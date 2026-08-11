#!/usr/bin/env python3
"""Train a shared 2D CNN from verified NumPy shards with masked multi-task heads."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HEADS = {
    'cough_type': ['dry', 'wet'],
    'diagnosis': ['COVID-19', 'healthy_cough', 'lower_infection', 'upper_infection', 'obstructive_disease'],
    'severity': ['mild', 'pseudocough', 'severe'],
    'overall_status': ['healthy', 'symptomatic', 'COVID-19'],
}
ABNORMALITIES = ['wheezing', 'dyspnea', 'congestion', 'nothing']


def target(meta, head):
    value = meta['targets'].get('target_' + head)
    labels = HEADS[head]
    return labels.index(value) if value in labels else 0


def abnormal(meta):
    value = str(meta['targets'].get('target_abnormalities') or '')
    return np.asarray([float(label in value.split('|')) for label in ABNORMALITIES], dtype='float32')


def shard_rows(paths, split, batch_size):
    for path in paths:
        z = np.load(path, allow_pickle=True)
        x = z['x'].astype('float32') / 16.0
        meta = z['meta']
        selected = [i for i, item in enumerate(meta) if item['split'] == split and item.get('valid')]
        for start in range(0, len(selected), batch_size):
            ids = selected[start:start + batch_size]
            xb = x[ids][..., None]
            y = {}
            w = {}
            for head in HEADS:
                y[head] = np.asarray([target(meta[i], head) for i in ids], dtype='int32')
                w[head] = np.asarray([float(meta[i]['masks'].get('mask_' + head, 0)) for i in ids], dtype='float32')
            y['abnormalities'] = np.stack([abnormal(meta[i]) for i in ids])
            w['abnormalities'] = np.asarray([float(meta[i]['masks'].get('mask_abnormalities', 0)) for i in ids], dtype='float32')
            yield xb, y, w


def count_split(paths, split):
    total = 0
    for path in paths:
        z = np.load(path, allow_pickle=True)
        total += sum(1 for item in z['meta'] if item['split'] == split and item.get('valid'))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit('TensorFlow is required: ' + str(exc))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.shards.glob('shard_*.npz'))
    if not paths:
        raise SystemExit('No shard files found')
    train_n = count_split(paths, 'train')
    val_n = count_split(paths, 'validation')
    test_n = count_split(paths, 'test')
    inputs = tf.keras.Input((128, 128, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = {head: tf.keras.layers.Dense(len(labels), activation='softmax', name=head)(x) for head, labels in HEADS.items()}
    outputs['abnormalities'] = tf.keras.layers.Dense(4, activation='sigmoid', name='abnormalities')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss={h: 'binary_crossentropy' if h == 'abnormalities' else 'sparse_categorical_crossentropy' for h in outputs}, metrics={h: ['accuracy'] for h in outputs})
    train_steps = max(1, int(np.ceil(train_n / args.batch_size)))
    val_steps = max(1, int(np.ceil(val_n / args.batch_size)))
    model.fit(shard_rows(paths, 'train', args.batch_size), steps_per_epoch=train_steps, validation_data=shard_rows(paths, 'validation', args.batch_size), validation_steps=val_steps, epochs=args.epochs, verbose=2)
    model.save(args.out_dir / 'respiratory_multitask_2dcnn.keras')
    report = {'architecture': 'shared 2D CNN with masked multi-task heads', 'shards': len(paths), 'train_valid_records': train_n, 'validation_valid_records': val_n, 'test_valid_records': test_n, 'note': 'Training completed from verified NumPy shards; test metrics require a separate deterministic evaluation pass.'}
    (args.out_dir / 'training_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
