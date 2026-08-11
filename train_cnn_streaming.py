#!/usr/bin/env python3
"""Streaming masked multi-task CNN trainer with epoch checkpoints."""
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


def make_generator(paths, split, class_weights):
    def generator():
        for path in paths:
            with np.load(path, allow_pickle=True) as shard:
                x = shard['x'].astype('float32') / 16.0
                for i, item in enumerate(shard['meta']):
                    if item.get('split') != split or not item.get('valid'):
                        continue
                    y = {}
                    w = {}
                    for head, labels in HEADS.items():
                        value = item['targets'].get('target_' + head)
                        index = labels.index(value) if value in labels else 0
                        y[head] = np.int32(index)
                        mask = float(item['masks'].get('mask_' + head, 0))
                        w[head] = np.float32(mask * class_weights[head][index])
                    yield x[i][..., None], y, w
    return generator


def count_split(paths, split):
    count = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as shard:
            count += sum(1 for item in shard['meta'] if item.get('split') == split and item.get('valid'))
    return count


def calculate_weights(paths, split):
    weights = {head: np.zeros(len(labels), dtype=np.float64) for head, labels in HEADS.items()}
    for path in paths:
        with np.load(path, allow_pickle=True) as shard:
            for item in shard['meta']:
                if item.get('split') != split or not item.get('valid'):
                    continue
                for head, labels in HEADS.items():
                    value = item['targets'].get('target_' + head)
                    if value in labels and item['masks'].get('mask_' + head, 0):
                        weights[head][labels.index(value)] += 1
    result = {}
    for head, counts in weights.items():
        total = counts.sum()
        result[head] = {i: float(total / (len(counts) * max(1, count))) for i, count in enumerate(counts)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    import tensorflow as tf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.shards.glob('shard_*.npz'))
    if not paths:
        raise SystemExit('No shard files found')
    shapes = (tf.TensorSpec((128, 128, 1), tf.float32), {h: tf.TensorSpec((), tf.int32) for h in HEADS}, {h: tf.TensorSpec((), tf.float32) for h in HEADS})
    train_count = count_split(paths, 'train')
    val_count = count_split(paths, 'validation')
    class_weights = calculate_weights(paths, 'train')
    train_ds = tf.data.Dataset.from_generator(make_generator(paths, 'train', class_weights), output_signature=shapes).shuffle(args.batch_size * 8).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(make_generator(paths, 'validation', class_weights), output_signature=shapes).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

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
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss={h: 'sparse_categorical_crossentropy' for h in HEADS}, metrics={h: ['accuracy'] for h in HEADS})
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(args.out_dir / 'checkpoint_epoch_{epoch:02d}.keras'), save_weights_only=False, save_freq='epoch')
    csv = tf.keras.callbacks.CSVLogger(str(args.out_dir / 'training_history.csv'), append=True)
    latest = sorted(args.out_dir.glob('checkpoint_epoch_*.keras'))
    initial_epoch = 0
    if latest:
        model = tf.keras.models.load_model(latest[-1])
        initial_epoch = int(latest[-1].stem.split('_')[-1])
    model.fit(train_ds, steps_per_epoch=(train_count + args.batch_size - 1) // args.batch_size, validation_data=val_ds, validation_steps=(val_count + args.batch_size - 1) // args.batch_size, initial_epoch=initial_epoch, epochs=args.epochs, callbacks=[checkpoint, csv], verbose=2)
    model.save(args.out_dir / 'respiratory_2dcnn_streaming.keras')
    (args.out_dir / 'training_manifest.json').write_text(json.dumps({'train_valid_records': train_count, 'validation_valid_records': val_count, 'epochs_requested': args.epochs, 'batch_size': args.batch_size, 'streaming': True, 'masked_sample_weights': True, 'class_weights': class_weights}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
