#!/usr/bin/env python3
"""Create complete NumPy shards directly from verified range chunks.

This avoids reparsing the 7.35 GB merged JSON artifact. The chunk ranges are
validated for contiguous coverage of row indices 0..34433.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--chunk-dir', type=Path, default=Path('artifacts'))
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--shard-size', type=int, default=256)
parser.add_argument('--start-row', type=int, default=0)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
chunks = []
for path in glob.glob(str(args.chunk_dir / 'tf_*.json')):
    match = re.search(r'tf_(\d+)_(\d+)\.json$', path)
    if match and 'check' not in path:
        chunks.append((int(match.group(1)), int(match.group(2)), Path(path)))
chunks.sort()
expected = 0
for start, end, path in chunks:
    if start != expected:
        raise SystemExit(f'non-contiguous chunk at {path.name}: expected {expected}, got {start}')
    expected = end
if expected != 34434:
    raise SystemExit(f'chunk coverage ends at {expected}, expected 34434')

manifest = args.out_dir / 'manifest.json'
records = args.start_row
valid = 0
shards = args.start_row // args.shard_size
meta = []
tensors = []
for start, end, path in chunks:
    if end <= args.start_row: continue
    with path.open(encoding='utf-8') as source:
        chunk = json.load(source)
    if len(chunk) != end - start:
        raise SystemExit(f'{path.name}: expected {end-start} records, got {len(chunk)}')
    for record in chunk:
        row = int(record['row_index'])
        if row < args.start_row:
            continue
        if row != records:
            raise SystemExit(f'row order mismatch: expected {records}, got {row}')
        ok = record.get('ok') is True
        tensor = np.zeros((128, 128), dtype=np.float16)
        if ok:
            tensor = np.asarray(record['timefreq'], dtype=np.float16)
            if tensor.shape != (128, 128):
                raise SystemExit(f'bad tensor shape at row {row}: {tensor.shape}')
            valid += 1
        tensors.append(tensor)
        meta.append({
            'row_index': row,
            'uuid': str(record.get('uuid', '')),
            'split': str(record.get('split', '')),
            'valid': ok,
            'targets': {key: record.get(key) for key in ('target_cough_type', 'target_abnormalities', 'target_diagnosis', 'target_severity', 'target_overall_status')},
            'masks': {key: int(record.get(key, 0) or 0) for key in ('mask_cough_type', 'mask_abnormalities', 'mask_diagnosis', 'mask_severity', 'mask_overall_status')},
        })
        records += 1
        if len(tensors) == args.shard_size:
            np.savez_compressed(args.out_dir / f'shard_{shards:05d}_{records-len(tensors):05d}_{records:05d}.npz', x=np.stack(tensors), meta=np.array(meta, dtype=object))
            shards += 1
            tensors, meta = [], []
            print(json.dumps({'records': records, 'valid_features': valid, 'shards': shards}), flush=True)
if tensors:
    np.savez_compressed(args.out_dir / f'shard_{shards:05d}_{records-len(tensors):05d}_{records:05d}.npz', x=np.stack(tensors), meta=np.array(meta, dtype=object))
    shards += 1
result = {'records': records, 'expected': 34434, 'valid_features': valid, 'invalid_records': records-valid, 'shards': shards, 'tensor_shape': [128, 128], 'complete': records == 34434}
manifest.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
if not result['complete']:
    raise SystemExit(2)
