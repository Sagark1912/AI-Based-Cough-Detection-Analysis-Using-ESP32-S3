#!/usr/bin/env python3
"""Create resumable NumPy shards for every verified time-frequency record."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import ijson

ap = argparse.ArgumentParser()
ap.add_argument('--input', type=Path, required=True)
ap.add_argument('--out-dir', type=Path, required=True)
ap.add_argument('--shard-size', type=int, default=256)
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
manifest_path = args.out_dir / 'manifest.json'
state = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
next_row = int(state.get('records_written', 0))
shard_index = int(state.get('shards_written', 0))
parsed = 0
rows = []
tensors = []

def write_shard():
    global rows, tensors, shard_index
    if not rows:
        return
    start = rows[0]['row_index']
    end = rows[-1]['row_index'] + 1
    path = args.out_dir / f'shard_{shard_index:05d}_{start:05d}_{end:05d}.npz'
    np.savez_compressed(path, x=np.stack(tensors).astype(np.float16), meta=np.array(rows, dtype=object))
    shard_index += 1
    rows = []
    tensors = []
    manifest_path.write_text(json.dumps({'records_written': parsed, 'shards_written': shard_index, 'expected_records': 34434}, indent=2), encoding='utf-8')
    print(json.dumps({'parsed': parsed, 'shards': shard_index}), flush=True)

with args.input.open('rb') as src:
    for record in ijson.items(src, 'item'):
        if parsed < next_row:
            parsed += 1
            continue
        idx = int(record['row_index'])
        if idx != parsed:
            raise SystemExit(f'row index mismatch: stream position {parsed}, record {idx}')
        valid = bool(record.get('ok') is True and isinstance(record.get('timefreq'), list))
        tensor = np.zeros((128, 128), dtype=np.float16)
        if valid:
            tensor = np.asarray(record['timefreq'], dtype=np.float16)
            if tensor.shape != (128, 128):
                raise SystemExit(f'bad tensor shape at row {idx}: {tensor.shape}')
        rows.append({'row_index': idx, 'uuid': str(record.get('uuid', '')), 'split': str(record.get('split', '')), 'valid': valid, 'targets': {k: record.get(k) for k in ('target_cough_type', 'target_abnormalities', 'target_diagnosis', 'target_severity', 'target_overall_status')}, 'masks': {k: int(record.get(k, 0) or 0) for k in ('mask_cough_type', 'mask_abnormalities', 'mask_diagnosis', 'mask_severity', 'mask_overall_status')}})
        tensors.append(tensor)
        parsed += 1
        if len(rows) >= args.shard_size:
            write_shard()
write_shard()
manifest_path.write_text(json.dumps({'records_written': parsed, 'shards_written': shard_index, 'expected_records': 34434, 'complete': parsed == 34434}, indent=2), encoding='utf-8')
print(json.dumps({'records_written': parsed, 'shards_written': shard_index, 'complete': parsed == 34434}, indent=2))
