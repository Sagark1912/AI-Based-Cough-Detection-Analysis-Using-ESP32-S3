#!/usr/bin/env python3
"""Run audio preprocessing over the complete cleaned dataset in resumable ranges."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from audio_preprocess import process

ap = argparse.ArgumentParser()
ap.add_argument('--cleaned', type=Path, required=True)
ap.add_argument('--out-audio', type=Path, required=True)
ap.add_argument('--out-features', type=Path, required=True)
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--end', type=int, default=0)
ap.add_argument('--sample-rate', type=int, default=16000)
args = ap.parse_args()
df = pd.read_csv(args.cleaned)
end = args.end if args.end else len(df)
if args.start < 0 or end > len(df) or args.start >= end: raise SystemExit('Invalid range')
rows=[]
for position, (_, row) in enumerate(df.iloc[args.start:end].iterrows(), start=args.start):
    raw = Path(str(row.get('raw_audio_path', '')))
    uuid = str(row.get('uuid', raw.stem))
    target = args.out_audio / (uuid + '.wav')
    result = process(raw, target, args.sample_rate) if raw.is_file() else {'ok': False, 'error': 'raw audio missing'}
    result.update({'row_index': position, 'uuid': uuid, 'processed_audio_path': str(target) if result.get('ok') else ''})
    rows.append(result)
out = pd.DataFrame(rows)
args.out_features.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out_features, index=False)
out.to_json(args.out_features.with_suffix('.report.json'), orient='records', indent=2)
print(json.dumps({'start': args.start, 'end': end, 'rows': len(out), 'success': int(out.ok.sum()), 'failed': int((~out.ok).sum())}, indent=2))
