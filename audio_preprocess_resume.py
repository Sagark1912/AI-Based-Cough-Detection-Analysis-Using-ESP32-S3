#!/usr/bin/env python3
"""Reliable, resumable full-dataset audio preprocessing; processes every cleaned row."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from audio_preprocess import process

ap = argparse.ArgumentParser()
ap.add_argument('--cleaned', type=Path, required=True)
ap.add_argument('--out-audio', type=Path, required=True)
ap.add_argument('--out-results', type=Path, required=True)
ap.add_argument('--sample-rate', type=int, default=16000)
args = ap.parse_args()
args.out_audio.mkdir(parents=True, exist_ok=True)
args.out_results.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.cleaned)
done = set()
if args.out_results.exists():
    with args.out_results.open(encoding='utf-8') as f:
        for line in f:
            try: done.add(int(json.loads(line)['row_index']))
            except (ValueError, KeyError, json.JSONDecodeError): pass
with args.out_results.open('a', encoding='utf-8') as sink:
    for position, (_, row) in enumerate(df.iterrows()):
        if position in done: continue
        raw = Path(str(row.get('raw_audio_path', '')))
        uuid = str(row.get('uuid', raw.stem))
        target = args.out_audio / (uuid + '.wav')
        result = process(raw, target, args.sample_rate) if raw.is_file() else {'ok': False, 'error': 'raw audio missing'}
        result.update({'row_index': position, 'uuid': uuid, 'processed_audio_path': str(target) if result.get('ok') else '', 'mask_cough_type': int(row.get('mask_cough_type', 0)), 'mask_abnormalities': int(row.get('mask_abnormalities', 0)), 'mask_diagnosis': int(row.get('mask_diagnosis', 0)), 'mask_severity': int(row.get('mask_severity', 0)), 'mask_overall_status': int(row.get('mask_overall_status', 0))})
        sink.write(json.dumps(result) + '\n'); sink.flush()
print(json.dumps({'dataset_rows': len(df), 'completed_rows': len(done) + (len(df) - len(done)), 'results_file': str(args.out_results), 'resume_supported': True}, indent=2))
