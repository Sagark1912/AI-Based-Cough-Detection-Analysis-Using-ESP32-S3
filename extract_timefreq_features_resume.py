#!/usr/bin/env python3
"""Resumable time-frequency extraction for every cleaned row."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from feature_core import extract
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cleaned',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args(); m=pd.read_csv(args.manifest);c=pd.read_csv(args.cleaned).reset_index().rename(columns={'index':'row_index'}); cols=['row_index','target_cough_type','mask_cough_type','target_abnormalities','mask_abnormalities','target_diagnosis','mask_diagnosis','target_severity','mask_severity','target_overall_status','mask_overall_status','split'];m=m.merge(c[cols],on='row_index',how='left');done=set()
if args.out.exists():
 for line in args.out.read_text(encoding='utf-8').splitlines():
  try:done.add(int(json.loads(line)['row_index']))
  except:pass
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('a',encoding='utf-8') as sink:
 for _,r in m.iterrows():
  if int(r.row_index) in done:continue
  o={k:(int(r[k]) if k.startswith('mask_') else r[k]) for k in cols};o['row_index']=int(r.row_index)
  try:o['timefreq'],o['complementary']=extract(Path(str(r.processed_audio_path)));o['timefreq']=o['timefreq'].tolist();o['complementary']=o['complementary'].tolist();o['ok']=True
  except Exception as e:o['ok']=False;o['error']=str(e)
  sink.write(json.dumps(o)+'\n');sink.flush()
print(json.dumps({'dataset_rows':len(m),'completed_rows':len(done)},indent=2))
