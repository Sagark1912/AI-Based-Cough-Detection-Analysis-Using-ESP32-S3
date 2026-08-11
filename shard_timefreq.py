#!/usr/bin/env python3
"""Convert verified time-frequency JSON to compressed NumPy shards."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, ijson
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--shard-size',type=int,default=512);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True); rows=[];total=0;shard=0
for r in ijson.items(args.input.open('rb'),'item'):
 total+=1
 if r.get('ok') is not True: continue
 tf=np.asarray(r.get('timefreq'),dtype=np.float16)
 if tf.shape!=(128,128): raise SystemExit(f'bad shape row {r.get("row_index")}: {tf.shape}')
 rows.append({'x':tf,'row_index':int(r['row_index']),'split':r.get('split',''),'targets':{k:r.get(k) for k in ['target_cough_type','target_abnormalities','target_diagnosis','target_severity','target_overall_status']},'masks':{k:r.get(k,0) for k in ['mask_cough_type','mask_abnormalities','mask_diagnosis','mask_severity','mask_overall_status']}})
 if len(rows)>=args.shard_size:
  np.savez_compressed(args.out_dir/f'shard_{shard:04d}.npz',x=np.stack([z['x'] for z in rows]),meta=np.array(rows,dtype=object));shard+=1;rows=[];print(f'parsed={total} written_shards={shard}',flush=True)
if rows:np.savez_compressed(args.out_dir/f'shard_{shard:04d}.npz',x=np.stack([z['x'] for z in rows]),meta=np.array(rows,dtype=object));shard+=1
print(json.dumps({'json_records_parsed':total,'valid_feature_records':sum(np.load(p,allow_pickle=True)['x'].shape[0] for p in args.out_dir.glob('shard_*.npz')),'shards':shard,'shape':[128,128]},indent=2))
