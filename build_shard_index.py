import json
from pathlib import Path
import numpy as np
folder=Path('artifacts/timefreq_shards_complete'); out=folder/'index.json'; index={'train':[],'validation':[],'test':[],'counts':{}}
for p in sorted(folder.glob('shard_*.npz')):
 z=np.load(p,allow_pickle=True); meta=z['meta']
 for i,item in enumerate(meta):
  if item.get('valid') and item.get('split') in index: index[item['split']].append({'file':p.name,'offset':i,'row_index':int(item['row_index'])})
index['counts']={k:len(v) for k,v in index.items() if isinstance(v,list)}; out.write_text(json.dumps(index),encoding='utf-8');print(json.dumps(index['counts'],indent=2))
