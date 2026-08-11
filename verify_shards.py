import json, re
from pathlib import Path
import numpy as np
folder=Path('artifacts/timefreq_shards_complete'); files=sorted(folder.glob('shard_*.npz')); total=valid=0; ranges=[]
for p in files:
 m=re.search(r'shard_\d+_(\d+)_(\d+)\.npz$',p.name)
 if not m: continue
 start,end=map(int,m.groups()); z=np.load(p,allow_pickle=True); x=z['x']; meta=z['meta']
 if len(meta)!=end-start or x.shape[0]!=end-start or x.shape[1:]!=(128,128): raise SystemExit(f'bad {p.name}')
 if start!=total: raise SystemExit(f'gap/overlap at {p.name}: expected {total}')
 total=end; valid+=sum(bool(item.get('valid',False)) for item in meta); ranges.append([start,end])
result={'folder':str(folder),'shards':len(files),'records':total,'expected':34434,'valid_features':valid,'complete':total==34434,'ranges':ranges}
(folder/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(0 if result['complete'] else 2)
