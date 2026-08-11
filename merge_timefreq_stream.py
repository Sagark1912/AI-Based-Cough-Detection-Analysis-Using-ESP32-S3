import glob,json,re,ijson,os
files=[]
for p in glob.glob('artifacts/tf_*.json'):
 m=re.search(r'tf_(\d+)_(\d+)\.json$',p)
 if m and 'check' not in p:files.append((int(m[1]),int(m[2]),p))
files.sort();expected=0
for s,e,p in files:
 if s!=expected:raise SystemExit(f'missing range {expected}')
 expected=e
if expected!=34434:raise SystemExit(f'end {expected}')
tmp='artifacts/features_timefreq_verified.jsonl';rows=0
with open(tmp,'w',encoding='utf-8') as out:
 for s,e,p in files:
  count=0
  with open(p,'rb') as src:
   for r in ijson.items(src,'item'):
    if not s<=int(r['row_index'])<e:raise SystemExit('range error')
    out.write(json.dumps(r,separators=(',',':'),default=float)+'\n');count+=1;rows+=1
  if count!=e-s:raise SystemExit(f'{p} count {count}')
  out.flush();print(os.path.basename(p),rows,flush=True)
if rows!=34434:raise SystemExit(f'rows {rows}')
os.replace(tmp,'artifacts/features_timefreq_all_34434.jsonl');print('COMPLETE',rows)
