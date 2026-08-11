import glob,json,re,struct
files=[]
for p in glob.glob('artifacts/tf_*.json'):
 m=re.search(r'tf_(\d+)_(\d+)\.json$',p)
 if m and 'check' not in p:files.append((int(m[1]),int(m[2]),p))
files.sort(); expected=0
for s,e,p in files:
 if s!=expected:raise SystemExit(f'missing {expected}')
 expected=e
if expected!=34434:raise SystemExit(expected)
out=open('artifacts/features_timefreq_all_34434.jsonl','wb');rows=0
for s,e,p in files:
 for r in json.load(open(p,encoding='utf-8')):
  out.write((json.dumps(r,separators=(',',':'))+'\n').encode());rows+=1
 out.flush();print(p,rows,flush=True)
out.close()
if rows!=34434:raise SystemExit(rows)
print('COMPLETE',rows)
