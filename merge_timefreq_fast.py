from pathlib import Path
import glob,re,os,ijson
parts=[]
for name in glob.glob('artifacts/tf_*.json'):
 m=re.search(r'tf_(\d+)_(\d+)\.json$',name)
 if m and 'check' not in name: parts.append((int(m.group(1)),int(m.group(2)),Path(name)))
parts.sort(); expected=0
for start,end,path in parts:
 if start != expected: raise SystemExit(f'missing range {expected}')
 expected=end
if expected != 34434: raise SystemExit(f'range ends at {expected}')
tmp=Path('artifacts/features_timefreq_all_34434.complete.json')
with tmp.open('wb') as out:
 out.write(b'[')
 for number,(start,end,path) in enumerate(parts):
  data=path.read_bytes().strip()
  if not (data.startswith(b'[') and data.endswith(b']')): raise SystemExit(f'not array: {path}')
  body=data[1:-1].strip()
  if body:
   if number: out.write(b',')
   out.write(body)
  print(f'{path.name} {start}-{end}',flush=True)
 out.write(b']')
# Stream-verify the combined JSON array without materializing it.
count=0
with tmp.open('rb') as src:
 for record in ijson.items(src,'item'):
  count += 1
if count != 34434: raise SystemExit(f'verification count={count}, expected=34434')
os.replace(tmp,'artifacts/features_timefreq_all_34434.json')
print('COMPLETE records=34434 unique_ranges=34434')
