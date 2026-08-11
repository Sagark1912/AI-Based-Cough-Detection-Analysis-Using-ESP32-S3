import glob, json, os, re
files=[]
for path in glob.glob('artifacts/tf_*.json'):
    m=re.search(r'tf_(\d+)_(\d+)\.json$', path)
    if m and 'check' not in path:
        files.append((int(m.group(1)), int(m.group(2)), path))
files.sort()
expected=0
for start,end,path in files:
    if start != expected:
        raise SystemExit(f'missing range before {path}: expected {expected}, got {start}')
    expected=end
if expected != 34434:
    raise SystemExit(f'ranges end at {expected}, expected 34434')
tmp='artifacts/features_timefreq_all_34434.complete.jsonl'
rows=0
with open(tmp,'w',encoding='utf-8') as out:
    for start,end,path in files:
        with open(path,encoding='utf-8') as src:
            chunk=json.load(src)
        if len(chunk) != end-start:
            raise SystemExit(f'{path}: expected {end-start} rows, got {len(chunk)}')
        for record in chunk:
            if int(record['row_index']) < start or int(record['row_index']) >= end:
                raise SystemExit(f'{path}: row outside declared range')
            out.write(json.dumps(record,separators=(',',':'))+'\n')
            rows += 1
        out.flush()
        print(f'{os.path.basename(path)} rows={rows}',flush=True)
if rows != 34434:
    raise SystemExit(f'wrote {rows}, expected 34434')
os.replace(tmp,'artifacts/features_timefreq_all_34434.jsonl')
print('COMPLETE rows=34434 unique-range-verified=34434')
