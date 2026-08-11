import glob, json, os
files = sorted(f for f in glob.glob('artifacts/tf_*.json') if 'tf_check' not in f)
target = 'artifacts/features_timefreq_all_34434_complete.jsonl'
seen = set()
rows = 0
with open(target, 'w', encoding='utf-8') as out:
    for number, path in enumerate(files, 1):
        with open(path, encoding='utf-8') as src:
            chunk = json.load(src)
        for record in chunk:
            index = int(record['row_index'])
            if index not in seen:
                out.write(json.dumps(record, separators=(',', ':')) + '\n')
                seen.add(index)
                rows += 1
        out.flush()
        print(f'{number}/{len(files)} {os.path.basename(path)} rows={rows}', flush=True)
print(f'COMPLETE rows={rows} unique={len(seen)} expected=34434', flush=True)
if rows != 34434 or len(seen) != 34434:
    raise SystemExit(2)
os.replace(target, 'artifacts/features_timefreq_all_34434.jsonl')
