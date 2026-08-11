import glob,json
seen=set(); total=0
with open('artifacts/features_timefreq_all_34434.jsonl','w',encoding='utf-8') as out:
 for f in sorted(glob.glob('artifacts/tf_*.json')):
  if 'check' in f: continue
  with open(f,encoding='utf-8') as src:
   for r in json.load(src):
    i=int(r['row_index'])
    if i not in seen:
     out.write(json.dumps(r)+'\n');seen.add(i);total+=1
print({'records':total,'unique':len(seen)})
