#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd
from feature_core import extract
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cleaned',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);a=ap.parse_args();m=pd.read_csv(a.manifest).iloc[a.start:a.end].copy();c=pd.read_csv(a.cleaned).reset_index().rename(columns={'index':'row_index'});cols=['row_index','target_cough_type','mask_cough_type','target_abnormalities','mask_abnormalities','target_diagnosis','mask_diagnosis','target_severity','mask_severity','target_overall_status','mask_overall_status','split'];m=m.merge(c[cols],on='row_index',how='left');rows=[]
for _,r in m.iterrows():
 o={'row_index':int(r.row_index),'uuid':r.uuid,'split':r.split,'target_cough_type':r.target_cough_type,'mask_cough_type':int(r.mask_cough_type),'target_abnormalities':r.target_abnormalities,'mask_abnormalities':int(r.mask_abnormalities),'target_diagnosis':r.target_diagnosis,'mask_diagnosis':int(r.mask_diagnosis),'target_severity':r.target_severity,'mask_severity':int(r.mask_severity),'target_overall_status':r.target_overall_status,'mask_overall_status':int(r.mask_overall_status)}
 try:o['timefreq'],o['complementary']=extract(Path(str(r.processed_audio_path)));o['timefreq']=o['timefreq'].tolist();o['complementary']=o['complementary'].tolist();o['ok']=True
 except Exception as e:o['ok']=False;o['error']=str(e)
 rows.append(o)
a.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_json(a.out,orient='records');print(json.dumps({'start':a.start,'end':a.end,'records':len(rows),'success':sum(x.get('ok',False) for x in rows)}))
