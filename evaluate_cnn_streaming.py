#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
HEADS={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--shards',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();import tensorflow as tf
model=tf.keras.models.load_model(a.model);results={'heads':{}}
for head,labels in HEADS.items():
 y=[];p=[]
 for path in sorted(a.shards.glob('shard_*.npz')):
  z=np.load(path,allow_pickle=True);x=z['x'].astype('float32')/16
  ids=[i for i,m in enumerate(z['meta']) if m.get('split')=='test' and m.get('valid') and m['masks'].get('mask_'+head,0)]
  if not ids:continue
  yy=[labels.index(z['meta'][i]['targets'].get('target_'+head)) for i in ids]; pp=model.predict(x[ids][...,None],verbose=0)[head].argmax(1); y.extend(yy);p.extend(pp)
 results['heads'][head]={'samples':len(y),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,p).tolist(),'classification_report':classification_report(y,p,target_names=labels,zero_division=0,output_dict=True)}
a.out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
