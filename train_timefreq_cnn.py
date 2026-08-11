#!/usr/bin/env python3
"""Train time-preserving CNN multi-task heads from the verified artifact."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import classification_report,balanced_accuracy_score,accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder

HEADS={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--epochs',type=int,default=20);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
try:
 import tensorflow as tf
except Exception as e: raise SystemExit('TensorFlow is required for 2D CNN training: '+str(e))
rows=[]
with args.features.open('rb') as f:
 import ijson
 for r in ijson.items(f,'item'):
  if r.get('ok') and isinstance(r.get('timefreq'),list): rows.append(r)
if not rows: raise SystemExit('No valid time-frequency records')
d=pd.DataFrame(rows); X=np.asarray(d.timefreq.tolist(),dtype='float32')[...,None]; X=(X-X.mean())/(X.std()+1e-6); report={'records':len(d),'architecture':'compact 2D CNN shared encoder; independent masked heads','heads':{}}
inputs=tf.keras.Input(shape=(128,128,1)); x=tf.keras.layers.Conv2D(16,3,padding='same',activation='relu')(inputs);x=tf.keras.layers.MaxPool2D()(x);x=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(x);x=tf.keras.layers.MaxPool2D()(x);x=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x);x=tf.keras.layers.GlobalAveragePooling2D()(x); shared=tf.keras.layers.Dense(64,activation='relu')(x)
for head,labels in HEADS.items():
 mask=d[f'mask_{head}'].astype(bool).to_numpy(); tr=(d.split=='train').to_numpy()&mask; te=(d.split=='test').to_numpy()&mask
 if tr.sum()<2 or te.sum()<1: report['heads'][head]={'status':'insufficient'};continue
 enc=LabelEncoder();enc.fit(labels);y=enc.transform(d[f'target_{head}']); out=tf.keras.layers.Dense(len(labels),activation='softmax',name=head)(shared); model=tf.keras.Model(inputs,out); model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); counts=np.bincount(y[tr],minlength=len(labels)); weights={i:float(len(y[tr])/(len(labels)*max(1,c))) for i,c in enumerate(counts)}; model.fit(X[tr],y[tr],epochs=args.epochs,batch_size=32,class_weight=weights,validation_data=(X[d.split.to_numpy()=='validation'],y[d.split.to_numpy()=='validation']),verbose=0); p=model.predict(X[te],verbose=0);pred=p.argmax(1);report['heads'][head]={'train':int(tr.sum()),'test':int(te.sum()),'accuracy':float(accuracy_score(y[te],pred)),'balanced_accuracy':float(balanced_accuracy_score(y[te],pred)),'macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0)),'mean_confidence':float(p.max(1).mean()),'labels':list(enc.classes_),'classification_report':classification_report(y[te],pred,target_names=enc.classes_,zero_division=0,output_dict=True)};model.save(args.out_dir/f'{head}.keras')
(args.out_dir/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
