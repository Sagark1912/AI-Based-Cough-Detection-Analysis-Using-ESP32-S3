#!/usr/bin/env python3
"""Class-balanced masked multi-task baseline with train-only oversampling."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
import joblib
HEADS={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
d=pd.read_json(args.features); d=d[d.feature.notna()].copy(); X=np.vstack(d.feature); scaler=StandardScaler(); X=scaler.fit_transform(X); report={'records':len(d),'strategy':'train-only inverse-frequency oversampling; test/validation untouched','heads':{}}
rng=np.random.default_rng(42)
for head,labels in HEADS.items():
 mask=d[f'mask_{head}'].astype(bool).to_numpy(); tr=(d.split=='train').to_numpy() & mask; te=(d.split=='test').to_numpy() & mask
 if tr.sum()<2 or te.sum()<1: report['heads'][head]={'status':'insufficient_valid_data'}; continue
 enc=LabelEncoder(); enc.fit(labels); ytr=enc.transform(d.loc[tr,f'target_{head}']); yte=enc.transform(d.loc[te,f'target_{head}']); classes,counts=np.unique(ytr,return_counts=True); max_count=max(counts); indices=[]
 for cls,count in zip(classes,counts): indices.extend(rng.choice(np.flatnonzero(tr)[ytr==cls],size=max_count,replace=True).tolist())
 Xtr=X[indices]; ybal=enc.transform(d.iloc[indices][f'target_{head}']); model=MLPClassifier(hidden_layer_sizes=(96,48),early_stopping=True,max_iter=180,random_state=42); model.fit(Xtr,ybal); pred=model.predict(X[te]); probs=model.predict_proba(X[te]); report['heads'][head]={'status':'trained','original_train':int(tr.sum()),'balanced_train':int(len(indices)),'test':int(te.sum()),'accuracy':float(accuracy_score(yte,pred)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro',zero_division=0)),'mean_confidence':float(np.max(probs,axis=1).mean()),'labels':list(enc.classes_),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'classification_report':classification_report(yte,pred,target_names=enc.classes_,zero_division=0,output_dict=True)}; joblib.dump({'model':model,'scaler':scaler,'encoder':enc,'labels':labels},args.out_dir/f'{head}.joblib')
(args.out_dir/'evaluation.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
