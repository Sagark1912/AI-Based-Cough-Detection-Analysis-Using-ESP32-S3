#!/usr/bin/env python3
"""Train/evaluate masked multi-task baseline from extracted log-mel features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report
import joblib

HEADS={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
d=pd.read_json(args.features); d=d[d.feature.notna()].copy(); X=np.vstack(d.feature); scaler=StandardScaler(); X=scaler.fit_transform(X); report={'records':len(d),'heads':{}}
for head,labels in HEADS.items():
    mask=d[f'mask_{head}'].astype(bool).to_numpy(); usable=mask & d.target_cough_type.notna().to_numpy() if head=='cough_type' else mask
    tr=(d.split=='train').to_numpy() & usable; te=(d.split=='test').to_numpy() & usable
    if tr.sum()<2 or te.sum()<1: report['heads'][head]={'status':'insufficient_valid_data','train':int(tr.sum()),'test':int(te.sum())}; continue
    enc=LabelEncoder(); enc.fit(labels); ytr=enc.transform(d.loc[tr,f'target_{head}']); yte=enc.transform(d.loc[te,f'target_{head}'])
    model=MLPClassifier(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=80,random_state=42); model.fit(X[tr],ytr); pred=model.predict(X[te]); conf=float(np.max(model.predict_proba(X[te]),axis=1).mean())
    report['heads'][head]={'status':'trained','train':int(tr.sum()),'test':int(te.sum()),'accuracy':float(accuracy_score(yte,pred)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),'mean_confidence':conf,'labels':list(enc.classes_),'classification_report':classification_report(yte,pred,target_names=enc.classes_,zero_division=0,output_dict=True)}
    joblib.dump({'model':model,'scaler':scaler,'encoder':enc,'labels':labels},args.out_dir/f'{head}.joblib')
(args.out_dir/'evaluation.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
