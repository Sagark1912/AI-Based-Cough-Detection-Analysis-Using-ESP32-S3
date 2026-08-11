#!/usr/bin/env python3
"""Time-aware feature training with validation calibration and abstention."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report
from sklearn.isotonic import IsotonicRegression
import joblib
HEADS={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
d=pd.read_json(args.features); d=d[d.feature.notna()].copy(); X=np.vstack(d.feature); scaler=StandardScaler(); X=scaler.fit_transform(X); report={'calibration':'validation-derived confidence threshold; abstain below threshold','heads':{}}
rng=np.random.default_rng(42)
for head,labels in HEADS.items():
 mask=d[f'mask_{head}'].astype(bool).to_numpy(); tr=(d.split=='train').to_numpy()&mask; va=(d.split=='validation').to_numpy()&mask; te=(d.split=='test').to_numpy()&mask
 if tr.sum()<2 or te.sum()<1 or va.sum()<1: report['heads'][head]={'status':'insufficient'}; continue
 enc=LabelEncoder();enc.fit(labels); ytr=enc.transform(d.loc[tr,f'target_{head}']); yva=enc.transform(d.loc[va,f'target_{head}']); yte=enc.transform(d.loc[te,f'target_{head}']); classes,counts=np.unique(ytr,return_counts=True); n=max(counts); idx=[]
 for cls,count in zip(classes,counts):idx.extend(rng.choice(np.flatnonzero(tr)[ytr==cls],n,replace=True))
 model=MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=220,random_state=42).fit(X[idx],enc.transform(d.iloc[idx][f'target_{head}']))
 pv=model.predict_proba(X[va]); pt=model.predict_proba(X[te]); va_conf=pv.max(1); va_correct=(pv.argmax(1)==yva); threshold=float(np.quantile(va_conf[~va_correct],0.8)) if (~va_correct).any() else 0.5; threshold=max(0.5,min(0.9,threshold)); pred=pt.argmax(1); abstain=pt.max(1)<threshold; kept=~abstain; report['heads'][head]={'train':int(len(idx)),'validation':int(len(yva)),'test':int(len(yte)),'threshold':threshold,'coverage':float(kept.mean()),'abstention_rate':float(abstain.mean()),'accuracy_all':float(accuracy_score(yte,pred)),'accuracy_non_abstained':float(accuracy_score(yte[kept],pred[kept])) if kept.any() else None,'balanced_accuracy_all':float(balanced_accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro',zero_division=0)),'mean_confidence':float(pt.max(1).mean()),'labels':list(enc.classes_),'classification_report':classification_report(yte,pred,target_names=enc.classes_,zero_division=0,output_dict=True)}; joblib.dump({'model':model,'scaler':scaler,'encoder':enc,'threshold':threshold,'labels':labels},args.out_dir/f'{head}.joblib')
(args.out_dir/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
