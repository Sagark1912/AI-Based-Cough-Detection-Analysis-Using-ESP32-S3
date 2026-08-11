#!/usr/bin/env python3
"""Extract time-preserving log-mel tensors and compact complementary audio features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf
from scipy import signal

def extract(path:Path,sr=16000,n_mels=128,n_frames=128):
 x,actual=sf.read(path,dtype='float32');x=np.asarray(x,dtype=np.float32);x=x.mean(1) if x.ndim>1 else x
 if actual!=sr:x=signal.resample_poly(x,sr,actual)
 x=x/(np.max(np.abs(x))+1e-8); _,_,z=signal.stft(x,fs=sr,nperseg=512,noverlap=384,nfft=512,boundary='zeros');p=np.abs(z)**2
 hz=np.linspace(0,sr/2,p.shape[0]);edges=np.linspace(0,sr/2,n_mels+2);mel=np.zeros((n_mels,p.shape[1]))
 for i in range(n_mels):
  l,c,r=edges[i:i+3];w=np.maximum(0,np.minimum((hz-l)/(c-l),(r-hz)/(r-c)));mel[i]=w@p
 log=np.log(np.maximum(mel,1e-10));
 if log.shape[1]<n_frames:log=np.pad(log,((0,0),(0,n_frames-log.shape[1])),mode='constant',constant_values=log.min())
 else:log=log[:,:n_frames]
 rms=np.sqrt(np.mean(x*x)); zcr=float(np.mean(np.abs(np.diff(np.signbit(x))))); centroid=float(np.sum(hz[:,None]*p)/(np.sum(p)+1e-9)); bandwidth=float(np.sqrt(np.sum(((hz[:,None]-centroid)**2)*p)/(np.sum(p)+1e-9))); duration=float(len(x)/sr)
 return log.astype('float32'),np.array([rms,zcr,centroid,bandwidth,duration,float(np.max(np.abs(x))),float(np.mean(log)),float(np.std(log))],dtype='float32')

ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cleaned',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=0);args=ap.parse_args();m=pd.read_csv(args.manifest);end=args.end or len(m);m=m.iloc[args.start:end].copy();c=pd.read_csv(args.cleaned).reset_index().rename(columns={'index':'row_index'});cols=['row_index','target_cough_type','mask_cough_type','target_abnormalities','mask_abnormalities','target_diagnosis','mask_diagnosis','target_severity','mask_severity','target_overall_status','mask_overall_status','split'];m=m.merge(c[cols],on='row_index',how='left');rows=[]
for _,r in m.iterrows():
 o={'row_index':int(r.row_index),'uuid':r.uuid,'split':r.split,'target_cough_type':r.target_cough_type,'mask_cough_type':int(r.mask_cough_type),'target_abnormalities':r.target_abnormalities,'mask_abnormalities':int(r.mask_abnormalities),'target_diagnosis':r.target_diagnosis,'mask_diagnosis':int(r.mask_diagnosis),'target_severity':r.target_severity,'mask_severity':int(r.mask_severity),'target_overall_status':r.target_overall_status,'mask_overall_status':int(r.mask_overall_status)}
 try:o['timefreq'],o['complementary']=extract(Path(str(r.processed_audio_path)));o['ok']=True
 except Exception as e:o['ok']=False;o['error']=str(e)
 rows.append(o)
args.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_json(args.out,orient='records');print(json.dumps({'records':len(rows),'success':sum(x.get('ok',False) for x in rows),'tensor':'128x128','complementary_features':8}))
