#!/usr/bin/env python3
"""Extract log-mel features for a defined chunk of the complete manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

def features(path: Path, sr=16000, n_mels=40):
    x, actual = sf.read(path, dtype='float32'); x=np.asarray(x,dtype=np.float32)
    if x.ndim>1: x=x.mean(1)
    if actual!=sr: x=signal.resample_poly(x,sr,actual)
    _,_,z=signal.stft(x,fs=sr,nperseg=512,noverlap=384,nfft=512,boundary='zeros')
    power=np.abs(z)**2; hz=np.linspace(0,sr/2,power.shape[0]); edges=np.linspace(0,sr/2,n_mels+2); mel=np.zeros((n_mels,power.shape[1]))
    for i in range(n_mels):
        left,center,right=edges[i:i+3]; w=np.maximum(0,np.minimum((hz-left)/(center-left),(right-hz)/(right-center))); mel[i]=w@power
    log=np.log(np.maximum(mel,1e-10)); return np.concatenate([log.mean(1),log.std(1),np.percentile(log,25,axis=1),np.percentile(log,75,axis=1)]).astype('float32')

ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--cleaned',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); args=ap.parse_args()
m=pd.read_csv(args.manifest); c=pd.read_csv(args.cleaned).reset_index().rename(columns={'index':'row_index'}); m=m.iloc[args.start:args.end].copy(); cols=['row_index','target_cough_type','mask_cough_type','target_abnormalities','mask_abnormalities','target_diagnosis','mask_diagnosis','target_severity','mask_severity','target_overall_status','mask_overall_status','split']; m=m.merge(c[cols],on='row_index',how='left'); rows=[]
for _,r in m.iterrows():
    out={'row_index':int(r.row_index),'uuid':r.uuid,'split':r.split,'target_cough_type':r.target_cough_type,'mask_cough_type':int(r.mask_cough_type),'target_abnormalities':r.target_abnormalities,'mask_abnormalities':int(r.mask_abnormalities),'target_diagnosis':r.target_diagnosis,'mask_diagnosis':int(r.mask_diagnosis),'target_severity':r.target_severity,'mask_severity':int(r.mask_severity),'target_overall_status':r.target_overall_status,'mask_overall_status':int(r.mask_overall_status)}
    try: out['feature']=features(Path(str(r.processed_audio_path))).tolist(); out['ok']=True
    except Exception as e: out['ok']=False; out['error']=str(e)
    rows.append(out)
args.out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_json(args.out,orient='records'); print(json.dumps({'start':args.start,'end':args.end,'records':len(rows),'success':sum(x.get('ok',False) for x in rows)}))
