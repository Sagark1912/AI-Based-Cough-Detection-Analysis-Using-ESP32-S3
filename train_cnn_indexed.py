#!/usr/bin/env python3
"""Train CNN using cached shard index; each shard is loaded once per epoch."""
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--shards',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--epochs',type=int,default=5);a=ap.parse_args()
import tensorflow as tf
idx=json.loads((a.shards/'index.json').read_text()); paths=sorted(a.shards.glob('shard_*.npz')); cache={p.name:np.load(p,allow_pickle=True) for p in paths}
heads={'cough_type':['dry','wet'],'diagnosis':['COVID-19','healthy_cough','lower_infection','upper_infection','obstructive_disease'],'severity':['mild','pseudocough','severe'],'overall_status':['healthy','symptomatic','COVID-19']}
def make(split):
 X=[];Y={h:[] for h in heads};W={h:[] for h in heads}
 for ref in idx[split]:
  m=cache[ref['file']]['meta'][ref['offset']]; X.append(cache[ref['file']]['x'][ref['offset']].astype('float32')[...,None]/16)
  for h,labs in heads.items():
   Y[h].append(labs.index(m['targets'].get('target_'+h)) if m['targets'].get('target_'+h) in labs else 0)
   W[h].append(float(m['masks'].get('mask_'+h,0)))
 return np.stack(X),{h:np.asarray(v) for h,v in Y.items()},{h:np.asarray(v,dtype='float32') for h,v in W.items()}
Xtr,Ytr,Wtr=make('train');Xv,Yv,Wv=make('validation');a.out_dir.mkdir(parents=True,exist_ok=True)
inputs=tf.keras.Input((128,128,1));x=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(inputs);x=tf.keras.layers.MaxPool2D()(x);x=tf.keras.layers.Conv2D(32,3,activation='relu')(x);x=tf.keras.layers.MaxPool2D()(x);x=tf.keras.layers.Conv2D(64,3,activation='relu')(x);x=tf.keras.layers.GlobalAveragePooling2D()(x);x=tf.keras.layers.Dense(64,activation='relu')(x);out={h:tf.keras.layers.Dense(len(l),activation='softmax',name=h)(x) for h,l in heads.items()};model=tf.keras.Model(inputs,out);model.compile('adam',{h:'sparse_categorical_crossentropy' for h in heads},metrics={h:['accuracy'] for h in heads});model.fit(Xtr,Ytr,sample_weight=Wtr,validation_data=(Xv,Yv,Wv),epochs=a.epochs,batch_size=64,verbose=2);model.save(a.out_dir/'respiratory_2dcnn_indexed.keras');(a.out_dir/'index_training.json').write_text(json.dumps({'train':len(Xtr),'validation':len(Xv),'test_indexed':len(idx['test']),'epochs':a.epochs},indent=2))
