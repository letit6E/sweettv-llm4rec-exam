import json
import numpy as np
import pandas as pd
from pathlib import Path
from threadpoolctl import threadpool_limits
from sweettv.experiment import Fold
from sweettv.improve import WINDOWS
from sweettv.finalize import metadata

with threadpool_limits(limits=1):
 f=Fold(*WINDOWS['final']); meta=metadata(f)
 examples=json.load(open('artifacts/final-examples.json'))
 novelty={}
 for label,path in [('v2','v2_cat200_horizon'),('v5','v5_channel_blend'),('v6','v6_discovery_blend')]:
  frame=pd.read_csv('submissions/'+path+'.csv',dtype={'user_id':str})
  assert frame.user_id.tolist()==f.template.user_id.astype(str).tolist()
  preds=[list(map(int,s.split())) for s in frame.tv_show_id]
  histories=f.train[f.train.fraction>=.8].groupby('user').tv_show_id.apply(set).to_dict()
  novelty[label]=sum(i not in histories.get(int(u),set()) for u,ps in zip(f.eval_users,preds) for i in ps)/(1260*5)
  for ex in examples:
   row=ex['template_row']; user=int(f.eval_users[row])
   hist=f.train[(f.train.user==user)&(f.train.fraction>=.8)].groupby('tv_show_id').size()
   ex['recommendations'][label]=[{'id':i,'title':meta.get(i,{}).get('tv_show_title','Unknown'),'category':meta.get(i,{}).get('tv_show_category','Unknown'),'train_watch_count':int(hist.get(i,0))} for i in preds[row]]
 Path('artifacts/final-examples-updated.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
 Path('artifacts/final-novelty-updated.json').write_text(json.dumps(novelty,indent=2))
 for e in examples:
  print(e['user_id'],[(x['title'],x['train_watch_count']) for x in e['recommendations']['v6']])
 print(novelty)
