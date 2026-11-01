"""Rebuild missing historical feature caches and selected models, without saved models.
Requires original data/processed/{schedule.pkl,events.pkl,views.npz,users.json},
data/raw/submission.csv and the two historical raw schedules for ID-zero context.
Run with PYTHONPATH=src .venv/bin/python scripts/reproduction/build_selected_from_processed.py.
All generated caches/models/CSVs go under output/session/from_processed.
A full run is expensive and was NOT performed during the ten-minute review.
"""
import argparse, inspect, pathlib, subprocess, sys
import numpy as np
from threadpoolctl import threadpool_limits
from sweettv import improve, discovery_context, discovery_ranker
from sweettv.experiment import Fold
from sweettv.rerank import generate, build_features

ROOT=pathlib.Path(__file__).resolve().parents[2]

def base_cache(key,dest):
    path=dest/(key+'-features.npz'); scores=dest/(key+'-scores.npz')
    if path.exists() and scores.exists():return
    f=Fold(*improve.WINDOWS[key]);s,r=generate(f)
    for name,value in improve.neighbor_scores(f).items():
        s[name]=value;a=value.copy();a[:,~f.eligible]=-np.inf
        order=np.argsort(-a,axis=1,kind='stable');rank=np.empty_like(order,dtype='int32')
        np.put_along_axis(rank,order,np.arange(len(f.items))[None,:],axis=1);r[name]=rank
    X,u,i,names,_=build_features(f,s,r);extra,en=improve.extra_features(f,u,i)
    X=np.column_stack([X,extra]);names+=en
    truth=[set(t) for t in f.truth]
    y=np.array([int(f.items[j] in truth[row]) for row,j in zip(u,i)],dtype='int8')
    valid=np.array([len(truth[row])==5 and not f.cold[row] for row in u])
    assert X.shape[1]==121
    np.savez_compressed(path,X=X,u=u,i=i,y=y,valid=valid,names=np.array(names),items=f.items)
    # Original HGB scores are diagnostics only, never a v2/v6 input feature.
    np.savez_compressed(scores,**s)
    print('BASE_CACHE',key,len(y),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--fold',choices=list(improve.WINDOWS),help='Prepare one fold only; implies --prepare-only.')
    args=parser.parse_args()
    dest=ROOT/'output/session/from_processed';old=dest/'v2';new=dest/'v6'
    old.mkdir(parents=True,exist_ok=True);new.mkdir(parents=True,exist_ok=True)
    # Existing discovery preparation consumes the v2 scores, dropping the unused
    # original HGB diagnostic. Make that single deletion optional in a private
    # function namespace; neither source files nor original artifacts are changed.
    source=inspect.getsource(discovery_ranker.prepare)
    assert source.count('old.pop("hgb")')==1,'Upstream prepare changed; review adapter.'
    namespace=dict(discovery_ranker.__dict__,OLD=old,DEST=new)
    exec(compile(source.replace('old.pop("hgb")','old.pop("hgb", None)'),str(__file__),'exec'),namespace)
    discovery_context.DEST=new
    keys=[args.fold] if args.fold else ['early1','early2','early3','dev','audit','final']
    with threadpool_limits(limits=1):
        for key in keys:
            base_cache(key,old);namespace['prepare'](key)
    if not (args.prepare_only or args.fold):
        subprocess.run([sys.executable,str(ROOT/'scripts/reproduction/retrain_selected.py'),
                        '--v2-cache',str(old),'--v6-cache',str(new),
                        '--output',str(dest/'trained'),'--skip-reference-models'],cwd=ROOT,check=True)

if __name__=='__main__':main()
