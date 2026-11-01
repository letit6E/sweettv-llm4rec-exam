"""Exactly two cached-data CatBoost refits; all outputs stay in output/session."""
import argparse,gc,hashlib,json,pathlib,pickle,time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sweettv.experiment import Fold
from sweettv.improve import WINDOWS
from sweettv.horizon import rescale
from sweettv.blend import rrf

ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/'output/session'
OLD=ROOT/'artifacts/improvement-v2'; NEW=ROOT/'artifacts/discovery-v6'
KEYS=['early1','early2','early3','dev','audit']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def diff(a,b):
    z=np.abs(a.astype('float64')-b.astype('float64'))
    return {'shape':list(a.shape),'exact_equal':bool(np.array_equal(a,b)),
            'different_values':int(np.count_nonzero(a!=b)),'max_abs_difference':float(z.max()),'mean_abs_difference':float(z.mean())}
def write():
    (OUT/'training_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
def announce(x): print(json.dumps(x),flush=True)

def main():
    global OUT,OLD,NEW,result
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-cache',type=pathlib.Path,default=OLD)
    parser.add_argument('--v6-cache',type=pathlib.Path,default=NEW)
    parser.add_argument('--output',type=pathlib.Path,default=OUT)
    parser.add_argument('--skip-reference-models',action='store_true',help='Build without historical model binaries; CSV comparison still runs.')
    args=parser.parse_args();OLD=args.v2_cache.resolve();NEW=args.v6_cache.resolve();OUT=args.output.resolve()
    assert OUT==ROOT/'output/session' or ROOT/'output/session' in OUT.parents
    OUT.mkdir(parents=True,exist_ok=True)
    original={str(p.relative_to(ROOT)):digest(p) for p in [OLD/'v2_cat200.pkl',NEW/'final_discovery_joint.cbm',*sorted((ROOT/'submissions').glob('*.csv'))] if p.exists()}
    result={'scope':'Two actual final-model refits from exact cached historical rows, no tuning.',
            'train_windows':KEYS,'models':{},'csv_comparisons':{},'status':'running'}
    start=time.perf_counter();write()
    with threadpool_limits(limits=1):
        for kind in ['v2','v6']:
            parts=[];labels=[];row_records=[]
            for key in KEYS:
                assert WINDOWS[key][1]<=WINDOWS['final'][0]
                path=(OLD/(key+'-features.npz')) if kind=='v2' else (NEW/(key+'_discovery_features.npz'))
                with np.load(path) as data:
                    mask=data['valid']; x=data['X'][mask]; y=data['y'][mask]
                    parts.append(x);labels.append(y)
                    row_records.append({'fold':key,'label_end':WINDOWS[key][1],'rows':len(y),'positives':int(y.sum()),
                                        'cache':str(path.relative_to(ROOT)),
                                        'selected_row_indices_sha256':hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()})
            X=np.concatenate(parts);y=np.concatenate(labels);del parts,labels;gc.collect()
            assert X.shape==((1713057,121) if kind=='v2' else (2584719,167))
            params=dict(iterations=200 if kind=='v2' else 600,depth=6,learning_rate=.05,l2_leaf_reg=10,
                        random_seed=20261031,thread_count=4 if kind=='v2' else 5,verbose=False,allow_writing_files=False)
            if kind=='v2':params['loss_function']='Logloss'
            announce({'event':'fit_started','model':kind,'rows':len(y),'features':X.shape[1],'params':params})
            t=time.perf_counter();model=CatBoostClassifier(**params);model.fit(X,y);elapsed=time.perf_counter()-t
            path=OUT/(kind+'_fresh.cbm');model.save_model(str(path))
            record={'params':params,'shape':list(X.shape),'positives':int(y.sum()),'rows':row_records,
                    'fit_seconds':elapsed,'model_file':str(path.relative_to(ROOT)),'model_sha256':digest(path)}
            result['models'][kind]=record;write();announce({'event':'fit_finished','model':kind,'seconds':elapsed})
            del X,y;gc.collect()
            final=np.load(OLD/'final-features.npz' if kind=='v2' else NEW/'final_discovery_features.npz')
            x=rescale(final['X'],final['names'].tolist(),12) if kind=='v2' else final['X']
            pred=model.predict_proba(x)[:,1]
            if not args.skip_reference_models:
                if kind=='v2':
                    with (OLD/'v2_cat200.pkl').open('rb') as h:saved=pickle.load(h)
                else:
                    saved=CatBoostClassifier();saved.load_model(str(NEW/'final_discovery_joint.cbm'))
                reference=saved.predict_proba(x,ntree_end=200 if kind=='v2' else 600)[:,1]
                record['fresh_vs_saved_probability']=diff(pred,reference)
                record['saved_model_parameters']=saved.get_params()
                del saved,reference
            f=Fold(*WINDOWS['final'])
            dense=np.full((len(f.eval_users),len(f.items)),-1e9,dtype='float32')
            dense[final['u'],final['i']]=pred
            stored_path=OLD/'final_horizon_True.npy' if kind=='v2' else NEW/'final_discovery_joint_600.npy'
            if stored_path.exists():
                record['fresh_vs_artifact_dense']=diff(dense,np.load(stored_path))
            np.save(OUT/(kind+'_fresh_scores.npy'),dense)
            write();announce({'event':'scores_compared','model':kind,'comparison':record.get('fresh_vs_saved_probability',{'not_requested':True})})
            del model,pred,dense,x,final;gc.collect()
        f=Fold(*WINDOWS['final'])
        old=np.load(OUT/'v2_fresh_scores.npy');new=np.load(OUT/'v6_fresh_scores.npy')
        # This channel signal was recomputed and checked by check_discovery in this review.
        channel=np.load(NEW/'final_discovery_signals.npz')['raw_channel_28']
        oldrank,newrank,channelrank=rrf(f,old),rrf(f,new),rrf(f,channel)
        scores={'v2_cat200_horizon':old,'v5_channel_blend':.9*oldrank+.1*channelrank,
                'v6_discovery_blend':.9*(.5*oldrank+.5*newrank)+.1*channelrank}
        for name,score in scores.items():
            frame=f.template.copy();frame['tv_show_id']=[' '.join(map(str,p)) for p in f.rank(score)]
            path=OUT/(name+'_fresh.csv');frame.to_csv(path,index=False)
            submitted=ROOT/'submissions'/(name+'.csv');expected=pd.read_csv(submitted,dtype=str)
            a=np.array([p.split() for p in frame.tv_show_id]);b=np.array([p.split() for p in expected.tv_show_id])
            row={'file':str(path.relative_to(ROOT)),'sha256':digest(path),'submitted_sha256':digest(submitted),
                 'byte_identical':path.read_bytes()==submitted.read_bytes(),'user_order_exact':frame.user_id.tolist()==expected.user_id.tolist(),
                 'changed_rows':int(np.any(a!=b,axis=1).sum()),'changed_positions':int((a!=b).sum()),
                 'changed_item_sets':sum(set(x)!=set(y) for x,y in zip(a,b)),
                 'changed_row_indices':np.flatnonzero(np.any(a!=b,axis=1)).tolist()}
            result['csv_comparisons'][name]=row;announce({'event':'csv_compared','name':name,**row})
        result['original_models_and_csvs_preserved']=all(digest(ROOT/p)==h for p,h in original.items())
        assert result['original_models_and_csvs_preserved']
        result['seconds']=time.perf_counter()-start;result['status']='completed';write()
        announce({'event':'completed','seconds':result['seconds']})

if __name__=='__main__':main()
