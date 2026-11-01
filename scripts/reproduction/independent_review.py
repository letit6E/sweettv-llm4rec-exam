"""Read-only independent ranking/AP checks; no sweettv imports or model fitting.
Run: .venv/bin/python output/session/independent_review.py
Only independent_review.json is written. Processed events are trusted input;
this is not a fresh raw interval join or an independent Kaggle evaluation.
"""
import csv, hashlib, json, pathlib, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SESSION = ROOT / 'output/session'
V2, V5, V6 = ('artifacts/'+x for x in ['improvement-v2','approaches-v5','discovery-v6'])
WINDOWS = {'dev':('2020-05-18','2020-06-22'), 'audit':('2020-06-22','2020-07-27'),
           'long':('2020-05-18','2020-07-27'), 'final':('2020-07-27','2020-10-19')}
def read(path): return json.loads((ROOT/path).read_text())
def array(path): return np.load(ROOT/path, mmap_mode='r')
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def rows(path):
    with (ROOT/path).open(newline='') as f:
        r=csv.DictReader(f); assert r.fieldnames==['user_id','tv_show_id']; return list(r)
def epoch(date): return int(pd.Timestamp(date,tz='Europe/Kyiv').timestamp())

def state(fold, schedule, events, users, template):
    start,end=map(epoch,WINDOWS[fold])
    past=schedule[(schedule.end<=start)&~schedule.source.str.contains('31-42')]
    future=schedule[schedule.source.str.contains('31-42')] if fold=='final' else schedule[(schedule.start>=start)&(schedule.start<end)]
    items=np.union1d(past.tv_show_id,future.tv_show_id)
    eligible=np.flatnonzero(np.isin(items,future.tv_show_id))
    full=events[(events.end<=start)&(events.fraction>=.8)]
    pop=full.groupby('tv_show_id').user.nunique().reindex(items,fill_value=0).to_numpy()
    uid={u:i for i,u in enumerate(users)}; evaluation=[uid[r['user_id']] for r in template]
    cold=~np.isin(evaluation,full.user.unique())
    target=events[(events.start>=start)&(events.end<=end)&(events.fraction>=.8)]
    counts=target.groupby(['user','tv_show_id']).size().reset_index(name='count')
    counts=counts.sort_values(['user','count','tv_show_id'],ascending=[True,False,True])
    lookup=counts.groupby('user').head(5).groupby('user').tv_show_id.apply(list).to_dict()
    truth=[set(lookup.get(u,[])) for u in evaluation]
    return {'items':items,'eligible':eligible,'pop':pop,'cold':cold,'truth':truth}

def ranking(scores,s):
    assert scores.shape==(len(s['cold']),len(s['items'])) and np.isfinite(scores).all()
    idx=s['eligible']; ids=s['items'][idx]; pop=s['pop'][idx]
    result=[]
    for n,row in enumerate(scores):
        v=pop if s['cold'][n] else row[idx]
        result.append(ids[np.lexsort((ids,-pop,-v))[:5]].tolist())
    return result

def reciprocal(scores,s):
    idx=s['eligible']; ids=s['items'][idx]; pop=s['pop'][idx]
    result=np.zeros(scores.shape,dtype='float32')
    weights=1/(5+np.arange(len(idx),dtype='float32'))
    for n,row in enumerate(scores):
        result[n,idx[np.lexsort((ids,-pop,-row[idx]))]]=weights
    return result

def metric(pred,s):
    ap=[]
    for truth,rec in zip(s['truth'],pred):
        hits=0; value=0.
        for rank,item in enumerate(rec,1):
            if item in truth: hits+=1; value+=hits/rank
        ap.append(value/min(5,len(truth)) if truth else 0.)
    ap=np.asarray(ap); n=np.array([len(t) for t in s['truth']])
    return {'map5_all':float(ap.mean()),'map5_nonempty':float(ap[n>0].mean()),
            'map5_five_targets':float(ap[n==5].mean()),'five_target_users':int((n==5).sum())}

def equal_metric(actual,expected):
    for key in actual.keys() & expected.keys():
        assert abs(actual[key]-expected[key])<1e-12,(key,actual[key],expected[key])


def main():
    started=time.perf_counter(); template=rows('data/raw/submission.csv')
    assert len(template)==1260 and len({r['user_id'] for r in template})==1260
    schedule=pd.read_pickle(ROOT/'data/processed/schedule.pkl')
    events=pd.read_pickle(ROOT/'data/processed/events.pkl')
    assert (schedule.tv_show_id>0).all() and events.end.max()<=epoch('2020-07-27')
    assert not set(events.airing_id)&set(schedule[schedule.source.str.contains('31-42')].airing_id)
    users=read('data/processed/users.json')
    allowed=set(pd.read_csv(ROOT/'data/raw/export_arh_31-42-final.csv',usecols=['tv_show_id']).tv_show_id)-{0}
    out={'scope':'Independent CSV, ranking, fusion and AP verification from saved artifacts; no final-ranker training or hidden-label evaluation.',
         'submissions':[], 'local_metrics':{}, 'course_methods':{}, 'limitations':['Processed events reused; raw interval matching not rebuilt.', 'Kaggle scores checked against saved observations only; hidden labels unavailable.']}
    s=state('final',schedule,events,users,template)
    b=array(V2+'/final_horizon_True.npy'); c=array(V6+'/final_discovery_signals.npz')['raw_channel_28']
    d=array(V6+'/final_discovery_joint_600.npy')
    br,cr,dr=reciprocal(b,s),reciprocal(c,s),reciprocal(d,s)
    computed=[b,.9*br+.1*cr,.9*(.5*br+.5*dr)+.1*cr]
    paths=[V2+'/v2_cat200_horizon.npy',V5+'/final_channel_blend.npy',V6+'/final_discovery_blend.npy']
    reports=[(V2+'/kaggle-submissions.json','v2_cat200_horizon.csv'),
             (V5+'/kaggle-results.json','v5_channel_blend.csv'),(V6+'/kaggle-result.json','v6_discovery_blend.csv')]
    for scores,source,(report,name) in zip(computed,paths,reports):
        assert np.array_equal(scores,array(source)),source
        observed=read(report); observed=observed if isinstance(observed,list) else [observed]
        record=next(r for r in observed if pathlib.Path(r['file']).name==name)
        path='submissions/'+name; data=rows(path); expected=ranking(scores,s)
        assert [r['user_id'] for r in data]==[r['user_id'] for r in template]
        assert len({r['user_id'] for r in data})==1260
        predictions=[list(map(int,r['tv_show_id'].split())) for r in data]
        assert predictions==expected
        assert all(len(p)==len(set(p))==5 and set(p)<=allowed for p in predictions)
        assert sha(path)==record['sha256']
        out['submissions'].append({'file':path,'rows':len(data),'rows_match_template_and_saved_ranking':True,
            'five_unique_official_ids':True,'sha256':sha(path),'hash_matches_source':True,
            'fusion_array_exact':True,'private':record['private'],'public':record['public'],'score_source':report})
    for fold in ['dev','audit','long']:
        s=state(fold,schedule,events,users,template)
        b=array(V2+'/'+(fold+'_cat6_200.npy' if fold!='long' else 'long_horizon_True.npy'))
        c=array(V5+'/'+fold+'_channel.npy'); d=array(V6+'/'+fold+'_discovery_joint_600.npy')
        br,cr,dr=reciprocal(b,s),reciprocal(c,s),reciprocal(d,s)
        scores=[b,.9*br+.1*cr,.9*(.5*br+.5*dr)+.1*cr]
        source=V6+'/'+{'dev':'context-selection.json','audit':'context-audit-confirmation.json','long':'context-long-confirmation.json'}[fold]
        evidence=read(source)
        if fold=='long': reported=next(r for r in read(V2+'/horizon-results.json') if r['fold']=='long' and r['five_week_normalization'])
        else: reported=next(r for r in read(V2+'/'+fold+'-results.json') if r['name']==fold+'_cat6_200')
        metrics=[metric(ranking(x,s),s) for x in scores]
        equal_metric(metrics[0],reported)
        equal_metric(metrics[1],{'map5_five_targets':evidence['baseline']})
        equal_metric(metrics[2],{'map5_five_targets':evidence['selected']['map5']})
        out['local_metrics'][fold]={'candidates':len(s['eligible']),'cold_users':int(s['cold'].sum()),
            'v2':metrics[0],'v5':metrics[1],'v6':metrics[2],'matches_sources':True,'source':source}
        if fold=='dev':
            frozen=read('artifacts/frozen-selection.json'); records=read('artifacts/development-results.json')
            audit=read('artifacts/audit-results.json')['metrics']
            for method,config in frozen['best_individuals'].items():
                record=next(r for r in records if r['config']==config)
                actual=metric(ranking(array('artifacts/cache/'+record['key']+'.npy'),s),s)
                equal_metric(actual,record)
                out['course_methods'][method]={'dev_recomputed':actual['map5_five_targets'],
                    'audit_reported':audit[method]['map5_five_targets'],'dev_matches_source':True}
    before=read('output/session/submission_hashes_before.json')
    assert all(sha(p)==h for p,h in before.items())
    out['all_submission_bytes_preserved']={'files':len(before),'passed':True}
    out['runs']=read('output/session/reproduction_runs.json')
    assert len(out['runs'])==5 and all(r['status']=='passed' for r in out['runs'])
    out['training_qualification']='Saved CatBoost rankers reloaded/scored; check_improvement_reproduction additionally refits ALS via generate -> Fold.model.'
    out['findings']=[
        {'issue':'Historical README scope','evidence':'README.md:54,77 versus improve_final.py and discovery_ranker.py',
         'detail':'Audit-never-fit and no-Kaggle-selection statements describe initial HGB only. Final v2/v6 use completed dev/audit labels; all later windows and Kaggle feedback are adaptively reused.'},
        {'issue':'Missing audit reproduction entrypoint','evidence':'scripts/discovery_select.py:65 versus scripts/finalize_discovery.py:26',
         'detail':'CLI writes dev and long confirmation only; required context-audit-confirmation.json exists but has no checked-in writer.'},
        {'issue':'Check scope overstatement','evidence':'README.md final paragraph versus scripts/check_discovery.py',
         'detail':'check_discovery recomputes 11 signals and checks shared 121 columns, not all 167 features or raw official candidate IDs. finalize_discovery supplies raw-ID and saved-model checks.'},
        {'issue':'Stale historical test counts','evidence':'README.md:110','detail':'Ten tests refers to initial study; actual current suite has 29.'},
        {'issue':'Training qualification','detail':out['training_qualification']}
    ]
    training=SESSION/'training_reproduction.json'
    if training.exists():
        tr=json.loads(training.read_text())
        out['fresh_training']={'source':'output/session/training_reproduction.json','status':tr['status'],
            'models':{k:{'shape':v['shape'],'fit_seconds':v['fit_seconds'],
                         'probability_comparison':v.get('fresh_vs_saved_probability')} for k,v in tr['models'].items()},
            'csv_comparisons':tr['csv_comparisons']}
    out['seconds']=round(time.perf_counter()-started,3);out['passed']=True
    (SESSION/'independent_review.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'passed':True,'seconds':out['seconds'],'csvs':3,'local_folds':3,'all_csvs_preserved':len(before)}))

if __name__=='__main__': main()
