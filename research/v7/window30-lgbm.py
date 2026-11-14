"""One fixed W30 real-data LightGBM recipe; no network or old-model refitting."""
import sys,json,time,resource,math,platform,importlib.metadata,subprocess
from pathlib import Path
from collections import defaultdict,Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import numpy as np
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H,file_sha256 as sha
from V6.experimental.v7_source_snapshot import _atomic_bytes
from V6.experimental.v7_e1_checkpoint import atomic_json
from V6.experimental.v7_research_artifacts import _safe,_lock
from V6.experimental.v7_research_evaluation import daily_metric
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7';ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window30-real-lightgbm');RESTORE=ROOT.parent/'window30-restore';OLD=ROOT.parent/'window29-expanded-development'
def recipe():return json.loads((OUT/'window30-recipe.json').read_text())
def enc(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def size():return sum(p.stat().st_size for root in (ROOT,RESTORE) if root.exists() for p in root.rglob('*') if p.is_file())
def capacity(n=0):
 if size()+n>268435456-8388608:raise ValueError('W30_ARTIFACT_256MIB_STOP_NO_GC')
def put(p,v):
 b=enc(v);capacity(len(b));p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('IMMUTABLE_W30_CONFLICT')
 else:_atomic_bytes(_safe(p),b)
 return {'path':str(p),'bytes':len(b),'sha256':sha(p)}
def checked(r):
 p=Path(r['path']);assert p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r.get('bytes',p.stat().st_size),'INPUT_NOT_RESTORABLE_OR_HASH_MISMATCH'
 return json.loads(p.read_text())
@contextmanager
def budget(stage):
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'CPU_BUDGET.json'
 with _lock(ROOT/'cpu.lock'):
  v=json.loads(p.read_text()) if p.exists() else {'limit':1200,'verification_reserve':60,'stages':[]}
  if any(x['status']=='RUNNING' for x in v['stages']):raise ValueError('PRIOR_STAGE_INCOMPLETE_STOP')
  remaining=1140-sum(x['cpu_seconds'] for x in v['stages'])
  if remaining<10:raise ValueError('CPU_TOTAL_BUDGET_STOP')
  cap=min(300,int(remaining));row={'stage':stage,'status':'RUNNING','cpu_seconds':cap,'reserved':cap};v['stages'].append(row);atomic_json(p,v)
 resource.setrlimit(resource.RLIMIT_CPU,(cap,cap));start=time.monotonic()
 try:yield
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);row.update(status='TERMINAL',cpu_seconds=u.ru_utime+u.ru_stime,wall_seconds=time.monotonic()-start)
  with _lock(ROOT/'cpu.lock'):atomic_json(p,v)

def environment():
 r=recipe();actual={'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['lightgbm','numpy','scipy']}}
 if actual!=r['dependency_versions']:raise ValueError('DEPENDENCY_VERSION_MISMATCH_NO_INSTALL')
 lib=Path(lgb.basic._LIB._name);assert lib.is_file();return {**actual,'scikit_learn':importlib.metadata.version('scikit-learn'),'native_library':str(lib),'native_library_sha256':sha(lib),'executable':sys.executable}

def inputs():
 cov=json.loads((OLD/'PREFIT.json').read_text());assert cov==json.loads((OUT/'window29-coverage.json').read_text())
 old=json.loads((OLD/'COMPLETE.json').read_text());assert old==json.loads((OUT/'window29-run.json').read_text())
 features=checked(cov['refs']['features']);labels=checked(cov['refs']['labels']);checked(cov['refs']['sources']);oldpred=checked(old['artifacts']['predictions']);daily=checked(old['artifacts']['daily']);support=checked(old['artifacts']['common_support'])
 index={(f['signal_session'],f['stock_id']):f for f in features};assert len(index)==len(features)
 assert recipe()['feature_order']==json.loads((OUT/'window29-plan.json').read_text())['features']
 return cov,old,features,labels,index,oldpred,daily,support

def training(index,labels,h):
 qs=[q for q in labels if q['eligible'] and q['key'][2]==h and '2021-01-01'<=q['key'][0]<'2022-01-01']
 if not qs or any(not q['key'][0]<q['entry']<=q['exit']<'2022-01-01' for q in qs):raise ValueError('TRAINING_LABEL_MATURITY_OR_SPLIT_FAILURE')
 x=np.asarray([index[tuple(q['key'][:2])]['features'] for q in qs],dtype=np.float64);y=np.asarray([q['label'] for q in qs],dtype=np.float64)
 if x.shape!=(len(qs),2) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('TRAINING_FEATURE_ORDER_OR_FINITE_FAILURE')
 return x,y,qs

def reserve(root,h):
 root.mkdir(parents=True,exist_ok=True)
 with (root/f'FIT_{h}_CLAIM.json').open('x') as f:json.dump({'horizon':h,'maximum_fits':1,'fresh':True,'at':datetime.now(timezone.utc).isoformat()},f)

def fit_arrays(x,y,r,factory=lgb.LGBMRegressor):
 estimator=factory(**r['config']);start=time.perf_counter();estimator.fit(x,y,feature_name=r['feature_order']);elapsed=time.perf_counter()-start
 return {'booster_text':estimator.booster_.model_to_string(num_iteration=-1),'actual_iterations':estimator.booster_.current_iteration(),'fit_wall_seconds':elapsed}

def assert_support(a,b):
 def keys(xs):return {(r['signal_session'],r['stock_id'],r['horizon'],r['label_id'],r['label_metadata_sha256'],r['label']) for r in xs}
 if keys(a)!=keys(b) or len(keys(a))!=len(a) or len(keys(b))!=len(b):raise ValueError('PAIRED_LABEL_SUPPORT_MISMATCH')

def model_sha(m):return H({k:m[k] for k in ['model_id','spec_sha256','recipe_sha256','booster_text','actual_iterations','feature_order','train_view_sha256']})
def prepare():
 env=environment();cov,old,features,labels,index,oldpred,daily,support=inputs();r=recipe();specs={}
 for h in [5,10]:
  x,y,qs=training(index,labels,h);specs[str(h)]={'model_id':r['planned_model_ids'][str(h)],'train_n':len(qs),'train_view_sha256':H(qs),'train_x_sha256':H(x.tolist()),'train_y_sha256':H(y.tolist()),'latest_exit':max(q['exit'] for q in qs),'initialization':'FRESH_STATE_NO_RESUME','spec_sha256':H({'recipe':H(r),'horizon':h,'training_view':H(qs)})}
 refs={'W29_coverage':{'path':str(OUT/'window29-coverage.json'),'sha256':sha(OUT/'window29-coverage.json')},'features':cov['refs']['features'],'labels':cov['refs']['labels'],'sources':cov['refs']['sources'],'old_predictions':old['artifacts']['predictions'],'old_daily':old['artifacts']['daily'],'old_support':old['artifacts']['common_support']}
 proof={'status':'FROZEN_BEFORE_ANY_FIT','recipe_sha256':H(r),'recipe_file_sha256':sha(OUT/'window30-recipe.json'),'code_sha256':sha(Path(__file__)),'dataset_id':cov['dataset_id'],'feature_id':cov['feature_id'],'label_id':cov['label_id'],'cohort_sha256':cov['cohort_sha256'],'fold_id':old['fold_id'],'environment':env,'models':specs,'inputs':refs,'source_evidence_id':H(cov['refs']['sources']),'expected_prediction_keys_sha256':cov['predicted_keys_expected_sha256'],'prepared_at':datetime.now(timezone.utc).isoformat()}
 put(ROOT/'PREPARED.json',proof);put(OUT/'window30-prepared.json',proof);print(json.dumps({'status':proof['status'],'models':specs,'environment':env}))

def frozen():
 p=json.loads((ROOT/'PREPARED.json').read_text());assert p==json.loads((OUT/'window30-prepared.json').read_text());assert p['recipe_sha256']==H(recipe()) and p['code_sha256']==sha(Path(__file__)) and p['environment']==environment()
 for ref in p['inputs'].values():checked(ref)
 return p

def fit(h):
 p=frozen();r=recipe()
 for path in [OUT/'window30-recipe.json',OUT/'window30-prepared.json',Path(__file__)]:assert subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(REPO))],cwd=REPO)==path.read_bytes(),'PREFIT_LOCAL_COMMIT_REQUIRED'
 if h==10:
  primary=json.loads((ROOT/'model-5.json').read_text());assert primary['recipe_sha256']==p['recipe_sha256'] and primary['dataset_id']==p['dataset_id']
 cov,old,features,labels,index,oldpred,daily,support=inputs();x,y,qs=training(index,labels,h);spec=p['models'][str(h)];assert H(qs)==spec['train_view_sha256'] and H(x.tolist())==spec['train_x_sha256'] and H(y.tolist())==spec['train_y_sha256']
 reserve(ROOT,h);learned=fit_arrays(x,y,r);assert 1<=learned['actual_iterations']<=64
 m={**spec,**learned,'recipe_sha256':p['recipe_sha256'],'feature_order':r['feature_order'],'dataset_id':p['dataset_id'],'feature_id':p['feature_id'],'label_id':p['label_id'],'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'preprocessing':'NONE; train-only internal bin mapper','source_evidence_ref':p['source_evidence_id']};m['model_sha256']=model_sha(m);m['model_text_bytes']=len(m['booster_text'].encode());put(ROOT/f'model-{h}.json',m)
 print(json.dumps({k:m[k] for k in ['model_id','model_sha256','train_n','actual_iterations','fit_wall_seconds','model_text_bytes']}))

def predict(models,features,p):
 # No label-table or eligibility-mask parameter in the scoring interface.
 result={};timings={};r=recipe();allkeys=[]
 for h,m in models.items():
  booster=lgb.Booster(model_str=m['booster_text']);assert booster.feature_name()==r['feature_order'];assert model_sha(m)==m['model_sha256']
  for role in ['VALIDATION','EXPOSED_DIAGNOSTIC']:
   fs=[f for f in features if f['split']==role];x=np.asarray([f['features'] for f in fs],dtype=np.float64);start=time.perf_counter();scores=booster.predict(x,num_iteration=m['actual_iterations'],num_threads=1);timings[f'{h}|{role}']=time.perf_counter()-start;assert np.isfinite(scores).all()
   rows=[{'stock_id':f['stock_id'],'signal_session':f['signal_session'],'horizon':int(h),'score':float(v),'model_id':m['model_id'],'model_sha256':m['model_sha256'],'feature_id':p['feature_id'],'dataset_id':p['dataset_id'],'recipe_id':r['recipe_id'],'source_evidence_ref':p['source_evidence_id']} for f,v in zip(fs,scores)];result[f'LightGBM|{h}|{role}']=rows;allkeys.extend((v['signal_session'],v['stock_id'],v['horizon']) for v in rows)
 assert H(sorted(allkeys))==p['expected_prediction_keys_sha256'],'PREDICTION_UNIVERSE_CHANGED_BY_OUTCOMES'
 return result,timings

def evaluate(pred,labels,oldpred,old_daily,old_support):
 li={tuple(q['key']):q for q in labels};assert len(li)==len(labels);oldidx={(r['model'],r['signal_session'],r['horizon']):r for r in old_daily};daily=[];paired=[];summary=[]
 for year,h in [(2022,5),(2022,10),(2023,5),(2023,10)]:
  role='VALIDATION' if year==2022 else 'EXPOSED_DIAGNOSTIC';rows=pred[f'LightGBM|{h}|{role}'];oldrows=oldpred[f'Ridge|{h}|{role}'];assert {(r['signal_session'],r['stock_id'],r['horizon']) for r in rows}=={(r['signal_session'],r['stock_id'],r['horizon']) for r in oldrows}
  joined=[];oldjoined=[]
  for r in rows:
   q=li.get((r['signal_session'],r['stock_id'],h))
   if q and q['eligible']:joined.append({**r,'label':q['label'],'label_id':q['label_id'],'label_metadata_sha256':q['label_metadata_sha256']})
  for r in oldrows:
   q=li.get((r['signal_session'],r['stock_id'],h))
   if q and q['eligible']:oldjoined.append({**r,'label':q['label'],'label_id':q['label_id'],'label_metadata_sha256':q['label_metadata_sha256']})
  assert_support(joined,oldjoined);by=defaultdict(list)
  for r in joined:by[r['signal_session']].append(r)
  group=[];diffs=defaultdict(list)
  for s in [s for s in old_support if int(s['signal_session'][:4])==year and s['horizon']==h]:
   day=s['signal_session'];rs=by[day];q=daily_metric(rs,day,h,{'min_valid_n':40});assert q['support_sha256']==s['support_sha256'] and q['stock_ids']==s['stock_ids'] and q['valid_n']==s['valid_n']
   q.update(model='LightGBM',year=year,score_ties_n=len(rs)-len({r['score'] for r in rs}),label_ties_n=len(rs)-len({r['label'] for r in rs}));daily.append(q);group.append(q)
   for model in ['L0_POS','L0_NEG','Ridge']:
    base=oldidx[(model,day,h)];assert base['support_sha256']==q['support_sha256'] and base['stock_ids']==q['stock_ids'];delta=q['rank_ic']-base['rank_ic'] if q['rank_ic'] is not None and base['rank_ic'] is not None else None;paired.append({'year':year,'signal_session':day,'horizon':h,'baseline':model,'baseline_RankIC':base['rank_ic'],'LightGBM_RankIC':q['rank_ic'],'delta':delta,'valid_n':q['valid_n'],'support_sha256':q['support_sha256']})
    if delta is not None:diffs[model].append(delta)
  vals=[q['rank_ic'] for q in group if q['rank_ic'] is not None];summary.append({'year':year,'horizon':h,'LightGBM_mean':float(np.mean(vals)) if vals else None,'LightGBM_median':float(np.median(vals)) if vals else None,'requested_days':len(group),'defined_days':len(vals),'daily_N':dict(Counter(q['valid_n'] for q in group)),'skipped_reasons':dict(Counter(reason for q in group for reason in q['reasons'])),'paired_mean_LightGBM_minus':{model:float(np.mean(v)) if v else None for model,v in diffs.items()},'evidence_class':'EXPOSED_DEVELOPMENT_DIAGNOSTIC_NO_WINNER'})
 return daily,paired,summary

def score(replay=False):
 p=frozen();cov,old,features,labels,index,oldpred,old_daily,old_support=inputs();models={str(h):json.loads((ROOT/f'model-{h}.json').read_text()) for h in [5,10]}
 if replay:
  closed=json.loads((ROOT/'COMPLETE.json').read_text())
  for ref in closed['artifacts'].values():checked(ref)
  assert H(models)==H(checked(closed['artifacts']['models']))
 pred,timing=predict(models,features,p);before=H(pred)
 if not replay:put(ROOT/'predictions.json',pred) # publish immutable predictions before Label Eligibility Join
 daily,paired,summary=evaluate(pred,labels,oldpred,old_daily,old_support);assert H(pred)==before
 values={'models':models,'predictions':pred,'daily':daily,'paired':paired,'summary':summary}
 if replay:
  for name,value in values.items():assert H(value)==H(checked(closed['artifacts'][name])),'REPLAY_MISMATCH:'+name
  result={'status':'INDEPENDENT_PROCESS_REPLAY_PASS','fits':0,'predictions':sum(map(len,pred.values())),'daily':len(daily),'paired':len(paired),'source_W29_model_metrics_unmodified':True,'common_support_exact':True,'new_Ridge_fits_API_GPU':0};put(OUT/'window30-replay.json',result);print(json.dumps(result));return
 refs={name:put(ROOT/(name+'.json'),value) for name,value in values.items()};closed={'status':'COMPLETE_DEVELOPMENT_ONLY_NOT_ADMITTED','study_id':recipe()['study_id'],'recipe_sha256':p['recipe_sha256'],'prepared':{'path':str(ROOT/'PREPARED.json'),'sha256':sha(ROOT/'PREPARED.json')},'input_dataset_id':p['dataset_id'],'feature_id':p['feature_id'],'label_id':p['label_id'],'cohort_sha256':p['cohort_sha256'],'fold_id':p['fold_id'],'environment':p['environment'],'code_sha256':sha(Path(__file__)),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'artifacts':refs,'LightGBM_fits':2,'Ridge_fits_API_GPU':0,'prediction_published_before_label_join':True,'model_info':{h:{k:m[k] for k in ['model_id','model_sha256','train_n','actual_iterations','fit_wall_seconds','model_text_bytes']} for h,m in models.items()},'inference_wall_seconds':timing,'actual_generated_at':datetime.now(timezone.utc).isoformat(),'backup':'BACKUP_PENDING_SAME_HOST'};put(ROOT/'COMPLETE.json',closed);put(OUT/'window30-run.json',closed);put(OUT/'window30-results.json',summary);print(json.dumps({'model_info':closed['model_info'],'inference':timing,'summary':summary}))
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)==2 else ''
 if mode not in ['prepare','fit5','fit10','evaluate','replay']:raise SystemExit('choose prepare/fit5/fit10/evaluate/replay')
 with budget(mode):
  if mode=='prepare':prepare()
  elif mode.startswith('fit'):fit(int(mode[3:]))
  else:score(mode=='replay')
