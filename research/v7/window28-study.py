"""W28: exactly two clean CPU Ridge fits and a bounded causal feature check.

Reuses W24 numeric adapter, W27 mask, W25 pair check and W08 ranking primitive.
No network, search, scheduler, GPU or changes to previous evidence.
"""
import json,sys,runpy,copy,subprocess,platform
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_classical_baseline import ridge_fit_state,ridge_predict
from V6.experimental.v7_retrospective_smoke import engineering_sample
from V6.experimental.v7_research_artifacts import verify_bundle,_safe
from V6.experimental.v7_source_snapshot import _atomic_bytes
from V6.experimental.v7_robustness_audit import leave_one as paired_daily
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7'
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window28-clean-refit')
W27=ROOT.parent/'window27-label-safety'
S=runpy.run_path(str(OUT/'window27-safety.py'))
def key(r):return (r['signal_session'],r['stock_id'],r['horizon'])
def put(path,value):
 b=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=b:raise ValueError('IMMUTABLE_W28_OUTPUT_CONFLICT')
 else:_atomic_bytes(_safe(path),b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def clean_training(data,mask,h):
 rows=[r for r in data if r['split']=='TRAIN' and r['horizon']==h and mask[key(r)]['eligible']]
 for r in rows:
  if not '2021-01-01'<=r['signal_session']<'2022-01-01' or not r['signal_session']<r['entry']<=r['exit']<'2022-01-01':raise ValueError('TRAINING_TIME_OR_MATURITY_VIOLATION')
 return rows

def same_support(a,b):
 def identities(rows):return {(key(r),r['label_id'],r['label_metadata_sha256'],r['label']) for r in rows}
 if len(identities(a))!=len(a) or len(identities(b))!=len(b) or identities(a)!=identities(b):raise ValueError('SUPPORT_OR_LABEL_IDENTITY_CHANGED')

def reserve_fit(root,h):
 root.mkdir(parents=True,exist_ok=True)
 with (root/f'FIT_{h}_CLAIM.json').open('x') as f:json.dump({'horizon':h,'limit':1,'initialization':'FRESH_STATE_NO_RESUME','claimed_at':datetime.now(timezone.utc).isoformat()},f)

def features(signal,h,days,prices):
 q=engineering_sample(signal=signal,horizon=h,sessions=days,prices=prices,market='TWSE_LISTED',calendar_ok=True,known_action_dates=[])
 if q['features'] is None:raise ValueError('FEATURE_CASE_UNAVAILABLE:'+str(q['reasons']))
 return q['features']

def poison_future(prices,signal):
 changed=copy.deepcopy(prices);n=0
 for i,day in enumerate(sorted(changed)):
  if day<=signal:continue
  n+=1;factor=11.+i/1000.;p=changed[day]
  for field in ('open','max','min','close'):p[field]*=factor
  for field in ('Trading_Volume','Trading_money','Trading_turnover'):
   if field in p:p[field]=p[field]*17+37
 return changed,n

def load_inputs():
 prior=S['input_pins']()
 for r in json.loads((OUT/'window27-local-artifacts.json').read_text())['files']:
  assert Path(r['path']).stat().st_size==r['bytes'] and file_sha256(Path(r['path']))==r['sha256']
 calc=json.loads((OUT/'window27-replay-repair.json').read_text())['original_calculation_source'];assert file_sha256(Path(calc['path']))==calc['sha256']==file_sha256(Path(calc['restore']))
 closed=json.loads((W27/'COMPLETE.json').read_text());assert closed==json.loads((OUT/'window27-diagnostic-manifest.json').read_text())
 for ref in closed['artifacts'].values():S['checked'](ref)
 frozen=json.loads((W27/'FROZEN_MASK.json').read_text());assert frozen==json.loads((OUT/'window27-mask-manifest.json').read_text())
 for p,pin in frozen['source_refs'].items():assert file_sha256(REPO/p)==pin
 assert file_sha256(OUT/'window27-safety.py')==frozen['predicate_sha256'] and file_sha256(OUT/'window27-policy.json')==frozen['policy_file_sha256']
 masks=S['checked'](frozen['mask']);mask={tuple(r['key']):r for r in masks}
 run,bundle,data,days,raw=S['source_views']();assert verify_bundle(S['W24']/'bundles',run['bundle_key'],run['manifest_sha256'])['status']=='ARTIFACTS_VERIFIED'
 assert len(mask)==len(data)
 for r in data:assert canonical_hash(r)==mask[key(r)]['source_sample_sha256']
 old=S['checked'](closed['artifacts']['frozen_prediction_subset']);old_daily=S['checked'](closed['artifacts']['daily']);plan=json.loads((OUT/'window28-plan.json').read_text());recipe=json.loads((OUT/'window24-label-recipe.json').read_text())
 assert plan['features']==recipe['features'] and plan['ridge']['alpha']==recipe['Ridge']['alpha']==1.0
 index={key(r):r for r in data}
 for name,rows in old.items():
  role=name.split('-',2)[2];h=int(name.split('-')[1]);expected={key(r) for r in data if r['split']==role and r['horizon']==h and mask[key(r)]['eligible']}
  assert {key(r) for r in rows}==expected and len(rows)==len(expected)
  for r in rows:assert r['label_metadata_sha256']==canonical_hash(index[key(r)]) and r['label']==index[key(r)]['diagnostic_label']
 return {'plan':plan,'recipe':recipe,'mask':mask,'data':data,'days':days,'raw':raw,'old':old,'old_daily':old_daily,'index':index,'W24_run':run,'W27_closed':closed,'prior_private_files':sum(prior.values())+9}

def preflight(inputs):
 p=inputs['plan'];data=inputs['data'];mask=inputs['mask']
 for h in p['horizons']:
  rows=clean_training(data,mask,h);assert len(rows)==p['expected_training_counts'][str(h)]['clean']
 for sec in p['poison_cases']['stocks']:
  for signal in p['poison_cases']['signals']:
   for h in p['poison_cases']['horizons']:
    k=(signal,sec,h);assert k in inputs['index'] and mask[k]['eligible'],'PREDECLARED_POISON_CASE_NOT_ELIGIBLE'
    assert features(signal,h,inputs['days'],inputs['raw'][sec])==inputs['index'][k]['features']
 return {'status':'PREFIT_PASS','clean_train_n':{str(h):len(clean_training(data,mask,h)) for h in p['horizons']},'poison_cases':18,'plan_sha256':canonical_hash(p),'source_pins':{str(path.relative_to(REPO)):file_sha256(path) for path in [OUT/'window28-plan.json',OUT/'window27-mask-manifest.json',OUT/'window27-policy.json',OUT/'window24-label-recipe.json']}}

def evaluate(inputs,pred):
 policy=inputs['recipe']['evaluation'];old=inputs['old'];stored={(r['model'],r['signal_session']):r for r in inputs['old_daily']};daily=[];pairs=[];summary=[]
 for year,h in ((2022,5),(2022,10),(2023,5),(2023,10)):
  role='VALIDATION' if year==2022 else 'FORWARD_DIAGNOSTIC';name=f'Ridge-{h}-{role}';a=old[name];b=pred[name];same_support(a,b)
  da=defaultdict(list);db=defaultdict(list)
  for r in a:da[r['signal_session']].append(r)
  for r in b:db[r['signal_session']].append(r)
  assert set(da)==set(db);days=sorted(da);group=[]
  for day in days:
   q=paired_daily(da[day],db[day],None,day,h,policy)  # None removes no security; no LOSO.
   assert q['L0']['rank_ic']==stored[(name,day)]['rank_ic'] and q['L0']['support_sha256']==stored[(name,day)]['support_sha256']
   assert q['L0']['valid_n']>=20 and q['Ridge']['valid_n']==q['L0']['valid_n']
   item={'year':year,'horizon':h,'signal_session':day,'valid_n':q['L0']['valid_n'],'support_sha256':q['L0']['support_sha256'],'old_Ridge':q['L0']['rank_ic'],'clean_Ridge':q['Ridge']['rank_ic'],'delta':q['delta'],'reasons':{'old':q['L0']['reasons'],'clean':q['Ridge']['reasons']}};pairs.append(item);group.append(item)
   for who,metric,rows in (('W27_OLD_RIDGE',q['L0'],da[day]),('W28_CLEAN_RIDGE',q['Ridge'],db[day])):
    daily.append({**metric,'model':who,'year':year,'score_ties_n':len(rows)-len({r['score'] for r in rows}),'label_ties_n':len(rows)-len({r['label'] for r in rows})})
  joint=[r for r in group if r['delta'] is not None];mean=lambda field:sum(r[field] for r in joint)/len(joint) if joint else None
  summary.append({'year':year,'horizon':h,'requested_days':len(days),'defined_paired_days':len(joint),'old_mean_RankIC':mean('old_Ridge'),'clean_mean_RankIC':mean('clean_Ridge'),'mean_paired_delta':mean('delta'),'daily_N':dict(Counter(r['valid_n'] for r in group)),'common_support_sha256':canonical_hash([r['support_sha256'] for r in group]),'evidence_class':'EXPOSED_DEVELOPMENT_DIAGNOSTIC_NOT_MODEL_SELECTION'})
 return daily,pairs,summary

def causality(inputs,models):
 cases=[];days=inputs['days'];plan=inputs['plan']
 for sec in plan['poison_cases']['stocks']:
  raw=inputs['raw'][sec];original_hash=canonical_hash(raw)
  for signal in plan['poison_cases']['signals']:
   poisoned,n=poison_future(raw,signal);assert n>0 and canonical_hash({d:p for d,p in raw.items() if d<=signal})==canonical_hash({d:p for d,p in poisoned.items() if d<=signal})
   for h in plan['poison_cases']['horizons']:
    before=features(signal,h,days,raw);after=features(signal,h,days,poisoned);a=ridge_predict(models[str(h)]['state'],np.asarray([before]));b=ridge_predict(models[str(h)]['state'],np.asarray([after]));np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(a,b)
    control=copy.deepcopy(raw)
    for field in ('open','max','min','close'):control[signal][field]*=2
    assert features(signal,h,days,control)!=before,'VACUOUS_CAUSAL_TEST'
    cases.append({'security':sec,'signal':signal,'horizon':h,'future_rows_modified':n,'features_before':before,'features_after':after,'prediction_before':a.tolist(),'prediction_after':b.tolist(),'past_prefix_sha256':canonical_hash({d:p for d,p in raw.items() if d<=signal}),'poisoned_future_sha256':canonical_hash({d:p for d,p in poisoned.items() if d>signal}),'positive_control':'CURRENT_SESSION_CHANGE_DETECTED','result':'PASS_EXACT'})
   assert canonical_hash(raw)==original_hash,'ORIGINAL_RAW_MUTATED'
 assert len(cases)==plan['poison_cases']['maximum_cases']==18
 return {'status':'PROGRAMMATIC_CAUSALITY_PASS_NOT_HISTORICAL_ASOF_CERTIFICATION','cases':cases,'real_raw_mutation':False,'model_state_sha256':canonical_hash(models)}

def predictions(inputs,models):
 result={}
 for name,rows in inputs['old'].items():
  if name.startswith('L0-'):result[name]=rows;continue
  h=str(rows[0]['horizon']);x=np.asarray([inputs['index'][key(r)]['features'] for r in rows]);scores=ridge_predict(models[h]['state'],x);assert np.isfinite(scores).all()
  result[name]=[{**r,'score':float(score),'run_id':models[h]['run_id'],'model_sha256':canonical_hash(models[h]),'prediction_role':inputs['index'][key(r)]['split']} for r,score in zip(rows,scores)]
  same_support(rows,result[name])
 return result

def execute(mode):
 inputs=load_inputs();proof=preflight(inputs)
 if mode=='preflight':put(OUT/'window28-prefit.json',proof);print(json.dumps(proof));return
 code={str(p.relative_to(REPO)):file_sha256(p) for p in [Path(__file__),OUT/'window28-plan.json',REPO/'V6/experimental/v7_classical_baseline.py',REPO/'V6/experimental/v7_retrospective_smoke.py',REPO/'V6/experimental/v7_research_evaluation.py',REPO/'V6/experimental/v7_robustness_audit.py']}
 if mode=='run':
  if (ROOT/'COMPLETE.json').exists():raise ValueError('W28_ALREADY_COMPLETE_NO_REFIT')
  assert json.loads((OUT/'window28-prefit.json').read_text())==proof
  models={}
  for h in inputs['plan']['horizons']:
   rows=clean_training(inputs['data'],inputs['mask'],h);x=np.asarray([r['features'] for r in rows]);y=np.asarray([r['diagnostic_label'] for r in rows]);reserve_fit(ROOT,h)
   state=ridge_fit_state(x,y,inputs['plan']['ridge']['alpha'])  # exactly one real fit for this horizon
   np.testing.assert_array_equal(state['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(state['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
   models[str(h)]={'run_id':f'w28-fresh-clean-Ridge-{h}-train2021','recipe_id':inputs['plan']['recipe_id'],'horizon':h,'initialization':'FRESH_STATE','resume_checkpoint':None,'train_n':len(rows),'old_train_n':inputs['plan']['expected_training_counts'][str(h)]['old'],'train_view_sha256':canonical_hash(rows),'feature_sha256':canonical_hash(inputs['recipe']['features']),'source_label_contract_sha256':canonical_hash(inputs['recipe']['label_formula']),'safety_policy_sha256':inputs['W27_closed']['policy_sha256'],'recipe':inputs['plan']['ridge'],'latest_label_exit':max(r['exit'] for r in rows),'state':state,'scaling_sha256':canonical_hash(state['scaling'])}
   put(ROOT/f'model-{h}.json',models[str(h)])
 else:
  complete=json.loads((ROOT/'COMPLETE.json').read_text());assert complete['code']==code
  for ref in complete['artifacts'].values():S['checked'](ref)
  models={str(h):json.loads((ROOT/f'model-{h}.json').read_text()) for h in inputs['plan']['horizons']}
  assert models==S['checked'](complete['artifacts']['models']), 'MODEL_STATE_CHANGED'
  for h in inputs['plan']['horizons']:
   rows=clean_training(inputs['data'],inputs['mask'],h);assert canonical_hash(rows)==models[str(h)]['train_view_sha256'];x=np.asarray([r['features'] for r in rows]);np.testing.assert_array_equal(models[str(h)]['state']['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(models[str(h)]['state']['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
 pred=predictions(inputs,models);daily,paired,summary=evaluate(inputs,pred);poison=causality(inputs,models)
 if mode=='replay':
  for name,value in {'models':models,'predictions':pred,'daily':daily,'paired':paired,'summary':summary,'causality':poison}.items():assert canonical_hash(value)==canonical_hash(S['checked'](complete['artifacts'][name])),'REPLAY_MISMATCH:'+name
  result={'status':'INDEPENDENT_PROCESS_REPLAY_PASS_NO_REFIT','daily_rows':len(daily),'paired_rows':len(paired),'causality_cases':len(poison['cases']),'prediction_rows_recomputed':sum(len(v) for k,v in pred.items() if k.startswith('Ridge-')),'model_fit_calls':0,'scaling_train_only_verified':True,'original_private_files_verified':inputs['prior_private_files']};put(OUT/'window28-replay.json',result);print(json.dumps(result));return
 refs={name:put(ROOT/(name+'.json'),value) for name,value in {'models':models,'predictions':pred,'daily':daily,'paired':paired,'summary':summary,'causality':poison}.items()}
 complete={'schema':'w28-clean-refit-diagnostic-v1','status':'COMPLETE_DEVELOPMENT_ONLY_NOT_ADMITTED','study_id':inputs['plan']['study_id'],'source_commit':inputs['plan']['source_commit'],'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'code':code,'artifacts':refs,'plan_sha256':canonical_hash(inputs['plan']),'fit_count':2,'resume':False,'source_data_sha256':inputs['W24_run']['identity']['data_sha256'],'feature_sha256':canonical_hash(inputs['recipe']['features']),'W27_policy_sha256':inputs['W27_closed']['policy_sha256'],'W27_mask_sha256':json.loads((W27/'FROZEN_MASK.json').read_text())['mask']['sha256'],'W27_prediction_subset_sha256':inputs['W27_closed']['artifacts']['frozen_prediction_subset']['sha256'],'models':{h:{'run_id':m['run_id'],'model_sha256':canonical_hash(m),'train_n':m['train_n'],'old_train_n':m['old_train_n'],'scaling_sha256':m['scaling_sha256']} for h,m in models.items()},'actual_generated_at':datetime.now(timezone.utc).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU','threads':'OMP/MKL/OPENBLAS=1'},'API_GPU_search':0,'admission':False,'backup':'BACKUP_PENDING_SAME_HOST'}
 put(ROOT/'COMPLETE.json',complete);put(OUT/'window28-run.json',complete);put(OUT/'window28-results.json',summary);put(OUT/'window28-causality.json',poison);print(json.dumps({'models':complete['models'],'summary':summary,'causality':poison['status']}))
if __name__=='__main__':
 if len(sys.argv)!=2 or sys.argv[1] not in ('preflight','run','replay'):raise SystemExit('choose preflight/run/replay')
 execute(sys.argv[1])
