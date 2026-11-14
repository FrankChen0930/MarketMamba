"""One fixed W24 pilot: source/mask preparation, conditional fit, offline replay.
No network/scheduler in this driver. Numerical models, RankIC and publisher reused.
"""
import sys,json,hashlib,platform,runpy
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_finmind_proof import put,verified_record,diagnose
from V6.experimental.v7_expanded_pilot import exploratory_mask,coverage_gate
from V6.experimental.v7_classical_baseline import ridge_fit_state,ridge_predict
from V6.experimental.v7_research_evaluation import daily_metric,aggregate
from V6.experimental.v7_research_artifacts import publish_bundle,verify_bundle,run_key,artifact_path
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7'
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot')
W22=Path('/home/frank/projects/MarketMamba/.artifacts/window22-historical-panel');W23=Path('/home/frank/projects/MarketMamba/.artifacts/window23-calendar-smoke')
CONTRACT=json.loads((OUT/'window24-label-recipe.json').read_text());COHORT=json.loads((ROOT/'FROZEN_COHORT.json').read_text())
DAILY_IDENTITY=runpy.run_path(str(OUT/'window23-pipeline.py'))['daily_identity']
CLASS='RETROSPECTIVE_EXPLORATORY_LIMITED'
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
def immutable(path,obj):
 b=enc(obj);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b,'IMMUTABLE_OUTPUT_CHANGED'
 else:
  with path.open('xb') as f:f.write(b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def pin_check():
 pin=json.load(open(ROOT/'COHORT_PIN.json'));assert canonical_hash(COHORT)==pin['semantic_sha256']==CONTRACT['cohort_semantic_sha256']
 assert file_sha256(ROOT/'FROZEN_COHORT.json')==pin['file_sha256'];assert json.load(open(OUT/'window24-cohort.json'))==COHORT
 assert json.load(open(ROOT/'source-at-capture/FROZEN_LABEL_RECIPE.json'))==CONTRACT
 for name in ['window22-local-artifacts.json','window23-local-artifacts.json']:
  for f in json.load(open(OUT/name))['files']:
   p=Path(f['path']);assert p.is_file() and file_sha256(p)==f['sha256'] and p.stat().st_size==f['bytes'],'PRIOR_RAW_EVIDENCE_NOT_RESTORABLE'

def build():
 pin_check();assert (ROOT/'finmind/CLOSED.json').is_file()
 calendar=json.load(open(W23/'diagnostics/calendar.json'));sessions=[x['date'] for x in calendar if x['price_observed'] and x['engineering_calendar_ok']];assert len(sessions)==729
 assert dict(Counter(d[:4] for d in sessions))=={'2021':244,'2022':246,'2023':239}
 new={};sources=[]
 for p in sorted((ROOT/'finmind/records').glob('*.json')):
  r=verified_record(ROOT/'finmind',str(p.relative_to(ROOT/'finmind')),file_sha256(p));assert r['cohort_sha256']==CONTRACT['cohort_semantic_sha256'];new[r['params']['data_id']]=(ROOT/'finmind',p,r)
 old={}
 for p in (W22/'finmind/records').glob('*.json'):
  r=json.load(open(p))
  if r['params']['data_id'] in COHORT['existing_W22_reuse']:old[r['params']['data_id']]=(W22/'finmind',p,verified_record(W22/'finmind',str(p.relative_to(W22/'finmind')),file_sha256(p)))
 masks=[];data=[];coverage=[];allrows=[]
 for c in COHORT['cohort']:
  sec=c['security_id'];item=old.get(sec) or new.get(sec);rows=[];r=None
  if item:
   base,p,r=item;rows=[x for x in json.loads((base/r['raw_path']).read_text())['data'] if '2021-01-01'<=x['date']<='2023-12-31']
   sources.append({'security':sec,'record_path':str(p),'record_sha256':file_sha256(p),'raw_path':str(base/r['raw_path']),'raw_sha256':r['raw_sha256'],'raw_bytes':r['raw_bytes'],'retrieved_at':r['retrieved_at'],'vintage':'CURRENT_RETROSPECTIVE_NOT_ORIGINAL_ASOF','reused_W22':sec in old})
  prices={x['date']:x for x in rows};assert len(prices)==len(rows),'DUPLICATE_SECURITY_SESSION';allrows.extend(rows)
  qc=diagnose(enc({'status':200,'data':rows}),{'dataset':'TaiwanStockPrice','data_id':sec,'start_date':'2021-01-01','end_date':'2023-12-31'})
  missing=sorted(set(sessions)-set(prices));extra=sorted(set(prices)-set(sessions));counts=Counter();reasoncounts=Counter()
  for day in sessions:
   split='TRAIN' if day<'2022-01-01' else 'VALIDATION' if day<'2023-01-01' else 'FORWARD_DIAGNOSTIC';boundary='2022-01-01' if split=='TRAIN' else '2023-01-01' if split=='VALIDATION' else '2024-01-01'
   for h in [5,10]:
    q=exploratory_mask(signal=day,horizon=h,sessions=sessions,prices=prices,cohort=c,next_split=boundary,known_action_dates=[])
    q.update(stock_id=sec,split=split,source_record_sha256=file_sha256(item[1]) if item else None,feature_decision_logical=day+'T13:30:00+08:00',actual_historical_publication='UNKNOWN')
    masks.append(q);reasoncounts.update(q['exploratory_reasons']);counts['engineering']+=q['engineering_eligible'];counts['exploratory']+=q['exploratory_eligible']
    if q['exploratory_eligible']:data.append(q)
  coverage.append({'security':sec,'selection_role':c['selection_role'],'rows':len(rows),'earliest':min(prices) if prices else None,'latest':max(prices) if prices else None,'missing_sessions':missing,'extra_sessions':extra,'qc':qc['counts'],'listing_start':c['reported_twse_start'],'delisting_end':c['reported_end'],'mask_counts':dict(counts),'exclusion_reasons':dict(reasoncounts),'daily_suspension_evidence':'UNKNOWN','corporate_actions':'UNKNOWN','source_present':item is not None})
 expected={};daily_stock_counts=Counter((r['split'],r['signal_session'],r['horizon']) for r in data)
 for i,day in enumerate(sessions):
  if day<'2022-01-01':continue
  split='VALIDATION' if day<'2023-01-01' else 'FORWARD_DIAGNOSTIC';boundary='2023-01-01' if split=='VALIDATION' else '2024-01-01'
  for h in [5,10]:
   if i+h<len(sessions) and sessions[i+h]<boundary:expected[f'{split}:{day}:{h}']=daily_stock_counts[(split,day,h)]
 gate=coverage_gate(expected,CONTRACT['evaluation']['min_valid_n']) and len(COHORT['cohort'])>=30 and len(new)==COHORT['planned_price_requests'] and not any(x['extra_sessions'] for x in coverage)
 refs={'dataset':immutable(ROOT/'derived/dataset.json',data),'masks':immutable(ROOT/'derived/three-masks.json',masks),'raw_panel':immutable(ROOT/'derived/panel.json',allrows),'calendar':{'path':str(W23/'diagnostics/calendar.json'),'sha256':file_sha256(W23/'diagnostics/calendar.json')},'sources':immutable(ROOT/'derived/source-refs.json',sources),'common_support_counts':immutable(ROOT/'derived/common-support-counts.json',expected)}
 summary={'schema':'w24-coverage-v1','cohort_sha256':canonical_hash(COHORT),'contract_sha256':canonical_hash(CONTRACT),'source_base':COHORT['source_commit'],'cohort_n':len(COHORT['cohort']),'actual_finmind_requests':len(new),'metadata':0,'reused_W22':len(old),'source_records':len(sources),'raw_rows_2021_2023':len(allrows),'raw_response_bytes_new':sum(x[2]['raw_bytes'] for x in new.values()),'sessions':len(sessions),'mask_rows':len(masks),'mask_totals':{'engineering':sum(m['engineering_eligible'] for m in masks),'exploratory':len(data),'strict':0},'eligible_securities':sorted({r['stock_id'] for r in data}),'split_horizon_counts':dict(Counter(r['split']+'_'+str(r['horizon']) for r in data)),'daily_core_N_distribution':dict(sorted(Counter(expected.values()).items())),'daily_core_groups':len(expected),'core_groups_below20':sum(n<20 for n in expected.values()),'fit_gate':bool(gate),'coverage':coverage,'references':refs,'all_strict_reasons':{'HISTORICAL_AVAILABILITY_UNKNOWN':len(masks),'COMPLETE_ACTION_HISTORY_UNKNOWN':len(masks),'DAILY_EXECUTABLE_UNIVERSE_UNKNOWN':len(masks)},'label_class':CONTRACT['label_evidence_class'],'research_admitted':False,'sample':'HINDSIGHT_SELECTED_NOT_HISTORICAL_FULL_MARKET','new_requests_closed':True}
 put(OUT/'window24-coverage.json',summary);print(json.dumps({k:summary[k] for k in ['raw_rows_2021_2023','mask_totals','daily_core_N_distribution','fit_gate']}));return summary

def shared_metrics(predictions,policy):
 results=[]
 for model in sorted(predictions):
  rows=predictions[model]
  for day,h in sorted({(r['signal_session'],r['horizon']) for r in rows}):
   subset=[r for r in rows if r['signal_session']==day and r['horizon']==h];q=daily_metric(rows,day,h,policy)
   results.append({'model':model,**q,'score_ties_n':len(subset)-len({r['score'] for r in subset}),'label_ties_n':len(subset)-len({r['label'] for r in subset}),'evidence_class':CLASS})
 return results

def paired(daily):
 idx={(r['model'],r['signal_session']):r for r in daily};result=[]
 for r in daily:
  if not r['model'].startswith('L0-'):continue
  other=idx[(r['model'].replace('L0-','Ridge-',1),r['signal_session'])]
  assert r['support_sha256']==other['support_sha256'] and r['stock_ids']==other['stock_ids'] and r['valid_n']==other['valid_n'],'UNFAIR_SUPPORT'
  result.append({'signal_session':r['signal_session'],'role_horizon':r['model'][3:],'valid_n':r['valid_n'],'support_sha256':r['support_sha256'],'L0':r['rank_ic'],'Ridge':other['rank_ic'],'delta':other['rank_ic']-r['rank_ic'] if r['rank_ic'] is not None and other['rank_ic'] is not None else None})
 return result

def run():
 pin_check();summary=json.load(open(OUT/'window24-coverage.json'));assert summary['fit_gate'],'COVERAGE_INSUFFICIENT_NO_FIT'
 assert not (ROOT/'FIT_STARTED.json').exists(),'ONE_SHOT_ONLY_NO_REFIT';immutable(ROOT/'FIT_STARTED.json',{'cohort':canonical_hash(COHORT),'contract':canonical_hash(CONTRACT),'scope':CLASS})
 ref=summary['references']['dataset'];assert file_sha256(Path(ref['path']))==ref['sha256'];data=json.load(open(ref['path']));fits={};predictions={};logs=[];policy=CONTRACT['evaluation']
 for h in [5,10]:
  train=[r for r in data if r['horizon']==h and r['split']=='TRAIN'];assert train and all(r['exit']<'2022-01-01' and r['exploratory_eligible'] and not r['strict_eligible'] for r in train)
  x=np.array([r['features'] for r in train],dtype=np.float64);y=np.array([r['diagnostic_label'] for r in train],dtype=np.float64);state=ridge_fit_state(x,y,CONTRACT['Ridge']['alpha']);fits[str(h)]={'state':state,'train_n':len(train),'train_view_sha256':canonical_hash(train),'signal_range':[min(r['signal_session'] for r in train),max(r['signal_session'] for r in train)],'latest_label_exit':max(r['exit'] for r in train),'run_id':f'w24-fixed-Ridge-{h}-train2021','resume':False,'research_admitted':False};frozen=canonical_hash(fits)
  for role in ['VALIDATION','FORWARD_DIAGNOSTIC']:
   rows=[r for r in data if r['split']==role and r['horizon']==h];xx=np.array([r['features'] for r in rows]);scores={'L0':xx[:,0],'Ridge':ridge_predict(state,xx)}
   for model,values in scores.items():
    assert np.isfinite(values).all();predictions[f'{model}-{h}-{role}']=[{'stock_id':r['stock_id'],'signal_session':r['signal_session'],'horizon':h,'score':float(value),'label':r['diagnostic_label'],'label_id':canonical_hash({'stock':r['stock_id'],'session':r['signal_session'],'h':h,'contract':canonical_hash(CONTRACT)}),'label_metadata_sha256':canonical_hash(r),'label_class':r['label_class']} for r,value in zip(rows,values)]
  assert canonical_hash(fits)==frozen;logs.append({'horizon':h,'train_n':len(train),'selection':False,'refit':False,'search':False,'forward_labels_used_in_fit':False,'fit':'existingRidgeclosedformCPU'})
 daily=shared_metrics(predictions,policy);comparison=paired(daily)
 payloads={'resolved_config':('RESOLVED_CONFIG',{'contract':CONTRACT,'cohort':COHORT}),'environment':('ENVIRONMENT',{'python':platform.python_version(),'numpy':np.__version__,'gpu':False,'threads':1}),'dataset_snapshot':('DATASET_SNAPSHOT',data),'data_split_eligibility':('DATA_SPLIT_ELIGIBILITY',summary),'training_logs':('TRAINING_LOGS',logs),'models':('CHECKPOINT_REFERENCE',fits),'predictions':('PREDICTIONS',predictions),'daily_metrics':('DAILY_METRICS',daily),'paired_comparison':('PAIRED_COMPARISON',comparison),'rejection_ledger':('REJECTION_LEDGER',json.load(open(summary['references']['masks']['path']))),'completion_evidence':('COMPLETION_EVIDENCE',{'scope':CLASS,'real_ridge_fits':2,'formal_research_admitted':False,'strict_pit':False})}
 sources={};arts=[]
 for name,(kind,obj) in payloads.items():
  obj={'schema':'w24-'+name+'-v1','evidence_class':CLASS,'value':obj};b=enc(obj);sources[name]=b;arts.append({'name':name,'kind':kind,'storage':'BUNDLE','path':name+'.json','sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'format':'JSON','schema_id':obj['schema']})
 codepaths=['V6/experimental/v7_classical_baseline.py','V6/experimental/v7_research_artifacts.py','V6/experimental/v7_research_evaluation.py','V6/experimental/v7_retrospective_smoke.py','V6/experimental/v7_expanded_pilot.py','research/v7/window24-pipeline.py','research/v7/window23-pipeline.py'];code={p:file_sha256(REPO/p) for p in codepaths}
 identity={'study_id':'w24-retrospective-exploratory-limited','recipe_id':'w24-fixed-W23-l0-ridge-no-retuning','run_id':'w24-one-shot-2horizon-CPU','fold_id':'train2021-validation2022-exposed2023','origin_id':'logical-forward2023-not-actual-generation','origin_at':'2023-01-01T00:00:00+08:00','seed':0,'code_commit':COHORT['source_commit'],'code_manifest_sha256':canonical_hash(code),'data_sha256':canonical_hash(data),'feature_sha256':canonical_hash(CONTRACT['features']),'label_sha256':canonical_hash(CONTRACT['label_formula']),'universe_sha256':canonical_hash(COHORT),'evaluation_policy_sha256':canonical_hash(policy),'model_sha256':canonical_hash(fits),'training_contract_sha256':canonical_hash(CONTRACT),'prediction_role':'FORWARD','checkpoint_status':'NOT_GENERATED_TINY_CLASSICAL_NO_RESUME','checkpoint_sha256':None,'source_vintages':[{'source_sha256':ref['sha256'],'vintage_id':'W24-current-retrospective-raw-proxy-not-asof'}]}
 manifest={'schema':'v7-study-run-bundle-v2','profile':CLASS,'evidence_class':CLASS,'training_status':'CPU_EXPLORATORY_PILOT_COMPLETED','authority_scope':'HumanWindow24_LIMITED_EXPLORATORY_PILOT','research_admitted':False,'identity':identity,'required_artifacts':list(sources),'artifacts':arts};result=publish_bundle(ROOT/'bundles',manifest,canonical_hash(manifest),sources);assert result['status']=='ARTIFACTS_VERIFIED',result
 summaries={name:{**aggregate([r for r in daily if r['model']==name],policy),'N_min':min(r['valid_n'] for r in daily if r['model']==name),'N_max':max(r['valid_n'] for r in daily if r['model']==name),'score_ties_total':sum(r['score_ties_n'] for r in daily if r['model']==name),'label_ties_total':sum(r['label_ties_n'] for r in daily if r['model']==name)} for name in predictions}
 runinfo={'status':'LIMITED_EXPLORATORY_RAW_PROXY_PILOT_COMPLETED_NOT_FORMAL_ADMISSION','identity':identity,'code_files':code,'bundle_key':run_key(identity),'manifest_sha256':canonical_hash(manifest),'bundle_path':result['bundle_path'],'ridge_fits':2,'daily_metric_rows':len(daily),'paired_rows':len(comparison),'period_diagnostics':summaries,'source_limits':'hindsight/current-version/raw actions unknown;2022/2023 alreadyexposed;no superiority/investingclaim','actual_generated_at':json.load(open(Path(result['bundle_path'])/'publication.json'))['generated_at'],'logical_origin_is_not_actual_generated_time':True}
 immutable(ROOT/'PILOT_CLOSED.json',runinfo);put(OUT/'window24-run-summary.json',runinfo);print(json.dumps({'daily':len(daily),'paired':len(comparison),'summary':summaries}));return runinfo

def replay():
 pin_check();run=json.load(open(ROOT/'PILOT_CLOSED.json'));v=verify_bundle(ROOT/'bundles',run['bundle_key'],run['manifest_sha256']);assert v['status']=='ARTIFACTS_VERIFIED'
 for p,pin in run['code_files'].items():assert file_sha256(REPO/p)==pin,'CODE_IDENTITY_CHANGED'
 ds={x['name']:x for x in v['manifest']['artifacts']};load=lambda name:json.load(open(artifact_path(v['bundle_path'],ds[name])))['value'];data=load('dataset_snapshot');fits=load('models');pred=load('predictions');config=load('resolved_config');assert config=={'contract':CONTRACT,'cohort':COHORT}
 assert all(r['exploratory_eligible'] and not r['strict_eligible'] and not r['formal_research_admitted'] for r in data)
 index={(r['stock_id'],r['signal_session'],r['horizon']):r for r in data};assert len(index)==len(data)
 for h in [5,10]:
  train=[r for r in data if r['split']=='TRAIN' and r['horizon']==h];model=fits[str(h)];assert canonical_hash(train)==model['train_view_sha256'] and all(r['exit']<'2022-01-01' for r in train)
  np.testing.assert_array_equal(np.mean([r['features'] for r in train],axis=0),model['state']['scaling']['mean'])
 for name,rows in pred.items():
  keys=set();h=int(name.split('-')[1]);role=name.split('-',2)[2];expected=[r for r in data if r['split']==role and r['horizon']==h];assert len(rows)==len(expected)
  for r in rows:
   k=(r['stock_id'],r['signal_session'],r['horizon']);assert k not in keys;keys.add(k);sample=index[k];assert sample['split']==role and sample['diagnostic_label']==r['label'] and canonical_hash(sample)==r['label_metadata_sha256']
  xx=np.array([index[(r['stock_id'],r['signal_session'],h)]['features'] for r in rows]);scores=xx[:,0] if name.startswith('L0-') else ridge_predict(fits[str(h)]['state'],xx);np.testing.assert_array_equal(scores,[r['score'] for r in rows])
 actual=shared_metrics(pred,CONTRACT['evaluation']);expected=load('daily_metrics');assert DAILY_IDENTITY(actual)==DAILY_IDENTITY(expected),'DAILY_METRIC_MISMATCH';comparison=paired(actual);assert canonical_hash(comparison)==canonical_hash(load('paired_comparison')),'PAIRED_MISMATCH'
 summary=load('data_split_eligibility')
 for src in json.load(open(summary['references']['sources']['path'])):
  assert file_sha256(Path(src['raw_path']))==src['raw_sha256'];assert file_sha256(Path(src['record_path']))==src['record_sha256']
 result={'status':'LIMITED_EXPLORATORY_EVALUATION_REPLAYABLE','independent_process':True,'bundle_key':run['bundle_key'],'manifest_sha256':run['manifest_sha256'],'daily_rows':len(actual),'paired_rows':len(comparison),'metric_identity':DAILY_IDENTITY(actual),'prediction_scores_recomputed_without_refit':True,'common_support_verified':True,'source_raw_hashes_verified':36,'research_admitted':False,'strict_pit':False,'untouched_test':False};put(OUT/'window24-replay.json',result);print(json.dumps(result))
if __name__=='__main__':{'build':build,'run':run,'replay':replay}[sys.argv[1]]()
