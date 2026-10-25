"""Bounded Window23 offline build/run/replay driver. No network or scheduling.

This is a single engineering recipe, not a trainer/evaluator/registry replacement.
All vendor payloads, labels, predictions and daily diagnostics stay in private root.
"""
import sys,json,csv,io,hashlib,platform,subprocess
from pathlib import Path
from datetime import date,timedelta
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_finmind_proof import put,verified_record
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_retrospective_smoke import schedule_row_role,engineering_sample
from V6.experimental.v7_classical_baseline import ridge_fit_state,ridge_predict
from V6.experimental.v7_research_evaluation import daily_metric
from V6.experimental.v7_research_artifacts import publish_bundle,verify_bundle,run_key,artifact_path
import numpy as np
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window23-calendar-smoke')
W22=Path('/home/frank/projects/MarketMamba/.artifacts/window22-historical-panel')
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7';CONTRACT=json.loads((OUT/'window23-smoke-contract.json').read_text())
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
def save_private(path,obj):
 b=enc(obj);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b,'NO_IMMUTABLE_OVERWRITE'
 else:
  with path.open('xb') as f:f.write(b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size}
def build():
 # Read-only prior source integrity pins, not guessed reconstruction.
 for f in json.loads((OUT/'window22-local-artifacts.json').read_text())['files']:
  p=Path(f['path']);assert file_sha256(p)==f['sha256'] and p.stat().st_size==f['bytes']
 records=[verified_record(W22/'finmind',str(p.relative_to(W22/'finmind')),file_sha256(p)) for p in sorted((W22/'finmind/records').glob('*.json'))]
 raw={r['params']['data_id']:json.loads((W22/'finmind'/r['raw_path']).read_text())['data'] for r in records};ref=records[0]['qc']['sessions'];assert ref==records[1]['qc']['sessions']
 declared={};calendar_qualified={};schedule_pins={}
 for y in range(2019,2024):
  r=json.loads((ROOT/'official-docs'/f'{y}.record.json').read_text());p=Path(r['raw']);rows=json.loads(p.read_text())['data'];schedule_pins[str(y)]=file_sha256(p)
  for row in rows:declared[row[0]]={'role':schedule_row_role(row),'source_sha256':file_sha256(p),'title':row[1]}
  d=date(y,1,1);expected=[]
  while d.year==y:
   day=d.isoformat()
   if d.weekday()<5 and declared.get(day,{}).get('role')!='OFFICIAL_CLOSURE':expected.append(day)
   d+=timedelta(days=1)
  if y==2023:
   # Official PDF bytes pinned; text independently read via web PDF parser (request10).
   r2=json.loads((ROOT/'official-docs/closure20230803.record.json').read_text());declared['2023-08-03']={'role':'TEMPORARY_CLOSURE','source_sha256':file_sha256(Path(r2['raw'])),'title':'TWSE20230802pressrelease;20230803full-dayclosure'};expected.remove('2023-08-03')
  observed=[d for d in ref if d.startswith(str(y))]
  calendar_qualified[str(y)]={'nonempty_official_schedule':bool(rows),'observed_matches_schedule_with_documented_closure':bool(rows) and observed==expected,'observed_n':len(observed),'expected_n':len(expected) if rows else None,'not_strict_pit_admission':True}
 days=[];d=date(2019,1,1)
 while d<=date(2023,12,31):
  day=d.isoformat();event=declared.get(day)
  days.append({'date':day,'TWSE_role':event['role'] if event else 'OBSERVED_MARKET_SESSION' if day in ref else 'UNKNOWN','source_sha256':event['source_sha256'] if event else schedule_pins[day[:4]],'price_observed':day in ref,'engineering_calendar_ok':calendar_qualified[day[:4]]['observed_matches_schedule_with_documented_closure'],'TPEX_role':'OBSERVED_MARKET_SESSION' if day in ref else 'UNKNOWN','TPEX_full_calendar':'NOT_VERIFIED;2023PDFarchived_no_full_machine_parse'})
  d+=timedelta(days=1)
 calref=save_private(ROOT/'diagnostics/calendar.json',days)
 life=json.loads((OUT/'window22-lifecycle.json').read_text())['rows'];life={r['security']:r for r in life};membership=[];samples=[];masks=[]
 smoke_days=[d for d in ref if calendar_qualified[d[:4]]['observed_matches_schedule_with_documented_closure']]
 for sec,rows in raw.items():
  prices={r['date']:r for r in rows};start=life[sec]['reported_current_market_start'];end=life[sec]['reported_delisting']
  for day in ref:
   role='UNKNOWN'
   if end and day>=end:role='DELISTED'
   elif sec=='2499' and '2020-09-30'<=day<end:role='SUSPENDED'
   elif start and day>=start['start']:role='TWSE_LISTED' if start['market']=='TWSE' else 'TPEX_LISTED'
   elif (sec,day) in [('6515','2020-03-19'),('8272','2023-10-17')]:role='EMERGING'
   membership.append({'security':sec,'session':day,'role':role,'daily_tradability':'UNKNOWN','evidence':'W22officialboundary;exactW21emergingexample;2499officialknownsuspensionbydate'})
   if day not in prices:continue
   for h in [5,10]:
    q=engineering_sample(signal=day,horizon=h,sessions=smoke_days,prices=prices,market=role,calendar_ok=calendar_qualified[day[:4]]['observed_matches_schedule_with_documented_closure'],known_action_dates=[])
    q['stock_id']=sec;masks.append(q)
    if q['engineering_eligible']:
     split='TRAIN' if day<'2022-01-01' else 'VALIDATION' if day<'2023-01-01' else 'FORWARD_DIAGNOSTIC'
     boundary='2022-01-01' if split=='TRAIN' else '2023-01-01' if split=='VALIDATION' else '2024-01-01'
     if q['exit']>=boundary:q['engineering_eligible']=False;q['reasons'].append('LABEL_CROSSES_SPLIT_BOUNDARY')
     else:samples.append({**q,'split':split,'feature_decision_logical':day+'T13:30:00+08:00','publication_availability':'UNKNOWN_NOT_HISTORICAL_SNAPSHOT'})
 # Original W22 mask remains byte-identical; decompose overlapping causes separately.
 strict=list(csv.DictReader((W22/'derived/label-geometry.csv').open()));assert len(strict)==19728
 reasons=Counter(x for r in strict for x in r['reasons'].split('|') if x)
 old={(r['stock_id'],r['signal'],int(r['horizon'])):r for r in strict};classes=Counter()
 for q in masks:
  key=(q['stock_id'],q['signal_session'],q['horizon']);orig=old[key]
  category='A_PRICE_GEOMETRY_NOT_COMPUTABLE' if orig['structural_ready']=='False' else 'B_ENGINEERING_DIAGNOSTIC_ONLY_NOT_STRICT_PIT' if q['engineering_eligible'] else 'C_OTHER_DATA_OR_CONTRACT_BLOCKED'
  classes[category]+=1;q['failure_class']=category
 refs={'calendar':calref,'membership':save_private(ROOT/'diagnostics/membership.json',membership),'mask':save_private(ROOT/'diagnostics/engineering-mask.json',masks),'dataset':save_private(ROOT/'diagnostics/dataset.json',samples)}
 summary={'schema':'w23-diagnostics-v1','original_mask_rows':len(strict),'strict_eligible':0,'original_reason_counts_overlapping':dict(reasons),'PIT_NOT_VERIFIED':len(strict),'exclusive_failure_classes':dict(classes),'engineering_samples':len(samples),'split_horizon_counts':dict(Counter(r['split']+'_'+str(r['horizon']) for r in samples)),'securities':sorted({r['stock_id'] for r in samples}),'calendar_by_year':calendar_qualified,'calendar_roles':dict(Counter(x['TWSE_role'] for x in days)),'membership_roles':dict(Counter(x['role'] for x in membership)),'references':refs,'contract_sha256':canonical_hash(CONTRACT),'FinMind_requests':0,'official_document_requests':10,'no_further_requests':True,'corporate_actions':'COMPLETE_EVIDENCE_UNKNOWN; no qualified known-event feed; every raw diagnostic conditional','historical_universe':'HINDSIGHT_SAMPLE_NOT_FULL_MARKET','study_admitted':False}
 put(OUT/'window23-diagnostics-summary.json',summary);print(json.dumps({'samples':len(samples),'counts':summary['split_horizon_counts'],'classes':dict(classes)}))
 return summary

def metrics(predictions,policy):
 out=[]
 for model,rows in predictions.items():
  for day,h in sorted({(r['signal_session'],r['horizon']) for r in rows}):out.append({'model':model,**daily_metric(rows,day,h,policy),'evidence_class':'RETROSPECTIVE_ENGINEERING_SMOKE','not_model_quality_evidence':True})
 return out

def run():
 assert not (ROOT/'RUN_CLOSED.json').exists(),'ALREADY_EXECUTED_NO_REFIT'
 summary=json.loads((OUT/'window23-diagnostics-summary.json').read_text());ref=summary['references']['dataset'];assert file_sha256(Path(ref['path']))==ref['sha256'];data=json.loads(Path(ref['path']).read_text())
 assert set(r['split'] for r in data)=={'TRAIN','VALIDATION','FORWARD_DIAGNOSTIC'}
 policy=CONTRACT['evaluation'];fits={};predictions={};logs=[]
 for h in [5,10]:
  train=[r for r in data if r['split']=='TRAIN' and r['horizon']==h];assert train and all(r['exit']<'2022-01-01' for r in train)
  x=np.array([r['features'] for r in train]);y=np.array([r['diagnostic_label'] for r in train]);state=ridge_fit_state(x,y,CONTRACT['Ridge']['alpha']);fits[str(h)]={'state':state,'train_n':len(train),'train_latest_exit':max(r['exit'] for r in train),'train_signal_range':[min(r['signal_session'] for r in train),max(r['signal_session'] for r in train)],'train_data_sha256':canonical_hash(train),'run_id':f'w23-ridge-{h}-fixed-fit','research_admitted':False}
  frozen=canonical_hash(fits)
  for split in ['VALIDATION','FORWARD_DIAGNOSTIC']:
   rows=[r for r in data if r['split']==split and r['horizon']==h];xx=np.array([r['features'] for r in rows]);scores={'L0':xx[:,0],'Ridge':ridge_predict(state,xx)}
   for name,ss in scores.items():
    k=f'{name}-{h}-{split}';predictions[k]=[{'stock_id':r['stock_id'],'signal_session':r['signal_session'],'horizon':h,'score':float(s),'label':r['diagnostic_label'],'label_id':canonical_hash({'stock':r['stock_id'],'day':r['signal_session'],'h':h,'contract':summary['contract_sha256']}),'label_metadata_sha256':canonical_hash(r),'source_evidence':'ENGINEERING_DIAGNOSTIC_NOT_PIT'} for r,s in zip(rows,ss)]
  assert canonical_hash(fits)==frozen,'EVALUATION_MUTATED_FIT'
  logs.append({'horizon':h,'optimizer':'closed_form_Ridge_existing_core','n_train':len(train),'search':False,'selection':False,'refit':False,'forward_labels_used_in_fit':False})
 daily=metrics(predictions,policy)
 payloads={'resolved_config':('RESOLVED_CONFIG',CONTRACT),'environment':('ENVIRONMENT',{'python':platform.python_version(),'numpy':np.__version__,'gpu':False}),'data_split_eligibility':('DATA_SPLIT_ELIGIBILITY',summary),'dataset_snapshot':('DATASET_SNAPSHOT',data),'training_logs':('TRAINING_LOGS',logs),'models':('CHECKPOINT_REFERENCE',fits),'predictions':('PREDICTIONS',predictions),'daily_metrics':('DAILY_METRICS',daily),'rejection_ledger':('REJECTION_LEDGER',json.loads(Path(summary['references']['mask']['path']).read_text())),'completion_evidence':('COMPLETION_EVIDENCE',{'training_scope':'W23_RETROSPECTIVE_ENGINEERING_ONLY','ridge_fits':2,'research_admitted':False,'strict_pit_verified':False})}
 sources={};arts=[]
 for name,(kind,obj) in payloads.items():
  obj={'schema':'w23-'+name+'-v1','evidence_class':'RETROSPECTIVE_ENGINEERING_SMOKE','value':obj};b=enc(obj);sources[name]=b;arts.append({'name':name,'kind':kind,'storage':'BUNDLE','path':name+'.json','sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'format':'JSON','schema_id':obj['schema']})
 codefiles=['V6/experimental/v7_classical_baseline.py','V6/experimental/v7_research_artifacts.py','V6/experimental/v7_research_evaluation.py','V6/experimental/v7_retrospective_smoke.py','research/v7/window23-pipeline.py']
 ident={'study_id':CONTRACT['study_id'],'recipe_id':'w23-fixed-l0-ridge-v1','run_id':'w23-one-shot-cpu-smoke','fold_id':'2019-2021-mask-to2021__2022__2023','origin_id':'w23-logical-2023-forward-diagnostic','origin_at':'2023-01-01T00:00:00+08:00','seed':0,'code_commit':'7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470','code_manifest_sha256':canonical_hash({p:file_sha256(REPO/p) for p in codefiles}),'data_sha256':canonical_hash(data),'feature_sha256':canonical_hash(CONTRACT['features_order']),'label_sha256':canonical_hash(CONTRACT['diagnostic_formula']),'universe_sha256':canonical_hash(summary['securities']),'evaluation_policy_sha256':canonical_hash(policy),'model_sha256':canonical_hash(fits),'training_contract_sha256':canonical_hash(CONTRACT),'prediction_role':'FORWARD','checkpoint_status':'NOT_GENERATED_TINY_CLASSICAL_NO_RESUME','checkpoint_sha256':None,'source_vintages':[{'source_sha256':ref['sha256'],'vintage_id':'W22-current-retrospective-not-original-asof'}]}
 manifest={'schema':'v7-study-run-bundle-v2','profile':'RETROSPECTIVE_ENGINEERING_SMOKE','training_status':'CPU_ENGINEERING_SMOKE_COMPLETED','evidence_class':'RETROSPECTIVE_ENGINEERING_SMOKE','authority_scope':'HumanWindow23_LIMITED_ENGINEERING_SMOKE','research_admitted':False,'identity':ident,'required_artifacts':list(sources),'artifacts':arts}
 result=publish_bundle(ROOT/'bundles',manifest,canonical_hash(manifest),sources);assert result['status']=='ARTIFACTS_VERIFIED',result
 runsummary={'schema':'w23-smoke-run-v1','status':'ENGINEERING_SMOKE_COMPLETED_NOT_RESEARCH_ADMISSION','identity':ident,'manifest_sha256':canonical_hash(manifest),'bundle_key':run_key(ident),'bundle_path':result['bundle_path'],'daily_rows':len(daily),'defined_rankic_days':sum(r['rank_ic'] is not None for r in daily),'skipped_rankic_days':sum(r['rank_ic'] is None for r in daily),'valid_N_counts':dict(Counter(r['valid_n'] for r in daily)),'ridge_fits':2,'training_actual_period':{h:fit['train_signal_range'] for h,fit in fits.items()},'models':'L0andRidge fixed diagnostic only; no model choice','interpretation':'Pipeline engineering only; hindsight selection, raw prices/actions and availability limits; no quality comparison','publication_generated_at':'Actual W09publication timestamp, distinct from logicalorigin2023','code_files':{p:file_sha256(REPO/p) for p in codefiles}}
 save_private(ROOT/'RUN_CLOSED.json',runsummary);put(OUT/'window23-run-summary.json',runsummary);print(json.dumps({'bundle':run_key(ident),'daily':len(daily),'defined':runsummary['defined_rankic_days']}))

def daily_identity(rows):
 keys=[(r['model'],r['signal_session'],r['horizon']) for r in rows]
 if len(set(keys))!=len(keys):raise ValueError('DUPLICATE_DAILY_METRIC_KEY')
 return canonical_hash(sorted(rows,key=lambda r:(r['model'],r['signal_session'],r['horizon'])))

def replay():
 for f in json.loads((OUT/'window22-local-artifacts.json').read_text())['files']:
  p=Path(f['path']);assert p.is_file() and file_sha256(p)==f['sha256'] and p.stat().st_size==f['bytes'],'RAW_SOURCE_NOT_RESTORABLE'
 run=json.loads((ROOT/'RUN_CLOSED.json').read_text());v=verify_bundle(ROOT/'bundles',run['bundle_key'],run['manifest_sha256']);assert v['status']=='ARTIFACTS_VERIFIED',v
 ds={a['name']:a for a in v['manifest']['artifacts']};load=lambda name:json.loads(artifact_path(v['bundle_path'],ds[name]).read_text())['value']
 for rel,pin in run['code_files'].items():
  assert file_sha256(ROOT/'source-at-run'/rel)==pin,'ORIGINAL_RUN_CODE_BYTES_MISSING'
  if rel!='research/v7/window23-pipeline.py':assert file_sha256(REPO/rel)==pin,'NUMERIC_OR_PUBLISHER_CODE_CHANGED'
 pred=load('predictions');expected=load('daily_metrics');actual=metrics(pred,load('resolved_config')['evaluation']);assert daily_identity(actual)==daily_identity(expected),'DAILY_REPLAY_MISMATCH'
 # Verify every persisted prediction label/identity against saved diagnostic dataset.
 data=load('dataset_snapshot');lookup={(r['stock_id'],r['signal_session'],r['horizon']):r for r in data}
 for key,rows in pred.items():
  for r in rows:
   p=lookup[(r['stock_id'],r['signal_session'],r['horizon'])];assert p['diagnostic_label']==r['label'] and p['engineering_eligible'] and not p['strict_eligible'] and canonical_hash(p)==r['label_metadata_sha256']
 evidence={'status':'EVALUATION_REPLAYABLE_ENGINEERING_ONLY','separate_process':True,'bundle_key':run['bundle_key'],'manifest_sha256':run['manifest_sha256'],'daily_metrics_sha256':daily_identity(actual),'metric_identity':'KEYED_MODEL_SESSION_HORIZON_ORDER_NORMALIZED_NO_VALUE_CHANGE','replay_adapter_sha256':file_sha256(Path(__file__)),'original_run_code_verified':True,'metrics_equal':True,'raw_restore':'W22all78filesverifiedbeforebuild;W23bundleallbytesverified','label_identity_count':sum(len(x) for x in pred.values()),'research_admitted':False,'strict_pit':False,'training_resume':'NOT_CLAIMED'}
 put(OUT/'window23-replay.json',evidence);print(json.dumps(evidence))
if __name__=='__main__':{'build':build,'run':run,'replay':replay}[sys.argv[1]]()
