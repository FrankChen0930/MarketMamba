"""Fixed W29 cohort study, feature-only predictions, post-hoc label join, no search."""
import sys,json,runpy,math,statistics,subprocess
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_classical_baseline import ridge_fit_state,ridge_predict
from V6.experimental.v7_research_evaluation import daily_metric
from V6.experimental.v7_source_snapshot import verify_snapshot
C=runpy.run_path(str(Path(__file__).with_name('window29-common.py')));P=runpy.run_path(str(Path(__file__).with_name('window29-panel.py')))
ROOT=C['ROOT'];OUT=C['OUT'];put=C['put'];checked=C['checked'];sha=C['file_sha256'];H=C['canonical_hash']
MODELS=['L0_POS','L0_NEG','Ridge']
def sources():
 cohort=json.loads((OUT/'window29-cohort.json').read_text());assert cohort==json.loads((ROOT/'FROZEN_COHORT.json').read_text());pin=json.loads((ROOT/'COHORT_PIN.json').read_text());assert H(cohort)==pin['semantic_sha256'] and sha(OUT/'window29-cohort.json')==pin['file_sha256']
 records={};refs=[]
 for p in sorted((ROOT/'finmind/records').glob('*.json')):
  r=json.loads(p.read_text());assert sha(Path(r['raw_path']))==r['raw_sha256'] and verify_snapshot(Path(r['manifest']))
  if r['status']=='CAPTURED':records[r['params']['data_id']]=r
 raw={}
 for c in cohort['cohort']:
  sec=c['security_id'];r=records.get(sec)
  if sec in cohort['cache']:
   ref=cohort['cache'][sec];assert sha(Path(ref['record']))==ref['record_sha256'] and sha(Path(ref['raw']))==ref['raw_sha256'];body=json.loads(Path(ref['raw']).read_text());refs.append(ref)
  elif r:
   body=json.loads(Path(r['raw_path']).read_text());refs.append({'security':sec,'raw':r['raw_path'],'sha256':r['raw_sha256'],'retrieved_at':r['retrieved_at'],'record_sha256':H(r),'license':r['license_identity']})
  else:raw[sec]={};continue
  rows=[x for x in body['data'] if '2021-01-01'<=x['date']<='2023-12-31'];raw[sec]={x['date']:x for x in rows};assert len(rows)==len(raw[sec])
 calendar_ref=json.loads((OUT/'window24-coverage.json').read_text())['references']['calendar'];cal=checked(calendar_ref);days=[r['date'] for r in cal if r['engineering_calendar_ok'] and r['price_observed']];assert len(days)==729 and len(set(days))==len(days) and days==sorted(days)
 return cohort,raw,refs,days,calendar_ref

def build():
 cohort,raw,refs,days,calref=sources();plan=json.loads((OUT/'window29-plan.json').read_text());net=json.loads((ROOT/'finmind/CLOSED.json').read_text());features=[];feature_rejections=[];labels=[];counts={};stock_coverage=[];events={};extras=[]
 for c in cohort['cohort']:
  sec=c['security_id'];prices=raw[sec];event=P['suspect_events'](prices);events[sec]=event;extra=sorted(set(prices)-set(days));extras.extend((sec,d) for d in extra)
  stock_coverage.append({'stock_id':sec,'raw_rows':len(prices),'first':min(prices) if prices else None,'last':max(prices) if prices else None,'raw_missing_market_sessions':len(set(days)-set(prices)),'zero_price_rows':sum(any(p.get(k)==0 for k in ('open','max','min','close')) for p in prices.values()),'extra_sessions':extra,'suspected_unit_dates':event,'market_or_cause':'UNKNOWN_NOT_INFERRED_FROM_PRICE_ABSENCE','membership_role':c['role']})
  for day in days:
   f=P['feature_at'](sec,day,days,prices,c,event)
   if not f['eligible']:
    feature_rejections.append({'key':[day,sec],'split':f['split'],'reasons':f['reasons']});continue
   features.append(f)
   for h in [5,10]:
    q=P['label_at'](f,h,days,prices,event);labels.append(q);g=counts.setdefault(f['split']+'_'+str(h),{'feature_candidates':0,'eligible':0,'excluded':0,'primary':Counter(),'all_flags':Counter(),'UNKNOWN_risk':0});g['feature_candidates']+=1;g['eligible']+=q['eligible'];g['excluded']+=not q['eligible'];g['UNKNOWN_risk']+=1
    if not q['eligible']:g['primary'][q.get('primary') or q['reasons'][0]]+=1;g['all_flags'].update(q['reasons'])
 for g in counts.values():assert sum(g['primary'].values())==g['excluded']
 n=Counter((q['key'][0],q['key'][2]) for q in labels if q['eligible']);core=[]
 for i,day in enumerate(days):
  if day<'2022-01-01':continue
  bound='2023-01-01' if day<'2023-01-01' else '2024-01-01'
  for h in [5,10]:
   if i+h<len(days) and days[i+h]<bound:core.append({'signal_session':day,'horizon':h,'N':n[(day,h)]})
 data_id=H({'cohort':H(cohort),'sources':refs,'calendar':calref,'policy':H(plan),'feature_label_code':sha(OUT/'window29-panel.py')});feature_id=H({'features':plan['features'],'formulas':plan['feature_formulas'],'interface':'FEATURE_ONLY_NO_FUTURE_LABEL_ACCESS','code':sha(OUT/'window29-panel.py')});label_id=H({'formula':'rawclose[t+h]/rawopen[t+1]-1','safety':'W27plusW29suspectedunitprecommitted','code':sha(OUT/'window29-panel.py'),'data':data_id})
 for q in labels:
  q['label_id']=H({'contract':label_id,'key':q['key']});q['label_metadata_sha256']=H(q)
 refs_out={name:put(ROOT/'derived'/(name+'.json'),value) for name,value in {'features':features,'labels':labels,'feature_rejections':feature_rejections,'sources':refs,'stock_coverage':stock_coverage,'suspected_events':events,'core_support':core}.items()}
 gate=P['coverage_gate']([q['N'] for q in core]) and not extras and not net['source_stopped'] and net['captured']==len(cohort['price_queries'])
 cover={'status':'TECHNICAL_DEVELOPMENT_PANEL_NOT_ADMITTED','dataset_id':data_id,'feature_id':feature_id,'label_id':label_id,'cohort_sha256':H(cohort),'plan_sha256':H(plan),'calendar':calref,'refs':refs_out,'raw_rows':sum(len(x) for x in raw.values()),'feature_rows':len(features),'feature_rejected_rows':len(feature_rejections),'label_counts':counts,'eligible_label_total':sum(q['eligible'] for q in labels),'suspected_event_dates':events,'core_group_count':len(core),'core_N_distribution':dict(Counter(q['N'] for q in core)),'core_groups_below40':sum(q['N']<40 for q in core),'fit_gate':bool(gate),'prediction_rule':'all eligible current/past feature rows in2022/2023, including label tail; feature-only inputs do not read label payload','predicted_keys_expected_sha256':H(sorted((f['signal_session'],f['stock_id'],h) for f in features if f['split']!='TRAIN' for h in [5,10])),'extra_sessions':extras,'units':'FinMind native raw OHLC/Trading_Volume share counts; no scaling/adjusted-price or lot conversion','asof_vintage_action_membership_risks':'UNKNOWN_NOT_CLEAN; current official survivor selection not historical universe','source_restrictions':'Private noncommercial/attribution; no raw redistribution','research_admitted':False}
 put(ROOT/'PREFIT.json',cover);put(OUT/'window29-coverage.json',cover);print(json.dumps({k:cover[k] for k in ['raw_rows','feature_rows','feature_rejected_rows','core_N_distribution','core_groups_below40','fit_gate']}))

def inputs():
 cover=json.loads((ROOT/'PREFIT.json').read_text());assert cover==json.loads((OUT/'window29-coverage.json').read_text());features=checked(cover['refs']['features']);labels=checked(cover['refs']['labels']);plan=json.loads((OUT/'window29-plan.json').read_text());assert H(plan)==cover['plan_sha256'];sources() # Reverify real raw bytes and source identity.
 return cover,features,labels,plan

def train(features,labels,plan):
 index={(f['signal_session'],f['stock_id']):f for f in features};models={}
 for h in [5,10]:
  rows=[q for q in labels if q['eligible'] and q['key'][2]==h and q['key'][0]<'2022-01-01'];assert rows and all(q['exit']<'2022-01-01' for q in rows)
  with (ROOT/f'RIDGE_{h}_FIT_CLAIM.json').open('x') as f:json.dump({'horizon':h,'fit_limit':1,'fresh':True},f)
  x=np.asarray([index[tuple(q['key'][:2])]['features'] for q in rows]);y=np.asarray([q['label'] for q in rows]);state=ridge_fit_state(x,y,1.0)
  np.testing.assert_array_equal(state['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(state['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
  models[str(h)]={'run_id':f'w29-fresh-ridge-{h}-train2021','state':state,'train_n':len(rows),'train_label_keys_sha256':H([q['key'] for q in rows]),'train_feature_sha256':H(x.tolist()),'alpha':1.0,'resume':False,'scaling':'TRAIN_ONLY_ZSCORE','selection':'NONE','latest_label_exit':max(q['exit'] for q in rows)}
  put(ROOT/f'model-{h}.json',models[str(h)])
 return models

def predict(features,models,expected_keys):
 # This interface receives NO label table, outcome mask, entry/exit or maturity.
 pred={};evaluation=[f for f in features if f['split']!='TRAIN'];keys=sorted((f['signal_session'],f['stock_id'],h) for f in evaluation for h in [5,10]);assert H(keys)==expected_keys
 for h in [5,10]:
  for role in ['VALIDATION','EXPOSED_DIAGNOSTIC']:
   fs=[f for f in evaluation if f['split']==role];x=np.asarray([f['features'] for f in fs]);scores={'L0_POS':x[:,0],'L0_NEG':-x[:,0],'Ridge':ridge_predict(models[str(h)]['state'],x)}
   for model,values in scores.items():
    assert np.isfinite(values).all();pred[f'{model}|{h}|{role}']=[{'stock_id':f['stock_id'],'signal_session':f['signal_session'],'horizon':h,'score':float(v)} for f,v in zip(fs,values)]
 return pred

def evaluate(pred,labels,days,policy,label_contract):
 li={tuple(q['key']):q for q in labels};assert len(li)==len(labels);daily=[];paired=[];summary=[];support=[];join_rejections=[]
 for year,h in [(2022,5),(2022,10),(2023,5),(2023,10)]:
  role='VALIDATION' if year==2022 else 'EXPOSED_DIAGNOSTIC';by={m:defaultdict(list) for m in MODELS}
  for model in MODELS:
   seen=set()
   for r in pred[f'{model}|{h}|{role}']:
    k=(r['signal_session'],r['stock_id'],r['horizon']);assert k not in seen;seen.add(k);q=li[k]
    if not q['eligible']:
     if model=='L0_POS':join_rejections.append({'key':list(k),'reasons':q['reasons']})
     continue
    by[model][r['signal_session']].append({**r,'label':q['label'],'label_id':q['label_id'],'label_metadata_sha256':q['label_metadata_sha256']})
  groups={m:[] for m in MODELS};pair_groups=defaultdict(list)
  for day in [d for d in days if int(d[:4])==year]:
   metrics={}
   for model in MODELS:
    rows=by[model][day];q=daily_metric(rows,day,h,policy);q.update(model=model,year=year,score_ties_n=len(rows)-len({r['score'] for r in rows}),label_ties_n=len(rows)-len({r['label'] for r in rows}));metrics[model]=q;daily.append(q);groups[model].append(q)
   first=metrics['L0_POS'];assert all(q['support_sha256']==first['support_sha256'] and q['stock_ids']==first['stock_ids'] for q in metrics.values())
   support.append({'signal_session':day,'horizon':h,'stock_ids':first['stock_ids'],'valid_n':first['valid_n'],'support_sha256':first['support_sha256']})
   for a,b in [('L0_POS','Ridge'),('L0_NEG','Ridge'),('L0_POS','L0_NEG')]:
    x,y=metrics[a]['rank_ic'],metrics[b]['rank_ic'];delta=y-x if x is not None and y is not None else None;r={'year':year,'horizon':h,'signal_session':day,'a':a,'b':b,'a_RankIC':x,'b_RankIC':y,'delta':delta,'valid_n':first['valid_n'],'support_sha256':first['support_sha256']};paired.append(r);pair_groups[(a,b)].append(delta)
  for model,ds in groups.items():
   vals=[q['rank_ic'] for q in ds if q['rank_ic'] is not None];summary.append({'year':year,'horizon':h,'model':model,'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None,'requested_days':len(ds),'defined_days':len(vals),'daily_N':dict(Counter(q['valid_n'] for q in ds)),'skipped_reasons':dict(Counter(reason for q in ds for reason in q['reasons'])),'paired_differences':{a+'->'+b:float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else None for (a,b),vs in pair_groups.items()},'statistical_inference':'NOT_CONFIGURED_NO_WINNER_CLAIM','label_contract':label_contract})
 return daily,paired,summary,support,join_rejections

def main(mode):
 with C['cpu_stage'](mode):
  if mode=='build':build();return
  cover,features,labels,plan=inputs();assert cover['fit_gate'],'INSUFFICIENT_COVERAGE_NO_FIT_MINIMUM_NOT_RELAXED'
  code={str(p.relative_to(C['REPO'])):sha(p) for p in [Path(__file__),OUT/'window29-panel.py',OUT/'window29-common.py',OUT/'window29-plan.json',OUT/'window29-cohort.json',C['REPO']/'V6/experimental/v7_classical_baseline.py',C['REPO']/'V6/experimental/v7_research_evaluation.py']}
  if mode=='run':
   if (ROOT/'COMPLETE.json').exists():raise ValueError('ALREADY_COMPLETE_NO_REFIT')
   models=train(features,labels,plan)
  else:
   completed=json.loads((ROOT/'COMPLETE.json').read_text());assert completed['code']==code
   for ref in completed['artifacts'].values():checked(ref)
   models=checked(completed['artifacts']['models'])
   # Replay validates train-only state algebra without performing another fit/solve.
   ix={(f['signal_session'],f['stock_id']):f for f in features}
   for h in [5,10]:
    qs=[q for q in labels if q['eligible'] and q['key'][2]==h and q['key'][0]<'2022-01-01'];x=np.asarray([ix[tuple(q['key'][:2])]['features'] for q in qs]);m=models[str(h)];assert m['train_label_keys_sha256']==H([q['key'] for q in qs]);np.testing.assert_array_equal(m['state']['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(m['state']['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
  pred=predict(features,models,cover['predicted_keys_expected_sha256']);pure_hash=H(pred)
  if mode=='run':prediction_ref=put(ROOT/'predictions.json',pred) # Immutable publication BEFORE evaluation label join.
  _,_,_,days,_=sources();daily,paired,summary,support,rejected=evaluate(pred,labels,days,plan['evaluation'],cover['label_id']);assert H(pred)==pure_hash,'EVALUATION_MUTATED_PREDICTIONS'
  values={'models':models,'predictions':pred,'daily':daily,'paired':paired,'summary':summary,'common_support':support,'evaluation_rejections':rejected}
  if mode=='replay':
   for name,value in values.items():assert H(value)==H(checked(completed['artifacts'][name])),'REPLAY_MISMATCH:'+name
   proof={'status':'INDEPENDENT_PROCESS_REPLAY_PASS','fits':0,'model_groups':len(pred),'predictions':sum(len(v) for v in pred.values()),'daily':len(daily),'paired':len(paired),'pure_prediction_hash':pure_hash,'labels_do_not_change_prediction_keys':True,'research_admitted':False};put(OUT/'window29-replay.json',proof);print(json.dumps(proof));return
  refs={name:put(ROOT/(name+'.json'),value) for name,value in values.items()};completed={'status':'COMPLETE_DEVELOPMENT_ONLY_NOT_ADMITTED','study_id':plan['study_id'],'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=C['REPO'],text=True).strip(),'code':code,'dataset_id':cover['dataset_id'],'feature_id':cover['feature_id'],'label_id':cover['label_id'],'cohort_sha256':cover['cohort_sha256'],'fold_id':'train2021-development2022-exposed2023-horizon-maturity','artifacts':refs,'Ridge_fits':2,'LightGBM_fits':0,'LightGBM_status':plan['LightGBM'],'pure_prediction_hash':pure_hash,'predictions_published_before_evaluation_join':True,'prediction_count':sum(len(v) for v in pred.values()),'model_ids':{h:H(m) for h,m in models.items()},'model_training_n':{h:m['train_n'] for h,m in models.items()},'publication_asof_action_risk':'UNKNOWN_NOT_CLEAN','backup':'BACKUP_PENDING_SAME_HOST'}
  put(ROOT/'COMPLETE.json',completed);put(OUT/'window29-run.json',completed);put(OUT/'window29-results.json',summary);print(json.dumps({'train_n':completed['model_training_n'],'predictions':completed['prediction_count'],'summary':summary}))
if __name__=='__main__':
 if len(sys.argv)!=2 or sys.argv[1] not in ('build','run','replay'):raise SystemExit('choose build/run/replay')
 main(sys.argv[1])
