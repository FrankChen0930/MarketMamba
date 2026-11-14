"""Three fixed origins; scoped mature-only views, validation diagnostics, fresh refit.

No network/search/resume or formal admission. Forward evaluation deferred until fits complete.
"""
import sys,json,runpy,time,subprocess,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_classical_baseline import ridge_fit_state,ridge_predict
from V6.experimental.v7_research_evaluation import daily_metric
from V6.experimental.v7_research_isolation import make_refit_run,make_forward_run,RUN_SCHEMA
C=runpy.run_path(str(Path(__file__).with_name('window31-common.py')));F=runpy.run_path(str(Path(__file__).with_name('window31-folds.py')));W=runpy.run_path(str(Path(__file__).with_name('window29-study.py')));L=runpy.run_path(str(Path(__file__).with_name('window30-lgbm.py')))
ROOT=C['ROOT'];OUT=C['OUT'];H=C['H'];sha=C['sha'];put=C['put'];checked=C['checked'];MODELS=['L0_POS','L0_NEG','Ridge','LightGBM']
def contract():return json.loads((OUT/'window31-contract.json').read_text())
def source():
 cohort,raw,refs,days,cal=W['sources']();cov=json.loads((OUT/'window29-coverage.json').read_text());features=checked(cov['refs']['features']);events=checked(cov['refs']['suspected_events']);return cov,features,raw,days,cal,events

def prepare():
 ct=contract();cov,features,raw,days,calref,events=source();env=L['environment']();index={(f['signal_session'],f['stock_id']):f for f in features};assert len(index)==len(features)
 labels=[]
 for f in features:
  for h in [5,10]:
   q=F['numeric_label'](f,h,days,raw[f['stock_id']],events[f['stock_id']]);q['label_id']=H({'source_data':cov['dataset_id'],'definition':ct['label_formula'],'key':q['key']});q['label_metadata_sha256']=H(q);labels.append(q)
 # Numerical values retain original W29 identity where its old split mask was eligible.
 old=checked(cov['refs']['labels']);li={tuple(q['key']):q for q in labels}
 for q in old:
  if q['eligible']:assert li[tuple(q['key'])]['eligible'] and li[tuple(q['key'])]['label']==q['label']
 label_contract=H({'formula':ct['label_formula'],'source_label':cov['label_id'],'temporal':ct['purge'],'numeric_code':sha(OUT/'window31-folds.py')});calendar,origins=F['resolve'](ct,days,calref['sha256'],cov['feature_id'],label_contract)
 prepared_origins=[]
 for o in origins:
  oid=o['origin_id'];origin=o['origin'];val=o['validation_start'];fold=o['fold'];views={};stats={};specs={};gate=True
  for h in [5,10]:
   masks={'TRAIN':F['view_ids'](labels,h,'2021-01-01',val,val),'VALIDATION':F['view_ids'](labels,h,val,origin,origin),'REFIT':F['view_ids'](labels,h,'2021-01-01',origin,origin),'FORWARD':F['view_ids'](labels,h,origin,o['next_quarter_anchor'],'2024-01-01')};views[str(h)]=masks
   stats[str(h)]={}
   for role,ids in masks.items():
    start,end,cutoff=('2021-01-01',val,val) if role=='TRAIN' else (val,origin,origin) if role=='VALIDATION' else ('2021-01-01',origin,origin) if role=='REFIT' else (origin,o['next_quarter_anchor'],'2024-01-01')
    candidates=[q for q in labels if q['key'][2]==h and start<=q['key'][0]<end];numeric_bad=[q for q in candidates if not q['eligible']];temporal=[q for q in candidates if q['eligible'] and not F['mature'](q,cutoff)]
    stats[str(h)][role]={'candidate_features':len(candidates),'numeric_rejected':len(numeric_bad),'numeric_reasons':dict(Counter(reason for q in numeric_bad for reason in q['reasons'])),'temporal_overlap_or_immature':len(temporal),'eligible':len(ids),'latest_label_exit':max((labels[i]['exit'] for i in ids),default=None),'cutoff':cutoff+'T00:00:00+08:00','view_sha256':H([labels[i] for i in ids])}
   for phase,role,cutoff in [('VALIDATION_FIT','TRAIN',val),('REFIT','REFIT',origin)]:
    for model in ['Ridge','LightGBM']:
     rid=f'W31-{oid}-H{h}-{model}-{phase}';recipe_id='Ridge-alpha1-seed0' if model=='Ridge' else L['recipe']()['recipe_id'];selection={'schema':RUN_SCHEMA,'kind':'SELECTION','study_id':ct['study_id'],'fold_id':fold['fold_id'],'fold_sha256':H(fold),'run_id':f'W31-{oid}-H{h}-{model}-VALIDATION_FIT','recipe_id':recipe_id,'seed':0,'execution':'DECLARED_ONLY_NOT_EXECUTED'}
     declared=selection if phase=='VALIDATION_FIT' else make_refit_run(selection,rid,fold,{'status':'BOUNDED_DEVELOPMENT_PROXY_NO_FORMAL_ADMISSION'})
     specs[rid]={'run_id':rid,'origin_id':oid,'horizon':h,'family':model,'phase':phase,'input_role':role,'label_cutoff':cutoff,'fit_n':len(masks[role]),'view_sha256':stats[str(h)][role]['view_sha256'],'declared_lineage':declared,'resume_checkpoint':None,'initialization':'FRESH_STATE','recipe_id':recipe_id}
     if phase=='REFIT':specs[rid]['forward_declared_lineage']=make_forward_run(declared,rid+'-FORWARD_PREDICTIONS',fold)
   gate=gate and bool(masks['TRAIN']) and bool(masks['REFIT'])
   for role,ds,cutoff in [('VALIDATION',[d for d in days if val<=d<origin],origin),('FORWARD',[d for d in days if origin<=d<o['next_quarter_anchor']],'2024-01-01')]:
    counts=Counter(labels[i]['key'][0] for i in masks[role]);core=[d for d in ds if days.index(d)+h<len(days) and days[days.index(d)+h]<cutoff];gate=gate and all(counts[d]>=40 for d in core)
  ref=put(ROOT/oid/'views.json',views);prepared_origins.append({**o,'view_ref':ref,'stats':stats,'model_specs':specs,'eligibility_gate':'READY_DEVELOPMENT_PROXY' if gate else 'BLOCKED_COVERAGE_OR_INPUT','not_admitted':True})
 proof={'status':'PREFIT_FROZEN_CONTRACT_AND_ORIGIN_VIEWS','contract_sha256':H(ct),'code':{str(p.relative_to(C['REPO'])):sha(p) for p in [Path(__file__),OUT/'window31-folds.py',OUT/'window31-common.py',OUT/'window31-contract.json',OUT/'window30-recipe.json']},'source_dataset_id':cov['dataset_id'],'feature_id':cov['feature_id'],'label_contract_id':label_contract,'source_calendar':calref,'calendar':calendar,'origins':prepared_origins,'numeric_labels':put(ROOT/'numeric-labels.json',labels),'features':cov['refs']['features'],'environment':env,'exposure':'2023exposed; later validation/refit can reuse earlier forward mature labels without metric feedback; overlapping folds dependent','source_risks':'UNKNOWN_PUBLICATION_VINTAGE_ACTIONS_UNIVERSE; geometry only'}
 put(ROOT/'PREPARED.json',proof);put(OUT/'window31-prepared.json',proof);print(json.dumps({'origins':[{'id':o['origin_id'],'date':o['origin'],'gate':o['eligibility_gate'],'stats':o['stats']} for o in prepared_origins]}))

def load():
 p=json.loads((ROOT/'PREPARED.json').read_text());assert p==json.loads((OUT/'window31-prepared.json').read_text()) and H(contract())==p['contract_sha256'];assert L['environment']()==p['environment']
 for name,pin in p['code'].items():assert sha(C['REPO']/name)==pin
 labels=checked(p['numeric_labels']);features=checked(p['features']);index={(f['signal_session'],f['stock_id']):f for f in features};return p,labels,features,index

def authorized(p,o,labels,h,role):
 ids=checked(o['view_ref'])[str(h)][role];cutoff=o['validation_start'] if role=='TRAIN' else o['origin'];rows=[labels[i] for i in ids];assert H(rows)==o['stats'][str(h)][role]['view_sha256']
 if any(not F['mature'](q,cutoff) or q['key'][0]>=cutoff for q in rows):raise ValueError('FUTURE_OR_IMMATURE_LABEL_IN_FIT')
 return rows

def claim(oid,h,model,phase):
 folder=ROOT/'claims';folder.mkdir(exist_ok=True);existing=[json.loads(p.read_text()) for p in folder.glob('*.json')]
 if sum(r['family']==model for r in existing)>=12:raise ValueError('FAMILY_FIT_BUDGET_EXHAUSTED')
 with (folder/f'{oid}-{h}-{model}-{phase}.json').open('x') as f:json.dump({'origin':oid,'horizon':h,'family':model,'phase':phase,'maximum_family_fits':12},f)

def fit_scoped(spec,rows,index):
 x=np.asarray([index[tuple(q['key'][:2])]['features'] for q in rows]);y=np.asarray([q['label'] for q in rows]);assert np.isfinite(x).all() and np.isfinite(y).all();start=time.perf_counter()
 if spec['family']=='Ridge':
  state=ridge_fit_state(x,y,1.0);np.testing.assert_array_equal(state['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(state['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
 else:state=L['fit_arrays'](x,y,L['recipe']())
 return {'run_id':spec['run_id'],'family':spec['family'],'phase':spec['phase'],'horizon':spec['horizon'],'initialization':'FRESH_STATE','resume_checkpoint':None,'view_sha256':H(rows),'n':len(rows),'train_x_sha256':H(x.tolist()),'train_y_sha256':H(y.tolist()),'state':state,'fit_wall_seconds':time.perf_counter()-start}

def prediction_packets(o,phase,models,features,p):
 a,b=o['fold']['validation'] if phase=='VALIDATION' else o['fold']['forward'];fs=[f for f in features if a<=f['signal_session']<=b];x=np.asarray([f['features'] for f in fs]);packets={}
 for h in [5,10]:
  fitphase='VALIDATION_FIT' if phase=='VALIDATION' else 'REFIT';scores={'L0_POS':x[:,0],'L0_NEG':-x[:,0]}
  for family in ['Ridge','LightGBM']:
   m=models[f'W31-{o["origin_id"]}-H{h}-{family}-{fitphase}'];scores[family]=ridge_predict(m['state'],x) if family=='Ridge' else lgb.Booster(model_str=m['state']['booster_text']).predict(x,num_iteration=m['state']['actual_iterations'],num_threads=1)
  for family,vs in scores.items():
   assert np.isfinite(vs).all();modelid=f'W31-{o["origin_id"]}-H{h}-{family}-{fitphase}';packet={'header':{'origin':o['origin_id'],'phase':phase,'family':family,'horizon':h,'model_id':modelid,'model_sha256':H(models[modelid]) if modelid in models else H({'L0':family,'direction':1 if family=='L0_POS' else -1}),'dataset_id':p['source_dataset_id'],'feature_id':p['feature_id'],'label_contract_id':p['label_contract_id'],'fold_sha256':H(o['fold']),'recipe_id':o['model_specs'][modelid]['recipe_id'] if modelid in o['model_specs'] else 'fixed-momentum-'+family,'exposure':'DEVELOPMENT_NOT_ADMITTED'},'rows':[{'stock_id':f['stock_id'],'signal_session':f['signal_session'],'horizon':h,'score':float(v)} for f,v in zip(fs,vs)]};packets[f'{h}|{family}']=packet
 return packets

def metrics(o,phase,packets,labels):
 views=checked(o['view_ref']);li={tuple(q['key']):q for q in labels};daily=[];pairs=[];summaries=[];rejections=[]
 for h in [5,10]:
  allowed={tuple(labels[i]['key']) for i in views[str(h)][phase]};by={m:defaultdict(list) for m in MODELS};requested={}
  for family in MODELS:
   rows=packets[f'{h}|{family}']['rows'];keys={(r['signal_session'],r['stock_id'],h) for r in rows};assert len(keys)==len(rows);requested[family]=keys
   for r in rows:
    k=(r['signal_session'],r['stock_id'],h);q=li[k]
    if k in allowed:by[family][r['signal_session']].append({**r,'label':q['label'],'label_id':q['label_id'],'label_metadata_sha256':q['label_metadata_sha256']})
    elif family=='L0_POS':rejections.append({'key':list(k),'reasons':q['reasons'] or ['ORIGIN_LABEL_IMMATURE_OR_OVERLAP']})
  assert all(requested[m]==requested['L0_POS'] for m in MODELS);days=sorted({k[0] for k in requested['L0_POS']});groups={m:[] for m in MODELS};deltas=defaultdict(list)
  for d in days:
   result={}
   for family in MODELS:
    rows=by[family][d];q=daily_metric(rows,d,h,contract()['evaluation']);q.update(origin=o['origin_id'],phase=phase,family=family,score_ties_n=len(rows)-len({r['score'] for r in rows}));result[family]=q;groups[family].append(q);daily.append(q)
   first=result['L0_POS'];assert all(q['stock_ids']==first['stock_ids'] and q['support_sha256']==first['support_sha256'] for q in result.values())
   for a,b in itertools.combinations(MODELS,2):
    x,y=result[a]['rank_ic'],result[b]['rank_ic'];delta=y-x if x is not None and y is not None else None;r={'origin':o['origin_id'],'phase':phase,'horizon':h,'signal_session':d,'a':a,'b':b,'a_RankIC':x,'b_RankIC':y,'delta':delta,'valid_n':first['valid_n'],'support_sha256':first['support_sha256']};pairs.append(r)
    if delta is not None:deltas[a+'->'+b].append(delta)
  for family,ds in groups.items():
   v=[q['rank_ic'] for q in ds if q['rank_ic'] is not None];summaries.append({'origin':o['origin_id'],'phase':phase,'horizon':h,'family':family,'requested_days':len(ds),'defined_days':len(v),'mean':float(np.mean(v)) if v else None,'median':float(np.median(v)) if v else None,'daily_N':dict(Counter(q['valid_n'] for q in ds)),'undefined_reasons':dict(Counter(r for q in ds for r in q['reasons'])),'mean_paired_deltas':{k:float(np.mean(v)) for k,v in deltas.items()},'dependency':'OVERLAPPING_EXPOSED_DEVELOPMENT_NOT_INDEPENDENT_TRIALS'})
 return {'daily':daily,'paired':pairs,'summary':summaries,'rejections':rejections}

def run_origin(oid):
 p,labels,features,index=load();o=next(o for o in p['origins'] if o['origin_id']==oid)
 if o['eligibility_gate']!='READY_DEVELOPMENT_PROXY':put(ROOT/oid/'BLOCKED.json',{'status':'BLOCKED_NO_FITS','reason':'ORIGIN_ELIGIBILITY_FAILED'});return
 for name,pin in p['code'].items():assert subprocess.check_output(['git','show','HEAD:'+name],cwd=C['REPO'])==(C['REPO']/name).read_bytes()
 assert subprocess.check_output(['git','show','HEAD:research/v7/window31-prepared.json'],cwd=C['REPO'])==(OUT/'window31-prepared.json').read_bytes(),'PREFIT_COMMIT_REQUIRED'
 models={}
 for phase,role in [('VALIDATION_FIT','TRAIN'),('REFIT','REFIT')]:
  for h in [5,10]:
   rows=authorized(p,o,labels,h,role)
   for family in ['Ridge','LightGBM']:
    rid=f'W31-{oid}-H{h}-{family}-{phase}';spec=o['model_specs'][rid];claim(oid,h,family,phase);models[rid]=fit_scoped(spec,rows,index);put(ROOT/oid/(rid+'.json'),models[rid])
  prediction_phase='VALIDATION' if phase=='VALIDATION_FIT' else 'FORWARD';pack=prediction_packets(o,prediction_phase,models,features,p);put(ROOT/oid/(prediction_phase+'-predictions.json'),pack)
  if prediction_phase=='VALIDATION':put(ROOT/oid/'VALIDATION-diagnostics.json',metrics(o,'VALIDATION',pack,labels)) # Diagnostic not passed to fit_scoped; recipe cannot change.
 result={'status':'ORIGIN_FITS_AND_PREDICTIONS_COMPLETE','origin':oid,'models':put(ROOT/oid/'models.json',models),'fits':{'Ridge':4,'LightGBM':4},'forward_evaluated_before_later_fits':False};put(ROOT/oid/'COMPLETE.json',result);print(json.dumps({'origin':oid,'fits':result['fits'],'model_ids':list(models)}))

def evaluate_all(replay=False):
 p,labels,features,index=load();allstats=[];origins=[];prediction_n=0;daily_n=paired_n=0
 if replay:
  closed=json.loads((ROOT/'COMPLETE.json').read_text());assert closed['prepared_sha256']==sha(ROOT/'PREPARED.json')
 for o in p['origins']:
  folder=ROOT/o['origin_id']
  if not (folder/'COMPLETE.json').exists():origins.append({'origin':o['origin_id'],'status':'BLOCKED_OR_INCOMPLETE'});continue
  completion=json.loads((folder/'COMPLETE.json').read_text());models=checked(completion['models']);stats=[]
  for phase in ['VALIDATION','FORWARD']:
   pack=json.loads((folder/(phase+'-predictions.json')).read_text());pred_hash=H(pack);prediction_n+=sum(len(x['rows']) for x in pack.values())
   if replay:
    for rid,m in models.items():
     role='TRAIN' if m['phase']=='VALIDATION_FIT' else 'REFIT';rows=authorized(p,o,labels,m['horizon'],role);x=np.asarray([index[tuple(q['key'][:2])]['features'] for q in rows]);assert H(rows)==m['view_sha256'] and H(x.tolist())==m['train_x_sha256']
     if m['family']=='Ridge':np.testing.assert_array_equal(m['state']['scaling']['mean'],x.mean(axis=0));np.testing.assert_array_equal(m['state']['scaling']['std'],np.where(x.std(axis=0)==0,1.,x.std(axis=0)))
    assert H(prediction_packets(o,phase,models,features,p))==pred_hash,'PREDICTION_REPLAY_MISMATCH'
   if phase=='VALIDATION' and not replay:r=json.loads((folder/'VALIDATION-diagnostics.json').read_text())
   else:r=metrics(o,phase,pack,labels)
   assert H(pack)==pred_hash,'LABEL_JOIN_MUTATED_PREDICTIONS'
   target=folder/(phase+'-diagnostics.json')
   if replay:assert H(r)==H(json.loads(target.read_text())),'METRIC_REPLAY_MISMATCH'
   else:put(target,r)
   stats.extend(r['summary']);daily_n+=len(r['daily']);paired_n+=len(r['paired'])
  allstats.extend(stats);origins.append({'origin':o['origin_id'],'status':'COMPLETE','fits':completion['fits']})
 if replay:
  assert H(allstats)==H(json.loads((OUT/'window31-results.json').read_text()));proof={'status':'INDEPENDENT_PROCESS_REPLAY_PASS','fits':0,'origins':origins,'prediction_rows':prediction_n,'daily':daily_n,'paired':paired_n,'no_Ridge_or_LGBM_refit':True};put(OUT/'window31-replay.json',proof);print(json.dumps(proof));return
 record={'status':'DEVELOPMENT_COMPLETE' if all(o['status']=='COMPLETE' for o in origins) else 'PARTIAL_BLOCKED_DEVELOPMENT','prepared_sha256':sha(ROOT/'PREPARED.json'),'origins':origins,'Ridge_fits':sum(o.get('fits',{}).get('Ridge',0) for o in origins),'LightGBM_fits':sum(o.get('fits',{}).get('LightGBM',0) for o in origins),'predictions':prediction_n,'daily':daily_n,'paired':paired_n,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=C['REPO'],text=True).strip(),'deferred_forward_evaluation':True,'admission':False};put(ROOT/'COMPLETE.json',record);put(OUT/'window31-run.json',record);put(OUT/'window31-results.json',allstats);print(json.dumps({'run':record,'forward_summary':[s for s in allstats if s['phase']=='FORWARD']}))
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)==2 else ''
 if mode not in ['prepare','O1','O2','O3','evaluate','replay']:raise SystemExit('prepare/O1/O2/O3/evaluate/replay')
 with C['budget'](mode):
  if mode=='prepare':prepare()
  elif mode in ['O1','O2','O3']:run_origin(mode)
  else:evaluate_all(mode=='replay')
