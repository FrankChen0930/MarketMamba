"""Frozen-prediction W27 sensitivity and independent-process replay; never fits/scores models."""
import sys,json,runpy,subprocess,hashlib
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_research_artifacts import verify_bundle
from V6.experimental.v7_source_snapshot import _atomic_bytes
from V6.experimental.v7_research_artifacts import _safe
# W25 shared paired-support routine. None removes NO security: not a LOSO analysis.
from V6.experimental.v7_robustness_audit import leave_one as paired_daily
M=runpy.run_path(str(Path(__file__).with_name('window27-safety.py')))
ROOT=M['ROOT'];OUT=M['OUT'];REPO=M['REPO'];checked=M['checked'];input_pins=M['input_pins']
FREEZE='e77f32d'  # full identity is resolved and recorded before the first metric call

def put(path,value):
 b=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=b:raise ValueError('IMMUTABLE_DIAGNOSTIC_CONFLICT')
 else:_atomic_bytes(_safe(path),b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def load_inputs():
 frozen=json.loads((ROOT/'FROZEN_MASK.json').read_text());assert frozen==json.loads((OUT/'window27-mask-manifest.json').read_text())
 policy=json.loads((OUT/'window27-policy.json').read_text())
 assert canonical_hash(policy)==frozen['policy_sha256'] and file_sha256(OUT/'window27-policy.json')==frozen['policy_file_sha256']
 assert file_sha256(OUT/'window27-safety.py')==frozen['predicate_sha256']
 for p,pin in frozen['source_refs'].items():assert file_sha256(REPO/p)==pin
 freeze=subprocess.check_output(['git','rev-parse',FREEZE],cwd=REPO,text=True).strip()
 for name in ('window27-policy.json','window27-mask-manifest.json','window27-safety.py'):
  old=subprocess.check_output(['git','show',freeze+':research/v7/'+name],cwd=REPO);assert old==(OUT/name).read_bytes()
 run=json.loads((M['W24']/'PILOT_CLOSED.json').read_text());v=verify_bundle(M['W24']/'bundles',run['bundle_key'],run['manifest_sha256']);assert v['status']=='ARTIFACTS_VERIFIED'
 bundle=Path(v['bundle_path']);get=lambda n:json.loads((bundle/(n+'.json')).read_text())['value']
 for path,pin in run['code_files'].items():assert file_sha256(REPO/path)==pin
 data=get('dataset_snapshot');pred=get('predictions');old=get('daily_metrics');mask=checked(frozen['mask'])
 index={(r['signal_session'],r['stock_id'],r['horizon']):r for r in data};mi={tuple(r['key']):r for r in mask};assert len(mi)==len(index)==len(data)
 for k,m in mi.items():assert canonical_hash(index[k])==m['source_sample_sha256']
 filtered={};diff=[]
 for name,rows in pred.items():
  h=int(name.split('-')[1]);role=name.split('-',2)[2];keys=[];kept=[]
  expected={k for k,r in index.items() if r['split']==role and r['horizon']==h}
  for r in rows:
   k=(r['signal_session'],r['stock_id'],r['horizon']);keys.append(k);sample=index[k]
   assert isinstance(r['stock_id'],str) and r['horizon']==h and sample['split']==role
   assert canonical_hash(sample)==r['label_metadata_sha256'] and sample['diagnostic_label']==r['label']
   assert r['label_id']==canonical_hash({'stock':r['stock_id'],'session':r['signal_session'],'h':h,'contract':canonical_hash(get('resolved_config')['contract'])})
   if mi[k]['eligible']:kept.append(r)
   elif name.startswith('L0-'):diff.append({'key':list(k),'role':role,'primary':mi[k]['primary'],'all_reasons':mi[k]['reasons'],'old_label_metadata_sha256':r['label_metadata_sha256']})
  assert len(keys)==len(set(keys)) and set(keys)==expected
  filtered[name]=kept
 return frozen,policy,freeze,run,pred,old,filtered,diff

def calculate(filtered,old,policy):
 by_model_day={name:defaultdict(list) for name in filtered}
 for name,rows in filtered.items():
  for r in rows:by_model_day[name][r['signal_session']].append(r)
 original={(d['model'],d['signal_session']):d for d in old};daily=[];paired=[];groups=[]
 for year,h in policy['allowed_groups']:
  role='VALIDATION' if year==2022 else 'FORWARD_DIAGNOSTIC';a=f'L0-{h}-{role}';b=f'Ridge-{h}-{role}'
  days=sorted(day for model,day in original if model==a);assert days==sorted(day for model,day in original if model==b)
  details=[]
  for day in days:
   l0=by_model_day[a][day];ridge=by_model_day[b][day]
   # Identical support is checked in the existing W25 helper; no security is removed here.
   q=paired_daily(l0,ridge,None,day,h,policy['evaluation'])
   for name,label,rows in ((a,'L0',l0),(b,'Ridge',ridge)):
    r=q[label];r={**r,'model':name,'score_ties_n':len(rows)-len({x['score'] for x in rows}),'label_ties_n':len(rows)-len({x['label'] for x in rows})};daily.append(r)
   olda=original[(a,day)];oldb=original[(b,day)];assert olda['support_sha256']==oldb['support_sha256']
   assert set(q['L0']['stock_ids'])<=set(olda['stock_ids'])
   p={'year':year,'horizon':h,'signal_session':day,'old_n':olda['valid_n'],'new_n':q['L0']['valid_n'],'excluded_securities':sorted(set(olda['stock_ids'])-set(q['L0']['stock_ids'])),'support_sha256':q['L0']['support_sha256'],'L0':q['L0']['rank_ic'],'Ridge':q['Ridge']['rank_ic'],'delta':q['delta'],'reasons':{'L0':q['L0']['reasons'],'Ridge':q['Ridge']['reasons']},'old_L0':olda['rank_ic'],'old_Ridge':oldb['rank_ic']};paired.append(p);details.append(p)
  def mean(values):
   values=[v for v in values if v is not None];return sum(values)/len(values) if values else None
  joint=[p for p in details if all(p[k] is not None for k in ('L0','Ridge','old_L0','old_Ridge'))]
  groups.append({'year':year,'horizon':h,'requested_days':len(details),'defined_paired_days':len(joint),'skipped_days':len(details)-len(joint),'old_daily_n':dict(Counter(p['old_n'] for p in details)),'new_daily_n':dict(Counter(p['new_n'] for p in details)),'removed_sample_keys':sum(p['old_n']-p['new_n'] for p in details),'changed_days':sum(p['old_n']!=p['new_n'] for p in details),'below20_days':sum(p['new_n']<20 for p in details),'means':{model:{'old_all_defined':mean(p['old_'+model] for p in details),'new_all_defined':mean(p[model] for p in details),'old_joint_defined_days':mean(p['old_'+model] for p in joint),'new_joint_defined_days':mean(p[model] for p in joint)} for model in ('L0','Ridge')},'mean_paired_delta':mean(p['delta'] for p in joint),'interpretation':'POST_HOC_DEVELOPMENT_SENSITIVITY_ONLY'})
 return daily,paired,groups

def main(replay=False):
 before=input_pins();frozen,policy,freeze,oldrun,pred,old,filtered,diff=load_inputs()
 result=calculate(filtered,old,policy)
 if replay:
  closed=json.loads((ROOT/'COMPLETE.json').read_text())
  for ref in closed['artifacts'].values():checked(ref)
  assert canonical_hash(result[0])==canonical_hash(checked(closed['artifacts']['daily'])) and canonical_hash(result[1])==canonical_hash(checked(closed['artifacts']['paired'])) and canonical_hash(result[2])==canonical_hash(checked(closed['artifacts']['summary']))
  assert filtered==checked(closed['artifacts']['frozen_prediction_subset']) and diff==checked(closed['artifacts']['support_difference'])
  # Regenerate the mask independently from raw sessions, without altering the frozen mask.
  _,_,data,days,raw=M['source_views']();mi={tuple(r['key']):r for r in checked(frozen['mask'])}
  for r in data:
   k=(r['signal_session'],r['stock_id'],r['horizon']);q=M['classify'](r,days,raw[r['stock_id']],['2023-02-17'] if r['stock_id']=='2321' else [])
   assert all(mi[k][field]==q[field] for field in ('eligible','primary','reasons','risk_status'))
  proof={'status':'INDEPENDENT_PROCESS_REPLAY_PASS','daily_rows':len(result[0]),'paired_rows':len(result[1]),'original_predictions_unchanged':True,'mask_raw_rederived':len(data),'original_artifacts_verified':before,'new_model_predictions_generated':False,'research_admitted':False}
  put(OUT/'window27-replay.json',proof);print(json.dumps(proof));return
 refs={name:put(ROOT/(name+'.json'),value) for name,value in zip(('daily','paired','summary','frozen_prediction_subset','support_difference'),(*result,filtered,diff))}
 complete={'schema':'w27-frozen-prediction-diagnostic-v1','status':'COMPLETE_NOT_RESEARCH_ADMITTED','diagnostic_id':'W27_POST_HOC_FROZEN_PREDICTION_LABEL_SENSITIVITY','freeze_commit':freeze,'policy_sha256':frozen['policy_sha256'],'mask':frozen['mask'],'W24_prediction_bundle':oldrun['bundle_key'],'W24_manifest_sha256':oldrun['manifest_sha256'],'code_sha256':file_sha256(Path(__file__)),'artifacts':refs,'original_byte_pins':input_pins(),'fit_newprediction_API_GPU':0,'backup':'BACKUP_PENDING_SAME_HOST','publisher_scope':'W09 used to verify original bundle; W14 atomic byte writer reused for W27 immutable diagnostic files because W09 does not admit a W27 no-training profile; no V6 policy changed'}
 put(ROOT/'COMPLETE.json',complete);put(OUT/'window27-diagnostic-manifest.json',complete);put(OUT/'window27-results.json',result[2]);print(json.dumps(result[2]))
if __name__=='__main__':main(sys.argv[1:] == ['replay'])
