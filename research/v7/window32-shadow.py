"""Synthetic-only prediction publication/late-label join preparation.

No model inference, training, network or scheduler. Supplied fixture clocks are
not runtime observations. Actual archive creation is recorded separately.
Reuse existing immutable publication/locks/strict JSON rather than a new store.
"""
import json,math,re,sys
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H,file_sha256 as sha
from V6.experimental.v7_research_temporal import instant
from V6.experimental.v7_temporal_contract import normalize_calendar
from V6.experimental.v7_research_artifacts import strict_json,_safe,_lock
from V6.experimental.v7_source_snapshot import _atomic_bytes
FAMILIES=('L0_POS','L0_NEG','Ridge','LightGBM')
FEATURE_ORDER=['raw_close_momentum1','raw_intraday_range']

def require(ok,reason):
 if not ok:raise ValueError(reason)

def verify_ref(ref):
 p=_safe(ref['path'])
 require(p.is_file() and re.fullmatch('[0-9a-f]{64}',ref['sha256']) and sha(p)==ref['sha256'],'MISSING_OR_CHANGED_ARTIFACT')
 return p

def payload_for(context,models,features,predictions,*,executed_at,published_at):
 """Validate supplied synthetic scores; labels are deliberately not an input."""
 require(context.get('evidence_class')=='SYNTHETIC','W32_REAL_PREDICTION_NOT_AUTHORIZED')
 for k in ('study_id','dataset_id','feature_id','universe_id','label_policy_id','recipe_set_id'):
  require(isinstance(context.get(k),str) and bool(context[k].strip()),'MISSING_IDENTITY:'+k)
 require(re.fullmatch('[0-9a-f]{40}',context.get('code_commit','')),'CODE_IDENTITY')
 require(context.get('qc_status')=='PASS','QC_NOT_PASS')
 require(context.get('calendar_status')=='SYNTHETIC_QUALIFIED','CALENDAR_UNQUALIFIED')
 calendar=context['calendar'];days=list(normalize_calendar([r['session'] for r in calendar]));require([r['session'] for r in calendar]==days,'CALENDAR_ORDER')
 for n,r in enumerate(calendar):
  opened=instant(r['open_at']);closed=instant(r['close_at'])
  require(opened.astimezone(ZoneInfo('Asia/Taipei')).date().isoformat()==r['session'] and closed.astimezone(ZoneInfo('Asia/Taipei')).date().isoformat()==r['session'] and opened<closed,'CALENDAR_SESSION_CLOCK_MISMATCH')
  if n:require(instant(calendar[n-1]['close_at'])<opened,'CALENDAR_CLOCK_ORDER')
 signal=context['signal_session'];i=days.index(signal)
 require(i+10<len(days) and context['entry_session']==days[i+1],'ENTRY_OR_HORIZON_CALENDAR_UNKNOWN')
 entry=instant(calendar[i+1]['open_at']);executed=instant(executed_at);published=instant(published_at)
 require(instant(context['observation_at'])<=instant(context['retrieved_at'])<=instant(context['qc_at'])<=executed<=published,'FUTURE_DATA_OR_CLOCK_ORDER')
 require(instant(calendar[i]['close_at'])<=executed and instant(context['observation_at'])==instant(calendar[i]['close_at']),'SIGNAL_OBSERVATION_MISMATCH')
 require(bool(context['raw_refs']),'MISSING_RAW_REFS')
 for ref in context['raw_refs']:verify_ref(ref)
 require(context['capture_class'] in {'ORIGINAL_FORWARD_CAPTURE','LATE_CAPTURE','HISTORICAL_BACKFILL','REVISION_CAPTURE'},'CAPTURE_CLASS_UNKNOWN')
 expected={(f,h) for f in FAMILIES for h in (5,10)}
 require(len(models)==8 and {(m['family'],m['horizon']) for m in models}==expected,'MODEL_SET_INCOMPLETE')
 require(len({m['model_id'] for m in models})==8,'MODEL_ID_DUPLICATE')
 for m in models:
  verify_ref(m);require(m['feature_id']==context['feature_id'] and m['feature_order']==FEATURE_ORDER and bool(m.get('recipe_id')),'MODEL_FEATURE_RECIPE_MISMATCH')
 stock_ids=[]
 for f in features:
  require(set(f)=={'stock_id','signal_session','values'},'LABEL_OR_UNKNOWN_FIELDS_IN_FEATURE_FRAME')
  require(f['signal_session']==signal and len(f['values'])==2 and all(type(v) in (int,float) and math.isfinite(v) for v in f['values']),'FEATURE_INVALID')
  stock_ids.append(f['stock_id'])
 require(bool(stock_ids) and len(set(stock_ids))==len(stock_ids),'UNIVERSE_EMPTY_OR_DUPLICATE')
 keys=[(r['stock_id'],r['family'],r['horizon']) for r in predictions]
 require(len(set(keys))==len(keys) and set(keys)=={(s,f,h) for s in stock_ids for f,h in expected},'PREDICTION_KEYS_NOT_FEATURE_UNIVERSE')
 require(all(r['signal_session']==signal and type(r['score']) in (int,float) and math.isfinite(r['score']) for r in predictions),'SCORE_INVALID')
 # Reject accidental future-label payload instead of allowing it into publication.
 require(all(set(r)=={'stock_id','signal_session','family','horizon','score'} for r in predictions),'LABEL_OR_UNKNOWN_FIELDS_IN_PREDICTION')
 timing='PREOPEN' if published<entry else 'MISSED_PREOPEN'
 return {'schema':'w32-synthetic-shadow-payload-v1','evidence_class':'SYNTHETIC','context':context,'models':models,'feature_order':FEATURE_ORDER,'feature_universe':stock_ids,'feature_payload_sha256':H(features),'predictions':predictions,'executed_at':executed_at,'published_at':published_at,'entry_open_at':calendar[i+1]['open_at'],'timing':timing,'prediction_role':'SYNTHETIC_FORWARD' if timing=='PREOPEN' else 'RESEARCH_ONLY','label_status':'PENDING_NOT_READ','planned_exits':{str(h):calendar[i+h]['close_at'] for h in (5,10)},'capture_class':context['capture_class'],'verified_forward_vintage':False,'source_publication_at':context.get('publication_at'),'source_publication_verified':False,'clock_scope':'SUPPLIED_SYNTHETIC_FIXTURE_CLOCKS_NOT_ACTUAL_HISTORICAL_PREDICTION'}

def publish_synthetic(root,context,models,features,predictions,*,executed_at,published_at):
 payload=payload_for(context,models,features,predictions,executed_at=executed_at,published_at=published_at)
 root=_safe(root);root.mkdir(parents=True,exist_ok=True)
 key=H({'study':context['study_id'],'entry_session':context['entry_session']});path=root/(key+'.json')
 with _lock(root/(key+'.lock')):
  if path.exists():
   prior=strict_json(path.read_bytes());require(prior['payload_sha256']==H(payload) and H(prior['payload'])==H(payload),'IMMUTABLE_PREDICTION_CONFLICT')
  else:
   record={'payload':payload,'payload_sha256':H(payload),'research_admitted':False,'actual_archive_created_at':datetime.now(timezone.utc).isoformat()}
   _atomic_bytes(path,(json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode())
 return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def read_prediction(ref):
 record=strict_json(verify_ref(ref).read_bytes())
 require(record.get('research_admitted') is False and record['payload']['evidence_class']=='SYNTHETIC' and H(record['payload'])==record['payload_sha256'],'PREDICTION_RECORD_IDENTITY')
 return record

def evaluation_gate(prediction_ref,label,*,now):
 """Eligibility only; no empirical scoring, universe selection or state mutation."""
 record=read_prediction(prediction_ref);p=record['payload'];ctx=p['context']
 def result(status,reason):return {'status':status,'reason':reason,'research_admitted':False,'prediction_sha256':prediction_ref['sha256']}
 try:
  h=label['horizon'];matched=any(r['stock_id']==label['stock_id'] and r['signal_session']==label['signal_session'] and r['horizon']==h for r in p['predictions'])
  if not matched or label['prediction_sha256']!=prediction_ref['sha256'] or label['label_policy_id']!=ctx['label_policy_id'] or label['exit_at']!=p['planned_exits'][str(h)]:return result('REJECTED','LABEL_IDENTITY_OR_EXIT_MISMATCH')
  exit_at=instant(label['exit_at']);available=instant(label['available_at']);clock=instant(now)
  if clock<instant(p['published_at']):return result('REJECTED','EVALUATION_BEFORE_PREDICTION_PUBLICATION')
  if available<exit_at:return result('REJECTED','AVAILABILITY_BEFORE_MATURITY')
  if clock<=exit_at or clock<available:return result('PENDING_MATURITY','NO_LABEL_JOIN_YET')
  if label.get('eligible') is not True or type(label.get('value')) not in (int,float) or not math.isfinite(label['value']):return result('REJECTED','LABEL_INELIGIBLE_OR_INVALID')
  return result('ELIGIBLE_SYNTHETIC_JOIN','NOT_REAL_LABEL_ADMISSION_OR_EVALUATION')
 except (KeyError,TypeError,ValueError):return result('REJECTED','UNKNOWN_LABEL_OR_MATURITY_EVIDENCE')

if __name__=='__main__':
 raise SystemExit('W32 preparation library only. Run window32-shadow-tests.py; no live inference/activation CLI.')
