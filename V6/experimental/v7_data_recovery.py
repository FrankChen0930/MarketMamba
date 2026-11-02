"""Recovery v0.4 offline decisions/attempt evidence. No GET, scheduler or inference.

Counters are OFFLINE reference-policy simulations, never W19 budget mutations.
A source claim is not authenticity evidence; UNKNOWN never becomes suspension
without a matching pinned document (or explicitly synthetic test evidence).
"""
import csv,io,json,math,re
from pathlib import Path
from datetime import date,datetime,timezone
from decimal import Decimal,InvalidOperation
from V6.experimental.v7_csv_capture_qc import CSV_HEADERS,MISSING,diagnose
from V6.experimental.v7_manual_collector import source
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H,file_sha256 as sha
from V6.experimental.v7_research_artifacts import _safe,_lock,strict_json
from V6.experimental.v7_source_snapshot import _atomic_bytes

def require(ok,reason):
 if not ok:raise ValueError(reason)
def write_once(path,value):
 _atomic_bytes(_safe(path),(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode())
 return {'path':str(path),'sha256':sha(path)}
def read_ref(ref):
 p=_safe(ref['path']);require(p.is_file() and sha(p)==ref['sha256'],'ARTIFACT_MISSING_OR_CHANGED');return strict_json(p.read_bytes())
def valid_ohlc(v):
 return all(type(v.get(k)) in (int,float) and math.isfinite(v[k]) and v[k]>0 for k in ('open','high','low','close')) and v['low']<=min(v['open'],v['close'])<=max(v['open'],v['close'])<=v['high']
def parse_csv(body,dataset):
 qc=diagnose(body,dataset);require(not qc['fatal_errors'],'SOURCE_SCHEMA_FAILURE')
 reader=csv.DictReader(io.StringIO(body.decode('utf-8-sig')));require(reader.fieldnames==CSV_HEADERS[dataset],'CSV_SCHEMA_MISMATCH');out=[]
 names=['開盤價','最高價','最低價','收盤價','成交股數'] if dataset=='11549' else ['開盤','最高','最低','收盤','成交股數']
 for r in reader:
  raw=r[reader.fieldnames[0]].strip();day=date(int(raw[:3])+1911,int(raw[3:5]),int(raw[5:])).isoformat();values={};missing=[];bad=[]
  for field,column in zip(['open','high','low','close','volume'],names):
   token=r[column].strip()
   if token in MISSING:values[field]=None;missing.append(field)
   else:
    try:n=Decimal(token.replace(',',''));require(n.is_finite(),'NONFINITE');values[field]=float(n)
    except (ValueError,InvalidOperation):values[field]=None;bad.append(field)
  ok=valid_ohlc(values);vol=values['volume'];basic=ok and type(vol) in (int,float) and vol>=0 and vol.is_integer()
  out.append({'market':source(dataset)['provider'],'dataset_id':dataset,'session':day,'security_id':r[reader.fieldnames[1]].strip(),'name':r[reader.fieldnames[2]],'values':values,'missing_fields':missing,'invalid_fields':bad,'ohlc_valid':ok,'basic_valid':basic,'reason':'MISSING_OHLCV_NO_FILL' if missing else 'INVALID_OHLCV' if not basic else 'PASS','instrument_hint':'LEXICAL_SPECIAL_PRODUCT_CANDIDATE_NOT_MASTER_VERIFIED' if len(r[reader.fieldnames[1]].strip())!=4 or r[reader.fieldnames[1]].startswith('0') else 'ORDINARY_TYPE_UNKNOWN_WITHOUT_SECURITY_MASTER'})
 return out,qc

def bound_evidence(e,row):
 if not isinstance(e,dict) or any(e.get(k)!=row[k] for k in ('market','security_id','session')):return False
 if e.get('evidence_class')=='SYNTHETIC':return str(e.get('evidence_ref','')).startswith('SYNTHETIC_')
 ref=e.get('evidence_ref')
 if not isinstance(ref,dict):return False
 try:return _safe(ref['path']).is_file() and sha(Path(ref['path']))==ref['sha256']
 except (ValueError,OSError,KeyError,TypeError):return False

def classify(row,universe,evidence=None):
 scope=(row['market'],row['security_id']) in universe;blocked=scope and not row['ohlc_valid']
 if row.get('source_failure'):state='SOURCE_LEVEL_FAILURE';reason='SOURCE_HTTP_SCHEMA_OR_IDENTITY_FAILURE'
 elif row['basic_valid']:state='VALID_DATA';reason='RAW_BASIC_QC_PASS_NOT_RESEARCH_ADMISSION'
 elif not scope:state='OUT_OF_RESEARCH_UNIVERSE';reason='EXPLICIT_MARKET_SECURITY_SCOPE_NOT_AN_INSTRUMENT_TYPE_ASSERTION'
 elif bound_evidence(evidence,row) and evidence.get('kind') in {'LEGITIMATE_NO_PRICE','KNOWN_SUSPENSION'}:state=evidence['kind'];reason='MATCHED_PINNED_EVIDENCE_NOT_INFERRED_FROM_ZERO_OR_PERSISTENCE'
 else:state='UNKNOWN_MISSING';reason='NO_BOUND_LEGAL_NO_PRICE_OR_SUSPENSION_EVIDENCE'
 return {'state':state,'reason':reason,'market':row['market'],'session':row['session'],'security_id':row['security_id'],'in_universe':scope,'blocks_features':blocked,'missing_fields':row['missing_fields'],'evidence':evidence if bound_evidence(evidence,row) else None,'research_admitted':False}

def checked_scope(scope):
 require(set(scope)=={'market','source','session'},'BUDGET_SCOPE_MUST_BE_RESOURCE_SESSION_NOT_STOCK')
 require(scope['source']==source(scope['source']['dataset_id']) and scope['market']==scope['source']['provider'],'UNAPPROVED_SOURCE_OR_MARKET')
 date.fromisoformat(scope['session']);return H(scope)

def decide(classification,*,scope,update_evidence,source_permitted,quota_available,ledger,http_status=200):
 checked_scope(scope);require(classification['market']==scope['market'] and classification['session']==scope['session'],'CLASSIFICATION_SCOPE_SESSION_MISMATCH');counts=ledger.counts(scope);state=classification['state'];candidate=False;reason=classification['reason']
 if http_status in {402,403,429} or ledger.halted(scope):state='SOURCE_LEVEL_FAILURE';reason='SOURCE_STOP_NO_BYPASS'
 elif not quota_available:reason='QUOTA_OR_BUDGET_UNAVAILABLE'
 elif counts['CONTENT_RECOVERY']>=ledger.policy['content_limit']:state='UNRECOVERABLE_AFTER_POLICY_LIMIT';reason='CONTENT_RECOVERY_CYCLE_LIMIT_NO_FURTHER_AUTOMATIC_GET'
 elif state in {'VALID_DATA','LEGITIMATE_NO_PRICE','KNOWN_SUSPENSION','OUT_OF_RESEARCH_UNIVERSE','SOURCE_LEVEL_FAILURE'}:reason='NO_USEFUL_CONTENT_RECOVERY_FOR_CLASSIFICATION'
 elif not classification['blocks_features']:reason='MISSING_ONLY_NONREQUIRED_FIELDS_FOR_FIXED_TWO_FEATURES'
 elif source_permitted is not True or not isinstance(update_evidence,str) or not update_evidence.strip():reason='SOURCE_PERMISSION_OR_UPDATE_RATIONALE_MISSING'
 else:state='RECOVERABLE_MISSING_CANDIDATE';candidate=True;reason='REQUIRED_INPUT_PLAUSIBLE_UPDATE_SOURCE_PERMISSION_AND_UNUSED_CYCLE'
 return {'state':state,'recovery_candidate':candidate,'reason':reason,'scope_sha256':H(scope),'classification':classification,'update_evidence':update_evidence,'attempt_counts':counts,'policy_sha256':H(ledger.policy),'execution_authorized':False,'status':'OFFLINE_PLAN_ONLY_NOT_A_GET'}

class AttemptLedger:
 """Small append-only claims/outcomes on existing no-clobber writer + file lock."""
 def __init__(self,root,policy):
  require(policy.get('status')=='OFFLINE_REFERENCE_NOT_ACTIVATED' and type(policy.get('content_limit')) is int and 0<=policy['content_limit']<=1 and type(policy.get('network_retry_limit')) is int and 0<=policy['network_retry_limit']<=2 and 1<=policy.get('physical_limit',0)<=4,'POLICY_NOT_WITHIN_OFFLINE_REFERENCE_SCOPE')
  self.root=_safe(root);self.root.mkdir(parents=True,exist_ok=True);self.policy=dict(policy);write_once(self.root/'policy.json',policy)
 def claims(self,scope):
  key=checked_scope(scope);return [strict_json(p.read_bytes()) for p in (self.root/'claims').glob('*.json') if strict_json(p.read_bytes())['scope_sha256']==key]
 def counts(self,scope):
  rows=self.claims(scope);return {kind:sum(r['kind']==kind for r in rows) for kind in ['NORMAL_CAPTURE','CONTENT_RECOVERY','NETWORK_RETRY']}
 def halted(self,scope):
  checked_scope(scope)
  for path in (self.root/'claims').glob('*.json'):
   c=strict_json(path.read_bytes())
   if c['scope']['source']!=scope['source'] or c['scope']['market']!=scope['market']:continue
   path=self.root/'outcomes'/(H(c['event_id'])+'.json')
   if path.exists():
    o=strict_json(path.read_bytes())['outcome']
    if o.get('http_status') in {402,403,429} or o.get('status') in {'SOURCE_HALT','SOURCE_IDENTITY_STOP'}:return True
    if c['scope_sha256']==H(scope) and o.get('status')=='SESSION_MISMATCH_STOP':return True
  return False
 def reserve(self,scope,kind,event_id,*,decision=None,retry_of=None):
  key=checked_scope(scope);require(kind in {'NORMAL_CAPTURE','CONTENT_RECOVERY','NETWORK_RETRY'},'ATTEMPT_KIND');intent={'scope':scope,'scope_sha256':key,'kind':kind,'event_id':event_id,'decision':decision,'retry_of':retry_of,'policy_sha256':H(self.policy),'actual_GET':False,'evidence_class':'OFFLINE_SIMULATED_ATTEMPT'};path=self.root/'claims'/(H(event_id)+'.json')
  with _lock(self.root/'attempt.lock'):
   if path.exists():
    prior=strict_json(path.read_bytes());prior.pop('actual_offline_recorded_at',None);require(prior==intent,'ATTEMPT_ID_CONFLICT');return {'path':str(path),'sha256':sha(path)}
   require(not self.halted(scope),'SOURCE_HALTED');counts=self.counts(scope);require(sum(counts.values())<self.policy['physical_limit'],'TOTAL_OFFLINE_DISPATCH_LIMIT')
   if kind=='NORMAL_CAPTURE':require(counts[kind]<1,'NORMAL_CAPTURE_LIMIT')
   elif kind=='CONTENT_RECOVERY':require(counts[kind]<self.policy['content_limit'] and decision and decision.get('recovery_candidate') is True and decision['scope_sha256']==key and decision['policy_sha256']==H(self.policy),'CONTENT_RECOVERY_NOT_AUTHORIZED_OR_LIMIT')
   else:
    require(counts[kind]<self.policy['network_retry_limit'] and retry_of,'NETWORK_RETRY_LIMIT');parent=read_ref(retry_of);require(_safe(retry_of['path'])==self.root/'claims'/(H(parent['event_id'])+'.json'),'RETRY_NOT_IN_THIS_LEDGER');require(parent['scope_sha256']==key and parent['policy_sha256']==H(self.policy),'RETRY_SCOPE_IDENTITY');out=read_ref({'path':str(self.root/'outcomes'/(H(parent['event_id'])+'.json')),'sha256':sha(self.root/'outcomes'/(H(parent['event_id'])+'.json'))});require(out['outcome'].get('status')=='NETWORK_FAILURE','RETRY_REQUIRES_TRANSIENT_FAILURE')
   if kind=='NETWORK_RETRY':
    # Consume each failure once. Compare immutable claim identity, not caller
    # annotations on a ref. Idempotent reuse of the same event returned above.
    used=any(c['kind']=='NETWORK_RETRY' and c.get('retry_of',{}).get('sha256')==retry_of['sha256'] for c in self.claims(scope))
    require(not used,'FAILURE_PARENT_ALREADY_CONSUMED_BY_RETRY')
   return write_once(path,{**intent,'actual_offline_recorded_at':datetime.now(timezone.utc).isoformat()})
 def finish(self,claim,outcome):
  c=read_ref(claim);require(_safe(claim['path'])==self.root/'claims'/(H(c['event_id'])+'.json') and c['policy_sha256']==H(self.policy),'OUTCOME_POLICY_OR_LEDGER_IDENTITY');path=self.root/'outcomes'/(H(c['event_id'])+'.json');intent={'claim':claim,'outcome':outcome,'actual_GET':False}
  if path.exists():
   old=strict_json(path.read_bytes());old.pop('actual_offline_recorded_at',None);require(old==intent,'OUTCOME_ID_CONFLICT');return {'path':str(path),'sha256':sha(path)}
  return write_once(path,{**intent,'actual_offline_recorded_at':datetime.now(timezone.utc).isoformat()})

def recovery_outcome(before,after):
 checked_scope({k:before[k] for k in ['market','source','session']});required=set(before['missing_required']);remaining=required;repaired=set();new=set()
 if before['source']!=after['source'] or before['market']!=after['market']:status='SOURCE_IDENTITY_STOP'
 elif before['session']!=after['session']:status='SESSION_MISMATCH_STOP'
 elif after.get('qc_fatal'):status='SOURCE_HALT'
 elif before['raw_sha256']==after['raw_sha256']:status='SAME_CONTENT_NO_REPAIR_STOP'
 else:
  missing=set(after['missing_required']);remaining=required&missing;new=missing-required;repaired=required-missing;status='REPAIRED_NEW_IMMUTABLE_VINTAGE' if repaired and not remaining and not new else 'PARTIAL_REPAIR_STOP' if repaired else 'STILL_MISSING_STOP'
 return {'status':status,'repaired':sorted(repaired),'remaining':sorted(remaining|new),'stop_after_single_content_cycle':True,'backfill_queue':[{'security_id':s,'session':before['session'],'status':'PENDING_APPROVED_HISTORICAL_SOURCE_NOT_CURRENT_LATEST_GUARANTEE'} for s in sorted(remaining|new)],'verified_forward_vintage':False,'original_snapshot_must_remain':True}
