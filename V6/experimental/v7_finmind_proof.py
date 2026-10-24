"""Bounded W21 anonymous evidence adapter, not a dataset/training platform.

Exact endpoint; <=18 price +3 metadata +1 identical repeat, <=6 symbols,
<=3 <=14-calendar-day windows/symbol. No redirect/retry/token/alternate API.
Current historical version is RETROSPECTIVE_ONLY; no ffill/admission.
"""
from pathlib import Path
from datetime import date,datetime,timezone
from decimal import Decimal,InvalidOperation
import hashlib,http.client,json,urllib.request,urllib.error
from urllib.parse import urlencode
from collections import Counter
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response,verify_snapshot,_atomic_bytes
from V6.experimental.v7_research_artifacts import _safe,_lock,_relative,strict_json
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_manual_collector import NoRedirect,framing_status
ENDPOINT='https://api.finmindtrade.com/api/v4/data'
META={'TaiwanStockInfo','TaiwanStockTradingDate','TaiwanStockDelisting'}
LIMIT=2097152
class ProofIntegrityError(ValueError):pass

def stamp():return datetime.now(timezone.utc).isoformat()
def put(p,o):_atomic_bytes(_safe(p),(json.dumps(o,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode())
def diagnose(body,params,*,market_context=None):
 out={'status':'SOURCE_ERROR','rows':0,'schema':[],'sessions':[],'earliest':None,'latest':None,'securities':[],'counts':{},'reasons':[],'missing_sessions_verified':False,'research_admitted':False}
 try:
  x=strict_json(body)
  if not isinstance(x,dict) or x.get('status')!=200 or not isinstance(x.get('data'),list):raise ValueError('API_APPLICATION_ERROR_OR_SCHEMA')
  rows=x['data'];out['rows']=len(rows)
  if not all(isinstance(r,dict) for r in rows):raise ValueError('ROW_SCHEMA_INVALID')
  out['schema']=sorted(set(k for r in rows for k in r))
  if params['dataset']!='TaiwanStockPrice':out['status']='OBSERVED_METADATA';return out
  counter=Counter();days=set();keys=set();secs=set()
  for r in rows:
   if not {'date','stock_id','open','max','min','close','Trading_Volume'}<=set(r):raise ValueError('PRICE_SCHEMA_INVALID')
   day=date.fromisoformat(r['date']).isoformat();sec=r['stock_id']
   if sec!=params['data_id'] or not params['start_date']<=day<=params['end_date'] or (day,sec) in keys:raise ValueError('STOCK_DATE_OR_DUPLICATE_IDENTITY')
   keys.add((day,sec));days.add(day);secs.add(sec);values=[r[k] for k in ['open','max','min','close','Trading_Volume']]
   if any(v is None for v in values):counter['MISSING_DATA']+=1;continue
   if any(type(v) not in (int,float) for v in values):raise ValueError('NUMERIC_SCHEMA_INVALID')
   op,hi,lo,cl,vol=[Decimal(str(v)) for v in values]
   if not all(v.is_finite() for v in (op,hi,lo,cl,vol)) or min(op,hi,lo,cl,vol)<0 or vol!=vol.to_integral_value():raise ValueError('INVALID_OHLCV')
   if op==hi==lo==cl==0:counter['NO_PUBLISHED_PRICE']+=1
   elif lo>0 and lo<=cl<=hi and (op==0 or not lo<=op<=hi):
    ctx=(market_context or {}).get(day,{})
    counter['MARKET_SEMANTIC_EXCEPTION' if ctx.get('market')=='emerging' and ctx.get('evidence') else 'UNVERIFIED']+=1
   elif 0<lo<=min(op,cl)<=max(op,cl)<=hi:counter['VALID_PRICE']+=1
   else:counter['UNVERIFIED']+=1
  out.update(status='MISSING_DATA' if not rows else 'OBSERVED_PRICE',counts=dict(counter),sessions=sorted(days),securities=sorted(secs),earliest=min(days) if days else None,latest=max(days) if days else None)
 except (OSError,ValueError,TypeError,KeyError,InvalidOperation) as exc:out['reasons']=[str(exc)]
 return out

def fetch_once(params):
 req=urllib.request.Request(ENDPOINT+'?'+urlencode(params),headers={'User-Agent':'MarketMamba-W21-Anonymous-Historical-Proof','Accept-Encoding':'identity'})
 op=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}))
 try:r=op.open(req,timeout=25)
 except urllib.error.HTTPError as e:r=e
 with r:
  body=r.read(LIMIT+1);headers={k.lower():v for k,v in r.headers.items()}
  return r.code,headers,body,len(body)<=LIMIT and framing_status(headers,len(body))!='CONTENT_LENGTH_MISMATCH_OR_INVALID'

def request(root,params,*,kind,fetch=fetch_once,market_context=None):
 root=_safe(root);root.mkdir(parents=True,exist_ok=True);claim=None
 try:
  if (root/'CLOSED.json').exists():return {'status':'WINDOW_CLOSED','reason':'COMPLETED_AUTHORIZATION_NO_UNUSED_BUDGET_REUSE'}
  if (root/'STOP.json').exists():return {'status':'STOPPED','reason':'PRIOR_SOURCE_STOP'}
  if kind not in {'price','metadata','repeat'} or not isinstance(params,dict):raise ValueError('INVALID_REQUEST_KIND')
  if set(params)-{'dataset','data_id','start_date','end_date'}:raise ValueError('UNAUTHORIZED_PARAMETER_OR_TOKEN')
  ds=params.get('dataset')
  if kind=='metadata':
   if ds not in META or set(params)!={'dataset'}:raise ValueError('METADATA_NOT_APPROVED')
  elif ds!='TaiwanStockPrice' or set(params)!={'dataset','data_id','start_date','end_date'} or not isinstance(params['data_id'],str) or not 0<=(date.fromisoformat(params['end_date'])-date.fromisoformat(params['start_date'])).days<=13:raise ValueError('PRICE_OR_WINDOW_NOT_APPROVED')
  (root/'claims').mkdir(exist_ok=True)
  with _lock(root/'budget.lock'):
   claims=[strict_json(p.read_bytes()) for p in sorted((root/'claims').glob('*.json'))]
   for old in claims:
    try:
     seq=old['sequence'];relative=f'records/{seq:02d}.json';rp=root/relative
     if not rp.is_file():raise ProofIntegrityError('PRIOR_CLAIM_WITHOUT_SUCCESSFUL_TERMINAL_RECORD')
     prev=verified_record(root,relative,file_sha256(rp))
     if prev['params']!=old['params'] or prev['kind']!=old['kind'] or prev['sequence']!=seq or prev['status']!='CAPTURED' or prev['http_status']!=200 or prev['response_complete'] is not True:raise ProofIntegrityError('PRIOR_QUERY_NOT_SUCCESSFUL_COMPLETE')
    except (OSError,ValueError,KeyError,TypeError) as exc:raise ProofIntegrityError('PRIOR_EVIDENCE_NOT_RESTORABLE_NO_GET_REPAIR:'+str(exc)) from exc
   if len(claims)>=22 or sum(c['kind']==kind for c in claims)>={'price':18,'metadata':3,'repeat':1}[kind]:raise ValueError('REQUEST_BUDGET_EXHAUSTED')
   same=[c for c in claims if c['params']==params]
   if kind=='repeat' and not any(c['kind']=='price' for c in same):raise ValueError('REPEAT_REQUIRES_EXISTING_EXACT_QUERY')
   if kind!='repeat' and same:raise ValueError('DUPLICATE_QUERY_NO_RETRY')
   symbols={c['params']['data_id'] for c in claims if c['kind']=='price'}
   if kind=='price' and ((params['data_id'] not in symbols and len(symbols)>=6) or sum(c['kind']=='price' and c['params']['data_id']==params['data_id'] for c in claims)>=3):raise ValueError('SYMBOL_OR_WINDOW_BUDGET')
   number=len(claims)+1;claim={'sequence':number,'kind':kind,'params':params,'endpoint':ENDPOINT,'claimed_at':stamp(),'authority':'HumanWindow21_ONLY','token_used':False};put(root/'claims'/f'{number:02d}.json',claim)
  code,headers,body,complete=fetch(params);at=stamp();safeheaders={k.lower():str(v) for k,v in headers.items() if k.lower() in {'content-type','content-length','date','retry-after','location'}}
  if len(body)>LIMIT+1:raise ValueError('RESPONSE_BOUND_EXCEEDED')
  m=snapshot_response(RequestSpec('GET',ENDPOINT,params=params),body,root/'snapshots',retrieved_at=at,parser_version='w21-finmind-proof-v1',content_type=safeheaders.get('content-type','application/octet-stream'),status_code=code)
  if not verify_snapshot(m):raise ValueError('SNAPSHOT_INTEGRITY_FAILED')
  md=strict_json(m.read_bytes());qc=diagnose(body,params,market_context=market_context)
  stopped=code!=200 or not complete or qc['status']=='SOURCE_ERROR'
  record={'schema':'w21-finmind-evidence-v1','sequence':number,'kind':kind,'params':params,'query_id':canonical_hash(params),'endpoint':ENDPOINT,'source':'FinMind','attribution':'FinMind; original data providers TWSE/TPEx per source documentation; noncommercial private research only, no raw redistribution','license_identity':'W20official FinMindFree/Pricing/Disclaimer reviewed, no commercial/redistribution grant','request_started_at':claim['claimed_at'],'retrieved_at':at,'published_at':None,'historical_version':'RETROSPECTIVE_ONLY_NOT_ORIGINAL_ASOF','http_status':code,'response_complete':complete,'headers':safeheaders,'framing_evidence':framing_status(safeheaders,len(body)),'manifest_path':str(m.relative_to(root)),'manifest_sha256':file_sha256(m),'raw_path':'snapshots/'+md['response']['path'],'raw_sha256':md['response']['sha256'],'raw_bytes':len(body),'qc':qc,'market_context':market_context or {},'research_admitted':False,'verified_historical_asof':False,'status':'STOPPED_SOURCE_ERROR' if stopped else 'CAPTURED','code_sha256':file_sha256(Path(__file__))}
  path=root/'records'/f'{number:02d}.json';put(path,record)
  if stopped:put(root/'STOP.json',{'sequence':number,'reason':'HTTP_APPLICATION_SCHEMA_OR_COMPLETENESS_STOP','at':at,'http_status':code})
  return {**record,'record_path':str(path.relative_to(root)),'record_sha256':file_sha256(path)}
 except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as exc:
  if isinstance(exc,ProofIntegrityError):
   stopped={'status':'STOPPED_EVIDENCE_FAILURE','reason':str(exc),'at':stamp()};put(root/'STOP.json',stopped);return stopped
  if claim is None:return {'status':'REFUSED','reason':str(exc)}
  failure={'status':'FAILED','sequence':claim['sequence'],'reason':str(exc),'at':stamp()};put(root/'failures'/f"{claim['sequence']:02d}.json",failure)
  if not (root/'STOP.json').exists():put(root/'STOP.json',failure)
  return failure

def verified_record(root,relative,pin):
 root=_safe(root);path=_safe(root/_relative(relative))
 if file_sha256(path)!=pin:raise ValueError('RECORD_IDENTITY_MISMATCH')
 r=strict_json(path.read_bytes());mp=_safe(root/_relative(r['manifest_path']));raw=_safe(root/_relative(r['raw_path']))
 if r['endpoint']!=ENDPOINT or r['research_admitted'] is not False or r['schema']!='w21-finmind-evidence-v1' or file_sha256(mp)!=r['manifest_sha256'] or not verify_snapshot(mp):raise ValueError('SOURCE_OR_MANIFEST_MISMATCH')
 m=strict_json(mp.read_bytes())
 if m['request']['params']!=r['params'] or m['request']['url']!=ENDPOINT or m['retrieved_at']!=r['retrieved_at'] or m['response']['status_code']!=r['http_status'] or 'snapshots/'+m['response']['path']!=r['raw_path'] or file_sha256(raw)!=r['raw_sha256'] or raw.stat().st_size!=r['raw_bytes'] or diagnose(raw.read_bytes(),r['params'],market_context=r['market_context'])!=r['qc']:raise ValueError('RAW_QUERY_SCHEMA_QC_MISMATCH')
 return r

def restore(root,relative,pin,dest):
 try:
  root=_safe(root);dest=_safe(dest)
  if dest==root:raise ValueError('RESTORE_DESTINATION_MUST_DIFFER')
  r=verified_record(root,relative,pin)
  for rel in [r['raw_path'],r['manifest_path'],relative]:_atomic_bytes(_safe(dest/_relative(rel)),_safe(root/_relative(rel)).read_bytes())
  if not verify_snapshot(dest/r['manifest_path']) or file_sha256(dest/relative)!=pin:raise ValueError('RESTORE_VERIFY_FAILED')
  return {'status':'RESTORED_BYTES_HASH_SCHEMA_RECORDS_VERIFIED','sequence':r['sequence'],'raw_sha256':r['raw_sha256'],'raw_bytes':r['raw_bytes'],'schema':r['qc']['schema'],'records':r['qc']['rows'],'query_id':r['query_id'],'source_http_status':r['http_status'],'research_admitted':False}
 except (OSError,ValueError,KeyError,TypeError) as e:return {'status':'NOT_RESTORABLE','reason':str(e)}
