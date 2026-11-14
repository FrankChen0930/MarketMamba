"""One approved non-price query for W26. No new framework or V6 modification."""
from pathlib import Path
from datetime import date,datetime,timezone
import sys,json,urllib.request,urllib.error,http.client
from urllib.parse import urlencode
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response,verify_snapshot,_atomic_bytes
from V6.experimental.v7_manual_collector import NoRedirect,framing_status
from V6.experimental.v7_research_artifacts import strict_json
from V6.experimental.v7_corrected_e5_replication import file_sha256
ENDPOINT='https://api.finmindtrade.com/api/v4/data'
PARAMS={'dataset':'TaiwanStockCapitalReductionReferencePrice','data_id':'2321','start_date':'2021-01-01','end_date':'2023-12-31'}
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window26-label-integrity/finmind')
def stamp():return datetime.now(timezone.utc).isoformat()
def put(path,obj):_atomic_bytes(path,(json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode())
def validate_query(params):
 if params!=PARAMS:raise ValueError('QUERY_NOT_EXACT_APPROVED_NONPRICE_PLAN')
def crosses_event(entry,exit,effective):
 return date.fromisoformat(entry)<date.fromisoformat(effective)<=date.fromisoformat(exit)
def reserve_request(root):
 root.mkdir(parents=True,exist_ok=True)
 if any(root.glob('*.claim.json')) or (root/'STOP.json').exists() or (root/'CLOSED.json').exists():raise ValueError('REQUEST_ALREADY_SPENT_OR_CLOSED_NO_RETRY')
 put(root/'finmind-01.claim.json',{'endpoint':ENDPOINT,'params':PARAMS,'claimed_at':stamp(),'authority':'HumanWindow26_ONE_PLANNED_OF_SIX_MAX_NONPRICE','token':False})
def capture():
 validate_query(PARAMS);reserve_request(ROOT)
 try:
  op=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}));req=urllib.request.Request(ENDPOINT+'?'+urlencode(PARAMS),headers={'User-Agent':'MarketMamba-W26-Event-Proof','Accept-Encoding':'identity'})
  try:r=op.open(req,timeout=25)
  except urllib.error.HTTPError as e:r=e
  with r:body=r.read(2097153);code=r.code;headers={k.lower():str(v) for k,v in r.headers.items()}
  at=stamp();mp=snapshot_response(RequestSpec('GET',ENDPOINT,params=PARAMS),body,ROOT/'snapshots',retrieved_at=at,parser_version='w26-capital-event-v1',content_type=headers.get('content-type','unknown'),status_code=code);assert verify_snapshot(mp);m=strict_json(mp.read_bytes());framing=framing_status(headers,len(body))
  stopped=code!=200 or len(body)>2097152 or framing=='CONTENT_LENGTH_MISMATCH_OR_INVALID';rows=[];reason=None
  if not stopped:
   payload=strict_json(body)
   if payload.get('status')!=200 or not isinstance(payload.get('data'),list):stopped=True;reason='API_ACCESS_OR_SCHEMA_NOT_APPROVED'
   else:
    rows=payload['data']
    for row in rows:
     if row.get('stock_id')!='2321' or not '2021-01-01'<=row.get('date','')<='2023-12-31':stopped=True;reason='REQUEST_IDENTITY_OR_RANGE_MISMATCH'
  record={'schema':'w26-finmind-event-evidence-v1','source':'FinMind','dataset':PARAMS['dataset'],'endpoint':ENDPOINT,'params':PARAMS,'license_scope':'Freeindividualstock/private noncommercial attribution/no raw redistribution per archivedofficialdocs','http_status':code,'retrieved_at':at,'publication_at':'UNKNOWN','historical_original_vintage':False,'manifest':str(mp),'manifest_sha256':file_sha256(mp),'raw_path':str(ROOT/'snapshots'/m['response']['path']),'raw_sha256':m['response']['sha256'],'bytes':len(body),'rows':len(rows),'schema_fields':sorted(set(k for row in rows for k in row)),'framing':framing,'status':'STOPPED' if stopped else 'OBSERVED_EVENT_METADATA','reason':reason,'code_sha256':file_sha256(Path(__file__)),'research_admitted':False}
  put(ROOT/'record.json',record)
  if stopped:put(ROOT/'STOP.json',{'reason':reason or 'HTTP_SIZE_FRAMING_STOP','http_status':code,'at':at})
  put(ROOT/'CLOSED.json',{'requests':1,'maximum':6,'unused':5,'no_budget_reuse':True,'at':stamp()});print(json.dumps(record));return record
 except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as exc:
  put(ROOT/'STOP.json',{'reason':str(exc),'at':stamp(),'request_spent':True});put(ROOT/'CLOSED.json',{'requests':1,'maximum':6,'unused':5,'reason':'ERROR_NO_RETRY'});raise
if __name__=='__main__':capture()
