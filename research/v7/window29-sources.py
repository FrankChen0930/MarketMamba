"""Single-concurrency W29 anonymous requests; no retries/redirects/fallback/token."""
import sys,json,runpy,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_finmind_proof import fetch_once,diagnose,stamp,ENDPOINT,LIMIT
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response,verify_snapshot,_atomic_bytes
from V6.experimental.v7_research_artifacts import _lock
C=runpy.run_path(str(Path(__file__).with_name('window29-common.py')));ROOT=C['ROOT'];OUT=C['OUT'];put=C['put'];sha=C['file_sha256'];hashof=C['canonical_hash'];enc=C['enc']
SHARED=ROOT.parent/'finmind-anonymous-window-budget.json'
def validate_query(q,cohort):
 if set(q)!={'dataset','data_id','start_date','end_date'} or q['dataset']!='TaiwanStockPrice' or q['data_id'] not in cohort['price_queries'] or q['start_date']!='2021-01-01' or q['end_date']!='2023-12-31':raise ValueError('QUERY_NOT_IN_FROZEN_W29_AUTHORIZATION')
def budget_check(total,price,known_hour):
 if total>=80 or price>=70 or known_hour>=100:raise ValueError('REQUEST_OR_KNOWN_SHARED_HOURLY_BUDGET_EXHAUSTED')
def source_stop(code,qc,complete):return code!=200 or not complete or qc['status']=='SOURCE_ERROR'
def prior_claims():
 paths=[]
 for rel in ['window21-finmind-proof/claims','window22-historical-panel/finmind/claims','window24-expanded-pilot/finmind/claims']:
  paths.extend((ROOT.parent/rel).glob('*.json'))
 paths.append(ROOT.parent/'window26-label-integrity/finmind/finmind-01.claim.json');result=[]
 for p in paths:
  if p.is_file():
   r=json.loads(p.read_text());at=r.get('claimed_at') or r.get('request_started_at')
   if at:result.append({'at':at,'source':str(p)})
 return result

def main():
 cohort=json.loads((OUT/'window29-cohort.json').read_text());assert cohort==json.loads((ROOT/'FROZEN_COHORT.json').read_text());pin=json.loads((ROOT/'COHORT_PIN.json').read_text());assert hashof(cohort)==pin['semantic_sha256'] and sha(OUT/'window29-cohort.json')==pin['file_sha256']
 plan=json.loads((OUT/'window29-plan.json').read_text());assert plan['requests']['minimum_interval_seconds']==5
 net=ROOT/'finmind';net.mkdir(exist_ok=True)
 if any((net/n).exists() for n in ('CLOSED.json','STOP.json')):raise ValueError('W29_NETWORK_ALREADY_CLOSED')
 if any((net/'claims').glob('*.json')):raise ValueError('NO_REQUEST_REDISPATCH_AFTER_PRIOR_ATTEMPT')
 with C['cpu_stage']('source-capture'):
  for sec in cohort['price_queries']:
   q={'dataset':'TaiwanStockPrice','data_id':sec,'start_date':'2021-01-01','end_date':'2023-12-31'};validate_query(q,cohort)
   claims=list((net/'claims').glob('*.json'));number=len(claims)+1;C['capacity'](2*(LIMIT+1))
   with _lock(SHARED.with_suffix('.lock')):
    ledger=json.loads(SHARED.read_text()) if SHARED.exists() else {'schema':'local-anonymous-finmind-rate-ledger-v1','external_other_process_account_usage':'UNKNOWN','events':[]}
    now=datetime.now(timezone.utc);events=prior_claims()+ledger['events'];last=max((datetime.fromisoformat(r['at']) for r in events),default=now-timedelta(seconds=6))
    if last>now:raise ValueError('CLOCK_ANOMALY_NO_GET')
    delay=max(0,5-(now-last).total_seconds())
    if delay:time.sleep(delay)
    now=datetime.now(timezone.utc);known=sum(datetime.fromisoformat(r['at'])>=now-timedelta(hours=1) for r in events);budget_check(len(claims),len(claims),known)
    claim={'sequence':number,'kind':'price','params':q,'endpoint':ENDPOINT,'claimed_at':now.isoformat(),'known_requests_previous_hour':known,'shared_counter_scope':'known local W21/W22/W24/W26 plus cooperating ledger; unrelated clients UNKNOWN','min_interval_seconds':5,'authorization':'HumanWindow29','cohort_sha256':pin['semantic_sha256']};put(net/'claims'/f'{number:02d}.json',claim)
    ledger['events'].append({'at':claim['claimed_at'],'source':'HumanWindow29','query_sha256':hashof(q)});_atomic_bytes(SHARED,enc(ledger))
    try:
     code,headers,body,complete=fetch_once(q);at=stamp();m=snapshot_response(RequestSpec('GET',ENDPOINT,params=q),body,net/'snapshots',retrieved_at=at,parser_version='w29-existing-finmind-fetch-v1',content_type=headers.get('content-type','UNKNOWN'),status_code=code);assert verify_snapshot(m);md=json.loads(m.read_text());qc=diagnose(body,q);stop=source_stop(code,qc,complete)
     r={'schema':'w29-finmind-capture-v1','sequence':number,'params':q,'request_started_at':claim['claimed_at'],'retrieved_at':at,'publication_time':'UNKNOWN','http_status':code,'headers':{k:v for k,v in headers.items() if k in ['content-type','content-length','date','retry-after','location']},'response_complete':complete,'manifest':str(m),'manifest_sha256':sha(m),'raw_path':str(net/'snapshots'/md['response']['path']),'raw_sha256':md['response']['sha256'],'raw_bytes':len(body),'qc':qc,'status':'SOURCE_STOP' if stop else 'CAPTURED','license_identity':cohort['FinMind_license'],'attribution':'FinMind; underlying TWSE/TPEx providers; private noncommercial, no raw redistribution','historical_version':'RETROSPECTIVE_CURRENT_NOT_ORIGINAL_ASOF','code_sha256':sha(Path(__file__))};put(net/'records'/f'{number:02d}.json',r)
    except Exception as e:
     put(net/'STOP.json',{'sequence':number,'reason':type(e).__name__+':'+str(e),'no_retry':True});print('STOP',number,sec,str(e),flush=True);break
   print(json.dumps({'request':number,'security':sec,'HTTP':code,'rows':qc['rows'],'bytes':len(body),'status':r['status']}),flush=True)
   if stop:put(net/'STOP.json',{'sequence':number,'HTTP':code,'reason':'HTTP_APPLICATION_SCHEMA_OR_COMPLETENESS_STOP','no_retry':True});break
  records=[json.loads(p.read_text()) for p in sorted((net/'records').glob('*.json'))];n=len(list((net/'claims').glob('*.json')))
  closed={'status':'CLOSED_NO_UNUSED_BUDGET_REUSE','requests':n,'price_requests':n,'metadata_requests':0,'max_total':80,'max_price':70,'unused_total':80-n,'source_stopped':(net/'STOP.json').exists(),'captured':sum(r['status']=='CAPTURED' for r in records),'raw_bytes':sum(r['raw_bytes'] for r in records),'shared_ledger_sha256':sha(SHARED),'no_retry_redirect_token_fallback':True};put(net/'CLOSED.json',closed);put(OUT/'window29-network.json',closed);print(json.dumps(closed),flush=True)
if __name__=='__main__':main()
