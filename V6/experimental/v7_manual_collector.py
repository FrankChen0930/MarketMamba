"""Window18 manual one-shot pilot. No scheduler, retries, redirects or fallback.

Reuse v1 immutable snapshots/W09 filesystem guards; W15 adapter stays synthetic.
Budget reservations precede dispatch and survive failure/process death. Raw capture
and restore do not admit data or authenticate publication/calendar/universe.
"""
import argparse,hashlib,http.client,json,subprocess,urllib.request,urllib.error,uuid
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response,verify_snapshot,_atomic_bytes
from V6.experimental.v7_research_artifacts import _safe,_lock,_relative,strict_json
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_research_temporal import instant
from V6.experimental.v7_capture_readiness import calendar_status
from V6.experimental.v7_csv_capture_qc import diagnose,CSV_HEADERS,MISSING

REPO=Path(__file__).resolve().parents[2]
MATRIX=REPO/'research/v7/window15-license-matrix.json'
MATRIX_SHA='274f464b6ca58381b2b8cca4d71109428502a3b2c09c942512f93e99f801cd06'
URLS={'11549':'https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data',
      '11371':'https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&se=EW&o=data'}
MAX_BYTES=2*1024*1024
TIMEOUT=25
LEGACY_QC_SHA='14b81444ef14c9001991c55627b464a639d99db0f94a88f4c32a3cd895bfd55d'

def encoded(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
def stamp():return datetime.now(timezone.utc).isoformat()
def source(dataset):
    if dataset not in URLS or file_sha256(MATRIX)!=MATRIX_SHA:raise ValueError('SOURCE_OR_LICENSE_MATRIX_NOT_APPROVED')
    row=next(r for r in strict_json(MATRIX.read_bytes())['records'] if r.get('dataset_id')==dataset)
    if row['exact_resource_url']!=URLS[dataset] or row['license_id']!='OGDL-1.0' or row['classification']!='LICENSE_CONFIRMED_FOR_SCOPED_USE':raise ValueError('RESOURCE_LICENSE_MISMATCH')
    return {'provider':row['provider'],'dataset_id':dataset,'resource_url':URLS[dataset],
            'license_id':row['license_id'],'license_matrix_sha256':MATRIX_SHA,
            'attribution':f"{row['provider']} / {row['dataset_name']} / OGDL-1.0 https://data.gov.tw/license; source year as observation session; local capture does not imply original vintage"}

def freshness(dataset,sessions,receipt):
    if dataset!='11549':return {'status':'UNKNOWN_EXCHANGE_CALENDAR_UNQUALIFIED'}
    panel=strict_json((REPO/'research/v7/window17-calendar-panel.json').read_bytes());cal=panel['calendar']
    r=calendar_status(cal,canonical_hash(cal),instant(receipt).astimezone(ZoneInfo('Asia/Taipei')).date().isoformat(),receipt,captured_sessions=sessions)
    return {'status':'BOUNDED_LAST_ACTUAL_MATCH_NOT_ADMITTED' if r['status']!='BLOCKED' and sessions==[r['last_actual_session']] else 'UNKNOWN_OR_UNMATCHED_NOT_AUTOMATICALLY_STALE','calendar_verdict':r,'calendar_sha256':canonical_hash(cal),'qualification':'PROVISIONAL_NOT_ADMITTED'}

def archive(root,dataset,body,*,retrieved_at,http_status,headers,mode,complete=True,request_started_at=None):
    binding=source(dataset);root=_safe(root);receipt=instant(retrieved_at).isoformat()
    if mode not in {'MANUAL_ONE_SHOT','OFFLINE_REPLAY','BACKFILL'} or not isinstance(body,bytes) or len(body)>MAX_BYTES+1:raise ValueError('INVALID_BOUNDED_CAPTURE')
    for rel in ('snapshots/raw','snapshots/manifests','captures/'+dataset):_safe(root/rel).mkdir(parents=True,exist_ok=True)
    safe_headers={k.lower():str(v) for k,v in headers.items() if k.lower() in {'content-type','content-length','date','last-modified','etag','retry-after','location'}}
    content_type=safe_headers.get('content-type','application/octet-stream')
    request=RequestSpec('GET',binding['resource_url'])
    digest=hashlib.sha256(body).hexdigest()
    ident=canonical_hash({'source':binding,'raw_sha256':digest,'receipt':receipt,'mode':mode,'http_status':http_status,'complete':complete})
    record=_safe(root/'captures'/dataset/(ident+'.json'))
    with _lock(root/'archive.lock'):
        if record.exists():
            try:existing=verified_record(root,str(record.relative_to(root)),file_sha256(record))
            except OSError as exc:raise ValueError('EXISTING_CAPTURE_NOT_RESTORABLE_NO_REPAIR') from exc
            if existing['raw_sha256']!=digest or existing['headers']!=safe_headers or existing['acquisition_mode']!=mode or existing['response_complete']!=complete or existing['http_status']!=http_status:
                raise ValueError('EXISTING_CAPTURE_IDENTITY_MISMATCH')
            return {**existing,'record_path':str(record.relative_to(root)),'record_sha256':file_sha256(record)}
        manifest=snapshot_response(request,body,root/'snapshots',retrieved_at=receipt,parser_version='window18-csv-raw-v1',content_type=content_type,status_code=http_status)
        if not verify_snapshot(manifest):raise ValueError('PUBLISHED_SNAPSHOT_NOT_VERIFIED')
        m=strict_json(manifest.read_bytes());qc=diagnose(body,dataset)
        csv_type='csv' in content_type.lower()
        framing=framing_status(safe_headers,len(body))
        status=('INCOMPLETE_RAW_RETAINED' if not complete or framing=='CONTENT_LENGTH_MISMATCH_OR_INVALID' else 'HTTP_REJECTED_RAW_RETAINED' if http_status!=200 else
                'RAW_CAPTURED_WITH_QC_FAILURE' if qc['status']=='QC_FAILED' or not csv_type else 'RAW_CAPTURED')
        previous=[]
        for p in sorted(record.parent.glob('*.json')):
            v=strict_json(_safe(p).read_bytes())
            if v['raw_sha256']!=m['response']['sha256']:previous.append({'record_path':str(p.relative_to(root)),'record_sha256':file_sha256(p),'raw_sha256':v['raw_sha256']})
        r={'schema':'v7-manual-capture-pilot-v1','source':binding,'status':status,
           'request_started_at':request_started_at,'retrieved_at_utc':receipt,'retrieved_at_taipei':instant(receipt).astimezone(ZoneInfo('Asia/Taipei')).isoformat(),
           'archive_created_at':stamp(),'published_at':None,'publication_evidence':'UNKNOWN_NOT_RECEIPT_IMPUTED','http_status':http_status,'headers':safe_headers,'response_complete':complete,'framing_evidence':framing,
           'manifest_path':str(manifest.relative_to(root)),'manifest_sha256':file_sha256(manifest),
           'raw_path':'snapshots/'+m['response']['path'],'raw_sha256':m['response']['sha256'],'raw_bytes':len(body),
           'capture_classification':'HISTORICAL_BACKFILL_NOT_ORIGINAL_VINTAGE' if mode=='BACKFILL' else 'OFFLINE_REPLAY_NOT_NEW_NETWORK_CAPTURE' if mode=='OFFLINE_REPLAY' else 'MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED',
           'acquisition_mode':mode,'previous_different_content':previous,'qc':qc,'freshness':freshness(dataset,qc['sessions'],receipt),
           'raw_capture_success':status in {'RAW_CAPTURED','RAW_CAPTURED_WITH_QC_FAILURE'},'research_admitted':False,'verified_forward_vintage':False,'daily_collector_enabled':False,
           'backup_status':'BACKUP_PENDING','capture_code_sha256':file_sha256(Path(__file__)),'qc_code_sha256':file_sha256(Path(__file__).with_name('v7_csv_capture_qc.py'))}
        _atomic_bytes(record,encoded(r))
    return {**r,'record_path':str(record.relative_to(root)),'record_sha256':file_sha256(record)}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def fetch_once(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MarketMamba-Window18-Manual-OneShot','Accept-Encoding':'identity'})
    opener=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}))
    try:response=opener.open(req,timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:response=exc
    with response:
        body=response.read(MAX_BYTES+1)
        headers=dict(response.headers.items())
        complete=len(body)<=MAX_BYTES and framing_status({k.lower():v for k,v in headers.items()},len(body))!='CONTENT_LENGTH_MISMATCH_OR_INVALID'
        return body,response.code,headers,complete

def run_one(dataset,root,budget_root,*,fetch=fetch_once):
    attempt=None
    try:
        binding=source(dataset);root=_safe(root);budget=_safe(budget_root);budget.mkdir(parents=True,exist_ok=True)
        with _lock(budget/'budget.lock'):
            claim=_safe(budget/(dataset+'.json'))
            if claim.exists():return {'status':'BUDGET_EXHAUSTED','dataset_id':dataset,'network_dispatched':False}
            started=stamp();_atomic_bytes(claim,encoded({'authority':'HumanWindow18_ONE_GET_PER_EXACT_RESOURCE','source':binding,'claimed_at':started,'attempt_id':uuid.uuid4().hex,'status':'RESERVED_NO_RETRY_EVEN_ON_FAILURE'}))
        attempt=_safe(budget/(dataset+'-result.json'))
        body,status,headers,complete=fetch(binding['resource_url'])
        r=archive(root,dataset,body,retrieved_at=stamp(),request_started_at=started,http_status=status,headers=headers,mode='MANUAL_ONE_SHOT',complete=complete)
        _atomic_bytes(attempt,encoded(r));return r
    except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as exc:
        result={'status':'FAILED','dataset_id':dataset,'reason':type(exc).__name__+':'+str(exc),'research_admitted':False}
        if attempt is not None:
            try:_atomic_bytes(attempt,encoded(result))
            except (OSError,ValueError):result['failure_log_storage']='FAILED_NOT_SUCCESS'
        return result

def framing_status(headers,size):
    value=headers.get('content-length')
    if value is None:return 'UNKNOWN_NO_CONTENT_LENGTH'
    try:
        if not str(value).isdigit() or int(value)!=size:return 'CONTENT_LENGTH_MISMATCH_OR_INVALID'
    except (ValueError,TypeError):return 'CONTENT_LENGTH_MISMATCH_OR_INVALID'
    return 'CONTENT_LENGTH_MATCH'


def verified_record(root,record_path,expected_record_sha256):
    """Read before any retry writes: missing committed bytes are not repaired."""
    record=_safe(root/_relative(record_path))
    if file_sha256(record)!=expected_record_sha256:raise ValueError('RECORD_HASH_MISMATCH')
    r=strict_json(record.read_bytes());binding=source(r['source']['dataset_id'])
    if r['schema']!='v7-manual-capture-pilot-v1' or r['source']!=binding or r['research_admitted'] is not False or r['verified_forward_vintage'] is not False or r['daily_collector_enabled'] is not False:raise ValueError('RECORD_SOURCE_IDENTITY_MISMATCH')
    mpath=_safe(root/_relative(r['manifest_path']));raw=_safe(root/_relative(r['raw_path']))
    if file_sha256(mpath)!=r['manifest_sha256'] or not verify_snapshot(mpath):raise ValueError('SNAPSHOT_NOT_RESTORABLE')
    m=strict_json(mpath.read_bytes())
    if m['request']['url']!=binding['resource_url'] or m['retrieved_at']!=r['retrieved_at_utc'] or m['response']['status_code']!=r['http_status'] or 'snapshots/'+m['response']['path']!=r['raw_path'] or file_sha256(raw)!=r['raw_sha256'] or raw.stat().st_size!=r['raw_bytes']:raise ValueError('RAW_OR_REQUEST_MISMATCH')
    code=r['qc_code_sha256'];current=file_sha256(Path(__file__).with_name('v7_csv_capture_qc.py'))
    if code not in {current,LEGACY_QC_SHA}:raise ValueError('QC_CODE_IDENTITY_NOT_SUPPORTED')
    # Preserve first-run diagnostic exactly, including its known token limitation.
    tokens=MISSING-{'----'} if code==LEGACY_QC_SHA else MISSING
    if diagnose(raw.read_bytes(),binding['dataset_id'],missing_tokens=tokens)!=r['qc']:raise ValueError('QC_ROW_REPLAY_MISMATCH')
    return r


def restore(root,record_path,expected_record_sha256,destination):
    try:
        root=_safe(root);r=verified_record(root,record_path,expected_record_sha256)
        binding=r['source'];raw=_safe(root/_relative(r['raw_path']))
        qc=diagnose(raw.read_bytes(),binding['dataset_id'])
        dest=_safe(destination)
        if dest==root:raise ValueError('RESTORE_DESTINATION_MUST_DIFFER')
        for rel in (r['raw_path'],r['manifest_path'],record_path):_atomic_bytes(_safe(dest/_relative(rel)),_safe(root/_relative(rel)).read_bytes())
        if not verify_snapshot(dest/r['manifest_path']) or file_sha256(dest/r['raw_path'])!=r['raw_sha256'] or file_sha256(dest/record_path)!=expected_record_sha256:raise ValueError('RESTORE_VERIFICATION_FAILED')
        return {'status':'RESTORED_BYTES_HASH_ROWS_VERIFIED','dataset_id':binding['dataset_id'],'rows':qc['rows'],'original_qc_status':r['qc']['status'],'current_qc_status':qc['status'],'qc_version_changed':r['qc_code_sha256']!=file_sha256(Path(__file__).with_name('v7_csv_capture_qc.py')),'framing_evidence':framing_status(r['headers'],r['raw_bytes']),'raw_bytes':r['raw_bytes'],'raw_sha256':r['raw_sha256'],'destination':str(dest),'research_admitted':False,'backup_status':'BACKUP_PENDING_SAME_HOST_COPY'}
    except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as exc:return {'status':'NOT_RESTORABLE','reason':str(exc)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['live','offline','restore']);p.add_argument('--root',required=True);p.add_argument('--dataset',choices=list(URLS)+['all']);p.add_argument('--input');p.add_argument('--retrieved-at');p.add_argument('--record');p.add_argument('--record-sha');p.add_argument('--destination');a=p.parse_args()
    if a.mode=='live':
        if not a.dataset:p.error('--dataset required')
        common=Path(subprocess.check_output(['git','-C',str(REPO),'rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()).parent
        root=_safe(a.root)
        if '.worktrees' in root.parts or not root.is_relative_to(common/'.artifacts/window18-manual-pilot'):p.error('live root must be repository .artifacts/window18-manual-pilot[/...] outside worktrees')
        results=[run_one(d,root,common/'.artifacts/window18-network-budget') for d in (URLS if a.dataset=='all' else [a.dataset])]
    elif a.mode=='offline':
        if a.dataset not in URLS or not a.input or not a.retrieved_at:p.error('offline requires dataset,input,retrieved-at; no network')
        with open(a.input,'rb') as f:body=f.read(MAX_BYTES+1)
        results=[archive(a.root,a.dataset,body,retrieved_at=a.retrieved_at,http_status=200,headers={'Content-Type':'text/csv'},mode='OFFLINE_REPLAY',complete=len(body)<=MAX_BYTES)]
    else:
        if not all([a.record,a.record_sha,a.destination]):p.error('restore requires record,record-sha,destination')
        results=[restore(a.root,a.record,a.record_sha,a.destination)]
    print(json.dumps(results,ensure_ascii=False,indent=2));return 0 if all(r['status'] in {'RAW_CAPTURED','RESTORED_BYTES_HASH_ROWS_VERIFIED'} for r in results) else 2

if __name__=='__main__':raise SystemExit(main())
