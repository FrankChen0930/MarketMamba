"""Independent-process W33 raw/derived restore; no network, inference or fitting."""
import json,runpy
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;C=runpy.run_path(str(O/'window33-common.py'));R=C['ROOT'];D=C['RESTORE'];put=C['put'];sha=C['sha']
from V6.experimental.v7_source_snapshot import verify_snapshot,_atomic_bytes
with C['budget']('independent-byte-restore'):
 data=json.loads((O/'window33-data-evidence.json').read_text());assert data['gates']['model_fit']=='BLOCKED' and data['fits_Ridge_LightGBM']==0
 net=json.loads((R/'finmind/CLOSED.json').read_text());assert net['captured']==net['price_requests']==64 and net['metadata_requests']==0
 records=[json.loads(p.read_text()) for p in (R/'finmind/records').glob('*.json')];claims=[json.loads(p.read_text()) for p in (R/'finmind/claims').glob('*.json')];times=sorted(datetime.fromisoformat(r['claimed_at']) for r in claims);intervals=[(b-a).total_seconds() for a,b in zip(times,times[1:])];assert min(intervals)>=5 and max(c['known_requests_previous_hour'] for c in claims)<100
 rows=0
 for r in records:
  assert verify_snapshot(Path(r['manifest'])) and sha(Path(r['raw_path']))==r['raw_sha256'];body=json.loads(Path(r['raw_path']).read_bytes());assert body['status']==200 and len(body['data'])==r['qc']['rows'];rows+=len(body['data'])
  assert r['params']['dataset']=='TaiwanStockPrice' and r['params']['end_date']=='2026-10-22'
 assert rows==data['raw_rows'];files=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.name=='CPU_BUDGET.json' or p.suffix=='.lock':continue
  b=p.read_bytes();C['capacity'](len(b));target=D/p.relative_to(R);_atomic_bytes(target,b);assert sha(p)==sha(target)
  files.append({'path':str(p),'restore':str(target),'bytes':len(b),'sha256':sha(p)})
 old=data['refs']['old_raw']
 for r in old:assert sha(Path(r['raw']))==r['sha256']
 proof={'status':'INDEPENDENT_PROCESS_RAW_DERIVED_RESTORE_PASS','files':files,'file_count':len(files),'bytes_per_copy':sum(f['bytes'] for f in files),'restored_rows':rows,'request_count':64,'price_requests':64,'metadata':0,'min_request_start_interval_seconds':min(intervals),'max_known_prior_hour':max(c['known_requests_previous_hour'] for c in claims),'external_anonymous_usage':'UNKNOWN_NOT_GLOBAL_PROVIDER_QUOTA_CONTROL','new_fit_API_inference_RankIC':0,'old_raw_pins_unchanged':len(old),'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','raw_redistribution':'PRIVATE_ONLY_NO_GIT_PUSH','publication_time':'UNKNOWN_NOT_RETRIEVAL_IMPUTED'};put(O/'window33-local-artifacts.json',proof)
resource={'observed_at':datetime.now(timezone.utc).isoformat(),'cpu':json.loads((R/'CPU_BUDGET.json').read_text()),'CPU_seconds':sum(x['cpu_seconds'] for x in json.loads((R/'CPU_BUDGET.json').read_text())['stages']),'verification_reserve_seconds':60,'CPU_limit':1800,'artifact_plus_restore_bytes':C['size'](),'artifact_limit':1073741824,'source_shell_timeout_seconds':600,'fit_timeout_seconds':300,'fit_attempts':0,'execution_ref':'source tool9ecd4d session94344;data tool414bc3 session38740; each timeout--signalTERM--killafter5s; independent raw restore in separate process','limits_scope':'task resource ledger, not aggregate cost enforcement of unrelated processes'};put(O/'window33-resource-evidence.json',resource)
print(json.dumps({k:proof[k] for k in ['status','file_count','bytes_per_copy','restored_rows','request_count','min_request_start_interval_seconds','old_raw_pins_unchanged','backup']}))
