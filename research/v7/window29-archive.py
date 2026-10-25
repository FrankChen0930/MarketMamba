"""Independent-process real raw-byte restore and finite resource evidence. No GET/fit."""
import sys,json,runpy,hashlib
from pathlib import Path
from datetime import datetime,timezone
C=runpy.run_path(str(Path(__file__).with_name('window29-common.py')));ROOT=C['ROOT'];OUT=C['OUT'];sha=C['file_sha256'];put=C['put']
with C['cpu_stage']('archive'):
 net=json.loads((ROOT/'finmind/CLOSED.json').read_text());records=[json.loads(p.read_text()) for p in sorted((ROOT/'finmind/records').glob('*.json'))];assert len(records)==net['requests']==68
 starts=[datetime.fromisoformat(r['request_started_at']) for r in records];gaps=[(b-a).total_seconds() for a,b in zip(starts,starts[1:])];assert all(x>=5 for x in gaps)
 assert len({C['canonical_hash'](r['params']) for r in records})==68 and all(r['http_status']==200 and r['status']=='CAPTURED' for r in records)
 for r in records:assert sha(Path(r['raw_path']))==r['raw_sha256']
 shared=ROOT.parent/'finmind-anonymous-window-budget.json';put(ROOT/'network-ledger-at-archive.json',json.loads(shared.read_text()))
 closed=json.loads((ROOT/'COMPLETE.json').read_text());assert closed['Ridge_fits']==2 and len(list(ROOT.glob('RIDGE_*_FIT_CLAIM.json')))==2
 for ref in closed['artifacts'].values():C['checked'](ref)
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.name=='CPU_BUDGET.json' or p.suffix=='.lock':continue
  b=p.read_bytes();C['capacity'](len(b));q=C['RESTORE']/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert q.read_bytes()==b
  else:
   with q.open('xb') as f:f.write(b)
  assert len(b)==q.stat().st_size and sha(p)==sha(q)
  files.append({'path':str(p),'restore':str(q),'bytes':len(b),'sha256':sha(p)})
 # Previous private source/result bytes, excluding protected W19 entirely.
 prior={}
 for name in ['window24-local-artifacts.json','window25-local-artifacts.json','window26-independent-restore.json','window27-local-artifacts.json','window28-local-artifacts.json']:
  inv=json.loads((OUT/name).read_text())['files']
  for f in inv:assert Path(f['path']).stat().st_size==f['bytes'] and sha(Path(f['path']))==f['sha256']
  prior[name]=len(inv)
 proof={'status':'INDEPENDENT_LOCAL_BYTES_RESTORED','files':files,'file_count':len(files),'bytes':sum(f['bytes'] for f in files),'raw_responses':68,'raw_response_bytes':net['raw_bytes'],'minimum_actual_GET_start_interval_seconds':min(gaps),'hourly_claims_max_known':max(json.loads(p.read_text())['known_requests_previous_hour'] for p in (ROOT/'finmind/claims').glob('*.json')),'prior_pins':prior,'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','mutable_cpu_counter':'excluded from immutable evidence inventory; final observation separately recorded'}
 put(OUT/'window29-local-artifacts.json',proof)
resource={'observed_at':datetime.now(timezone.utc).isoformat(),'cpu':json.loads((ROOT/'CPU_BUDGET.json').read_text()),'artifact_plus_restore_bytes':C['size'](),'artifact_limit_bytes':1073741824,'verification_reserve_seconds':60,'global_other_program_CPU_and_external_anonymous_requests':'not globally enforced/observable; study stages OS CPU limits and locks enforced','recovery_charge_600_seconds_not_reset':True};put(OUT/'window29-resource-evidence.json',resource)
print(json.dumps({k:proof[k] for k in ('status','file_count','bytes','raw_responses','raw_response_bytes','minimum_actual_GET_start_interval_seconds','backup')}))
