"""Bounded W29 local artifact/resource helpers, reusing existing atomic writer/locks."""
import sys,json,resource,time,math
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_source_snapshot import _atomic_bytes
from V6.experimental.v7_e1_checkpoint import atomic_json as mutable_json
from V6.experimental.v7_research_artifacts import _safe,_lock
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7';ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window29-expanded-development');RESTORE=ROOT.parent/'window29-restore'
def enc(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def size():return sum(p.stat().st_size for r in (ROOT,RESTORE) if r.exists() for p in r.rglob('*') if p.is_file())
def capacity(extra=0):
 if size()+extra>1073741824-16777216:raise ValueError('W29_STORAGE_LIMIT_STOP_NO_GC')
def put(path,value):
 b=enc(value);capacity(len(b));path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=b:raise ValueError('IMMUTABLE_W29_OUTPUT_CONFLICT:'+str(path))
 else:_atomic_bytes(_safe(path),b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def checked(ref):
 p=Path(ref['path']);assert p.is_file() and p.stat().st_size==ref.get('bytes',p.stat().st_size) and file_sha256(p)==ref['sha256'],'NOT_RESTORABLE_OR_HASH_MISMATCH'
 return json.loads(p.read_text())
@contextmanager
def cpu_stage(stage):
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'CPU_BUDGET.json'
 with _lock(ROOT/'cpu-budget.lock'):
  ledger=json.loads(p.read_text()) if p.exists() else {'limit':3600,'verification_reserve':60,'stages':[]}
  if any(r['status']=='RUNNING' for r in ledger['stages']):raise ValueError('PRIOR_CPU_STAGE_INCOMPLETE_NO_UNBOUNDED_RETRY')
  used=sum(r['cpu_seconds'] for r in ledger['stages']);remaining=3540-used
  if remaining<10:raise ValueError('CPU_TOTAL_BUDGET_EXHAUSTED')
  cap=min(600,int(remaining));r={'stage':stage,'status':'RUNNING','reserved_seconds':cap,'cpu_seconds':cap};ledger['stages'].append(r);mutable_json(p,ledger)
 resource.setrlimit(resource.RLIMIT_CPU,(cap,cap));start=time.monotonic()
 try:yield
 finally:
  usage=resource.getrusage(resource.RUSAGE_SELF);r.update(status='TERMINAL',cpu_seconds=usage.ru_utime+usage.ru_stime,wall_seconds=time.monotonic()-start)
  with _lock(ROOT/'cpu-budget.lock'):mutable_json(p,ledger)
