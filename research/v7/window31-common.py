"""W31 bounded local artifacts/resources using existing atomic writers and locks."""
import sys,json,resource,time
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H,file_sha256 as sha
from V6.experimental.v7_source_snapshot import _atomic_bytes
from V6.experimental.v7_e1_checkpoint import atomic_json
from V6.experimental.v7_research_artifacts import _safe,_lock
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window31-expanding');RESTORE=ROOT.parent/'window31-restore';REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7'
def enc(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def size():return sum(p.stat().st_size for r in (ROOT,RESTORE) if r.exists() for p in r.rglob('*') if p.is_file())
def capacity(n=0):
 if size()+n>536870912-16777216:raise ValueError('ARTIFACT512MIB_STOP_NO_GC')
def put(p,x):
 b=enc(x);capacity(len(b));p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('IMMUTABLE_W31_CONFLICT:'+str(p))
 else:_atomic_bytes(_safe(p),b)
 return {'path':str(p),'bytes':len(b),'sha256':sha(p)}
def checked(r):
 p=Path(r['path']);assert p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r.get('bytes',p.stat().st_size)
 return json.loads(p.read_text())
@contextmanager
def budget(stage):
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'CPU_BUDGET.json'
 with _lock(ROOT/'cpu.lock'):
  x=json.loads(p.read_text()) if p.exists() else {'limit':1800,'verification_reserve':60,'stages':[]}
  if any(r['status']=='RUNNING' for r in x['stages']):raise ValueError('PRIOR_STAGE_INCOMPLETE')
  rem=1740-sum(r['cpu_seconds'] for r in x['stages'])
  if rem<10:raise ValueError('CPU1800_TOTAL_BUDGET_STOP')
  cap=min(300,int(rem));r={'stage':stage,'status':'RUNNING','cpu_seconds':cap,'reserved':cap};x['stages'].append(r);atomic_json(p,x)
 resource.setrlimit(resource.RLIMIT_CPU,(cap,cap));t=time.monotonic()
 try:yield
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);r.update(status='TERMINAL',cpu_seconds=u.ru_utime+u.ru_stime,wall_seconds=time.monotonic()-t)
  with _lock(ROOT/'cpu.lock'):atomic_json(p,x)
