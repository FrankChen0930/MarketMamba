"""W30 immutable output restore proof; zero Fit/API and no old evidence changes."""
from pathlib import Path
import json,runpy
from datetime import datetime,timezone
M=runpy.run_path(str(Path(__file__).with_name('window30-lgbm.py')));root=M['ROOT'];out=M['OUT'];sha=M['sha'];put=M['put']
with M['budget']('archive'):
 closed=json.loads((root/'COMPLETE.json').read_text());assert closed['LightGBM_fits']==2 and len(list(root.glob('FIT_*_CLAIM.json')))==2
 for r in closed['artifacts'].values():M['checked'](r)
 files=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.name=='CPU_BUDGET.json' or p.suffix=='.lock':continue
  b=p.read_bytes();M['capacity'](len(b));q=M['RESTORE']/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert q.read_bytes()==b
  else:
   with q.open('xb') as f:f.write(b)
  assert p.stat().st_size==q.stat().st_size and sha(p)==sha(q)
  files.append({'path':str(p),'restore':str(q),'bytes':len(b),'sha256':sha(p)})
 original=json.loads((out/'window29-local-artifacts.json').read_text())['files']
 for f in original:assert Path(f['path']).stat().st_size==f['bytes'] and sha(Path(f['path']))==f['sha256']
 proof={'status':'INDEPENDENT_LOCAL_BYTES_RESTORED','files':files,'file_count':len(files),'bytes':sum(f['bytes'] for f in files),'W29_original_files_unchanged':len(original),'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','mutable_CPU_counter':'excluded; separate final resource observation'};put(out/'window30-local-artifacts.json',proof)
resource={'observed_at':datetime.now(timezone.utc).isoformat(),'cpu':json.loads((root/'CPU_BUDGET.json').read_text()),'artifact_plus_restore_bytes':M['size'](),'artifact_limit_bytes':268435456,'CPU_limit_seconds':1200,'verification_reserve_seconds':60,'scope':'OS CPU/stage limits and shell timeout for study, verification reserve; not global control of unrelated processes'};put(out/'window30-resource-evidence.json',resource)
print(json.dumps({k:proof[k] for k in ['status','file_count','bytes','W29_original_files_unchanged','backup']}))
