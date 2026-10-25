"""Same-host restore proof for W28 outputs; no evaluation or model fitting."""
from pathlib import Path
import json,hashlib,runpy
M=runpy.run_path(str(Path(__file__).with_name('window28-study.py')));root=M['ROOT'];target=root.parent/'window28-restore';files=[]
for p in sorted(root.rglob('*')):
 if not p.is_file():continue
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();q=target/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==b
 else:
  with q.open('xb') as f:f.write(b)
 assert q.stat().st_size==len(b) and hashlib.sha256(q.read_bytes()).hexdigest()==sha
 files.append({'path':str(p),'restore':str(q),'bytes':len(b),'sha256':sha})
assert len(list(root.glob('FIT_*_CLAIM.json')))==2
closed=json.loads((root/'COMPLETE.json').read_text());assert closed['fit_count']==2
for ref in closed['artifacts'].values():M['S']['checked'](ref)
prior=M['S']['input_pins']()
for r in json.loads((M['OUT']/'window27-local-artifacts.json').read_text())['files']:assert M['file_sha256'](Path(r['path']))==r['sha256']
result={'status':'REAL_BYTES_RESTORED','source':str(root),'restore':str(target),'files':files,'file_count':len(files),'total_bytes':sum(r['bytes'] for r in files),'fit_claims':2,'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','prior_pins':prior,'W27_diagnostic_files':8}
M['put'](M['OUT']/'window28-local-artifacts.json',result);print(json.dumps({k:result[k] for k in ('status','file_count','total_bytes','fit_claims','backup')}))
