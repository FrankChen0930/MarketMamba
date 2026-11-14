"""Independent filesystem restore proof; never loads model payloads or computes metrics."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path('/home/frank/projects/MarketMamba')
root=BASE/'.artifacts/window26-label-integrity';target=BASE/'.artifacts/window26-restore'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[{'path':str(p),'relative':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]
assert files
for f in files:
 p=Path(f['path']);q=target/f['relative'];q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert sha(q)==f['sha256'] and q.stat().st_size==f['bytes']
 else:
  with q.open('xb') as out:out.write(p.read_bytes())
 assert sha(q)==f['sha256'] and q.stat().st_size==f['bytes']
rec=json.loads((target/'finmind/record.json').read_text());raw=target/Path(rec['raw_path']).relative_to(root)
assert sha(raw)==rec['raw_sha256']
data=json.loads(raw.read_text());assert len(data['data'])==1 and data['data'][0]['stock_id']=='2321'
prior={}
for name in ['window24','window25']:
 inv=json.loads((Path(__file__).parent/(name+'-local-artifacts.json')).read_text())['files']
 for f in inv:assert Path(f['path']).stat().st_size==f['bytes'] and sha(Path(f['path']))==f['sha256']
 prior[name]=len(inv)
result={'status':'BYTES_HASH_RESTORED','separate_process':True,'source':str(root),'restore':str(target),'files':files,'event_records':1,'prior_immutable_files':prior,'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','unavailable_annual_report':'NO_BYTES_NO_RESTORE_CLAIM','research_admission':False}
p=Path(__file__).with_name('window26-independent-restore.json');p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'files':len(files),'bytes':sum(f['bytes'] for f in files),'prior':prior,'event_records':1,'backup':result['backup']}))
