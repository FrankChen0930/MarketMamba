"""Persist tiny synthetic acceptance bytes; no real inference, Fit or metric."""
import json,runpy,shutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
T=runpy.run_path(str(OUT/'window32-shadow-tests.py'));M=T['M']
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window32-shadow-preparation')
RESTORE=ROOT.parent/'window32-shadow-restore'
x=T['ShadowAcceptance']();x.setUp()
try:
 fixture=ROOT/'fixture';fixture.mkdir(parents=True,exist_ok=True)
 for src in x.root.glob('*.json'):
  target=fixture/src.name;M['_atomic_bytes'](target,src.read_bytes())
 for ref in x.ctx['raw_refs']+x.models:ref['path']=str(fixture/Path(ref['path']).name)
 x.root=ROOT
 pre=x.publish();payload=M['read_prediction'](pre)['payload']
 label={'stock_id':'SYNTH_A','signal_session':x.ctx['signal_session'],'horizon':5,'label_policy_id':x.ctx['label_policy_id'],'prediction_sha256':pre['sha256'],'exit_at':payload['planned_exits']['5'],'available_at':'2030-01-07T14:00:00+08:00','eligible':True,'value':.1}
 pending=M['evaluation_gate'](pre,label,now=x.pub);mature=M['evaluation_gate'](pre,label,now='2030-01-07T14:01:00+08:00')
 x.ctx['study_id']='W32_SYNTHETIC_LATE';late=x.publish(published_at='2030-01-03T09:01:00+08:00')
 files=[]
 for src in sorted(ROOT.rglob('*.json')):
  dest=RESTORE/src.relative_to(ROOT);M['_atomic_bytes'](dest,src.read_bytes());assert M['sha'](dest)==M['sha'](src)
  files.append({'path':str(src),'restore':str(dest),'bytes':src.stat().st_size,'sha256':M['sha'](src)})
 proof={'status':'SYNTHETIC_OFFLINE_ACCEPTANCE_ONLY','real_inference_API_fit_RankIC':0,'preopen':pre,'late':late,'pending_label':pending,'mature_label':mature,'files':files,'bytes_per_copy':sum(f['bytes'] for f in files),'backup':'BACKUP_PENDING_SAME_HOST_SYNTHETIC_ONLY','actual_archive_observed_at':datetime.now(timezone.utc).isoformat(),'clock_note':'2030 fixtures are simulated clocks, not real observations or verified market sessions'}
 M['_atomic_bytes'](OUT/'window32-offline-evidence.json',(json.dumps(proof,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
 print(json.dumps({'status':proof['status'],'files':len(files),'bytes_per_copy':proof['bytes_per_copy'],'pending':pending['status'],'mature':mature['status'],'real_inference':0}))
finally:x.doCleanups()
