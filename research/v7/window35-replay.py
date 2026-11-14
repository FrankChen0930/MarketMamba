"""Independent offline replay of saved W19 classification and synthetic recovery evidence."""
import json,runpy,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;C=runpy.run_path(str(O/'window35-common.py'));R=C['ROOT'];sha=C['sha'];put=C['put'];H=C['H']
from V6.experimental import v7_data_recovery as M
with C['budget']('independent-recovery-replay'):
 closure=json.loads((O/'window35-closure.json').read_text());proof=json.loads((O/'window35-offline-evidence.json').read_text());co=json.loads((O/'window35-cohort-reference.json').read_text());universe={('TWSE',s) for s in co['security_ids']};totals=Counter();states=Counter();freq=Counter()
 for ref in closure['capture_refs']:
  assert sha(Path(ref['raw']['path']))==ref['raw']['sha256'];rows,qc=M.parse_csv(Path(ref['raw']['path']).read_bytes(),ref['dataset_id']);totals.update({'rows':len(rows),'PASS':sum(r['basic_valid'] for r in rows),'missing':sum(not r['basic_valid'] for r in rows),'fatal':len(qc['fatal_errors'])})
  for r in rows:
   if not r['basic_valid']:states[M.classify(r,universe)['state']]+=1;freq[(r['market'],r['security_id'])]+=1
 assert dict(totals)==closure['totals'] and dict(states)==closure['classification_counts'];assert dict(Counter(freq.values()))=={int(k):v for k,v in closure['missing_day_frequency'].items()}
 records=sorted([json.loads(f.read_text()) for f in (R/'synthetic-snapshots/captures/11549').glob('*.json')],key=lambda r:r['retrieved_at_utc']);assert len(records)==2
 snapshots=[]
 for r in records:
  rows,_=M.parse_csv((R/'synthetic-snapshots'/r['raw_path']).read_bytes(),'11549');snapshots.append({'market':'TWSE','source':r['source'],'session':r['qc']['sessions'][0],'raw_sha256':r['raw_sha256'],'missing_required':[v['security_id'] for v in rows if not v['ohlc_valid']],'qc_fatal':False})
 assert M.recovery_outcome(*snapshots)==proof['synthetic_repair'];claim=M.read_ref(proof['synthetic_claim']);ledger=M.AttemptLedger(R/'synthetic-attempt-ledger',proof['synthetic_policy']);denied=M.decide(proof['after_limit_decision']['classification'],scope=claim['scope'],update_evidence='synthetic',source_permitted=True,quota_available=True,ledger=ledger);assert H(denied)==H(proof['after_limit_decision'])
 for f in proof['files']:
  for k in ['path','restore']:assert Path(f[k]).stat().st_size==f['bytes'] and sha(Path(f[k]))==f['sha256']
 before=json.loads((O/'window35-w19-inventory-before.json').read_text())
 for f in before:assert Path(f['path']).stat().st_size==f['bytes'] and sha(Path(f['path']))==f['sha256']
 result={'status':'INDEPENDENT_OFFLINE_RECOVERY_REPLAY_PASS','rows':totals['rows'],'missing':totals['missing'],'missing_security_count':len(freq),'W19_original_files_unchanged':len(before),'offline_restore_files_verified':len(proof['files']),'synthetic_outcome_reproduced':True,'budget_stop_decision_reproduced':True,'actual_GET_Fit_GPU_ranking_task_changes':0,'actual_replayed_at':datetime.now(timezone.utc).isoformat()};result['recovery_code_sha256']=sha(C['REPO']/'V6/experimental/v7_data_recovery.py');put(O/('window35-replay-closure.json' if '--closure' in sys.argv else 'window35-replay.json'),result);print(json.dumps(result))
