"""Persist offline evidence using W19 read-only bytes + explicitly synthetic recovery cases."""
import json,runpy
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;C=runpy.run_path(str(O/'window35-common.py'));R=C['ROOT'];put=C['put'];sha=C['sha'];H=C['H']
from V6.experimental import v7_data_recovery as M
from V6.experimental.v7_manual_collector import archive,verified_record,restore,source
from V6.experimental.v7_source_snapshot import _atomic_bytes
closure=json.loads((O/'window35-closure.json').read_text());root=Path('/home/frank/projects/MarketMamba/.artifacts/window19-daily-pilot')
policy={'status':'OFFLINE_REFERENCE_NOT_ACTIVATED','content_limit':1,'network_retry_limit':2,'physical_limit':4}
with C['budget']('offline-recovery-acceptance'):
 ledger=M.AttemptLedger(R/'synthetic-attempt-ledger',policy);binding=source('11549');scope={'market':'TWSE','source':binding,'session':'2030-01-02'}
 row={'market':'TWSE','session':'2030-01-02','security_id':'SYNTH_ONLY','basic_valid':False,'ohlc_valid':False,'missing_fields':['close']};classified=M.classify(row,{('TWSE','SYNTH_ONLY')});decision=M.decide(classified,scope=scope,update_evidence='SYNTHETIC_FIXTURE_EXPECTS_PROVIDER_REVISION',source_permitted=True,quota_available=True,ledger=ledger);ref=ledger.reserve(scope,'CONTENT_RECOVERY','synthetic-recovery-one',decision=decision)
 body=(','.join(M.CSV_HEADERS['11549'])+'\n1190102,SYNTH_ONLY,synthetic,1000,10000,10,11,9,--,0,1\n').encode();first=archive(R/'synthetic-snapshots','11549',body,retrieved_at='2030-01-02T19:00:00+08:00',http_status=200,headers={'content-type':'text/csv'},mode='OFFLINE_REPLAY');second=archive(R/'synthetic-snapshots','11549',body.replace(b'9,--',b'9,10'),retrieved_at='2030-01-03T07:00:00+08:00',http_status=200,headers={'content-type':'text/csv'},mode='OFFLINE_REPLAY')
 before={'market':'TWSE','source':binding,'session':'2030-01-02','raw_sha256':first['raw_sha256'],'missing_required':['SYNTH_ONLY'],'qc_fatal':False};after={**before,'raw_sha256':second['raw_sha256'],'missing_required':[]};out=M.recovery_outcome(before,after);ledger.finish(ref,out)
 denied=M.decide(classified,scope=scope,update_evidence='synthetic',source_permitted=True,quota_available=True,ledger=ledger);assert denied['state']=='UNRECOVERABLE_AFTER_POLICY_LIMIT'
 real_replays=[]
 for r in closure['capture_refs']:
  original=verified_record(root/'data',str(Path(r['record']['path']).relative_to(root/'data')),r['record']['sha256']);body=Path(r['raw']['path']).read_bytes();new=archive(R/'w19-readonly-replay',''+r['dataset_id'],body,retrieved_at=r['retrieved_at'],http_status=200,headers=original['headers'],mode='OFFLINE_REPLAY');assert new['raw_sha256']==r['raw']['sha256'] and not new['verified_forward_vintage'];same=archive(R/'w19-readonly-replay',r['dataset_id'],body,retrieved_at=r['retrieved_at'],http_status=200,headers=original['headers'],mode='OFFLINE_REPLAY');assert new==same;real_replays.append({'session':r['session'],'dataset_id':r['dataset_id'],'original_raw_ref':r['raw'],'replay_raw_sha256':new['raw_sha256'],'replay_record_sha256':H(new),'classification':new['capture_classification'],'same_content_reuses_identity':True})
 # Independent restore of actual source bytes into W35's own namespace only.
 for r in closure['capture_refs']:
  p=Path(r['record']['path']);rel=str(p.relative_to(root/'data'));result=restore(root/'data',rel,r['record']['sha256'],C['RESTORE']/'w19-source-copy');assert result['status']=='RESTORED_BYTES_HASH_ROWS_VERIFIED'
 files=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.name=='CPU_BUDGET.json' or p.suffix=='.lock':continue
  b=p.read_bytes();C['capacity'](len(b));q=C['RESTORE']/'offline'/p.relative_to(R);_atomic_bytes(q,b);assert sha(q)==sha(p);files.append({'path':str(p),'restore':str(q),'bytes':len(b),'sha256':sha(p)})
 for old in json.loads((O/'window35-w19-inventory-before.json').read_text()):assert Path(old['path']).stat().st_size==old['bytes'] and sha(Path(old['path']))==old['sha256'],'W19_ASSET_CHANGED'
 proof={'status':'OFFLINE_RECOVERY_REPLAY_AND_W19_BYTES_RESTORE_PASS','actual_GET_Fit_GPU_ranking_scheduler_changes':0,'real_W19_readonly_replays':real_replays,'synthetic_policy':policy,'synthetic_claim':ref,'synthetic_repair':out,'after_limit_decision':denied,'synthetic_clock_note':'2030fixture timestamps are not real captures; actual archive timestamps separate','files':files,'file_count':len(files),'bytes_per_copy':sum(f['bytes'] for f in files),'W19_original_inventory_unchanged':True,'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP'};put(O/'window35-offline-evidence.json',proof)
resource={'observed_at':datetime.now(timezone.utc).isoformat(),'cpu':json.loads((R/'CPU_BUDGET.json').read_text()),'artifact_plus_restore_bytes':C['size'](),'real_requests':0,'new_fits':0,'W19_policy_budget_task_unchanged':True};put(O/'window35-resource-evidence.json',resource)
print(json.dumps({'status':proof['status'],'W19_replays':len(real_replays),'synthetic_repair':out['status'],'after_limit':denied['state'],'files':len(files),'bytes_per_copy':proof['bytes_per_copy'],'real_requests':0}))
