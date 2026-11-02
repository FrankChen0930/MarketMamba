"""Read-only W19 closure from complete CSVs; zero GET, Task mutation or model score."""
import json,runpy,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;C=runpy.run_path(str(O/'window35-common.py'));R=C['ROOT'];put=C['put'];sha=C['sha'];H=C['H'];W19=Path('/home/frank/projects/MarketMamba/.artifacts/window19-daily-pilot')
from V6.experimental.v7_manual_collector import verified_record
from V6.experimental.v7_data_recovery import parse_csv,classify
from V6.experimental.v7_research_temporal import instant
P=runpy.run_path(str(O/'window29-panel.py'))
def main():
 inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(W19.rglob('*')) if p.is_file()];put(O/'window35-w19-inventory-before.json',inventory)
 policy=json.loads((W19/'policy.json').read_text());auth=json.loads((W19/'AUTHORIZED.json').read_text());assert H(policy)==auth['policy_sha256'] and policy['slots']==['2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02'];assert policy['expires_at']=='2026-11-02T23:59:59+08:00'
 w34=json.load(open('/tmp/w35-w34-prepared.json'));co=json.loads((O/'window33-cohort.json').read_text());ids=w34['offline_feature_security_ids'];assert sorted(ids)==co['securities'] and len(ids)==64;members={r['security_id']:r for r in co['cohort']};universe={('TWSE',s) for s in ids}
 rows=[];daily=[];refs=[];price_by_sec=defaultdict(dict);totals=Counter();exceptions=Counter();invocations=[]
 for f in sorted((W19/'invocations').glob('*finish.json')):
  x=json.loads(f.read_text());invocations.append({'path':str(f),'sha256':sha(f),'started_at':x['started_at'],'finished_at':x['finished_at'],'result':x.get('result')})
 for day in policy['slots']:
  run=json.loads((W19/'runs'/(day+'.json')).read_text());assert run['status']=='COMPLETE_NEW_SESSION' and run['GETs_dispatched_by_this_invocation']==2
  for ds,v in run['sources'].items():
   capture=verified_record(W19/'data',v['record_path'],v['record_sha256']);restored=verified_record(W19/'restore',v['record_path'],v['record_sha256']);assert capture==restored
   raw=W19/'data'/capture['raw_path'];parsed,qc=parse_csv(raw.read_bytes(),ds);assert qc==capture['qc'] and qc['sessions']==[day] and not qc['fatal_errors'];counts=Counter(r['reason'] for r in parsed);assert counts['PASS']==qc['counts'].get('PASS',0);assert counts['MISSING_OHLCV_NO_FILL']==qc['counts'].get('MISSING_OHLCV_NO_FILL',0)
   for row in parsed:
    verdict=classify(row,universe);row['classification']=verdict;rows.append(row)
    if row['market']=='TWSE' and row['security_id'] in ids:
     x=row['values'];price_by_sec[row['security_id']][day]={'open':x['open'],'max':x['high'],'min':x['low'],'close':x['close'],'Trading_Volume':x['volume']}
   daily.append({'session':day,'market':capture['source']['provider'],'dataset_id':ds,'rows':len(parsed),'basic_pass':counts['PASS'],'missing':counts['MISSING_OHLCV_NO_FILL'],'invalid':counts['INVALID_OHLCV'],'exception_list_n':len(qc['exceptions']),'retrieved_at':capture['retrieved_at_utc'],'calendar':v['calendar'],'raw_capture_status':capture['status'],'capture_classification':capture['capture_classification']});totals.update({'rows':len(parsed),'PASS':counts['PASS'],'missing':counts['MISSING_OHLCV_NO_FILL'],'fatal':len(qc['fatal_errors'])});exceptions[ds]+=len(qc['exceptions'])
   refs.append({'session':day,'dataset_id':ds,'source':capture['source'],'raw':{'path':str(raw),'sha256':capture['raw_sha256'],'bytes':capture['raw_bytes']},'record':{'path':str(W19/'data'/v['record_path']),'sha256':v['record_sha256']},'manifest':{'path':str(W19/'data'/capture['manifest_path']),'sha256':capture['manifest_sha256']},'retrieved_at':capture['retrieved_at_utc']})
 requests=[]
 for day in policy['slots']:
  for ds in policy['sources']:
   claim=W19/'requests'/day/(ds+'.json');out=W19/'requests'/day/(ds+'-result.json');assert claim.is_file() and out.is_file();requests.append({'claim':str(claim),'sha256':sha(claim),'result_sha256':sha(out)})
 assert len(requests)==10 and dict(totals)=={'rows':12031,'PASS':11814,'missing':217,'fatal':0},totals
 missing=[r for r in rows if not r['basic_valid']];bysec=defaultdict(list)
 for row in missing:bysec[(row['market'],row['security_id'])].append(row)
 distribution=[{'market':m,'security_id':s,'name':rs[0]['name'],'missing_days':len({r['session'] for r in rs}),'sessions':sorted({r['session'] for r in rs}),'fields':dict(Counter(f for r in rs for f in r['missing_fields'])),'volume_values':sorted({str(r['values']['volume']) for r in rs}),'state':rs[0]['classification']['state'],'instrument_hint':rs[0]['instrument_hint'],'cause':'UNKNOWN_NO_TRADE_SUSPENSION_OR_PROVIDER_ISSUE_NOT_INFERRED'} for (m,s),rs in sorted(bysec.items())]
 # Prior close context for the first pilot day comes from immutable W16 Oct22,
 # not future labels or any W19 recovery request.
 oldrecord=next(r for r in json.loads((O/'window16-resource-coverage.json').read_text())['records'] if r['dataset_id']=='11549');assert sha(Path(oldrecord['raw_path']))==oldrecord['raw_sha256'];priorrows,_=parse_csv(Path(oldrecord['raw_path']).read_bytes(),'11549')
 for row in priorrows:
  if row['security_id'] in ids:
   v=row['values'];price_by_sec[row['security_id']]['2026-10-22']={'open':v['open'],'max':v['high'],'min':v['low'],'close':v['close'],'Trading_Volume':v['volume']}
 days=['2026-10-22']+policy['slots'];impact=[];impact_details=[]
 for day in policy['slots']:
  current=[r for r in rows if r['market']=='TWSE' and r['session']==day and r['security_id'] in ids];valid=[];blocked=[]
  for s in ids:
   f=P['feature_at'](s,day,days,price_by_sec[s],members[s],[])
   if f['eligible']:valid.append(s)
   else:blocked.append({'security_id':s,'reasons':f['reasons']})
  impact.append({'session':day,'cohort_n':64,'present':len(current),'basic_ohlcv_valid':sum(r['basic_valid'] for r in current),'ohlc_valid':sum(r['ohlc_valid'] for r in current),'feature_input_eligible':len(valid),'blocked':blocked,'calendar_scope':'observed pilot snapshots/slots, not formal data or inference admission'});impact_details.extend({'session':day,**b} for b in blocked)
 now=datetime.now(timezone.utc).isoformat();state='EXPIRED_PROTECTED_EVIDENCE' if instant(now)>instant(policy['expires_at']) else 'EXPIRING_PROTECTED_EVIDENCE'
 task=json.load(open('/tmp/w35-task-observation.json',encoding='utf-8-sig'));assert task['registered'] and task['triggers'][0]['end']==policy['expires_at'];assert task['last_result']==0
 w34ref={'source_commit':'82cd49c25e585d857c05e9835af6c77f6ac9e860','prepared_sha256':sha(Path('/tmp/w35-w34-prepared.json')),'security_ids':ids,'market':'TWSE','feature_pipeline_sha256':sha(O/'window29-panel.py'),'model_loaded_or_scored':False};put(O/'window35-cohort-reference.json',w34ref)
 put(R/'complete-raw-row-diagnostics.json',rows);put(R/'complete-missing-distribution.json',distribution);put(O/'window35-missing-distribution.json',distribution)
 summary={'status':'FIVE_SESSION_EVIDENCE_VERIFIED_NO_NEW_GET','observed_at':now,'pilot_expiry_status':state,'expiry_at':policy['expires_at'],'policy_sha256':H(policy),'sessions':5,'sources':10,'requests':len(requests),'totals':dict(totals),'daily':daily,'missing_by_market':dict(Counter(r['market'] for r in missing)),'exceptions_recorded_by_dataset':dict(exceptions),'missing_security_count':len(distribution),'missing_day_frequency':dict(Counter(r['missing_days'] for r in distribution)),'classification_counts':dict(Counter(r['classification']['state'] for r in missing)),'missing_field_counts':dict(Counter(f for r in missing for f in r['missing_fields'])),'cohort_impact':impact,'cohort_blocked_details':impact_details,'invocations':invocations,'capture_refs':refs,'source_level_failures':0,'known_legitimate_no_price':0,'known_suspensions':0,'unknown_cause_missing':217,'backup':'BACKUP_PENDING','task_observation':task,'runtime_port':'NOT_NEEDED_existing_main_verified_record_QC_snapshot_restore_accept_W19_records; no old runtime merged','prior_context':{'path':oldrecord['raw_path'],'sha256':oldrecord['raw_sha256'],'session':'2026-10-22'},'strict_or_research_admission':False};put(O/'window35-closure.json',summary);print(json.dumps({k:summary[k] for k in ['status','pilot_expiry_status','totals','missing_by_market','missing_security_count','missing_day_frequency','classification_counts','cohort_impact']}))
if __name__=='__main__':
 with C['budget']('w19-readonly-closure'):main()
