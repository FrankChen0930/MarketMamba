"""Bounded W26 event/label analysis only. No fit, prediction or RankIC code."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import file_sha256
from V6.experimental.v7_source_snapshot import verify_snapshot
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window26-label-integrity');OUT=Path(__file__).parent
W24=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot')
def save(path,obj):
 b=(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b,'NO_EVIDENCE_OVERWRITE'
 else:
  with path.open('xb') as f:f.write(b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def crosses(entry,exit,effective):return entry<effective<=exit
# Exact byte identity, no metric/prediction replay or loading of model payloads.
for name in ['window24-local-artifacts.json','window25-local-artifacts.json']:
 for f in json.load(open(OUT/name))['files']:
  p=Path(f['path']);assert p.is_file() and file_sha256(p)==f['sha256'] and p.stat().st_size==f['bytes']
rec=json.load(open(ROOT/'finmind/record.json'));assert rec['status']=='OBSERVED_EVENT_METADATA' and verify_snapshot(rec['manifest']) and file_sha256(Path(rec['raw_path']))==rec['raw_sha256']
events=json.load(open(rec['raw_path']))['data'];event=next(x for x in events if x['stock_id']=='2321' and x['date']=='2023-02-17')
assert event['ReasonforCapitalReduction']=='Making up losses'
run=json.load(open(W24/'PILOT_CLOSED.json'));bundle=Path(run['bundle_path']);data=json.load(open(bundle/'dataset_snapshot.json'))['value'];coverage=json.load(open(bundle/'data_split_eligibility.json'))['value'];sources=json.load(open(coverage['references']['sources']['path']));raw={}
for r in sources:
 p=Path(r['raw_path']);assert file_sha256(p)==r['raw_sha256'];raw[r['security']]={x['date']:x for x in json.load(open(p))['data']}
calendar=json.load(open(coverage['references']['calendar']['path']));days=[x['date'] for x in calendar if x['engineering_calendar_ok'] and x['price_observed']];index={d:i for i,d in enumerate(days)}
p=raw['2321'];gap=[d for d in days if '2023-02-09'<=d<='2023-02-16'];assert len(gap)==6 and not any(d in p for d in gap)
market_evidence={d:[sec for sec,q in raw.items() if d in q and q[d]['open']>0 and q[d]['close']>0] for d in gap};assert all('2330' in xs for xs in market_evidence.values())
assert p['2023-02-08']['close']==event['ClosingPriceonTheLastTradingDay']==6.38
assert event['PostReductionReferencePrice']==19.93 and p['2023-02-17']['open']==19.6
case_rows=[]
for r in data:
 if r['stock_id']=='2321' and crosses(r['entry'],r['exit'],event['date']):
  expected=p[r['exit']]['close']/p[r['entry']]['open']-1;assert expected==r['diagnostic_label']
  i=index[r['signal_session']];assert r['entry']==days[i+1] and r['exit']==days[i+r['horizon']]
  case_rows.append({'sample_key':[r['stock_id'],r['signal_session'],r['horizon']],'signal_index':i,'entry_index':i+1,'exit_index':i+r['horizon'],'entry':r['entry'],'exit':r['exit'],'entry_open':p[r['entry']]['open'],'exit_close':p[r['exit']]['close'],'stored_raw_label':r['diagnostic_label'],'formula_check':'PASS_EXACT_ORIGINAL','economic_label_status':'CORPORATE_ACTION_CROSSING_NOT_COMPARABLE_SAME_SHARE_UNIT_RETURN','model_role':r['split'],'affected_ref':'existingW24samplehashonly_nochange','source_sample_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()})
assert len(case_rows)==3 and all(x['sample_key'][2]==10 for x in case_rows)
private=save(ROOT/'2321-case-details.json',{'provider_event':event,'case_rows':case_rows,'missing_market_sessions':gap,'market_observations':market_evidence,'before_event_close':p['2023-02-08'],'after_event_first_quote':p['2023-02-17'],'original_raw_source':next(r for r in sources if r['security']=='2321')})
cohort={c['security_id']:c for c in json.load(open(OUT/'window24-cohort.json'))['cohort']};gap_events=[]
for sec,prices in raw.items():
 c=cohort[sec];start=c['reported_twse_start'];end=c['reported_end'];eligible=[d for d in days if start and start<=d and (not end or d<end)]
 missing=[d for d in eligible if d not in prices];groups=[]
 for d in missing:
  if not groups or index[d]!=index[groups[-1][-1]]+1:groups.append([d])
  else:groups[-1].append(d)
 for g in groups:gap_events.append({'security':sec,'first_missing':g[0],'last_missing':g[-1],'market_sessions':len(g),'dates':g,'cause':'CAPITAL_EVENT_ALIGNED_INDIVIDUAL_GAP_PRIMARY_SUSPENSION_NOTICE_UNKNOWN' if sec=='2321' and g==gap else 'UNKNOWN_SOURCE_OMISSION_SUSPENSION_OR_NONTRADING_NOT_INFERRED','market_closed':False,'event_scope':'reportedlistinginterval_notdailytradability'})
label_flags=Counter();affected=[]
for r in data:
 path=days[index[r['entry']]:index[r['exit']]+1];prices=raw[r['stock_id']];missing=[d for d in path if d not in prices];zero=[d for d in path if d in prices and all(prices[d][k]==0 for k in ['open','max','min','close'])]
 if missing:label_flags[r['split']+'_MISSING_PATH']+=1
 if zero:label_flags[r['split']+'_ZERO_PATH']+=1
 if r['stock_id']=='2321' and crosses(r['entry'],r['exit'],event['date']):label_flags[r['split']+'_CONFIRMED_CAPITAL_EVENT']+=1
 if missing or zero:affected.append({'stock':r['stock_id'],'signal':r['signal_session'],'horizon':r['horizon'],'role':r['split'],'missing':missing,'zero':zero,'cause':'UNQUALIFIED_INTERIOR_PATH_NOT_AUTOMATIC_ACTION'})
private_path=save(ROOT/'broader-label-flags.json',{'gap_events':gap_events,'flagged_samples':affected,'flags':dict(label_flags),'uncovered_action_history':'UNKNOWN_ALL_OTHER_EVENTS_NOT_INVESTIGATED'})
issuer_url='https://www.tecom.com.tw/wp-content/uploads/2025/01/2023%E5%B9%B4%E5%BA%A6-FS-AllReport%E5%90%AB%E9%99%84%E8%A8%BB%E9%99%84%E8%A1%A8-CHT-IFRS.pdf'
matrix={'schema':'w26-event-evidence-matrix-v1','status':'EVENT_INTEGRITY_FINDING_NOT_NEW_LABEL_ADMISSION','sources':{'FinMindCapital':{'record':str(ROOT/'finmind/record.json'),'raw_sha256':rec['raw_sha256'],'current_version_only':True},'issuer112financialreport':{'url':issuer_url,'raw_record':str(ROOT/'documents/tecom112-financial-report.record.json'),'page_zero_based':37,'report_approval':'2024-03-04','claim':'68percentlossoffsetcapitalreduction;shareface10;approved2022-12-06;accountingbase2022-12-08','stop_resume_dates_not_in_this_note':True}},'events':[
 {'id':'2321-share-capital-reduction','status':'CONFIRMED_ISSUER_AND_PROVIDER','type':'LOSS_OFFSET_CAPITAL_REDUCTION','ratio':.68,'share_retention_fraction':.32,'cash_or_totalreturn_ledger':'NOT_ESTABLISHED;no automaticcashassumption','accounting_base_date':'2022-12-08','accounting_date_is_not_market_price_unit_effective_date':True},
 {'id':'2321-market-reference-change','status':'PROVIDER_EVENT_VERIFIED_RAW_PRICE_CORROBORATED','effective_reference_date':'2023-02-17','last_old_close':6.38,'post_reduction_reference':19.93,'opening_reference':19.95,'actual_observed_open':19.6,'scale_ratio':19.93/6.38,'ideal_share_unit_factor':1/.32,'reference_is_not_executable_entry':True,'minus1_ExrightReferencePrice':'UNKNOWN_SENTINEL_NOT_A_CASH_FLOW'},
 {'id':'2321-six-session-gap','status':'CONFIRMED_INDIVIDUAL_RECORD_GAP_NOT_MARKET_CLOSURE','sessions':gap,'market_open_evidence':'existingTWSEcalendar andcohortpricesincluding2330eachdate','suspension_cause':'CORROBORATED_BY_RESUMPTION_REFERENCE_AND_SECONDARY_NOTICE;ORIGINAL_SUSPENSION_NOTICE_NOT_VERIFIED','secondary_notice':'2023-01-10issuerstatementcarriedbyYahoo/CNA,notusedasprimaryproof;20230209..16stop/20230217resume','no_global_suspension_inference':True},
 {'id':'2321-crossing-labels','status':'ARITHMETIC_CORRECT_BUT_INVALID_AS_UNADJUSTED_SAME_UNIT_HOLDING_RETURN','affected_samples':3,'horizon':10,'role':'FORWARD_DIAGNOSTIC2023','signal_sessions':[r['sample_key'][1] for r in case_rows],'stored_label_max':max(x['stored_raw_label'] for x in case_rows),'original_artifacts_or_metrics_modified':False,'metric_impact':'NOT_RECALCULATED_NOT_ESTIMATED'},
 {'id':'other-cohort-actions','status':'UNKNOWN_NOT_EXHAUSTIVELY_COVERED','new_queries_for_others':0,'candidate_types':'capitalreduction/split/parvalue/dividend/sourcegaps'}],
 'broader_flags':dict(label_flags),'individual_gap_event_count':len(gap_events),'private_details':private,'private_broader':private_path,'no_new_price_fit_prediction_rankic':True,'formal_admission':False}
save(OUT/'window26-event-matrix.json',matrix);print(json.dumps({'affected':len(case_rows),'missing':gap,'broader':dict(label_flags),'gap_events':len(gap_events)}))
