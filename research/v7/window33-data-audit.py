"""Bounded recent raw/CSV/maturity diagnostics. No fitting, inference or RankIC.
Recent market-session union is OBSERVED only, never qualified by rewriting a flag.
"""
import json,runpy,csv,io
from pathlib import Path
from collections import Counter
O=Path(__file__).resolve().parent;C=runpy.run_path(str(O/'window33-common.py'));ROOT=C['ROOT'];put=C['put'];checked=C['checked'];sha=C['sha'];H=C['H']
from V6.experimental.v7_research_temporal import _calendar
W=runpy.run_path(str(O/'window29-study.py'));P=runpy.run_path(str(O/'window29-panel.py'));F=runpy.run_path(str(O/'window31-folds.py'));A=runpy.run_path(str(O/'window33-compatibility.py'))

def main():
 co=json.loads((O/'window33-cohort.json').read_text());net=json.loads((ROOT/'finmind/CLOSED.json').read_text());assert not net['source_stopped'] and net['captured']==len(co['securities'])
 records=[json.loads(p.read_text()) for p in sorted((ROOT/'finmind/records').glob('*.json'))];new={};refs=[]
 for r in records:
  assert r['params']['start_date']=='2024-01-01' and r['params']['end_date']=='2026-10-22' and sha(Path(r['raw_path']))==r['raw_sha256'] and W['verify_snapshot'](Path(r['manifest']))
  rows=json.loads(Path(r['raw_path']).read_text())['data'];sec=r['params']['data_id'];assert all(x['stock_id']==sec and '2024-01-01'<=x['date']<='2026-10-22' for x in rows);new[sec]={x['date']:x for x in rows};assert len(new[sec])==len(rows);refs.append({'path':r['raw_path'],'sha256':r['raw_sha256'],'bytes':r['raw_bytes'],'retrieved_at':r['retrieved_at']})
 assert sorted(new)==co['securities'];oldco,old,oldrefs,olddays,calref=W['sources']();days=sorted({d for rows in new.values() for d in rows});all_days=sorted(set(olddays)|set(days));joined={s:{**old[s],**new[s]} for s in co['securities']};members={r['security_id']:r for r in co['cohort']}
 official_record=next(r for r in json.loads((O/'window16-resource-coverage.json').read_text())['records'] if r['dataset_id']=='11549');official_path=Path(official_record['raw_path']);assert sha(official_path)==official_record['raw_sha256'];body=official_path.read_bytes();official=[A['official_row'](r) for r in csv.DictReader(io.StringIO(body.decode('utf-8-sig')))];official_index={(r['stock_id'],r['date']):r for r in official};assert len(official_index)==len(official)
 w18pins=json.loads((O/'window18-local-artifacts.json').read_text())['files'];same=[f for f in w18pins if f['sha256']==sha(official_path)];assert same
 for f in same:assert sha(Path(f['path']))==f['sha256']
 comparison=[];field_counts=Counter();hybrid_count=0
 for sec in co['securities']:
  key=(sec,'2026-10-22');a=official_index.get(key);b=new[sec].get(key[1]);row={'security':sec,'session':key[1],'official_present':a is not None,'finmind_present':b is not None,'matching_fields':[],'mismatches':{}}
  if a and b:
   for field in ['open','max','min','close','Trading_Volume']:
    if a[field]==b[field]:row['matching_fields'].append(field);field_counts[field]+=1
    else:row['mismatches'][field]={'official':a[field],'finmind':b[field]}
   ref=P['feature_at'](sec,key[1],all_days,joined[sec],members[sec],[]);hybrid={**joined[sec],key[1]:a};alt=P['feature_at'](sec,key[1],all_days,hybrid,members[sec],[])
   row['same_formula_hybrid_prior']=ref['eligible'] and alt['eligible'] and ref['features']==alt['features'];hybrid_count+=bool(row['same_formula_hybrid_prior'])
  comparison.append(row)
 features=[];masks=[];per_security=[];reasons=Counter();numeric=Counter();event_flags={};stock_sets=Counter()
 for sec in co['securities']:
  prices=joined[sec];events=P['suspect_events'](prices);event_flags[sec]=events;coverage=Counter(x[:4] for x in new[sec]);valid=sum(P['valid'](p) for p in new[sec].values());zero=sum(any(p.get(k)==0 for k in ['open','max','min','close']) for p in new[sec].values());missing=sorted(set(days)-set(new[sec]));feature_n=0
  for day in days:
   f=P['feature_at'](sec,day,all_days,prices,members[sec],events)
   if not f['eligible']:reasons.update(f['reasons']);continue
   features.append(f);feature_n+=1;stock_sets[day]+=1
   for h in [5,10]:
    q=F['numeric_label'](f,h,all_days,prices,events);numeric[str(h)+'_candidate_ok']+=bool(q['eligible']);numeric[str(h)+'_candidate_rejected']+=not q['eligible']
    masks.append({'key':q['key'],'entry':q['entry'],'exit':q['exit'],'numeric_candidate':q['eligible'],'candidate_mature_by_data_end':bool(q['eligible'] and q['exit']<='2026-10-22'),'eligible_for_fit':False,'reasons':q['reasons']+['RECENT_CALENDAR_QUALIFICATION_UNKNOWN'],'event_status':'UNKNOWN_UNLESS_SUSPECTED_DISCONTINUITY_FLAG','safety_policy':'W27/W29/W31 inherited path predicates, no ffill/zero imputation'})
  per_security.append({'security':sec,'rows':len(new[sec]),'by_year':dict(coverage),'earliest':min(new[sec],default=None),'latest':max(new[sec],default=None),'valid_ohlc':valid,'zero_ohlc':zero,'missing_observed_sessions':len(missing),'missing_sessions':missing,'feature_candidates':feature_n,'membership':'W29 conditional TWSE identity; current Oct22 official observation does not prove full lifecycle','suspension':'UNKNOWN_not_inferred_from_missing','suspected_discontinuities':events})
 _,_,calendar_reason=_calendar({'source_sha256':H(days),'vintage_id':'W33_CURRENT_RETROSPECTIVE','qualification_ref':'OBSERVED prices only; official complete closure/session evidence missing','qualification':'OBSERVED_MARKET_SESSION','sessions':[{'session':d} for d in days]});assert calendar_reason=='CALENDAR_QUALIFICATION_UNKNOWN'
 feature_ref=put(ROOT/'derived/recent-feature-candidates.json',features);mask_ref=put(ROOT/'derived/label-candidate-masks.json',masks);panel_ref=put(ROOT/'derived/recent-raw-panel.json',new)
 put(ROOT/'derived/cross-source-comparison.json',comparison);put(ROOT/'derived/security-coverage.json',per_security)
 source_diff=any(r['mismatches'] for r in comparison);compat={'status':'MISMATCH_BLOCKED' if source_diff else 'BOUNDED_SAME_DAY_PARITY_NOT_TWO_OFFICIAL_SESSIONS','source_W16_raw':{'path':str(official_path),'sha256':sha(official_path)},'W18_same_bytes_refs':same,'official_sessions':['2026-10-22'],'comparison_rows':len(comparison),'matching_field_counts':dict(field_counts),'hybrid_feature_exact':hybrid_count,'hybrid_limitation':'same formula using official current row plus explicitly FinMind prior row, NOT independent two-session official feature parity','official_previous_session_snapshot':'NOT_PRESENT_IN_W16_W18; do not infer prior close from price-change field','feature_order':['raw_close_momentum1','raw_intraday_range'],'volume_unit':'SHARES field values compared without conversion','adjustment':'RAW source prices; neither total nor executable return; current historical versions not original asof','mismatched_rows':[r for r in comparison if r['mismatches']]}
 put(O/'window33-compatibility.json',compat)
 result={'status':'RECENT_DATA_PRESERVED_FIT_BLOCKED','cohort_n':len(co['securities']),'raw_rows':sum(len(v) for v in new.values()),'observed_sessions':len(days),'first':min(days),'last':max(days),'observed_sessions_by_year':dict(Counter(d[:4] for d in days)),'calendar_qualification':'UNKNOWN_OBSERVED_MARKET_SESSION_UNION_NOT_QUALIFIED_REFERENCE','W07_machine_rejection':calendar_reason,'calendar_missingness':'Only union-relative gaps diagnosed; synchronized provider omissions vs actual market closures unresolved','official_calendar_scope':'W23/W29 qualified2021–2023; W17 partial2026-10-22–27 only','numeric_label_candidates':dict(numeric),'actual_training_n':0,'fits_Ridge_LightGBM':0,'feature_candidates':len(features),'daily_feature_N_distribution':dict(Counter(stock_sets[d] for d in days)),'feature_rejection_reason_flags_non_disjoint':dict(reasons),'zero_ohlc_rows':sum(r['zero_ohlc'] for r in per_security),'missing_union_relative_records':sum(r['missing_observed_sessions'] for r in per_security),'suspected_unit_events':event_flags,'event_classification':'Suspected threshold inherited>=2 or<=.5 across last valid closes; not confirmed corporate actions; no new event sources','CA_original_vintage_status':'UNKNOWN retained under Development-only, not sole reason to refuse data','dataset_id':H({'cohort':H(co),'raw':refs,'period':co['period']}),'refs':{'features':feature_ref,'mask':mask_ref,'panel':panel_ref,'new_raw':refs,'old_raw':oldrefs,'old_calendar':calref},'gates':{'recent_calendar':'BLOCKED_UNKNOWN','source_csv':compat['status'],'model_fit':'BLOCKED','real_offline_scores':'NOT_RUN_DEPENDENCY_BLOCKED'},'exposure':'2024–2026 previously exposed, not independent test','authority':'SHADOW_PREPARATION_ONLY_NOT_FORMAL_ADMISSION','W19_access':0,'API_metadata':0};put(O/'window33-data-evidence.json',result)
 print(json.dumps({k:result[k] for k in ['status','cohort_n','raw_rows','observed_sessions','first','last','numeric_label_candidates','feature_candidates','zero_ohlc_rows','missing_union_relative_records','gates']}))
if __name__=='__main__':
 with C['budget']('recent-data-audit'):main()
