"""Fixed read-only W24 robustness audit. No request, fit, inference or policy mutation.
Uses saved scores, W08 daily_metric, E1 circular-block helper and W24 identities.
"""
import sys,json,hashlib,runpy
from pathlib import Path
from collections import Counter,defaultdict
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_research_artifacts import verify_bundle,artifact_path
from V6.experimental.v7_robustness_audit import describe,circular_bounds,leave_one
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7';ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window25-robustness-audit');W24=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot')
PLAN=json.load(open(OUT/'window25-analysis-plan.json'))
def save(path,obj):
 b=(json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b,'IMMUTABLE_AUDIT_OUTPUT_CONFLICT'
 else:
  with path.open('xb') as f:f.write(b)
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(b)}
def pins():
 inv=json.load(open(OUT/'window24-local-artifacts.json'))
 for f in inv['files']:
  p=Path(f['path']);assert p.is_file() and file_sha256(p)==f['sha256'] and p.stat().st_size==f['bytes'],'W24_INPUT_CHANGED_OR_NOT_RESTORABLE'
 return len(inv['files'])
def main():
 before=pins();run=json.load(open(W24/'PILOT_CLOSED.json'));v=verify_bundle(W24/'bundles',run['bundle_key'],run['manifest_sha256']);assert v['status']=='ARTIFACTS_VERIFIED'
 for rel,pin in run['code_files'].items():assert file_sha256(REPO/rel)==pin
 descriptors={a['name']:a for a in v['manifest']['artifacts']};load=lambda name:json.load(open(artifact_path(v['bundle_path'],descriptors[name])))['value']
 config=load('resolved_config');contract=config['contract'];pred=load('predictions');daily=load('daily_metrics');data=load('dataset_snapshot');fits=load('models');coverage=load('data_split_eligibility');policy=contract['evaluation']
 w24=runpy.run_path(str(OUT/'window24-pipeline.py'));assert w24['DAILY_IDENTITY'](w24['shared_metrics'](pred,policy))==w24['DAILY_IDENTITY'](daily)
 assert canonical_hash(w24['paired'](daily))==canonical_hash(load('paired_comparison'))
 calendar=json.load(open(coverage['references']['calendar']['path']));sessions=[r['date'] for r in calendar if r['engineering_calendar_ok'] and r['price_observed']];positions={d:i for i,d in enumerate(sessions)}
 sources=json.load(open(coverage['references']['sources']['path']));raw={}
 for r in sources:
  p=Path(r['raw_path']);assert file_sha256(p)==r['raw_sha256'];raw[r['security']]={x['date']:x for x in json.load(open(p))['data']}
 cohort={r['security_id']:r for r in config['cohort']['cohort']};temporal_errors=[];source_label_count=0
 for row in data:
  sec=row['stock_id'];day=row['signal_session'];h=row['horizon'];i=positions[day];prev,entry,exit=sessions[i-1],sessions[i+1],sessions[i+h];p=raw[sec];start=cohort[sec]['reported_twse_start'];end=cohort[sec]['reported_end']
  expected_x=[p[day]['close']/p[prev]['close']-1,(p[day]['max']-p[day]['min'])/p[day]['close']]
  boundary='2022-01-01' if row['split']=='TRAIN' else '2023-01-01' if row['split']=='VALIDATION' else '2024-01-01'
  ok=(prev<day<entry<=exit<boundary and row['entry']==entry and row['exit']==exit and row['features']==expected_x and row['diagnostic_label']==p[exit]['close']/p[entry]['open']-1 and start<=prev and (not end or exit<end) and row['exploratory_eligible'] and not row['strict_eligible'])
  if not ok:temporal_errors.append({'stock':sec,'day':day,'horizon':h,'reason':'RAW_FEATURE_LABEL_OR_BOUNDARY_MISMATCH'})
  source_label_count+=1
 training=[]
 for h in [5,10]:
  rows=[r for r in data if r['split']=='TRAIN' and r['horizon']==h];model=fits[str(h)];x=np.asarray([r['features'] for r in rows]);std=x.std(axis=0);std=np.where(std==0,1.,std)
  assert canonical_hash(rows)==model['train_view_sha256'] and len(rows)==model['train_n'];np.testing.assert_array_equal(x.mean(axis=0),model['state']['scaling']['mean']);np.testing.assert_array_equal(std,model['state']['scaling']['std'])
  training.append({'horizon':h,'n':len(rows),'signal_range':[min(r['signal_session'] for r in rows),max(r['signal_session'] for r in rows)],'latest_exit':max(r['exit'] for r in rows),'logical_fit_cutoff':'2022-01-01T00:00:00+08:00','label_exit_before_cutoff':all(r['exit']<'2022-01-01' for r in rows),'train_hash_matches':True,'train_only_scaling_matches':True,'saved_standardized_coefficients':model['state']['coef'],'intercept':model['state']['intercept'],'actual_historical_label_publication':'UNKNOWN'})
 c23=json.load(open(OUT/'window23-smoke-contract.json'));same_recipe=all(contract[k]==c23[k] for k in ['L0','Ridge','seed']);same_features=contract['features']==c23['features_order'];assert same_recipe and same_features
 leakage={'schema':'w25-leakage-matrix-v1','actual_crossperiod_errors':temporal_errors,'raw_formula_samples_checked':source_label_count,'training':training,'checks':[
  {'id':'logical_feature_label_timing','status':'PASS' if not temporal_errors else 'INVALID_OR_CONTAMINATED','evidence':'all44266source-recomputedfeatures/prior/signal/entry/exit andsplitboundary'},
  {'id':'real_feature_observation_publication_availability','status':'UNKNOWN','evidence':'logicalsessiondates not actualhistoricalpublication/snapshot timestamps'},
  {'id':'train_label_maturity','status':'GEOMETRY_PASS_AVAILABILITY_UNKNOWN','evidence':'exitclosebeforelogical2022cutoff;historicalpublication/vintageUNKNOWN'},
  {'id':'train_validation_forward_purge','status':'PASS_WITHIN_W24_DIAGNOSTIC_CONTRACT','evidence':'allhorizonlabels before nextsplit; no otherformalpurge/embargoguarantee claimed'},
  {'id':'scaling_and_fit_partition','status':'PASS_RECORDED_PIPELINE','evidence':'savedtrainingviewhash/mean/std/sourcecode;nofit performed to prove coefficients,unrecordedexperimentsnotprovable'},
  {'id':'recipe_and_selection','status':'RECORDED_NO_ADAPTATION_EXPOSURE_REMAINS','evidence':'L0/Ridge/seed/features exactlyW23;pipelinehasnoselection/refit/search;cannotprove subjective/unrecordedchoices;2022/2023alreadyseen'},
  {'id':'cohort_capital_universe','status':'KNOWN_HINDSIGHT_SELECTION_BIAS','evidence':'2026currentindustry/capital/survivors;3retirednamesunknownstartnotcompared'},
  {'id':'revision_vintage','status':'UNKNOWN_HISTORICAL_ASOF','evidence':'old5/new31currentretrievalvintages;hashpinsprovebytes not2021availability'},
  {'id':'corporate_actions','status':'UNKNOWN','evidence':'noqualifiedcompleteeventfeed,rawprices nottotal/executablereturn'}],
  'actual_generated_at':run['actual_generated_at'],'logical_origin_at':run['identity']['origin_at'],'do_not_confuse_times':True,'original_research_admitted':False,'no_full_no_leakage_certification':True}
 save(OUT/'window25-leakage-matrix.json',leakage)
 groups=[];loso_details=[];risk_cases=[];eligible_securities=coverage['eligible_securities']
 for gi,(year,role,h) in enumerate([(2022,'VALIDATION',5),(2022,'VALIDATION',10),(2023,'FORWARD_DIAGNOSTIC',5),(2023,'FORWARD_DIAGNOSTIC',10)]):
  names={m:f'{m}-{h}-{role}' for m in ['L0','Ridge']};original={m:{r['signal_session']:r for r in daily if r['model']==names[m]} for m in names};days=sorted(original['L0']);assert days==sorted(original['Ridge'])
  byday={m:defaultdict(list) for m in names}
  for m in names:
   for r in pred[names[m]]:byday[m][r['signal_session']].append(r)
  for day in days:assert original['L0'][day]['support_sha256']==original['Ridge'][day]['support_sha256']
  vals={m:np.array([original[m][d]['rank_ic'] for d in days]) for m in names};vals['delta']=vals['Ridge']-vals['L0'];pos=[positions[d] for d in days]
  stats={m:describe(x) for m,x in vals.items()};boot={str(b):{m:circular_bounds(vals[m],pos,b,PLAN['bootstrap']['replicates'],PLAN['bootstrap']['seed_base']+gi*100+b) for m in ['Ridge','delta']} for b in PLAN['bootstrap']['block_lengths']}
  top=sorted(np.flatnonzero(vals['Ridge']>0),key=lambda i:(-vals['Ridge'][i],days[i]))[:5];keep=[i for i in range(len(days)) if i not in set(top)]
  date_effect={'removed_for_sensitivity_only':[days[i] for i in top],'removed_values':[float(vals['Ridge'][i]) for i in top],'Ridge_mean_without_top5positive':float(vals['Ridge'][keep].mean()),'not_cleaning_or_revised_performance':True}
  influence=[]
  for sec in eligible_securities:
   per=[]
   for day in days:
    q=leave_one(byday['L0'][day],byday['Ridge'][day],sec,day,h,policy);row={'removed_security':sec,'period':str(year),'horizon':h,'signal_session':day,'N':q['L0']['valid_n'],'L0':q['L0']['rank_ic'],'Ridge':q['Ridge']['rank_ic'],'delta':q['delta'],'L0_reasons':q['L0']['reasons'],'Ridge_reasons':q['Ridge']['reasons'],'support_sha256':q['L0']['support_sha256']};per.append(row);loso_details.append(row)
   valid={k:[r[k] for r in per if r[k] is not None] for k in ['L0','Ridge','delta']};means={k:float(np.mean(v)) if v else None for k,v in valid.items()}
   influence.append({'removed_security':sec,'days_defined':{k:len(v) for k,v in valid.items()},'N_min':min(r['N'] for r in per),'N_max':max(r['N'] for r in per),'means':means,'changes_from_original':{k:means[k]-stats[k]['mean'] if means[k] is not None else None for k in means}})
  lo={k:{'min':min(r['means'][k] for r in influence),'max':max(r['means'][k] for r in influence),'largest_absolute_mean_change':max(abs(r['changes_from_original'][k]) for r in influence)} for k in ['L0','Ridge','delta']}
  core=[d for i,d in enumerate(sessions) if d.startswith(str(year)) and i+h<len(sessions) and sessions[i+h]<f'{year+1}-01-01'];assert core==days
  all_year=[d for d in sessions if d.startswith(str(year))];excluded=[d for d in all_year if d not in days]
  label_rows=[r for r in data if r['split']==role and r['horizon']==h];label=np.array([r['diagnostic_label'] for r in label_rows]);ranked=sorted(label_rows,key=lambda r:(-abs(r['diagnostic_label']),r['signal_session'],r['stock_id']))[:10]
  risk_cases.extend({'period':year,'horizon':h,'stock_id':r['stock_id'],'session':r['signal_session'],'entry':r['entry'],'exit':r['exit'],'raw_diagnostic_label':r['diagnostic_label'],'cause':'UNKNOWN_NOT_A_CONFIRMED_ACTION','not_threshold_exclusion':True} for r in ranked)
  groups.append({'period':year,'role':role,'horizon':h,'days':len(days),'stats':stats,'bootstrap':boot,'daily_N':dict(sorted(Counter(original['L0'][d]['valid_n'] for d in days).items())),'score_ties':{m:sum(original[m][d]['score_ties_n'] for d in days) for m in names},'label_ties':sum(original['L0'][d]['label_ties_n'] for d in days),'missing_core_sessions':[],'undefined_original_days':0,'excluded_tail_or_split_sessions':excluded,'date_concentration':date_effect,'LOSO':influence,'LOSO_ranges':lo,'LOSO_undefined_days':sum(len(days)-r['days_defined']['delta'] for r in influence),'raw_label_descriptives':dict(zip(['min','p01','median','p99','max'],map(float,np.quantile(label,[0,.01,.5,.99,1])))),'known_action_sensitivity':'UNKNOWN_NOT_ESTIMABLE_NO_DATED_QUALIFIED_ACTION_EVENTS','overlapping_labels':'adjacent5d/10dlabels_sharefuture_sessions;dailyrowsnotIID;block10/20notlongregimecertification'})
  print('completed',year,h,'Ridge mean/median/positive',stats['Ridge']['mean'],stats['Ridge']['median'],stats['Ridge']['positive_fraction'],'LOSO',lo['Ridge'],flush=True)
 refs={'LOSO_daily':save(ROOT/'loso-daily.json',loso_details),'raw_label_cases':save(ROOT/'raw-label-candidates.json',risk_cases)}
 result={'schema':'w25-four-group-robustness-v1','source_bundle_key':run['bundle_key'],'source_manifest_sha256':run['manifest_sha256'],'source_data_sha256':run['identity']['data_sha256'],'analysis_plan_sha256':canonical_hash(PLAN),'groups':groups,'references':refs,'causal_leakage_logical_errors':len(temporal_errors),'corporate_actions':'UNKNOWN_FOR_ALL_PROXY_LABELS; noeventexclusionsinvented','baseline_semantics':{'L0':'fixedpositive1daymomentum','Ridge':'savedsupervisedtwolinearfeatures,signedcoefficientsfrom2021only','target':'bothhigherscoresrankhigherrawdiagnosticreturn','direction_flip_performed':False,'no_claim_of_architecture_advantage':True},'fit_refit_newpredictions_network':0,'research_admitted':False,'statistics':'EXPLORATORY_DEPENDENCE_SENSITIVITY_NOT_FORMAL_SIGNIFICANCE'}
 save(OUT/'window25-robustness.json',result);assert pins()==before
 audit={'status':'AUDIT_COMPLETED_WITH_UNKNOWN_PIT_ACTIONS','input_W24_files_unchanged':before,'existing_predictions_replayed':1880,'paired_daily_rows':940,'LOSO_daily_rows':len(loso_details),'LOSO_cases':len(eligible_securities)*4,'source_label_metadata_checked':source_label_count,'runtime_guards':'urllibopen/socketconnect/np.linalg.solve/ridgefit prohibited','new_requests':0,'fit_refit':0,'original_predictions_changed':False,'original_W24_policy_changed':False,'source_commit':PLAN['input_commit'],'analysis_plan_sha256':canonical_hash(PLAN),'code_files':{str(p.relative_to(REPO)):file_sha256(p) for p in [Path(__file__),REPO/'V6/experimental/v7_robustness_audit.py',REPO/'V6/experimental/v7_e1_evaluate.py']}}
 save(OUT/'window25-completion.json',audit);print(json.dumps(audit))
if __name__=='__main__':
 deny=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('W25_NETWORK_OR_FIT_FORBIDDEN'))
 with patch('urllib.request.OpenerDirector.open',side_effect=deny),patch('socket.socket.connect',side_effect=deny),patch('numpy.linalg.solve',side_effect=deny),patch('V6.experimental.v7_classical_baseline.ridge_fit_state',side_effect=deny),patch('V6.experimental.v7_classical_baseline.fit_baseline',side_effect=deny):main()
