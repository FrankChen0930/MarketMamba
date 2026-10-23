"""One bounded named synthetic study, not a trainer/orchestrator or real admission.

Reuses executable W07/W10 fixture factories only for this engineering demonstration.
Forward label capability is created after selection/fresh-refit/prediction; selection
consumers receive only the W07-supervised subset. No search or runtime scheduling.
"""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import hashlib,json
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_classical_baseline import fit_baseline,score_view
from V6.experimental.v7_classical_baseline_test import panel,selection,ids_for
from V6.experimental.v7_research_temporal import make_pin
from V6.experimental.v7_research_isolation import authorize_view,make_refit_run,make_forward_run
from V6.experimental.v7_research_evaluation import evaluate,prediction_hash
from V6.experimental.v7_research_comparison import compare
from V6.experimental.v7_research_artifacts import publish_bundle,run_key
from V6.experimental.v7_research_replay import runtime_identity

ROOT=Path(__file__).resolve().parents[2]
# Exact existing adapter/evaluator source snapshot. Whole study execution is NOT
# asserted fully reproducible; containing W10 commit separately pins this driver.
SOURCE_COMMIT='4ec22a4330ed6db7df0b3dafb4106cea820516e2'
CODE_PATHS=['V6/experimental/'+x+'.py' for x in ('v7_corrected_e5_train','v7_corrected_e5_replication','v7_research_temporal','v7_temporal_contract','v7_research_evaluation','v7_research_comparison','v7_classical_baseline')]


def _bytes(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def _require(result,status):
    if result['status']!=status:raise ValueError(result['status']+':'+','.join(result.get('reasons',[])))
    return result


def _authorized(p,run,role,indices,supervised):
    cal,fold,days,features,labels=p
    pins={sid:make_pin(fold,cal,f,None if role=='FORWARD_FEATURES' else supervised[sid]) for sid,f in features.items() if role=='FORWARD_FEATURES' or sid in supervised}
    return _require(authorize_view(run,canonical_hash(run),role,ids_for(p,indices),features,
       None if role=='FORWARD_FEATURES' else supervised,pins,fold,cal,make_pin(fold,cal),model_pin=run.get('model_sha256')),'ELIGIBLE')


def _fit(p,rec,run,role,indices,supervised):
    pins={sid:make_pin(p[1],p[0],p[3][sid],label) for sid,label in supervised.items()}
    return fit_baseline(rec,canonical_hash(rec),run,canonical_hash(run),role,ids_for(p,indices),p[3],supervised,pins,p[1],p[0],make_pin(p[1],p[0]))


def _evaluation(p,scores,fits,runs,label_capability,role,policy,extra=None):
    cal,fold,days,features,labels=p;metadata={};payloads={};rows={};aliases=list(scores)
    snapshot=canonical_hash({'feature_contract':fold['feature_contract_id'],'features':{r['sample_id']:features[r['sample_id']] for out in scores.values() for r in out['scores']}})
    source=canonical_hash({'provider':'window10-synthetic-generator','version':1})
    for alias,scored in scores.items():
        rows[alias]=[]
        for row in scored['scores']:
            h=row['horizon'];lab=deepcopy(label_capability[row['sample_id']][str(h)]);lid=row['sample_id']+'-'+str(h)
            meta={k:v for k,v in lab.items() if k!='value'}
            meta.update(horizon=h,snapshot_sha256=snapshot,payload_sha256=canonical_hash(lab),eligibility_status='ELIGIBLE_SYNTHETIC',eligibility_ref=canonical_hash({'label':lid,'role':role}))
            metadata[lid]=meta;payloads[lid]=lab
            rows[alias].append({k:row[k] for k in ('stock_id','signal_session','horizon','score','prediction_at','feature_eligibility_ref')}|{'label_id':lid,'label_metadata_sha256':canonical_hash(meta)})
    raw={'schema':'v7-ranking-synthetic-input-evidence-v1','evidence_class':'SYNTHETIC','calendar':cal,'fold':fold,'policy':policy,'scores':rows,'label_metadata':metadata,'label_payloads':payloads}
    if extra:raw['study_inputs']=extra
    data=_bytes(raw);dsha=hashlib.sha256(data).hexdigest();bundles={};evaluations={}
    for alias in aliases:
        run=runs[alias];fit=fits[alias];sessions=sorted({r['signal_session'] for r in rows[alias]})
        # Declared sessions are role support, even if one model covers fewer stocks.
        header={'schema':'v7-ranking-input-v1','study_id':fold['study_id'],'recipe_id':run['recipe_id'],'run_id':run['run_id'],
            'fold_id':fold['fold_id'],'origin_at':fold['origin_at'],'fold_sha256':canonical_hash(fold),'model_sha256':canonical_hash(fit['model']),
            'source_sha256':source,'snapshot_sha256':snapshot,'vintage_id':'window10-toy-original','evidence_class':'SYNTHETIC','prediction_role':role,
            'evaluation_role':'FORWARD_EVALUATION' if role=='FORWARD' else 'SELECTION_VALIDATION',
            'evaluation_at':days[42]+'T18:00:00+08:00' if role=='FORWARD' else fold['selection_cutoff_at'],
            'sessions':sessions,'horizons':[5],'raw_predictions_ref':{'path':'dataset.json' if role=='FORWARD' else 'validation_snapshot.json','sha256':dsha}}
        bundle={'header':header,'rows':rows[alias]};bundles[alias]=bundle
        evaluations[alias]=_require(evaluate(bundle,prediction_hash(bundle),fold,cal,make_pin(fold,cal),policy,canonical_hash(policy),metadata,payloads),'EVALUATED')
    return raw,data,bundles,evaluations


def _bundle(p,config,raw,rawbytes,bundles,evaluations,paired,ledger,validation):
    cal,fold,days,features,labels=p;h=bundles['a']['header']
    code={'schema':'v7-replay-code-manifest-v1','commit':SOURCE_COMMIT,'files':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in CODE_PATHS}}
    identity={'study_id':h['study_id'],'recipe_id':h['recipe_id'],'run_id':h['run_id'],'fold_id':h['fold_id'],'origin_id':'window10-toy-origin','origin_at':h['origin_at'],'seed':0,
      'code_commit':SOURCE_COMMIT,'prediction_role':'FORWARD','code_manifest_sha256':canonical_hash(code),'data_sha256':canonical_hash(raw),
      'feature_sha256':canonical_hash({'feature_contract_id':fold['feature_contract_id']}),
      'label_sha256':canonical_hash({'label_contract_id':fold['label_contract_id'],'label_definition':fold['label_definition']}),
      'universe_sha256':canonical_hash(sorted({r['stock_id'] for rs in raw['scores'].values() for r in rs})),
      'evaluation_policy_sha256':canonical_hash(raw['policy']),'model_sha256':h['model_sha256'],'training_contract_sha256':canonical_hash(config),
      'checkpoint_status':'NOT_GENERATED_TINY_CLASSICAL_NO_RESUME','checkpoint_sha256':None,
      'source_vintages':[{'source_sha256':h['source_sha256'],'vintage_id':h['vintage_id']}]}
    # v2 explicitly records fitted toy models without a resumable checkpoint.
    objects={
      'resolved_config':('RESOLVED_CONFIG',{'schema':'v7-replay-config-v1','identity':identity,'study_configuration':config}),
      'environment':('ENVIRONMENT',{'schema':'v7-replay-environment-v1','runtime':runtime_identity(),'optional_dependencies':ledger['dependency_probe']}),
      'data_split_eligibility':('DATA_SPLIT_ELIGIBILITY',{'schema':'v7-replay-split-v1','fold_sha256':canonical_hash(fold),'calendar_sha256':canonical_hash(cal),'label_metadata_sha256':canonical_hash(raw['label_metadata'])}),
      'predictions':('PREDICTIONS',{'schema':'v7-replay-predictions-v1','bundles':bundles}),
      'daily_metrics':('DAILY_METRICS',{'schema':'v7-replay-expected-evaluations-v1','evaluations':evaluations}),
      'paired_comparison':('PAIRED_COMPARISON',paired),
      'rejection_ledger':('REJECTION_LEDGER',{'schema':'v7-replay-rejections-v1','evaluations':{k:v['rejections'] for k,v in evaluations.items()},'comparison':paired['exclusions']}),
      'exposure_ledger':('EXPOSURE_LEDGER',{'schema':'v7-replay-exposure-v1','events':[]}),
      'completion_evidence':('COMPLETION_EVIDENCE',{'schema':'v7-replay-completion-v1','training_executed':True,'training_scope':'TINY_SYNTHETIC_CPU','historical_live_prediction_proven':False}),
      'code_manifest':('CODE_MANIFEST',code),'training_logs':('TRAINING_LOGS',ledger),
      'validation_predictions':('PREDICTIONS',{'schema':'v7-classical-validation-predictions-v1','bundles':validation['bundles']}),
      'validation_metrics':('DAILY_METRICS',{'schema':'v7-classical-validation-metrics-v1','evaluations':validation['evaluations']})}
    files={};artifacts=[]
    for name,(kind,value) in objects.items():
        data=_bytes(value);files[name]=data;artifacts.append({'name':name,'kind':kind,'storage':'BUNDLE','path':name+'.json','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'format':'JSON','schema_id':value['schema']})
    for name,path,data,schema in [('dataset_snapshot','dataset.json',rawbytes,raw['schema']),('validation_snapshot','validation_snapshot.json',validation['data'],validation['raw']['schema'])]:
        files[name]=data;artifacts.append({'name':name,'kind':'DATASET_SNAPSHOT','storage':'BUNDLE','path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'format':'JSON','schema_id':schema})
    manifest={'schema':'v7-study-run-bundle-v2','profile':'SYNTHETIC_CLASSICAL_FIT','training_status':'SYNTHETIC_CPU_FIT_COMPLETED','evidence_class':'SYNTHETIC','research_admitted':False,'identity':identity,'required_artifacts':[a['name'] for a in artifacts],'artifacts':artifacts,'logical_observation_period':[days[1],days[42]]}
    return manifest,files


def run_study(root,*,inputs=None,interrupt_at=None,drop_l0_forward_stock=False,comparison_model='L1'):
    result={'schema':'v7-classical-study-evidence-v1','status':'STUDY_REJECTED','research_admitted':False,'reasons':[]}
    try:
        if comparison_model not in {'L1','L2'}:raise ValueError('UNSUPPORTED_FIXED_COMPARISON')
        participants=('L0',comparison_model)
        p=inputs if inputs is not None else panel()
        if not isinstance(p,(tuple,list)) or len(p)!=5 or any(not isinstance(v,Mapping) for v in (p[0],p[1],p[3],p[4])):raise ValueError('INVALID_SYNTHETIC_PANEL')
        cal,fold,days,features,all_labels=p
        config=json.loads((ROOT/'research/v7/window10-synthetic-recipes.json').read_text())
        if comparison_model=='L2':config['selection_policy']['participants']=list(participants);config['comparison_model']='L2'
        configs={r['model_kind']:r for r in config['recipes']}
        eval_policy={'schema':'v7-ranking-policy-prototype-v1','policy_id':'window10-average-equal-toy-v1','status':'SYNTHETIC_NOT_ADMITTED','tie_policy':'AVERAGE','min_valid_n':2,'aggregation_weighting':'EQUAL_SESSION','invalid_support':'EXCLUDE_WITH_LEDGER','statistical_policy':'NOT_CONFIGURED_NO_INFERENCE'}
        # Capability subset excludes forward labels. They are not passed to fit/selection.
        supervised={sid:l for sid,l in all_labels.items() if features[sid]['signal_session']<=fold['refit'][1]}
        selections={};selection_fits={};val_scores={}
        for kind in participants:
            run=selection(p,configs[kind]);selections[kind]=run
            fitted=_require(_fit(p,configs[kind],run,'SELECTION_TRAIN',(1,2,3),supervised),'FIT_ENGINEERING_ONLY');selection_fits[kind]=fitted
            auth=_authorized(p,run,'SELECTION_VALIDATION',(15,16),supervised)
            val_scores[kind]=_require(score_view(fitted,canonical_hash(fitted),auth,canonical_hash(auth),run,canonical_hash(run)),'SCORED')
        dependency_probe=selection_fits['L2'] if comparison_model=='L2' else _fit(p,configs['L2'],selection(p,configs['L2']),'SELECTION_TRAIN',(1,2,3),supervised)
        if dependency_probe['status'] not in {'DEPENDENCY_BLOCKED','FIT_ENGINEERING_ONLY'}:raise ValueError('L2_PROBE_INVALID')
        vraw,vdata,vbundles,vevals=_evaluation(p,val_scores,selection_fits,selections,supervised,'SELECTION',eval_policy)
        policy=config['selection_policy']
        if policy['status']!='SYNTHETIC_NOT_ADMITTED' or policy['metric']!='VALIDATION_MEAN_DAILY_RANK_IC' or policy['horizon']!=5:raise ValueError('SYNTHETIC_SELECTION_POLICY_REQUIRED')
        means={k:v['summary']['by_horizon'][0]['mean_rank_ic'] for k,v in vevals.items()}
        selected=sorted(means,key=lambda k:(-means[k],policy['tie_break'].index(k)))[0]
        decision={'schema':'v7-classical-selection-v1','policy':deepcopy(policy),'selected_model_kind':selected,'selected_recipe_id':configs[selected]['recipe_id'],'validation_means':means,'evaluation_identities':{k:v['evaluation_identity'] for k,v in vevals.items()},'role':'VALIDATION_MODEL_SELECTION_NOT_FINAL_TEST','research_admitted':False}
        before=canonical_hash(decision);refit_fits={};refits={};forwards={};forward_scores={}
        d=days[29];ready={k:d+v for k,v in {'model_built_at':'T07:10:00+08:00','artifact_verified_at':'T07:20:00+08:00','approval_at':'T07:30:00+08:00','ready_at':'T07:40:00+08:00'}.items()}
        for kind in participants:
            refit=make_refit_run(selections[kind],'refit-'+kind,fold,ready);refit['selection_decision_sha256']=before;refits[kind]=refit
            fitted=_require(_fit(p,configs[kind],refit,'REFIT',(1,2,3,15,16,18),supervised),'FIT_ENGINEERING_ONLY');refit_fits[kind]=fitted
            forward=make_forward_run(refit,'forward-'+kind,fold);forwards[kind]=forward
            auth=_authorized(p,forward,'FORWARD_FEATURES',(30,31),supervised)
            forward_scores[kind]=_require(score_view(fitted,canonical_hash(fitted),auth,canonical_hash(auth),forward,canonical_hash(forward)),'SCORED')
        if drop_l0_forward_stock:
            forward_scores['L0']['scores']=[r for r in forward_scores['L0']['scores'] if r['stock_id']!='toy-3']
        # Only now make matured forward labels available to evaluation, never selection.
        forward_labels={sid:all_labels[sid] for sid in ids_for(p,(30,31))}
        scores={'a':forward_scores[comparison_model],'b':forward_scores['L0']};fits={'a':refit_fits[comparison_model],'b':refit_fits['L0']};runs={'a':forwards[comparison_model],'b':forwards['L0']}
        extra={'features':features,'supervised_labels':supervised,'selection_policy_sha256':canonical_hash(policy)}
        raw,data,bundles,fevals=_evaluation(p,scores,fits,runs,forward_labels,'FORWARD',eval_policy,extra)
        pair=_require(compare(fevals['a'],canonical_hash(fevals['a']),fevals['b'],canonical_hash(fevals['b'])),'COMPARED')
        after=canonical_hash(decision)
        if before!=after:raise ValueError('SAME_FOLD_FORWARD_CHANGED_SELECTION')
        ledger={'schema':'v7-classical-fit-ledger-v1','scope':'TINY_SYNTHETIC_CPU','selection_fits':selection_fits,'selection_declarations':selections,'selection_decision':decision,'selection_decision_sha256':before,
                'refit_fits':refit_fits,'refit_declarations':refits,'forward_declarations':forwards,'dependency_probe':dependency_probe,'forward_prediction_artifact_sha256':{k:canonical_hash(v) for k,v in forward_scores.items()},'permission_declarations_are_initial_intent_not_actual_execution_status':True}
        manifest,files=_bundle(p,config,raw,data,bundles,fevals,pair,ledger,{'raw':vraw,'data':vdata,'bundles':vbundles,'evaluations':vevals})
        pub=publish_bundle(root,manifest,canonical_hash(manifest),files,interrupt_at=interrupt_at)
        return {**result,'status':'STUDY_ARTIFACTS_VERIFIED' if pub['status'] in {'ARTIFACTS_VERIFIED','ALREADY_PUBLISHED'} else 'STUDY_INCOMPLETE',
          'publication':pub,'manifest':manifest,'manifest_sha256':canonical_hash(manifest),'run_key':run_key(manifest['identity']),
          'selection_fits':selection_fits,'refit_fits':refit_fits,'selection_decision':decision,'selection_decision_before_sha256':before,'selection_decision_after_sha256':after,
          'validation_evaluations':vevals,'forward_evaluations':fevals,'paired_comparison':pair,'dependency_probe':dependency_probe}
    except (KeyError,TypeError,ValueError,IndexError) as exc:return {**result,'reasons':[str(exc)]}
