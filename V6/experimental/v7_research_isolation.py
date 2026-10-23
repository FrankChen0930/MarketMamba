"""Small research read-scope guards; not a trainer, registry, OS sandbox or admission.

Manifests describe proposed/mock identity and readiness, not actual state reset or
model execution. Only guarded returned views are suitable for the mock consumer;
callers retaining original stores are outside this Python-library trust boundary.
"""
from collections.abc import Mapping
from copy import deepcopy
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal import check_fold,check_sample,role_check,instant,_sha

RUN_SCHEMA='v7-research-run-engineering-v1'
STATES=('model','optimizer','scheduler','rng')


def make_refit_run(selection,new_id,fold,readiness):
    """Construct a DECLARED manifest. It must pass authorization before data use."""
    return {'schema':RUN_SCHEMA,'kind':'REFIT','study_id':fold['study_id'],'fold_id':fold['fold_id'],
            'fold_sha256':canonical_hash(fold),'run_id':new_id,'recipe_id':selection['recipe_id'],'seed':selection['seed'],
            'execution':'DECLARED_ONLY_NOT_EXECUTED','selection':deepcopy(selection),
            'selection_sha256':canonical_hash(selection),'initialization':{k:'NEW' for k in STATES},
            'resume_checkpoint':None,'readiness':deepcopy(readiness)}


def make_forward_run(refit,new_id,fold):
    return {'schema':RUN_SCHEMA,'kind':'FORWARD','study_id':fold['study_id'],'fold_id':fold['fold_id'],
            'fold_sha256':canonical_hash(fold),'run_id':new_id,'recipe_id':refit['recipe_id'],'seed':refit['seed'],
            'execution':'DECLARED_ONLY_NOT_EXECUTED','model':deepcopy(refit),'model_sha256':canonical_hash(refit)}


def _text(x):return isinstance(x,str) and bool(x.strip()) and x==x.strip()


def _run_issue(run,fold,kind):
    if not isinstance(run,Mapping) or run.get('schema')!=RUN_SCHEMA or run.get('kind')!=kind:return 'RUN_KIND_OR_SCHEMA_MISMATCH'
    if any(run.get(k)!=fold[k] for k in ('study_id','fold_id')) or run.get('fold_sha256')!=canonical_hash(fold):return 'RUN_FOLD_IDENTITY_MISMATCH'
    if not all(_text(run.get(k)) for k in ('run_id','recipe_id')) or type(run.get('seed')) is not int or run['seed']<0:return 'RUN_IDENTITY_MISSING'
    if run.get('execution')!='DECLARED_ONLY_NOT_EXECUTED':return 'EXECUTION_SCOPE_NOT_SUPPORTED'
    if kind=='REFIT':
        parent=run.get('selection')
        issue=_run_issue(parent,fold,'SELECTION')
        if issue:return issue
        if canonical_hash(parent)!=run.get('selection_sha256'):return 'SELECTION_IDENTITY_MISMATCH'
        if run['run_id']==parent['run_id']:return 'REFIT_ID_NOT_NEW'
        if run.get('resume_checkpoint') is not None or run.get('initialization')!={k:'NEW' for k in STATES}:return 'REFIT_IS_NOT_RESUME'
        if any(run.get(k)!=parent[k] for k in ('recipe_id','seed')):return 'REFIT_RECIPE_OR_SEED_MISMATCH'
        try:
            built,verified,approved,ready=(instant(run['readiness'][k]) for k in ('model_built_at','artifact_verified_at','approval_at','ready_at'))
            if not instant(fold['refit_cutoff_at'])<=built<=verified<=approved<=ready:return 'INCONSISTENT_MODEL_READINESS'
            if ready>instant(fold['model_ready_deadline_at']):return 'MODEL_READY_DEADLINE_MISSED'
        except (KeyError,TypeError,ValueError):return 'MODEL_READINESS_UNKNOWN'
    if kind=='FORWARD':
        model=run.get('model');issue=_run_issue(model,fold,'REFIT')
        if issue:return issue
        if canonical_hash(model)!=run.get('model_sha256'):return 'MODEL_IDENTITY_MISMATCH'
        if run['run_id'] in {model['run_id'],model['selection']['run_id']}:return 'FORWARD_ID_NOT_NEW'
        if any(run.get(k)!=model[k] for k in ('recipe_id','seed')):return 'FORWARD_RECIPE_OR_SEED_MISMATCH'
    return None


def authorize_view(run,expected_run_sha256,role,sample_ids,feature_store,label_store,sample_pins,fold,calendar,fold_pin,*,model_pin=None):
    base={'research_admitted':False,'view':[],'decisions':[],'run_id':run.get('run_id') if isinstance(run,Mapping) else None,'role':role}
    def reject(status,reason):return {**base,'status':status,'reasons':[reason]}
    f=check_fold(fold,calendar,fold_pin)
    if f['status']!='ELIGIBLE':return {**base,'status':f['status'],'reasons':f['reasons'],'fold_decision':f}
    try:
        if not isinstance(run,Mapping) or canonical_hash(run)!=expected_run_sha256:return reject('REJECTED','RUN_IDENTITY_MISMATCH')
        kind={'SELECTION_TRAIN':'SELECTION','SELECTION_VALIDATION':'SELECTION','REFIT':'REFIT','FORWARD_FEATURES':'FORWARD'}.get(role)
        if kind is None:return reject('REJECTED','ROLE_UNKNOWN')
        issue=_run_issue(run,fold,kind)
        if issue:return reject('BLOCKED' if issue.endswith('_UNKNOWN') else 'REJECTED',issue)
        if role=='FORWARD_FEATURES' and (model_pin is None or run.get('model_sha256')!=model_pin):return reject('REJECTED','PINNED_REFIT_MODEL_MISMATCH')
        if not isinstance(sample_ids,list) or not sample_ids or not all(_text(x) for x in sample_ids) or len(set(sample_ids))!=len(sample_ids):
            return reject('REJECTED','INVALID_SAMPLE_REQUEST')
        views=[];decisions=[]
        for sid in sample_ids:
            if sid not in feature_store or sid not in sample_pins:
                decisions.append({'status':'BLOCKED','reasons':['SAMPLE_SOURCE_OR_PIN_MISSING'],'sample_id':sid});continue
            feature=deepcopy(feature_store[sid]);pin=sample_pins[sid]
            if not isinstance(feature,Mapping) or feature.get('sample_id')!=sid:
                decisions.append({'status':'REJECTED','reasons':['SAMPLE_KEY_MISMATCH'],'sample_id':sid});continue
            issue=role_check(feature,fold,calendar,role)
            if issue:
                decisions.append({'status':'REJECTED','reasons':[issue],'sample_id':sid});continue
            # This branch deliberately does not inspect, query, iterate or load label_store.
            if role=='FORWARD_FEATURES':labels=None
            else:
                try:labels=deepcopy(label_store[sid])
                except KeyError:
                    decisions.append({'status':'BLOCKED','reasons':['LABEL_SOURCE_MISSING'],'sample_id':sid});continue
            check=check_sample(feature,labels,fold,calendar,pin,role);decisions.append(check)
            if check['status']=='ELIGIBLE':views.append({'features':feature} if role=='FORWARD_FEATURES' else {'features':feature,'labels':labels})
        base['decisions']=decisions
        if any(d['status']!='ELIGIBLE' for d in decisions):
            return {**base,'status':'REJECTED' if any(d['status']=='REJECTED' for d in decisions) else 'BLOCKED','reasons':['BATCH_HAS_INELIGIBLE_SAMPLE']}
        return {**base,'status':'ELIGIBLE','reasons':[],'view':views,'run_sha256':expected_run_sha256,'view_sha256':canonical_hash(views),'fold_sha256':fold_pin['fold_sha256']}
    except (KeyError,TypeError,ValueError):return reject('REJECTED','MALFORMED_RUN_OR_REQUEST')


def expose_forward_result(result,expected_result_sha256,target_fold,calendar,fold_pin,used_at,decision_ref,ledger,expected_ledger_sha256):
    """Return a new explicit exposure list + authorized result; never mutate input.

    This records use of a later-fold research result, not approval of its statistics.
    The caller persists the returned root/list; there is no hidden global registry.
    """
    def refuse(status,reason):return {'status':status,'reasons':[reason],'research_admitted':False,'result':None,'ledger':None}
    f=check_fold(target_fold,calendar,fold_pin)
    if f['status']!='ELIGIBLE':return refuse(f['status'],f['reasons'][0])
    try:
        if not isinstance(result,Mapping) or canonical_hash(result)!=expected_result_sha256:return refuse('REJECTED','RESULT_IDENTITY_MISMATCH')
        if not isinstance(ledger,list) or canonical_hash(ledger)!=expected_ledger_sha256:return refuse('REJECTED','LEDGER_IDENTITY_MISMATCH')
        if not all(_text(result.get(k)) for k in ('source_study_id','source_fold_id','source_run_id','source_sha256')) or not _text(decision_ref):return refuse('BLOCKED','EXPOSURE_IDENTITIES_UNKNOWN')
        if not _sha(result.get('source_sha256')):return refuse('BLOCKED','FORWARD_SOURCE_IDENTITY_UNKNOWN')
        if result['source_fold_id']==target_fold['fold_id'] and result['source_study_id']==target_fold['study_id']:return refuse('REJECTED','SAME_FOLD_FORWARD_CANNOT_SELECT')
        available,origin,used=map(instant,(result['available_at'],result['source_origin_at'],used_at))
        if not origin<=available<=used<=instant(target_fold['selection_cutoff_at']):return refuse('REJECTED','FORWARD_RESULT_NOT_AVAILABLE_FOR_SELECTION')
        if origin>=instant(target_fold['origin_at']):return refuse('REJECTED','NON_LATER_TARGET_FOLD')
        if result.get('evidence_class') not in {'SYNTHETIC','QUALIFIED_REFERENCE'}:return refuse('BLOCKED','RESULT_EVIDENCE_UNKNOWN')
        event={'schema':'v7-research-exposure-engineering-v1','source_study_id':result['source_study_id'],'source_fold_id':result['source_fold_id'],
               'source_run_id':result['source_run_id'],'target_study_id':target_fold['study_id'],'target_fold_id':target_fold['fold_id'],
               'role':'DEVELOPMENT_EXPOSED_NOT_FINAL','used_at':used_at,'decision_ref':decision_ref,'result_sha256':expected_result_sha256,
               'prior_ledger_sha256':expected_ledger_sha256,'evidence_class':result['evidence_class']}
        new=deepcopy(ledger)+[event]
        return {'status':'EXPOSURE_RECORDED','reasons':[],'research_admitted':False,'result':deepcopy(result),'ledger':new,'ledger_sha256':canonical_hash(new)}
    except (KeyError,TypeError,ValueError):return refuse('BLOCKED','EXPOSURE_CLOCK_OR_RECORD_UNKNOWN')
