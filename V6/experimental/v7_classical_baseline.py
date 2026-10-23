"""Bounded synthetic classical adapters; W07 owns roles/eligibility, W08 metrics.

No data loading, split policy, factor admission, search, checkpoint resume or GPU.
Legacy ridge math was inspected; importing its driver mutates legacy feature config,
so this small NumPy solve avoids importing that historical data/runtime wrapper.
"""
from collections.abc import Mapping
from copy import deepcopy
from importlib.util import find_spec
from importlib.metadata import version
import math
import numpy as np
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_isolation import authorize_view
from V6.experimental.v7_research_temporal import _text

SCHEMA='v7-classical-fit-v1'


def dependency_status():
    try:
        installed=version('lightgbm') if find_spec('lightgbm') else None
        return {'available':installed is not None and installed.split('.')[0]=='4','installed_version':installed,
                'requirement_candidate':'lightgbm>=4,<5; exact version lock requires actual CPU verification',
                'status':'SYNTHETIC_ENGINEERING_ONLY_NO_INSTALL_AUTHORITY'}
    except (ValueError,ImportError):return {'available':False,'installed_version':None,'requirement_candidate':'lightgbm>=4,<5; exact lock unverified'}


def _recipe(r,expected):
    if not isinstance(r,Mapping) or canonical_hash(r)!=expected:raise ValueError('RECIPE_IDENTITY_MISMATCH')
    if r['schema']!='v7-classical-recipe-v1' or r['status']!='SYNTHETIC_ENGINEERING_ONLY' or not _text(r['recipe_id']):raise ValueError('SYNTHETIC_RECIPE_REQUIRED')
    if type(r['seed']) is not int or r['seed']<0 or type(r['horizon']) is not int or r['horizon']<=0:raise ValueError('SEED_OR_HORIZON_INVALID')
    if type(r['max_samples']) is not int or not 1<=r['max_samples']<=64 or type(r['max_features']) is not int or not 1<=r['max_features']<=4:raise ValueError('TINY_SYNTHETIC_BOUNDS_REQUIRED')
    fs=r['features']
    if not isinstance(fs,list) or not fs or not all(_text(f) for f in fs) or len(set(fs))!=len(fs) or len(fs)>r['max_features']:raise ValueError('FEATURE_ORDER_INVALID')
    c=r['config'];kind=r['model_kind']
    if not isinstance(c,Mapping):raise ValueError('CONFIG_INVALID')
    if kind=='L0':
        if set(c)!={'factor','direction'} or c['factor'] not in fs or type(c['direction']) not in (int,float) or c['direction'] not in (-1,1):raise ValueError('EXPLICIT_FACTOR_DIRECTION_REQUIRED')
    elif kind=='L1':
        if set(c)!={'alpha','scaling','zero_variance','fit_intercept'} or type(c['alpha']) not in (int,float) or not math.isfinite(c['alpha']) or c['alpha']<=0 or c['scaling']!='TRAIN_ONLY_ZSCORE' or c['zero_variance']!='UNIT_SCALE' or c['fit_intercept'] is not True:raise ValueError('RIDGE_CONFIG_UNSUPPORTED')
    elif kind=='L2':
        if set(c)!={'n_estimators','num_leaves','max_depth','learning_rate','min_child_samples','n_jobs'}:raise ValueError('L2_CONFIG_UNSUPPORTED')
        if any(type(c[k]) is not int for k in ('n_estimators','num_leaves','max_depth','min_child_samples','n_jobs')) or not(1<=c['n_estimators']<=8 and 2<=c['num_leaves']<=4 and 1<=c['max_depth']<=2 and c['min_child_samples']==1 and c['n_jobs']==1):raise ValueError('L2_TINY_CPU_BOUNDS_REQUIRED')
        if type(c['learning_rate']) not in (int,float) or not 0<c['learning_rate']<=1:raise ValueError('L2_LEARNING_RATE_INVALID')
    else:raise ValueError('MODEL_KIND_UNSUPPORTED')


def _matrix(rows,features):
    if any(not isinstance(row,Mapping) or not isinstance(row.get('features'),Mapping) or set(row['features']['values'])!=set(features) for row in rows):raise ValueError('FEATURE_SCHEMA_MISMATCH')
    x=np.asarray([[row['features']['values'][f] for f in features] for row in rows],dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('INVALID_FEATURE_MATRIX')
    return x


def fit_baseline(recipe,expected_recipe_sha256,run,expected_run_sha256,phase,sample_ids,features,labels,sample_pins,fold,calendar,fold_pin):
    base={'schema':SCHEMA,'research_admitted':False,'model':None,'reasons':[]}
    def reject(reason,status='REJECTED'):return {**base,'status':status,'reasons':[reason]}
    try:
        _recipe(recipe,expected_recipe_sha256)
        if any(not isinstance(v,Mapping) for v in (calendar,fold,fold_pin)):return reject('CALENDAR_OR_FOLD_CONTAINER_INVALID')
        if calendar.get('qualification')!='SYNTHETIC':return reject('SYNTHETIC_CALENDAR_REQUIRED')
        if phase not in {'SELECTION_TRAIN','REFIT'}:return reject('FIT_ROLE_FORBIDDEN')
        if not isinstance(sample_ids,list) or len(sample_ids)>recipe['max_samples']:return reject('FIT_SAMPLE_BOUNDS')
        if not isinstance(run,Mapping) or any(run.get(k)!=recipe[k] for k in ('recipe_id','seed')):return reject('RUN_RECIPE_IDENTITY_MISMATCH')
        if recipe['horizon'] not in fold['required_horizons']:return reject('HORIZON_MISMATCH')
        auth=authorize_view(run,expected_run_sha256,phase,sample_ids,features,labels,sample_pins,fold,calendar,fold_pin)
        if auth['status']!='ELIGIBLE':return {**reject('TEMPORAL_AUTHORIZATION_DENIED'),'authorization':auth}
        if recipe['model_kind']=='L2' and not dependency_status()['available']:return {**reject('lightgbm dependency missing or version not qualified','DEPENDENCY_BLOCKED'),'dependency':dependency_status()}
        rows=auth['view'];x=_matrix(rows,recipe['features']);h=str(recipe['horizon'])
        y=np.asarray([row['labels'][h]['value'] for row in rows],dtype=np.float64)
        if not np.isfinite(y).all():return reject('INVALID_LABEL_VECTOR')
        model={'kind':recipe['model_kind'],'recipe':deepcopy(recipe),'recipe_sha256':expected_recipe_sha256,
               'study_id':fold['study_id'],'fold_id':fold['fold_id'],'fold_sha256':fold_pin['fold_sha256'],
               'feature_contract_id':fold['feature_contract_id'],'label_contract_id':fold['label_contract_id'],
               'fit_run_id':run['run_id'],'fit_run_sha256':expected_run_sha256,'fit_phase':phase,
               'fit_view_sha256':auth['view_sha256'],'fit_n':len(rows),'seed':recipe['seed'],
               'universe_sha256':canonical_hash(sorted({r['features']['stock_id'] for r in rows})),
               'initialization':'NEW_SYNTHETIC_STATE','resume_checkpoint':None}
        if recipe['model_kind']=='L1':
            mean=x.mean(axis=0);std=x.std(axis=0);std=np.where(std==0,1.,std);z=(x-mean)/std;ym=float(y.mean())
            coef=np.linalg.solve(z.T@z+recipe['config']['alpha']*np.eye(x.shape[1]),z.T@(y-ym))
            model.update(scaling={'mean':mean.tolist(),'std':std.tolist()},coef=coef.tolist(),intercept=ym)
        elif recipe['model_kind']=='L2':
            import lightgbm as lgb
            estimator=lgb.LGBMRegressor(**recipe['config'],random_state=recipe['seed'],deterministic=True,force_col_wise=True,verbosity=-1,device_type='cpu')
            estimator.fit(x,y);model.update(booster_text=estimator.booster_.model_to_string(),dependency=dependency_status())
        # Ensure serializable finite numeric state before exposing successful fit.
        canonical_hash(model)
        return {**base,'status':'FIT_ENGINEERING_ONLY','model':model,
                'training_status':'DIRECT_SCORE_NO_OPTIMIZER' if recipe['model_kind']=='L0' else 'FITTED_TINY_SYNTHETIC_CPU',
                'identity':{'study_id':fold['study_id'],'recipe_id':recipe['recipe_id'],'run_id':run['run_id'],
                            'fold_id':fold['fold_id'],'origin_at':fold['origin_at'],'horizon':recipe['horizon'],
                            'config_sha256':expected_recipe_sha256,'data_view_sha256':auth['view_sha256']}}
    except (KeyError,TypeError,ValueError,OverflowError,np.linalg.LinAlgError) as exc:return reject(str(exc))


def score_view(fit,expected_fit_sha256,authorized,expected_authorization_sha256,prediction_run,expected_run_sha256):
    """Consume immutable W07 authorization; never accept a Forward Label Store.

    Callers own authentic pins; as with W07 this is a Python API, not OS isolation.
    For forward scores the actual fitted artifact AND declarative refit run are bound.
    """
    base={'schema':'v7-classical-scores-v1','research_admitted':False,'scores':[],'reasons':[]}
    def fail(reason):return {**base,'status':'REJECTED','reasons':[reason]}
    try:
        if not isinstance(fit,Mapping) or canonical_hash(fit)!=expected_fit_sha256 or fit['status']!='FIT_ENGINEERING_ONLY':return fail('FIT_IDENTITY_MISMATCH')
        if not isinstance(authorized,Mapping) or canonical_hash(authorized)!=expected_authorization_sha256 or authorized['status']!='ELIGIBLE':return fail('VIEW_IDENTITY_MISMATCH')
        if not isinstance(prediction_run,Mapping) or canonical_hash(prediction_run)!=expected_run_sha256 or authorized['run_sha256']!=expected_run_sha256:return fail('PREDICTION_RUN_IDENTITY_MISMATCH')
        m=fit['model'];r=m['recipe'];role=authorized['role']
        if role not in {'SELECTION_VALIDATION','FORWARD_FEATURES'}:return fail('SCORING_ROLE_FORBIDDEN')
        if authorized['fold_sha256']!=m['fold_sha256'] or prediction_run['fold_sha256']!=m['fold_sha256'] or prediction_run['recipe_id']!=r['recipe_id'] or prediction_run['seed']!=r['seed']:return fail('MODEL_FOLD_OR_RECIPE_MISMATCH')
        if role=='SELECTION_VALIDATION':
            if m['fit_phase']!='SELECTION_TRAIN' or prediction_run['run_id']!=m['fit_run_id'] or expected_run_sha256!=m['fit_run_sha256']:return fail('SELECTION_MODEL_SCOPE_MISMATCH')
        elif m['fit_phase']!='REFIT' or prediction_run['model_sha256']!=m['fit_run_sha256']:return fail('FORWARD_REQUIRES_FRESH_REFIT')
        rows=authorized['view'];x=_matrix(rows,r['features'])
        if any(row['features']['feature_contract_id']!=m['feature_contract_id'] for row in rows):return fail('FEATURE_CONTRACT_MISMATCH')
        if m['kind']=='L0':scores=x[:,r['features'].index(r['config']['factor'])]*r['config']['direction']
        elif m['kind']=='L1':scores=((x-np.asarray(m['scaling']['mean']))/np.asarray(m['scaling']['std']))@np.asarray(m['coef'])+m['intercept']
        else:
            if not dependency_status()['available']:return fail('lightgbm dependency unavailable for scores')
            import lightgbm as lgb
            scores=lgb.Booster(model_str=m['booster_text']).predict(x,num_threads=1)
        if not np.isfinite(scores).all():return fail('NONFINITE_PREDICTIONS')
        result=[]
        for row,value in zip(rows,scores):
            f=row['features'];result.append({'sample_id':f['sample_id'],'stock_id':f['stock_id'],'signal_session':f['signal_session'],
               'horizon':r['horizon'],'score':float(value),'prediction_at':f['decision_at'],
               'feature_eligibility_ref':canonical_hash(f)})
        return {**base,'status':'SCORED','scores':result,'prediction_run_id':prediction_run['run_id'],
                'prediction_role':role,'fit_artifact_sha256':expected_fit_sha256,'authorization_sha256':expected_authorization_sha256,
                'model_configuration':deepcopy(r),'fit_metadata':deepcopy(fit['identity']),'training_status':fit['training_status']}
    except (KeyError,TypeError,ValueError,OverflowError) as exc:return fail(str(exc))
