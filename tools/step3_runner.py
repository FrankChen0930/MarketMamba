"""Step3 fixed CPU fit queue and durable job artifacts, no evaluation imports.

This phase's real-data admission remains STOP. fit_job checks it before fitting.
No CLI implicitly starts training; caller supplies already verified matrices.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import uuid
import numpy as np
from tools.step3 import (FEATURES, PARAMS, SEEDS, YEARS, SPEC_SHA, ROOT, private_dir,
                        require_admission, sha256, verify_file, validate_arms)


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def array_digest(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def queue():
    return [(year,seed,arm) for year in YEARS for seed in SEEDS for arm in 'AB']


def feature_support(x):
    a=np.asarray(x)
    if a.ndim!=2 or a.shape[1]!=5:raise ValueError('STOP monthly feature width')
    out=[]
    for j,name in enumerate(FEATURES):
        v=a[:,j];v=v[np.isfinite(v)]
        if len(v)<2 or v.min()==v.max():raise ValueError('STOP unobservable/constant monthly feature '+name)
        out.append(dict(feature=name,finite=int(len(v)),missing=int(len(a)-len(v))))
    return out


def validate_outputs(outputs):
    if set(outputs)!={(arm,seed) for arm in 'AB' for seed in SEEDS}:raise ValueError('STOP six outputs required')
    expected=outputs['A',17]
    for keys in outputs.values():
        a=np.asarray(keys)
        if a.ndim!=2 or a.shape[1]!=2 or len(np.unique(a,axis=0))!=len(a) or not np.array_equal(a,expected):
            raise ValueError('STOP output keys differ/duplicate')


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def resume(directory,request):
    p=Path(directory)
    try:
        if p.is_symlink() or (p/'manifest.json').is_symlink():raise ValueError('symlink')
        m=json.loads((p/'manifest.json').read_text())
        if m['state']!='COMPLETE' or m['request']!=request or m['request_sha256']!=digest_json(request):raise ValueError('identity')
        if not {'keys.npy','sample.npy','model.txt'}<=set(m['files']):raise ValueError('missing required outputs')
        actual={v.name for v in p.iterdir()}
        if actual!=set(m['files'])|{'manifest.json'}:raise ValueError('unexpected outputs')
        for name,info in m['files'].items():
            if Path(name).name!=name or (p/name).is_symlink():raise ValueError('unsafe file')
            verify_file(p/name,info['sha256'],info['bytes'])
        return m
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('STOP corrupt/incomplete/mismatched job '+str(p)) from exc


def publish(root,name,request,arrays,model_bytes,telemetry=None):
    root=private_dir(root)
    if Path(name).name!=name or name.startswith('.'):raise ValueError('STOP job name')
    final=root/name
    if final.exists():resume(final,request);return final
    stage=root/('.partial-'+name+'-'+uuid.uuid4().hex);stage.mkdir()
    # On interruption the partial namespace is kept for explicit recovery/audit.
    for key,value in arrays.items():
        if Path(key).name!=key or not key.endswith('.npy'):raise ValueError('STOP artifact filename')
        with (stage/key).open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    with (stage/'model.txt').open('xb') as f:f.write(model_bytes);f.flush();os.fsync(f.fileno())
    files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in stage.iterdir()}
    manifest=dict(state='COMPLETE',request=request,request_sha256=digest_json(request),files=files,telemetry=telemetry or {})
    with (stage/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    _sync_dir(stage);resume(stage,request)
    # No overwrite: an existing final namespace is a conflict, never silently replaced.
    if final.exists():raise ValueError('STOP concurrent publication')
    stage.rename(final);_sync_dir(root)
    return final


def recover_orphan(stage,final,request):
    stage,final=Path(stage),Path(final)
    private_dir(stage.parent)
    if stage.parent!=final.parent or not stage.name.startswith('.partial-') or final.exists():raise ValueError('STOP orphan policy')
    resume(stage,request);stage.rename(final);_sync_dir(final.parent)
    return final


def fit_job(gate,root,year,seed,arm,keys,fit_rows,y_fit,x_fit,x_eval,input_identity):
    """One complete fit+inference, sampled training rows provided by shared P1 plan.

    No correlation, labels at evaluation time, feature importance, ranking or
    performance summary is computed. Scores are never printed or returned.
    """
    require_admission(gate)
    if year not in YEARS or seed not in SEEDS or arm not in ('A','B'):raise ValueError('STOP queue identity')
    width=185 if arm=='A' else 190
    if x_fit.shape!=(len(fit_rows),width) or x_eval.shape!=(len(keys),width):raise ValueError('STOP matrix dimensions')
    if len(fit_rows)>500_000 or not len(fit_rows) or len(y_fit)!=len(fit_rows) or not np.isfinite(y_fit).all():raise ValueError('STOP fit support')
    if np.isinf(x_fit).any() or np.isinf(x_eval).any():raise ValueError('STOP infinite matrix')
    if len(np.unique(keys,axis=0))!=len(keys):raise ValueError('STOP duplicate inference key')
    request=dict(contract=SPEC_SHA,source=sha256(__file__),primitives=sha256(ROOT/'tools/step3.py'),
        year=year,seed=seed,arm=arm,params=PARAMS,input_identity=input_identity,
        sample_sha256=array_digest(fit_rows),target_sha256=array_digest(y_fit),keys_sha256=array_digest(keys),
        fit_sha256=array_digest(x_fit),eval_sha256=array_digest(x_eval))
    root=private_dir(root);name=f'fold{year}-{arm}-seed{seed}'
    if (root/name).exists():resume(root/name,request);return root/name
    from lightgbm import LGBMRegressor
    start=time.monotonic();model=LGBMRegressor(**PARAMS,random_state=seed).fit(x_fit,y_fit)
    fit_seconds=time.monotonic()-start;predict_start=time.monotonic()
    prediction=np.asarray(model.predict(x_eval),np.float64)
    if prediction.shape!=(len(keys),) or not np.isfinite(prediction).all():raise ValueError('STOP nonfinite/incomplete predictions')
    telemetry=dict(fit_seconds=fit_seconds,inference_seconds=time.monotonic()-predict_start,
        total_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fit_matrix_bytes=x_fit.nbytes,eval_matrix_bytes=x_eval.nbytes,rows_fit=len(fit_rows),rows_eval=len(keys))
    return publish(root,name,request,{'keys.npy':keys,'sample.npy':fit_rows,'prediction.npy':prediction},
                   model.booster_.model_to_string().encode(),telemetry)


def build_arms(data,rows,monthly_values):
    """Build only requested P1 rows, avoiding a 6-million-row 185-column copy.

    data exposes c0plus's exact dense-panel interface. monthly_values is keyed in
    original matrix order and contains exactly the five frozen numeric columns.
    This function performs no fitting, ranking or evaluation.
    """
    from research_lab.c0plus import build_features
    rows=np.asarray(rows)
    if len(data.feature_names)!=48 or len(set(data.feature_names))!=48 or np.asarray(monthly_values).shape!=(len(data.X),5):
        raise ValueError('STOP frozen feature schema')
    a,names=build_features(data,rows,data.feature_names)
    b=np.column_stack([a,np.asarray(monthly_values[rows],np.float32)])
    keys=np.column_stack([data.date_idx[rows],data.stock_idx[rows]])
    validate_arms(keys,a,keys,b)
    return a,b,names+FEATURES


def run_fold(gate,root,data,dates,base_splits,labels,monthly_values,present,year,input_identity,calendar):
    """Deferred end-to-end fold driver; does not call the evaluation module.

    The complete original-day target is ranked before seed sampling. Inference
    eligibility depends only on matrix presence through t, never on labels or B.
    This phase's unconditional admission lock prevents all actual fits here.
    """
    require_admission(gate)
    from tools.step3 import fold_rows,sample_training,w60_eligible
    tr,ev=fold_rows(dates,base_splits,year,np.isfinite(labels),calendar=calendar)
    eligible=w60_eligible(present)[data.date_idx,data.stock_idx]
    ev=ev[eligible[ev]]
    if not len(ev):raise ValueError('STOP empty W60 inference support')
    feature_support(monthly_values[tr]);feature_support(monthly_values[ev])
    keys=np.column_stack([data.date_idx[ev],data.stock_idx[ev]])
    outputs={};xe_a,xe_b,_=build_arms(data,ev,monthly_values)
    for seed in SEEDS:
        rows,target=sample_training(tr,labels,data.date_idx,seed)
        xf_a,xf_b,_=build_arms(data,rows,monthly_values)
        for arm,xf,xe in [('A',xf_a,xe_a),('B',xf_b,xe_b)]:
            path=fit_job(gate,root,year,seed,arm,keys,rows,target,xf,xe,input_identity)
            outputs[arm,seed]=np.load(path/'keys.npy',allow_pickle=False)
    validate_outputs(outputs)
    return dict(fold=year,jobs=6,expected_keys=len(keys),status='ARTIFACTS_ONLY_NO_EVALUATION')
