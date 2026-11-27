"""Step5 frozen CPU fits; no evaluation or label statistics.

RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED.
Durable primitives copied verbatim from step3_runner (Step3 bytes stay frozen).
The 5a admission is immutable: this additive executor is bound separately in
requests, never retroactively inserted into the admitted source-code glob.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import uuid
import numpy as np
from tools import step5 as s
from tools.step3 import sha256, verify_file

BUNDLE_SHA = 'f7d67e3dc7dfd7ea6b1eb7b7b076d07927f7927320a4f9deaae55edd502d1169'
ADMISSION_SHA = 'c873e4cfd0a3992b4aca73bdabd765b663bf2c14e89cce16c2227f33cc0ba3e8'
ADMITTED_CODE = {'tools/' + n for n in ('step3.py', 'step3_audit.py',
    'step3_evaluation.py', 'step3_prepare.py', 'step3_runner.py',
    'step4.py', 'step4_audit.py', 'step4_prepare.py',
    'step5.py', 'step5_features.py', 'step5_prepare.py', 'step5_audit.py')}


def private_dir(path):
    p = Path(path).absolute()
    base = s.OUTPUT_ROOT
    if p.resolve() != p or not any(p == base / n or p.is_relative_to(base / n)
                                 for n in ('fits', 'measurement')):
        raise ValueError('STOP private Step5 fits/measurement namespace')
    p.mkdir(parents=True, exist_ok=True)
    return p


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def array_digest(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


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
        if not {'keys.npy','sample.npy','prediction.npy','model.txt'}<=set(m['files']):raise ValueError('missing required outputs')
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



def queue():
    return [s.job_id(y, seed, arm) for y in s.YEARS for seed in s.SEEDS for arm in s.ARMS]


def require_admission(directory):
    """Consume the exact independently approved 5a evidence, not a new audit.

    Verify every admitted file and original implementation byte. Adding this
    executor does not change the old glob contract or grant a new admission.
    """
    directory = Path(directory).absolute()
    if directory.resolve() != directory:
        raise ValueError('STOP bundle symlink')
    verify_file(directory / 'admission.json', ADMISSION_SHA)
    gate = json.loads((directory / 'admission.json').read_text())
    verify_file(directory / 'bundle.json', BUNDLE_SHA, gate['bundle']['bytes'])
    if (gate['status'] != 'ADMIT' or gate['stops'] or gate['bundle']['sha256'] != BUNDLE_SHA
            or gate['contract_sha256'] != s.SPEC_SHA or gate['decision_sha256'] != s.DECISION_SHA):
        raise ValueError('STOP admission identity/status')
    bundle = json.loads((directory / 'bundle.json').read_text())
    if bundle['contract'] != s.contract() or set(bundle['jobs']) != set(queue()):
        raise ValueError('STOP frozen contract/queue')
    if set(bundle['implementation']) != ADMITTED_CODE:
        raise ValueError('STOP admitted implementation set')
    for path, info in bundle['implementation'].items():
        verify_file(s.ROOT / path, info['sha256'], info['bytes'])
    seen = set()
    for info in bundle['files'].values():
        path = s.file_path(directory, info)
        if path not in seen:
            verify_file(path, info['sha256'], info['bytes'])
            seen.add(path)
    return bundle


def executor_identity():
    paths = [Path(__file__), Path(__file__).with_name('run_step5_queue.py')]
    return {p.name: sha256(p) for p in paths}


def run_job(directory, root, name, bundle=None):
    """Verify exact job inputs before fit or resume; only fit targets are loaded.

    A supplied bundle is a queue-local cache of require_admission. Its content
    must still match the pinned bundle bytes before every job.
    """
    directory = Path(directory).absolute()
    if bundle is None:
        bundle = require_admission(directory)
    verify_file(directory / 'admission.json', ADMISSION_SHA)
    verify_file(directory / 'bundle.json', BUNDLE_SHA)
    if bundle != json.loads((directory / 'bundle.json').read_text()):
        raise ValueError('STOP cached bundle identity')
    if name not in queue() or name not in bundle['jobs']:
        raise ValueError('STOP queue job identity')
    job = bundle['jobs'][name]
    year, seed, arm = job['year'], job['seed'], job['arm']
    if (year not in s.YEARS or seed not in s.SEEDS or arm not in s.ARMS
            or s.job_id(year, seed, arm) != name):
        raise ValueError('STOP job identity')
    roles = ['sample_file', 'target_file', 'keys_file', 'fit_base', 'eval_base', 'fit_keys', 'eval_rows']
    if arm != 'A':
        roles += ['fit_extra', 'eval_extra']
    inputs = {}
    for role in roles:
        info = bundle['files'][job[role]]
        verify_file(s.file_path(directory, info), info['sha256'], info['bytes'])
        inputs[role] = dict(ref=job[role], sha256=info['sha256'], bytes=info['bytes'])
    load = lambda role: s.load(directory, bundle, job[role])
    rows, target, keys = load('sample_file'), load('target_file'), load('keys_file')
    base_fit, base_eval = load('fit_base'), load('eval_base')
    arrays = dict(sample=rows, target=target, keys=keys, fit=base_fit, eval=base_eval)
    if any(array_digest(a) != job['source_array_identities'][k] for k, a in arrays.items()):
        raise ValueError('STOP source array identity')
    x_fit, x_eval = s.matrix(directory, bundle, job, 'fit'), s.matrix(directory, bundle, job, 'eval')
    width = len(s.FEATURE_ORDER[arm])
    if (x_fit.shape != (len(rows), width) or x_eval.shape != (len(keys), width)
            or rows.ndim != 1 or not 0 < len(rows) <= s.SAMPLE_CAP
            or target.shape != (len(rows),) or not np.isfinite(target).all()
            or keys.shape != (len(x_eval), 2) or len(np.unique(keys, axis=0)) != len(keys)
            or np.isinf(x_fit).any() or np.isinf(x_eval).any()):
        raise ValueError('STOP matrix/key/fit target support')
    import lightgbm
    request = dict(contract=s.SPEC_SHA, decision=s.DECISION_SHA,
        bundle_sha256=BUNDLE_SHA, admission_sha256=ADMISSION_SHA, job=name,
        year=year, seed=seed, arm=arm, params=s.PARAMS, feature_order=list(s.FEATURE_ORDER[arm]),
        executor=executor_identity(), inputs=inputs,
        array_sha256={k: array_digest(v) for k, v in dict(sample=rows, target=target,
                        keys=keys, fit=x_fit, eval=x_eval).items()},
        environment=dict(python=platform.python_version(), executable=sys.executable,
            numpy=np.__version__, lightgbm=lightgbm.__version__, device='cpu'),
        evidence_class='HISTORICAL_RESEARCH_CPU_ENGINEERING_ONLY')
    root = private_dir(root)
    final = root / name
    if final.exists() or final.is_symlink():
        resume(final, request)
        return final
    start = time.monotonic()
    model = lightgbm.LGBMRegressor(**s.PARAMS, random_state=seed).fit(x_fit, target)
    fit_seconds = time.monotonic() - start
    predict_start = time.monotonic()
    prediction = np.asarray(model.predict(x_eval), np.float64)
    if prediction.shape != (len(keys),) or not np.isfinite(prediction).all():
        raise ValueError('STOP nonfinite/incomplete predictions')
    telemetry = dict(fit_seconds=fit_seconds, inference_seconds=time.monotonic()-predict_start,
        total_seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fit_matrix_bytes=x_fit.nbytes, eval_matrix_bytes=x_eval.nbytes,
        rows_fit=len(rows), rows_eval=len(keys))
    return publish(root, name, request, {'keys.npy':keys, 'sample.npy':rows, 'prediction.npy':prediction},
                   model.booster_.model_to_string().encode(), telemetry)
