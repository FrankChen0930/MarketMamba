"""Synthetic Step5 durable CPU execution; no real bundles or labels."""
import json
from pathlib import Path
import numpy as np
import pytest
from tools import step5 as s
from tools import step5_runner as r


@pytest.fixture
def runner_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'OUTPUT_ROOT',tmp_path/'.artifacts/step5')
    monkeypatch.setattr(s,'PARAMS',dict(s.PARAMS,n_estimators=2,min_child_samples=1,verbosity=-1))
    directory=tmp_path/'bundle';directory.mkdir()
    files={};jobs={}
    def save(ref,a):
        p=directory/(ref+'.npy');np.save(p,a)
        files[ref]=dict(path=p.name,sha256=s.sha256(p),bytes=p.stat().st_size)
        return ref
    arrays=dict(sample=np.arange(6),target=np.arange(6,dtype=np.float32),
        keys=np.column_stack([np.zeros(24,dtype=int),np.arange(24)]),
        fit=np.arange(6*190,dtype=np.float32).reshape(6,190),
        eval=np.arange(24*190,dtype=np.float32).reshape(24,190),
        extra_fit=np.ones((6,4),np.float32),extra_eval=np.ones((24,4),np.float32),
        fit_keys=np.column_stack([np.zeros(6,dtype=int),np.arange(6)]),
        eval_rows=np.arange(24),labels=np.arange(24,dtype=float))
    refs={k:save(k,a) for k,a in arrays.items()}
    for y in s.YEARS:
        for seed in s.SEEDS:
            for arm in s.ARMS:
                jobs[s.job_id(y,seed,arm)]=dict(year=y,seed=seed,arm=arm,
                    sample_file=refs['sample'],target_file=refs['target'],keys_file=refs['keys'],
                    fit_base=refs['fit'],eval_base=refs['eval'],fit_extra=refs['extra_fit'],eval_extra=refs['extra_eval'],
                    fit_keys=refs['fit_keys'],eval_rows=refs['eval_rows'],eval_labels_file=refs['labels'],
                    source_array_identities={k:r.array_digest(arrays[k]) for k in ('sample','target','keys','fit','eval')})
    code=tmp_path/'tools/fixture.py';code.parent.mkdir();code.write_text('# synthetic source identity\n')
    info=dict(sha256=s.sha256(code),bytes=code.stat().st_size)
    monkeypatch.setattr(r,'ADMITTED_CODE',{'tools/fixture.py'})
    bundle=dict(contract=s.contract(),jobs=jobs,files=files,implementation={'tools/fixture.py':info})
    p=directory/'bundle.json';p.write_text(json.dumps(bundle))
    monkeypatch.setattr(r,'BUNDLE_SHA',s.sha256(p))
    gate=dict(status='ADMIT',stops=[],bundle=dict(sha256=r.BUNDLE_SHA,bytes=p.stat().st_size),contract_sha256=s.SPEC_SHA,decision_sha256=s.DECISION_SHA)
    p=directory/'admission.json';p.write_text(json.dumps(gate))
    monkeypatch.setattr(r,'ADMISSION_SHA',s.sha256(p))
    return r,directory,bundle


def test_runner_roundtrip_and_seed_identity(runner_bundle, monkeypatch):
    r, directory, bundle = runner_bundle
    gate = r.require_admission(directory)
    root = s.OUTPUT_ROOT / 'measurement'
    name = s.job_id(2015, 17, 'B')
    path = r.run_job(directory, root, name, gate)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['request']['arm'] == 'B'
    assert manifest['request']['admission_sha256'] == r.ADMISSION_SHA
    assert manifest['telemetry']['rows_fit'] == 6
    import lightgbm
    monkeypatch.setattr(lightgbm.LGBMRegressor, 'fit', lambda *a, **k: pytest.fail('resume fitted again'))
    assert r.run_job(directory, root, name, gate) == path
    (path / 'prediction.npy').unlink()
    with pytest.raises(ValueError, match='STOP'):
        r.run_job(directory, root, name, gate)


def assert_missing_job_rejected(resumer, path, request):
    with pytest.raises(ValueError, match='STOP'):
        resumer(path, request)


def test_mutation_resume_accepts_missing_job_killed(runner_bundle):
    r, directory, _ = runner_bundle
    gate = r.require_admission(directory)
    path = r.run_job(directory, s.OUTPUT_ROOT / 'measurement', s.job_id(2015, 17, 'A'), gate)
    manifest = json.loads((path / 'manifest.json').read_text())
    (path / 'model.txt').unlink()
    assert_missing_job_rejected(r.resume, path, manifest['request'])
    with pytest.raises(BaseException) as caught:
        assert_missing_job_rejected(lambda *a: manifest, path, manifest['request'])
    assert 'DID NOT RAISE' in str(caught.value)


def assert_identity_rejected(r, directory, root, gate):
    with pytest.raises(ValueError, match='STOP'):
        r.run_job(directory, root, s.job_id(2015, 17, 'A'), gate)


def test_mutation_job_identity_mismatch_killed(runner_bundle, monkeypatch):
    r, directory, _ = runner_bundle
    gate = r.require_admission(directory)
    gate['jobs'][s.job_id(2015, 17, 'A')]['seed'] = 29
    import lightgbm
    calls = []
    monkeypatch.setattr(lightgbm.LGBMRegressor, 'fit', lambda *a, **k: calls.append(1))
    root = s.OUTPUT_ROOT / 'measurement'
    assert_identity_rejected(r, directory, root, gate)
    assert calls == []
    monkeypatch.setattr(r, 'run_job', lambda *a: calls.append(1))
    with pytest.raises(BaseException) as caught:
        assert_identity_rejected(r, directory, root, gate)
    assert 'DID NOT RAISE' in str(caught.value)


def test_runner_never_loads_eval_labels(runner_bundle, monkeypatch):
    r, directory, bundle = runner_bundle
    gate = r.require_admission(directory)
    real_load = np.load
    forbidden = {s.file_path(directory, bundle['files'][j['eval_labels_file']]) for j in bundle['jobs'].values()}
    def guarded(path, *args, **kwargs):
        assert Path(path) not in forbidden, 'evaluation labels must remain opaque'
        return real_load(path, *args, **kwargs)
    from pathlib import Path
    monkeypatch.setattr(np, 'load', guarded)
    r.run_job(directory, s.OUTPUT_ROOT / 'measurement', s.job_id(2015, 17, 'B'), gate)


@pytest.mark.parametrize('mutation', ['input', 'admission', 'manifest', 'artifact', 'request'])
def test_runner_corruption_stops(runner_bundle, mutation):
    r, directory, bundle = runner_bundle
    gate = r.require_admission(directory)
    name = s.job_id(2015, 17, 'A')
    root = s.OUTPUT_ROOT / 'measurement'
    path = r.run_job(directory, root, name, gate)
    if mutation == 'input':
        target = s.file_path(directory, bundle['files'][bundle['jobs'][name]['fit_base']])
        with target.open('ab') as f:
            f.write(b'corrupt')
    elif mutation == 'admission':
        with (directory / 'admission.json').open('a') as f:
            f.write(' ')
    elif mutation == 'manifest':
        (path / 'manifest.json').write_text('{')
    elif mutation == 'artifact':
        (path / 'prediction.npy').write_bytes(b'bad')
    else:
        manifest = json.loads((path / 'manifest.json').read_text())
        manifest['request']['seed'] = 29
        (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='STOP'):
        r.run_job(directory, root, name, gate)


def test_runner_source_mutation_identity_gate_killed(runner_bundle, monkeypatch):
    """Mutate the actual guard: a wrong seed must never reach model.fit."""
    import inspect
    import lightgbm
    r, directory, bundle = runner_bundle
    gate = r.require_admission(directory)
    name = s.job_id(2015, 17, 'A')
    gate['jobs'][name]['seed'] = 29
    monkeypatch.setattr(s, 'SEEDS', (17, 29))
    (directory / 'bundle.json').write_text(json.dumps(gate))
    monkeypatch.setattr(r, 'BUNDLE_SHA', s.sha256(directory / 'bundle.json'))
    root = s.OUTPUT_ROOT / 'measurement'
    with pytest.raises(ValueError, match='STOP job identity'):
        r.run_job(directory, root, name, gate)
    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('MUTANT_REACHED_FIT')
    monkeypatch.setattr(lightgbm.LGBMRegressor, 'fit', forbidden_fit)
    source = inspect.getsource(r.run_job)
    changed = source.replace('or s.job_id(year, seed, arm) != name', 'or False')
    assert changed != source
    namespace = dict(r.__dict__)
    exec(compile(changed, '<identity-mutant>', 'exec'), namespace)
    with pytest.raises(RuntimeError, match='MUTANT_REACHED_FIT'):
        namespace['run_job'](directory, root, name, gate)


def test_runner_source_array_sha_mismatch_stop_and_mutant_killed(runner_bundle, monkeypatch):
    """Self-consistent file/bundle hashes cannot bypass the source-array pin."""
    import inspect
    import lightgbm
    r, directory, bundle = runner_bundle
    gate = r.require_admission(directory)
    name = s.job_id(2015, 17, 'A')
    gate['jobs'][name]['source_array_identities']['fit'] = '0' * 64
    (directory / 'bundle.json').write_text(json.dumps(gate))
    monkeypatch.setattr(r, 'BUNDLE_SHA', s.sha256(directory / 'bundle.json'))
    root = s.OUTPUT_ROOT / 'measurement'

    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('MUTANT_REACHED_FIT')

    monkeypatch.setattr(lightgbm.LGBMRegressor, 'fit', forbidden_fit)

    def oracle(run_job):
        with pytest.raises(ValueError, match='STOP source array identity'):
            run_job(directory, root, name, gate)

    oracle(r.run_job)
    source = inspect.getsource(r.run_job)
    changed = source.replace("if any(array_digest(a) != job['source_array_identities'][k] for k, a in arrays.items()):", 'if False:')
    assert changed != source
    namespace = dict(r.__dict__)
    exec(compile(changed, '<source-array-sha-mutant>', 'exec'), namespace)
    with pytest.raises(RuntimeError, match='MUTANT_REACHED_FIT'):
        oracle(namespace['run_job'])


def test_runner_source_mutation_missing_file_killed(runner_bundle):
    import inspect
    r, directory, _ = runner_bundle
    gate = r.require_admission(directory)
    path = r.run_job(directory, s.OUTPUT_ROOT / 'measurement', s.job_id(2015, 17, 'A'), gate)
    request = json.loads((path / 'manifest.json').read_text())['request']
    (path / 'prediction.npy').unlink()
    assert_missing_job_rejected(r.resume, path, request)
    source = inspect.getsource(r.resume)
    changed = source.replace("if actual!=set(m['files'])|{'manifest.json'}:", 'if False:')
    changed = changed.replace("verify_file(p/name,info['sha256'],info['bytes'])", 'pass')
    assert changed != source
    namespace = dict(r.__dict__)
    exec(compile(changed, '<resume-mutant>', 'exec'), namespace)
    # The same oracle rejects this source mutant because it accepts missing data.
    with pytest.raises(BaseException) as caught:
        assert_missing_job_rejected(namespace['resume'], path, request)
    assert 'DID NOT RAISE' in str(caught.value)


def test_runner_orphan_recovery_and_output_boundary(runner_bundle, monkeypatch):
    r, directory, _ = runner_bundle
    root = s.OUTPUT_ROOT / 'measurement'
    root.mkdir(parents=True)
    request = {'seed': 17}
    arrays = {'keys.npy': np.array([[1, 2]]), 'sample.npy': np.array([0]), 'prediction.npy': np.array([.1])}
    from pathlib import Path
    rename = Path.rename
    def interrupted(*args):
        raise InterruptedError('publish interruption after durable stage')
    monkeypatch.setattr(Path, 'rename', interrupted)
    with pytest.raises(InterruptedError):
        r.publish(root, 'synthetic', request, arrays, b'model')
    stage, = root.glob('.partial-*')
    assert not (root / 'synthetic').exists()
    monkeypatch.setattr(Path, 'rename', rename)
    final = r.recover_orphan(stage, root / 'synthetic', request)
    assert r.resume(final, request)['state'] == 'COMPLETE'
    with pytest.raises(ValueError, match='STOP'):
        r.recover_orphan(final, root / 'other', request)
    with pytest.raises(ValueError, match='STOP'):
        r.private_dir(directory)


def test_queue_72_and_resume_progress(runner_bundle, monkeypatch, capsys):
    from tools import run_step5_queue as q
    r, directory, _ = runner_bundle
    monkeypatch.setattr(r, 'require_admission', lambda path: {})
    calls = []
    monkeypatch.setattr(r, 'run_job', lambda d, out, name, b: calls.append(name))
    monkeypatch.setattr(s, 'YEARS', tuple(range(2015, 2027)))
    monkeypatch.setattr(s, 'SEEDS', (17, 29, 43))
    assert q.run_all(directory, s.ROOT / '.artifacts/step5/fits') == 72
    assert len(set(calls)) == 72
    assert calls[0] == 'fold2015-A-seed17' and calls[-1] == 'fold2026-B-seed43'
    assert len(capsys.readouterr().out.splitlines()) == 72
