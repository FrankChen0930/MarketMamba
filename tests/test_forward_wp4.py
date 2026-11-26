"""Only synthetic WP4 persistence and blind-boundary fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import numpy as np
import pytest
from tests.forward_tmp import tmp_path  # noqa: F401  (overrides builtin)

from tools import forward_wp4 as w


def test_three_seed_roundtrip_and_environment(tmp_path):
    root = tmp_path / 'store'
    result = w.synthetic_run(root)
    loaded = w.load_current(root, w.synthetic_context())
    assert set(loaded) == {17, 29, 43}
    for seed, (model, directory, manifest) in loaded.items():
        probe = np.load(directory / 'probe.npy', allow_pickle=False)
        expected = np.load(directory / 'prediction.npy', allow_pickle=False)
        assert model.predict(probe, num_threads=4).tobytes() == expected.tobytes()
        env = json.loads((directory / 'environment.json').read_text())
        assert env['lightgbm_version'] == '4.7.0'
        assert env['packages'] and env['cpu_flags'] and env['native_libraries']
        assert env['fit_threads'] == 4
        assert env['fit_layout']['shape'] == [800, 185]
        assert env['fit_layout']['c_contiguous'] is True
        assert manifest['request']['seed'] == seed
    assert result['roundtrip_equal'] == [True, True, True]


@pytest.mark.parametrize('key', ['matrix_sha256', 'contract_sha256', 'seeds'])
def test_identity_mismatch_rejected(tmp_path, key):
    root = tmp_path / 'store'; w.synthetic_run(root)
    ctx = w.synthetic_context()
    ctx[key] = [17, 29, 47] if key == 'seeds' else 'f' * 64
    with pytest.raises(w.Stop): w.load_current(root, ctx)


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'traversal', 'manifest', 'model', 'env', 'seed_swap'])
def test_corruption_does_not_fallback_or_promote_orphan(tmp_path, kind):
    root = tmp_path / 'store'; w.synthetic_run(root)
    pointer = json.loads((root / 'current.json').read_text())
    version = root / 'versions' / pointer['version']
    if kind == 'missing': (root / 'current.json').unlink()
    elif kind == 'corrupt': (root / 'current.json').write_text('{')
    elif kind == 'traversal':
        pointer['version'] = '../bad'; (root / 'current.json').write_text(json.dumps(pointer))
    elif kind == 'manifest': (version / 'manifest.json').write_text('{}')
    elif kind == 'model': (version / 'seed-17/model.txt').write_bytes(b'bad')
    elif kind == 'env': (version / 'seed-17/environment.json').write_text('{}')
    else:
        (version / 'seed-17/model.txt').write_bytes((version / 'seed-29/model.txt').read_bytes())
    before = (root / 'current.json').read_bytes() if (root / 'current.json').exists() else None
    with pytest.raises(w.Stop): w.load_current(root, w.synthetic_context())
    assert ((root / 'current.json').read_bytes() if (root / 'current.json').exists() else None) == before


@pytest.mark.parametrize('phase', ['local_verified', 'durable_verified', 'version_renamed', 'pointer_ready', 'pointer_committed'])
def test_real_sigkill_restart_never_promotes_orphan(tmp_path, phase):
    import time
    root = tmp_path / 'store'; w.synthetic_run(root)
    before = (root / 'current.json').read_bytes()
    ready = tmp_path / 'ready'
    script = '''
import os, signal, sys
from pathlib import Path
from tools import forward_wp4 as w
root, ready, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def checkpoint(value):
    if value == phase:
        w.write_sync(ready, value.encode())
        os.kill(os.getpid(), signal.SIGSTOP)
w._checkpoint = checkpoint
w.synthetic_run(root)
'''
    log = tmp_path / 'child.log'
    with log.open('wb') as output:
        proc = subprocess.Popen([sys.executable, '-c', script, str(root), str(ready), phase],
                                stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 30
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.read_text() == phase, log.read_text()
            proc.kill()
            assert proc.wait(timeout=10) == -signal.SIGKILL
        finally:
            if proc.poll() is None: proc.kill(); proc.wait(timeout=10)
    after = (root / 'current.json').read_bytes()
    assert (after != before) == (phase == 'pointer_committed')
    # Actual new interpreter, not in-process cached state. It must only read
    # the pointer, even when a newer, complete unreferenced version exists.
    script = '''
from pathlib import Path
import sys
from tools import forward_wp4 as w
root = Path(sys.argv[1]); w.load_current(root, w.synthetic_context())
w.write_sync(Path(sys.argv[2]), (root / 'current.json').read_bytes())
'''
    check = tmp_path / 'restart.json'
    restarted = subprocess.run([sys.executable, '-c', script, str(root), str(check)],
                               capture_output=True, timeout=30)
    assert restarted.returncode == 0, restarted.stderr.decode()
    assert check.read_bytes() == after
    assert list(root.glob('local-*'))  # retained, never reclaimed as latest/best


@pytest.mark.parametrize('kind', ['prediction', 'native_seed', 'seed_field', 'context', 'incomplete'])
def test_invalid_candidate_never_switches_pointer(tmp_path, kind):
    root = tmp_path / 'store'; w.synthetic_run(root)
    before = (root / 'current.json').read_bytes()
    trained = w.synthetic_train(tmp_path / ('fit-' + uuid.uuid4().hex))
    if kind == 'prediction': trained[17]['prediction'][0] += 1
    elif kind == 'native_seed': trained[17]['model_text'] = trained[29]['model_text']
    elif kind == 'seed_field': trained[17]['seed'] = 29
    elif kind == 'context': trained[17]['context'] = dict(trained[17]['context'], matrix_sha256='f'*64)
    else: trained.pop(43)
    with pytest.raises(w.Stop): w.publish_models(root, w.synthetic_context(), trained)
    assert (root / 'current.json').read_bytes() == before
    w.load_current(root, w.synthetic_context())


def test_output_paths_and_symlinks_rejected(tmp_path):
    for path in [w.ROOT, Path('/tmp/wp4-forbidden'), tmp_path / '..' / 'escape']:
        with pytest.raises(w.Stop): w.private_path(path)
    alias = tmp_path / 'alias'; alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(w.Stop): w.synthetic_run(alias / 'store')


def test_missing_pointer_with_complete_orphan_blocks_retraining(tmp_path):
    root = tmp_path / 'store'; w.synthetic_run(root)
    (root / 'current.json').unlink()
    count = len(list((root / 'versions').iterdir()))
    with pytest.raises(w.Stop): w.synthetic_run(root)
    assert not (root / 'current.json').exists()
    assert len(list((root / 'versions').iterdir())) == count


def test_durable_copy_corruption_never_publishes(tmp_path, monkeypatch):
    original = w.write_sync
    def corrupt(path, payload):
        if '.partial-' in str(path) and Path(path).name == 'model.txt': payload += b'corrupt'
        original(path, payload)
    monkeypatch.setattr(w, 'write_sync', corrupt)
    with pytest.raises(w.Stop): w.synthetic_run(tmp_path / 'store')
    assert not (tmp_path / 'store/current.json').exists()


def test_two_source_mutations_are_killed(tmp_path):
    import types
    from _pytest.outcomes import Failed
    source = Path(w.__file__).read_text()
    cases = [
        ('native_seed', "model.params['seed'] != seed", 'False'),
        ('roundtrip', 'prediction.tobytes() != expected.tobytes()', 'False'),
    ]
    killed = []
    for name, old, new in cases:
        assert source.count(old) == 1
        mutant = types.ModuleType('wp4_mutant'); mutant.__file__ = w.__file__
        exec(compile(source.replace(old, new), w.__file__, 'exec'), mutant.__dict__)
        trained = w.synthetic_train(tmp_path / ('fit-' + uuid.uuid4().hex))
        if name == 'native_seed':
            trained[17]['model_text'] = trained[29]['model_text']
            trained[17]['prediction'] = trained[29]['prediction']
        else: trained[17]['prediction'][0] += 1
        # The same behavioral assertion must PASS for real code and FAIL for
        # the mutated code. No source file is edited, no test is disabled.
        with pytest.raises(w.Stop):
            w.publish_models(tmp_path / ('real-' + name), w.synthetic_context(), trained)
        try:
            with pytest.raises(w.Stop):
                mutant.publish_models(tmp_path / ('mutant-' + name), w.synthetic_context(), trained)
        except Failed:
            killed.append(name)
    assert killed == ['native_seed', 'roundtrip']
    w.write_sync(tmp_path / 'mutations.json', w.canonical(dict(killed=killed, total=2)))


def test_explicit_cross_matrix_switch_preserves_old_immutable_version(tmp_path, monkeypatch):
    root = tmp_path / 'store'; w.synthetic_run(root)
    old = json.loads((root/'current.json').read_text())
    old_version = root/'versions'/old['version']
    inventory = {str(p.relative_to(old_version)):p.read_bytes() for p in old_version.rglob('*') if p.is_file()}
    old_context = w.synthetic_context()
    x, y, probe = w.synthetic_arrays()
    x = x.copy(); x[:, 0] += .125
    monkeypatch.setattr(w, 'synthetic_arrays', lambda: (x, y, probe))
    trained = w.synthetic_train(tmp_path / ('fit-' + uuid.uuid4().hex)); context = w.synthetic_context()
    assert context['matrix_sha256'] != old_context['matrix_sha256']
    with pytest.raises(w.Stop): w.publish_models(root, context, trained)
    assert json.loads((root/'current.json').read_text()) == old
    w.publish_models(root, context, trained, previous_context=old_context)
    pointer = json.loads((root/'current.json').read_text())
    assert pointer['previous'] == {k:old[k] for k in ('version','manifest_sha256')}
    assert pointer['version'] != old['version']
    assert inventory == {str(p.relative_to(old_version)):p.read_bytes() for p in old_version.rglob('*') if p.is_file()}
    w.load_current(root, context)
    with pytest.raises(w.Stop): w.load_current(root,old_context)


@pytest.mark.parametrize('field', ['cpu_flags','native_libraries','packages','fit_threads','fit_matrix_sha256'])
def test_incomplete_environment_or_matrix_binding_rejects_before_pointer(tmp_path, field):
    trained = w.synthetic_train(tmp_path / ('fit-' + uuid.uuid4().hex))
    trained[17]['environment'][field] = 0 if field == 'fit_threads' else 'f'*64 if field == 'fit_matrix_sha256' else []
    with pytest.raises(w.Stop): w.publish_models(tmp_path/'store', w.synthetic_context(),trained)
    assert not (tmp_path/'store/current.json').exists()


@pytest.mark.parametrize('kind', ['pointer', 'manifest', 'model', 'directory'])
def test_symlink_is_rejected_before_any_target_read(tmp_path, monkeypatch, kind):
    root = tmp_path/'store'; w.synthetic_run(root)
    pointer = json.loads((root/'current.json').read_text())
    version = root/'versions'/pointer['version']
    target = tmp_path/'unread-target'; target.write_text('MUST_NOT_READ')
    if kind == 'pointer': path = root/'current.json'
    elif kind == 'manifest': path = version/'manifest.json'
    elif kind == 'model': path = version/'seed-17/model.txt'
    else:
        old = version/'seed-17'; old.rename(version/'retained-seed-17')
        old.symlink_to(tmp_path, target_is_directory=True)
        path = None
    if path is not None:
        path.unlink(); path.symlink_to(target)
    original_read = Path.read_text
    original_hash = w.sha256
    def guarded_read(path, *args, **kwargs):
        assert not path.is_symlink(), 'followed symlink before validation'
        return original_read(path,*args,**kwargs)
    def guarded_hash(path):
        assert not Path(path).is_symlink(), 'hashed symlink target before validation'
        return original_hash(path)
    monkeypatch.setattr(Path,'read_text',guarded_read)
    monkeypatch.setattr(w,'sha256',guarded_hash)
    with pytest.raises(w.Stop): w.load_current(root,w.synthetic_context())


def test_native_environment_hash_cache_invalidates_changed_binary(tmp_path):
    binary = tmp_path/'synthetic-library.so'; binary.write_bytes(b'first')
    before = dict(w._native_hash(str(binary),w._native_stat(binary)))
    binary.write_bytes(b'other')  # same length; row count/size alone is inadequate
    after = w._native_hash(str(binary),w._native_stat(binary))
    assert before['bytes'] == after['bytes'] and before['sha256'] != after['sha256']


def test_first_publication_sigkill_leaves_unpromoted_orphan(tmp_path):
    root = tmp_path/'store'
    script = '''
import os, signal, sys
from pathlib import Path
from tools import forward_wp4 as w
def checkpoint(phase):
    if phase == 'version_renamed': os.kill(os.getpid(), signal.SIGKILL)
w._checkpoint = checkpoint
w.synthetic_run(Path(sys.argv[1]))
'''
    proc = subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True,timeout=30)
    assert proc.returncode == -signal.SIGKILL
    assert len(list((root/'versions').glob('version-*'))) == 1
    assert not (root/'current.json').exists()
    with pytest.raises(w.Stop): w.load_current(root,w.synthetic_context())
    with pytest.raises(w.Stop): w.synthetic_run(root)
    assert not (root/'current.json').exists()


def test_environment_is_fsynced_before_a_failed_fit(tmp_path, monkeypatch):
    from lightgbm import LGBMRegressor
    attempt = tmp_path/'fit-attempt'
    def fail_fit(*args, **kwargs):
        env = json.loads((attempt/'seed-17/environment.json').read_text())
        req = json.loads((attempt/'seed-17/request.json').read_text())
        assert env['lightgbm_version'] == '4.7.0'
        assert req['environment_sha256'] == w.sha256(attempt/'seed-17/environment.json')
        raise RuntimeError('synthetic interrupted fit')
    monkeypatch.setattr(LGBMRegressor,'fit',fail_fit)
    with pytest.raises(RuntimeError): w.synthetic_train(attempt)
    assert (attempt/'seed-17/environment.json').exists()
    assert not (attempt/'seed-29').exists()
