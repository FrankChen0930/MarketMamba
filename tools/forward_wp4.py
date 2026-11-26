"""WP4 synthetic-only runner and durable three-seed store.

CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED.
No real-data loader, production path, scheduler or public private-state logger.
Persistence follows step3_runner publish/resume; its root is intentionally not
changed. Reuse its strict per-job validator, array identity and fsync primitives.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import fcntl
from functools import lru_cache
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

import numpy as np

from tools.forward_wp1 import Stop
from tools.forward_wp2 import PARAMS, SEEDS
from tools.step3_runner import _sync_dir, array_digest, digest_json, resume
from tools.step3 import sha256

ROOT = Path(__file__).resolve().parents[1] / '.artifacts/forward'
SPEC = Path(__file__).resolve().parents[1] / 'tasks/spec-forward-p2-primary-20261124.md'
HEX = re.compile(r'[0-9a-f]{64}')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def private_path(path):
    """Only this checkout's forward namespace; reject symlink ancestors."""
    p = Path(path).absolute()
    if p == ROOT or not p.is_relative_to(ROOT) or p.resolve() != p:
        raise Stop('WP4 output namespace')
    return p


def write_sync(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def file_info(path):
    return dict(sha256=sha256(path), bytes=Path(path).stat().st_size)


def _checkpoint(phase):
    """No-op fault-injection seam; no CLI or environment-controlled kill hook."""


@contextmanager
def locked(root):
    root = private_path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.lock'
    if lock.is_symlink(): raise Stop('WP4 lock symlink')
    with lock.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try: yield root
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_context(context):
    if (set(context) != {'schema', 'matrix_sha256', 'contract_sha256', 'seeds', 'synthetic'}
            or context['schema'] != 'forward-wp4-context-v1'
            or context['synthetic'] is not True or context['seeds'] != list(SEEDS)
            or any(not isinstance(context[k], str) or not HEX.fullmatch(context[k])
                   for k in ('matrix_sha256', 'contract_sha256'))):
        raise Stop('WP4 context/seed/matrix/contract')


@lru_cache(maxsize=512)
def _native_hash(path, stat_identity):
    # Large loaded binaries (including transitive runtime libraries) are stable
    # within a fit process. Every capture rechecks inode/size/mtime/ctime; changed
    # files miss this cache. Each fit still stores the FULL version/hash map.
    info = file_info(path)
    if _native_stat(path) != stat_identity: raise Stop('native library changed during hash')
    return info


def _native_stat(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def capture_environment(x):
    """Capture before EACH fit; metadata is private, not a replay guarantee."""
    import lightgbm
    from threadpoolctl import threadpool_info
    import lightgbm.basic
    cpu = Path('/proc/cpuinfo').read_text()
    flags = sorted(set(line.split(':', 1)[1].strip() for line in cpu.splitlines()
                       if line.startswith(('flags\t', 'Features\t'))))
    if not flags: raise Stop('CPU flags unavailable')
    config = io.StringIO()
    with redirect_stdout(config): np.show_config()
    pools = threadpool_info()
    native = {lightgbm.basic._LIB._name, sys.executable}
    native.update(p['filepath'] for p in pools if p.get('filepath'))
    # Include numpy extension binaries as well as BLAS/OpenMP/LightGBM.
    native.update(str(p) for p in Path(np.__file__).parent.rglob('*.so'))
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5] and Path(parts[5]).is_file():
            native.add(parts[5])
    return dict(schema='forward-fit-environment-v1', python=sys.version,
        executable=str(Path(sys.executable).resolve()), platform=platform.platform(),
        machine=platform.machine(), cpu_flags=flags, cpuinfo=cpu,
        python_compiler=platform.python_compiler(), libc=platform.libc_ver(),
        cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
        packages=sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
        lightgbm_version=lightgbm.__version__, fit_threads=PARAMS['n_jobs'],
        thread_environment={k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_SCHEDULE', 'OMP_PROC_BIND',
            'OMP_PLACES', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        threadpools=pools, numpy_config=config.getvalue(),
        native_libraries={str(Path(p).resolve()): dict(_native_hash(str(Path(p).resolve()), _native_stat(p))) for p in sorted(native)},
        fit_layout=dict(shape=list(x.shape), dtype=str(x.dtype), strides=list(x.strides),
                        c_contiguous=bool(x.flags.c_contiguous), f_contiguous=bool(x.flags.f_contiguous)),
        fit_matrix_sha256=array_digest(x))


def _validate_version(directory, context):
    """Validate all bytes before parsing native model text; no orphan search."""
    validate_context(context)
    p = private_path(directory)
    try:
        if any(f.is_symlink() for f in p.rglob('*')): raise ValueError('version symlink')
        m = json.loads((p / 'manifest.json').read_text())
        if (set(m) != {'schema', 'context', 'state', 'source', 'files'}
                or m['schema'] != 'forward-wp4-version-v1' or m['state'] != 'COMPLETE'
                or m['context'] != context): raise ValueError('version context')
        expected = {'manifest.json'} | {f'seed-{s}' for s in SEEDS}
        if {f.name for f in p.iterdir()} != expected: raise ValueError('version members')
        actual = {str(f.relative_to(p)): file_info(f) for f in p.rglob('*') if f.is_file()}
        actual.pop('manifest.json')
        if actual != m['files']: raise ValueError('version files')
        out = {}
        for seed in SEEDS:
            d = p / f'seed-{seed}'
            job = json.loads((d / 'manifest.json').read_text())
            request = job['request']
            env = json.loads((d / 'environment.json').read_text())
            if (env['schema'] != 'forward-fit-environment-v1' or not env['packages']
                    or not env['cpu_flags'] or not env['native_libraries']
                    or env['fit_threads'] != PARAMS['n_jobs']
                    or env['fit_matrix_sha256'] != context['matrix_sha256']
                    or request['inputs']['fit'] != context['matrix_sha256']):
                raise ValueError('environment/matrix identity')
            if (request['context'] != context or request['seed'] != seed
                    or request['parameters'] != dict(PARAMS, random_state=seed)
                    or request['environment_sha256'] != sha256(d / 'environment.json')):
                raise ValueError('seed identity')
            resume(d, request)
            out[seed] = (d, job)
        return m, out
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Stop('WP4 invalid version') from exc


def _read_current(root, context):
    root = private_path(root)
    try:
        pointer = root / 'current.json'
        if pointer.is_symlink(): raise ValueError('pointer symlink')
        p = json.loads(pointer.read_text())
        if (set(p) != {'schema', 'version', 'manifest_sha256', 'previous'}
                or p['schema'] != 'forward-wp4-pointer-v1'
                or not re.fullmatch(r'version-[0-9a-f]{32}', p['version'])
                or not isinstance(p['manifest_sha256'], str)
                or not HEX.fullmatch(p['manifest_sha256'])): raise ValueError('pointer schema')
        if p['previous'] is not None:
            old = p['previous']
            if (set(old) != {'version', 'manifest_sha256'}
                    or not re.fullmatch(r'version-[0-9a-f]{32}', old['version'])
                    or not HEX.fullmatch(old['manifest_sha256'])): raise ValueError('previous pointer')
        version = private_path(root / 'versions' / p['version'])
        if (version / 'manifest.json').is_symlink(): raise ValueError('manifest symlink')
        if sha256(version / 'manifest.json') != p['manifest_sha256']: raise ValueError('pointer hash')
        manifest, jobs = _validate_version(version, context)
        return p, manifest, jobs
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Stop('WP4 missing/corrupt/mismatched pointer; no fallback') from exc


def _load_jobs(jobs):
    from lightgbm import Booster
    result = {}
    for seed, (directory, job) in jobs.items():
        try:
            model = Booster(model_file=str(directory / 'model.txt'))
            if model.params['seed'] != seed or model.params['num_threads'] != PARAMS['n_jobs']:
                raise ValueError('native model seed/threads')
            probe = np.load(directory / 'probe.npy', allow_pickle=False)
            expected = np.load(directory / 'prediction.npy', allow_pickle=False)
            prediction = np.asarray(model.predict(probe, num_threads=4), dtype=np.float64)
            if (expected.dtype != np.float64 or not np.isfinite(prediction).all()
                    or prediction.shape != expected.shape or prediction.tobytes() != expected.tobytes()):
                raise ValueError('roundtrip')
        except Exception as exc:
            raise Stop('WP4 model roundtrip') from exc
        result[seed] = (model, directory, job)
    return result


def load_current(root, context):
    with locked(root):
        _, _, jobs = _read_current(root, context)
        return _load_jobs(jobs)


def publish_models(root, context, trained, *, previous_context=None):
    """Three-seed transaction. Retain partials and orphans; never auto-promote.

    serialize/fsync/hash locally -> durable copy/fsync/reverify -> version rename
    -> atomic pointer replace/fsync. A single flock spans old-pointer validation
    and commit. Old versions are retained, not overwritten or auto-fallbacks.
    """
    validate_context(context)
    if set(trained) != set(SEEDS): raise Stop('three seeds required')
    with locked(root) as root:
        private_path(root / 'versions')
        previous = None
        if (root / 'current.json').exists() or (root / 'current.json').is_symlink():
            previous, _, old_jobs = _read_current(root, previous_context or context)
            _load_jobs(old_jobs)
            previous = {k: previous[k] for k in ('version', 'manifest_sha256')}
        elif (root / 'versions').exists() and any((root / 'versions').iterdir()):
            raise Stop('missing pointer with existing versions; manual audit required')
        token = uuid.uuid4().hex
        local = root / ('local-' + token); local.mkdir()
        for seed in SEEDS:
            r = trained[seed]
            if r['seed'] != seed or r['context'] != context: raise Stop('fit identity mismatch')
            d = local / f'seed-{seed}'; d.mkdir()
            arrays = dict(keys=np.arange(len(r['probe']), dtype=np.int64), sample=r['sample'],
                          prediction=r['prediction'], probe=r['probe'])
            for name, value in arrays.items():
                with (d / f'{name}.npy').open('xb') as stream:
                    np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            write_sync(d / 'model.txt', r['model_text'].encode())
            write_sync(d / 'environment.json', canonical(r['environment']))
            request = dict(seed=seed, context=context, parameters=dict(PARAMS, random_state=seed),
                environment_sha256=sha256(d / 'environment.json'), inputs=r['inputs'])
            files = {p.name: file_info(p) for p in d.iterdir()}
            write_sync(d / 'manifest.json', canonical(dict(state='COMPLETE', request=request,
                       request_sha256=digest_json(request), files=files, telemetry=r['telemetry'])))
            _sync_dir(d); resume(d, request)
        files = {str(f.relative_to(local)): file_info(f) for f in local.rglob('*') if f.is_file()}
        source = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            implementation={str(p.relative_to(SPEC.parents[1])): file_info(p) for p in (
                Path(__file__).resolve(), SPEC, SPEC.parents[1]/'tools/step3_runner.py',
                SPEC.parents[1]/'tools/forward_wp2.py', SPEC.parents[1]/'tools/forward_wp1.py',
                SPEC.parents[1]/'tools/step3.py', SPEC.parents[1]/'tools/forward_wp4_summary.py')}, evidence_class='SYNTHETIC_CPU_ONLY')
        write_sync(local / 'manifest.json', canonical(dict(schema='forward-wp4-version-v1',
                   context=context, state='COMPLETE', source=source, files=files)))
        _sync_dir(local); _sync_dir(root)
        _validate_version(local, context); _checkpoint('local_verified')
        versions = root / 'versions'; versions.mkdir(exist_ok=True); _sync_dir(root)
        stage = versions / ('.partial-' + token); stage.mkdir()
        for p in local.rglob('*'):
            target = stage / p.relative_to(local)
            if p.is_dir(): target.mkdir()
            else: write_sync(target, p.read_bytes())
        for seed in SEEDS: _sync_dir(stage / f'seed-{seed}')
        _sync_dir(stage); _, durable_jobs = _validate_version(stage, context)
        _load_jobs(durable_jobs)
        # The durable copy must match the LOCAL manifest as well, not only itself.
        if (stage / 'manifest.json').read_bytes() != (local / 'manifest.json').read_bytes():
            raise Stop('durable manifest differs')
        _checkpoint('durable_verified')
        final = versions / ('version-' + token)
        stage.rename(final); _sync_dir(versions); _checkpoint('version_renamed')
        pointer = dict(schema='forward-wp4-pointer-v1', version=final.name,
            manifest_sha256=sha256(final / 'manifest.json'), previous=previous)
        temp = root / ('.pointer-' + token)
        write_sync(temp, canonical(pointer)); _checkpoint('pointer_ready')
        temp.replace(root / 'current.json'); _sync_dir(root); _checkpoint('pointer_committed')
        return pointer


def synthetic_context():
    x, _, _ = synthetic_arrays()
    return dict(schema='forward-wp4-context-v1', matrix_sha256=array_digest(x),
                contract_sha256=sha256(SPEC), seeds=list(SEEDS), synthetic=True)


def synthetic_arrays():
    rng = np.random.default_rng(74)
    x = rng.normal(size=(800, 185)).astype('float32')
    y = (.3 * x[:, 0] + rng.normal(size=800)).astype('float64')
    probe = rng.normal(size=(37, 185)).astype('float32')
    return x, y, probe


def synthetic_train(environment_directory):
    """Only generated arrays; unchanged WP2 params; no score/log stdout."""
    from lightgbm import LGBMRegressor
    attempt = private_path(environment_directory)
    attempt.mkdir(parents=True, exist_ok=False)
    x, y, probe = synthetic_arrays(); context = synthetic_context(); trained = {}
    for seed in SEEDS:
        environment = capture_environment(x)
        before_fit = attempt / f'seed-{seed}'; before_fit.mkdir()
        write_sync(before_fit / 'environment.json', canonical(environment))
        write_sync(before_fit / 'request.json', canonical(dict(context=context, seed=seed,
                   parameters=dict(PARAMS, random_state=seed), target_sha256=array_digest(y),
                   environment_sha256=sha256(before_fit / 'environment.json'))))
        _sync_dir(before_fit); _sync_dir(attempt); _sync_dir(attempt.parent)
        _checkpoint('fit_environment_written')
        started = time.monotonic()
        model = LGBMRegressor(**PARAMS, random_state=seed).fit(x, y)
        fit_seconds = time.monotonic() - started
        prediction_started = time.monotonic()
        prediction = np.asarray(model.predict(probe), dtype=np.float64)
        predict_seconds = time.monotonic() - prediction_started
        trained[seed] = dict(seed=seed, context=context, environment=environment,
            model_text=model.booster_.model_to_string(), probe=probe,
            prediction=prediction, telemetry=dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds),
            sample=np.arange(len(x), dtype=np.int64),
            inputs=dict(fit=array_digest(x), target=array_digest(y), probe=array_digest(probe),
                        split='SYNTHETIC_ALL_TRAIN_800_PROBE_37', sampling='SYNTHETIC_NO_SAMPLE'))
    return trained


def synthetic_run(root):
    root = private_path(root)
    # Diagnose damaged state before paying even a synthetic refit cost.
    with locked(root):
        private_path(root / 'versions')
        if (root / 'current.json').exists() or (root / 'current.json').is_symlink():
            _read_current(root, synthetic_context())
        elif (root / 'versions').exists() and any((root / 'versions').iterdir()):
            raise Stop('missing pointer with existing versions; manual audit required')
    trained = synthetic_train(root / ('fit-attempt-' + uuid.uuid4().hex))
    publish_models(root, synthetic_context(), trained)
    loaded = load_current(root, synthetic_context())
    return dict(schema='forward-wp4-synthetic-check-v1', seeds=list(SEEDS),
                roundtrip_equal=[loaded[s][0].predict(trained[s]['probe'], num_threads=4).tobytes()
                                 == trained[s]['prediction'].tobytes() for s in SEEDS],
                evidence_class='SYNTHETIC_CPU_ONLY', forward_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-out', type=Path, required=True)
    args = parser.parse_args()
    # No public stdout: even failures are a private, generic incident.
    from tools.forward_wp4_summary import run_synthetic_session
    run_synthetic_session(args.synthetic_out)
