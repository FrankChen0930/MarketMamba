"""Thin, resumable Colab/CLI runner for matrix baselines (B4). RESEARCH_ONLY.

Orchestration only: identity checks, local staging, memory plan, per-model caching, result files.
All model logic lives in matrix_baselines.py; significance in significance.py.
  python -m research_lab.colab_runner --source DRIVE_MATRIX_DIR --expect E5 --label y5 --out RESULTS_ROOT
Outputs under RESULTS_ROOT/<EXP>_<label>_<id12>/ ; an existing complete result is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import matrix_baselines as mb
from .significance import compare_models

# Identities taken from the two manifests read on 2026-11-15 (manifest.json on Drive).
_COMMON = {
    'X.npy': '67358805f50b0dafaf70c29971def766191f537042cd1dc48518e2f45df18aed',
    'dates.npy': 'b0632562242f107855770742762f15fa1f82b6d8e012dce8e4cfd1d68c641016',
    'end_indices.npy': 'dd6c6e8e8e4c44ad28f42e9dd266f87c3baebbdc0f986e9aa6865789c9d3516f',
    'stock_ids.npy': '8d5cc83037d2470f051d8ff40e5e7c8edb2d436e0db195dea78df5fca8567c21',   # identical in both manifests
}
REGISTRY = {
    'E5': dict(rows=7423927, logical_identity='f6f00cef79abf0c47b597c6c2bcef67a6ce44e483b83864629b967c145fc17d4',
               schema='v7-corrected-e5-matrix-v1', train_end='2023-11-17', eval_start='2024-01-02',
               files={**_COMMON, 'splits.npy': 'de2941fe18cac4d1c87a3a7fc77d8fe88b47c6b006c26f85384993234b275d24',
                      'y5.npy': '21ba2cc6b1c39832ce55e47972bedd650b3bb51e3a0c5556fc67d988c806b503',
                      'y10.npy': 'b907a005fd7d708fcb66606aae0982098c7d65441b0aab16d9d4c94d3d4a3c5a'}),
    'E1': dict(rows=7423927, logical_identity='a590b280f0700dcf36fde9ac51e767d735cc2737b613cbb3b49ef3f472947f62',
               schema='v7-e1-matrix-v1', train_end='2025-11-18', eval_start='2026-01-02',
               files={**_COMMON, 'splits.npy': 'd56aa8568d478be42b1337f1db6f004051e9932ed5a898ab28cf1c2d42c9f8fa',
                      'y5.npy': '4f3bfd94346ce2abf9c904a8882d8dad8fac6b77f3e3720c9415c5661e4d6cda',
                      'y10.npy': 'f098008c1f100ab594c46e74a8f20fe1981d788813a4d86654c6d36a9f826716'}),
}
MODELS = ('reversal_ret5', 'momentum_ret20', 'lowvol_20d', 'ridge', 'lightgbm')
CODE_FILES = ('matrix_baselines.py', 'significance.py', 'colab_runner.py')


class IdentityError(RuntimeError):
    pass


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def code_identity():
    here = Path(__file__).parent
    return {n: sha256_file(here/n) for n in CODE_FILES}


def needed_files(label, extra=()):
    return tuple(dict.fromkeys(('X.npy', f'{label}.npy', 'dates.npy', 'splits.npy', 'end_indices.npy', *extra)))


def find_matrix(expect, roots, registry=REGISTRY, max_depth=6, log=print):
    """Locate the matrix directory for `expect` under `roots` by manifest identity, not by guessed path.
    Candidates are any directory holding manifest.json + X.npy; the first whose manifest passes check_manifest wins."""
    seen = []
    for root in map(Path, roots):
        if not root.exists():
            continue
        base = len(root.parts)
        for cur, dirs, files in os.walk(root):
            if len(Path(cur).parts)-base >= max_depth:
                dirs[:] = []
            if 'manifest.json' in files and 'X.npy' in files:
                try:
                    check_manifest(json.loads((Path(cur)/'manifest.json').read_text()), expect, registry)
                    return Path(cur)
                except (IdentityError, ValueError, OSError) as e:
                    seen.append(f'{cur}: {e}')
    raise FileNotFoundError(f'no {expect} matrix found under {[str(r) for r in roots]}; candidates rejected: {seen[:5]}')


def check_manifest(manifest, expect, registry=REGISTRY):
    """Cheap pre-copy check: the manifest must be exactly the expected experiment's manifest."""
    if expect not in registry:
        raise IdentityError(f'unknown experiment {expect}')
    r = registry[expect]
    problems = []
    if manifest.get('logical_identity') != r['logical_identity']:
        problems.append('logical_identity')
    if manifest.get('schema_version') != r['schema']:
        problems.append('schema_version')
    if manifest.get('rows') != r['rows']:
        problems.append('rows')
    for name, h in r['files'].items():
        if manifest.get('files', {}).get(name) != h:
            problems.append(f'manifest hash {name}')
    if problems:
        raise IdentityError(f'manifest is not {expect}: ' + ', '.join(problems))


def stage_and_verify(source, dest, label, expect, registry=REGISTRY, copy=True, extra=()):
    """Copy needed arrays to local disk while hashing the bytes read; fail on any mismatch."""
    source, dest = Path(source), Path(dest)
    manifest = json.loads((source/'manifest.json').read_text())
    check_manifest(manifest, expect, registry)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'manifest.json', dest/'manifest.json')
    hashes = {}
    for name in needed_files(label, extra):
        want = registry[expect]['files'][name]
        src, dst = source/name, dest/name
        if copy and not (dst.exists() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == want):
            tmp = dst.with_suffix('.part')
            h = hashlib.sha256()
            with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
                for b in iter(lambda: fi.read(1 << 24), b''):
                    h.update(b)
                    fo.write(b)
            if h.hexdigest() != want:
                tmp.unlink()
                raise IdentityError(f'{name}: sha256 {h.hexdigest()[:12]} != expected {want[:12]} ({expect})')
            os.replace(tmp, dst)
        elif not copy:
            if sha256_file(src) != want:
                raise IdentityError(f'{name}: sha256 mismatch for {expect}')
        hashes[name] = want
    return dest if copy else source, hashes


def check_semantics(directory, label, expect, registry=REGISTRY):
    """Array-level checks that hashes alone do not give: shapes, ordering, split/date contract."""
    r = registry[expect]
    manifest, a = mb.load_matrix(directory, labels=(label,))
    n = r['rows']
    if a['X'].shape != (n, len(manifest['feature_order'])) or a['X'].dtype != np.float32:
        raise IdentityError('X shape/dtype mismatch')
    ends = np.asarray(a['end_indices'])
    dates = np.asarray(a['dates'])
    seg = dates[ends-1]
    if (np.diff(seg.astype('int64')) <= 0).any():
        raise IdentityError('dates are not strictly increasing across sessions')
    starts = np.concatenate(([0], ends[:-1]))
    if not (dates[starts] == seg).all():
        raise IdentityError('date changes inside a session segment')
    splits = np.asarray(a['splits'])
    train_d, eval_d = dates[splits == mb.TRAIN], dates[splits == mb.EVAL]
    if not len(train_d) or not len(eval_d):
        raise IdentityError('empty train or evaluation split')
    if train_d.max() > np.datetime64(r['train_end']):
        raise IdentityError(f'train rows after {r["train_end"]}')
    if eval_d.min() < np.datetime64(r['eval_start']):
        raise IdentityError(f'evaluation rows before {r["eval_start"]}')
    if train_d.max() >= eval_d.min():
        raise IdentityError('train overlaps evaluation')
    y = np.asarray(a[label])
    if not np.isfinite(y[splits == mb.EVAL]).any() or not np.isfinite(y[splits == mb.TRAIN]).any():
        raise IdentityError('label has no finite values in train or evaluation')
    return dict(train=[str(train_d.min()), str(train_d.max())], evaluation=[str(eval_d.min()), str(eval_d.max())],
                train_rows=int((splits == mb.TRAIN).sum()), eval_rows=int((splits == mb.EVAL).sum()),
                eval_label_finite_fraction=float(np.isfinite(y[splits == mb.EVAL]).mean()))


def memory_plan(directory, label, max_train_rows, available_bytes=None):
    manifest, a = mb.load_matrix(directory, labels=(label,))
    splits = np.asarray(a['splits'])
    y = np.asarray(a[label])
    n_train = int(((splits == mb.TRAIN) & np.isfinite(y)).sum())
    n_eval = int((splits == mb.EVAL).sum())
    f = len(manifest['feature_order'])
    need = (min(n_train, max_train_rows)+n_eval)*f*4 + len(y)*4 + len(splits) + 4*n_eval  # +scores
    need *= 1.5  # headroom for sklearn/lightgbm working copies
    if available_bytes is None:
        try:
            available_bytes = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_AVPHYS_PAGES')
        except (ValueError, OSError):
            available_bytes = None
    ok = available_bytes is None or need < 0.6*available_bytes
    return dict(estimated_bytes=int(need), available_bytes=available_bytes, ok=bool(ok))


def experiment_id(expect, label, config, code, registry=REGISTRY):
    r = registry[expect]
    payload = dict(identity=r['logical_identity'], y=r['files'][f'{label}.npy'], splits=r['files']['splits.npy'],
                   label=label, config=config, code={k: code[k] for k in ('matrix_baselines.py', 'significance.py')})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def environment():
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               cpus=os.cpu_count(), platform=platform.platform())
    for mod in ('sklearn', 'lightgbm'):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    return env


def run_experiment(source, out_root, expect, label='y5', smoke=False, seed=17, max_train_rows=500_000,
                   n_boot=1000, stage_dir=None, registry=REGISTRY, models=MODELS, params=None, log=print, fit_target='rank'):
    cfg = dict(seed=seed, max_train_rows=50_000 if smoke else max_train_rows, n_boot=200 if smoke else n_boot,
               params={**mb.DEFAULT_PARAMS, **(dict(lgbm_estimators=50) if smoke else {}), **(params or {})},
               models=list(models), smoke=bool(smoke), fit_target=fit_target)
    code = code_identity()
    eid = experiment_id(expect, label, cfg, code, registry)
    root = Path(out_root)/f'{expect}_{label}{"_smoke" if smoke else ""}_{eid}'
    final = root/'result.json'
    if final.exists():
        log(f'[skip] complete result exists: {final}')
        return json.loads(final.read_text())
    (root/'models').mkdir(parents=True, exist_ok=True)
    (root/'daily_ic').mkdir(exist_ok=True)
    t0 = time.time()
    local, hashes = stage_and_verify(source, stage_dir or root/'_stage', label, expect, registry)
    log(f'[ok] identity verified for {expect}; files hashed: {sorted(hashes)}')
    semantics = check_semantics(local, label, expect, registry)
    plan = memory_plan(local, label, cfg['max_train_rows'])
    log(f'[plan] estimated {plan["estimated_bytes"]/1e9:.2f} GB, available {plan["available_bytes"]}')
    if not plan['ok']:
        raise MemoryError(f'estimated need {plan["estimated_bytes"]} exceeds 60% of available memory')
    atomic_write(root/'identity.json', json.dumps(dict(experiment=expect, label=label, id=eid, manifest_logical_identity=
                 registry[expect]['logical_identity'], file_sha256=hashes, semantics=semantics), indent=2))
    atomic_write(root/'config.json', json.dumps(dict(config=cfg, code=code, environment=environment()), indent=2))
    ctx = mb.Context(local, label, cfg['max_train_rows'], seed, fit_target)
    daily = {}
    model_info = {}
    for m in models:
        done, csv = root/'models'/f'{m}.json', root/'daily_ic'/f'{m}.csv'
        if done.exists() and csv.exists():
            log(f'[resume] {m} already complete')
            daily[m] = pd.read_csv(csv, parse_dates=['date']).set_index('date').ic
            model_info[m] = json.loads(done.read_text())
            continue
        s = time.time()
        ic = mb.model_daily_ic(ctx, mb.compute_scores(ctx, m, cfg['params']))
        tmp = ic.rename('ic').rename_axis('date').reset_index()
        atomic_write(csv, tmp.to_csv(index=False))
        info = dict(model=m, **mb.summarize_daily(ic), seconds=time.time()-s, fit_rows=int(len(ctx.fit_idx)) if m in ('ridge', 'lightgbm') else 0)
        atomic_write(done, json.dumps(info, indent=2))
        daily[m], model_info[m] = ic, info
        log(f'[done] {m}: IC {info["ic_mean"]:+.4f} over {info["days"]} sessions ({info["seconds"]:.0f}s)')
    common = pd.concat(daily, axis=1).dropna().index
    aligned = {k: v.loc[common] for k, v in daily.items()}
    sig = compare_models(aligned, 'reversal_ret5', ctx.horizon, n_boot=cfg['n_boot'], seed=seed) if len(common) >= 20 else []
    sig_gbm = compare_models({k: aligned[k] for k in ('lightgbm', 'ridge') if k in aligned}, 'ridge', ctx.horizon,
                             n_boot=cfg['n_boot'], seed=seed) if {'lightgbm', 'ridge'} <= set(aligned) and len(common) >= 20 else []
    result = dict(experiment=expect, label=label, id=eid, smoke=bool(smoke), seed=seed, horizon=ctx.horizon,
                  label_evidence='HISTORICAL_SIMULATION_PROXY', evidence_tier='RESEARCH_ONLY',
                  semantics=semantics, fit_rows=int(len(ctx.fit_idx)), train_rows_available=ctx.train_rows_available,
                  eval_rows=int(len(ctx.eval_idx)), common_sessions=int(len(common)),
                  models=model_info, vs_reversal_ret5=sig, lightgbm_vs_ridge=sig_gbm, memory_plan=plan,
                  fit_target=fit_target, file_sha256=hashes, code=code, wall_seconds=time.time()-t0,
                  caveats=['IC is comparable only within this experiment (same X/label/split)',
                           'research-used evaluation window; not out-of-sample for architecture selection',
                           'Holm correction is per family (5 vs reversal_ret5; lightgbm vs ridge separately)',
                           'tabular single-day features; not a Mamba/Samba result'])
    atomic_write(final, json.dumps(result, indent=2, allow_nan=False))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path, help='matrix directory (Drive, read-only)')
    p.add_argument('--expect', required=True, choices=sorted(REGISTRY))
    p.add_argument('--label', choices=['y5', 'y10'], default='y5')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--stage-dir', type=Path, help='local SSD staging dir (default: under result dir)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-train-rows', type=int, default=500_000)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--fit-target', choices=['raw', 'demean', 'rank'], default='rank')
    a = p.parse_args(argv)
    r = run_experiment(a.source, a.out, a.expect, a.label, a.smoke, a.seed, a.max_train_rows, a.n_boot, a.stage_dir, fit_target=a.fit_target)
    for m, i in r['models'].items():
        print(f'{m:15s} IC {i["ic_mean"]:+.4f}  days {i["days"]}')
    for s in r['vs_reversal_ret5']:
        print(f'{s["model"]:15s} vs reversal  dIC {s["mean_diff"]:+.4f}  p_holm {s["p_hac_holm"]:.3f}')


if __name__ == '__main__':
    main()
