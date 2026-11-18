"""P1 score export, E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Reuses c0plus.run exactly as rolling_runner does, including its default fitting
recipe and 60-session evaluation eligibility. No new matrix or cache is built.
Default dry run; each execution requires a fresh .artifacts/p1 namespace.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .backtest import file_hash
from .rolling_runner import SYSTEMS

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / '.artifacts' / 'p1'
MATRIX_FILES = ('manifest.json', 'X.npy', 'dates.npy', 'end_indices.npy',
                'splits.npy', 'stock_ids.npy', 'stock_idx.npy', 'y5.npy', 'y10.npy')


def output_path(path):
    root, out = ARTIFACT_ROOT.resolve(), Path(path).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError('output must be below .artifacts/p1/')
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'refusing overwrite: {out}')
    return out


def identity(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=file_hash(p))


def provenance():
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((Path(__file__).parents[1]/'experimental').glob('v7_*metrics*.py'))
    sources += [Path(__file__).parents[2]/'tasks/spec-portfolio-eval-20261119.md']
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                source_status='UNMERGED / NOT_DEPLOYED / uncommitted',
                implementation={str(p.relative_to(Path(__file__).parents[2])): file_hash(p) for p in sources},
                environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in ('numpy','pandas','lightgbm','pyarrow')},
                    device='CPU', MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT')))


def write_json(path, result):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rank_average(scores):
    keys = ['fold','system','date','code']
    s = scores.copy()
    if s.empty or s.duplicated(keys+['seed']).any() or not np.isfinite(s.score).all():
        raise ValueError('empty, duplicate or nonfinite scores')
    seeds = sorted(s.seed.unique())
    if not s.groupby(keys).seed.nunique().eq(len(seeds)).all():
        raise ValueError('incomplete seed support')
    # Require the identical row universe across both systems as well.
    if s.system.nunique() > 1 and not s.groupby(['fold','date','code','seed']).system.nunique().eq(s.system.nunique()).all():
        raise ValueError('incomplete system support')
    s['score'] = s.groupby(['fold','system','seed','date']).score.rank(method='average', pct=True)
    agg = dict(score='mean')
    if 'label_y5' in s:
        if not s.groupby(keys).label_y5.nunique(dropna=False).eq(1).all():
            raise ValueError('label differs across seeds')
        agg['label_y5'] = 'first'
    return s.groupby(keys, as_index=False).agg(agg).sort_values(keys).reset_index(drop=True)


def collect(fold_dir, seed, max_train_rows=500_000):
    from .seq_runner import SeqData
    from .c0plus import run
    # SeqData would otherwise write stock_idx.npy into external matrix storage.
    if not (Path(fold_dir)/'stock_idx.npy').is_file():
        raise ValueError('read-only matrix requires existing stock_idx.npy')
    started = time.perf_counter()
    data = SeqData(fold_dir, normalize=None)
    result = run(fold_dir, 'y5', max_train_rows, seed, n_boot=1,
                 row_mask=data.eligible, log=lambda *a, **k: None)
    ev = result['eval_idx']
    ids = np.load(Path(fold_dir)/'stock_ids.npy', mmap_mode='r', allow_pickle=False)
    dates = np.load(Path(fold_dir)/'dates.npy', mmap_mode='r', allow_pickle=False)
    frames = [pd.DataFrame(dict(code=np.asarray(ids[ev]), date=pd.to_datetime(dates[ev]),
                  score=result['scores'][system], fold=int(Path(fold_dir).name.removeprefix('fold_')),
                  seed=int(seed), system=system, label_y5=data.y['y5'][ev])) for system in SYSTEMS]
    return pd.concat(frames, ignore_index=True), dict(seed=int(seed), fit_rows=result['fit_rows'],
                eval_rows=result['eval_rows'], wall_seconds=time.perf_counter()-started)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds-root', type=Path, required=True)
    p.add_argument('--years', type=int, nargs='+', default=list(range(2021,2027)))
    p.add_argument('--seeds', type=int, nargs='+', default=[17,29,43])
    p.add_argument('--systems', choices=SYSTEMS, nargs='+', default=list(SYSTEMS))
    p.add_argument('--max-train-rows', type=int, default=500_000)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    if (len(set(a.years)) != len(a.years) or len(set(a.seeds)) != len(a.seeds)
            or len(set(a.systems)) != len(a.systems) or min(a.seeds) < 0 or a.max_train_rows < 1):
        p.error('years/seeds/systems must be unique; seeds nonnegative and row cap positive')
    out = output_path(a.out_dir)
    if not a.run:
        print(json.dumps(dict(dry_run=True, years=a.years, seeds=a.seeds, systems=a.systems,
                              fits=2*len(a.years)*len(a.seeds), output=str(out))))
        return
    started = time.perf_counter()
    prov = provenance()
    paths = [a.folds_root/f'fold_{year}'/name for year in a.years for name in MATRIX_FILES]
    inputs = {str(path.absolute()): identity(path) for path in paths}
    out.mkdir(parents=True, exist_ok=False)
    jobs, frames = [], []
    for year in a.years:
        for seed in a.seeds:
            frame, timing = collect(a.folds_root/f'fold_{year}', seed, a.max_train_rows)
            frame = frame[frame.system.isin(a.systems)]
            with (out/f'fold_{year}_seed_{seed}.parquet').open('xb') as stream:
                frame.to_parquet(stream, index=False)
            frames.append(frame)
            jobs.append(dict(fold=year, **timing))
            print(json.dumps(jobs[-1]), flush=True)
    ranks = rank_average(pd.concat(frames, ignore_index=True))
    with (out/'rank_average.parquet').open('xb') as stream:
        ranks.to_parquet(stream, index=False)
    if inputs != {str(path.absolute()): identity(path) for path in paths} or prov['implementation'] != provenance()['implementation']:
        raise RuntimeError('input/source changed during run; no completion manifest')
    manifest = dict(version='p1-scores-v1', artifact_class='GENERATED', evidence_class='E2 RESEARCH_ONLY',
        provenance=prov, inputs=inputs, years=a.years, seeds=a.seeds, systems=a.systems,
        parameters=dict(label='y5', max_train_rows=a.max_train_rows, eligibility_window=60,
                        fitting='c0plus.run defaults; same rolling_runner recipe', rank_method='average percentile per date then mean across seeds'),
        jobs=jobs, outputs={f.name: identity(f) for f in sorted(out.glob('*.parquet'))},
        wall_seconds=time.perf_counter()-started)
    write_json(out/'manifest.json', manifest)
    print(json.dumps(dict(manifest=str(out/'manifest.json'), rows=len(ranks), wall_seconds=manifest['wall_seconds'])))


if __name__ == '__main__':
    main()
