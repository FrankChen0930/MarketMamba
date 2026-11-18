"""Roadmap §12.39 H3; E2 / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Evaluation-only known-disposal exclusion, with unchanged H1 training/recipe.
False disposal membership is not verified tradability. Default dry run; no GPU,
network or production writes. One writer per output directory. Resume checks
readability only, as H1 does; use a distinct namespace for changed input identity.
H3 is a robustness check, never an independent adoption claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import rolling_analysis as analysis
from . import rolling_runner as rr
from .disposal import STATUS, disposal_known
from .seq_runner import SeqData

YEARS = tuple(range(2021, 2027))
SEEDS = (17, 29, 43, 71, 97)


def run_fold_seed(fold_dir, seed, status_df, label='y5', max_train_rows=500_000,
                  n_boot=50, run_fn=None):
    """Return H1-format daily IC plus eligible known-disposal exclusion count.

    Existing stock_idx is mandatory to avoid creating caches in input matrices.
    Dates in SeqData are session-level; date_idx expands them to matrix rows.
    excluded_rows counts all eligible matrix rows, not only evaluation splits.
    """
    fold_dir = Path(fold_dir)
    if not (fold_dir/'stock_idx.npy').is_file():
        raise ValueError('stock_idx.npy required: H3 must not create matrix caches')
    if run_fn is None:
        from .c0plus import run
        run_fn = run
    started = time.perf_counter()
    data = SeqData(fold_dir, normalize=None)
    first = np.full(data.n_stocks, len(data.stock_idx), dtype=np.int64)
    np.minimum.at(first, data.stock_idx, np.arange(len(data.stock_idx)))
    if (first == len(data.stock_idx)).any():
        raise ValueError('noncontiguous stock_idx')
    ids = np.load(fold_dir/'stock_ids.npy', mmap_mode='r')[first]
    known = disposal_known(status_df, data.dates[data.date_idx], ids[data.stock_idx])
    row_mask = data.eligible & ~known
    result = run_fn(fold_dir, label, max_train_rows, seed, n_boot=n_boot,
                    row_mask=row_mask, log=lambda *args, **kwargs: None)
    return dict(fold=fold_dir.name, seed=int(seed), label=label,
                wall_seconds=time.perf_counter()-started,
                fit_rows=int(result['fit_rows']), eval_rows=int(result['eval_rows']),
                excluded_rows=int(np.count_nonzero(data.eligible & known)),
                daily_ic={name: {str(date): float(ic) for date, ic in result['daily_ic'][name].items()}
                          for name in rr.SYSTEMS})


def _output_dir(out_dir):
    out = rr._output_dir(out_dir)
    h1 = (rr.LOCAL_DATA_ROOT/'rolling_h1').resolve()
    if out == h1 or out.is_relative_to(h1) or h1.is_relative_to(out):
        raise ValueError('out_dir must be independent of rolling_h1')
    return out


def run_all(folds_root, out_dir, status_path=STATUS, years=YEARS, seeds=SEEDS,
            dry_run=True):
    """Fixed H1 recipe, readable JSON resume and shared atomic-write helper.

    Dry run neither reads status/matrices nor creates output directories.
    Preflight every existing JSON before training or writing any output.
    """
    out = _output_dir(out_dir)
    jobs = [(y, s, out/f'fold_{y}_seed_{s}.json') for y in years for s in seeds]
    completed = {(y, s) for y, s, path in jobs if rr._completed(path)}
    status = None
    for year, seed, path in jobs:
        if (year, seed) in completed:
            print(f'SKIP fold_{year} seed={seed} {path}', flush=True)
        elif dry_run:
            print(f'RUN fold_{year} seed={seed} {path}', flush=True)
        else:
            if status is None:
                status = pd.read_parquet(status_path)
            result = run_fold_seed(Path(folds_root)/f'fold_{year}', seed, status)
            out.mkdir(parents=True, exist_ok=True)
            rr._atomic_json(path, result)
            print(f'DONE fold_{year} seed={seed} wall_seconds={result["wall_seconds"]:.3f}', flush=True)
    if dry_run:
        print(f'DRY_RUN total={len(jobs)} pending={len(jobs)-len(completed)} skipped={len(completed)}', flush=True)


def analyze(h1_dir, h3_dir, n_boot=2000, seed=0):
    """Full H1 statistics and preregistered H3 decision; read-only, call once.

    Recompute H1 point estimate instead of using the rounded roadmap figure.
    Absolute C0+ IC differences use paired dates after averaging the five seeds
    within each source; report support and means descriptively, without tests.
    """
    h1, h3 = analysis.load_results(h1_dir), analysis.load_results(h3_dir)
    required = {(y, s) for y in YEARS for s in SEEDS}
    for records in (h1, h3):
        if {(y, s) for y, fold in records.items() for s in fold} != required:
            raise ValueError('H3 requires all 30 registered fold/seed pairs in both sources')
    report = analysis.report(h3_dir, n_boot=n_boot, seed=seed)
    reference = analysis.report(h1_dir, n_boot=n_boot, seed=seed)
    point, h1_point = report['clean']['hac']['mean'], reference['clean']['hac']['mean']
    half = h1_point/2
    retained = point >= half
    report['preregistration'] = 'roadmap 12.39 H3'
    report['h3'] = dict(clean_classification=report['classification'], point=point,
                        h1_point=h1_point, half_h1=half, at_least_half_h1=retained,
                        verdict='穩健' if report['classification'] == '證實有益' and retained
                        else '對處置宇宙敏感')
    report['h1_inputs'] = reference['inputs']
    differences = {}
    for year in YEARS:
        paired = pd.concat({'h1': analysis.system_daily(h1, 'c0plus', year),
                            'h3': analysis.system_daily(h3, 'c0plus', year)},
                           axis=1, join='inner').dropna()
        if paired.empty:
            raise ValueError(f'no common H1/H3 C0+ days: {year}')
        differences[str(year)] = dict(h1_ic=float(paired.h1.mean()), h3_ic=float(paired.h3.mean()),
                                      difference=float((paired.h3-paired.h1).mean()),
                                      n=len(paired), descriptive_only=True)
    report['absolute_ic_difference'] = differences
    report['limitations'][1:2] = [
        'H3 robustness check only; no new alpha budget or independent adoption claim.',
        'Only known disposal excluded; remaining rows are not verified tradable.',
        'Three clean folds; confidence intervals do not cover training randomness.']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds-root', type=Path, default=rr.LOCAL_DATA_ROOT/'matrices/rolling')
    parser.add_argument('--out-dir', type=Path, default=rr.LOCAL_DATA_ROOT/'rolling_h3')
    parser.add_argument('--status-path', type=Path, default=STATUS)
    parser.add_argument('--run', action='store_true', help='execute registered jobs (default: dry run)')
    args = parser.parse_args(argv)
    run_all(args.folds_root, args.out_dir, args.status_path, dry_run=not args.run)


if __name__ == '__main__':
    main()
