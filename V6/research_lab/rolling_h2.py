"""Roadmap §12.36 H2 collection; E2 / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Default dry run. One writer per output directory; resume validates payload but
not full training/input provenance. Use a distinct directory for changed inputs.
No GPU, network, production writes or adoption authorization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import rolling_analysis as analysis
from . import rolling_runner as rr
from .fundamentals import make_extra
from .seq_runner import SeqData

YEARS = tuple(range(2021, 2027))
SEEDS = (17, 29, 43, 71, 97)
DATA_DIR = Path('/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6')


def _load_financials(data_dir):
    root = Path(data_dir)
    return (pd.read_parquet(root/'financials_raw.parquet'),
            pd.read_parquet(root/'stock_info.parquet'))


def run_fold_seed(fold_dir, seed, financials_long, stock_info,
                  max_train_rows=500_000, n_boot=50, run_fn=None):
    """Read-only matrices; return only the augmented C0+ daily IC, without writes.

    Training row cap/bootstrap overrides are for smoke tests only. Formal run_all
    always uses the registered 500,000/50 configuration.
    """
    if run_fn is None:
        from .c0plus import run
        run_fn = run
    fold_dir = Path(fold_dir)
    if not (fold_dir/'stock_idx.npy').is_file():
        raise ValueError('stock_idx.npy required: H2 must not create matrix caches')
    started = time.perf_counter()
    data = SeqData(fold_dir, normalize=None)
    first = np.full(data.n_stocks, len(data.stock_idx), dtype=np.int64)
    np.minimum.at(first, data.stock_idx, np.arange(len(data.stock_idx)))
    if (first == len(data.stock_idx)).any():
        raise ValueError('noncontiguous stock_idx')
    ids = np.load(fold_dir/'stock_ids.npy', mmap_mode='r')[first]
    extra = make_extra(financials_long, stock_info, data.dates, ids)
    result = run_fn(fold_dir, 'y5', max_train_rows, seed, n_boot=n_boot,
                    row_mask=data.eligible, extra=extra, log=lambda *args, **kwargs: None)
    return dict(fold=fold_dir.name, seed=int(seed), label='y5',
                wall_seconds=time.perf_counter()-started,
                fit_rows=int(result['fit_rows']), eval_rows=int(result['eval_rows']),
                daily_ic={'c0plus_fund': result['daily_ic']['c0plus']})


def _completed(path, year, seed):
    if not rr._completed(path):
        return False
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise ValueError(f'invalid H2 record: {path}')
    if record.get('fold') != f'fold_{year}' or type(record.get('seed')) is not int or record['seed'] != seed or record.get('label') != 'y5':
        raise ValueError(f'H2 identity mismatch: {path}')
    wall = record.get('wall_seconds')
    if type(wall) not in (float, int) or not math.isfinite(wall) or wall < 0:
        raise ValueError(f'invalid wall_seconds: {path}')
    daily = record.get('daily_ic')
    if not isinstance(daily, dict) or set(daily) != {'c0plus_fund'} or not isinstance(daily['c0plus_fund'], dict) or not daily['c0plus_fund']:
        raise ValueError(f'invalid H2 daily IC: {path}')
    series = pd.Series(daily['c0plus_fund'], dtype=float)
    dates = pd.to_datetime(series.index, errors='raise')
    if dates.tz is not None or dates.has_duplicates or not (dates.year == year).all() or not np.isfinite(series).all() or (series.abs() > 1).any():
        raise ValueError(f'invalid H2 dates/IC: {path}')
    return True


def run_all(folds_root, out_dir, data_dir=DATA_DIR, years=YEARS, seeds=SEEDS, dry_run=True):
    """Registered jobs, atomic JSON and fail-closed resume; default has no writes."""
    out = rr._output_dir(out_dir)
    years, seeds = tuple(years), tuple(seeds)
    jobs = [(y, s, out/f'fold_{y}_seed_{s}.json') for y in years for s in seeds]
    # Validate all existing jobs before any training or output writes.
    completed = {(y, s) for y, s, p in jobs if _completed(p, y, s)}
    financials = None
    for year, seed, path in jobs:
        if (year, seed) in completed:
            print(f'SKIP fold_{year} seed={seed} {path}', flush=True)
        elif dry_run:
            print(f'RUN fold_{year} seed={seed} {path}', flush=True)
        else:
            if financials is None:
                financials = _load_financials(data_dir)
            result = run_fold_seed(Path(folds_root)/f'fold_{year}', seed, *financials)
            out.mkdir(parents=True, exist_ok=True)
            rr._atomic_json(path, result)
            print(f'DONE fold_{year} seed={seed} wall_seconds={result["wall_seconds"]:.3f}', flush=True)
    if dry_run:
        print(f'DRY_RUN total={len(jobs)} pending={len(jobs)-len(completed)} skipped={len(completed)}', flush=True)


def analyze(h1_dir, h2_dir, n_boot=2000, seed=0):
    """Same H1 statistics on paired H2−H1; require every registered fold/seed.

    Call only once for formal H2. Apply the registered descriptive gate and
    H2 Holm-order threshold to the shared clean-fold statistics. The p value
    is the clean fixed-effect HAC Student t test, not exploratory p_boot.
    """
    h1 = analysis.load_results(h1_dir)
    h2 = analysis.load_results(h2_dir, ('c0plus_fund',))
    required = {(y, s) for y in YEARS for s in SEEDS}
    for records in (h1, h2):
        identities = {(y, s) for y, fold in records.items() for s in fold}
        if identities != required:
            raise ValueError('H2 requires all registered fold/seed pairs in both sources')
    report = analysis.report(h1_dir, n_boot=n_boot, seed=seed, candidate='c0plus_fund',
                             baseline='c0plus', candidate_dir=h2_dir, baseline_dir=h1_dir)
    report['preregistration'] = 'roadmap 12.36 H2'
    p = report['clean']['hac']['p']
    holm_pass = p is not None and p < .05
    report['h2_holm'] = dict(p=p, threshold=.05, passed=holm_pass,
                             method='clean fixed-effect HAC Student t; H1 passed, H2 threshold .05')
    if report['classification'] == '證實有益' and (
            report['descriptive']['noninferiority'] != '通過' or not holm_pass):
        report['classification'] = '不確定'
    report['limitations'][0] = 'H2 beneficial classification requires clean bootstrap/SESOI, descriptive noninferiority and HAC p<.05.'
    report['limitations'][1] = 'H1 passed as recorded in roadmap 12.36; H2 Holm-order threshold .05, no adoption authorization.'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds-root', type=Path, default=rr.LOCAL_DATA_ROOT/'matrices/rolling')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--run', action='store_true', help='execute all registered jobs (default: dry run)')
    args = parser.parse_args(argv)
    run_all(args.folds_root, args.out_dir, data_dir=args.data_dir, dry_run=not args.run)


if __name__ == '__main__':
    main()
