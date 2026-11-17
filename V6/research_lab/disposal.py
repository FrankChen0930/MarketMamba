"""Read-only A-2 research audit; UNMERGED / NOT_DEPLOYED / RESEARCH_ONLY.

Daily disposal membership is NOT an interval to extrapolate. False means no
known disposal under this rule, never verified tradability. The observed matrix
calendar is only a proxy. No model, production universe or label is modified.
This standalone diagnostic reuses existing matrices/folds; existing universe
code does not implement announcement-lagged daily disposal membership.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATUS = Path('/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/trading_status_raw.parquet')


def _days(values):
    return np.asarray(pd.to_datetime(values, errors='raise'), dtype='datetime64[D]')


def _disposals(status_df):
    d = status_df.loc[status_df['status'].eq('disposal'),
                      ['Date', 'stock_id', 'announced']].copy()
    d['Date'] = _days(d['Date'])
    d['announced'] = _days(d['announced'])
    d['stock_id'] = d['stock_id'].astype('string')
    return d


def disposal_known(status_df, dates, stock_ids):
    """Bool per aligned matrix row, using exact (stock, day) disposal keys.

    Effective session = lower_bound(announcement) + 1. Thus a non-session
    announcement first rolls forward, then waits one further observed session.
    Missing announcements and exhausted calendars produce no known key.
    Duplicates use OR across independently known announcements, without
    multiplying rows or inferring membership on unreported days.
    """
    days, ids = _days(dates), np.asarray(stock_ids)
    if days.ndim != 1 or ids.ndim != 1 or len(days) != len(ids):
        raise ValueError('dates and stock_ids must be aligned 1D row arrays')
    if np.isnat(days).any() or pd.isna(ids).any():
        raise ValueError('matrix row keys must not be missing')
    result = np.zeros(len(days), dtype=bool)
    if not len(days):
        return result
    calendar = np.unique(days)
    d = _disposals(status_df).dropna(subset=['Date', 'stock_id', 'announced'])
    if d.empty:
        return result
    announced = _days(d['announced'])
    effective = np.searchsorted(calendar, announced, side='left') + 1
    within = effective < len(calendar)
    known = within & (_days(d['Date']) >= calendar[np.minimum(effective, len(calendar) - 1)])
    keys = pd.MultiIndex.from_frame(d.loc[known, ['stock_id', 'Date']]).unique()
    for start in range(0, len(days), 500_000):
        stop = start + 500_000
        rows = pd.MultiIndex.from_arrays([ids[start:stop].astype(str), days[start:stop]])
        result[start:stop] = rows.isin(keys)
    return result


def _identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(realpath=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return dict(rows=int(len(values)), finite_rows=int(len(finite)),
                mean=float(finite.mean()) if len(finite) else None)


def universe_audit(status_path=STATUS, matrix_dir=ROOT / '.local-data/matrices/E5_roll_base',
                   folds_dir=ROOT / '.local-data/matrices/rolling'):
    """Return JSON-ready diagnostic; all input arrays opened read-only.

    Excluded return is row-weighted y5 over eval & known disposal, not a unique
    stock mean, portfolio return, or an executable return. Volume (if present)
    is a raw matrix feature proxy; units/normalization are not inferred.
    """
    base, folds = Path(matrix_dir), Path(folds_dir)
    paths = [Path(status_path), base / 'manifest.json', base / 'dates.npy',
             base / 'stock_ids.npy', base / 'y5.npy']
    manifest = json.loads(paths[1].read_text())
    load = lambda name: np.load(base / name, mmap_mode='r', allow_pickle=False)
    dates, ids, y5 = load('dates.npy'), load('stock_ids.npy'), load('y5.npy')
    if dates.ndim != 1 or ids.shape != dates.shape or y5.shape != dates.shape:
        raise ValueError('matrix arrays must be aligned 1D row arrays')
    status = pd.read_parquet(status_path)
    known = disposal_known(status, dates, ids)
    d = _disposals(status)
    delta = (_days(d['Date']) - _days(d['announced'])) / np.timedelta64(1, 'D')
    valid_delta = delta[np.isfinite(delta)]
    quantiles = {str(q): float(np.quantile(valid_delta, q)) if len(valid_delta) else None
                 for q in (0, .25, .5, .75, .95, 1)}
    calendar = np.unique(dates)
    names = manifest.get('feature_order', [])
    volume = None
    if 'Volume' in names:
        x = load('X.npy')
        if x.shape != (len(dates), len(names)):
            raise ValueError('X shape disagrees with manifest feature_order')
        volume = x[:, names.index('Volume')]
        paths.append(base / 'X.npy')
    annual = []
    for year in range(2021, 2027):
        path = folds / f'fold_{year}' / 'splits.npy'
        split = np.load(path, mmap_mode='r', allow_pickle=False)
        if split.shape != dates.shape or split.dtype.kind not in 'iu' or not np.isin(split, [0, 1, 2, 3, 4]).all():
            raise ValueError(f'invalid split: {path}')
        evaluation = split == 2
        if np.any(evaluation & (dates.astype('datetime64[Y]') != np.datetime64(str(year), 'Y'))):
            raise ValueError(f'eval rows outside fold year {year}')
        excluded = evaluation & known
        n, k = int(evaluation.sum()), int(excluded.sum())
        row = dict(year=year, eval_rows=n, disposal_known_rows=k,
                   disposal_known_ratio=k / n if n else None,
                   excluded_stock_count=int(len(np.unique(ids[excluded]))),
                   excluded_y5_alpha=_stats(y5[excluded]))
        if volume is not None:
            row['excluded_volume_proxy'] = _stats(volume[excluded])
        annual.append(row)
        paths.append(path)
    return dict(schema='disposal-audit-v1', evidence_class='RESEARCH_ONLY',
                state=['UNMERGED', 'NOT_DEPLOYED'], dry_run=True,
                observed_at=datetime.now(timezone.utc).isoformat(),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_files={str(p.relative_to(ROOT)): _identity(p) for p in
                              (Path(__file__), ROOT / 'V6/research_lab/tests/test_disposal.py')},
                environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                contract=dict(effective_session='lower_bound(announced, unique_sorted_matrix_dates) + 1',
                              membership='exact daily key; duplicate OR', eval_split=2,
                              label='matrix y5 (Alpha name does not certify excess return)',
                              aggregation='eval & disposal_known; row-weighted finite means', model='N/A'),
                limitations=['False is not verified tradability', 'Observed-union calendar, not qualified exchange calendar',
                             'Historical diagnostic, not PIT/execution certification',
                             'Volume is raw matrix proxy, not turnover rate; units unknown'],
                inputs=[_identity(p) for p in paths], matrix_rows=len(dates),
                disposal_rows=len(d), attention_rows_ignored=int(status.status.eq('attention').sum()),
                date_minus_announced_days=dict(finite_rows=len(valid_delta), quantiles=quantiles,
                                               negative_rows=int((valid_delta < 0).sum())),
                announcement_not_in_calendar_rows=int((~np.isin(_days(d.announced), calendar) & d.announced.notna()).sum()),
                missing_announcement_rows=int(d.announced.isna().sum()),
                duplicate_stock_date_excess_rows=int(d.duplicated(['stock_id', 'Date']).sum()),
                duplicate_stock_date_member_rows=int(d.duplicated(['stock_id', 'Date'], keep=False).sum()),
                duplicate_scope='disposal only; attention ignored', annual=annual)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, default=STATUS)
    parser.add_argument('--matrix', type=Path, default=ROOT / '.local-data/matrices/E5_roll_base')
    parser.add_argument('--folds', type=Path, default=ROOT / '.local-data/matrices/rolling')
    parser.add_argument('--output', type=Path, default=ROOT / '.local-data/disposal_audit.json')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='always report-only; no data/universe writes')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / '.local-data').resolve()):
        parser.error('output must be inside repository .local-data')
    if out.exists():
        parser.error('output already exists; use an independent --output namespace')
    report = universe_audit(args.status, args.matrix, args.folds)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(out)


if __name__ == '__main__':
    main()
