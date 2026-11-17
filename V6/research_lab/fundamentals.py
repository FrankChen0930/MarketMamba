"""Roadmap §12.36 frozen H2 features; RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Legal deadlines are UNVERIFIED PIT: restatements and late filings are unknown.
Ranks use all present matrix names on each session, before row sampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = ['GM', 'OM', 'EPS_YoY']


def availability_dates(period_ends, availability_shift_days=0):
    """Pure deadline mapping, with no weekend adjustment; positive shift is earlier."""
    ends = pd.DatetimeIndex(period_ends)
    if ends.isna().any() or not ends.is_quarter_end.all():
        raise ValueError('expected calendar quarter ends')
    values = []
    for d in ends:
        year, month, day = {3: (d.year, 5, 15), 6: (d.year, 8, 14),
                            9: (d.year, 11, 14), 12: (d.year+1, 3, 31)}[d.month]
        values.append(pd.Timestamp(year=year, month=month, day=day))
    return pd.DatetimeIndex(values) - pd.Timedelta(days=availability_shift_days)


def latest_period(dates, availability_shift_days=0):
    """Latest calendar quarter strictly available on each signal date, even if absent."""
    dates = pd.DatetimeIndex(dates)
    if dates.empty or dates.isna().any() or dates.tz is not None:
        raise ValueError('need nonempty naive signal dates')
    shifted = dates + pd.Timedelta(days=availability_shift_days)
    quarters = pd.period_range(f'{shifted.min().year-2}Q1', f'{shifted.max().year}Q4', freq='Q')
    deadlines = availability_dates(quarters.to_timestamp(how='end').normalize(), availability_shift_days)
    positions = deadlines.searchsorted(dates, side='left') - 1
    return quarters[positions]


def make_extra(financials_long, stock_info, dates, stock_ids, availability_shift_days=0):
    """Return extra(data, rows): float32 GM, OM, EPS_YoY; exclusions remain NaN.

    stock_ids is ordered like SeqData.stock_idx (lexically sorted unique IDs).
    First-year age counts matrix calendar sessions from each stock's first row.
    Average ranks resolve ties deterministically; valid count is per feature.
    """
    ids = pd.Index([str(s) for s in stock_ids])
    dates = pd.DatetimeIndex(dates)
    latest = latest_period(dates, availability_shift_days)
    fin = financials_long.loc[financials_long['type'].isin(
        ['Revenue', 'GrossProfit', 'OperatingIncome', 'EPS']), ['stock_id', 'Date', 'type', 'value']].copy()
    fin['stock_id'] = fin.stock_id.astype(str)
    fin['Date'] = pd.to_datetime(fin.Date)
    if not fin.Date.dt.is_quarter_end.all():
        raise ValueError('financial Date must be calendar quarter end')
    if fin.duplicated(['stock_id', 'Date', 'type']).any():
        raise ValueError('duplicate financial key')
    fin['q'] = fin.Date.dt.to_period('Q')
    table = fin.pivot(index=['q', 'stock_id'], columns='type', values='value')
    def lookup(quarters, name):
        index = pd.MultiIndex.from_product([quarters, ids], names=['q', 'stock_id'])
        series = table[name] if name in table else pd.Series(dtype=float)
        return series.reindex(index).to_numpy(float).reshape(len(quarters), len(ids))
    quarters = pd.period_range(latest.min(), latest.max(), freq='Q')
    rev = lookup(quarters, 'Revenue')
    with np.errstate(all='ignore'):
        raw = np.stack([lookup(quarters, 'GrossProfit')/rev,
                        lookup(quarters, 'OperatingIncome')/rev,
                        lookup(quarters, 'EPS')-lookup(quarters-4, 'EPS')], axis=2)
    raw[(~np.isfinite(rev)) | (rev <= 0)] = np.nan
    raw[~np.isfinite(raw)] = np.nan
    info = stock_info.copy()
    financial = set(info.loc[info.industry_category.fillna('').str.contains('金融保險|金融業'), 'stock_id'].astype(str))
    raw[:, ids.isin(financial)] = np.nan
    panel = raw[quarters.get_indexer(latest)]
    cached_data, ranked = None, None

    def extra(data, rows):
        nonlocal cached_data, ranked
        if cached_data is not data:
            t, s = np.asarray(data.date_idx), np.asarray(data.stock_idx)
            first = np.full(len(ids), len(dates), dtype=int)
            np.minimum.at(first, s, t)
            present = np.zeros(panel.shape[:2], dtype=bool)
            present[t, s] = True
            valid = present & (np.arange(len(dates))[:, None] - first[None, :] >= 252)
            values = np.where(valid[:, :, None], panel, np.nan)
            ranked = np.stack([pd.DataFrame(values[:, :, j]).rank(axis=1, pct=True).to_numpy()-.5
                               for j in range(3)], axis=2).astype(np.float32)
            cached_data = data
        return ranked[data.date_idx[rows], data.stock_idx[rows]], list(NAMES)
    return extra
