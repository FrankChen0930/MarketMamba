"""Monthly-revenue features with a point-in-time rule. RESEARCH_ONLY.

Availability rule (conservative): Taiwan listed/OTC companies must report month-m revenue by the 10th of month m+1, so on a
session dated d the latest usable revenue month is  period(d - 10 days) - 1.  (d = Oct 10 -> August; d = Oct 11 -> September.)
The provider's `Date` column (first day of month m+1) is NOT used as availability: it would let sessions on days 1-10 see
revenue that is not guaranteed public yet. Revenue <= 0 is treated as missing. Restatements are not tracked (the stored
value is the latest one), which is an UNVERIFIED-PIT caveat for any result using this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = ['rev_yoy', 'rev_mom', 'rev_yoy3']


def monthly_panel(rev):
    """rev: DataFrame[stock_id, revenue_year, revenue_month, revenue] -> (periods DatetimeIndex[M], panel (n_periods, n_stocks) float, stock_ids)."""
    r = rev[rev.revenue > 0].copy()
    r['p'] = pd.PeriodIndex.from_fields(year=r.revenue_year.astype(int), month=r.revenue_month.astype(int), freq='M')
    r = r.drop_duplicates(['stock_id', 'p'], keep='last')
    pv = r.pivot(index='p', columns='stock_id', values='revenue')
    full = pd.period_range(pv.index.min(), pv.index.max(), freq='M')
    pv = pv.reindex(full)
    return full, pv.to_numpy(dtype=np.float64), list(pv.columns)


def growth_panels(panel):
    """-> dict name -> (n_periods, n_stocks): yoy = r_m/r_{m-12}-1, mom = r_m/r_{m-1}-1, yoy3 = sum(r_m..r_{m-2})/sum(same a year earlier)-1."""
    def shift(a, k):
        out = np.full_like(a, np.nan); out[k:] = a[:-k]; return out
    with np.errstate(all='ignore'):
        s3 = panel + shift(panel, 1) + shift(panel, 2)                       # NaN if any of the 3 months is missing
        return {'rev_yoy': panel/shift(panel, 12)-1, 'rev_mom': panel/shift(panel, 1)-1, 'rev_yoy3': s3/shift(s3, 12)-1}


def latest_period(dates):
    """Latest revenue month usable on each session date (see module docstring)."""
    return (pd.DatetimeIndex(dates)-pd.Timedelta(days=10)).to_period('M')-1


def make_extra(rev, dates, stock_ids):
    """-> callable(data, rows) -> (float32 (len(rows), 3), NAMES) aligned with c0plus.build_features rows. NaN where unavailable."""
    periods, panel, rev_ids = monthly_panel(rev)
    gp = growth_panels(panel)
    col = {s: i for i, s in enumerate(rev_ids)}
    sidx = np.array([col.get(str(s), -1) for s in stock_ids])
    lp = latest_period(dates)
    pos = np.array([periods.get_loc(p) if p in periods else -1 for p in lp])      # -1: before data starts or after it ends

    def extra(data, rows):
        t, s = data.date_idx[rows], data.stock_idx[rows]
        pi, ci = pos[t], sidx[s]
        ok = (pi >= 0) & (ci >= 0)
        out = np.full((len(rows), len(NAMES)), np.nan, dtype=np.float32)
        for j, n in enumerate(NAMES):
            out[ok, j] = gp[n][pi[ok], ci[ok]]
        return out, list(NAMES)
    return extra
