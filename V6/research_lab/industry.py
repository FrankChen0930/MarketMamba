"""Industry-relative features from the exchanges' public company-profile open data (TWSE t187ap03_L, TPEx mopsfin_t187ap03_O).

POINT-IN-TIME CAVEAT (stated in every result that uses this): the files are a CURRENT snapshot (2026-11). Stocks that were delisted
before then are absent, and a stock that changed industry is assigned its current industry for all of history. So this is an
UNVERIFIED back-fill and survivorship-biased; stocks without an industry code keep NaN features (no guessing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REL = ['Return_5d', 'Return_20d', 'Volatility_20d', 'RS_20d']


def load_industry_map(raw_dir):
    """-> {stock_code(str): industry_code(str)} from the two archived raw payloads."""
    raw = Path(raw_dir)
    twse = json.loads((raw/'twse.json').read_text(encoding='utf-8'))
    tpex = json.loads((raw/'tpex.json').read_text(encoding='utf-8'))
    m = {}
    for r in twse:
        if r.get('公司代號') and r.get('產業別'):
            m[str(r['公司代號']).strip()] = 'L'+str(r['產業別']).strip()
    for r in tpex:
        if r.get('SecuritiesCompanyCode') and r.get('SecuritiesIndustryCode'):
            m[str(r['SecuritiesCompanyCode']).strip()] = 'O'+str(r['SecuritiesIndustryCode']).strip()
    return m


def industry_codes(stock_ids, mapping):
    """Integer industry id per row (-1 = unknown) for an array of stock id strings."""
    uniq, inv = np.unique(np.asarray(stock_ids).astype(str), return_inverse=True)
    labels = {k: i for i, k in enumerate(sorted(set(mapping.values())))}
    per = np.array([labels.get(mapping.get(u.strip(), ''), -1) for u in uniq])
    return per[inv]


def relative_features(values, date_idx, ind, min_peers=3):
    """values (n, k) float; returns (n, 2k): value minus the same-session industry mean, and the industry mean itself.
    Industry mean EXCLUDES the stock itself (leave-one-out) so a stock cannot pull its own benchmark; groups with fewer than
    `min_peers` other members, and unknown industries, give NaN."""
    n, k = values.shape
    df = pd.DataFrame(values.astype(np.float64))
    key = pd.Series(date_idx.astype(np.int64)*10_000+ind.astype(np.int64)+1)      # ind=-1 -> bucket 0 of that day
    g = df.groupby(key.values)
    s, c = g.transform('sum').to_numpy(), g.transform('count').to_numpy()
    own_ok = np.isfinite(values)
    s_loo = s-np.where(own_ok, values, 0.)
    c_loo = c-own_ok.astype(int)
    with np.errstate(all='ignore'):
        mean = np.where(c_loo >= min_peers, s_loo/c_loo, np.nan)
    mean[ind < 0] = np.nan
    return np.hstack([values-mean, mean]).astype(np.float32)


def panel_relative(panel, per_stock_ind, min_peers=3):
    """panel (T, S) float32 with NaN holes; per_stock_ind (S,) int (-1 unknown) -> (rel, mean) panels (T, S), leave-one-out
    same-session industry mean over ALL stocks present that session (not just a sample)."""
    T, S = panel.shape
    known = per_stock_ind >= 0
    n_ind = int(per_stock_ind.max())+1 if known.any() else 0
    onehot = np.zeros((S, max(n_ind, 1)), np.float64)
    onehot[np.flatnonzero(known), per_stock_ind[known]] = 1.
    ok = np.isfinite(panel)
    sums = np.nan_to_num(panel).astype(np.float64) @ onehot          # (T, I)
    cnts = ok.astype(np.float64) @ onehot
    idx = np.where(known, per_stock_ind, 0)
    s_row, c_row = sums[:, idx], cnts[:, idx]                         # (T, S): the stock's own industry totals that session
    s_loo = s_row-np.where(ok, panel, 0.)
    c_loo = c_row-ok
    with np.errstate(all='ignore'):
        mean = np.where((c_loo >= min_peers) & known[None, :], s_loo/c_loo, np.nan)
    return (panel-mean).astype(np.float32), mean.astype(np.float32)
