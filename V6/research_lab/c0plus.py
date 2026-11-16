"""C0+: LightGBM with temporal context (lags, rolling means, market aggregates) on the same rows/label/split as B4,
plus tradability-style checks that need no extra data: IC by size tercile and IC with the signal delayed one session.
RESEARCH_ONLY. Dense (session x stock) panels make lags exact and gap-aware: a missing session is NaN, never shifted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import matrix_baselines as mb
from .seq_runner import SeqData
from .significance import compare_models

CORE = ['Return_1d', 'Return_5d', 'Return_20d', 'RS_5d', 'RS_20d', 'RS_60d', 'RSI_14', 'ATR_14', 'Volatility_20d', 'Volume',
        'Foreign_Net', 'Investment_Trust_Net', 'Dealer_Net', 'Margin_Balance', 'Short_Balance', 'Day_Trade_Volume']
LAGS, ROLLS = (1, 2, 5, 10, 20), (5, 20, 60)
MARKET = ['Return_5d', 'Return_20d', 'Volatility_20d']


def dense_panel(data, col):
    p = np.full((len(data.ends), data.n_stocks), np.nan, dtype=np.float32)
    p[data.date_idx, data.stock_idx] = np.asarray(data.X[:, col], dtype=np.float32)
    return p


def lagged(p, t, s, lag):
    tt = t-lag
    out = np.full(len(t), np.nan, dtype=np.float32)
    ok = tt >= 0
    out[ok] = p[tt[ok], s[ok]]
    return out


def rolling_mean(p, w, min_frac=.5):
    m = ~np.isnan(p)
    z = np.zeros((1, p.shape[1]))
    s = np.vstack([z, np.cumsum(np.nan_to_num(p).astype(np.float64), axis=0)])
    c = np.vstack([z, np.cumsum(m, axis=0)])
    idx = np.arange(1, p.shape[0]+1)
    lo = np.maximum(idx-w, 0)
    cnt = c[idx]-c[lo]
    out = (s[idx]-s[lo])/np.where(cnt > 0, cnt, np.nan)
    out[cnt < np.ceil(w*min_frac)] = np.nan
    return out.astype(np.float32)


def build_features(data, rows, names, ind_stock=None, market=True, extra=None):
    """(len(rows), n_features) float32 and feature names. Base 48 columns + lags + rolling means + market aggregates."""
    col = {n: i for i, n in enumerate(names)}
    t, s = data.date_idx[rows], data.stock_idx[rows]
    cols = [np.asarray(data.X[rows], dtype=np.float32)]
    fn = list(names)
    for c in CORE:
        p = dense_panel(data, col[c])
        for lag in LAGS:
            cols.append(lagged(p, t, s, lag)[:, None]); fn.append(f'{c}_lag{lag}')
        for w in ROLLS:
            r = rolling_mean(p, w)
            cols.append(r[t, s][:, None]); fn.append(f'{c}_mean{w}')
    if ind_stock is not None:                       # industry-relative block (UNVERIFIED current-snapshot industry, see industry.py)
        from .industry import REL, panel_relative
        for c in REL:
            rel, mean = panel_relative(dense_panel(data, col[c]), ind_stock)
            cols += [rel[t, s][:, None], mean[t, s][:, None]]; fn += [f'{c}_vs_ind', f'{c}_ind_mean']
    for c in (MARKET if market else ()):          # market=False = ablation without same-day cross-sectional aggregates
        p = dense_panel(data, col[c])
        with np.errstate(all='ignore'):
            agg = {'mean': np.nanmean(p, axis=1), 'std': np.nanstd(p, axis=1), 'pos': np.nanmean(np.where(np.isnan(p), np.nan, p > 0), axis=1)}
        for k, v in agg.items():
            cols.append(v[t][:, None].astype(np.float32)); fn.append(f'mkt_{c}_{k}')
    if extra is not None:                              # extra(data, rows) -> (array, names): e.g. revenue.make_extra
        a, n = extra(data, rows); cols.append(a); fn += list(n)
    return np.hstack(cols), fn


def daily_ic(values, y, date_idx, dates, min_names=10):
    df = pd.DataFrame(dict(d=date_idx, s=values, y=y))
    df = df[np.isfinite(df.s) & np.isfinite(df.y)]
    r = df.groupby('d').apply(lambda g: g.s.rank().corr(g.y.rank()) if len(g) >= min_names else np.nan, include_groups=False).dropna()
    r.index = dates[r.index.to_numpy()]
    return r


def delayed_labels(data, rows, label='y5', delay=1):
    """Label of the same stock `delay` sessions later (NaN if absent) = the signal acted on `delay` sessions late."""
    t, s = data.date_idx[rows], data.stock_idx[rows]
    tt = t+delay
    ok = tt < len(data.ends)
    out = np.full(len(rows), np.nan, dtype=np.float32)
    r2 = np.full(len(rows), -1, dtype=np.int64)
    r2[ok] = data.row_of[tt[ok], s[ok]]
    good = r2 >= 0
    out[good] = data.y[label][r2[good]]
    return out


def tercile_ic(values, y, date_idx, dates, strata_value):
    """IC within the lowest / middle / highest third of `strata_value` (ranked within each session)."""
    df = pd.DataFrame(dict(d=date_idx, s=values, y=y, v=strata_value))
    df['q'] = df.groupby('d').v.rank(pct=True)
    out = {}
    for name, lo, hi in (('small', 0, 1/3), ('mid', 1/3, 2/3), ('large', 2/3, 1.0001)):
        g = df[(df.q > lo) & (df.q <= hi)] if name != 'small' else df[(df.q >= lo) & (df.q <= hi)]
        out[name] = daily_ic(g.s.to_numpy(), g.y.to_numpy(), g.d.to_numpy(), dates)
    return out


def run(directory, label='y5', max_train_rows=500_000, seed=17, n_boot=1000, lgbm=None, log=print, row_mask=None, extra_daily=None, industry_dir=None, ind_only_variant=False, market=True, exclude=None, extra=None):
    """row_mask: evaluate only these eval rows (e.g. SeqData.eligible, so the comparison is on the rows a sequence model can score).
    extra_daily: {name: daily-IC Series} of other models on the SAME rows, added to the paired tests."""
    from lightgbm import LGBMRegressor
    data = SeqData(directory, window=1, normalize=None)
    ctx = mb.Context(directory, label, max_train_rows, seed, 'rank', row_mask=row_mask)
    names = data.feature_names
    p = dict(n_estimators=200, num_leaves=31, learning_rate=.05, min_child_samples=200, subsample=.8, subsample_freq=1,
             colsample_bytree=.8, reg_lambda=10., random_state=seed, n_jobs=4, verbose=-1)
    p.update(lgbm or {})                                   # both models share these settings so the comparison isolates the features
    ind_stock = None
    if industry_dir is not None:
        from .industry import industry_codes, load_industry_map
        uniq = np.unique(np.load(Path(directory)/'stock_ids.npy', mmap_mode='r'))
        ind_stock = industry_codes(uniq, load_industry_map(industry_dir))
        log(f'[c0plus] industry known for {(ind_stock >= 0).mean():.1%} of {len(uniq)} stock codes')
    xf, fn = build_features(data, ctx.fit_idx, names, ind_stock, market, extra)
    keep = np.array([not (exclude and exclude(n)) for n in fn])      # exclude: name -> bool, drops those columns from fit AND eval
    xf, fn = xf[:, keep], [n for n, k in zip(fn, keep) if k]
    base_scores = mb.compute_scores(ctx, 'lightgbm', dict(lgbm_estimators=p['n_estimators'], lgbm_leaves=p['num_leaves'], lgbm_lr=p['learning_rate'],
                                                          lgbm_min_child=p['min_child_samples'], lgbm_reg_lambda=p['reg_lambda'], lgbm_threads=p['n_jobs']))
    plus = LGBMRegressor(**p).fit(xf, ctx.y_fit)
    ev = ctx.eval_idx
    xe = build_features(data, ev, names, ind_stock, market, extra)[0][:, keep]
    plus_scores = plus.predict(xe)
    log(f'[c0plus] features {len(fn)}; fit {len(ctx.fit_idx)}; eval {len(ev)}')
    y, dix, dates = data.y[label][ev], data.date_idx[ev], data.dates
    size = np.asarray(data.X[ev][:, names.index('Market_Cap_Log')], dtype=np.float32)
    y_next = delayed_labels(data, ev, label, 1)
    res, daily = {}, {}
    for name, sc in (('lightgbm_day_t', base_scores), ('c0plus', plus_scores), ('reversal_ret5', -np.asarray(data.X[ev][:, names.index('Return_5d')], dtype=np.float32))):
        ic = daily_ic(sc, y, dix, dates)
        ic1 = daily_ic(sc, y_next, dix, dates)
        terc = tercile_ic(sc, y, dix, dates, size)
        daily[name] = ic
        res[name] = dict(ic=mb.summarize_daily(ic), ic_delayed_1d=mb.summarize_daily(ic1),
                         by_size={k: float(v.mean()) for k, v in terc.items()},
                         by_half_year={f'{y_}H{h}': float(g.mean()) for (y_, h), g in ic.groupby([ic.index.year, (ic.index.month > 6).astype(int)+1])})
    daily.update(extra_daily or {})
    common = pd.concat(daily, axis=1).dropna().index
    sig = compare_models({k: v.loc[common] for k, v in daily.items()}, 'lightgbm_day_t', int(label[1:]), n_boot=n_boot, seed=seed)
    imp = sorted(zip(fn, plus.feature_importances_), key=lambda x: -x[1])[:15]
    return dict(label=label, fit_rows=int(len(ctx.fit_idx)), eval_rows=int(len(ev)), n_features=len(fn), models=res,
                vs_lightgbm_day_t=sig, daily_ic={k: {str(d.date()): float(v) for d, v in s_.items()} for k, s_ in daily.items()}, top_features=[(a, int(b)) for a, b in imp], eval_idx=ev, scores=dict(c0plus=plus_scores, lightgbm_day_t=base_scores),
                caveats=['same fit rows and eval rows as B4; evaluation window is research-used; labels are simulation proxy; survivorship-biased',
                         'size/volume strata use per-session scaled Market_Cap_Log (rank within session); raw volume and limit-up/down are not in the matrix',
                         'delayed-1d IC = score at t vs label of the same stock at t+1 (entry one session later)'])
