"""Paired significance for daily rank-IC series. RESEARCH_ONLY.

Compares two models scored on the same sessions: HAC (Newey-West, Bartlett,
lag=horizon-1) t-test on the daily IC difference plus a circular block bootstrap,
with Holm correction across a family of comparisons. Differences are only
formed on sessions where both ICs are finite. Pure numpy/stdlib.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_OBS = 20


def _normal_two_sided(t):
    return math.erfc(abs(t)/math.sqrt(2))


def holm_adjust(pvalues):
    p = [float(x) for x in pvalues]
    if any(not (0 <= x <= 1) or x != x for x in p):
        raise ValueError('p-values must be in [0, 1]')
    order = sorted(range(len(p)), key=lambda i: p[i])
    out = [0.]*len(p)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(p)-rank)*p[i]))
        out[i] = running
    return out


def _hac_se(d, lag):
    n = len(d)
    z = d-d.mean()
    s = float(np.dot(z, z))
    for k in range(1, min(lag, n-1)+1):
        s += 2*(1-k/(lag+1))*float(np.dot(z[k:], z[:-k]))
    return math.sqrt(max(s, 0.))/n


def paired_ic_test(ic_a, ic_b, horizon, n_boot=2000, block=None, seed=17, min_obs=MIN_OBS):
    """Test H0: mean(IC_a - IC_b) = 0. Positive mean_diff favours model a."""
    both = pd.concat([pd.Series(ic_a, dtype=float).rename('a'),
                      pd.Series(ic_b, dtype=float).rename('b')], axis=1, join='inner')
    both = both[np.isfinite(both.a) & np.isfinite(both.b)].sort_index()
    n = len(both)
    if n < min_obs:
        raise ValueError(f'need >= {min_obs} common finite sessions, got {n}')
    d = (both.a-both.b).to_numpy(float)
    mean = float(d.mean())
    se = _hac_se(d, max(horizon-1, 0))
    if np.ptp(d) == 0 or se == 0:
        t_hac, p_hac = (0., 1.) if mean == 0 else (math.copysign(math.inf, mean), 0.)
    else:
        t_hac = mean/se
        p_hac = _normal_two_sided(t_hac)
    block = block or max(horizon, 5)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    nb = (n+block-1)//block
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None]+np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    centred = means-mean
    p_boot = float(min(1., (np.sum(np.abs(centred) >= abs(mean)-1e-15)+1)/(n_boot+1))) if mean != 0 else 1.
    lo, hi = np.quantile(means, [.025, .975])
    return dict(n=n, mean_diff=mean, se_hac=se, t_hac=float(t_hac), p_hac=float(p_hac),
                p_boot=p_boot, ci95_boot=[float(lo), float(hi)], hac_lag=max(horizon-1, 0),
                boot_block=block, n_boot=n_boot, seed=seed,
                first=str(both.index[0]), last=str(both.index[-1]))


def compare_models(daily_ic, baseline, horizon, n_boot=2000, seed=17):
    """Each model vs `baseline`; Holm-adjusted over the whole family (HAC and bootstrap p)."""
    if baseline not in daily_ic:
        raise KeyError(baseline)
    rows = []
    for name in sorted(daily_ic):
        if name == baseline:
            continue
        r = paired_ic_test(daily_ic[name], daily_ic[baseline], horizon, n_boot=n_boot, seed=seed)
        r['model'], r['baseline'] = name, baseline
        rows.append(r)
    for key in ('p_hac', 'p_boot'):
        for r, adj in zip(rows, holm_adjust([r[key] for r in rows])):
            r[key+'_holm'] = adj
    return rows
