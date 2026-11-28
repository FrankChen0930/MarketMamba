"""Offline, exposed-history leakage audit primitives; no production authority."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path('/home/frank/projects/MarketMamba')
SHARED = Path('/home/frank/projects/p4-bundles/prepared/shared')
PRICE_LEVEL = ('Open', 'High', 'Low', 'Close', 'MA_20', 'MA_60', 'ATR_14')


def feature_names():
    from research_lab.c0plus import CORE, LAGS, ROLLS, MARKET
    from tools.step3 import FEATURES
    policy = Path(__file__).resolve().parents[1] / 'research/v7/corrected-baseline-v1/policy.json'
    base = json.loads(policy.read_text())['features']['kept']
    return (base + [f'{c}_{suffix}' for c in CORE
                    for suffix in [*[f'lag{v}' for v in LAGS], *[f'mean{v}' for v in ROLLS]]]
            + [f'mkt_{c}_{v}' for c in MARKET for v in ('mean', 'std', 'pos')]
            + FEATURES)


def price_columns(names):
    return [j for j, name in enumerate(names)
            if name in PRICE_LEVEL or name.startswith('ATR_14_lag') or name.startswith('ATR_14_mean')]


def scalefree(x, names, actual_average):
    """Replace all price-dimension columns by ratios to signal-date adjusted Close.

    ATR lags/means also divide by signal-date Close. Other columns retain exact
    bytes; OBV is already a volume-only rolling z score. No scaler is learned.
    Missing actual turnover gives NaN log price, never adjusted-price fallback.
    """
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(names) or len(actual_average) != len(x):
        raise ValueError('STOP transform axes')
    close = x[:, names.index('Close')]
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('STOP unknown/nonpositive Close')
    out = np.empty((len(x), len(names) + 1), dtype=np.float32)
    out[:, :-1] = x
    cols = price_columns(names)
    out[:, cols] = x[:, cols] / close[:, None]
    actual_average = np.asarray(actual_average)
    out[:, -1] = np.nan
    ok = np.isfinite(actual_average) & (actual_average > 0)
    out[ok, -1] = np.log(actual_average[ok])
    return out


def perturb_future(frame, cutoff, seed, date_col='Date'):
    """Perturb numeric payload strictly after cutoff; preserve dates and keys."""
    out = frame.copy(deep=True)
    if date_col is None:
        return out
    future = pd.to_datetime(out[date_col]) > pd.Timestamp(cutoff)
    rng = np.random.default_rng(seed)
    for c in out.select_dtypes(include='number'):
        if c not in ('stock_id', 'stock_idx', 'session', 'date_idx'):
            value = out[c].astype(float).to_numpy(copy=True)
            value[future] = value[future] * rng.uniform(.25, 4., int(future.sum())) + rng.normal(size=int(future.sum()))
            out[c] = value
    return out


def bitwise_prefix(a, b, mask):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a[mask].tobytes() == b[mask].tobytes()
