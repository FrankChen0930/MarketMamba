"""Research proxy labels, NEVER verified executable returns.

Signal after close t; enter raw open t+1, exit raw close t+h. Missing, invalid,
zero-volume or flagged observations anywhere t..t+h invalidate the label.
No rolling to another session. Rank ties use average percentile; singleton NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .data import KEYS, calendar_panel

HORIZONS = (5,10,20)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def compute_labels(prices: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    horizons = tuple(positive_integer(h,'horizon') for h in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError('horizons must be nonempty and unique')
    panel, days = calendar_panel(prices)
    pieces = []
    for code, stock in panel.groupby(level='code', sort=True):
        out = pd.DataFrame(index=stock.index)
        dates = pd.Series(days.to_numpy(),index=stock.index)
        out['label_entry'] = dates.shift(-1)
        valid = stock.observation_valid.eq(True).astype(int)
        for h in horizons:
            complete = valid.rolling(h+1,min_periods=h+1).sum().shift(-h).eq(h+1)
            out[f'label_end_{h}d'] = dates.shift(-h)
            out[f'fwd_return_{h}d'] = (stock.close.shift(-h)/stock.open.shift(-1)-1).where(complete)
        pieces.append(out)
    out = pd.concat(pieces)
    for h in horizons:
        groups = out[f'fwd_return_{h}d'].groupby(level='date')
        out[f'rank_label_{h}d'] = groups.rank(method='average',pct=True).where(groups.transform('count').ge(2))
    original = pd.MultiIndex.from_frame(prices[KEYS])
    return out.reindex(original).reset_index()


def purged_training_mask(frame: pd.DataFrame, test_start, *, horizon=20, purge=None, embargo=0) -> pd.Series:
    """Remove purge+embargo observed sessions before test; require mature labels.

    Embargo is an additional pre-test safety gap for past-only expanding folds;
    no post-test samples are trained in the same fold. All stocks share dates.
    """
    horizon = positive_integer(horizon,'horizon')
    purge = horizon if purge is None else positive_integer(purge,'purge')
    if purge < horizon or isinstance(embargo,bool) or not isinstance(embargo,(int,np.integer)) or embargo < 0:
        raise ValueError('purge must be >= horizon; embargo must be a nonnegative integer')
    end, target = f'label_end_{horizon}d', f'fwd_return_{horizon}d'
    if end not in frame or target not in frame:
        raise ValueError('label end and target required for leakage checks')
    days = pd.DatetimeIndex(frame.date.unique()).sort_values()
    expected = pd.Series(days,index=days).shift(-horizon)
    actual_expected = frame.date.map(expected)
    if not (frame[end].eq(actual_expected) | (frame[end].isna() & actual_expected.isna())).all():
        raise ValueError('label end violates horizon/calendar alignment')
    start = pd.Timestamp(test_start)
    if start not in days:
        raise ValueError('test_start not in observed sessions')
    cutoff = days.get_loc(start)-purge-embargo
    eligible = frame.date.isin(days[:max(0,cutoff)])
    return eligible & frame[end].lt(start) & frame[end].notna() & np.isfinite(frame[target])
