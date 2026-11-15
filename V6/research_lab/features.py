"""Close-t inputs only; predict t+1 open to t+h close (never same-bar entry).

Raw close avoids Yahoo retrospective adj_close factors. Source revisions and
survivor selection still prevent a claim of strict historical PIT evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .data import KEYS, calendar_panel

BASE_FEATURES = ([f'return_{h}d' for h in (1,5,10,20,60)] +
                 ['volatility_20d', 'volume_z_20d', 'volume_ratio_20d',
                  'dollar_volume', 'drawdown_60d', 'rsi_14d'])
FEATURE_COLUMNS = BASE_FEATURES + [f'{c}_cs_{suffix}' for c in BASE_FEATURES for suffix in ('rank','z')]


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    panel, _ = calendar_panel(prices)
    pieces = []
    for code, stock in panel.groupby(level='code', sort=True):
        valid = stock.observation_valid.eq(True)
        close = stock.close.where(valid)
        volume = stock.volume.where(valid)
        f = pd.DataFrame(index=stock.index)
        for h in (1,5,10,20,60):
            complete = close.rolling(h+1, min_periods=h+1).count().eq(h+1)
            f[f'return_{h}d'] = (close/close.shift(h)-1).where(complete)
        ret = close/close.shift(1)-1
        f['volatility_20d'] = ret.rolling(20, min_periods=20).std(ddof=0)
        mean, std = volume.rolling(20,min_periods=20).mean(), volume.rolling(20,min_periods=20).std(ddof=0)
        f['volume_z_20d'] = (volume-mean)/std.replace(0,np.nan)
        f['volume_ratio_20d'] = volume/mean.replace(0,np.nan)
        f['dollar_volume'] = close*volume
        f['drawdown_60d'] = close/close.rolling(60,min_periods=60).max()-1
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14,min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
        # RSI-like simple moving average; flat window is 50, all gains 100.
        f['rsi_14d'] = 100*gain/(gain+loss).replace(0,np.nan)
        f.loc[gain.eq(0) & loss.eq(0), 'rsi_14d'] = 50.
        pieces.append(f)
    f = pd.concat(pieces).replace([np.inf,-np.inf],np.nan)
    for col in BASE_FEATURES:
        grouped = f[col].groupby(level='date')
        f[f'{col}_cs_rank'] = grouped.rank(method='average', pct=True).where(grouped.transform('count').ge(2))
        scale = grouped.transform('std', ddof=0).replace(0,np.nan)
        f[f'{col}_cs_z'] = (f[col]-grouped.transform('mean'))/scale
    original = pd.MultiIndex.from_frame(prices[KEYS])
    out = prices.copy()
    out[FEATURE_COLUMNS] = f.reindex(original)[FEATURE_COLUMNS].to_numpy()
    return out.sort_values(KEYS,kind='stable').reset_index(drop=True)
