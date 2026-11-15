"""Pure validation; no imputation, price adjustment, or inferred tradability."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['code', 'date']
PRICE_COLUMNS = ['open', 'high', 'low', 'close']
RAW_COLUMNS = KEYS + PRICE_COLUMNS + ['adj_close', 'volume']
OUTLIER_THRESHOLD = 0.35


def load_prices(path: str | Path) -> pd.DataFrame:
    return validate_prices(pd.read_parquet(path))


def validate_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Sort keys, collapse exact duplicates, reject conflicts; retain invalid rows.

    Invalid OHLC and volume become NaN with flags. Zero volume is retained but
    observation_valid=False. >35% raw close jumps flag unknown events, not splits
    confirmed by evidence. adj_close is provenance only, never a model input.
    """
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing or raw.empty:
        raise ValueError(f'empty prices or missing columns: {sorted(missing)}')
    d = raw[RAW_COLUMNS].copy()
    if d.code.isna().any():
        raise ValueError('missing code')
    d['code'] = d.code.astype(str).str.strip()
    if d.code.eq('').any():
        raise ValueError('empty code')
    d['date'] = pd.to_datetime(d.date, errors='raise')
    if d.date.isna().any() or d.date.dt.tz is not None or not d.date.eq(d.date.dt.normalize()).all():
        raise ValueError('dates must be non-null timezone-naive daily session dates')
    for col in PRICE_COLUMNS + ['adj_close', 'volume']:
        d[col] = pd.to_numeric(d[col], errors='coerce').astype('float64')
    n = len(d)
    d = d.drop_duplicates()
    if d.duplicated(KEYS).any():
        raise ValueError('conflicting duplicate code/date prices')
    duplicates = n - len(d)
    d = d.sort_values(KEYS, kind='stable').reset_index(drop=True)
    positive = np.isfinite(d[PRICE_COLUMNS]) & d[PRICE_COLUMNS].gt(0)
    geometry = (d.high >= d[PRICE_COLUMNS].max(axis=1)) & (d.low <= d[PRICE_COLUMNS].min(axis=1))
    d['price_valid'] = positive.all(axis=1) & geometry
    d['volume_valid'] = np.isfinite(d.volume) & d.volume.ge(0)
    d['adj_close_valid'] = np.isfinite(d.adj_close) & d.adj_close.gt(0)
    d.loc[~d.price_valid, PRICE_COLUMNS] = np.nan
    d.loc[~d.volume_valid, 'volume'] = np.nan
    d.loc[~d.adj_close_valid, 'adj_close'] = np.nan
    previous = d.groupby('code', sort=False).close.shift(1)
    d['outlier_flag'] = (d.close / previous - 1).abs().gt(OUTLIER_THRESHOLD)
    # Possible unit/split event; cause UNKNOWN, never automatically repair prices.
    d['split_or_outlier_flag'] = d.outlier_flag
    d['observation_valid'] = d.price_valid & d.volume_valid & d.volume.gt(0) & ~d.outlier_flag
    d.attrs['quality'] = dict(input_rows=n, duplicates_removed=duplicates,
                              invalid_price_rows=int((~d.price_valid).sum()),
                              invalid_volume_rows=int((~d.volume_valid).sum()),
                              zero_volume_rows=int(d.volume.eq(0).sum()),
                              outlier_rows=int(d.outlier_flag.sum()))
    return d


def calendar_panel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Observed date union proxy, NOT a qualified exchange calendar.

    Reindex internally so missing stock sessions cannot be silently compressed.
    Never emit synthetic rows or infer listing/membership from future coverage.
    """
    if prices.duplicated(KEYS).any() or not prices.index.is_unique:
        raise ValueError('duplicate keys or row indices')
    days = pd.DatetimeIndex(prices.date.unique()).sort_values()
    index = pd.MultiIndex.from_product([sorted(prices.code.unique()), days], names=KEYS)
    panel = prices.set_index(KEYS).reindex(index)
    return panel, days


def coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    total = prices.date.nunique()
    rows = []
    for code, p in prices.groupby('code', sort=True):
        rows.append(dict(code=code, rows=len(p), date_min=p.date.min().isoformat(),
                         date_max=p.date.max().isoformat(), union_sessions=total,
                         missing_union_sessions=total-len(p),
                         observed_coverage=len(p)/total, valid_rows=int(p.observation_valid.sum()),
                         invalid_price_rows=int((~p.price_valid).sum()),
                         zero_volume_rows=int(p.volume.eq(0).sum()), outlier_rows=int(p.outlier_flag.sum())))
    return pd.DataFrame(rows)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(frame: pd.DataFrame, path: str | Path | None = None) -> dict:
    """Content hash: UTF-8 CSV, sorted keys/columns, ISO dates, %.17g, NA=null.

    Path is metadata, excluded from sha256; row order and attrs are irrelevant.
    Schema and canonical representation are part of the versioned contract.
    """
    if frame.duplicated(KEYS).any():
        raise ValueError('manifest requires unique code/date keys')
    d = frame.sort_values(KEYS, kind='stable').reindex(sorted(frame.columns), axis=1).copy()
    for c in d.select_dtypes(include=['datetime']).columns:
        d[c] = d[c].dt.strftime('%Y-%m-%d')
    canonical = d.to_csv(index=False, na_rep='null', float_format='%.17g', lineterminator='\n')
    return dict(path=str(Path(path).resolve()) if path is not None else None,
                rows=len(d), codes=int(frame.code.nunique()),
                date_min=frame.date.min().isoformat(), date_max=frame.date.max().isoformat(),
                columns=list(d.columns), canonical_form='sorted-csv-iso-dates-float17-v1',
                sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest())
