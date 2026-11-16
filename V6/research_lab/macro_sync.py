"""Incremental, classified macro-data sync (replacement design for fetcher._sync_macro_data). Public yfinance data only.

Defects of the production version this fixes (see tasks/roadmap-20261115.md 12.21):
  * `if cache.exists(): return` -> the file was never refreshed after its first build;
  * empty/failed download -> `continue` with no retry and no record -> downstream silently fills defaults (VIX=20 ...).
Here: every ticker gets a status (OK / NO_DATA / TRANSIENT_FAILED / SCHEMA), only transient errors are retried, existing
non-NaN values are never overwritten, nothing is written if nothing was fetched, and a freshness report names every
column whose last valid value is too old. No default value is ever generated.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = {'^VIX': 'VIX', '^GSPC': 'SPX', '^TWII': 'TWII_Close', 'GC=F': 'Gold', 'CL=F': 'Oil', '^TNX': 'TNX', 'TWD=X': 'USD_TWD'}
OK, NO_DATA, FAILED, SCHEMA = 'OK', 'NO_DATA', 'TRANSIENT_FAILED', 'SCHEMA'


def yf_download(ticker, start, end):
    import yfinance as yf
    return yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)


def _close(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if 'Close' not in df.columns:
        raise KeyError('Close')
    s = df['Close'].astype(float); s.index = pd.to_datetime(s.index).tz_localize(None).normalize(); s.index.name = 'Date'
    return s.dropna()


def fetch_one(ticker, start, end, download=yf_download, retries=3, base_delay=2., sleep=time.sleep):
    """-> (status, Series|None, attempts, reason). Only exceptions are retried; an empty frame is NO_DATA (not retried)."""
    last = ''
    for a in range(1, retries+1):
        try:
            df = download(ticker, start, end)
        except Exception as e:                                       # network / rate-limit / provider error
            last = f'{type(e).__name__}: {e}'
            if a < retries:
                sleep(base_delay*2**(a-1))
            continue
        if df is None or len(df) == 0:
            return NO_DATA, None, a, 'empty response'
        try:
            s = _close(df)
        except KeyError as e:
            return SCHEMA, None, a, f'missing column {e}'
        except (ValueError, TypeError) as e:
            return SCHEMA, None, a, f'unparseable payload: {type(e).__name__}: {e}'
        return (OK, s, a, '') if len(s) else (NO_DATA, None, a, 'no valid Close')
    return FAILED, None, retries, last


def freshness(df, today, max_lag_days=7, expected=()):
    """Per column: last valid date, lag in calendar days, stale flag. A column stale beyond `max_lag_days` is reported, never defaulted."""
    d = df.copy(); d['Date'] = pd.to_datetime(d['Date']); d = d.set_index('Date')
    out = {}
    for c in list(dict.fromkeys(list(d.columns)+list(expected))):             # an expected column that is absent / all-NaN is stale, not invisible
        if c not in d.columns:
            out[c] = dict(last_valid=None, lag_days=None, stale=True); continue
        v = d[c].dropna()
        last = v.index.max() if len(v) else None
        lag = (pd.Timestamp(today)-last).days if last is not None else None
        out[c] = dict(last_valid=None if last is None else str(last.date()), lag_days=lag, stale=lag is None or lag > max_lag_days)
    return out


SAME_DAY_OK = {'TWII_Close'}      # Taiwan index is closed when the 22:30 job runs; US/FX/commodity bars dated `today` are still forming


def sync_macro(path, today, tickers=TICKERS, download=yf_download, overlap_days=10, retries=3, sleep=time.sleep, max_lag_days=7, same_day_ok=SAME_DAY_OK, now=None, tw_close_minutes=13*60+45, conflict_tol=.01):
    """Update `path` (parquet with Date + one column per ticker) up to `today`. Returns a report dict; writes only if something new arrived."""
    now = pd.Timestamp.now(tz='Asia/Taipei') if now is None else pd.Timestamp(now)
    if today is None:
        today = now.strftime('%Y-%m-%d')                                           # Taipei calendar day, not the host's
    tw_closed = pd.Timestamp(today).date() < now.date() or now.hour*60+now.minute >= tw_close_minutes
    path = Path(path)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['Date']+list(tickers.values()))
    old['Date'] = pd.to_datetime(old['Date']); m = old.set_index('Date').sort_index()
    end = (pd.Timestamp(today)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')            # yfinance `end` is exclusive
    rep, got = {}, {}
    for tk, col in tickers.items():
        last = m[col].dropna().index.max() if col in m.columns and m[col].notna().any() else None
        start = (last-pd.Timedelta(days=overlap_days)).strftime('%Y-%m-%d') if last is not None else '2005-01-01'
        st, s, att, why = fetch_one(tk, start, end, download, retries, sleep=sleep)
        rep[col] = dict(status=st, attempts=att, reason=why, rows=0 if s is None else len(s))
        if s is not None:
            if col not in same_day_ok or not tw_closed:                                 # an unfinished bar must never be stored (it is never overwritten)
                s = s[s.index < pd.Timestamp(today).normalize()]
            if col in m.columns:                                                        # provider vs stored disagreement is reported, not silently dropped
                j = pd.concat([m[col], s], axis=1, keys=['o', 'n'], sort=True).dropna()
                bad = j[(abs(j.o-j.n)/abs(j.o).where(j.o != 0)) > conflict_tol]
                rep[col]['conflicts'] = dict(n=len(bad), dates=[str(d.date()) for d in bad.index[:10]], max_rel=float((abs(bad.o-bad.n)/abs(bad.o)).max()) if len(bad) else 0.)
            if len(s):
                got[col] = s
    if got:
        new = m.reindex(m.index.union(pd.DatetimeIndex(sorted(set().union(*[s.index for s in got.values()]))))).sort_index()
        for col, s in got.items():
            if col not in new.columns:
                new[col] = np.nan
            add = s.reindex(new.index)
            take = new[col].isna() & add.notna()                                      # never overwrite an existing value
            new.loc[take, col] = add[take]
            rep[col]['filled'] = int(take.sum())
        new.index.name = 'Date'
        out = new.reset_index()
        if any(rep[c].get('filled', 0) for c in got):
            tmp = path.with_suffix('.tmp')
            try:
                out.to_parquet(tmp); os.replace(tmp, path)                                # atomic
            finally:
                if tmp.exists(): tmp.unlink()
            rep['_written'] = True
    cur = pd.read_parquet(path) if path.exists() else pd.DataFrame({'Date': pd.to_datetime([])})
    rep['_freshness'] = freshness(cur, today, max_lag_days, expected=list(tickers.values()))
    rep['_stale'] = sorted(c for c, f in rep['_freshness'].items() if f['stale'] and c != 'Date')
    return rep
