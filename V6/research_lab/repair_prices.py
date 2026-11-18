"""Repair interior holes in production prices_raw.parquet from exchange-direct (TWSE/TPEx) data. Dry-run by default.

Why: the daily job fetches only "today" for prices; a day that failed is never retried once a later day succeeded, so
prices_raw had missing sessions and sessions with 13-18 of ~1,950 stocks (see tasks/roadmap-20261115.md 12.25). Downstream
daytrade / market_value use prices_raw Volume as denominator, so prices must be repaired FIRST.

Safety: (1) additions only -- an existing (Date, stock_id) row is never changed; (2) a day is repaired only if the exchange
data covers >= `min_cover` of the typical session size; (3) bad rows (non-positive price, High<Low, negative volume) are dropped,
never written; (4) apply = backup -> temp file -> verify -> atomic replace; refuses inside the nightly-update window.
Exchange RAW prices are the same basis as production prices_raw (checked 2026-11-16: Close identical on 3 sample days).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ['Date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Volume']
UPDATE_WINDOW = (22*60+20, 22*60+50)          # Taipei minutes-of-day during which the nightly job writes prices_raw


def trading_calendar(macro_path, start='2026-04-01'):
    """Taiwan trading days = dates where the TWII close exists (macro_raw)."""
    m = pd.read_parquet(macro_path, columns=['Date', 'TWII_Close']); m['Date'] = pd.to_datetime(m['Date'])
    return pd.DatetimeIndex(sorted(m.loc[m.TWII_Close.notna() & (m.Date >= start), 'Date']))


def day_counts(prices):
    return prices.groupby('Date').size()


def clean_fill(fill):
    f = fill[COLS].copy()
    f['Date'] = pd.to_datetime(f['Date']).dt.strftime('%Y-%m-%d'); f['stock_id'] = f['stock_id'].astype(str)
    px = f[['Open', 'High', 'Low', 'Close']]
    ok = np.isfinite(px).all(axis=1) & (px > 0).all(axis=1) & (f.High >= f.Low) & np.isfinite(f.Volume) & (f.Volume >= 0)
    return f[ok].drop_duplicates(['Date', 'stock_id']), int((~ok).sum())


def plan(prices, fill, calendar, bad_frac=.5, min_cover=.9, typical_window=60):
    """prices: production frame (Date as 'YYYY-MM-DD'); returns (additions frame, per-day report)."""
    cnt = day_counts(prices); cnt.index = pd.to_datetime(cnt.index)
    typical = float(cnt.loc[cnt.index.isin(calendar)].tail(typical_window).median())
    f, dropped = clean_fill(fill)
    key = set(zip(prices.Date, prices.stock_id))
    rep, add = {}, []
    for d in calendar:
        have = int(cnt.get(d, 0))
        if have >= bad_frac*typical:
            continue
        ds = d.strftime('%Y-%m-%d'); fd = f[f.Date == ds]
        new = fd[[(a, b) not in key for a, b in zip(fd.Date, fd.stock_id)]]
        status = 'REPAIR' if len(fd) >= min_cover*typical else 'REFUSED_EXCHANGE_COVERAGE_TOO_LOW'
        rep[ds] = dict(existing=have, exchange=int(len(fd)), to_add=int(len(new)) if status == 'REPAIR' else 0, status=status)
        if status == 'REPAIR':
            add.append(new)
    nontrading = {k.strftime('%Y-%m-%d'): int(v) for k, v in cnt.items() if k not in calendar and calendar.min() <= k <= calendar.max()}
    out = pd.concat(add, ignore_index=True) if add else pd.DataFrame(columns=COLS)
    return out, dict(typical_rows_per_day=typical, exchange_rows_dropped_invalid=dropped, days=rep, rows_on_non_trading_days_NOT_TOUCHED=nontrading, total_to_add=int(len(out)))


def _digest(df):
    return int(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).astype('uint64').sum() % (1 << 63))


def apply(target, additions, backup_dir, now=None, force=False):
    target = Path(target)
    now = pd.Timestamp.now(tz='Asia/Taipei') if now is None else pd.Timestamp(now)
    mod = now.hour*60 + now.minute
    if not force and UPDATE_WINDOW[0] <= mod <= UPDATE_WINDOW[1]:
        raise RuntimeError('refusing to write inside the nightly update window (22:20-22:50 Taipei)')
    before = pd.read_parquet(target)
    assert list(before.columns) == COLS, 'unexpected schema'
    add = additions[COLS].astype(before.dtypes.to_dict())
    keys_b = set(zip(before.Date, before.stock_id)); assert not any(k in keys_b for k in zip(add.Date, add.stock_id)), 'addition collides with existing row'
    backup_dir = Path(backup_dir); backup_dir.mkdir(parents=True, exist_ok=True)
    bk = backup_dir/f'{target.stem}.before_repair.parquet'; assert not bk.exists(), 'backup already exists'
    shutil.copy2(target, bk)
    out = pd.concat([before, add], ignore_index=True).sort_values(['stock_id', 'Date']).reset_index(drop=True)
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    try:
        out.to_parquet(tmp, index=False)
        chk = pd.read_parquet(tmp)
        assert len(chk) == len(before)+len(add), 'row count mismatch'
        assert not chk.duplicated(['Date', 'stock_id']).any(), 'duplicate keys after repair'
        assert chk.dtypes.to_dict() == before.dtypes.to_dict(), 'dtype changed'
        tdays = set(add.Date)
        assert _digest(chk[~chk.Date.isin(tdays)].sort_values(['stock_id', 'Date'])) == _digest(before[~before.Date.isin(tdays)].sort_values(['stock_id', 'Date'])), 'untouched rows changed'
        old_t = before[before.Date.isin(tdays)].merge(chk, on=['Date', 'stock_id'], suffixes=('', '_n'))
        assert all((old_t[c] == old_t[c+'_n']).all() for c in COLS[2:]), 'existing rows on repaired days changed'
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dict(backup=str(bk), rows_before=len(before), rows_after=len(chk), added=len(add))


def main(argv=None):
    import argparse
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--target', required=True); a.add_argument('--fill', required=True); a.add_argument('--macro', required=True)
    a.add_argument('--backup-dir'); a.add_argument('--apply', action='store_true')
    a = a.parse_args(argv)
    prices = pd.read_parquet(a.target); fill = pd.read_parquet(a.fill)
    add, rep = plan(prices, fill, trading_calendar(a.macro))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.apply:
        assert a.backup_dir, '--backup-dir required with --apply'
        print(json.dumps(apply(a.target, add, a.backup_dir), indent=1))
    else:
        print('DRY RUN: nothing written (use --apply)')


if __name__ == '__main__':
    main()
