"""Month-end all-market quote snapshots -> point-in-time listed universe. RESEARCH_ONLY.

TWSE: rwd/zh/afterTrading/MI_INDEX?type=ALLBUT0999 (verified back to 2008); TPEx: www/zh-tw/afterTrading/dailyQuotes
(verified back to 2008). Raw prices only; ordinary 4-digit codes. A snapshot proves "this code quoted on this day"
(PIT listing evidence); it says nothing about adjusted prices or the cause of a disappearance.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .pit_universe import is_stock_code

TWSE_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json'
TPEX_URL = 'https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?date={y}/{m}/{dd}&type=EW&response=json'


def http_json(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client'}), timeout=timeout) as r:
        return json.loads(r.read())


def _num(x):
    try:
        return float(str(x).replace(',', ''))
    except ValueError:
        return float('nan')


def parse_twse(payload, date):
    best = None
    for t in payload.get('tables', []):
        if '證券代號' in t.get('fields', []) and (best is None or len(t['data']) > len(best['data'])):
            best = t
    if payload.get('stat') != 'OK' or best is None:
        return pd.DataFrame(columns=['market', 'date', 'code', 'name', 'close'])
    f = best['fields']
    i, n, c = f.index('證券代號'), f.index('證券名稱'), f.index('收盤價')
    return pd.DataFrame([dict(market='TWSE', date=pd.Timestamp(date), code=r[i], name=r[n], close=_num(r[c])) for r in best['data']])


def parse_tpex(payload, date):
    if payload.get('stat') != 'ok' or not payload.get('tables'):
        return pd.DataFrame(columns=['market', 'date', 'code', 'name', 'close'])
    t = payload['tables'][0]
    f = t['fields']
    i, n, c = f.index('代號'), f.index('名稱'), f.index('收盤')
    return pd.DataFrame([dict(market='TPEX', date=pd.Timestamp(date), code=r[i], name=r[n], close=_num(r[c])) for r in t['data']])


def legitimate(payload, market):
    """A real answer (data, or the exchange's explicit 'no matching data' for non-trading days). Anything else
    (rate limit, maintenance, malformed) must not be cached as 'no quotes'."""
    stat = str(payload.get('stat', ''))
    return stat == 'OK' or '沒有符合條件' in stat if market == 'TWSE' else stat.lower() == 'ok'


def month_end_snapshot(market, year, month, cache_dir, fetch=http_json, sleep=3., max_back=10, log=print):
    """Last calendar day with quotes in the month (walks back over weekends/holidays). Raw JSON cached, never overwritten."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    day = (pd.Timestamp(year, month, 1)+pd.offsets.MonthEnd(0)).normalize()
    for _ in range(max_back):
        f = cache_dir/f'{market}_{day:%Y%m%d}.json'
        if f.exists():
            payload = json.loads(f.read_text())
        else:
            url = TWSE_URL.format(d=f'{day:%Y%m%d}') if market == 'TWSE' else TPEX_URL.format(y=day.year, m=f'{day.month:02d}', dd=f'{day.day:02d}')
            payload = fetch(url)
            if not legitimate(payload, market):
                raise RuntimeError(f'unexpected {market} payload for {day:%Y-%m-%d}: {str(payload.get("stat"))[:80]} (not cached)')
            f.write_text(json.dumps(payload, ensure_ascii=False))
            time.sleep(sleep)
        df = (parse_twse if market == 'TWSE' else parse_tpex)(payload, day)
        if len(df):
            return df
        day -= pd.Timedelta(days=1)
    log(f'no quotes found for {market} {year}-{month:02d}')
    return pd.DataFrame(columns=['market', 'date', 'code', 'name', 'close'])


def stock_only(df):
    return df[df.code.map(is_stock_code)].reset_index(drop=True)


def presence_matrix(snapshots):
    """code x snapshot-date boolean table (any market)."""
    s = snapshots.drop_duplicates(['date', 'code'])
    return s.assign(p=True).pivot(index='code', columns='date', values='p').fillna(False).astype(bool)


def explain_disappearances(snapshots, events, window_days=45):
    """For every code present at snapshot t and absent at t+1: matched delisting event in (t - window, t+1], a market
    transfer (code present at t+1 on the other market is not a disappearance by construction), or UNEXPLAINED."""
    pm = presence_matrix(snapshots)
    dates = list(pm.columns)
    ev = events[events.stock_like]
    rows = []
    for a, b in zip(dates[:-1], dates[1:]):
        gone = pm.index[pm[a] & ~pm[b]]
        for code in gone:
            e = ev[(ev.code == code) & (ev.date > a-pd.Timedelta(days=window_days)) & (ev.date <= b)]
            rows.append(dict(code=code, last_snapshot=a, next_snapshot=b,
                             event_date=e.date.iloc[0] if len(e) else pd.NaT,
                             reason_class=e.reason_class.iloc[0] if len(e) else None,
                             status='EXPLAINED_BY_DELISTING_EVENT' if len(e) else 'UNEXPLAINED'))
    return pd.DataFrame(rows, columns=['code', 'last_snapshot', 'next_snapshot', 'event_date', 'reason_class', 'status'])


def events_missing_from_snapshots(snapshots, events):
    """Delisting events inside the snapshot span whose code was never seen disappearing (data gaps in the event list or code absent)."""
    pm = presence_matrix(snapshots)
    lo, hi = min(pm.columns), max(pm.columns)
    ev = events[events.stock_like & (events.date > lo) & (events.date <= hi)]
    seen_gone = set(explain_disappearances(snapshots, events).code)
    return ev[~ev.code.isin(seen_gone)]
