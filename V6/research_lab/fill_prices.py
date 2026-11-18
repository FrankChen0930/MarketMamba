"""Isolated public-endpoint fill of daily OHLCV (TWSE MI_INDEX ALLBUT0999 + TPEx afterTrading/otc type=EW). RESEARCH_ONLY.

Parsers follow the layouts documented in the (read-only) original fetcher.py, including the traps recorded there:
TPEx lists Close BEFORE Open and silently answers with another day's data unless the response date is checked, so every
response is bound to the requested date (date_check) and anything else is NO_DATA, never written under the requested date.
Prices are exchange RAW (unadjusted); they are not the same basis as Yahoo adjusted prices. No credentials, no FinMind.
"""
from __future__ import annotations

import pandas as pd

from . import fetch_core as fc

TWSE = ('https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX', lambda d: dict(response='json', date=d.replace('-', ''), type='ALLBUT0999'))
TPEX = ('https://www.tpex.org.tw/www/zh-tw/afterTrading/otc', lambda d: dict(date=d.replace('-', '/'), type='EW', response='json'))
NEED = ('代號', '收盤', '開盤', '最高', '最低', '成交股數')


class EmptyDay(Exception):
    """Valid price layout, zero rows: a closed day. Cross-check with the other market before trusting it."""


def _num(s):
    try:
        return float(str(s).replace(',', '').strip())
    except (ValueError, TypeError):
        return None


def _resp_date(body):
    return str(body.get('date', '')).replace('-', '').replace('/', '')


def parse(body, date, market):
    """-> DataFrame[Date, stock_id, Open, High, Low, Close, Volume, market]; raises ValueError('SCHEMA ...') on layout drift."""
    rows, seen_layout = [], False
    for t in body.get('tables', []):
        f = t.get('fields')
        if not isinstance(f, list) or not all(isinstance(x, str) for x in f):
            continue
        f = [x.strip() for x in f]
        if market == 'TWSE':
            if not any('開盤' in x for x in f) or len(t.get('data', [])) <= 100:
                continue
            ix = {n: next((i for i, x in enumerate(f) if (n if n != '代號' else '代號') in x), -1) for n in NEED}
        else:
            if not any('代號' in x for x in f):
                continue
            if any(f.count(n) != 1 for n in NEED):
                raise ValueError(f'SCHEMA ambiguous TPEx fields {f}')
            ix = {n: f.index(n) for n in NEED}
        if any(v < 0 for v in ix.values()):
            raise ValueError(f'SCHEMA missing fields {f}')
        seen_layout = True
        for r in t['data']:
            if len(r) <= max(ix.values()):
                continue
            sid = str(r[ix['代號']]).strip()
            c = _num(r[ix['收盤']])
            if not sid.isdigit() or len(sid) != 4 or c is None or c <= 0:
                continue
            rows.append(dict(Date=date, stock_id=sid, Open=_num(r[ix['開盤']]) or c, High=_num(r[ix['最高']]) or c,
                             Low=_num(r[ix['最低']]) or c, Close=c, Volume=_num(r[ix['成交股數']]) or 0., market=market))
    if not rows:
        if seen_layout:                                  # correct layout but zero rows = exchange closed (holiday skeleton)
            raise EmptyDay('price table present but empty')
        raise ValueError('SCHEMA no price table found')
    return pd.DataFrame(rows)


def fetch_day(session, date, market, archive_dir=None, **kw):
    """-> (Result, DataFrame|None). NO_DATA for holidays and for a response whose date differs from the request."""
    url, mk = TWSE if market == 'TWSE' else TPEX
    want = date.replace('-', '')
    r = fc.fetch_json(session, url, mk(date), expect_keys=('tables',), date_check=lambda b: _resp_date(b) == want,
                      archive_dir=archive_dir, key=f'{market}_{want}', **kw)
    if r.status != fc.OK:
        return r, None
    try:
        return r, parse(r.payload, date, market)
    except EmptyDay as e:
        return fc.Result(fc.NO_DATA, None, str(e), r.attempts, r.raw_path), None
    except ValueError as e:
        return fc.Result(fc.SCHEMA, None, str(e), r.attempts, r.raw_path), None


def fetch_range(session, dates, archive_dir=None, sleep=None, pause=1.0, **kw):
    """Fetch both markets for each date. -> (prices DataFrame, {(date, market): Result}). Pauses `pause` s between requests."""
    import time
    sleep = sleep or time.sleep
    frames, results = [], {}
    for d in dates:
        for m in ('TWSE', 'TPEX'):
            r, df = fetch_day(session, d, m, archive_dir, sleep=sleep, **kw)
            results[(d, m)] = r
            if df is not None:
                frames.append(df)
            sleep(pause)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), results
