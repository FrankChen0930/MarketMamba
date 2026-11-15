"""Delisting event table + coverage vs a matrix universe. RESEARCH_ONLY.

Sources (all public, anonymous): TWSE openapi company/suspendListingCsvAndHtml (code, name, ROC date;
no reason), TPEx www/zh-tw/company/deListed?date=<year> (reason text; server returns only first 10 rows per
year, totalCount gives true size), FinMind TaiwanStockDelisting (date, id, name; no market/reason).
Reason text is mostly a rule citation; only explicit wording is classified, the rest stays UNSPECIFIED_RULE.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def roc_date(text):
    m = re.fullmatch(r'\s*(\d{2,3})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*', str(text))
    if not m:
        raise ValueError(f'bad ROC date: {text!r}')
    y, mo, d = map(int, m.groups())
    return pd.Timestamp(y+1911, mo, d)


def is_stock_code(code):
    """Ordinary 4-digit stock codes (ETF/ETN/warrants/bonds excluded)."""
    return bool(re.fullmatch(r'[1-9]\d{3}', str(code)))


def load_twse(path):
    d = pd.DataFrame(json.loads(Path(path).read_text()))
    return pd.DataFrame(dict(code=d.Code.astype(str), name=d.Company, date=d.DelistingDate.map(roc_date),
                             market='TWSE', reason='', source='twse_official'))


def load_tpex(paths):
    rows, declared = [], {}
    for p in paths:
        j = json.loads(Path(p).read_text())
        t = j['tables'][0]
        declared[int(j['date'])] = (int(t.get('totalCount', 0)), len(t['data']))
        for r in t['data']:
            rows.append(dict(code=str(r[0]), name=r[1], date=roc_date(r[2]), market='TPEX',
                             reason=(r[3] or '').strip(), source='tpex_official'))
    return pd.DataFrame(rows, columns=['code', 'name', 'date', 'market', 'reason', 'source']), declared


def load_finmind(path):
    d = pd.DataFrame(json.loads(Path(path).read_text())['data'])
    return pd.DataFrame(dict(code=d.stock_id.astype(str), name=d.stock_name, date=pd.to_datetime(d.date),
                             market='UNKNOWN', reason='', source='finmind'))


EXPLICIT = [  # (pattern, class) - only wording that states the cause itself
    (r'合併|消滅公司', 'MERGER'), (r'股份轉換', 'SHARE_SWAP'), (r'轉(列)?上市|轉(列)?上櫃|轉板', 'MARKET_TRANSFER'),
    (r'主動|申請終止|董事會決議向|股東臨時會決議向|股東會決議向', 'VOLUNTARY'), (r'管理股票', 'MANAGED_STOCK_TRANSFER'),
]


def classify_reason(text):
    t = (text or '').strip()
    if not t:
        return 'NO_REASON_GIVEN'
    for pat, cls in EXPLICIT:
        if re.search(pat, t):
            return cls
    return 'UNSPECIFIED_RULE' if re.search(r'規則|辦法|規定', t) else 'OTHER_TEXT'


def build_events(*frames, tolerance_days=7):
    """Union of source tables, one row per (code, delisting event); sources flagged, reason from the richest source."""
    allr = pd.concat([f for f in frames if len(f)], ignore_index=True).sort_values(['code', 'date']).reset_index(drop=True)
    out, cur = [], None
    for r in allr.itertuples(index=False):
        if cur is not None and cur['code'] == r.code and abs((r.date-cur['date']).days) <= tolerance_days:
            cur['sources'].add(r.source)
            if cur['market'] == 'UNKNOWN' and r.market != 'UNKNOWN':
                cur['market'] = r.market
            if not cur['reason'] and r.reason:
                cur['reason'] = r.reason
        else:
            if cur is not None:
                out.append(cur)
            cur = dict(code=r.code, name=r.name, date=r.date, market=r.market, reason=r.reason, sources={r.source})
    if cur is not None:
        out.append(cur)
    e = pd.DataFrame(out)
    e['sources'] = e.sources.map(lambda s: '+'.join(sorted(s)))
    e['stock_like'] = e.code.map(is_stock_code)
    e['reason_class'] = e.reason.map(classify_reason)
    return e.sort_values(['date', 'code']).reset_index(drop=True)


def matrix_status(events, ranges, end_tolerance_days=10):
    """ranges: DataFrame index=code with first,last (Timestamps). Status per delisting event."""
    r = ranges.copy()
    r['first'], r['last'] = pd.to_datetime(r['first']), pd.to_datetime(r['last'])
    m = events.join(r, on='code')
    def status(row):
        if pd.isna(row['first']):
            return 'ABSENT'
        if row['first'] > row['date']:
            return 'CODE_REUSED_AFTER_DELIST'   # present only after the delisting: a different/new listing
        if abs((row['last']-row['date']).days) <= end_tolerance_days:
            return 'COVERED_ENDS_AT_DELIST'
        if row['last'] > row['date']:
            return 'PRESENT_AFTER_DELIST_OR_TRANSFER'
        return 'ENDS_BEFORE_DELIST'
    m['matrix_status'] = m.apply(status, axis=1)
    return m


def coverage_table(events_status, start='2005-01-01', end=None):
    e = events_status[(events_status.date >= pd.Timestamp(start)) & events_status.stock_like]
    if end:
        e = e[e.date <= pd.Timestamp(end)]
    by_year = e.groupby([e.date.dt.year.rename('year'), 'matrix_status']).size().unstack(fill_value=0)
    by_year['events'] = by_year.sum(axis=1)
    covered = by_year.get('COVERED_ENDS_AT_DELIST', 0)
    by_year['covered_fraction'] = covered/by_year['events']
    return by_year
