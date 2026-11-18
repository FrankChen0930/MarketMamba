import json
import pandas as pd
import pytest
from research_lab.pit_snapshots import (parse_twse, parse_tpex, month_end_snapshot, stock_only, presence_matrix,
                                        explain_disappearances, events_missing_from_snapshots)

TW = {'stat': 'OK', 'tables': [{'title': 'x', 'fields': ['證券代號', '證券名稱', '收盤價'], 'data': [['2330', '台積電', '1,000.00'], ['0050', 'ETF', '90']]},
                               {'fields': ['a'], 'data': []}]}
TP = {'stat': 'ok', 'tables': [{'fields': ['代號', '名稱', '收盤'], 'data': [['8069', '元太', '120.5'], ['006201', 'ETF', '10']]}]}


def test_parsers_and_stock_filter():
    a = parse_twse(TW, '2012-01-31'); b = parse_tpex(TP, '2012-01-31')
    assert a.close.tolist() == [1000., 90.] and a.market.iloc[0] == 'TWSE' and b.market.iloc[0] == 'TPEX'
    assert stock_only(pd.concat([a, b])).code.tolist() == ['2330', '8069']
    assert parse_twse({'stat': '很抱歉，沒有符合條件的資料!'}, '2012-02-04').empty and parse_tpex({'stat': 'error'}, '2012-02-04').empty


def test_month_end_walks_back_caches_and_never_refetches(tmp_path):
    calls = []
    def fetch(url):
        calls.append(url)
        return TW if '20120130' in url else {'stat': '很抱歉，沒有符合條件的資料!'}      # 31 Jan empty (holiday), 30 Jan has data
    df = month_end_snapshot('TWSE', 2012, 1, tmp_path, fetch=fetch, sleep=0)
    assert df.date.iloc[0] == pd.Timestamp('2012-01-30') and len(calls) == 2
    month_end_snapshot('TWSE', 2012, 1, tmp_path, fetch=fetch, sleep=0)
    assert len(calls) == 2                                              # fully served from cache
    assert (tmp_path/'TWSE_20120130.json').exists()
    empty = month_end_snapshot('TWSE', 2012, 2, tmp_path, fetch=lambda u: {'stat': '很抱歉，沒有符合條件的資料!'}, sleep=0, max_back=2, log=lambda *_: None)
    assert empty.empty


def snaps(rows):
    return pd.DataFrame([dict(market=m, date=pd.Timestamp(d), code=c, name='n', close=1.) for d, m, c in rows])


def events(rows):
    e = pd.DataFrame([dict(code=c, name='n', date=pd.Timestamp(d), reason_class=r) for c, d, r in rows])
    e['stock_like'] = True
    return e


def test_disappearances_explained_unexplained_and_transfer_not_flagged():
    s = snaps([('2012-01-31', 'TPEX', '1111'), ('2012-01-31', 'TPEX', '2222'), ('2012-01-31', 'TPEX', '3333'),
               ('2012-02-29', 'TWSE', '3333')])                          # 3333 moved TPEX->TWSE: still present
    ev = events([('1111', '2012-02-10', 'MERGER')])
    d = explain_disappearances(s, ev).set_index('code')
    assert set(d.index) == {'1111', '2222'} and 'transfer' not in d
    assert d.loc['1111', 'status'] == 'EXPLAINED_BY_DELISTING_EVENT' and d.loc['1111', 'reason_class'] == 'MERGER'
    assert d.loc['2222', 'status'] == 'UNEXPLAINED'


def test_events_missing_from_snapshots_flags_event_list_vs_quotes_disagreement():
    s = snaps([('2012-01-31', 'TWSE', '1111'), ('2012-01-31', 'TWSE', '2222'), ('2012-02-29', 'TWSE', '2222')])
    ev = events([('1111', '2012-02-10', 'MERGER'), ('9999', '2012-02-15', None)])     # 9999 never quoted in either snapshot
    miss = events_missing_from_snapshots(s, ev)
    assert miss.code.tolist() == ['9999']


def test_error_payloads_are_never_cached_as_no_data(tmp_path):
    with pytest.raises(RuntimeError, match='not cached'):
        month_end_snapshot('TWSE', 2012, 1, tmp_path, fetch=lambda u: {'stat': '查詢過於頻繁，請稍後再試'}, sleep=0)
    assert not list(tmp_path.glob('*.json'))
    with pytest.raises(RuntimeError):
        month_end_snapshot('TPEX', 2012, 1, tmp_path, fetch=lambda u: {'stat': 'error'}, sleep=0)
