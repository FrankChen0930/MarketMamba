import pytest
from research_lab import fill_prices as fp, fetch_core as fc
from test_fetch_core import Resp, Sess

TWSE_FIELDS = ['證券代號', '證券名稱', '成交股數', '成交筆數', '成交金額', '開盤價', '最高價', '最低價', '收盤價']
TPEX_FIELDS = ['代號', '名稱', '收盤', '漲跌', '開盤', '最高', '最低', '成交股數', '成交金額']


def twse(date, n=120, close='10.5'):
    data = [[f'{1000+i}', 'x', '1,000', '5', '9', '10.0', '11.0', '9.5', close] for i in range(n)] + [['00878', 'etf', '1', '1', '1', '1', '1', '1', '1'], ['2330', 'tsmc', '5', '1', '1', '--', '--', '--', '--']]
    return {'stat': 'OK', 'date': date, 'tables': [{'fields': ['a'], 'data': []}, {'fields': TWSE_FIELDS, 'data': data}]}


def tpex(date):
    return {'stat': 'ok', 'date': date, 'tables': [{'fields': TPEX_FIELDS, 'data': [['6488', 'a', '12.5', '0.1', '12.0', '13.0', '11.5', '2,000', '1']]}]}


def test_twse_parse_skips_non_4digit_and_no_trade_rows():
    df = fp.parse(twse('20261113'), '2026-11-13', 'TWSE')
    assert len(df) == 120 and set(df.stock_id) == {f'{1000+i}' for i in range(120)}
    assert df.iloc[0].Open == 10.0 and df.iloc[0].Close == 10.5 and df.iloc[0].Volume == 1000


def test_tpex_close_precedes_open_and_lowercase_ok_accepted():
    s = Sess([Resp(200, tpex('20261113'))])
    r, df = fp.fetch_day(s, '2026-11-13', 'TPEX', sleep=lambda x: None)
    assert r.status == fc.OK and df.iloc[0].Close == 12.5 and df.iloc[0].Open == 12.0 and df.iloc[0].Volume == 2000


def test_response_for_other_day_is_no_data_never_written_under_requested_date():
    s = Sess([Resp(200, tpex('20261112'))])
    r, df = fp.fetch_day(s, '2026-11-13', 'TPEX', sleep=lambda x: None)
    assert r.status == fc.NO_DATA and df is None


def test_layout_drift_is_schema_not_silent_empty():
    body = tpex('20261113'); body['tables'][0]['fields'] = ['代號', '名稱', '收盤', '成交股數']
    r, df = fp.fetch_day(Sess([Resp(200, body)]), '2026-11-13', 'TPEX', sleep=lambda x: None)
    assert r.status == fc.SCHEMA and df is None


def test_fetch_range_collects_and_reports_each_market_status():
    seq = [Resp(200, twse('20261113')), Resp(200, tpex('20261113')), Resp(200, {'stat': '很抱歉，沒有符合條件的資料!'}), Resp(200, {'stat': 'no data'})]
    df, res = fp.fetch_range(Sess(seq), ['2026-11-13', '2026-11-14'], sleep=lambda x: None)
    assert set(df.market) == {'TWSE', 'TPEX'} and res[('2026-11-14', 'TWSE')].status == fc.NO_DATA
    assert fc.coverage(res)['counts']['OK'] == 2


def test_empty_price_table_is_no_data_but_missing_table_is_schema():
    empty = tpex('20261113'); empty['tables'][0]['data'] = []
    r, df = fp.fetch_day(Sess([Resp(200, empty)]), '2026-11-13', 'TPEX', sleep=lambda x: None)
    assert r.status == fc.NO_DATA and df is None
    r, df = fp.fetch_day(Sess([Resp(200, {'stat': 'ok', 'date': '20261113', 'tables': [{'fields': ['x'], 'data': []}]})]), '2026-11-13', 'TPEX', sleep=lambda x: None)
    assert r.status == fc.SCHEMA
