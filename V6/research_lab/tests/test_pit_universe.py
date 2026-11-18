import json
import pandas as pd
import pytest
from research_lab.pit_universe import (roc_date, is_stock_code, classify_reason, build_events, matrix_status,
                                       coverage_table, load_tpex, load_twse, load_finmind)


def test_roc_date_formats_and_errors():
    assert roc_date('115/09/01') == pd.Timestamp('2026-09-01')
    assert roc_date('99-12-30') == pd.Timestamp('2010-12-30')
    assert roc_date('090/01/20') == pd.Timestamp('2001-01-20')
    with pytest.raises(ValueError):
        roc_date('2026-09-01x')


def test_stock_code_filter():
    assert is_stock_code('2330') and is_stock_code('8908')
    assert not any(is_stock_code(c) for c in ('00732', '00747B', '0050', '911622', '030001', 'abc'))


def test_classification_only_uses_explicit_wording():
    assert classify_reason('') == 'NO_REASON_GIVEN'
    assert classify_reason('股份轉換') == 'SHARE_SWAP'
    assert classify_reason('因與花旗(台灣)商業銀行股份有限公司合併，並為消滅公司，核有本中心業務規則第12條之2第1項第19款規定之情事') == 'MERGER'
    assert classify_reason('主動申請下櫃') == 'VOLUNTARY'
    assert classify_reason('經該公司董事會決議向本中心申請終止有價證券櫃檯買賣') == 'VOLUNTARY'
    assert classify_reason('依本中心業務規則第12條之2第1項第1款') == 'UNSPECIFIED_RULE'  # rule citation says nothing about cause


def frame(rows, source, market):
    return pd.DataFrame([dict(code=c, name=n, date=pd.Timestamp(d), market=market, reason=r, source=source) for c, n, d, r in rows])


def test_build_events_merges_sources_by_code_and_date_tolerance():
    a = frame([('4910', '陽慶', '2005-12-01', '股份轉換')], 'tpex_official', 'TPEX')
    b = frame([('4910', '陽慶', '2005-12-02', ''), ('2432', '倚天', '2008-09-01', '')], 'finmind', 'UNKNOWN')
    e = build_events(a, b)
    assert len(e) == 2
    r = e[e.code == '4910'].iloc[0]
    assert r.sources == 'finmind+tpex_official' and r.market == 'TPEX' and r.reason_class == 'SHARE_SWAP'
    assert e[e.code == '2432'].iloc[0].market == 'UNKNOWN'


def test_matrix_status_cases_including_code_reuse():
    ev = frame([('1111', 'a', '2010-06-01', ''), ('2222', 'b', '2010-06-01', ''), ('3333', 'c', '2010-06-01', ''),
                ('4444', 'd', '2010-06-01', ''), ('5555', 'e', '2010-06-01', '')], 'x', 'TWSE')
    ev['stock_like'] = True
    ranges = pd.DataFrame({'first': ['2005-01-03', '2015-01-05', '2005-01-03', '2005-01-03'],
                           'last': ['2010-05-31', '2026-09-04', '2026-09-04', '2008-01-01']},
                          index=['1111', '2222', '3333', '4444'])
    s = matrix_status(ev, ranges).set_index('code').matrix_status
    assert s.to_dict() == {'1111': 'COVERED_ENDS_AT_DELIST', '2222': 'CODE_REUSED_AFTER_DELIST',
                           '3333': 'PRESENT_AFTER_DELIST_OR_TRANSFER', '4444': 'ENDS_BEFORE_DELIST', '5555': 'ABSENT'}


def test_coverage_table_counts_only_stock_like_in_window():
    ev = frame([('1111', 'a', '2010-06-01', ''), ('2222', 'b', '2010-07-01', ''), ('00732', 'etf', '2010-08-01', ''),
                ('3333', 'c', '2001-01-01', '')], 'x', 'TWSE')
    ev['stock_like'] = ev.code.map(lambda c: len(c) == 4)
    ranges = pd.DataFrame({'first': ['2005-01-03'], 'last': ['2010-06-01']}, index=['1111'])
    t = coverage_table(matrix_status(ev, ranges))
    assert list(t.index) == [2010] and t.loc[2010, 'events'] == 2 and t.loc[2010, 'covered_fraction'] == .5


def test_loaders(tmp_path):
    (tmp_path/'t.json').write_text(json.dumps([{'DelistingDate': '098/03/01', 'Company': 'x', 'Code': '2019'}]))
    (tmp_path/'p.json').write_text(json.dumps({'date': '2010', 'tables': [{'totalCount': 9, 'data': [['4733', 'n', '99-12-30', '股份轉換', 'u']]}]}))
    (tmp_path/'f.json').write_text(json.dumps({'data': [{'date': '2008-03-01', 'stock_id': '9915', 'stock_name': 'y'}]}))
    assert load_twse(tmp_path/'t.json').date.iloc[0] == pd.Timestamp('2009-03-01')
    p, declared = load_tpex([tmp_path/'p.json'])
    assert declared == {2010: (9, 1)} and p.market.iloc[0] == 'TPEX'
    assert load_finmind(tmp_path/'f.json').market.iloc[0] == 'UNKNOWN'
