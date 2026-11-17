"""Offline contracts; all writes are confined to pytest temporary directories."""
import hashlib
import json

import pandas as pd
import pytest

from research_lab import repair_derived as rd

DAY = '2026-09-24'
IDS = [str(1000+i) for i in range(20)]


def frames():
    prices = pd.DataFrame({'Date': DAY, 'stock_id': IDS, 'Close': 5., 'Volume': 100.})
    old = pd.DataFrame({'Date': pd.to_datetime([DAY]), 'stock_id': IDS[:1], 'Day_Trade_Volume': [.1]})
    fill = pd.DataFrame({'Date': pd.to_datetime([DAY]*20), 'stock_id': IDS, 'Day_Trade_Volume': .5})
    return old, prices, fill


def test_coverage_below_90_percent_is_refused():
    old, prices, fill = frames()
    add, report = rd.plan(old, prices, fill.iloc[:17], [DAY], 'daytrade')
    assert add.empty
    assert report[DAY]['status'] == 'REFUSED_COVERAGE'
    assert report[DAY]['to_add'] == 0


def test_boundary_90_percent_and_conflicts_only_reported():
    old, prices, fill = frames()
    add, report = rd.plan(old, prices, fill.iloc[:18], [DAY], 'daytrade')
    assert len(add) == 17
    assert report[DAY]['conflicts'] == 1
    assert IDS[0] not in set(add.stock_id)
    assert old.Day_Trade_Volume.tolist() == [.1]


@pytest.mark.parametrize('instant', ['2026-11-17 22:20+08:00', '2026-11-17 22:50+08:00', '2026-11-17 14:35Z'])
def test_window_prevents_any_write(tmp_path, instant):
    old, prices, fill = frames(); target = tmp_path/'daytrade.parquet'; old.to_parquet(target)
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match='window'):
        rd.apply(target, prices, fill, [DAY], 'daytrade', tmp_path/'backup', now=instant)
    assert target.read_bytes() == before
    assert not (tmp_path/'backup').exists()


def test_apply_preserves_existing_rows_schema_and_backups(tmp_path):
    old, prices, fill = frames(); target = tmp_path/'daytrade.parquet'; old.to_parquet(target, index=False)
    before = target.read_bytes()
    rd.apply(target, prices, fill, [DAY], 'daytrade', tmp_path/'backup', now='2026-11-17 10:00+08:00')
    out = pd.read_parquet(target)
    assert len(out) == 20
    pd.testing.assert_frame_equal(out[out.stock_id == IDS[0]].reset_index(drop=True), old)
    assert out.dtypes.to_dict() == old.dtypes.to_dict()
    assert next((tmp_path/'backup').glob('*.parquet')).read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))


def test_apply_also_enforces_coverage(tmp_path):
    old, prices, fill = frames(); target = tmp_path/'daytrade.parquet'; old.to_parquet(target)
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match='REFUSED'):
        rd.apply(target, prices, fill.iloc[:17], [DAY], 'daytrade', tmp_path/'backup', now='2026-11-17 10:00+08:00')
    assert target.read_bytes() == before


def test_sunday_and_holiday_are_untouched():
    old, prices, fill = frames()
    for day, calendar in [('2026-09-20', ['2026-09-20']), (DAY, [])]:
        p = prices.assign(Date=day); f = fill.assign(Date=pd.Timestamp(day))
        add, _ = rd.plan(old, p, f, calendar, 'daytrade')
        assert add.empty


def test_ratios_use_prices_volume_and_drop_missing_zero_and_over_one():
    _, prices, _ = frames()
    raw = pd.DataFrame({'Date': DAY, 'stock_id': IDS[:4]+['9999'], 'DayTrade_Shares': [50., 101., 0., 25., 50.]})
    prices.loc[2, 'Volume'] = 0
    result = rd.derive('daytrade', raw, prices)
    assert result.stock_id.tolist() == [IDS[0], IDS[3]]
    assert result.Day_Trade_Volume.tolist() == [.5, .25]


def test_market_value_is_close_times_current_mops_shares():
    _, prices, _ = frames()
    raw = pd.DataFrame({'stock_id': IDS[:2], 'shares_outstanding': [1000., 2000.]})
    out = rd.derive('market_value', raw, prices)
    assert out.market_value.tolist() == [5000., 10000.]


@pytest.mark.parametrize('market,width,indices', [('TWSE',19,[2,3,4,8,9,10,11]), ('TPEX',24,[2,3,4,11,12,13,22])])
def test_institutional_matches_fetcher_indices_and_legacy_nan(market,width,indices):
    row = ['1000', 'name'] + [str(i*1000) for i in range(2,width)]
    fields = (['證券代號', '證券名稱', '外陸資買進股數(不含外資自營商)',
               '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
               '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
               '投信買進股數', '投信賣出股數', '投信買賣超股數', '自營商買賣超股數',
               '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
               '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)', '三大法人買賣超股數']
              if market == 'TWSE' else ['代號', '名稱'] +
              ['買進股數', '賣出股數', '買賣超股數']*7 + ['三大法人買賣超股數合計'])
    body = {'stat':'OK', 'date':'20260924', 'data':[row], 'fields':fields,
            'tables':[{'data':[row], 'fields':fields}]}
    out = rd.parse_institutional(body, DAY, market)
    assert out[rd.INST_VALUES].iloc[0].tolist() == [i*1000 for i in indices]
    assert out[['buy_Dealer', 'sell_Dealer']].isna().all().all()
    fields[2], fields[3] = fields[3], fields[2]
    with pytest.raises(ValueError, match='schema'):
        rd.parse_institutional(body, DAY, market)


def test_response_date_mismatch_and_schema_drift_fail_closed():
    with pytest.raises(ValueError, match='date'):
        rd.parse_institutional({'date':'20260923'}, DAY, 'TWSE')
    with pytest.raises(ValueError):
        rd.parse_daytrade({'date':'20260924', 'tables':[{'fields':['代號'], 'data':[['1000']]}]}, DAY)


def test_duplicate_fill_and_schema_mismatch_are_rejected():
    old, prices, fill = frames()
    for bad in [pd.concat([fill,fill]), fill.drop(columns='Day_Trade_Volume')]:
        with pytest.raises(ValueError):
            rd.plan(old, prices, bad, [DAY], 'daytrade')


def test_archive_is_content_addressed_and_verifiable(tmp_path):
    path = rd.archive(tmp_path, b'{"date":"20260924"}', 'test', {'url':'https://example.test'})
    assert hashlib.sha256(path.read_bytes()).hexdigest() in path.name
    meta = json.loads(path.with_suffix('.manifest.json').read_text())
    assert meta['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_reads_bounded_rows_and_defaults_to_dry_run(tmp_path, monkeypatch):
    old, prices, fill = frames()
    target = tmp_path/'daytrade.parquet'; old.to_parquet(target, index=False)
    price_path = tmp_path/'prices.parquet'; prices.to_parquet(price_path, index=False)
    pd.DataFrame({'Date':[DAY], 'TWII_Close':[100.]}).to_parquet(tmp_path/'macro_raw.parquet')
    before = target.read_bytes()
    seen = []
    read = rd.pd.read_parquet
    def bounded(path, *args, **kwargs):
        if path in [target, price_path]:
            seen.append(kwargs.get('filters'))
            assert kwargs.get('filters'), 'CLI must not load the entire history'
        return read(path, *args, **kwargs)
    monkeypatch.setattr(rd.pd, 'read_parquet', bounded)
    raw = pd.DataFrame({'Date':DAY, 'stock_id':IDS, 'DayTrade_Shares':50.})
    monkeypatch.setattr(rd, 'fetch', lambda *a: (raw, []))
    monkeypatch.setattr(rd, 'apply', lambda *a: pytest.fail('dry-run called writer'))
    monkeypatch.chdir(tmp_path)
    assert rd.main(['--dataset','daytrade','--target',str(target),'--prices',str(price_path)]) == 0
    assert len(seen) == 2 and target.read_bytes() == before
    report = json.loads(next((tmp_path/'.local-data/public_fill').glob('*/report.json')).read_text())
    assert report['total_to_add'] == 19 and report['inputs_unchanged']


def test_market_value_rejects_negative_inputs_even_when_product_positive():
    _, prices, _ = frames()
    prices.loc[0, 'Close'] = -5.
    raw = pd.DataFrame({'stock_id':IDS[:2], 'shares_outstanding':[-1000., float('inf')]})
    assert rd.derive('market_value', raw, prices).empty


def test_entering_update_window_during_serialization_aborts(tmp_path):
    old, prices, fill = frames(); target = tmp_path/'daytrade.parquet'; old.to_parquet(target, index=False)
    before = target.read_bytes()
    moments = iter([pd.Timestamp('2026-11-17 22:19+08:00'), pd.Timestamp('2026-11-17 22:20+08:00')])
    with pytest.raises(RuntimeError, match='window'):
        rd.apply(target, prices, fill, [DAY], 'daytrade', tmp_path/'backup', clock=lambda: next(moments))
    assert target.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
