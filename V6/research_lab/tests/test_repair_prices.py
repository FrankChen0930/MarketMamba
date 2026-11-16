import numpy as np, pandas as pd, pytest
from research_lab import repair_prices as rp

DAYS = pd.to_datetime(['2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24', '2026-09-25'])
IDS = [f'{1000+i}' for i in range(40)]


def rows(day, ids, px=100.):
    d = pd.DataFrame({'Date': day.strftime('%Y-%m-%d'), 'stock_id': ids})
    for c, v in zip(['Open', 'High', 'Low', 'Close'], [px, px+1, px-1, px]): d[c] = float(v)
    d['Volume'] = 1000.; return d


def prod():
    # 9/21, 9/22 full; 9/23 partial (3 stocks); 9/24 missing entirely; 9/25 full
    return pd.concat([rows(DAYS[0], IDS), rows(DAYS[1], IDS), rows(DAYS[2], IDS[:3]), rows(DAYS[4], IDS)], ignore_index=True)


def fill(px=101.):
    return pd.concat([rows(d, IDS, px) for d in DAYS], ignore_index=True).assign(market='TWSE')


def test_plan_finds_missing_and_partial_days_and_adds_only_absent_keys():
    add, rep = rp.plan(prod(), fill(), pd.DatetimeIndex(DAYS))
    assert set(rep['days']) == {'2026-09-23', '2026-09-24'}
    assert rep['days']['2026-09-23'] == dict(existing=3, exchange=40, to_add=37, status='REPAIR')
    assert rep['days']['2026-09-24']['to_add'] == 40 and rep['total_to_add'] == 77 and len(add) == 77
    assert not ((add.Date == '2026-09-23') & add.stock_id.isin(IDS[:3])).any()      # existing rows are never re-added


def test_exchange_coverage_too_low_is_refused_not_partially_written():
    f = fill(); f = f[~((f.Date == '2026-09-24') & f.stock_id.isin(IDS[5:]))]       # exchange only has 5 of 40
    add, rep = rp.plan(prod(), f, pd.DatetimeIndex(DAYS))
    assert rep['days']['2026-09-24']['status'] == 'REFUSED_EXCHANGE_COVERAGE_TOO_LOW' and (add.Date != '2026-09-24').all()


def test_invalid_exchange_rows_are_dropped():
    f = fill(); f.loc[f.index[0], 'Close'] = 0.; f.loc[f.index[1], ['High', 'Low']] = [5., 9.]; f.loc[f.index[2], 'Volume'] = -1.
    _, rep = rp.plan(prod(), f, pd.DatetimeIndex(DAYS))
    assert rep['exchange_rows_dropped_invalid'] == 3


def test_non_trading_day_rows_are_reported_but_never_touched():
    p = pd.concat([prod(), rows(pd.Timestamp('2026-09-26'), IDS[:2])], ignore_index=True)
    add, rep = rp.plan(p, fill(), pd.DatetimeIndex(DAYS))
    assert rep['rows_on_non_trading_days_NOT_TOUCHED'] == {'2026-09-26': 2}
    assert (add.Date != '2026-09-26').all()


def test_apply_adds_rows_keeps_existing_values_and_dtypes(tmp_path):
    t = tmp_path/'prices_raw.parquet'; p = prod(); p.to_parquet(t, index=False)
    add, _ = rp.plan(p, fill(px=555.), pd.DatetimeIndex(DAYS))                      # exchange px differs: existing rows must still win
    r = rp.apply(t, add, tmp_path/'bk', now='2026-11-17 10:00+08:00')
    out = pd.read_parquet(t)
    assert r['added'] == 77 and len(out) == len(p)+77 and not out.duplicated(['Date', 'stock_id']).any()
    assert (out[(out.Date == '2026-09-23') & out.stock_id.isin(IDS[:3])].Close == 100.).all()
    assert (out[(out.Date == '2026-09-24')].Close == 555.).all()
    assert out.dtypes.to_dict() == p.dtypes.to_dict() and (tmp_path/'bk'/'prices_raw.before_repair.parquet').exists()
    assert list(tmp_path.glob('*.tmp')) == []


def test_apply_refuses_inside_nightly_window_unless_forced(tmp_path):
    t = tmp_path/'prices_raw.parquet'; prod().to_parquet(t, index=False)
    add, _ = rp.plan(prod(), fill(), pd.DatetimeIndex(DAYS))
    with pytest.raises(RuntimeError):
        rp.apply(t, add, tmp_path/'bk', now='2026-11-17 22:35+08:00')
    assert len(pd.read_parquet(t)) == len(prod()) and not (tmp_path/'bk').exists()


def test_apply_aborts_and_leaves_target_untouched_when_verification_fails(tmp_path):
    t = tmp_path/'prices_raw.parquet'; p = prod(); p.to_parquet(t, index=False)
    bad = rows(DAYS[3], IDS[:2]); bad = pd.concat([bad, bad])                         # duplicate keys in the additions
    with pytest.raises(AssertionError):
        rp.apply(t, bad, tmp_path/'bk', now='2026-11-17 10:00+08:00')
    pd.testing.assert_frame_equal(pd.read_parquet(t), p); assert list(tmp_path.glob('*.tmp')) == []


def test_apply_rejects_addition_colliding_with_existing_row(tmp_path):
    t = tmp_path/'prices_raw.parquet'; prod().to_parquet(t, index=False)
    with pytest.raises(AssertionError):
        rp.apply(t, rows(DAYS[0], IDS[:2]), tmp_path/'bk', now='2026-11-17 10:00+08:00')
