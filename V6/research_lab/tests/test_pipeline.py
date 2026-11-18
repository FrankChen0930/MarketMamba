import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research_lab.data import load_prices, validate_prices, dataset_manifest, coverage_report
from research_lab.features import compute_features, FEATURE_COLUMNS
from research_lab.labels import compute_labels, purged_training_mask
from research_lab.splits import walk_forward_splits

REAL = Path('/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/public_tw/prices_daily.parquet')

@pytest.fixture
def prices():
    days = pd.bdate_range('2020-01-01', periods=120)
    rows = []
    for j, code in enumerate(['0050', '1101', '2330']):
        for i, day in enumerate(days):
            close = 100 + i * (j + 1) + np.sin(i)
            rows.append(dict(date=day, code=code, open=close-.5, high=close+1,
                             low=close-1, close=close, adj_close=close*.8, volume=1000+i*(j+1)))
    return pd.DataFrame(rows)


def test_future_changes_cannot_change_any_past_feature(prices):
    cutoff = prices.date.unique()[85]
    original = compute_features(validate_prices(prices))
    changed = prices.copy()
    changed.loc[changed.date > cutoff, ['open','high','low','close','adj_close','volume']] *= 2
    changed = compute_features(validate_prices(changed))
    pd.testing.assert_frame_equal(original.loc[original.date <= cutoff], changed.loc[changed.date <= cutoff])
    truncated = compute_features(validate_prices(prices[prices.date <= cutoff]))
    pd.testing.assert_frame_equal(original.loc[original.date <= cutoff].reset_index(drop=True), truncated)


def test_label_alignment_and_tail(prices):
    p = validate_prices(prices)
    labels = compute_labels(p)
    stock = p[p.code == '0050'].reset_index(drop=True)
    y = labels[labels.code == '0050'].reset_index(drop=True)
    for h in (5,10,20):
        assert y.loc[0, f'fwd_return_{h}d'] == pytest.approx(stock.close.iloc[h]/stock.open.iloc[1]-1)
        assert y.loc[0, f'label_end_{h}d'] == stock.date.iloc[h]
        assert y[f'fwd_return_{h}d'].tail(h).isna().all()
    assert y.loc[0, 'label_entry'] == stock.date.iloc[1]
    assert labels['rank_label_5d'].dropna().between(0,1).all()


def test_hash_and_features_are_order_independent(prices):
    a, b = validate_prices(prices), validate_prices(prices.sample(frac=1, random_state=7))
    assert dataset_manifest(a)['sha256'] == dataset_manifest(b)['sha256']
    pd.testing.assert_frame_equal(compute_features(a), compute_features(b))
    altered = a.copy(); altered.loc[0, 'close'] += .01
    assert dataset_manifest(a)['sha256'] != dataset_manifest(altered)['sha256']


def test_bad_data_deduplication_and_conflicts(prices):
    duplicated = pd.concat([prices, prices.iloc[:1]], ignore_index=True)
    p = validate_prices(duplicated)
    assert len(p) == len(prices)
    assert p.attrs['quality']['duplicates_removed'] == 1
    conflict = prices.iloc[:1].copy(); conflict['close'] += .1
    with pytest.raises(ValueError, match='conflicting'):
        validate_prices(pd.concat([prices, conflict]))
    for value in [0, -1, np.nan, np.inf]:
        bad = prices.copy(); bad.loc[10, 'close'] = value
        q = validate_prices(bad)
        assert not q.loc[(q.code=='0050') & (q.date==prices.date.iloc[10]), 'price_valid'].item()
        f = compute_features(q)
        assert f.loc[(f.code=='0050') & (f.date==prices.date.iloc[10]), FEATURE_COLUMNS].isna().all(axis=None)
        assert compute_labels(q).query("code == '0050'").iloc[5].fwd_return_10d != compute_labels(q).query("code == '0050'").iloc[5].fwd_return_10d
    bad = prices.copy(); bad.loc[0,'date'] = pd.NaT
    with pytest.raises(ValueError): validate_prices(bad)


def test_gaps_outliers_zero_volume_are_not_filled(prices):
    days = sorted(prices.date.unique())
    gap = validate_prices(prices[~((prices.code=='0050') & (prices.date==days[10]))])
    assert coverage_report(gap).set_index('code').loc['0050','missing_union_sessions'] == 1
    y = compute_labels(gap).query("code == '0050'").set_index('date')
    assert pd.isna(y.loc[days[5], 'fwd_return_10d'])
    f = compute_features(gap).query("code == '0050'").set_index('date')
    assert pd.isna(f.loc[days[11], 'return_1d'])
    bad = prices.copy(); bad.loc[10, ['open','high','low','close']] *= 2
    q = validate_prices(bad)
    assert q.outlier_flag.sum() >= 1
    assert compute_labels(q).query("code == '0050'").iloc[5].fwd_return_10d != compute_labels(q).query("code == '0050'").iloc[5].fwd_return_10d
    zero = prices.copy(); zero.loc[10, 'volume'] = 0
    q = validate_prices(zero)
    assert not q.loc[(q.code=='0050') & (q.date==days[10]), 'observation_valid'].item()


def test_splits_purge_embargo_and_maturity(prices):
    d = compute_labels(validate_prices(prices))
    folds = walk_forward_splits(d, min_train_sessions=30, test_sessions=10, horizon=20, purge=20, embargo=3)
    assert folds
    dates = sorted(d.date.unique())
    for fold in folds:
        tr, te = fold['train'], fold['test']
        assert not set(tr.row_index) & set(te.row_index)
        assert tr.label_end_20d.max() < te.date.min()
        assert dates.index(te.date.min()) - dates.index(tr.date.max()) - 1 >= 23
        assert d.loc[tr.row_index, 'fwd_return_20d'].notna().all()
    pd.testing.assert_frame_equal(folds[0]['train'], walk_forward_splits(d, min_train_sessions=30,test_sessions=10,horizon=20,purge=20,embargo=3)[0]['train'])
    with pytest.raises(ValueError): walk_forward_splits(d, horizon=20, purge=19)
    with pytest.raises(ValueError): walk_forward_splits(d, embargo=-1)
    with pytest.raises(ValueError): walk_forward_splits(d.drop(columns='label_end_20d'))
    forged = d.copy(); forged['label_end_20d'] = forged.date
    with pytest.raises(ValueError): walk_forward_splits(forged)
    assert not purged_training_mask(d, d.date.min(), horizon=20, purge=20).any()


def test_cross_section_scaling_and_no_adjusted_price_dependency(prices):
    p = validate_prices(prices)
    f = compute_features(p)
    day = f[f.date==f.date.max()]
    assert day.return_20d_cs_z.mean() == pytest.approx(0,abs=1e-12)
    assert day.return_20d_cs_rank.between(0,1).all()
    changed = prices.copy(); changed['adj_close'] *= 2
    pd.testing.assert_frame_equal(f[FEATURE_COLUMNS], compute_features(validate_prices(changed))[FEATURE_COLUMNS])


@pytest.mark.skipif(not REAL.exists(), reason='external read-only parquet absent')
def test_real_data():
    p = load_prices(REAL)
    assert len(p) == 378571
    assert p.code.nunique() == 150
    assert p.date.min() == pd.Timestamp('2016-01-04')
    assert p.date.max() == pd.Timestamp('2026-11-13')
    assert '0050' in set(p.code)
    assert not p.duplicated(['code','date']).any()
    assert p.groupby('code').date.apply(lambda s:s.is_monotonic_increasing).all()
    assert dataset_manifest(p)['sha256'] == dataset_manifest(load_prices(REAL))['sha256']


def test_cli_round_trip_and_no_clobber(prices, tmp_path):
    from research_lab.build_dataset import build_dataset
    src, out = tmp_path/'prices.parquet', tmp_path/'dataset.parquet'
    prices.to_parquet(src,index=False)
    manifest = build_dataset(src,out)
    result = pd.read_parquet(out)
    assert len(result) == len(prices)
    assert dataset_manifest(result)['sha256'] == manifest['dataset']['sha256']
    assert out.with_suffix('.manifest.json').exists()
    with pytest.raises(FileExistsError): build_dataset(src,out)


def test_matches_existing_holding_window_contract(prices):
    from V6.experimental.v7_executable_labels import build_executable_labels
    p = validate_prices(prices)
    raw = [dict(stock_id=r.code, Date=r.date.strftime('%Y-%m-%d'),
                Open=r.open, Close=r.close, observation_valid=True,
                open_executable=True) for r in p.itertuples()]
    days = sorted(p.date.dt.strftime('%Y-%m-%d').unique())
    reference = build_executable_labels(raw,days,horizons=(5,10,20))
    labels = compute_labels(p).set_index(['code','date'])
    for r in reference:
        row = labels.loc[(r['stock_id'],pd.Timestamp(r['signal_date']))]
        for h in (5,10,20):
            assert row[f'fwd_return_{h}d'] == pytest.approx(r[f'Alpha_{h}d'],nan_ok=True)


@pytest.mark.parametrize('horizons', [(), (0,), (True,), (5,5), (5.5,)])
def test_reject_invalid_horizons(prices, horizons):
    with pytest.raises(ValueError): compute_labels(validate_prices(prices), horizons)


def test_conservative_constants_and_invalid_warm_rows(prices):
    flat = prices.copy()
    flat[['open','close','adj_close']] = 100.
    flat['low'],flat['high'],flat['volume'] = 99.,101.,1000.
    f = compute_features(validate_prices(flat))
    last = f[f.date==f.date.max()]
    assert last.rsi_14d.eq(50).all()
    assert last.return_20d_cs_rank.eq(2/3).all()
    assert last.return_20d_cs_z.isna().all()
    assert last.volume_z_20d.isna().all()
    flat.loc[90,'close'] = np.nan
    f = compute_features(validate_prices(flat))
    assert f.query("code == '0050'").iloc[90][FEATURE_COLUMNS].isna().all()
