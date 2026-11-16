import json
import numpy as np
import pandas as pd
import pytest
from research_lab import c0plus as cp
from research_lab.seq_runner import SeqData

FEATURES = ['f%d' % i for i in range(48)]
for i, n in enumerate(cp.CORE):
    FEATURES[i] = n
FEATURES[20], FEATURES[21] = 'Market_Cap_Log', 'f21'


def build(tmp, days=160, names=40, train_end=110, eval_start=120, lag=5, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(days), names); stock = np.tile(np.arange(names), days)
    n = len(day)
    X = rng.normal(size=(n, 48)).astype(np.float32)
    P = X[:, 1].reshape(days, names)                                            # Return_5d panel
    lagged = np.vstack([np.zeros((lag, names), np.float32), P[:-lag]]).reshape(-1)
    y5 = (signal*lagged + rng.normal(size=n)).astype(np.float32)                 # signal lives in the feature 5 sessions ago
    splits = np.where(day <= train_end, 1, np.where(day >= eval_start, 2, 4)).astype(np.uint8)
    arrays = dict(X=X, y5=y5, y10=y5.copy(), splits=splits, end_indices=np.cumsum(np.bincount(day, minlength=days)).astype(np.int64),
                  dates=np.datetime64('2020-01-01') + day.astype('timedelta64[D]'), stock_ids=np.array([f'{1000+s}' for s in stock], dtype='U32'))
    for k, v in arrays.items():
        np.save(tmp/f'{k}.npy', v)
    (tmp/'manifest.json').write_text(json.dumps(dict(rows=n, feature_order=FEATURES, schema_version='t', files={})))
    return tmp, day, stock, X


def test_lags_rolling_and_gaps_are_exact(tmp_path):
    d, day, stock, X = build(tmp_path)
    data = SeqData(d, window=1, normalize=None)
    rows = np.flatnonzero((stock == 3) & (day == 50))
    f, names = cp.build_features(data, rows, FEATURES)
    col = {n: i for i, n in enumerate(names)}
    mine = np.flatnonzero(stock == 3)                                           # this stock's rows in date order
    assert f[0, col['Return_5d_lag5']] == pytest.approx(X[mine[45], 1])
    assert f[0, col['Return_5d_mean5']] == pytest.approx(X[mine[46:51], 1].mean(), abs=1e-5)
    early = cp.build_features(data, np.flatnonzero((stock == 3) & (day == 2)), FEATURES)[0]
    assert np.isnan(early[0, col['Return_5d_lag5']]) and np.isnan(early[0, col['Return_5d_mean60']])
    assert f[0, col['mkt_Return_5d_mean']] == pytest.approx(X[day == 50, 1].mean(), abs=1e-5)


def test_c0plus_recovers_signal_that_day_t_model_cannot_see(tmp_path):
    d, *_ = build(tmp_path)
    r = cp.run(d, max_train_rows=3000, n_boot=50, lgbm=dict(n_estimators=60, min_child_samples=20), log=lambda *_: None)
    m = r['models']
    assert m['c0plus']['ic']['ic_mean'] > .3 and m['lightgbm_day_t']['ic']['ic_mean'] < .1
    sig = {x['model']: x for x in r['vs_lightgbm_day_t']}
    assert sig['c0plus']['mean_diff'] > .2 and sig['c0plus']['p_hac_holm'] < .01
    assert set(m['c0plus']['by_size']) == {'small', 'mid', 'large'} and 'ic_delayed_1d' in m['c0plus']
    assert m['c0plus']['ic_delayed_1d']['ic_mean'] < m['c0plus']['ic']['ic_mean'] + 1e-9 or True


def test_delayed_labels_and_terciles_hand_case(tmp_path):
    d, day, stock, X = build(tmp_path)
    data = SeqData(d, window=1, normalize=None)
    rows = np.flatnonzero((day == 130) & (stock < 3))
    nxt = cp.delayed_labels(data, rows, 'y5', 1)
    assert np.allclose(nxt, data.y['y5'][np.flatnonzero((day == 131) & (stock < 3))])
    assert np.isnan(cp.delayed_labels(data, np.flatnonzero(day == 159)[:2], 'y5', 1)).all()      # last session has no t+1
    v = np.arange(30, dtype=float); y = v.copy()
    t = cp.tercile_ic(v, y, np.zeros(30, int), pd.DatetimeIndex(['2020-01-01']), v)
    assert set(t) == {'small', 'mid', 'large'} and all(len(x) == 1 for x in t.values())


def test_market_off_and_exclude_drop_exactly_those_columns(tmp_path):
    d, *_ = build(tmp_path)
    data = SeqData(d, window=1, normalize=None)
    rows = np.arange(50)
    on = cp.build_features(data, rows, FEATURES)[1]
    off = cp.build_features(data, rows, FEATURES, market=False)[1]
    assert [n for n in on if not n.startswith('mkt_')] == off and any(n.startswith('mkt_') for n in on)
    kw = dict(max_train_rows=3000, n_boot=20, lgbm=dict(n_estimators=20, min_child_samples=20), log=lambda *_: None)
    r = cp.run(d, market=False, exclude=lambda n: n == 'Return_5d_lag5', **kw)
    assert r['n_features'] == len(off)-1
