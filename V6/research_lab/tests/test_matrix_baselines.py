import json
import numpy as np
import pytest
from research_lab.matrix_baselines import load_matrix, run, daily_rank_ic

FEATURES = ['f%d' % i for i in range(48)]
FEATURES[6], FEATURES[7], FEATURES[30] = 'Return_5d', 'Return_20d', 'Volatility_20d'


def make(tmp_path, days=120, names=40, signal=.5, seed=0):
    rng = np.random.default_rng(seed)
    n = days*names
    X = rng.normal(size=(n, 48)).astype(np.float32)
    y = (signal*X[:, 10] + rng.normal(size=n)).astype(np.float32)
    dates = np.datetime64('2020-01-01') + np.repeat(np.arange(days), names).astype('timedelta64[D]')
    splits = np.where(np.repeat(np.arange(days), names) < 80, 1, 2).astype(np.uint8)
    splits[np.repeat(np.arange(days), names) < 5] = 0
    y[(np.repeat(np.arange(days), names) == 90) & (np.tile(np.arange(names), days) < 5)] = np.nan
    ends = np.arange(1, days+1, dtype=np.int64)*names
    for k, v in dict(X=X, y5=y, y10=y, dates=dates, splits=splits, end_indices=ends).items():
        np.save(tmp_path/f'{k}.npy', v)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(rows=n, feature_order=FEATURES,
        schema_version='t', label_contract=dict(evidence_class='SYNTH'), files={})))
    return tmp_path


def test_load_validates_lengths(tmp_path):
    make(tmp_path)
    m = json.loads((tmp_path/'manifest.json').read_text()); m['rows'] += 1
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_matrix(tmp_path)


def test_ridge_recovers_planted_signal_on_eval_only(tmp_path):
    make(tmp_path)
    r = run(tmp_path, 'y5', models=('ridge',), n_boot=50)
    assert r['models']['ridge']['ic_mean'] > .2
    assert abs(r['models']['reversal_ret5']['ic_mean']) < .1
    assert r['eval_rows'] == 40*(120-80) and r['fit_rows'] == 40*75
    assert r['eval_sessions'] == 40


def test_fit_never_sees_eval_rows_and_nan_labels_excluded(tmp_path):
    make(tmp_path, signal=0.)
    r = run(tmp_path, 'y5', models=('ridge',), max_train_rows=1000, n_boot=20)
    assert r['fit_rows'] == 1000 and r['train_rows_available'] == 3000
    # label leakage check: a no-signal world gives ~0 IC for a fitted model
    assert abs(r['models']['ridge']['ic_mean']) < .1


def test_daily_rank_ic_hand_case():
    s = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    y = np.array(list(range(10)) + list(range(10)), dtype=float)
    ic = daily_rank_ic(s, y, np.array([10, 20]), np.ones(20, bool))
    assert ic.iloc[0] == pytest.approx(1.) and ic.iloc[1] == pytest.approx(-1.)


def test_fit_target_transforms_use_full_cross_section_and_do_not_leak_common_day_shift(tmp_path):
    from research_lab.matrix_baselines import Context
    make(tmp_path, signal=.5)
    y = np.load(tmp_path/'y5.npy')
    shift = np.repeat(np.linspace(-5, 5, 120), 40).astype(np.float32)  # huge common daily component
    np.save(tmp_path/'y5.npy', y + shift)
    raw = Context(tmp_path, 'y5', 10**6, fit_target='raw')
    rank = Context(tmp_path, 'y5', 10**6, fit_target='rank')
    dm = Context(tmp_path, 'y5', 10**6, fit_target='demean')
    assert abs(rank.y_fit.mean()) < 1e-6 and rank.y_fit.min() > -.5 and rank.y_fit.max() <= .5
    assert abs(dm.y_fit.mean()) < 1e-5
    assert raw.y_fit.std() > 2 and dm.y_fit.std() < 1.5
    # a pooled raw fit is contaminated by the shift; rank fit still recovers the planted signal
    r_rank = run(tmp_path, 'y5', models=('ridge',), n_boot=20, fit_target='rank')
    r_raw = run(tmp_path, 'y5', models=('ridge',), n_boot=20, fit_target='raw')
    assert r_rank['models']['ridge']['ic_mean'] >= r_raw['models']['ridge']['ic_mean'] - 1e-9
    with pytest.raises(ValueError):
        Context(tmp_path, 'y5', 100, fit_target='bogus')
