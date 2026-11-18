import numpy as np
import pandas as pd
import pytest
from research_lab.significance import holm_adjust, paired_ic_test, compare_models


def series(vals, start='2024-01-01'):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_holm_hand_arithmetic_monotone_and_capped():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([1.2])


def test_identical_models_have_zero_difference_and_p_one():
    x = series(np.random.default_rng(0).normal(.05, .1, 200))
    r = paired_ic_test(x, x.copy(), horizon=5, n_boot=200, seed=1)
    assert r['mean_diff'] == 0 and r['p_hac'] == 1.0 and r['p_boot'] == 1.0
    assert r['n'] == 200


def test_clear_winner_detected_and_sign():
    rng = np.random.default_rng(1)
    b = series(rng.normal(0, .1, 400))
    a = b + 0.08 + series(rng.normal(0, .02, 400))
    r = paired_ic_test(a, b, horizon=5, n_boot=500, seed=1)
    assert r['mean_diff'] > 0.07 and r['p_hac'] < 1e-6 and r['p_boot'] < 0.01
    assert r['ci95_boot'][0] > 0
    swapped = paired_ic_test(b, a, horizon=5, n_boot=500, seed=1)
    assert swapped['mean_diff'] == pytest.approx(-r['mean_diff'])


def test_aligns_on_common_finite_dates_only():
    a = series([.1, np.nan, .2, .3, .1, .2])
    b = series([.0, .1, np.nan, .1, .0, .1])
    r = paired_ic_test(a, b, horizon=1, n_boot=50, seed=0, min_obs=2)
    assert r['n'] == 4
    assert r['mean_diff'] == pytest.approx(np.mean([.1, .2, .1, .1]))


def test_noise_models_not_falsely_significant_and_deterministic():
    rng = np.random.default_rng(2)
    ps = []
    for i in range(30):
        a, b = series(rng.normal(0, .1, 250)), series(rng.normal(0, .1, 250))
        ps.append(paired_ic_test(a, b, horizon=5, n_boot=100, seed=i)['p_hac'])
    assert np.mean(np.array(ps) < .05) <= .2
    a, b = series(rng.normal(0, .1, 100)), series(rng.normal(0, .1, 100))
    assert paired_ic_test(a, b, 5, n_boot=100, seed=3) == paired_ic_test(a, b, 5, n_boot=100, seed=3)


def test_too_few_observations_fail_closed():
    with pytest.raises(ValueError):
        paired_ic_test(series([.1, .2]), series([.1, .3]), horizon=5)


def test_autocorrelation_widens_hac_vs_naive():
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 600)
    d = np.zeros(600)
    for t in range(1, 600):
        d[t] = .8*d[t-1] + e[t]
    r = paired_ic_test(series(d*.01+.003), series(np.zeros(600)), horizon=10, n_boot=50, seed=0)
    naive_t = r['mean_diff']/(np.std(d*.01, ddof=1)/np.sqrt(600))
    assert abs(r['t_hac']) < abs(naive_t)


def test_compare_models_holm_over_family():
    rng = np.random.default_rng(5)
    base = series(rng.normal(0, .1, 300))
    daily = {'base': base, 'good': base + .1 + series(rng.normal(0, .02, 300)),
             'noise': series(rng.normal(0, .1, 300))}
    out = compare_models(daily, baseline='base', horizon=5, n_boot=200, seed=1)
    assert [r['model'] for r in out] == ['good', 'noise']
    assert all(r['p_hac_holm'] >= r['p_hac'] for r in out)
    assert out[0]['p_hac_holm'] < .01 and out[1]['p_hac_holm'] > .05
    with pytest.raises(KeyError):
        compare_models(daily, baseline='zzz', horizon=5)
