"""Small synthetic checks; no historical inputs or model fits."""
import numpy as np
import pytest

from tools.tradability_diagnostic import (
    composite, daily_ic, limit_locked, turnover, trading_cost, quintiles, top_decile,
)
from tools.step3_evaluation import evaluate


def test_cost_and_turnover():
    buy, sell = turnover({"a": .5, "b": .5}, {"b": .5, "c": .5})
    assert (buy, sell) == (.5, .5)
    assert trading_cost(buy, sell, .001) == pytest.approx(.003925)
    assert turnover({}, {"a": 1.}) == (1., 0.)
    assert turnover({"a": 1.}, {}) == (0., 1.)
    assert turnover({"a": -.5}, {"a": .5}) == (1., 0.)
    buy, sell = turnover({"a": .6, "b": .4}, {"a": .5, "b": .5})
    assert buy == pytest.approx(.1)
    assert sell == pytest.approx(.1)
    with pytest.raises(ValueError):
        trading_cost(1, 1, -.001)


def test_limit_change_tick_and_unknown_fail_closed():
    assert limit_locked(107, 100, "2015-05-29")
    assert not limit_locked(107, 100, "2015-06-01")
    assert limit_locked(109.5, 100, "2015-06-01")
    assert not limit_locked(109.49, 100, "2015-06-01")
    assert limit_locked(np.nan, 100, "2020-01-02")
    assert limit_locked(11, np.nan, "2020-01-02")


def test_ic_exactly_matches_frozen_evaluator_with_ties_and_immaturity():
    rng = np.random.default_rng(17)
    keys = np.array([(d, s) for d in range(4) for s in range(25)])
    labels = rng.normal(size=100); labels[-9:] = np.nan
    predictions = {(a, seed): rng.integers(0, 8, 100).astype(float)
                   for a in 'AB' for seed in (17, 29, 43)}
    dates = np.array(['2015-01-02', '2015-01-05', '2021-01-04', '2021-01-05'], dtype='datetime64[D]')
    ref = evaluate({k: keys for k in predictions}, predictions, labels,
                   keys[:, 0], dates, label_keys=keys, frozen_labels=labels,
                   independent_execution_authorized=True)
    for arm in 'AB':
        score = composite(keys, labels, [predictions[arm, s] for s in (17, 29, 43)])
        got = daily_ic(keys, score, labels, len(dates))
        np.testing.assert_allclose(got, ref['daily'][arm], atol=1e-15, rtol=0, equal_nan=True)


def test_quintiles_have_stable_stock_tiebreak_and_missing_excluded():
    q = quintiles(np.array([2., 2., np.nan, 2., 2., 2.]), np.array([5, 4, 3, 2, 1, 0]))
    assert q.tolist() == [5, 4, 0, 3, 2, 1]


def test_top_decile_rounds_up_and_breaks_ties_by_stock():
    score = np.ones(21)
    stock = np.arange(21)[::-1]
    assert top_decile(score, stock).tolist() == [20, 19, 18]
