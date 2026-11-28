"""Small synthetic checks; no historical inputs or model fits."""
import numpy as np
import pytest

from tools.tradability_diagnostic import (
    composite, daily_ic, limit_locked, turnover, trading_cost, quintiles, top_decile,
)
from tools.step3_evaluation import evaluate


def test_amount_uses_prior_twenty_sessions_and_missing_breaks_window():
    from tools.tradability_diagnostic import prior_amount
    x = np.arange(1., 24.)[:, None]
    got = prior_amount(x)
    assert np.isnan(got[:20]).all()
    assert got[20, 0] == 10.5
    x[19] = np.nan
    assert np.isnan(prior_amount(x)[20:]).all()


def test_admission_requires_actual_price_volume_and_known_next_row():
    from tools.tradability_diagnostic import admission
    # adjusted close is irrelevant to the actual money/volume price gate
    amount = np.array([1e7] * 5)
    volume = np.array([1e6, 2e6, 1e6, 1e6, 1e6])
    ok, known = admission(amount, amount, volume,
                          np.array([100., 100., 110., np.nan, 100.]),
                          np.array([100.] * 5), np.array([1., 1., 1., 1., 0.]), '2024-01-02', 1e7)
    assert ok.tolist() == [True, False, False, False, False]
    assert known.tolist() == [True, True, True, False, True]


@pytest.mark.parametrize('date', ['2015-05-29', '2015-06-01', '2024-07-04'])
def test_vectorized_limit_filter_matches_scalar_at_every_tick_tier(date):
    from tools.tradability_diagnostic import admission, tick_size
    previous = np.array([1., 9., 10., 49., 50., 99., 100., 499., 500., 999., 1000.])
    limit = .07 if date < '2015-06-01' else .10
    boundary = previous * (1 + limit)
    boundary -= np.array([tick_size(v) for v in boundary])
    for epsilon in [-1e-7, 0., 1e-7]:
        opens = boundary + epsilon
        ok, known = admission(np.full(len(opens), 1e8), np.full(len(opens), 1e8),
                              np.full(len(opens), 1e6), opens, previous, np.ones(len(opens)), date, 1e7)
        expected = [not limit_locked(o, p, date) for o, p in zip(opens, previous)]
        assert known.all()
        assert ok.tolist() == expected


def test_cycle_equal_weight_paths_and_cost_and_unknown_exit():
    from tools.tradability_diagnostic import cycle_path
    closes = np.array([[11., 18.], [12., 22.]])
    path = cycle_path(np.array([10., 20.]), closes, np.array([.5, .5]), .01)
    np.testing.assert_allclose(np.cumprod(1 + path), [.99, 1.14])
    closes[-1, 0] = np.nan
    assert cycle_path(np.array([10., 20.]), closes, np.array([.5, .5]), .01) is None


def test_equal_weight_list_turnover_not_drifted():
    # Explicit second-round Human choice: turnover from equal-weight lists.
    assert turnover({1: .5, 2: .5}, {1: .5, 2: .5}) == (0., 0.)
    assert turnover({1: .5, 2: .5}, {2: .5, 3: .5}) == (.5, .5)


def test_twenty_offsets_cash_boundaries_and_common_unknown_support():
    from tools.tradability_diagnostic import simulate_portfolios
    days = np.arange('2020-01-01', '2020-03-06', dtype='datetime64[D]')
    keys = np.array([(d, s) for d in range(45) for s in range(21)])
    scores = {a: -keys[:, 1].astype(float) for a in 'AB'}
    price = {k: np.full((len(days), 21), 10.) for k in ['Open', 'Close']}
    universe = np.ones(len(keys), dtype=bool)
    paths, buys, sells, failed = simulate_portfolios(keys, scores, universe, price, days, days[-1])
    assert failed == dict(insufficient_universe=0, source_horizon=0, missing_selected_price=0)
    for off in range(20):
        assert np.nansum(buys['B', 'top10'][off]) == pytest.approx(1.)
        assert np.nansum(sells['B', 'top10'][off]) == pytest.approx(1.)
        assert np.prod(1 + paths['B', 'top10', 0.][off][np.isfinite(paths['B', 'top10', 0.][off])]) == pytest.approx(
            (1-.001425)*(1-.004425))
        assert np.nanmax(np.abs(paths['B', 'longshort', -1.][off])) == 0.
    # Leak-audit T6 policy: a holding-period gap no longer drops the cycle; the
    # stock is liquidated at its last observed close (benchmark stock20 here).
    price['Close'][10, 20] = np.nan
    paths, _, _, failed = simulate_portfolios(keys, scores, universe, price, days, days[-1])
    assert failed['missing_selected_price'] == 0
    assert np.isfinite(paths['B', 'top10', -1.][0, 1:21]).all()
    assert np.isfinite(paths['A', 'equal', -1.][0, 1:21]).all()


def test_performance_drawdown_includes_initial_nav():
    from tools.tradability_diagnostic import performance
    r = np.array([-.1, .1])
    result = performance(r, np.zeros(2), np.ones(2), np.ones(2))
    assert result[4] == pytest.approx(-.1)
    assert result[5] == 252.
    assert result[6] == 2


def test_finmind_loader_preserves_missing_calendar_and_rejects_duplicates(tmp_path):
    import pandas as pd
    from tools.tradability_diagnostic import load_finmind
    days = np.array(['2014-01-02', '2014-01-03', '2014-01-06'], dtype='datetime64[D]')
    frame = pd.DataFrame(dict(date=['2014-01-02'], stock_id=['2412'],
                              Trading_money=[100.], Trading_Volume=[2.], open=[47.], close=[48.], spread=[1.]))
    frame.to_parquet(tmp_path / '2014-01-02_adj_price.parquet')
    mats, identities = load_finmind(days, np.array(['2412', '3045']), tmp_path)
    assert len(identities) == 1
    assert mats['Trading_money'][0, 0] == 100.
    assert np.isnan(mats['Trading_money'][1:]).all()
    assert np.isnan(mats['Trading_money'][:, 1]).all()
    pd.concat([frame, frame]).to_parquet(tmp_path / '2014-01-02_adj_price.parquet')
    with pytest.raises(ValueError, match='duplicate'):
        load_finmind(days, np.array(['2412']), tmp_path)


def test_insolvent_short_cycle_stops_instead_of_excluding_loss():
    from tools.tradability_diagnostic import simulate_portfolios
    days = np.arange('2020-01-01', '2020-02-01', dtype='datetime64[D]')
    keys = np.array([(0, s) for s in range(20)])
    scores = {a: -keys[:, 1].astype(float) for a in 'AB'}
    price = {k: np.full((len(days), 20), 10.) for k in ['Open', 'Close']}
    price['Close'][1:21, 18:] = 40.
    with pytest.raises(ValueError, match='insolvent'):
        simulate_portfolios(keys, scores, np.ones(20,dtype=bool), price, days, days[-1])


def test_label_reconstruction_uses_full_frozen_calendar_beyond_inference_tail():
    from tools.tradability_diagnostic import validate_label_marks
    full = np.arange('2020-01-01', '2020-02-01', dtype='datetime64[D]')
    prices = {k: np.full((len(full),1),10.) for k in ['Open','Close']}
    prices['Close'][24,0] = 12.
    result = validate_label_marks(np.array([[4,0]]),np.array([.2]),prices,full[:5],full)
    assert result['rows'] == 1
    assert result['max_absolute_error'] < 1e-15
    with pytest.raises(ValueError, match='calendar prefix'):
        validate_label_marks(np.array([[4,0]]),np.array([.2]),prices,full[1:6],full)
    with pytest.raises(ValueError, match='horizon'):
        validate_label_marks(np.array([[15,0]]),np.array([.2]),prices,full[:20],full)


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
