import numpy as np
import pandas as pd
import pytest
from research_lab.portfolio import Config, run_backtest, transaction_cost


def fixture(n=9):
    days = pd.bdate_range('2024-01-01', periods=n)
    prices = pd.DataFrame([dict(code=c, date=d, open=10., close=10., volume=1000.)
                           for d in days for c in ['A', 'B']])
    pred = prices[['code', 'date']].assign(fold=0, model='fixture', horizon=2)
    pred['score'] = np.where(pred.code.eq('A'), 1., 0.)
    return prices, pred


def cfg(**kw):
    return Config(top_k=1, rebalance_sessions=2, commission=0., sell_tax=0., **kw)


def test_cost_hand_arithmetic():
    c = Config(commission=.001425, commission_discount=.6, sell_tax=.003, slippage_bps=5)
    assert transaction_cost(1000, 2000, c) == pytest.approx(1000*.001355 + 2000*.004355)


def test_schedule_turnover_and_benchmark():
    p, s = fixture()
    out = run_backtest(p, s, cfg())
    assert [x['date'] for x in out['trades']] == ['2024-01-02', '2024-01-10']
    assert out['metrics']['total_return'] == '0'
    assert out['metrics']['benchmark']['excess_return'] == '0'
    assert sum(float(x['turnover']) for x in out['metrics']['sessions']) == pytest.approx(1.)
    assert out['schedule'][1]['entry'] == '2024-01-04'


def test_drawdown_and_no_future_leakage():
    p, s = fixture(12)
    s = s[s.date <= pd.Timestamp('2024-01-11')]
    p.loc[(p.code == 'A') & (p.date == '2024-01-03'), 'close'] = 8.
    a = run_backtest(p, s, cfg())
    assert float(a['metrics']['max_drawdown']['fraction']) == pytest.approx(.2)
    p.loc[p.date > pd.Timestamp(a['end']), ['open', 'close']] = 1000.
    assert run_backtest(p, s, cfg()) == a


def test_determinism_hash_and_ties():
    p, s = fixture()
    a = run_backtest(p, s, cfg())
    assert run_backtest(p.sample(frac=1, random_state=1), s.sample(frac=1, random_state=2), cfg()) == a
    assert run_backtest(p, s, cfg(seed=99))['config_sha256'] != a['config_sha256']


def test_cost_cash_conservation():
    p, s = fixture()
    c = Config(top_k=1, rebalance_sessions=2, commission=.01, sell_tax=.02)
    out = run_backtest(p, s, c)
    assert float(out['metrics']['final_net_value']) == pytest.approx((1-.03)/1.01)
    assert float(out['metrics']['cost_drag']) == pytest.approx(1-(1-.03)/1.01)
    assert min(x['cash'] for x in out['daily']) >= -1e-10


def test_missing_open_keeps_position():
    p, s = fixture()
    s.loc[s.date >= '2024-01-03', 'score'] *= -1
    p.loc[(p.code == 'A') & (p.date == '2024-01-04'), 'open'] = np.nan
    out = run_backtest(p, s, cfg())
    assert out['blocked_orders'] >= 1
    assert not any(t['date'] == '2024-01-04' and t['code'] == 'A' for t in out['trades'])


def test_liquidity_and_capped_scores():
    p, s = fixture()
    p.loc[p.code == 'A', 'volume'] = 0.
    out = run_backtest(p, s, cfg(min_dollar_volume=1))
    assert out['trades'][0]['code'] == 'B'
    out = run_backtest(*fixture(), cfg(weighting='score', weight_cap=.4))
    assert out['daily'][1]['cash'] == pytest.approx(.6)


def test_contract_rejects_ambiguity():
    p, s = fixture()
    with pytest.raises(ValueError):
        run_backtest(p, pd.concat([s, s]), cfg())
    s['fold'] = .5
    with pytest.raises(ValueError):
        run_backtest(p, s, cfg())


def test_oracle_random_and_momentum_causality():
    from research_lab.backtest import baseline_predictions
    days = pd.bdate_range('2020-01-01', periods=81)
    codes = [f'{i:04}' for i in range(40)]
    rng = np.random.default_rng(31)
    op = 100*np.cumprod(1+rng.normal(0, .015, (81, 40)), axis=0)
    p = pd.DataFrame([dict(date=day, code=code, open=op[i,j], close=op[i,j], volume=1e6)
                      for i,day in enumerate(days) for j,code in enumerate(codes)])
    config = Config(top_k=5, rebalance_sessions=2, bootstrap_samples=30)
    oracle = baseline_predictions(p, 'oracle', config)
    random = baseline_predictions(p, 'random', config)
    random = random[random.date <= oracle.date.max()]
    a = run_backtest(p, oracle, config)
    b = run_backtest(p, random, config)
    assert float(a['metrics']['benchmark']['excess_return']) > .5
    assert abs(float(b['metrics']['benchmark']['excess_return']) + b['gross_minus_net_return']) < .10
    assert b['gross_minus_net_return'] > 0
    momentum = baseline_predictions(p, 'momentum20', config)
    p.loc[p.date > days[40], ['open', 'close']] *= 3
    changed = baseline_predictions(p, 'momentum20', config)
    pd.testing.assert_frame_equal(momentum[momentum.date <= days[40]].reset_index(drop=True),
                                  changed[changed.date <= days[40]].reset_index(drop=True))


def test_reused_metrics_hash_and_single_name_benchmark():
    from V6.experimental.v7_portfolio_metrics import canonical_sha256
    p, s = fixture()
    out = run_backtest(p, s, cfg(benchmark_code='A'))
    assert out['metrics']['canonical_sha256'] == canonical_sha256(out['metrics'])
    assert out['canonical_sha256'] == canonical_sha256(out)
    assert out['statistics']['total_return'] == 0


def test_missing_terminal_does_not_invent_liquidation():
    p, s = fixture()
    p.loc[(p.code == 'A') & (p.date == '2024-01-10'), 'open'] = np.nan
    out = run_backtest(p, s, cfg())
    assert out['unresolved_positions'] == 1
    assert out['trades'][-1]['date'] == '2024-01-02'


def test_entry_volume_cannot_leak_into_open_fill():
    p, s = fixture()
    expected = run_backtest(p, s, cfg())
    p.loc[p.date == '2024-01-02', 'volume'] = 0
    changed = run_backtest(p, s, cfg())
    assert expected['trades'][0] == changed['trades'][0]


def test_large_nav_cash_conservation():
    p, s = fixture(61)
    scale = {d: 1.8**i for i,d in enumerate(sorted(p.date.unique()))}
    p['open'] *= p.date.map(scale)
    p['close'] = p.open
    s['score'] = [float((pd.Timestamp(d).day % 2) == (c == 'A')) for c,d in zip(s.code,s.date)]
    out = run_backtest(p, s, Config(top_k=1, rebalance_sessions=2, bootstrap_samples=5))
    assert min(d['cash'] for d in out['daily']) >= 0
    assert float(out['metrics']['final_net_value']) > 1e12


def test_prediction_types_fail_closed():
    p, s = fixture()
    s['score'] = s.score.astype(str)
    with pytest.raises(ValueError):
        run_backtest(p, s, cfg())
    _, s = fixture()
    s['model'] = 123
    with pytest.raises(ValueError):
        run_backtest(p, s, cfg())


def test_random_keys_invariant_to_future_listings():
    from research_lab.backtest import baseline_predictions
    p, _ = fixture(30)
    c = cfg()
    before = baseline_predictions(p, 'random', c)
    future = p.iloc[-1:].assign(code='C', date=p.date.max()+pd.Timedelta(days=1))
    after = baseline_predictions(pd.concat([p, future]), 'random', c)
    after = after[after.date <= p.date.max()]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_cli_parquet_shards_provenance_and_no_clobber(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from V6.experimental.v7_portfolio_metrics import canonical_sha256
    p, s = fixture()
    prices, a, b, output = [tmp_path/name for name in ['prices.parquet','a.parquet','b.parquet','result.json']]
    p.to_parquet(prices)
    s[s.code == 'A'].to_parquet(a)
    s[s.code == 'B'].to_parquet(b)
    cmd = [sys.executable, '-m', 'research_lab.backtest', '--prices', str(prices), '--predictions',
           str(a), str(b), '--out', str(output), '--rebalance-sessions', '2', '--top-k', '1', '--bootstrap-samples', '5']
    result = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH':'V6'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    d = json.loads(output.read_text())
    assert d['canonical_sha256'] == canonical_sha256(d)
    assert len(d['provenance']['implementation_hashes']) == 4
    assert len(d['provenance']['inputs']['predictions']) == 2
    before = output.read_bytes()
    result = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH':'V6'}, capture_output=True, text=True)
    assert result.returncode != 0
    assert output.read_bytes() == before


def test_cli_hashes_loaded_bytes_even_if_shared_path_changes(tmp_path, monkeypatch):
    import json
    import sys
    import research_lab.backtest as cli
    p, s = fixture()
    pp, sp, out = [tmp_path/n for n in ['p.parquet', 's.parquet', 'out.json']]
    p.to_parquet(pp)
    s.to_parquet(sp)
    original_hash = cli.file_hash(sp)
    original_run = cli.run_backtest
    def replaced(*args):
        result = original_run(*args)
        s.assign(score=-s.score).to_parquet(sp)
        return result
    monkeypatch.setattr(cli, 'run_backtest', replaced)
    monkeypatch.setattr(sys, 'argv', ['backtest', '--prices', str(pp), '--predictions', str(sp),
                        '--out', str(out), '--rebalance-sessions', '2', '--bootstrap-samples', '5'])
    cli.main()
    result = json.loads(out.read_text())
    assert result['provenance']['inputs']['predictions'][0]['sha256'] == original_hash
