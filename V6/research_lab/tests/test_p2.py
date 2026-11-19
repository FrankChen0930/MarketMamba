"""P2 causal, currency and default compatibility regressions."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from research_lab import portfolio as pf, portfolio_eval as pe
from test_portfolio import fixture, cfg


def buffer_fixture():
    p, s = fixture()
    # A is first initially, then second. Both are inside buffer 2.
    s.loc[s.date.ge('2024-01-03'), 'score'] *= -1
    return p, s


def test_buffer_retains_incumbent():
    p, s = buffer_fixture()
    out = pf.run_backtest(p, s, cfg(hold_buffer_rank=2, bootstrap_samples=2))
    assert {t['code'] for t in out['trades']} == {'A'}
    assert len(out['trades']) == 2


def test_minimum_fee_per_order():
    p, s = fixture()
    c = cfg(min_fee=20., initial_capital=1e6, bootstrap_samples=2)
    out = pf.run_backtest(p, s, c)
    assert out['daily'][-1]['nav'] == pytest.approx(1-40/1e6, abs=1e-12)
    assert out['minimum_fee_cost_fraction'] == pytest.approx(1.)


def test_initial_capital_conversion_includes_nav():
    p, s = fixture()
    # Double NAV after initial fill. Terminal delta is 2 initial NAV units.
    p.loc[p.date.ge('2024-01-03'), ['open','close']] = 20.
    c = cfg(min_fee=20., initial_capital=1e6, bootstrap_samples=2)
    out = pf.run_backtest(p, s, c)
    assert out['daily'][-1]['nav'] == pytest.approx(2-60/1e6, abs=1e-12)
    larger = pf.run_backtest(p, s, replace(c, initial_capital=1e7))
    assert larger['daily'][-1]['nav'] == pytest.approx(2-60/1e7, abs=1e-12)
    # At NAV 2, normalized sale 2 => TWD 2m; proportional fee wins.
    assert pf.transaction_cost(0., 2., replace(c, commission=.01), order_deltas=np.array([-2.])) == pytest.approx(.02)


def test_buffer_never_reads_future_ranking():
    p, s = buffer_fixture()
    c = cfg(hold_buffer_rank=2, bootstrap_samples=2)
    out = pf.run_backtest(p, s, c)
    future = s.copy()
    future.loc[future.date.eq('2024-01-06'), 'score'] = 0  # no such observation
    # Change entry-day (after the Jan 3 signal) to rank B first / remove A.
    future = future[~(future.date.eq('2024-01-04') & future.code.eq('A'))]
    changed = pf.run_backtest(p, future, c)
    assert [t for t in out['trades'] if t['date'] <= '2024-01-04'] == [t for t in changed['trades'] if t['date'] <= '2024-01-04']
    assert out['daily'][:4] == changed['daily'][:4]


@pytest.mark.parametrize('kw', [dict(hold_buffer_rank=19),dict(hold_buffer_rank=True),dict(hold_buffer_rank=20.5),dict(min_fee=-1),dict(initial_capital=0),dict(min_fee=20),dict(min_fee=20,initial_capital=100)])
def test_config_rejects_invalid_p2(kw):
    with pytest.raises(ValueError):
        pf.Config(**kw)


def test_default_buffer_exact_daily_bits():
    p, s = buffer_fixture()
    a = pf.run_backtest(p,s,cfg(bootstrap_samples=2))
    b = pf.run_backtest(p,s,cfg(hold_buffer_rank=1, initial_capital=1e6, bootstrap_samples=2))
    assert a['daily'] == b['daily']
    assert a['trades'] == b['trades']


def test_p2_frozen_grid():
    cells = list(pe.p2_configurations())
    assert len(cells) == 972
    assert len({tuple(vars(c).items()) for c in cells}) == 972
    assert pe.p2_primary_config() in cells
    assert pe.primary_config().rebalance_sessions == 10


def test_minimum_fee_no_phantom_orders_and_cost_components():
    p,s = fixture()
    c = cfg(min_fee=20,initial_capital=1e6,bootstrap_samples=2)
    out = pf.run_backtest(p,s,c)
    assert len(out['trades']) == 2
    d = np.array([.001,-.002])
    paid = pf.transaction_cost(.001,.002,replace(c,commission=.001,sell_tax=.003,slippage_bps=10),d)
    assert paid == pytest.approx(40/1e6+.003*.002+.001*.003)
    assert pf.transaction_cost(0,0,c,np.zeros(2)) == 0


def test_p2_resume_input_and_payload_binding(tmp_path,monkeypatch):
    from test_p1 import review_inputs
    import json
    data = review_inputs()
    root = tmp_path/'parts'
    a = pe.evaluate(*data,mode='p2',primary_only=True,bootstrap_samples=2,part_dir=root,input_identity={'sha256':'fixture'})
    part = next(root.glob('part-*.json'))
    original = part.read_bytes()
    def refuse(*a):
        raise AssertionError('resume recomputed a durable cell')
    monkeypatch.setattr(pe,'checked_cell',refuse)
    assert pe.evaluate(*data,mode='p2',primary_only=True,bootstrap_samples=2,part_dir=root,input_identity={'sha256':'fixture'}) == a
    assert part.read_bytes() == original
    assert len(a['results'])==2  # main plus same-score P1 reference, no P1 strata
    assert all(t['status']=='INCOMPLETE_ENGINEERING_ONLY' for t in a['hypotheses']['tests'].values())
    saved=json.loads(original)
    assert saved['contract']['source_commit']
    assert saved['contract']['inputs']=={'sha256':'fixture'}
    saved['payload']['results'][0]['fold']=2022
    part.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='part identity/hash mismatch'):
        pe.evaluate(*data,mode='p2',primary_only=True,bootstrap_samples=2,part_dir=root,input_identity={'sha256':'fixture'})
    part.write_bytes(original)
    with pytest.raises(ValueError,match='part identity/hash mismatch'):
        pe.evaluate(*data,mode='p2',primary_only=True,bootstrap_samples=2,part_dir=root,input_identity={'sha256':'changed'})


def test_p2_holm_and_exact_pair_dates():
    manifest=dict(seeds=[17,29,43],parameters={'max_train_rows':500000})
    rows={}
    refs={}
    for y in (2021,2022,2023):
        rows[y,'c0plus']=dict(daily_returns=[dict(date=str(i),net=.01,benchmark=0.) for i in range(45)])
        refs[y,'c0plus']=dict(daily_returns=[dict(date=str(i),net=.005,benchmark=0.) for i in range(46)])
    result=pe.p2_hypotheses(rows,refs,manifest,False,1000)
    assert all(t['p_holm']==pytest.approx(2/1001) and t['status']=='CONFIRMATORY' for t in result['tests'].values())
    assert result['tests']['H-P2a']['decision']=='淨有益'
    assert result['tests']['H-P2b']['decision']=='低換手設計有改善'
    assert all(r['days']==45 and r['excluded_p1_dates']==['45'] for r in result['pairing'])
    del refs[2022,'c0plus']
    missing=pe.p2_hypotheses(rows,refs,manifest,False,1000)
    assert all(t['decision']=='未證實' for t in missing['tests'].values())
