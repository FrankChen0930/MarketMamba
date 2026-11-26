"""Synthetic endpoint A oracles. No formal data or calendar admission."""
from dataclasses import replace
from datetime import date, timedelta
import copy

import pytest
from tests.forward_tmp import tmp_path  # noqa: F401
from tools.forward_wp1 import Calendar, Stop
from tools import forward_wp6 as w


def fixture():
    days = [(date(2027, 4, 14) + timedelta(days=i)).isoformat() for i in range(410)]
    days = [d for d in days if date.fromisoformat(d).weekday() < 5]
    c = Calendar(days, days[0], days[-1], {'synthetic': True})
    sessions = days[:253]
    bars = {d: {code: w.Bar(10., 10., True, False, True, 'SYNTHETIC',
                           d+'T08:00:00+08:00') for code in ('A', 'B', 'C')}
            for d in days}
    signals = {d: w.Signal({'A': 3., 'B': 2., 'C': 1.}, ('A', 'B', 'C'),
                           {'A': 3e7, 'B': 3e7, 'C': 3e7}, 'model-0',
                           d+'T18:00:00+08:00', 'SYNTHETIC') for d in days}
    cfg = replace(w.p2_primary_config(), top_k=1, hold_buffer_rank=2,
                  commission=0., sell_tax=0., slippage_bps=0.)
    return c, sessions, bars, signals, cfg


def run(f=None, **kwargs):
    c, s, b, p, cfg = f or fixture()
    return w.run_endpoint(c, s, b, p, config=cfg, **kwargs)


def test_hand_r1_includes_first_minimum_and_r252_has_close_without_exit():
    f = fixture(); c, s, b, p, cfg = f
    b[s[1]]['A'] = replace(b[s[1]]['A'], close=11.)
    # Return to 10 next day, with terminal intraday +20%.
    b[s[252]]['A'] = replace(b[s[252]]['A'], close=12.)
    out = run(f); a = out['strategy']
    assert len(a['returns']) == 252 and len(a['daily']) == 253
    assert a['daily'][0]['nav'] == 1.
    assert a['returns'][0] == pytest.approx(.099978, abs=1e-13)
    assert a['returns'][-1] == pytest.approx(.2, abs=1e-12)
    assert a['daily'][-1]['at'] == s[252]+'T13:30:00+08:00'
    assert a['daily'][-1]['quantities']['A'] > 0
    assert a['daily'][-1]['cost'] == 0
    assert not [t for t in a['trades'] if t['date'] == s[252]]
    assert out['forward_admitted'] is False


def test_same_benchmark_clock_and_cost_asymmetry():
    out = run()
    a, b = out['strategy'], out['benchmark']
    assert [r['at'] for r in a['daily']] == [r['at'] for r in b['daily']]
    assert a['returns'][0] == pytest.approx(-20/1e6)
    assert b['returns'][0] == pytest.approx(0, abs=1e-13)
    assert len(out['excess_returns']) == 252
    assert a['daily'][0] == b['daily'][0]


@pytest.mark.parametrize('length', [252, 254])
def test_exact_252_253_boundary(length):
    c, s, b, p, cfg = fixture()
    with pytest.raises(Stop, match='253'):
        w.run_endpoint(c, c.sessions[:length], b, p, config=cfg)


def test_s253_and_terminal_signal_cannot_affect_endpoint():
    f = fixture(); expected = run(f)
    c, s, b, p, cfg = f
    p[s[-1]] = None
    b[c.shift(s[-1], 1)] = None
    p[c.shift(s[-1], 1)] = None
    assert run(f) == expected
    assert s[252] == '2028-03-31' and c.shift(s[252], 1) == '2028-04-03'


def test_phase_and_normal_rebalance_fees():
    f = fixture(); c, s, b, p, cfg = f
    for d in c.sessions[20:]:
        p[d] = replace(p[d], scores={'A': -1., 'B': 2., 'C': 3.})
    out = run(f)['strategy']
    orders = [t for t in out['trades'] if t['date'] == s[21]]
    assert {t['code'] for t in orders} == {'A', 'C'}
    assert out['daily'][21]['cost'] == pytest.approx(40/1e6)
    assert {t['date'] for t in out['trades']} <= {s[k] for k in range(1, 253, 20)}
    assert out['daily'][21]['signal'] == s[20]


def test_suspension_qualified_stale_then_resume():
    f = fixture(); c, s, b, p, cfg = f
    b[s[-1]]['A'] = replace(b[s[-1]]['A'], close=None, open=None,
                            suspended=True, executable=False)
    out = run(f)
    for arm in ('strategy', 'benchmark'):
        assert out[arm]['daily'][-1]['stale'] == ['A']
        assert out[arm]['daily'][-1]['nav'] == out[arm]['daily'][-2]['nav']
    b[s[5]]['A'] = replace(b[s[5]]['A'], close=None, open=None,
                           suspended=True, executable=False)
    b[s[6]]['A'] = replace(b[s[6]]['A'], close=11.)
    assert run(f)['strategy']['returns'][5] == pytest.approx(.1)


@pytest.mark.parametrize('change', [dict(close=None), dict(listed=None), dict(listed=False),
    dict(source=''), dict(asof=None), dict(event=True), dict(close=0.), dict(close=float('nan')),
    dict(asof='2029-01-01T08:00:00+08:00')])
def test_unknown_held_valuation_invalidates_pair(change):
    f = fixture(); c, s, b, p, cfg = f
    # C is benchmark-only: may not silently intersect remaining strategy days.
    b[s[-1]]['C'] = replace(b[s[-1]]['C'], **change)
    with pytest.raises(Stop, match='INVALID_FOR_CONFIRMATION'):
        run(f)


@pytest.mark.parametrize('missing', ['market', 'signal', 'universe', 'session'])
def test_missing_day_or_signal_support_stops(missing):
    f = fixture(); c, s, b, p, cfg = f
    if missing == 'market': del b[s[17]]
    if missing == 'signal': del p[s[17]]
    if missing == 'universe': p[s[17]] = replace(p[s[17]], eligible=None)
    if missing == 'session': s[17] = s[16]
    with pytest.raises(Stop): run(f)


def test_first_blocked_fill_keeps_cash_does_not_restart_clock():
    f = fixture(); c, s, b, p, cfg = f
    b[s[1]]['A'] = replace(b[s[1]]['A'], open=None, executable=False)
    out = run(f)['strategy']
    assert out['returns'][0] == 0 and out['daily'][1]['cash'] == 1.
    assert out['daily'][1]['blocked'] == ['A']
    assert not [t for t in out['trades'] if t['date'] < s[21]]
    assert len(out['returns']) == 252


def test_switch_does_not_reset_nav_holdings_buffer_pending_or_phase():
    f = fixture(); c, s, b, p, cfg = f
    for d in c.sessions[20:]:
        p[d] = replace(p[d], scores={'A': -1., 'B': 2., 'C': 3.})
    b[s[21]]['A'] = replace(b[s[21]]['A'], open=None, executable=False)
    expected = run(f)
    for d in c.sessions[22:]: p[d] = replace(p[d], model_id='model-1')
    actual = run(f)
    for arm in ('strategy', 'benchmark'):
        assert actual[arm]['returns'] == expected[arm]['returns']
        assert [{k:v for k,v in t.items() if k != 'signal_model'} for t in actual[arm]['trades']] == [
            {k:v for k,v in t.items() if k != 'signal_model'} for t in expected[arm]['trades']]
        for left, right in zip(actual[arm]['daily'], expected[arm]['daily']):
            assert {k:v for k,v in left.items() if k != 'model_id'} == {
                k:v for k,v in right.items() if k != 'model_id'}
    state = actual['strategy']['daily'][23]
    assert state['pending'] == ['A'] and state['quantities']['A'] > 0
    assert state['phase'] == 2 and state['model_id'] == 'model-1'


def test_switch_on_s251_has_no_extra_trade_and_tie_break_is_code():
    f = fixture(); c, s, b, p, cfg = f
    p[s[251]] = replace(p[s[251]], model_id='last', scores={'C': 9., 'A': 0., 'B': 0.})
    assert run(f)['strategy']['daily'][-1]['model_id'] == 'last'
    assert run(f)['strategy']['daily'][-1]['cost'] == 0
    for d in p: p[d] = replace(p[d], scores={'C': 1., 'B': 1., 'A': 1.})
    assert run(f)['strategy']['trades'][0]['code'] == 'A'


def test_dryrun_is_separate_and_input_state_cannot_be_inherited():
    formal = run()
    trial = run(phase='TRIAL')
    assert trial['phase'] == 'TRIAL' and formal['phase'] == 'FORMAL'
    assert formal['strategy']['daily'][0]['quantities'] == {}
    with pytest.raises(Stop, match='trial|inherit'):
        run(initial_state=trial['strategy']['daily'][-1])


def test_does_not_call_legacy_simulator(monkeypatch):
    from research_lab import portfolio
    def forbidden(*a, **kw): raise AssertionError('legacy endpoint called')
    monkeypatch.setattr(portfolio, 'run_backtest', forbidden)
    monkeypatch.setattr(portfolio, '_simulate', forbidden)
    assert len(run()['strategy']['returns']) == 252


def test_endpoint_gate_delegates_wp5_with_exact_sessions(monkeypatch):
    c, s, b, p, cfg = fixture(); out = run((c,s,b,p,cfg)); calls=[]
    monkeypatch.setattr(w.wp5, 'manual_unblind', lambda **kw: calls.append(kw) or {'synthetic':True})
    w.endpoint_unblind(out, calendar=c, directories=[], external_head='synthetic',
        state='VALID', at=c.shift(s[-1],11)+'T18:00:00+08:00', publication_complete=True,
        human='Human', manual=True, frozen_path='synthetic', pin={}, output='synthetic')
    assert calls[0]['expected_sessions'] == s[1:]
    broken = copy.deepcopy(out); broken['strategy']['returns'].pop()
    with pytest.raises(Stop): w.endpoint_unblind(broken, calendar=c)


@pytest.mark.parametrize('change', [dict(publication_complete=False), dict(human='Codex'),
    dict(manual=False), dict(state='INVALID_FOR_CONFIRMATION'),
    dict(at='2028-04-17T17:59:59+08:00')])
def test_real_wp5_unblind_prerequisites(tmp_path, change):
    c,s,b,p,cfg = fixture(); out=run((c,s,b,p,cfg))
    frozen=tmp_path/'frozen.json'; pin=w.wp5.freeze_script(frozen)
    args=dict(calendar=c,directories=[],external_head='synthetic',state='VALID',
        at=c.shift(s[-1],11)+'T18:00:00+08:00',publication_complete=True,
        human='Human',manual=True,frozen_path=frozen,pin=pin,output=tmp_path/'opening.json')
    with pytest.raises(Stop): w.endpoint_unblind(out,**dict(args,**change))
    assert not (tmp_path/'opening.json').exists()


def test_mature_but_missing_chain_still_cannot_open(tmp_path):
    c,s,b,p,cfg = fixture(); out=run((c,s,b,p,cfg))
    frozen=tmp_path/'frozen.json'; pin=w.wp5.freeze_script(frozen)
    with pytest.raises(Stop):
        w.endpoint_unblind(out,calendar=c,directories=[],external_head='synthetic',
            state='VALID',at=c.shift(s[-1],11)+'T18:00:00+08:00',publication_complete=True,
            human='Human',manual=True,frozen_path=frozen,pin=pin,output=tmp_path/'opening.json')
    with pytest.raises(Stop): w.endpoint_unblind(run(phase='TRIAL'),calendar=c)


def test_proportional_commission_tax_and_slippage_each_actual_order():
    c,s,b,p,cfg=fixture()
    cfg=replace(cfg,commission=.001,sell_tax=.003,slippage_bps=10.)
    for d in c.sessions[20:]: p[d]=replace(p[d],scores={'A':-1.,'B':2.,'C':3.})
    out=run((c,s,b,p,cfg))['strategy']
    assert out['returns'][0] == pytest.approx(1/(1+.001*.6+.001)-1,abs=1e-13)
    orders=[t for t in out['trades'] if t['date']==s[21]]
    buy=sum(t['notional'] for t in orders if t['notional']>0)
    sell=-sum(t['notional'] for t in orders if t['notional']<0)
    expected=buy*(.001*.6+.001)+sell*(.001*.6+.003+.001)
    assert out['daily'][21]['cost']==pytest.approx(expected,abs=1e-13)


def test_future_scores_universe_prices_do_not_change_prefix():
    f=fixture(); c,s,b,p,cfg=f; old=run(f)
    for d in c.sessions[100:]:
        p[d]=replace(p[d],scores={'B':5.},eligible=('B',),dollar_volume={'B':3e7})
        b[d]['A']=replace(b[d]['A'],close=13.)
    changed=run(f)
    for arm in ('strategy','benchmark'):
        assert old[arm]['daily'][:100]==changed[arm]['daily'][:100]


@pytest.mark.parametrize('patch', [dict(model_id=''),dict(source=''),
    dict(asof='2027-04-14T18:00:01+08:00'),dict(scores={'A':float('nan')}),
    dict(scores={'unknown':1.}),dict(dollar_volume={})])
def test_invalid_signals_fail_closed(patch):
    f=fixture(); c,s,b,p,cfg=f; p[s[0]]=replace(p[s[0]],**patch)
    with pytest.raises(Stop): run(f)


@pytest.mark.parametrize('mutation', ['terminal_order', 'extra_session'])
def test_source_mutations_are_killed(monkeypatch, mutation):
    """Run the same hand oracle against actual mutated module source in memory."""
    import sys
    import types
    from pathlib import Path
    source = Path(w.__file__).read_text()
    if mutation == 'terminal_order':
        old = 'if (k-1)%20 == 0:'
        new = ('if (k-1)%20 == 0 or k == 252:\n'
               '            if k == 252: signal=replace(signal,scores={},eligible=(),dollar_volume={})')
    else:
        old = 'sessions=endpoint_sessions(calendar,sessions)\n    config='
        new = ('sessions=endpoint_sessions(calendar,sessions)\n'
               '    sessions=sessions+(calendar.shift(sessions[-1],1),)\n    config=')
    assert source.count(old) == 1
    module = types.ModuleType('wp6_mutant_'+mutation)
    module.__file__ = w.__file__
    monkeypatch.setitem(sys.modules,module.__name__,module)
    exec(compile(source.replace(old,new),module.__file__,'exec'),module.__dict__)
    monkeypatch.setitem(globals(),'w',module)
    with pytest.raises(AssertionError):
        test_hand_r1_includes_first_minimum_and_r252_has_close_without_exit()
