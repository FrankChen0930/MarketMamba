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
    assert len(cells) == 14
    assert len({tuple(vars(c).items()) for c in cells}) == 14
    assert pe.p2_primary_config() in cells
    assert pe.primary_config().rebalance_sessions == 10
    primary=pe.p2_primary_config()
    assert replace(primary,initial_capital=1e7) in cells
    assert replace(primary,initial_capital=1e7,commission_discount=.28) in cells
    assert replace(primary,rebalance_sessions=40) in cells
    assert replace(primary,top_k=30,hold_buffer_rank=60) in cells


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


def test_p2_benchmark_proportional_and_descriptive_floor():
    p,s=fixture(51)
    c=replace(pe.p2_primary_config(),top_k=1,hold_buffer_rank=2,min_dollar_volume=0.,initial_capital=10000.,bootstrap_samples=2)
    out=pf.run_backtest(p,s,c)
    proportional=pf.run_backtest(p,s,replace(c,min_fee=0.))
    floor=pf.run_backtest(p,s,replace(c,benchmark_proportional_only=False))
    assert out['benchmark_daily']==proportional['benchmark_daily']
    assert out['daily']==floor['daily']
    sensitivity=out['benchmark_min_fee_sensitivity']
    assert sensitivity['role']=='descriptive'
    assert sensitivity['statistics']['benchmark_return'] < out['statistics']['benchmark_return']
    assert sensitivity['daily_returns']==pe.summary(floor)['daily_returns']
    assert out['benchmark_diagnostics']['minimum_fee_cost']==0


def test_frozen_overflow_keeps_tradable_incumbent_without_new_buys():
    p,s=fixture(11)
    p=pd.concat([p,p.assign(code=p.code.map({'A':'C','B':'D'}))],ignore_index=True)
    s=p[['code','date']].assign(fold=0,model='fixture',horizon=2)
    s['score']=s.code.map({'A':4.,'B':3.,'C':2.,'D':1.})
    later=s.date.ge('2024-01-03')
    s.loc[later,'score']=s.loc[later,'code'].map({'A':0.,'B':3.,'C':4.,'D':2.})
    # A stays valid at signal close but cannot fill at subsequent opens.
    p.loc[p.code.eq('A') & p.date.ge('2024-01-04'),'open']=np.nan
    out=pf.run_backtest(p,s,pf.Config(top_k=2,hold_buffer_rank=3,rebalance_sessions=2,commission=0.,sell_tax=0.,bootstrap_samples=2))
    middle=[t for t in out['trades'] if '2024-01-04'<=t['date']<out['end']]
    assert middle==[]
    assert {t['code'] for t in out['trades']}=={'A','B'}
    assert out['unresolved_positions']==1


@pytest.mark.parametrize('values,expected', [([.01,.04],[.02,.04]),([.03,.04],[.06,.06]),([.04,.01],[.04,.02])])
def test_p2_holm_unequal_and_cumulative_max(values,expected):
    assert pe.holm_adjust(values)==pytest.approx(expected)


def test_subset_never_grid_complete(tmp_path):
    from test_p1 import review_inputs
    out=pe.evaluate(*review_inputs(),mode='p2',bootstrap_samples=2,part_dir=tmp_path,selection_partial=True)
    assert out['grid_complete'] is False
    assert out['expected_grid_cells_per_fold_system']==14


def test_sparse_fold_system_pairs_never_grid_complete(tmp_path):
    # Years and systems are each complete as sets, but only 6 of 12 pairs exist.
    from test_p1 import review_inputs
    scores,prices,caps,manifest,*rest=review_inputs()
    pairs=[(2021,'c0plus'),(2022,'c0plus'),(2023,'c0plus'),(2024,'c0plus'),(2025,'c0plus'),(2026,'lightgbm_day_t')]
    base=scores.iloc[:0]
    parts=[]
    for fold,system in pairs:
        part=scores.copy(); part['fold']=fold; part['system']=system
        parts.append(part)
    sparse=pd.concat(parts,ignore_index=True)
    out=pe.evaluate(sparse,prices,caps,manifest,*rest,mode='p2',bootstrap_samples=2,part_dir=tmp_path)
    assert set(zip(sparse.fold,sparse.system))!={(f,y) for f in range(2021,2027) for y in ('c0plus','lightgbm_day_t')}
    assert out['grid_complete'] is False


def test_revision_caveats_explicit():
    text=' '.join(pe.P2_CAVEATS)
    for phrase in ('schema','run_hash','P1 parts','daily strategy and benchmark returns','proportional','descriptive'):
        assert phrase in text


def test_default_returns_match_original_p1_engine_golden():
    # Generated from git ace2eef:V6/research_lab/portfolio.py (before P2),
    # independent of this engine's branches. IEEE754 little-endian net/benchmark.
    import hashlib
    p,s=fixture(31)
    p['open']=10+np.arange(len(p))*.013
    p['close']=p.open+np.where(p.code.eq('A'),.07,-.03)
    s['score']=np.where((s.date.dt.day%3).eq(0),-s.score,s.score)
    out=pf.run_backtest(p,s,pf.Config(top_k=1,rebalance_sessions=2,commission=.001425,sell_tax=.003,slippage_bps=10,bootstrap_samples=2))
    pairs=np.array([[r['net'],r['benchmark']] for r in pe.summary(out)['daily_returns']],dtype='<f8')
    assert pairs.shape==(29,2)
    assert hashlib.sha256(pairs.tobytes()).hexdigest()=='67d4f183fcc8124611be78fc508ed5a4db2d7b515468e7e574223260e479ee98'


def test_p2_holm_one_pass_one_fail_with_unequal_bootstrap_pvalues():
    manifest=dict(seeds=[17,29,43],parameters={'max_train_rows':500000})
    rows,refs={},{}
    rng=np.random.default_rng(23)
    for y in (2021,2022,2023):
        noise=rng.normal(0,.01,100)
        rows[y,'c0plus']=dict(daily_returns=[dict(date=str(i),net=.01,benchmark=0.) for i in range(100)])
        refs[y,'c0plus']=dict(daily_returns=[dict(date=str(i),net=.01-v,benchmark=0.) for i,v in enumerate(noise)])
    tests=pe.p2_hypotheses(rows,refs,manifest,False,1000)['tests']
    a,b=tests['H-P2a'],tests['H-P2b']
    assert a['p_one_sided']==pytest.approx(1/1001)
    assert b['p_one_sided']==pytest.approx(522/1001)
    assert a['p_holm']==pytest.approx(2/1001)
    assert b['p_holm']==pytest.approx(522/1001)
    assert a['decision']=='淨有益' and b['decision']=='未證實'


@pytest.mark.parametrize('flag,value', [('--fold','2023'),('--system','c0plus')])
def test_cli_propagates_partial_selection_and_revision_caveats(tmp_path,monkeypatch,flag,value):
    from test_p1 import review_inputs
    import json
    scores,prices,caps,manifest=review_inputs()
    manifest['years']=[2023]
    monkeypatch.setattr(pe,'__file__',str(tmp_path/'V6/research_lab/portfolio_eval.py'))
    monkeypatch.setattr(pe,'load_inputs',lambda *a:(scores,prices,caps,manifest,
        {k:{'path':k,'sha256':'fixture'} for k in ('prices','market_value','scores')}, {'missing_prediction_price_keys':0}))
    monkeypatch.setattr(pe,'identity',lambda path:{'path':str(path),'sha256':'fixture'})
    monkeypatch.setattr(pe,'provenance',lambda:{'implementation':{}})
    def evaluate(*args,**kwargs):
        assert kwargs['selection_partial'] is True
        return dict(grid_complete=False,results=[])
    monkeypatch.setattr(pe,'evaluate',evaluate)
    out=tmp_path/'.artifacts/p2/smoke.json'
    pe.main(['--mode','p2','--scores-dir',str(tmp_path),'--out',str(out),flag,value,'--run'])
    saved=json.loads(out.read_text())
    assert saved['grid_complete'] is False
    assert all(c in saved['caveats'] for c in pe.P2_CAVEATS)


def test_p2_equal_buffer_still_preserves_frozen_overflow():
    assert pe.p2_primary_config().preserve_frozen_overflow is True
    p,s=fixture(11)
    p=pd.concat([p,p.assign(code=p.code.map({'A':'C','B':'D'}))],ignore_index=True)
    s=p[['code','date']].assign(fold=0,model='fixture',horizon=2)
    s['score']=s.code.map({'A':4.,'B':3.,'C':2.,'D':1.})
    later=s.date.ge('2024-01-03')
    s.loc[later,'score']=s.loc[later,'code'].map({'A':0.,'B':1.,'C':4.,'D':3.})
    p.loc[p.code.eq('A') & p.date.ge('2024-01-04'),'open']=np.nan
    c=pf.Config(top_k=2,hold_buffer_rank=2,preserve_frozen_overflow=True,
                rebalance_sessions=2,commission=0.,sell_tax=0.,bootstrap_samples=2)
    out=pf.run_backtest(p,s,c)
    assert [t for t in out['trades'] if '2024-01-04'<=t['date']<out['end']]==[]
    assert {t['code'] for t in out['trades']}=={'A','B'}
    assert pf.Config().preserve_frozen_overflow is False
