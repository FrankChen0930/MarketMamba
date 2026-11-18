"""P1 causal and accounting regressions; fixtures have hand-checkable outcomes."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from research_lab import portfolio_eval as pe, rolling_scores as rs
from research_lab import portfolio as pf
from test_portfolio import fixture, cfg


def test_cost_removal_control():
    p, s = fixture()
    out = pe.evaluate_cell(p, s, pf.Config(top_k=1, rebalance_sessions=2, commission=.01,
                           sell_tax=.02, slippage_bps=0, bootstrap_samples=5))
    assert out['statistics']['total_return'] == pytest.approx((1-.03)/1.01-1)
    assert out['gross_minus_net_return'] > .03


def test_same_day_signal_control():
    p, s = fixture()
    s.loc[s.date == pd.Timestamp('2024-01-02'), 'score'] *= -1
    out = pe.evaluate_cell(p, s, cfg(bootstrap_samples=5))
    assert out['trades'][0]['code'] == 'A'
    assert out['trades'][0]['date'] == '2024-01-02'


def test_blocked_holding_control():
    p, s = fixture()
    p.loc[(p.code == 'A') & (p.date == '2024-01-10'), 'open'] = np.nan
    out = pe.evaluate_cell(p, s, cfg(bootstrap_samples=5))
    assert out['unresolved_positions'] == 1
    assert out['blocked_orders'] == 1
    assert out['holding_count']['terminal'] == 1


def test_future_strata_control():
    days = pd.bdate_range('2024-01-01', periods=4)
    caps = pd.DataFrame([dict(code=c, date=d, market_value=v) for d in days
                         for c,v in zip('ABC', [1.,2.,3.])])
    original = pe.lagged_strata(caps, days)
    caps.loc[caps.date >= days[2], 'market_value'] *= -1
    changed = pe.lagged_strata(caps, days)
    assert original.loc[(days[2], 'A'), 'stratum'] == 'small'
    pd.testing.assert_frame_equal(original.loc[:days[2]], changed.loc[:days[2]])
    assert pd.isna(original.loc[(days[0], 'A'), 'stratum'])


def test_rank_average_requires_paired_seed_support():
    s = pd.DataFrame([dict(code=c, date='2023-01-01', fold=2023, seed=seed,
                          system='c0plus', score=v) for seed, vals in [(17,[100.,0.]),(29,[0.,1.])]
                      for c,v in zip('AB',vals)])
    a = rs.rank_average(s)
    assert a.score.tolist() == [.75,.75]
    with pytest.raises(ValueError, match='seed support'):
        rs.rank_average(s.iloc[:-1])
    with pytest.raises(ValueError):
        rs.rank_average(pd.concat([s,s.iloc[:1]]))


def test_capacity_uses_signal_day_not_entry_day():
    p,s = fixture()
    p.loc[p.date == '2024-01-02', 'volume'] = 1e12
    out = pe.evaluate_cell(p,s,cfg(bootstrap_samples=5))
    assert out['capacity']['1000000']['median'] == pytest.approx(100.)


def test_grid_and_price_basis():
    cells = list(pe.configurations())
    assert len(cells) == 432
    assert len({tuple(vars(c).items()) for c in cells}) == 432
    assert pe.primary_config() in cells


def test_universe_mask_does_not_block_exiting_holdings():
    p,s = fixture()
    p['universe_eligible'] = True
    p.loc[(p.code=='A') & (p.date >= '2024-01-03'), 'universe_eligible'] = False
    out = pe.evaluate_cell(p,s,cfg(bootstrap_samples=5))
    assert out['blocked_orders'] == 0
    assert any(t['code']=='A' and t['notional'] < 0 for t in out['trades'])


def test_output_guard_and_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, 'ARTIFACT_ROOT', tmp_path/'p1')
    with pytest.raises(ValueError):
        rs.output_path(tmp_path/'escape')
    rs.main(['--folds-root','missing','--years','2023','--seeds','17',
             '--out-dir',str(tmp_path/'p1'/'test')])
    assert not (tmp_path/'p1').exists()


def test_collect_reuses_recipe_and_rows_read_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from research_lab import c0plus, seq_runner
    fold = tmp_path/'fold_2023'
    fold.mkdir()
    np.save(fold/'stock_idx.npy', [0,1,0])
    np.save(fold/'stock_ids.npy', ['A','B','A'])
    np.save(fold/'dates.npy', np.array(['2023-01-01','2023-01-01','2023-01-02'], dtype='datetime64[D]'))
    mask = np.array([False,True,True])
    monkeypatch.setattr(seq_runner,'SeqData',lambda directory, normalize: SimpleNamespace(
        eligible=mask, y={'y5':np.array([0.,.2,.3])}))
    def run(directory, label, max_train_rows, seed, **kw):
        assert (directory,label,max_train_rows,seed) == (fold,'y5',123,17)
        assert kw['row_mask'] is mask
        assert kw['n_boot']==1
        assert set(kw)=={'row_mask','n_boot','log'}
        return dict(eval_idx=np.array([1,2]),fit_rows=123,eval_rows=2,
                    scores={'c0plus':np.array([2.,3.]),'lightgbm_day_t':np.array([4.,5.])})
    monkeypatch.setattr(c0plus,'run',run)
    before = {f.name:f.read_bytes() for f in fold.iterdir()}
    scores, timing = rs.collect(fold,17,123)
    assert scores.code.tolist()==['B','A','B','A']
    assert scores.score.tolist()==[4.,5.,2.,3.]
    assert timing['fit_rows']==123 and timing['eval_rows']==2
    assert {f.name:f.read_bytes() for f in fold.iterdir()}==before
    (fold/'stock_idx.npy').unlink()
    with pytest.raises(ValueError,match='read-only'):
        rs.collect(fold,17,123)


def test_export_manifest_rank_roundtrip_no_overwrite(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(rs,'ARTIFACT_ROOT',tmp_path/'p1')
    fold = tmp_path/'inputs'/'fold_2023'
    fold.mkdir(parents=True)
    for name in rs.MATRIX_FILES:
        (fold/name).write_bytes(b'fixture')
    def collect(directory, seed, max_train_rows):
        return pd.DataFrame([dict(code=c,date=pd.Timestamp('2023-01-01'),score=float(i),
                       fold=2023,system=system,seed=seed,label_y5=float(i))
                       for system in rs.SYSTEMS for i,c in enumerate('AB')]), dict(fit_rows=2,eval_rows=2,wall_seconds=.01,seed=seed)
    monkeypatch.setattr(rs,'collect',collect)
    args = ['--folds-root',str(fold.parent),'--years','2023','--seeds','17','29',
            '--out-dir',str(tmp_path/'p1'/'export'),'--run']
    rs.main(args)
    out = tmp_path/'p1'/'export'
    mf = json.loads((out/'manifest.json').read_text())
    rank = pd.read_parquet(out/'rank_average.parquet')
    assert len(rank)==4 and rank.score.tolist()==[.5,1.,.5,1.]
    assert mf['seeds']==[17,29]
    assert mf['outputs']['rank_average.parquet']==rs.identity(out/'rank_average.parquet')
    with pytest.raises(FileExistsError):
        rs.main(args)


def test_pooled_inference_does_not_bridge_fold_navs():
    rows=[dict(daily_returns=[dict(net=.02,benchmark=.01)]*30),
          dict(daily_returns=[dict(net=.02,benchmark=.01)]*30)]
    r=pe.pooled_test(rows,n_boot=20)
    assert r['days']==60
    assert r['mean_daily_excess']==pytest.approx(.01)
    assert r['ci95']==pytest.approx([.01,.01])


def test_stratum_benchmark_uses_same_eligible_universe():
    p,s=fixture()
    p['universe_eligible']=p.code.eq('A')
    p.loc[(p.code=='B') & (p.date >= '2024-01-03'),['open','close']]=100.
    r=pe.evaluate_cell(p,s,cfg(bootstrap_samples=5))
    assert r['statistics']['benchmark_return']==0.
    assert r['statistics']['mean_daily_excess']==0.


def test_evaluator_cli_pinned_inputs_raw_fallback_no_clobber(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(rs,'ARTIFACT_ROOT',tmp_path/'p1')
    scores_dir=tmp_path/'scores'
    data=tmp_path/'data'
    scores_dir.mkdir();data.mkdir()
    prices,pred=fixture(31)
    scores=pred.rename(columns={'model':'system'}).assign(system='c0plus',fold=2023,label_y5=.1)
    scores.to_parquet(scores_dir/'rank_average.parquet',index=False)
    rs.write_json(scores_dir/'manifest.json',dict(outputs={'rank_average.parquet':rs.identity(scores_dir/'rank_average.parquet')},
                  seeds=[17],years=[2023],parameters={'max_train_rows':2000}))
    prices.rename(columns=dict(code='stock_id',date='Date',open='Open',close='Close',volume='Volume')).to_parquet(data/'prices_raw.parquet',index=False)
    caps=prices[['code','date']].assign(market_value=np.where(prices.code.eq('A'),1.,2.))
    caps.rename(columns=dict(code='stock_id',date='Date')).to_parquet(data/'market_value_raw.parquet',index=False)
    out=tmp_path/'p1'/'test.json'
    args=['--scores-dir',str(scores_dir),'--data-root',str(data),'--out',str(out),
          '--primary-only','--bootstrap-samples','5']
    pe.main(args)
    assert not out.exists()
    pe.main(args+['--run'])
    result=json.loads(out.read_text())
    assert result['status']=='COMPLETED_RESEARCH_ONLY'
    assert result['grid_complete'] is False
    assert len(result['results'])==4
    assert result['price_basis_fallback'].startswith('raw:')
    assert result['inputs']['prices']==rs.identity(data/'prices_raw.parquet')
    assert result['clean_fold_hypothesis']['c0plus']['status']=='INCOMPLETE_ENGINEERING_ONLY'
    with pytest.raises(FileExistsError):
        pe.main(args+['--run'])
    scores.assign(score=0.).to_parquet(scores_dir/'rank_average.parquet',index=False)
    with pytest.raises(ValueError,match='identity mismatch'):
        pe.load_inputs(scores_dir,data)


def test_full_grid_reports_unavailable_basis_instead_of_dropping_cells(monkeypatch):
    import copy
    p,s=fixture(31)
    s=s.rename(columns={'model':'system'}).assign(system='c0plus',fold=2023)
    caps=p[['code','date']].assign(market_value=np.where(p.code.eq('A'),1.,2.))
    template=pe.evaluate_cell(p,s.assign(model='c0plus'),cfg(bootstrap_samples=2))
    def cell(prices,predictions,config):
        r=copy.deepcopy(template)
        r['config']=vars(config)
        return r
    monkeypatch.setattr(pe,'evaluate_cell',cell)
    r=pe.evaluate(s,p,caps,dict(seeds=[17],parameters={'max_train_rows':2000}),bootstrap_samples=2)
    all_cells=[x for x in r['results'] if x['stratum']=='all']
    assert len(all_cells)==432
    assert sum(x['status']=='UNAVAILABLE_NO_ADJ_CLOSE' for x in all_cells)==216
    assert sum(x['status']=='COMPUTED' for x in all_cells)==216
    assert len(r['results'])==435


def test_evaluation_benchmark_excludes_unscored_names():
    p,s=fixture(31)
    s=s[s.code.eq('A')].rename(columns={'model':'system'}).assign(system='c0plus',fold=2023)
    p['volume']=1e8
    p.loc[(p.code=='B') & (p.date >= '2024-01-03'),['open','close']]=100.
    caps=p[['code','date']].assign(market_value=np.where(p.code.eq('A'),1.,2.))
    r=pe.evaluate(s,p,caps,dict(seeds=[17],parameters={'max_train_rows':2000}),primary_only=True,bootstrap_samples=2)
    result=next(x for x in r['results'] if x['stratum']=='all')
    # B cannot be ranked by either strategy and must not lift the benchmark.
    assert result['statistics']['benchmark_return'] < 0.
