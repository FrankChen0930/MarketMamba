"""P1 causal and accounting regressions; fixtures have hand-checkable outcomes."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from research_lab import portfolio_eval as pe, rolling_scores as rs
from research_lab import portfolio as pf
from test_portfolio import fixture, cfg


@pytest.fixture(autouse=True)
def isolated_p1_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, 'ARTIFACT_ROOT', tmp_path/'p1')


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
    prices['volume']=3e7
    prices['code']=prices.code.replace({'A':'2330'})
    pred['code']=pred.code.replace({'A':'2330'})
    scores=pred.rename(columns={'model':'system'}).assign(system='c0plus',fold=2023,label_y5=.1)
    scores.to_parquet(scores_dir/'rank_average.parquet',index=False)
    rs.write_json(scores_dir/'manifest.json',dict(outputs={'rank_average.parquet':rs.identity(scores_dir/'rank_average.parquet')},
                  seeds=[17],years=[2023],parameters={'max_train_rows':2000}))
    probes=unit_prices()
    probes=pd.concat([probes.iloc[:3].assign(date=d) for d in prices.date.unique()],ignore_index=True)
    prices=pd.concat([prices[~prices.code.eq('2330')],probes],ignore_index=True)
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
    original=out.read_bytes()
    pe.main(args+['--run','--retry-errors'])
    assert out.read_bytes()==original
    assert (out.parent/'test-retry-000001.json').exists()
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


def review_inputs():
    p,s=fixture(31)
    p['volume']=1e8
    # Six codes, two per stratum; highest scores belong to large names.
    p=pd.concat([p.assign(code=c) for c in 'ABCDEF']).drop_duplicates(['date','code'])
    s=p[['date','code']].assign(score=lambda x:x.code.map(dict(zip('ABCDEF',range(6)))),system='c0plus',fold=2023)
    caps=p[['date','code']].assign(market_value=lambda x:x.code.map(dict(zip('ABCDEF',range(1,7)))))
    return s,p,caps,dict(seeds=[17],parameters={'max_train_rows':2000})


def test_review_r1_parts_resume_integrity(tmp_path,monkeypatch):
    import json
    args=review_inputs()
    first=pe.evaluate(*args,primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'run')
    parts=list((tmp_path/'run').glob('part-*.json'))
    assert len(parts)==1
    original=parts[0].read_bytes()
    monkeypatch.setattr(pe,'evaluate_cell',lambda *a:pytest.fail('completed part recomputed'))
    assert pe.evaluate(*args,primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'run')==first
    assert parts[0].read_bytes()==original
    payload=json.loads(original);payload['payload']['results'][0]['status']='tampered'
    parts[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='part.*hash'):
        pe.evaluate(*args,primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'run')


@pytest.mark.parametrize('nonfinite',[False,True])
def test_review_r1_cell_error_and_nan(tmp_path,monkeypatch,nonfinite):
    real=pe.evaluate_cell
    calls=[]
    def cell(*a):
        calls.append(1)
        if len(calls)==1:
            if not nonfinite:
                raise RuntimeError('injected cell fault')
            r=real(*a);r['statistics']['beta']=float('nan');return r
        return real(*a)
    monkeypatch.setattr(pe,'evaluate_cell',cell)
    r=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'run')
    assert r['results'][0]['status']=='ERROR'
    assert r['results'][0]['error']['type']==('NonFiniteResult' if nonfinite else 'RuntimeError')
    assert r['results'][0]['error']['message']
    assert sum(x['status']=='COMPUTED' for x in r['results'])==3
    import json
    json.dumps(r,allow_nan=False)


def test_review_r2_terminated_price_exposure():
    p,s=fixture(31)
    s=s[s.code.eq('A')]
    p=p[~(p.code.eq('A') & p.date.gt(p.date.unique()[3]))]
    r=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))
    d=r['price_termination']
    assert d['count']==1
    a=d['codes'][0]
    assert a['code']=='A' and a['last_price_date']=='2024-01-04'
    assert a['held_sessions']>0 and a['held_value_session_sum']>0
    assert a['blocked_orders']>0 and a['blocked_value_sum']>0


def test_review_r3_m3_caveats_and_window():
    r=pe.summary(pe.evaluate_cell(*fixture(31),cfg(bootstrap_samples=2)))
    assert r['return_days']==len(r['daily_returns'])
    assert r['start']=='2024-01-01' and r['end']==r['schedule'][-1]['entry']
    text=' '.join(pe.CAVEATS)
    assert 'last scheduled entry' in text and 'net-to-net' in text
    assert 'cross fold boundaries' in text and 'rebalance' in text


def test_review_m1_candidate_role(tmp_path):
    r=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2)
    assert r['results'][0]['role']=='candidate_confirmatory'
    assert r['clean_fold_hypothesis']['c0plus']['status']!='CONFIRMATORY'


def test_review_m2_separate_paired_eras(tmp_path,monkeypatch):
    s,p,c,m=review_inputs()
    s=pd.concat([s.assign(fold=y,system=sys) for y in (2023,2024) for sys in ('c0plus','lightgbm_day_t')])
    r=pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2)
    paired=r['paired_system_difference_descriptive']
    assert paired['2021_2023']['days']==21
    assert paired['2024_2026']['days']==21


def test_review_m4_volume_units():
    p=unit_prices()
    shares=pe.volume_evidence(p)
    lots=pe.volume_evidence(p.assign(volume=p.volume/1000))
    assert shares['unit']=='shares' and shares['shares_per_volume_unit']==1
    assert lots['unit']=='lots' and lots['shares_per_volume_unit']==1000
    assert (pe.normalize_volume(p.assign(volume=p.volume/1000),lots).volume==p.volume).all()
    with pytest.raises(ValueError,match='UNKNOWN'):
        pe.normalize_volume(p.assign(code='A'),pe.volume_evidence(p.assign(code='A')))


def test_review_m6_evaluate_strata_selection(tmp_path):
    r=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2)
    for stratum,allowed in [('small',set('AB')),('mid',set('CD')),('large',set('EF'))]:
        cell=next(x for x in r['results'] if x['stratum']==stratum)
        assert set(cell['selected_buy_codes'])==allowed


def test_review_m4_lots_liquidity_and_capacity():
    p,s=fixture(31)
    p['code']=p.code.replace({'A':'2330'})
    s['code']=s.code.replace({'A':'2330'})
    p['volume']=30000.  # lots: 30 million shares, TWD 300 million
    normalized=pe.normalize_volume(p,pe.volume_evidence(unit_prices(.001)))
    r=pe.evaluate_cell(normalized,s,cfg(bootstrap_samples=2,min_dollar_volume=2e7))
    assert r['selected_buy_codes']==['2330']
    assert r['capacity']['1000000']['median']==pytest.approx(1e6/3e8)


@pytest.mark.parametrize('gap', ['single_day', 'zero_volume', 'short_tail', 'recovered'])
def test_required1_transient_or_zero_volume_is_not_termination(gap):
    p,s=fixture(31)
    dates=sorted(p.date.unique())
    if gap=='zero_volume':
        p.loc[p.code.eq('A') & p.date.eq(dates[-1]), 'volume']=0.
    else:
        missing=dates[5:6] if gap=='single_day' else dates[-9:] if gap=='short_tail' else dates[3:17]
        p=p[~(p.code.eq('A') & p.date.isin(missing))]
    d=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))['price_termination']
    assert d['in_fold_terminated']['count']==0
    assert d['pre_fold_terminated']['count']==0
    assert d['transient_gap']['count']==(0 if gap=='zero_volume' else 1)
    if gap!='zero_volume':
        assert 'held_sessions' not in d['transient_gap']['codes'][0]


def test_required1_pre_fold_terminated_has_no_fold_exposure():
    p,s=fixture(31)
    s=s[s.date.ge('2024-01-15')]
    last=pd.Timestamp('2024-01-04')
    p=p[~(p.code.eq('A') & p.date.gt(last))]
    d=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))['price_termination']
    assert d['in_fold_terminated']['count']==0
    assert d['pre_fold_terminated']['count']==1
    assert d['pre_fold_terminated']['codes']==[dict(code='A',last_price_date='2024-01-04', trailing_invalid_sessions=21)]


def test_required1_ten_session_boundary():
    p,s=fixture(31)
    dates=sorted(p.date.unique())
    p=p[~(p.code.eq('A') & p.date.isin(dates[-10:]))]
    d=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))['price_termination']
    assert d['in_fold_terminated']['count']==1
    assert d['in_fold_terminated']['codes'][0]['trailing_invalid_sessions']==10


def test_required2_capacity_approximation_is_explicit():
    p,s=fixture(31)
    r=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))
    for row in r['capacity'].values():
        assert row['approximation'] is True
        assert 'NAV=1' in row['basis'] and 'drift' in row['basis']
    assert any('NAV=1' in x and 'drift' in x for x in pe.CAVEATS)


def unit_prices(factor=1.):
    return pd.DataFrame([dict(code=c,date=d,close=close,open=close,volume=v*factor)
        for d in pd.bdate_range('2023-01-02',periods=5)
        for c,close,v in [('2330',500.,4e7),('8069',1000.,2e6),('1236',20.,45000.)]])


@pytest.mark.parametrize('factor,unit',[(1.,'shares'),(.001,'lots')])
def test_required3_measured_multistock_and_daily_evidence(factor,unit):
    p=unit_prices(factor)
    e=pe.volume_evidence(p)
    assert e['unit']==unit
    assert {x['code'] for x in e['checks']}=={'2330','8069','1236'}
    assert all(x['unit']==unit and x['rows']==5 for x in e['checks'])
    assert e['universe_daily']['raw_close_times_volume_quantiles']['p50']==pytest.approx(22000900000.*factor)
    assert e['universe_daily']['unit']==unit
    changed=pe.volume_evidence(p.assign(volume=p.volume*1.1))
    assert changed['checks']!=e['checks']
    assert changed['universe_daily']!=e['universe_daily']
    assert isinstance(e['corroboration'],dict)


@pytest.mark.parametrize('mixed', ['8069','1236','universe','time_segment'])
def test_required3_inconsistent_units_fail_closed(mixed):
    p=unit_prices()
    if mixed=='universe':
        p=pd.concat([p,p[p.code.eq('2330')].assign(code='OTHER',volume=1e13)])
    elif mixed=='time_segment':
        q=p.assign(date=p.date+pd.DateOffset(years=1),volume=p.volume/1000)
        p=pd.concat([p,q])
    else:
        p.loc[p.code.eq(mixed),'volume']/=1000
    with pytest.raises(ValueError,match='inconsistent|UNKNOWN'):
        pe.normalize_volume(p,pe.volume_evidence(p))


def test_required3_missing_reference_fails_closed():
    p=unit_prices().query("code != '8069'")
    with pytest.raises(ValueError,match='UNKNOWN'):
        pe.normalize_volume(p,pe.volume_evidence(p))


@pytest.mark.parametrize('dimension',['fold','system'])
def test_minor1_part_payload_bound_to_fold_and_system(tmp_path,dimension):
    s,p,c,m=review_inputs()
    extra=s.assign(fold=2022) if dimension=='fold' else s.assign(system='lightgbm_day_t')
    s=pd.concat([extra,s])
    root=tmp_path/'parts'
    pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2,part_dir=root)
    parts=sorted(root.glob('part-*.json'))
    parts[1].write_bytes(parts[0].read_bytes())
    with pytest.raises(ValueError,match='part.*identity|part.*fold|part.*system'):
        pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2,part_dir=root)


def test_minor3_missing_paired_primary_folds_reported(tmp_path,monkeypatch):
    s,p,c,m=review_inputs()
    s=pd.concat([s.assign(fold=y,system=sys) for y in (2022,2023) for sys in ('c0plus','lightgbm_day_t')])
    real=pe.checked_cell
    def cell(prices,predictions,config):
        if predictions.fold.iloc[0]==2022 and predictions.system.iloc[0]=='lightgbm_day_t':
            return dict(status='ERROR',config=vars(config),error=dict(type='RuntimeError',message='temporary'))
        return real(prices,predictions,config)
    monkeypatch.setattr(pe,'checked_cell',cell)
    r=pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'parts')
    era=r['paired_system_difference_descriptive']['2021_2023']
    assert era['paired_folds']==[2023]
    assert dict(fold=2022,missing_systems=['lightgbm_day_t']) in era['missing_primary_folds']
    assert dict(fold=2021,missing_systems=['c0plus','lightgbm_day_t']) in era['missing_primary_folds']
    assert era['status']=='INCOMPLETE_PAIRED_SUPPORT'


def test_minor2_retry_errors_only_and_preserve_original_parts(tmp_path,monkeypatch):
    real=pe.checked_cell
    calls=[]
    def fail(*args):
        calls.append(1)
        if len(calls)==1:
            return dict(status='ERROR',config=vars(args[2]),error=dict(type='RuntimeError',message='temporary'))
        return real(*args)
    monkeypatch.setattr(pe,'checked_cell',fail)
    root=tmp_path/'parts'
    first=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root)
    part=next(root.glob('part-*.json'))
    original=part.read_bytes()
    assert first['results'][0]['status']=='ERROR'
    calls.clear()
    def count(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(pe,'checked_cell',count)
    same=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root)
    assert same==first and not calls
    retried=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root,retry_errors=True)
    assert len(calls)==1
    assert all(x['status']=='COMPUTED' for x in retried['results'])
    assert part.read_bytes()==original
    assert len(list(root.glob('part-*.json')))==2
    calls.clear()
    assert pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root,retry_errors=True)==retried
    assert not calls
    assert any('ERROR' in x and 'transient' in x for x in pe.CAVEATS)



def test_required1_fold_slice_preserves_old_terminated_code(tmp_path):
    s,p,c,m=review_inputs()
    end=pd.Timestamp('2024-01-04')
    p=p[~(p.code.eq('A') & p.date.gt(end))]
    s=s[s.date.ge('2024-01-15') & ~s.code.eq('A')]
    r=pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2,part_dir=tmp_path/'parts')
    d=r['results'][0]['price_termination']
    assert any(x['code']=='A' and x['last_price_date']=='2024-01-04' for x in d['pre_fold_terminated']['codes'])


@pytest.mark.parametrize('mismatch',['parameters','seeds','matrix'])
def test_minor1_resume_rejects_changed_run_identity(tmp_path,mismatch):
    s,p,c,m=review_inputs()
    root=tmp_path/'parts'
    pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=2,part_dir=root)
    samples=2
    if mismatch=='parameters':
        samples=3
    elif mismatch=='seeds':
        m=dict(m,seeds=[29])
    else:
        p=p.assign(close=p.close*1.01)
    with pytest.raises(ValueError,match='part identity/hash mismatch'):
        pe.evaluate(s,p,c,m,primary_only=True,bootstrap_samples=samples,part_dir=root)


@pytest.mark.parametrize('damage',['corrupt_retry','missing_base','missing_retry'])
def test_minor2_retry_chain_fails_closed(tmp_path,monkeypatch,damage):
    import json
    root=tmp_path/'parts'
    def fail(*args):
        return dict(status='ERROR',config=vars(args[2]),error=dict(type='RuntimeError',message='transient'))
    monkeypatch.setattr(pe,'checked_cell',fail)
    pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root)
    pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root,retry_errors=True)
    pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root,retry_errors=True)
    retry=next(root.glob('*-retry-000001.json'))
    if damage=='corrupt_retry':
        saved=json.loads(retry.read_text())
        saved['previous_payload_sha256']='bad'
        retry.write_text(json.dumps(saved))
    elif damage=='missing_base':
        next(x for x in root.glob('part-*.json') if '-retry-' not in x.name).unlink()
    else:
        retry.unlink()
    with pytest.raises(ValueError,match='part identity'):
        pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root,retry_errors=True)


def test_minor2_retry_survives_kill_after_durable_publish(tmp_path,monkeypatch):
    import os
    import signal
    import subprocess
    import sys
    root=tmp_path/'parts'
    real=pe.checked_cell
    count=0
    def fail_once(*args):
        nonlocal count
        count+=1
        if count==1:
            return dict(status='ERROR',config=vars(args[2]),error=dict(type='RuntimeError',message='transient'))
        return real(*args)
    monkeypatch.setattr(pe,'checked_cell',fail_once)
    pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=root)
    base=next(root.glob('part-*.json'))
    original=base.read_bytes()
    script='''
import os, signal, sys
from pathlib import Path
from test_p1 import review_inputs
from research_lab import portfolio_eval as pe
if sys.argv[2]=='kill':
    publish=pe.publish_part
    def durable_then_kill(*args):
        publish(*args)
        os.kill(os.getpid(),signal.SIGKILL)
    pe.publish_part=durable_then_kill
else:
    def refuse_recomputation(*args):
        raise AssertionError('durable successful retry must be reused')
    pe.checked_cell=refuse_recomputation
r=pe.evaluate(*review_inputs(),primary_only=True,bootstrap_samples=2,part_dir=Path(sys.argv[1]),retry_errors=True)
assert all(x['status']=='COMPUTED' for x in r['results'])
'''
    env=dict(os.environ, PYTHONPATH='V6:V6/research_lab/tests:.')
    killed=subprocess.run([sys.executable,'-c',script,str(root),'kill'],env=env,capture_output=True,text=True,timeout=60)
    assert killed.returncode==-signal.SIGKILL, killed.stderr
    assert len(list(root.glob('part-*.json')))==2
    restarted=subprocess.run([sys.executable,'-c',script,str(root),'restart'],env=env,capture_output=True,text=True,timeout=60)
    assert restarted.returncode==0, restarted.stderr
    assert 'SKIP_HASH_MATCH' in restarted.stdout
    assert base.read_bytes()==original


def test_required1_history_metadata_isolated_from_execution(monkeypatch):
    p,s=fixture(31)
    history={'OLD':'2020-01-02T00:00:00'}
    p.attrs['prior_valid_price_dates']=history
    real=pe.run_backtest
    def engine(prices,*args):
        assert not prices.attrs, 'diagnostic history must not propagate through engine operations'
        return real(prices,*args)
    monkeypatch.setattr(pe,'run_backtest',engine)
    r=pe.evaluate_cell(p,s,cfg(bootstrap_samples=2))
    assert r['price_termination']['pre_fold_terminated']['codes'][0]['code']=='OLD'
    assert p.attrs['prior_valid_price_dates']==history
