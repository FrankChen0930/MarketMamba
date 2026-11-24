"""CPU-only P4b contract counterexamples; no training or external services."""
import importlib
import numpy as np
import pandas as pd
import pytest
from research_lab import p4_eval


def module():
    return importlib.import_module('research_lab.p4b_blend')


def fixture(n=60, years=(2021,), days=2):
    rng=np.random.default_rng(81)
    neural=[];tree=[];dates={}
    for y in years:
        dates[y]=list(pd.date_range(f'{y}-01-04',periods=days))
        for a in ('mamba2','crossstock','gru','c0plus'):
            for s in ((17,29,43) if a=='c0plus' else (17,29)):
                for d in dates[y]:
                    for i,x in enumerate(rng.normal(size=n)):
                        row=dict(fold=y,date=d,code=str(i).zfill(4),system=a,seed=s,window=60,lr=.0003,
                                 score=x,label_y5=np.nan if d==dates[y][-1] else i/100)
                        (tree if a=='c0plus' else neural).append(row)
    return pd.DataFrame(neural),pd.DataFrame(tree),dates


def oracle(n,c):
    # Independent pandas arithmetic; no implementation helpers.
    out={}
    for name,f in [('c0plus',c)]+list(n.groupby('system')):
        f=f.copy();f['r']=f.groupby(['fold','date','seed']).score.rank(pct=True)
        g=f.groupby(['fold','date','code'],as_index=False).r.mean().sort_values(['fold','date','code'])
        out[name]=g.groupby(['fold','date']).r.rank(pct=True).to_numpy()
    base=out['c0plus'];temp=pd.DataFrame({'date':g.date,'fold':g.fold,'v':sum(out[a] for a in ('mamba2','crossstock','gru'))/3})
    allrank=temp.groupby(['fold','date']).v.rank(pct=True).to_numpy()
    return dict(B1=(base+out['mamba2'])/2,B2=(base+allrank)/2,B3=(base+out['gru'])/2)


def test_formulas_nan_and_shared_path():
    m=module();n,c,d=fixture();frames,report=m.blend(n,c,d)
    for b,expected in oracle(n,c).items():np.testing.assert_allclose(frames[b].score,expected)
    assert report['eligible'] and len(report['daily'])==8
    assert frames['B1'].label_y5.isna().sum()==60
    shared=p4_eval.averaged_scores(n)
    for a in shared:
        expected=shared[a].groupby(['fold','date']).score.rank(pct=True)
        np.testing.assert_allclose(frames[a].score,expected)
    assert m.common_rows is p4_eval.common_rows
    assert m.averaged_scores is p4_eval.averaged_scores
    assert m.portfolio_rows is p4_eval.portfolio_rows


@pytest.mark.parametrize('bad',['seed','day','fold','all_day','extra','duplicate','inf_score','inf_label','label','nan_label'])
def test_bad_shapes_stop(bad):
    n,c,d=fixture(years=(2021,2022))
    arm=n.system.eq('gru')
    if bad=='seed':n=n[~(arm & n.seed.eq(29))]
    elif bad=='day':n=n[~(arm & n.date.eq(d[2021][1]))]
    elif bad=='fold':n=n[~(arm & n.fold.eq(2022))]
    elif bad=='all_day':
        n=n[n.date.ne(d[2021][1])];c=c[c.date.ne(d[2021][1])]
    elif bad=='extra':d[2021]=d[2021][:1]
    elif bad=='duplicate':n=pd.concat([n,n.iloc[:1]])
    elif bad=='inf_score':n.loc[0,'score']=np.inf
    elif bad=='inf_label':n.loc[0,'label_y5']=np.inf
    elif bad=='label':n.loc[0,'label_y5']=5
    elif bad=='nan_label':n.loc[0,'label_y5']=np.nan
    with pytest.raises(ValueError):module().blend(n,c,d)


def test_common_before_rank_and_daily_gates():
    n,c,d=fixture(n=100)
    # New listings: both seeds agree on reduced history; cross-arm support differs.
    n=n[~(n.system.eq('gru') & n.code.lt('0004'))]
    frames,r=module().blend(n,c,d)
    support=set(frames['B1'].code)
    expected=oracle(n[n.code.isin(support)],c[c.code.isin(support)])
    for b,x in expected.items():np.testing.assert_allclose(frames[b].score,x)
    assert r['eligible'] and r['common_rows']==192
    n=n[~(n.system.eq('gru') & n.code.eq('0004'))]
    assert module().blend(n,c,d)[1]['eligible'] # exactly 95%
    n=n[~(n.system.eq('gru') & n.code.eq('0005'))]
    assert not module().blend(n,c,d)[1]['eligible']
    n,c,d=fixture(n=49)
    assert not module().blend(n,c,d)[1]['eligible']


def test_future_invariance_ties_and_shuffle():
    n,c,d=fixture();m=module();f,_=m.blend(n,c,d)
    n.loc[n.date.eq(d[2021][-1]),'score']=9
    c.loc[c.date.eq(d[2021][-1]),'score']=-10
    g,_=m.blend(n.sample(frac=1,random_state=2),c.sample(frac=1,random_state=3),d)
    pd.testing.assert_frame_equal(f['B2'].query('date == @d[2021][0]'),g['B2'].query('date == @d[2021][0]'))
    for x in (n,c):x['score']=1
    g,_=m.blend(n,c,d)
    assert g['B1'].score.eq(30.5/60).all()
    diag=m.signal_diagnostics(g)
    assert all(x['spearman'] is None for x in diag['daily'] if x['system']=='mamba2')
    assert all(x['top50_overlap']==1 for x in diag['daily'])


def test_statistics_family_and_candidates():
    m=module();rows={}
    for y in range(2021,2027):
        for b in ('c0plus','B1','B2','B3'):
            rows[y,b]=dict(status='COMPUTED',daily_returns=[dict(date=str(i),net=0 if b=='c0plus' else .001,benchmark=0) for i in range(40)])
    result=m.inference(rows,True,True)
    assert result['m']==3 and result['priority_candidates']==['B1','B2','B3']
    assert result['tests']['B1']['registered']['n']==120
    assert result['tests']['B1']['all']['n']==240
    assert result['tests']['B1']['registered']['n_boot']==1000
    assert result['tests']['B1']['registered']['seed']==17
    del rows[2026,'B3']
    with pytest.raises(ValueError):m.inference(rows,True,True)


def test_fixed_subset_missing_stops():
    m=module();from research_lab.p4_colab import fit_queue
    import json
    from pathlib import Path
    contract=json.loads(Path('tasks/p4a-colab-contract-20261121.json').read_text())
    jobs=fit_queue(contract)
    assert len(m.subset_jobs(jobs))==36
    with pytest.raises(ValueError):m.subset_jobs(jobs[1:])
    bad=[dict(x) for x in jobs];bad[0]['window']=120
    with pytest.raises(ValueError):m.subset_jobs(bad)


def test_portfolio_degenerate_and_holdings():
    from research_lab import portfolio_eval as pe
    m=module();n,c,d=fixture(days=42)
    # identical scores across all seeds/arms imply identical engine behaviour.
    for f in (n,c):f['score']=f.code.astype(int)
    frames,support=m.blend(n,c,d)
    calendar=pd.date_range('2021-01-03',periods=44)
    prices=pd.DataFrame([dict(date=day,code=str(i).zfill(4),open=100+i,close=101+i,volume=1e7) for day in calendar for i in range(60)])
    rows=m.portfolio_rows({k:frames[k] for k in ('B1','c0plus')},prices,1000)
    assert rows[2021,'B1']['status']=='COMPUTED'
    assert rows[2021,'B1']['daily_returns']==rows[2021,'c0plus']['daily_returns']
    assert rows[2021,'B1']['config']==pe.asdict(pe.p2_primary_config())
    diagnostics=m.holding_diagnostics({k:frames[k] for k in ('B1','c0plus')},prices)
    assert diagnostics['complete']
    assert all(x['transaction_cost']>0 for x in diagnostics['risk'])
    assert all(x['jaccard']==1 for x in diagnostics['daily'] if x['jaccard'] is not None)
    assert any(x['cash_c0plus']>0 for x in diagnostics['daily'])


def test_holm_centered_two_sided_and_duplicate_dates():
    m=module();assert m.holm_adjust([.01,.04,.03])==pytest.approx([.03,.06,.06])
    x=pd.Series(np.random.default_rng(1).normal(.1,1,80));zero=x*0
    a=m.paired_ic_test(x,zero,5,n_boot=1000,block=20,seed=17)
    b=m.paired_ic_test(-x,zero,5,n_boot=1000,block=20,seed=17)
    assert a['p_boot']==b['p_boot']
    assert a==m.paired_ic_test(x,zero,5,n_boot=1000,block=20,seed=17)
    with pytest.raises(ValueError,match='duplicate'):
        m.pe.p3_pair(dict(daily_returns=[dict(date='a',net=1)]*2),dict(daily_returns=[]))


def test_in_memory_mutants(monkeypatch):
    """Execute each mutant and kill with a numerical assertion or missing STOP."""
    import inspect
    m=module();source=inspect.getsource(m.blend)
    mutants={
        'weights':(".5*f.score+.5*frames[a].score",".7*f.score+.3*frames[a].score"),
        'outer_rank':("allframe['score']=rank(allframe)","allframe['score']=allframe['score']"),
        'raw_logits':("c.groupby(['fold','date','seed']).score.rank(method='average',pct=True)","c['score']"),
        'nan_drop':("return frames,report","frames={k:v.dropna() for k,v in frames.items()};return frames,report"),
        'date_denominator':("for y,d in sorted(expected):","for y,d in sorted(set(counts)):"),
        'gate_success':("report.update(daily=daily,eligible=all(r['passed'] for r in daily)","report.update(daily=daily,eligible=True"),
    }
    for name,(old,new) in mutants.items():
        assert old in source
        scope=dict(m.__dict__);exec(compile(source.replace(old,new),'<p4b-mutant-'+name+'>','exec'),scope)
        fn=scope['blend'];killed=False
        try:
            if name=='date_denominator':
                n,c,d=fixture();n=n[n.date.ne(d[2021][1])];c=c[c.date.ne(d[2021][1])]
                try:fn(n,c,d)
                except ValueError:pass
                else:raise AssertionError('missing planned day must STOP')
            elif name=='gate_success':
                n,c,d=fixture(n=49);assert not fn(n,c,d)[1]['eligible']
            else:
                n,c,d=fixture();frames,_=fn(n,c,d)
                for b,x in oracle(n,c).items():np.testing.assert_allclose(frames[b].score,x)
                assert frames['B1'].label_y5.isna().sum()==60
        except AssertionError:killed=True
        assert killed,name+' survived'
        print('ASSERTION_KILLED',name)
    source=inspect.getsource(m.subset_jobs);scope=dict(m.__dict__)
    exec(source.replace("if actual!=expected:","if False:"),scope)
    contract=m.read_json(m.ROOT/'tasks/p4a-colab-contract-20261121.json')
    with pytest.raises(AssertionError,match='missing fit'):
        try:scope['subset_jobs'](m.fit_queue(contract)[1:])
        except ValueError:pass
        else:raise AssertionError('missing fit must STOP')
    print('ASSERTION_KILLED missing_fit')


def synthetic_export(tmp_path,monkeypatch):
    m=module();contract=m.read_json(m.ROOT/'tasks/p4a-colab-contract-20261121.json')
    binding=dict(contract=contract,environment={'fixture':'CPU'},queue=m.fit_queue(contract),speed_test=False,matrix={},bundle={})
    run_hash=m.hashed(binding);monkeypatch.setattr(m,'RUN_HASH',run_hash)
    run=tmp_path/('run-'+run_hash);run.mkdir();(run/'run_identity.json').write_text(m.json.dumps(binding))
    jobs=[];completed=[]
    for job in m.subset_jobs(binding['queue']):
        fc=dict(run_hash=run_hash,**job,recipe=contract['recipe'],rankgauss=m.RANKGAUSS,speed_test=False)
        directory=run/f"{job['queue']}-{job['year']}-{job['arm']}-{job['seed']}-w60-lr0.0003";directory.mkdir()
        f=pd.DataFrame(dict(fold=[job['year']],system=[job['arm']],seed=[job['seed']],date=[pd.Timestamp(f"{job['year']}-01-01")],code=['A'],score=[1],label_y5=[np.nan]))
        path=directory/'attempt-active';path.mkdir();f.to_parquet(path/'scores.parquet')
        entry=dict(contract=fc,outputs={'attempt-active/scores.parquet':m.digest(path/'scores.parquet')})
        (directory/'complete.json').write_text(m.json.dumps(entry));jobs.append(entry);completed.append(fc)
    manifest=dict(binding=binding,jobs=jobs);(run/'manifest.json').write_text(m.json.dumps(manifest))
    state=dict(contract=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85),completed=completed)
    event=tmp_path/'event-000001.json';event.write_text(m.json.dumps(dict(previous=None,state=state,event='END')))
    (tmp_path/'head.json').write_text(m.json.dumps(dict(sequence=1,identity=m.digest(event))))
    # Only expensive checkpoint verification is doubled; metadata/queue/ledger/readers are real.
    def validate(directory,fc):
        entry=m.read_json(directory/'complete.json')
        for name,ident in entry['outputs'].items():
            if m.digest(directory/name)!=ident:raise ValueError('fixture output hash')
        return entry
    monkeypatch.setattr(m,'resume_fit',validate)
    return m,run


@pytest.mark.parametrize('damage',['none','missing','seed','rankgauss','run_hash','ledger','head','score','speed','upgrade'])
def test_subset_read_only_and_corruption(tmp_path,monkeypatch,damage):
    m,run=synthetic_export(tmp_path,monkeypatch)
    first=next(run.glob('Q1*/complete.json'))
    if damage in ('seed','rankgauss','run_hash','speed'):
        entry=m.read_json(first);entry['contract'][{'seed':'seed','rankgauss':'rankgauss','run_hash':'run_hash','speed':'speed_test'}[damage]]=None
        first.write_text(m.json.dumps(entry))
    elif damage=='missing':first.unlink()
    elif damage=='ledger':(tmp_path/'event-000001.json').write_text('{}')
    elif damage=='head':(tmp_path/'head.json').write_text('{}')
    elif damage=='score':(first.parent/'attempt-active/scores.parquet').write_bytes(b'corrupt')
    elif damage=='upgrade':(run/'source-upgrade-unrecorded.json').write_text('{}')
    before={str(p):m.digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    if damage=='none':
        f,identity=m.load_subset(run,retain_years=(2021,));assert len(f)==6 and len(identity['fits'])==36
    else:
        with pytest.raises((ValueError,KeyError,FileNotFoundError)):m.load_subset(run,retain_years=(2021,))
    assert before=={str(p):m.digest(p) for p in tmp_path.rglob('*') if p.is_file()}


def test_mutation_common_rows_and_holm(monkeypatch):
    """Two more semantic mutations, killed by concrete assertions."""
    import inspect
    m=module();n,c,d=fixture(n=100)
    n=n[~(n.system.eq('gru') & n.code.lt('0004'))]
    source=inspect.getsource(m.blend)
    old="frames=averaged_scores(pd.concat([restricted[a] for a in ARMS],ignore_index=True))"
    new="frames={k:v[pd.MultiIndex.from_frame(v[KEYS]).isin(support)].reset_index(drop=True) for k,v in averaged_scores(neural).items()}"
    assert old in source;scope=dict(m.__dict__);exec(source.replace(old,new),scope)
    bad,_=scope['blend'](n,c,d);good,_=m.blend(n,c,d)
    with pytest.raises(AssertionError):np.testing.assert_allclose(bad['B2'].score,good['B2'].score)
    print('ASSERTION_KILLED rank_before_intersection')
    rows={}
    rng=np.random.default_rng(12)
    for y in m.YEARS:
        noise=rng.normal(.00004,.001,80)
        for b in ('c0plus',)+m.BLENDS:
            rows[y,b]=dict(status='COMPUTED',daily_returns=[dict(date=str(i),net=0 if b=='c0plus' else float(v),benchmark=0) for i,v in enumerate(noise)])
    good=m.inference(rows,True,True)
    monkeypatch.setattr(m,'holm_adjust',lambda ps:ps)
    bad=m.inference(rows,True,True)
    with pytest.raises(AssertionError):assert bad['tests']['B1']['p_holm']==good['tests']['B1']['p_holm']
    print('ASSERTION_KILLED ignore_holm')


def test_ic_mature_mask_and_unknown_tail():
    m=module();n,c,d=fixture();frames,_=m.blend(n,c,d)
    result=m.signal_diagnostics(frames)
    for row in result['daily']:
        if pd.Timestamp(row['date'])==d[2021][-1]:
            assert row['ic_used']==0 and row['ic'] is None
        else:
            f=frames[row['system']];g=f[f.date.eq(d[2021][0])]
            assert row['ic']==pytest.approx(g.score.corr(g.label_y5,method='spearman'))
