"""P3 CPU contracts; synthetic evidence never certifies CUDA/model quality."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from research_lab import rolling_seq_scores as seq
from research_lab import portfolio_eval as pe


def scores():
    return pd.DataFrame([dict(fold=2023, system='mamba2', seed=s, date=pd.Timestamp(d), code=c,
                              score=v+s, label_y5=v)
        for s in (17,29) for d in ('2023-01-03','2023-01-04') for c,v in [('A',1.),('B',2.)]])


def test_p3_exact_p1_rows():
    a = scores().query('seed==17')
    seq.require_same_rows(a, a)
    with pytest.raises(ValueError, match='row'):
        seq.require_same_rows(a.iloc[:-1], a)
    with pytest.raises(ValueError, match='row'):
        seq.require_same_rows(pd.concat([a,a.iloc[:1]]), a)


def test_p3_rank_average_pit():
    a = scores()
    before = seq.rank_average(a)
    a.loc[a.date.eq(pd.Timestamp('2023-01-04')), 'score'] *= -1000
    after = seq.rank_average(a)
    pd.testing.assert_frame_equal(before.query("date=='2023-01-03'"), after.query("date=='2023-01-03'"))
    assert before.score.tolist() == [.5,1.,.5,1.]
    with pytest.raises(ValueError, match='support'):
        seq.rank_average(a.iloc[:-1])


def split_data():
    from research_lab.seq_runner import SeqData
    d = SeqData.__new__(SeqData)
    d.date_idx = np.repeat(np.arange(220),2)
    d.splits = np.where(d.date_idx < 180,1,np.where(d.date_idx < 200,4,2))
    d.eligible = np.ones(440,bool)
    d.y = {'y5':np.ones(440),'y10':np.ones(440)}
    d.y['y10'][0] = np.nan
    return d


def test_p3_training_purge_maturity():
    d = split_data()
    tr,va,ev = seq.checked_splits(d, val_dates=20)
    assert 0 not in tr
    assert d.date_idx[tr].max() <= 140  # inner embargo, same SeqData semantics
    assert d.date_idx[va].max()+20 < d.date_idx[ev].min()
    d.splits[d.date_idx==199] = 1
    with pytest.raises(ValueError, match='purge'):
        seq.checked_splits(d, val_dates=20)


def test_p3_common_dates():
    a = dict(daily_returns=[dict(date=str(x),net=float(x),benchmark=0.) for x in (1,2,3)])
    b = dict(daily_returns=[dict(date=str(x),net=.5,benchmark=0.) for x in (2,3,4)])
    paired, support = pe.p3_pair(a,b)
    assert [r['date'] for r in paired['daily_returns']] == ['2','3']
    assert [r['net'] for r in paired['daily_returns']] == [1.5,2.5]
    assert support['excluded_model_dates']==['1'] and support['excluded_c0plus_dates']==['4']


def test_p3_temperature_hysteresis_stop():
    readings = iter([79,80,76,73,72,85])
    sleeps = []
    g = seq.TemperatureGuard(lambda: next(readings), sleep=sleeps.append, interval=1)
    g.check(); g.check()
    assert sleeps==[1,1,1] and [x['temperature_c'] for x in g.records]==[79,80,76,73,72]
    with pytest.raises(seq.ThermalStop):
        g.check()
    with pytest.raises(seq.ThermalStop):
        g.check()  # latched; no auto restart


@pytest.mark.parametrize('bad',[float('nan'),None,-1,200])
def test_p3_temperature_unknown_fails(bad):
    g = seq.TemperatureGuard(lambda: bad)
    with pytest.raises(seq.ThermalStop):
        g.check()


def test_p3_resume_hash(tmp_path):
    frame = scores()
    p = tmp_path/'fit.parquet'
    seq.save_fit(p, frame, {'seed':17}, {'seconds':1})
    pd.testing.assert_frame_equal(seq.resume_fit(p, {'seed':17})[0], frame)
    with pytest.raises(ValueError, match='identity'):
        seq.resume_fit(p, {'seed':29})
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        seq.resume_fit(p, {'seed':17})


def test_p3_work_links_no_source_cache(tmp_path, monkeypatch):
    source=tmp_path/'external'; source.mkdir()
    work=tmp_path/'work'
    monkeypatch.setattr(seq,'WORK_ROOT',work)
    for n in seq.MATRIX_FILES:
        (source/n).write_bytes(b'fixture')
    dest=seq.prepare_work(source,2023)
    assert all((dest/n).is_symlink() for n in seq.MATRIX_FILES)
    assert not (source/'X_rankgauss.npy').exists()
    (dest/'X_rankgauss.npy').write_bytes(b'cache')
    assert not (source/'X_rankgauss.npy').exists()


def test_p3_portfolio_synthetic_path(tmp_path):
    days=pd.bdate_range('2023-01-02',periods=65)
    prices=pd.DataFrame([dict(date=d,code=c,open=100.+i,close=100.+i,volume=1e8)
                         for i,d in enumerate(days) for c in ('A','B')])
    frames=[]
    for model in ('c0plus','mamba2','crossstock'):
        frames.append(pd.DataFrame([dict(date=d,code=c,score=v,fold=2023,system=model,label_y5=v)
                                   for d in days[:-1] for c,v in [('A',1.),('B',2.)]]))
    scores=pd.concat(frames,ignore_index=True)
    result=pe.evaluate_p3(scores,prices,pd.DataFrame(),{},bootstrap_samples=5,part_dir=tmp_path/'parts')
    assert len(result['results'])==3
    assert all(len(r['daily_returns'])>20 for r in result['results'])
    assert all(r['decision']=='未證實' for r in result['hypotheses']['tests'].values())
    assert result['primary_config']['hold_buffer_rank']==100
    assert result['primary_config']['benchmark_proportional_only']
    again=pe.evaluate_p3(scores,prices,pd.DataFrame(),{},bootstrap_samples=5,part_dir=tmp_path/'parts')
    assert result==again
    changed=scores.iloc[1:]
    with pytest.raises(ValueError,match='row'):
        pe.evaluate_p3(changed,prices,pd.DataFrame(),{},bootstrap_samples=5,part_dir=tmp_path/'bad')


def test_p3_cli_dry_run_and_contract_stop(tmp_path,capsys):
    out=seq.ARTIFACT_ROOT/'test-no-write'
    args=['--folds-root',str(tmp_path/'missing'),'--out-dir',str(out)]
    seq.main(args)
    assert 'UNRESOLVED' in capsys.readouterr().out
    assert not out.exists()
    with pytest.raises(RuntimeError,match='GPU smoke'):
        seq.main(args+['--run'])
    assert not out.exists()


@pytest.mark.parametrize('model',['mamba2','crossstock'])
def test_p3_collect_cpu_injected(tmp_path,model):
    import torch
    from torch import nn
    from test_seq_runner import build, CFG
    torch.set_num_threads(1)
    source=tmp_path/'matrix'; source.mkdir()
    build(source,days=220,names=12,train_end=179,eval_start=200)
    from research_lab.seq_runner import SeqData
    data=SeqData(source,normalize=None)
    ev=np.flatnonzero((data.splits==2)&data.eligible)
    reference=pd.DataFrame(dict(fold=2023,date=pd.to_datetime(np.load(source/'dates.npy')[ev]),
                                 code=np.load(source/'stock_ids.npy')[ev],label_y5=data.y['y5'][ev]))
    cfg=dict(CFG,epochs=1,steps_per_epoch=1,dates_per_batch=2,stocks_per_date=12,min_names=10,val_dates=20,val_max_rows=240)
    guard=seq.TemperatureGuard(lambda:60)
    frame,job=seq.collect(source,2023,model,17,cfg,reference,guard,device='cpu',mixer_factory=lambda d:nn.Linear(d,d))
    assert len(frame)==240 and np.isfinite(frame.score).all()
    assert job['history'][0]['epoch']==1 and len(job['temperatures'])>=3
    assert job['gpu_model']=='CPU injected mixer'


def test_p3_holm_complete_gate():
    primaries={}
    for year in (2021,2022,2023):
        for model,value in [('c0plus',0.),('mamba2',.001),('crossstock',-.001)]:
            primaries[year,model]=dict(daily_returns=[dict(date=str(i),net=value,benchmark=0.) for i in range(40)])
    result=pe.p3_hypotheses(primaries,True,1000)
    assert result['tests']['H-P3a']['decision']=='優於 C0+'
    assert result['tests']['H-P3b']['decision']=='劣於 C0+'
    assert all(x['p_holm']<.05 for x in result['tests'].values())
    partial=pe.p3_hypotheses(primaries,False,1000)
    assert all(x['decision']=='未證實' for x in partial['tests'].values())


def test_p3_run_jobs_resume_and_input_binding(tmp_path,monkeypatch):
    import json
    import torch
    from torch import nn
    from test_seq_runner import build, CFG
    from research_lab.seq_runner import SeqData, stock_index
    torch.set_num_threads(1)
    source=tmp_path/'folds'/'fold_2023'; source.mkdir(parents=True)
    build(source,days=220,names=12,train_end=179,eval_start=200)
    stock_index(source)
    data=SeqData(source,normalize=None)
    ev=np.flatnonzero((data.splits==2)&data.eligible)
    reference=pd.DataFrame(dict(fold=2023,system='c0plus',date=pd.to_datetime(np.load(source/'dates.npy')[ev]),
        code=np.load(source/'stock_ids.npy')[ev],label_y5=data.y['y5'][ev],score=np.arange(len(ev),dtype=float)))
    p1=tmp_path/'p1'; p1.mkdir(); reference.to_parquet(p1/'rank_average.parquet',index=False)
    (p1/'manifest.json').write_text(json.dumps(dict(outputs={'rank_average.parquet':seq.identity(p1/'rank_average.parquet')})))
    root=tmp_path/'artifacts'; monkeypatch.setattr(seq,'ARTIFACT_ROOT',root); monkeypatch.setattr(seq,'WORK_ROOT',root/'work')
    cfg=dict(CFG,epochs=1,steps_per_epoch=1,dates_per_batch=2,stocks_per_date=12,min_names=10,val_dates=20,val_max_rows=240)
    kwargs=dict(folds_root=source.parent,years=[2023],models=['mamba2','crossstock'],seeds=[17],out=root/'scores',
                p1_dir=p1,cfg=cfg,smoke=True,device='cpu',mixer_factory=lambda d:nn.Linear(d,d),
                guard_factory=lambda **kw:seq.TemperatureGuard(lambda:60,**kw))
    first=seq.run_jobs(**kwargs)
    assert first['parameters']['smoke'] and len(first['jobs'])==2
    assert first['frozen_effective_config']==seq.FROZEN_CFG
    assert first['parameters']==dict(**cfg,smoke=True,eligibility_window=60,rank_method='average percentile per date then mean across seeds')
    assert {'torch','mamba_ssm','causal_conv1d'}<=first['provenance']['environment']['packages'].keys()
    assert first['mixer_parameters']['factory']=='INJECTED_CPU_ENGINEERING_ONLY'
    def unexpected(*a,**k): raise AssertionError('completed fit was retrained')
    monkeypatch.setattr(seq,'collect',unexpected)
    resumed=seq.run_jobs(**kwargs)
    assert first['outputs']==resumed['outputs']
    assert not (source/'X_rankgauss.npy').exists()
    np.save(source/'y5.npy',np.zeros(len(data.splits),np.float32))
    with pytest.raises(ValueError,match='identity'):
        seq.run_jobs(**kwargs)


def test_p3_no_best_checkpoint_rejected(tmp_path,monkeypatch):
    from torch import nn
    from test_seq_runner import build, CFG
    from research_lab import seq_runner as sr
    source=tmp_path/'matrix'; source.mkdir()
    build(source,days=220,names=12,train_end=179,eval_start=200)
    data=sr.SeqData(source,normalize=None)
    ev=np.flatnonzero((data.splits==2)&data.eligible)
    reference=pd.DataFrame(dict(fold=2023,date=pd.to_datetime(np.load(source/'dates.npy')[ev]),
        code=np.load(source/'stock_ids.npy')[ev],label_y5=data.y['y5'][ev]))
    monkeypatch.setattr(sr,'train_one',lambda *a,**kw:(None,{'best':-9.}))
    with pytest.raises(ValueError,match='best validation'):
        seq.collect(source,2023,'mamba2',17,dict(CFG,val_dates=20),reference,
                    seq.TemperatureGuard(lambda:60),device='cpu',mixer_factory=lambda d:nn.Linear(d,d))


def test_p3_frozen_effective_config(monkeypatch):
    from research_lab import seq_runner as sr
    cfg=seq.effective_config()
    assert cfg==seq.FROZEN_CFG=={**sr.DEFAULT_CFG,**seq.RECIPE}
    assert cfg['weight_decay']==.01 and cfg['val_max_rows']==80000
    cfg['mixer_kw']['expand']=99
    assert seq.effective_config()['mixer_kw']['expand']==2
    monkeypatch.setitem(sr.DEFAULT_CFG,'weight_decay',.2)
    with pytest.raises(ValueError,match='frozen'):
        seq.effective_config()
    with pytest.raises(ValueError,match='frozen'):
        seq.main(['--folds-root','unused','--out-dir',str(seq.ARTIFACT_ROOT/'unused')])


@pytest.mark.parametrize('ci,p,expected',[
    ([.01,.1],.01,'優於 C0+'),([-.1,-.01],.01,'劣於 C0+'),
    ([-.01,.1],.01,'未證實'),([.01,.1],.05,'未證實'),
    ([-.1,-.01],.05,'未證實'),([0.,.1],.01,'未證實'),([-.1,0.],.01,'未證實')])
def test_p3_amendment_decision(ci,p,expected,monkeypatch):
    from research_lab import significance
    monkeypatch.setattr(significance,'paired_ic_test',lambda *a,**k:dict(p_boot=p,ci95_boot=ci))
    monkeypatch.setattr(pe,'holm_adjust',lambda p:p)
    primaries={(y,m):dict(daily_returns=[dict(date=str(i),net=.001,benchmark=0.) for i in range(20)])
               for y in (2021,2022,2023) for m in ('c0plus','mamba2','crossstock')}
    r=pe.p3_hypotheses(primaries,True,1000)
    assert all(t['decision']==expected for t in r['tests'].values())


def test_p3_budget_subprocess_crash_restart(tmp_path):
    import subprocess,sys,json,os
    # A hard process exit bypasses finally; recovery must append an interrupted
    # debit and latch STOP, never silently reset the namespace's budget.
    code='''
import os,sys
from research_lab.rolling_seq_scores import BudgetLedger
with BudgetLedger(sys.argv[1]) as ledger:
    ledger.start({'fold':2023})
    os._exit(19)
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=os.environ.copy())
    assert r.returncode==19
    with pytest.raises(seq.ThermalStop,match='STOP'):
        with seq.BudgetLedger(tmp_path): pass
    events=[json.loads(x) for x in (tmp_path/'budget.jsonl').read_text().splitlines()]
    assert [x['event'] for x in events]==['START','END']
    assert events[-1]['status']=='INTERRUPTED' and events[-1]['seconds']>0
    stop=next(tmp_path.glob('STOP-*.json'))
    with pytest.raises(seq.ThermalStop,match='STOP'):
        with seq.BudgetLedger(tmp_path): pass
    stop.unlink()  # simulated explicit Human clearance; ledger stays intact
    code='''
import sys
from research_lab.rolling_seq_scores import BudgetLedger, ThermalStop
with BudgetLedger(sys.argv[1],limit_seconds=0.000000001) as ledger:
    print('spent_seconds=',ledger.spent,flush=True)
    ledger.start({'fold':2023})
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert r.returncode!=0 and 'budget' in r.stderr


@pytest.mark.parametrize('error',[RuntimeError('fit failure'),KeyboardInterrupt(),seq.ThermalStop('hot')])
def test_p3_budget_failed_attempts(tmp_path,error):
    import json
    with pytest.raises(type(error)):
        with seq.BudgetLedger(tmp_path) as ledger:
            with ledger.attempt({'seed':17}):
                raise error
    events=[json.loads(x) for x in (tmp_path/'budget.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='END' and events[-1]['seconds']>0
    assert list(tmp_path.glob('STOP-*.json'))
    with pytest.raises(seq.ThermalStop,match='STOP'):
        with seq.BudgetLedger(tmp_path): pass


def test_p3_official_mixer_contract(monkeypatch):
    import sys,types
    import torch
    from torch import nn
    from research_lab import seq_models,cross_stock
    calls=[]
    class Official(nn.Module):
        def __init__(self,d_model,d_state=128,d_conv=4,expand=2,headdim=64,**kw):
            super().__init__(); self.d_conv=d_conv; self.weight=nn.Parameter(torch.ones(1))
            calls.append(dict(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand,headdim=headdim))
        def forward(self,x): return x*self.weight
    monkeypatch.setitem(sys.modules,'mamba_ssm',types.SimpleNamespace(Mamba2=Official))
    model=cross_stock.CrossStockRanker(mixer_kw=seq.RECIPE['mixer_kw'])
    assert [x['d_conv'] for x in calls]==[4,4,2,2]
    params=seq.mixer_parameters(seq.effective_config())
    assert params['temporal']['d_conv']==4 and params['cross']['d_conv']==2
    assert params['temporal']['d_state']==16


def test_p3_environment_manifest_fields():
    env=seq.runtime_environment('cpu')
    assert {'torch','mamba_ssm','causal_conv1d'}<=env['packages'].keys()
    assert {'platform','system','release','python','executable','gpu_model','device'}<=env.keys()
    assert env['gpu_model'] is None


@pytest.mark.parametrize('year,sha',[
    (2021,'6c70097cb2f8bbbe27d16b5bc86cacbab506d3fabddc989a7eb28a3a290979d6'),
    (2022,'f55a74081ba43359edff6231aab488b0df9944c607de8489746d399e06e4bf89'),
    (2023,'20a8ad5b9d963130919eafa6c9b425c14bd5cd44f60c4f8c7964d169dc199eac'),
    (2024,'8e6c57aaa8a8611c31440c3e673693d40f879abd708f3615c1f8e7cb9eeec98c'),
    (2025,'281442ccb444f4a60d8b94a1a53ef5ae96a4a9a3d9b5b4662fd93ac91890727d'),
    (2026,'dfd3fa021b5d29447b67c65ecbc602e990cf92778828d151123c714fce57fa6b')])
def test_p3_real_fold_purge(year,sha):
    from pathlib import Path
    from research_lab import rolling_runner
    p=Path('/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling')/f'fold_{year}'
    if not p.exists(): pytest.skip('external read-only pilot09 fold unavailable')
    assert seq.identity(p/'splits.npy')['sha256']==sha
    data=rolling_runner.SeqData(p,normalize=None)
    tr,va,ev=seq.checked_splits(data)
    for a,b in zip((tr,va,ev),data.split_rows()): np.testing.assert_array_equal(a,b)
    def observe(directory,label,max_train_rows,seed,**kw):
        np.testing.assert_array_equal(kw['row_mask'],data.eligible)
        # rolling_runner uses split==1 plus eligibility for C0+; the neural
        # inner fit/validation are mature subsets of that exact outer support.
        outer=(data.splits==1)&kw['row_mask']
        assert outer[tr].all() and outer[va].all()
        assert data.date_idx[outer].max()+20<data.date_idx[ev].min()
        return dict(fit_rows=int(outer.sum()),eval_rows=len(ev),daily_ic={m:{} for m in rolling_runner.SYSTEMS})
    observed=rolling_runner.run_fold_seed(p,17,run_fn=observe)
    assert observed['eval_rows']==len(ev)
    data.splits=data.splits.copy()
    data.splits[np.flatnonzero(data.date_idx==data.date_idx[ev].min()-20)[0]]=1
    with pytest.raises(ValueError,match='purge'): seq.checked_splits(data)
    print(f'fold={year} splits_sha256={sha} fit={len(tr)} val={len(va)} eval={len(ev)} inner_gap=20')


def p3_cli_fixture(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    root=Path(pe.__file__).resolve().parents[2]/'.artifacts/p3'
    # Existing artifact namespace; tmp dirs are isolated and never formal Data.
    import tempfile
    root.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='revision1-e2e-',dir=root))
    days=pd.bdate_range('2021-01-04',periods=25).append(pd.bdate_range('2022-01-03',periods=25)).append(pd.bdate_range('2023-01-02',periods=25))
    rows=pd.DataFrame([dict(date=d,code=c,score=v,fold=d.year,label_y5=v) for d in days for c,v in [('A',1.),('B',2.)]])
    p1=tmp_path/'p1'; neural=tmp_path/'neural'; data=tmp_path/'data'
    for d in (p1,neural,data): d.mkdir()
    rows.assign(system='c0plus').to_parquet(p1/'rank_average.parquet',index=False)
    pd.concat([rows.assign(system=m) for m in seq.SYSTEMS],ignore_index=True).to_parquet(neural/'rank_average.parquet',index=False)
    p1_mf=dict(seeds=[17,29,43],parameters=dict(max_train_rows=500000),outputs={'rank_average.parquet':seq.identity(p1/'rank_average.parquet')})
    (p1/'manifest.json').write_text(json.dumps(p1_mf))
    neural_mf=dict(version='p3-scores-v1',seeds=[17,29],contract_status='RESOLVED',admission=dict(status='RESOLVED'),
        frozen_effective_config=seq.effective_config(),parameters=dict(seq.effective_config(),smoke=False),
        outputs={'rank_average.parquet':seq.identity(neural/'rank_average.parquet')})
    (neural/'manifest.json').write_text(json.dumps(neural_mf))
    prices=pd.DataFrame([dict(date=d,code=c,open=100.+i%25,close=100.+i%25,volume=1e8) for i,d in enumerate(days) for c in ('A','B')])
    prices.to_parquet(data/'prices_raw.parquet',index=False)
    pd.DataFrame(dict(date=days,code='A',market_value=1e9)).to_parquet(data/'market_value_raw.parquet',index=False)
    # Volume provenance has its own P1 tests. Keep all real disk readers,
    # pinned hashes, portfolio engine, parts and final JSON in this E2E.
    monkeypatch.setattr(pe,'volume_evidence',lambda p:dict(unit='shares',multiplier=1))
    monkeypatch.setattr(pe,'normalize_volume',lambda p,u:p)
    return work,p1,neural,data,neural_mf


@pytest.mark.parametrize('mutation,ready',[(None,True),('smoke',False),('unresolved',False),('seeds',False),('config',False),('admission',False)])
def test_p3_main_end_to_end(tmp_path,monkeypatch,mutation,ready):
    import json,shutil
    work,p1,neural,data,mf=p3_cli_fixture(tmp_path,monkeypatch)
    if mutation=='smoke': mf['parameters']['smoke']=True
    if mutation=='unresolved': mf['contract_status']='UNRESOLVED'
    if mutation=='seeds': mf['seeds']=[17]
    if mutation=='config': mf['parameters']['patience']=4
    if mutation=='admission': mf['admission']['status']='CPU_ENGINEERING_ONLY'
    (neural/'manifest.json').write_text(json.dumps(mf))
    # Observe the real evaluator's confirmation gate, retaining all computation.
    evaluate=pe.evaluate_p3
    observed=[]
    def record(*args,**kwargs):
        observed.append(args[3]['confirmatory_ready'])
        return evaluate(*args,**kwargs)
    monkeypatch.setattr(pe,'evaluate_p3',record)
    out=work/'report'  # suffixless: must not collide with its parts directory
    args=['--mode','p3','--scores-dir',str(neural),'--p1-scores-dir',str(p1),'--data-root',str(data),
          '--out',str(out),'--bootstrap-samples','1000','--run']
    try:
        pe.main(args)
        result=json.loads(out.read_text())
        assert observed==[ready] and len(result['results'])==9
        assert all(r['status']==('CONFIRMATORY' if ready else 'INCOMPLETE_ENGINEERING_ONLY') for r in result['hypotheses']['tests'].values())
        assert len(list((work/'report.parts').glob('part-*.json')))==9
        with pytest.raises(FileExistsError): pe.main(args)
        # Pinned reference corruption is rejected through the real CLI reader.
        (p1/'rank_average.parquet').write_bytes(b'corrupt')
        args[args.index(str(out))]=str(work/'corrupt.json')
        with pytest.raises(Exception): pe.main(args)
        assert not (work/'corrupt.json').exists()
    finally:
        shutil.rmtree(work)  # unique synthetic test fixture, reconstructible


def test_p3_main_part_dir_boundaries(tmp_path,monkeypatch):
    import shutil
    work,p1,neural,data,mf=p3_cli_fixture(tmp_path,monkeypatch)
    args=['--mode','p3','--scores-dir',str(neural),'--p1-scores-dir',str(p1),'--data-root',str(data),'--out',str(work/'out.json'),'--run','--bootstrap-samples','5']
    try:
        with pytest.raises(ValueError,match='part_dir'):
            pe.main(args+['--part-dir',str(work/'out.json')])
        with pytest.raises(ValueError,match='part_dir'):
            pe.main(args+['--part-dir',str(tmp_path/'external')])
        pe.main(args+['--part-dir',str(work/'selected-parts')])
        assert len(list((work/'selected-parts').glob('part-*.json')))==9
    finally: shutil.rmtree(work)


def test_p3_official_admission_and_collect_reachable(tmp_path,monkeypatch):
    import json
    from torch import nn
    from test_seq_runner import build
    from research_lab import seq_runner as sr
    cfg=seq.effective_config()
    source=tmp_path/'matrix'; source.mkdir()
    build(source,days=400,names=12,train_end=349,eval_start=371)
    data=sr.SeqData(source,normalize=None)
    ev=np.flatnonzero((data.splits==2)&data.eligible)
    ref=pd.DataFrame(dict(fold=2023,date=pd.to_datetime(np.load(source/'dates.npy')[ev]),
                         code=np.load(source/'stock_ids.npy')[ev],label_y5=data.y['y5'][ev]))
    env=dict(packages=dict(torch='2.10.0+cu128',mamba_ssm='2.3.2.post1',causal_conv1d='1.7.0'),gpu_model='SYNTHETIC GPU',device='cuda')
    mixer=dict(factory='mamba_ssm.Mamba2',temporal=dict(d_conv=4),cross=dict(d_conv=2))
    monkeypatch.setattr(seq,'runtime_environment',lambda device:env)
    monkeypatch.setattr(seq,'mixer_parameters',lambda cfg:mixer)
    evidence=tmp_path/'synthetic-smoke.json'
    payload=dict(version='p3-official-gpu-smoke-v1',human_reviewed=True,forward_finite=True,backward_finite=True,
                 parameters=cfg,mixer_parameters=mixer,implementation=seq.provenance()['implementation'],environment=env)
    evidence.write_text(json.dumps(payload))
    assert seq.official_admission(evidence,env,cfg)['status']=='RESOLVED'
    monkeypatch.setattr(sr,'SeqData',lambda *a,**k:data)
    reached=[]
    def fit(kind,seed,data,tr,va,cfg,device,mixer_factory,**kw):
        reached.append(mixer_factory)
        assert mixer_factory is None
        return nn.Linear(1,1),dict(best=.01,history=[dict(epoch=1)])
    monkeypatch.setattr(sr,'train_one',fit)
    monkeypatch.setattr(sr,'predict',lambda model,data,rows,device:np.zeros((len(rows),2)))
    # Exercise official adapter branch entirely on CPU with training boundary
    # stubbed. This synthetic document is never accepted as GPU evidence.
    frame,job=seq.collect(source,2023,'mamba2',17,cfg,ref,seq.TemperatureGuard(lambda:60),
                          device='cpu',gpu_smoke_evidence=evidence)
    assert reached==[None] and len(frame)==len(ev)
    payload['backward_finite']=False; evidence.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='mismatch'): seq.official_admission(evidence,env,cfg)


def test_p3_budget_lock_corrupt_and_completed(tmp_path):
    import json
    with seq.BudgetLedger(tmp_path) as ledger:
        with pytest.raises(OSError):
            with seq.BudgetLedger(tmp_path): pass
        with ledger.attempt({'seed':17}): pass
        spent=ledger.spent
        assert spent>0
    with seq.BudgetLedger(tmp_path) as ledger:
        assert ledger.spent==spent
    with (tmp_path/'budget.jsonl').open('a') as f: f.write('{truncated')
    with pytest.raises(json.JSONDecodeError):
        with seq.BudgetLedger(tmp_path): pass


def test_p3_holm_significant_ci_is_not_sufficient(monkeypatch):
    from research_lab import significance
    monkeypatch.setattr(significance,'paired_ic_test',lambda *a,**k:dict(p_boot=.03,ci95_boot=[.001,.01]))
    primaries={(y,m):dict(daily_returns=[dict(date=str(i),net=.001,benchmark=0.) for i in range(20)])
               for y in (2021,2022,2023) for m in ('c0plus','mamba2','crossstock')}
    result=pe.p3_hypotheses(primaries,True,1000)
    assert all(t['p_holm']==.06 and t['decision']=='未證實' for t in result['tests'].values())


def test_p3_budget_exhausted_refuses_even_after_stop_clearance(tmp_path,monkeypatch):
    tick=[100.]
    monkeypatch.setattr(seq.time,'monotonic',lambda:tick[0])
    with pytest.raises(seq.ThermalStop,match='budget'):
        with seq.BudgetLedger(tmp_path) as ledger:
            with ledger.attempt({'seed':17}): tick[0]+=14*3600+1
    for stop in tmp_path.glob('STOP-*.json'): stop.unlink()  # simulated Human action
    with pytest.raises(seq.ThermalStop,match='budget'):
        with seq.BudgetLedger(tmp_path): pass


def test_p3_stop_checked_before_environment_or_inputs(tmp_path,monkeypatch):
    (tmp_path/'STOP-human.json').write_text('{}')
    def unexpected(*a,**k): raise AssertionError('must stop before runtime/input access')
    monkeypatch.setattr(seq,'runtime_environment',unexpected)
    monkeypatch.setattr(seq,'pinned_reference',unexpected)
    with pytest.raises(seq.ThermalStop,match='STOP'):
        seq.run_jobs(tmp_path,[2023],['mamba2'],[17],tmp_path,tmp_path,seq.effective_config())
