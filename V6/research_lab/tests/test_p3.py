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
    assert 'STOP_MODEL_CONTRACT_AMBIGUITY' in capsys.readouterr().out
    assert not out.exists()
    with pytest.raises(RuntimeError,match='no-conv'):
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
