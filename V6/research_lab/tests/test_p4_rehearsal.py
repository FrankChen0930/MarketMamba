"""Tail-label regression and full CPU rehearsal with a mixer boundary double."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_lab import p4_colab as p, p4_eval as e, seq_runner as sr
from test_seq_runner import build, CausalConvMixer, CFG


def matrix(root):
    root.mkdir()
    _,day,_=build(root,days=390,names=20,train_end=330,eval_start=355,gap=(0,0,210))
    for label,horizon in [('y5',5),('y10',10)]:
        a=np.load(root/f'{label}.npy');a[day>=390-horizon]=np.nan;np.save(root/f'{label}.npy',a)
    data=sr.SeqData(root,window=60);_,_,rows=p.checked_splits(data,15)
    ref=pd.DataFrame(dict(fold=2021,date=pd.to_datetime(np.load(root/'dates.npy')[rows]),code=np.load(root/'stock_ids.npy')[rows],label_y5=data.y['y5'][rows]))
    return ref


@pytest.mark.parametrize('window',[120,250])
def test_real_seq_support_tail_nan_and_listing(tmp_path,window):
    ref=matrix(tmp_path/'matrix');data=sr.SeqData(tmp_path/'matrix',window=window)
    _,_,rows=p.checked_splits(data,15)
    model=pd.DataFrame(dict(fold=2021,date=pd.to_datetime(np.load(tmp_path/'matrix/dates.npy')[rows]),code=np.load(tmp_path/'matrix/stock_ids.npy')[rows],label_y5=data.y['y5'][rows]))
    aligned,report=e.common_rows({'model':model,'c0plus':ref})
    assert aligned['model'].label_y5.isna().sum()==5*len(model.code.unique())
    assert len(model)<=len(ref) and report['common_rows']==len(model)
    if window==250:assert report['systems']['c0plus']['excluded_rows']==35


def test_common_nan_identity_and_infinity():
    a=pd.DataFrame(dict(fold=2021,date=pd.to_datetime(['2021-01-01','2021-01-02']),code='A',label_y5=[1.,np.nan]))
    aligned,_=e.common_rows({'a':a,'b':a.iloc[::-1]});assert aligned['a'].label_y5.isna().sum()==1
    for value in (np.inf,-np.inf,2.):
        bad=a.copy();bad.loc[1,'label_y5']=value
        with pytest.raises(ValueError,match='label'):e.common_rows({'a':a,'b':bad})


def test_permutation_nan_uses_only_mature_support():
    dates=np.repeat(np.arange(3),12);codes=np.tile(np.arange(12),3);y=codes.astype(float);y[-12:]=np.nan
    out=e.permutation_summary(dates,codes,y,codes,-codes,[codes.copy() for _ in range(10)])
    assert len(out['daily'])==20 and out['label_support']==dict(input_rows=36,used_rows=24,excluded_nan_rows=12,excluded_dates=['2'])
    bad=y.copy();bad[0]=np.inf
    with pytest.raises(ValueError,match='nonfinite'):e.permutation_summary(dates,codes,bad,codes,-codes,[codes]*10)
    with pytest.raises(ValueError,match='support'):e.permutation_summary(dates,codes,np.full(36,np.nan),codes,-codes,[codes]*10)


@pytest.mark.parametrize('queue,window,arm',[('Q1',60,'gru'),('Q3',120,'gru'),('Q3',250,'mamba2'),('Q4',60,'gru'),('Q5',60,'crossstock')])
def test_rehearsal_full_cpu(tmp_path,monkeypatch,queue,window,arm):
    ref=matrix(tmp_path/'matrix')
    if queue=='Q4':ref['fold']=2022
    ref.to_parquet(tmp_path/'ref.parquet',index=False)
    before={str(f):p.digest(f) for f in (tmp_path/'matrix').iterdir() if f.is_file()}
    monkeypatch.setattr(p,'telemetry',lambda:dict(gpu='CPU_TEST',temperature_c=40,clock_mhz=0,power_w=0))
    cfg=dict(CFG,n_cross=1,dropout=.1,epochs=1,steps_per_epoch=1,d_model=8,dates_per_batch=1,stocks_per_date=12,min_names=10,val_max_rows=0)
    result=p.rehearse(tmp_path/'matrix',tmp_path/'ref.parquet',queue=queue,window=window,arm=arm,steps=1,device='cpu',mixer_factory=CausalConvMixer,recipe=cfg)
    assert result['evidence_class']=='REHEARSAL / 不是研究結果'
    out=Path(result['output']);assert out.is_relative_to(Path('/tmp'))
    assert len(result['fits'])==(2 if queue=='Q5' else 1)
    for fit in result['fits']:
        path=out/fit/'REHEARSAL-complete.json';saved=json.loads(path.read_text())
        assert saved['evidence_class']==result['evidence_class']
        assert {'attempt-active/scores.parquet','attempt-active/reference.parquet','attempt-active/best.pt'}<=saved['outputs'].keys()
        assert all(p.digest(out/fit/name)==ident for name,ident in saved['outputs'].items())
    if queue=='Q5':
        assert result['diagnostic']['evaluation']['label_support']['excluded_nan_rows']==100
        for split in result['diagnostic'].values():
            p.validate_rehearsal_schema(split,p.rehearsal_schema('q5'))
        dates=set(np.load(tmp_path/'matrix/dates.npy').astype(str))
        assert all(row['date'] in dates for split in result['diagnostic'].values() for row in split['daily'])
    assert before=={str(f):p.digest(f) for f in (tmp_path/'matrix').iterdir() if f.is_file()}
    import shutil
    shutil.rmtree(out)


def test_family_c_tail_nan_and_distinct_listing_support(monkeypatch):
    from research_lab import p4_bundle as b
    contract=json.loads((b.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    rows=[];base=[]
    for year in (2021,2022,2023):
        for day in range(2):
            for i,code in enumerate(['old','ipo120','ipo60']):
                key=dict(fold=year,date=pd.Timestamp(f'{year}-02-01')+pd.Timedelta(days=day),code=code,label_y5=np.nan if day else float(i))
                base.append(dict(key,system='c0plus',score=float(i)))
                for arm in ('mamba2','crossstock','gru'):
                    for w in ((60,) if arm=='crossstock' else (60,120,250)):
                        if w==120 and code=='ipo60' or w==250 and code!='old':continue
                        for seed in (17,29):rows.append(dict(key,system=arm,score=-float(i),seed=seed,window=w,lr=3e-4))
    supports=[]
    def portfolios(frames,*_):
        supports.append(frames)
        return {}
    monkeypatch.setattr(e,'portfolio_rows',portfolios)
    out=e.evaluate(pd.DataFrame(rows),pd.DataFrame(base),pd.DataFrame(),contract)
    report=out['intersections']['C'];assert report['common_rows']==6
    assert report['systems']['mamba2']['excluded_rows']==12
    assert report['systems']['mamba2_w120']['excluded_rows']==6
    assert report['systems']['mamba2_w250']['excluded_rows']==0
    assert all(set(f.code)=={'old'} and f.label_y5.isna().sum()==3 for f in supports[2].values())
    assert all(f.score.eq(1.).all() for name,f in supports[2].items() if name!='c0plus')


def test_completed_q1_q2_paths_never_call_common_rows(tmp_path,monkeypatch):
    from test_p4_source_compat import old_module
    from research_lab import p4_bundle as b
    import torch
    old=old_module(tmp_path,'p4_colab');ref=matrix(tmp_path/'matrix')
    frozen=json.loads((b.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    calls=[];trains=[];frames=[]
    monkeypatch.setattr(p,'long_window_data',lambda *a,**k:pytest.fail('W60 entered new preflight'))
    monkeypatch.setattr(e,'common_rows',lambda *a,**k:calls.append(a) or pytest.fail('W60 entered changed common_rows'))
    model=torch.nn.Linear(1,1)
    def train(*a,**k):
        trains.append(a[0]);return model,dict(best=.2,history=[dict(epoch=1,val_ic5=.2)])
    monkeypatch.setattr(sr,'train_one',train)
    monkeypatch.setattr(sr,'predict',lambda model,data,ev,device:np.ones((len(ev),2)))
    class Observer:
        speed=False;device='cpu';records=[];contract={}
        def check(self):pass
        def load_resume(self):return None
    for mod in (old,p):
        for i,job in enumerate(p.fit_queue(frozen)[:36]):
            obs=Observer();obs.root=tmp_path/f'{mod.__name__}-{i}';obs.root.mkdir()
            cfg=dict(frozen['recipe'],window=60,val_dates=15,val_max_rows=0)
            frame,_=mod.train_fit(tmp_path/'matrix',job['year'],job['arm'],job['seed'],cfg,ref.assign(fold=job['year']),obs)
            assert frame.label_y5.isna().any()
            frames.append(frame)
    for old_frame,new_frame in zip(frames[:36],frames[36:],strict=True):
        pd.testing.assert_frame_equal(old_frame,new_frame,check_exact=True)
        assert old_frame.score.to_numpy().tobytes()==new_frame.score.to_numpy().tobytes()
    assert calls==[] and len(trains)==72


def test_portfolio_outcomes_ignore_identity_labels():
    from test_portfolio import fixture,cfg
    from research_lab import portfolio_eval as pe
    prices,pred=fixture(45)
    pred['label_y5']=np.arange(len(pred),dtype=float)
    finite=pe.checked_cell(prices,pred,cfg(bootstrap_samples=2))
    pred['label_y5']=np.nan
    immature=pe.checked_cell(prices,pred,cfg(bootstrap_samples=2))
    assert finite['status']=='COMPUTED' and finite==immature


def test_rehearsal_cli_defaults_and_no_drive(monkeypatch,capsys,tmp_path):
    seen=[]
    monkeypatch.setattr(p,'rehearse',lambda *a,**kw:seen.append((a,kw)) or dict(evidence_class='REHEARSAL / 不是研究結果'))
    monkeypatch.setattr(p,'run',lambda *a,**k:pytest.fail('rehearsal touched formal run'))
    p.main(['--rehearse','--data-dir',str(tmp_path),'--reference',str(tmp_path/'keys.parquet'),'--queue','Q3','--window','250','--arm','gru'])
    assert seen[0][1]==dict(queue='Q3',window=250,arm='gru',steps=1)
    assert 'REHEARSAL' in capsys.readouterr().out


@pytest.mark.parametrize('queue,window,arm,steps',[('Q1',250,'gru',1),('Q5',120,None,1),('Q3',120,'crossstock',1),('Q4',60,'gru',0)])
def test_invalid_rehearsal_selection_fail_closed(tmp_path,queue,window,arm,steps):
    with pytest.raises(ValueError,match='rehearsal|Q5'):
        p.rehearse(tmp_path,tmp_path/'missing.parquet',queue=queue,window=window,arm=arm,steps=steps,device='cpu')
