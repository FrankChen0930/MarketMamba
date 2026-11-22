"""Offline CPU progress regression evidence; no Colab acceptance."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from research_lab import p4_bundle as b, p4_colab as p, seq_runner as sr


def stream_helper():
    cells=b.colab_notebook()['cells']
    cell=next(c['source'] for c in cells if 'def stream_training' in c['source'])
    tree=ast.parse(cell)
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    ns=dict(subprocess=subprocess,os=os)
    exec(compile(tree,'stream helper','exec'),ns)
    return ns['stream_training']


def test_stream_large_output_and_nonzero(capsys):
    stream=stream_helper()
    stream([sys.executable,'-u','-c',"import sys; [print('x'*1024) for _ in range(2048)]; print('ERR',file=sys.stderr)"],os.environ.copy())
    output=capsys.readouterr().out
    assert output.count('x'*1024)==2048 and 'ERR' in output
    with pytest.raises(subprocess.CalledProcessError):
        stream([sys.executable,'-c','raise SystemExit(7)'],os.environ.copy())
    output=capsys.readouterr().out
    assert all(s in output for s in ('訓練','期望','實際','7','下一步'))


def test_progress_eta_atomic_and_failure(tmp_path,monkeypatch,capsys):
    clock=[0.]
    progress=p.Progress(tmp_path,speed=False,clock=lambda:clock[0])
    progress.start_fit(dict(queue='Q1',arm='gru',year=2021,seed=17,window=60,lr=.0003),1,4,dict(epochs=2,steps_per_epoch=100))
    assert progress.snapshot()['queue_eta_seconds'] is None
    progress.update(epoch=1,step=50,loss=.25,val_ic5=.1)
    progress.write()
    data=json.loads((tmp_path/'progress.json').read_text())
    assert data['loss']==.25 and '預估中' in (tmp_path/'PROGRESS.txt').read_text()
    assert not list(tmp_path.glob('*.tmp'))
    clock[0]=20
    progress.finish_fit()
    progress.start_fit(dict(queue='Q2',arm='gru'),2,4,dict(epochs=2,steps_per_epoch=100))
    assert progress.snapshot()['queue_eta_seconds']==60
    previous=(tmp_path/'progress.json').read_bytes()
    monkeypatch.setattr(p.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('drive unavailable')))
    progress.write()  # failure is observational only; old complete document remains
    assert (tmp_path/'progress.json').read_bytes()==previous
    assert 'drive unavailable' in capsys.readouterr().out


def test_speed_redaction_and_wallclock_heartbeat(tmp_path,capsys):
    import time
    with p.Progress(tmp_path,speed=True,interval=.02) as progress:
        progress.update(val_ic5=.99,loss=.4)
        time.sleep(.09)  # no training steps: heartbeat still writes
        assert (tmp_path/'progress.json').exists()
        content=(tmp_path/'progress.json').read_text()+(tmp_path/'PROGRESS.txt').read_text()
        assert 'val_ic' not in content and '.99' not in content and 'IC' not in content
    assert not progress.thread.is_alive()


def test_speed_then_formal_ledgers_and_missing_head(tmp_path):
    c=dict(policy='record_only',units_per_hour=0)
    speed=p.budget_root(tmp_path/'budget',True)
    formal=p.budget_root(tmp_path/'budget',False)
    assert speed==tmp_path/'budget_speedtest' and formal==tmp_path/'budget'
    for root,contract in ((speed,dict(c,mode='speed')),(formal,dict(c,mode='formal'))):
        with p.Ledger(root,contract) as ledger:
            ledger.start({'fit':1},0);ledger.end({'fit':1},True)
        with p.Ledger(root,contract) as ledger:assert len(ledger.state['completed'])==1
        (root/'head.json').unlink()
        with pytest.raises(ValueError,match='head'):
            with p.Ledger(root,contract):pass


def test_progress_weights_predictions_bitwise(tmp_path,monkeypatch):
    from test_seq_runner import build,CFG
    torch.set_num_threads(1);(tmp_path/'matrix').mkdir();build(tmp_path/'matrix')
    data=sr.SeqData(tmp_path/'matrix');tr,va,ev=data.split_rows(15)
    cfg=dict(CFG,epochs=2,steps_per_epoch=51,d_model=8,n_layers=1)
    plain,_=sr.train_one('gru',17,data,tr,va,cfg,'cpu',log=lambda *_:None)
    rng_before=torch.get_rng_state().clone()
    class Ledger:
        def check(self):pass
    monkeypatch.setattr(p,'telemetry',lambda:dict(gpu='CPU',temperature_c=40,clock_mhz=0,power_w=0))
    with p.Progress(tmp_path/'progress',interval=.02) as progress:
        progress.start_fit(dict(queue='Q1',arm='gru',year=2021,seed=17,window=60,lr=cfg['lr']),1,1,cfg)
        observer=p.Observer(tmp_path/'ckpt',dict(cfg=cfg),Ledger(),device='cpu',progress=progress)
        observed,_=sr.train_one('gru',17,data,tr,va,cfg,'cpu',log=lambda *_:None,observer=observer)
    assert torch.equal(rng_before,torch.get_rng_state())
    assert all(v.detach().numpy().tobytes()==observed.state_dict()[k].detach().numpy().tobytes() for k,v in plain.state_dict().items())
    assert sr.predict(plain,data,ev,'cpu').tobytes()==sr.predict(observed,data,ev,'cpu').tobytes()


def test_generated_notebook_progress_compile():
    nb=b.colab_notebook()
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'generated notebook','exec')
    assert any('目前進度摘要' in c['source'] for c in nb['cells'])
    source='\n'.join(c['source'] for c in nb['cells'])
    assert "PYTHONUNBUFFERED='1'" in source and "'-u'" in source


def test_stream_is_live_before_child_exit(tmp_path):
    helper=stream_helper()
    marker=tmp_path/'ack'
    printed=[]
    def output(line,**kwargs):
        assert kwargs.get('flush') is True
        printed.append(line)
        if line.strip()=='READY':marker.write_text('ack')
    helper.__globals__['print']=output
    script="""import pathlib,sys,time
p=pathlib.Path(sys.argv[1])
print('READY',flush=True)
end=time.monotonic()+5
while not p.exists() and time.monotonic()<end:time.sleep(.01)
assert p.exists(), 'parent buffered until exit'
print('DONE',flush=True)
"""
    helper([sys.executable,'-u','-c',script,str(marker)],os.environ.copy())
    assert printed==['READY\n','DONE\n']


def test_atomic_replace_keeps_readers_on_complete_document(tmp_path,monkeypatch):
    progress=p.Progress(tmp_path)
    progress.update(loss=1.);progress.write()
    real=p.os.replace;observations=[]
    def replace(source,target):
        if Path(target).name=='progress.json':
            observations.append(json.loads(Path(target).read_text())['loss'])
            assert json.loads(Path(source).read_text())['loss']==2.
        real(source,target)
    monkeypatch.setattr(p.os,'replace',replace)
    progress.update(loss=2.);progress.write()
    assert observations==[1.] and json.loads((tmp_path/'progress.json').read_text())['loss']==2.


def test_progress_frequency_and_speed_training_output(tmp_path,monkeypatch,capsys):
    from test_seq_runner import build,CFG
    torch.set_num_threads(1);build(tmp_path)
    data=sr.SeqData(tmp_path);tr,va,_=data.split_rows(15)
    cfg=dict(CFG,epochs=2,steps_per_epoch=51,d_model=8,n_layers=1)
    class Ledger:
        def check(self):pass
    monkeypatch.setattr(p,'telemetry',lambda:dict(gpu='CPU',temperature_c=40,clock_mhz=0,power_w=0))
    progress=p.Progress(tmp_path/'progress',speed=True)
    progress.start_fit(dict(queue='Q1',arm='gru'),1,3,cfg)
    observer=p.Observer(tmp_path/'ckpt',dict(cfg=cfg),Ledger(),device='cpu',speed=True,progress=progress)
    sr.train_one('gru',17,data,tr,va,cfg,'cpu',log=lambda *_:None,observer=observer)
    output=capsys.readouterr().out
    assert output.count('累計 step:')==5  # fit start + two step-50 lines + two epochs
    assert 'IC' not in output
    assert not (tmp_path/'ckpt').exists()


def test_in_memory_mutations_killed(tmp_path,monkeypatch):
    import inspect
    source=inspect.getsource(p.Progress)
    def mutant(old,new):
        assert old in source
        ns=dict(p.__dict__);exec(compile(source.replace(old,new),'in-memory Progress','exec'),ns)
        return ns['Progress']
    # 1. Remove speed redaction: the invariant must reject leaked IC.
    cls=mutant("if not self.speed:allowed.add('val_ic5')","if True:allowed.add('val_ic5')")
    changed=cls(tmp_path,speed=True);changed.update(val_ic5=.123)
    with pytest.raises(AssertionError):assert 'val_ic5' not in changed.snapshot()
    # 2. Fabricate an ETA before any completed fit.
    cls=mutant('queue_eta=None','queue_eta=0.')
    with pytest.raises(AssertionError):assert cls(tmp_path).snapshot()['queue_eta_seconds'] is None
    # 3. Turn observational Drive failure into a fatal training exception.
    cls=mutant('except Exception as error:', 'except Exception as error:\n            raise')
    with monkeypatch.context() as failure:
        failure.setattr(p.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('mutant Drive')))
        with pytest.raises(OSError,match='mutant Drive'):cls(tmp_path).write()
    # 4. Shared path regression reproduces ledger identity/chain mismatch.
    contract=dict(policy='record_only',units_per_hour=0)
    with p.Ledger(tmp_path/'shared',dict(contract,mode='speed')) as ledger:ledger.start({'fit':1},0);ledger.end({'fit':1},True)
    with pytest.raises(ValueError,match='identity/chain mismatch'):
        with p.Ledger(tmp_path/'shared',dict(contract,mode='formal')):pass


def resumed_progress(root, clock):
    """Load an identity/hash-validated synthetic epoch-39 checkpoint."""
    cfg=dict(epochs=50,steps_per_epoch=100)
    job=dict(queue='Q3',arm='gru',year=2024,seed=29,window=60,lr=.0003)
    progress=p.Progress(root/'progress',clock=lambda:clock[0])
    progress.start_fit(job,40,64,cfg)
    checkpoint=root/'checkpoint';checkpoint.mkdir()
    contract=dict(recipe=cfg)
    payload=dict(contract=contract,state=dict(epoch=39,best=.2),best_sd={},
                 model={},opt={},sched={},rng_state={},numpy_rng=None,torch_rng=None,cuda_rng=None)
    torch.save(payload,checkpoint/'epoch.pt')
    entry=dict(file='epoch.pt',identity=b.digest(checkpoint/'epoch.pt'),epoch=39)
    (checkpoint/'checkpoint-pointer.json').write_text(json.dumps(dict(latest=entry,previous=None,best=entry)))
    p.Observer(checkpoint,contract,None,device='cpu',progress=progress).load_resume()
    return progress


def test_resumed_eta_uses_only_this_launch_steps(tmp_path):
    clock=[0.];progress=resumed_progress(tmp_path,clock)
    clock[0]=10.;progress.update(epoch=40,step=4000)
    data=progress.snapshot()
    assert data['fit_eta_seconds']==pytest.approx(100.)  # 100 new steps / 10s; 1000 remain
    assert data['queue_eta_seconds'] is None  # no completed measured fits


@pytest.mark.parametrize('steps,seconds',[(0,10.),(9,10.),(100,.5)])
def test_resumed_eta_requires_reliable_sample(tmp_path,steps,seconds):
    clock=[0.];progress=resumed_progress(tmp_path,clock)
    clock[0]=seconds;progress.update(step=3900+steps)
    assert progress.snapshot()['fit_eta_seconds'] is None
    assert '預估中' in progress.render(progress.snapshot())


def test_completed_duration_is_frozen_and_skips_are_not_samples(tmp_path,capsys):
    clock=[0.];progress=p.Progress(tmp_path,clock=lambda:clock[0])
    cfg=dict(epochs=50,steps_per_epoch=100)
    job=dict(queue='Q1',arm='gru',year=2021,seed=17,window=60,lr=.0003)
    progress.start_fit(job,1,4,cfg);progress.update(step=100)
    clock[0]=20.;progress.finish_fit()
    clock[0]=1000.  # time between fits must never inflate the stored duration
    progress.start_fit(dict(job,queue='Q2',arm='mamba2'),2,4,cfg)
    clock[0]=1005.
    progress.finish_fit(skipped=True)
    assert progress.durations==[20.]
    progress.start_fit(dict(job,queue='Q3'),3,4,cfg)
    clock[0]=1015.;progress.update(step=1000)
    data=progress.snapshot()
    assert data['fit_eta_seconds']==40.
    assert data['queue_eta_seconds']==60.  # current 40 + future measured mean 20
    progress.finish_fit()
    progress.start_fit(dict(job,queue='Q4',year=2026),4,4,cfg)
    progress.finish_fit(skipped=True)
    data=json.loads((tmp_path/'progress.json').read_text())
    assert data['queue_eta_seconds']==data['fit_eta_seconds']==0.
    assert data['status']=='已完成（略過）'
    assert '已完成（略過）' in capsys.readouterr().out


def test_interrupted_fit_resume_progress_is_continuous(tmp_path,monkeypatch):
    from test_seq_runner import build,CFG
    torch.set_num_threads(1);build(tmp_path)
    data=sr.SeqData(tmp_path);tr,va,_=data.split_rows(15)
    cfg=dict(CFG,epochs=3,steps_per_epoch=12,d_model=8,n_layers=1,patience=9)
    job=dict(queue='Q2',arm='gru',year=2023,seed=29,window=60,lr=cfg['lr'])
    class Ledger:
        def check(self):pass
    monkeypatch.setattr(p,'telemetry',lambda:dict(gpu='CPU',temperature_c=40,clock_mhz=0,power_w=0))
    clock=[0.];progress=p.Progress(tmp_path/'progress',clock=lambda:clock[0])
    progress.start_fit(job,18,64,cfg)
    contract=dict(recipe=cfg)
    observer=p.Observer(tmp_path/'ckpt',contract,Ledger(),device='cpu',progress=progress)
    steps=[0]
    def interrupt(event,payload):
        observer(event,payload)
        if event=='step_end':
            steps[0]+=1
            if steps[0]==17:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        sr.train_one('gru',29,data,tr,va,cfg,'cpu',observer=interrupt,log=lambda *_:None)
    # A fresh launch rolls back the five uncommitted steps to epoch 1.
    clock[0]=100.;resumed=p.Progress(tmp_path/'progress',clock=lambda:clock[0])
    resumed.start_fit(job,18,64,cfg)
    observer=p.Observer(tmp_path/'ckpt',contract,Ledger(),device='cpu',progress=resumed)
    state=observer.load_resume();resumed.write()
    saved=json.loads((tmp_path/'progress/progress.json').read_text())
    assert saved['epoch']==1 and saved['step']==12
    assert saved['fit_index']==18 and all(saved[k]==v for k,v in job.items())
    assert saved['fit_eta_seconds'] is None
    snapshots=[]
    def observe(event,payload):
        if event=='step_end':clock[0]+=.1
        observer(event,payload)
        if event=='progress':snapshots.append(resumed.snapshot())
    sr.train_one('gru',29,data,tr,va,cfg,'cpu',observer=observe,resume_state=state,log=lambda *_:None)
    assert [s['step'] for s in snapshots]==[24,36]
    assert snapshots[0]['fit_eta_seconds']==pytest.approx(1.2)
    assert all(s['fit_index']==18 and s['seed']==29 for s in snapshots)
    resumed.finish_fit()
    assert resumed.snapshot()['fit_eta_seconds']==0.


def test_last_trained_fit_reports_completed_zero_eta(tmp_path,capsys):
    clock=[0.];progress=p.Progress(tmp_path,clock=lambda:clock[0])
    progress.start_fit(dict(queue='Q4',arm='gru'),64,64,dict(epochs=2,steps_per_epoch=100))
    progress.completed=63;clock[0]=10.;progress.update(step=200);progress.finish_fit()
    saved=json.loads((tmp_path/'progress.json').read_text())
    assert saved['completed_fits']==64 and saved['fit_index']==64
    assert saved['queue_eta_seconds']==saved['fit_eta_seconds']==0.
    assert saved['status']=='已完成'
    assert '狀態: 已完成' in capsys.readouterr().out


def test_cached_final_snapshot_preserves_skip_status(tmp_path):
    with p.Progress(tmp_path) as progress:
        progress.start_fit(dict(queue='Q4',arm='gru',year=2026,seed=29,window=60,lr=.0003),64,64,
                           dict(epochs=50,steps_per_epoch=100))
        progress.completed=63;progress.finish_fit(skipped=True)
    saved=json.loads((tmp_path/'progress.json').read_text())
    assert saved['status']=='已完成（略過）'
    assert saved['fit_index']==saved['completed_fits']==64
    assert saved['queue_eta_seconds']==0.
