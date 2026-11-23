"""Offline gate control-flow tests; no CUDA or official-mixer claim."""
import inspect
import json
from pathlib import Path
import subprocess
import pytest
from research_lab import p4_colab as p, p4_bundle as b
from test_seq_runner import build


def launch(tmp_path, fail=None, identity=None, enabled=True, implementation=None, seen=None, run="budget"):
    if seen is None:seen=[]
    def stream(cmd,env):
        seen.append(cmd)
        if fail and len(seen)==fail:raise subprocess.CalledProcessError(7,cmd)
    common=['python','-m','research_lab.p4_colab','--bundle','bundle','--data-dir','local','--drive-root',run]
    (implementation or p.launch_with_rehearsal)(common,{},tmp_path,identity or {'source':'a','data':'b','environment':'c'},stream,enabled=enabled)
    return seen


def test_gate_success_cache_and_identity_invalidation(tmp_path):
    calls=launch(tmp_path)
    assert len(calls)==4 and '--rehearse' not in calls[-1]
    assert [c[c.index('--queue')+1] for c in calls[:-1]]==['Q3','Q3','Q5']
    assert [c[c.index('--window')+1] for c in calls[:2]]==['120','250']
    for c in calls[:-1]:
        assert c[c.index('--steps')+1]=='1' and '--drive-root' not in c
        assert c[c.index('--data-dir')+1]=='local' and '--bundle' in c
    for c in calls[:2]:assert c[c.index('--arm')+1]=='mamba2'
    assert len(launch(tmp_path))==1
    for key in ('source','data','environment'):
        changed={'source':'a','data':'b','environment':'c'};changed[key]='changed'
        assert len(launch(tmp_path,identity=changed))==4
    marker=tmp_path/'rehearsal_passed.json';marker.write_text('{corrupt')
    assert len(launch(tmp_path))==4
    marker.write_text('null');assert len(launch(tmp_path))==4
    assert len(launch(tmp_path,run='another-run'))==4
    marker.unlink();assert len(launch(tmp_path,enabled=False))==1 and not marker.exists()


@pytest.mark.parametrize('failure',[1,2,3])
def test_failure_never_launches_formal(tmp_path,failure,capsys):
    seen=[]
    with pytest.raises(subprocess.CalledProcessError):launch(tmp_path,fail=failure,seen=seen)
    assert len(seen)==failure and all('--rehearse' in c for c in seen)
    assert not (tmp_path/'rehearsal_passed.json').exists()
    output=capsys.readouterr().out
    for required in ('階段','/tmp','期望','實際','下一步','Claude'):assert required in output


@pytest.mark.parametrize('days,train_end,window,val_dates,actual,needed',[(100,70,250,15,100,250),(270,260,250,15,12,35)])
def test_long_window_actionable_preflight(tmp_path,days,train_end,window,val_dates,actual,needed):
    matrix=tmp_path/'matrix';matrix.mkdir();build(matrix,days=days,names=12,train_end=train_end,eval_start=train_end+1)
    with pytest.raises(ValueError) as error:p.long_window_data(matrix,2021,window,val_dates)
    message=str(error.value)
    for value in ('2021',str(window),str(actual),str(needed),'下一步'):assert value in message


def test_notebook_gate_defaults_compile_and_validation():
    nb=b.colab_notebook()
    assert "REHEARSE_FIRST = MODE != 'SPEED_TEST'" in nb['cells'][0]['source']
    cell=next(c['source'] for c in nb['cells'] if 'def stream_training' in c['source'])
    assert 'launch_with_rehearsal' in cell and 'prepared.json' in cell and 'environment_snapshot' in cell
    assert 'stream_training(common+' not in cell
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(c['source'],f'cell-{i}','exec')
    nbformat = pytest.importorskip('nbformat', reason='offline CPU runtime has no nbformat')
    nbformat.validate(nbformat.from_dict(nb))


def test_three_in_memory_mutations_are_killed(tmp_path):
    source=inspect.getsource(p.launch_with_rehearsal)
    mutations=[("stream(command,env)","pass # bypass rehearsal"),
               ("saved.get('identity')==binding","True"),
               ("raise\n", "return\n")]
    for i,(before,after) in enumerate(mutations):
        assert before in source
        ns=dict(p.__dict__);exec(compile(source.replace(before,after),f'in-memory-gate-{i}','exec'),ns)
        mutated=ns['launch_with_rehearsal'];out=tmp_path/str(i)
        with pytest.raises(AssertionError):
            if i==0:assert len(launch(out,implementation=mutated))==4
            elif i==1:
                launch(out);assert len(launch(out,identity={'source':'new'},implementation=mutated))==4
            else:
                try:launch(out,fail=1,implementation=mutated)
                except subprocess.CalledProcessError:pass
                else:raise AssertionError('failure swallowed')
        print(f'IN_MEMORY gate-{i}: KILLED')


def test_run_identity_and_stale_marker_failure(tmp_path):
    launch(tmp_path)
    seen=[]
    with pytest.raises(subprocess.CalledProcessError):
        launch(tmp_path,identity={'source':'new'},fail=2,seen=seen)
    assert len(seen)==2 and not (tmp_path/'rehearsal_passed.json').exists()
    assert len(launch(tmp_path))==4


def test_notebook_actual_runner_cell_control_flow(tmp_path,monkeypatch):
    from types import SimpleNamespace
    nb=b.colab_notebook(run_name='03_formal_resume')
    source=next(c['source'] for c in nb['cells'] if 'def stream_training' in c['source'])
    # Execute the generated cell itself; replace subprocess boundary only.
    prepared=tmp_path/'prepared.json';prepared.write_text('{}')
    runtime=tmp_path/'source/environments/p4a-colab-runtime.json';runtime.parent.mkdir(parents=True)
    runtime.write_text('{"packages":[]}')
    monkeypatch.setattr(p,'environment_snapshot',lambda names:{'CPU_TEST':True})
    for failure in (1,2,3,None):
        out=tmp_path/str(failure);out.mkdir()
        calls=[]
        class Process:
            def __init__(self,command,**kwargs):calls.append(command);self.stdout=[]
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def wait(self):return 7 if failure and len(calls)==failure else 0
        # Redirect only the prepared identity read, leaving generated paths/commands intact.
        def path(value):return prepared if value=='/content/data/prepared.json' else Path(value)
        ns=dict(MODE='FORMAL',REHEARSE_FIRST=True,ALLOW_NEW_RUN=False,root=tmp_path,
                EXPERIMENT_DIR=str(out),Path=path,json=json,
                os=__import__('os'),sys=__import__('sys'),
                subprocess=SimpleNamespace(Popen=Process,PIPE=-1,STDOUT=-2,CalledProcessError=subprocess.CalledProcessError),
                p4=SimpleNamespace(prepare_source=lambda *a:None,verify=lambda *a,**k:None,digest=lambda *a:{'sha256':'test'}))
        if failure:
            with pytest.raises(subprocess.CalledProcessError):exec(source,ns)
            assert len(calls)==failure and all('--rehearse' in c for c in calls)
        else:
            exec(source,ns);assert len(calls)==4 and '--rehearse' not in calls[-1]


@pytest.mark.parametrize('arm',['gru','crossstock'])
def test_w60_train_fit_bitwise_against_original(tmp_path,monkeypatch,arm):
    import numpy as np
    import torch
    from test_p4_source_compat import old_module
    from test_p4_rehearsal import matrix
    from test_seq_runner import CFG,CausalConvMixer
    from research_lab import seq_runner as sr
    train,predict=sr.train_one,sr.predict
    def cpu_train(*args,**kwargs):
        args=list(args);args[6]='cpu'
        return train(*args,**kwargs,mixer_factory=CausalConvMixer)
    monkeypatch.setattr(sr,'train_one',cpu_train)
    monkeypatch.setattr(sr,'predict',lambda model,data,rows,device:predict(model,data,rows,'cpu'))
    ref=matrix(tmp_path/'matrix');old=old_module(tmp_path,'p4_colab')
    monkeypatch.setattr(p,'long_window_data',lambda *a,**k:pytest.fail('W60 entered preflight'))
    class Observer:
        speed=False;device='cpu';records=[];contract={}
        def check(self):pass
        def load_resume(self):return None
        def __call__(self,*a):pass
    cfg=dict(CFG,window=60,epochs=1,steps_per_epoch=1,d_model=8,n_cross=1,dates_per_batch=1,stocks_per_date=12,min_names=10,val_dates=15,val_max_rows=0)
    results=[]
    for i,mod in enumerate((old,p)):
        obs=Observer();obs.root=tmp_path/str(i);obs.root.mkdir()
        np.random.seed(91);torch.manual_seed(91)
        frame,_=mod.train_fit(tmp_path/'matrix',2021,arm,17,cfg,ref,obs)
        results.append((frame,torch.load(obs.root/'best.pt',weights_only=False)['model'],torch.get_rng_state().numpy().tobytes(),np.random.get_state()))
    a,z=results
    assert a[0].score.to_numpy().tobytes()==z[0].score.to_numpy().tobytes()
    for key in a[1]:assert a[1][key].numpy().tobytes()==z[1][key].numpy().tobytes()
    assert a[2]==z[2]
    assert a[3][1].tobytes()==z[3][1].tobytes() and a[3][2:]==z[3][2:]
