"""Offline gate control-flow tests; no CUDA or official-mixer claim."""
import inspect
import json
from pathlib import Path
import subprocess
import pytest
from research_lab import p4_colab as p, p4_bundle as b
from test_seq_runner import build


def evidence(tmp_path,request,common):
    """Synthetic publications, real completed-output checks; no CUDA claim."""
    import tempfile,torch
    from test_p4 import formal_fixture
    root=Path(tempfile.mkdtemp(prefix='p4-REHEARSAL-fixture-',dir='/tmp'))
    frozen=json.loads((b.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    jobs=[j for j in p.fit_queue(frozen) if j['queue']==('Q1' if request['queue']=='Q5' else request['queue'])]
    jobs=([next(j for j in jobs if j['arm']==a) for a in ('crossstock','mamba2')] if request['queue']=='Q5' else
          [j for j in jobs if j['window']==request['window'] and j['arm']==request['arm']][:1])
    fits=[]
    for job in jobs:
        name=f"{job['arm']}-w{job['window']}";fit=root/name;attempt=fit/'attempt-active';attempt.mkdir(parents=True)
        _,files=formal_fixture(attempt,values=(.3,),job=job)
        contract=dict(**job,recipe=dict(frozen['recipe'],window=job['window'],lr=job['lr'],epochs=1,steps_per_epoch=1),speed_test=False,evidence_class=p.REHEARSAL_EVIDENCE,rehearsal_binding=p.hashed(request))
        for file in attempt.glob('*.pt'):
            payload=torch.load(file,weights_only=False);payload['contract']=contract;torch.save(payload,file)
        pointer=json.loads((attempt/'checkpoint-pointer.json').read_text())
        for entry in pointer.values():
            if entry:entry['identity']=p.digest(attempt/entry['file'])
        (attempt/'checkpoint-pointer.json').write_text(json.dumps(pointer))
        p.atomic_json(fit/'REHEARSAL-complete.json',dict(contract=contract,outputs={'attempt-active/'+n:p.digest(attempt/n) for n in files},evidence_class=p.REHEARSAL_EVIDENCE))
        fits.append(name)
    data=Path(common[common.index('--data-dir')+1]);data.mkdir(exist_ok=True,parents=True)
    (data/'manifest.json').write_text('{}')
    bundle=Path(common[common.index('--bundle')+1]);ref=bundle/'reference/keys.parquet';ref.parent.mkdir(exist_ok=True,parents=True);ref.write_text('synthetic reference identity')
    result=dict(evidence_class=p.REHEARSAL_EVIDENCE,artifact_class='TEMPORARY',output=str(root),
        queue=request['queue'],window=request['window'],arm=request['arm'],steps=1,fits=fits,
        inputs=dict(data_realpath=str(data.resolve()),matrix_manifest=p.digest(data/'manifest.json'),reference=p.digest(ref)),
        implementation={n:p.digest(Path(p.__file__).with_name(n)) for n in ('p4_colab.py','p4_eval.py','seq_runner.py')},
        environment=dict(python='test',torch='test',numpy='test',device='cuda',mixer='official'),
        cuda_max_memory_allocated=1,request=request,context=request['binding']['identity'])
    if request['queue']=='Q5':result['diagnostic']={k:dict(R=10,label_support=dict(input_rows=10,used_rows=10,excluded_nan_rows=0,excluded_dates=[]),daily=[dict(permutation=r,date='2021-01-01',score_spearman=1.,ic=.2,ic_change=0.,top50_overlap=1.) for r in range(10)],permutation_mean_ic=[.2]*10,ic_std=0.,mamba_ic_gap=0.,order_sensitive=True,rule='IC std >= absolute crossstock minus mamba2 IC gap',evidence_class='DESCRIPTIVE_ONLY') for k in ('validation','evaluation')}
    p.atomic_json(root/'REHEARSAL.json',result)
    return result


def write_evidence(tmp_path,command,env):
    request=json.loads(Path(env['P4_REHEARSAL_REQUEST']).read_text())
    result=evidence(tmp_path,request,command)
    subprocess.run([__import__('sys').executable,'-c',"import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])",env['P4_REHEARSAL_RESULT'],json.dumps(result)],check=True,env=env)


TEST_ENV=dict(python='test',packages=dict(torch='test',numpy='test'))


def launch(tmp_path, fail=None, identity=None, enabled=True, implementation=None, seen=None, run="budget"):
    if seen is None:seen=[]
    def stream(cmd,env):
        seen.append(cmd)
        if fail and len(seen)==fail:raise subprocess.CalledProcessError(7,cmd)
        if '--rehearse' in cmd:write_evidence(tmp_path,cmd,env)
    common=['python','-m','research_lab.p4_colab','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'local'),'--drive-root',run]
    (implementation or p.launch_with_rehearsal)(common,{},tmp_path,dict(identity or {'source':'a','data':'b','environment':'c'},environment=(identity or {}).get('environment') if isinstance((identity or {}).get('environment'),dict) else dict(TEST_ENV,tag=(identity or {}).get('environment','c'))),stream,enabled=enabled)
    return seen


def test_gate_success_cache_and_identity_invalidation(tmp_path):
    calls=launch(tmp_path)
    assert len(calls)==4 and '--rehearse' not in calls[-1]
    assert [c[c.index('--queue')+1] for c in calls[:-1]]==['Q3','Q3','Q5']
    assert [c[c.index('--window')+1] for c in calls[:2]]==['120','250']
    for c in calls[:-1]:
        assert c[c.index('--steps')+1]=='1' and '--drive-root' not in c
        assert c[c.index('--data-dir')+1]==str(tmp_path/'local') and '--bundle' in c
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
    mutations=[("stream(command,stage_env)","pass # bypass rehearsal"),
               ("validate_rehearsal_marker(saved,binding)","True"),
               ("raise\n", "return\n")]
    for i,(before,after) in enumerate(mutations):
        assert before in source
        ns=dict(p.__dict__);exec(compile(source.replace(before,after),f'in-memory-gate-{i}','exec'),ns)
        mutated=ns['launch_with_rehearsal'];out=tmp_path/str(i)
        with pytest.raises(AssertionError):
            if i==0:
                try:launch(out,implementation=mutated)
                except ValueError:raise AssertionError('missing evidence rejected')
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
    source=next(c['source'] for c in nb['cells'] if 'def stream_training' in c['source']).replace("'--data-dir','/content/data'",repr('--data-dir')+','+repr(str(tmp_path/'local')))
    # Execute the generated cell itself; replace subprocess boundary only.
    prepared=tmp_path/'prepared.json';prepared.write_text('{}')
    runtime=tmp_path/'source/environments/p4a-colab-runtime.json';runtime.parent.mkdir(parents=True)
    runtime.write_text('{"packages":[]}')
    monkeypatch.setattr(p,'environment_snapshot',lambda names:TEST_ENV)
    for failure in (1,2,3,None):
        out=tmp_path/str(failure);out.mkdir()
        calls=[]
        class Process:
            def __init__(self,command,**kwargs):
                calls.append(command);self.stdout=[]
                if '--rehearse' in command and not (failure and len(calls)==failure):
                    command=list(command);command[command.index('--data-dir')+1]=str(tmp_path/'local')
                    write_evidence(tmp_path,command,kwargs['env'])
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


@pytest.mark.parametrize('behavior',['empty','swallowed','mismatch'])
def test_real_subprocess_zero_exit_requires_evidence(tmp_path,behavior):
    import ast,sys
    cell=next(c['source'] for c in b.colab_notebook()['cells'] if 'def stream_training' in c['source'])
    function=next(n for n in ast.parse(cell).body if isinstance(n,ast.FunctionDef) and n.name=='stream_training')
    ns=dict(subprocess=subprocess)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'notebook-stream','exec'),ns)
    seen=[]
    scripts={'empty':'pass','swallowed':'try:\n raise RuntimeError("lost")\nexcept Exception:\n pass',
             'mismatch':'import os,json; from pathlib import Path; p=os.environ.get("P4_REHEARSAL_RESULT"); p and Path(p).write_text(json.dumps({"queue":"Q4"}))'}
    def stream(command,env):
        seen.append(command)
        if behavior=='mismatch':
            request=json.loads(Path(env['P4_REHEARSAL_REQUEST']).read_text())
            result=evidence(tmp_path,request,command);result['window']=250
            p.atomic_json(Path(result['output'])/'REHEARSAL.json',result)
            script="import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])"
            ns['stream_training']([sys.executable,'-c',script,env['P4_REHEARSAL_RESULT'],json.dumps(result)],env)
        else:ns['stream_training']([sys.executable,'-c',scripts[behavior]],env)
    common=[sys.executable,'-m','research_lab.p4_colab','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data'),'--drive-root','budget']
    with pytest.raises((ValueError,OSError),match='rehearsal|REHEARSAL|排練'):
        p.launch_with_rehearsal(common,dict(__import__('os').environ),tmp_path,{'source':'test','environment':TEST_ENV},stream)
    assert len(seen)==1 and '--rehearse' in seen[0]
    assert not (tmp_path/'rehearsal_passed.json').exists()


@pytest.mark.parametrize('field,value',[('stages',[]),('evidence_class','TAMPERED'),('unknown',True)])
def test_valid_json_marker_tamper_reruns(tmp_path,field,value):
    launch(tmp_path)
    marker=tmp_path/'rehearsal_passed.json';saved=json.loads(marker.read_text());saved[field]=value
    marker.write_text(json.dumps(saved))
    assert len(launch(tmp_path))==4


@pytest.mark.parametrize('damage',['fits','window','arm','queue','steps','inputs','implementation','context','request','environment','class','missing_scores','missing_reference','missing_complete','complete_identity','q5_validation','q5_evaluation'])
def test_gate_independently_rejects_invalid_publications(tmp_path,damage):
    common=['python','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data')]
    request=dict(binding={'identity':{'environment':TEST_ENV}},queue='Q5' if damage.startswith('q5') else 'Q3',window=None if damage.startswith('q5') else 120,arm=None if damage.startswith('q5') else 'mamba2',steps=1,nonce='unique')
    result=evidence(tmp_path,request,common)
    path=tmp_path/'result.json';root=Path(result['output'])
    if damage in ('fits','inputs','implementation','context','request','environment'):result[damage]=[]
    elif damage in ('window','steps'):result[damage]=True
    elif damage in ('queue','arm'):result[damage]='WRONG'
    elif damage=='class':result['evidence_class']='TAMPERED'
    elif damage.startswith('q5'):result['diagnostic'].pop(damage.removeprefix('q5_'))
    else:
        fit=root/result['fits'][0]
        if damage=='complete_identity':
            saved=json.loads((fit/'REHEARSAL-complete.json').read_text());saved['contract']['rehearsal_binding']='stale';p.atomic_json(fit/'REHEARSAL-complete.json',saved)
        elif damage=='missing_complete':(fit/'REHEARSAL-complete.json').unlink()
        else:(fit/'attempt-active'/('scores.parquet' if damage=='missing_scores' else 'reference.parquet')).unlink()
    p.atomic_json(root/'REHEARSAL.json',result);p.atomic_json(path,result)
    with pytest.raises(ValueError,match='REHEARSAL'):p.validate_rehearsal_result(path,request,common)


@pytest.mark.parametrize('damage',['stage_count','duplicate','identity','seconds','fit_count','stage_class','stage_unknown','digest','passed','stage_type','steps_type'])
def test_marker_complete_schema(tmp_path,damage):
    launch(tmp_path)
    path=tmp_path/'rehearsal_passed.json';saved=json.loads(path.read_text());binding=saved['identity']
    if damage=='stage_count':saved['stages'].pop()
    elif damage=='duplicate':saved['stages'][1]=saved['stages'][0]
    elif damage=='identity':saved['stages'][0]['identity']={}
    elif damage=='seconds':saved['stages'][0]['seconds']=True
    elif damage=='fit_count':saved['stages'][0]['fit_count']=True
    elif damage=='stage_class':saved['stages'][0]['evidence_class']='TAMPERED'
    elif damage=='stage_unknown':saved['stages'][0]['unknown']=1
    elif damage=='digest':saved['digest']='0'*64
    elif damage=='passed':saved['passed']=1
    elif damage=='stage_type':saved['stages'][0]=None
    elif damage=='steps_type':saved['stages'][0]['identity']['steps']=True
    # Recompute digest so schema checks must reject independently of checksum.
    if damage!='digest':saved['digest']=p.hashed({k:v for k,v in saved.items() if k!='digest'})
    assert not p.validate_rehearsal_marker(saved,binding)
    path.write_text(json.dumps(saved));assert len(launch(tmp_path))==4


def test_three_new_in_memory_mutations_are_killed(tmp_path):
    source=inspect.getsource(p.validate_rehearsal_result)
    common=['python','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data')]
    request=dict(binding={'identity':{'environment':TEST_ENV}},queue='Q3',window=120,arm='mamba2',steps=1,nonce='unique')
    result=evidence(tmp_path,request,common);path=tmp_path/'result.json'
    mutations=[("validate_rehearsal_schema(result,rehearsal_schema('result',request))",'pass # bypass schema', 'window'),
               ('validate_fit_outputs(fit,contract,complete[\'outputs\'],rehearsal=True)','pass # missing artifact check','artifact'),
               ("validate_rehearsal_schema(result,rehearsal_schema('result',request))",'pass # bypass schema','context')]
    import copy
    for i,(before,after,damage) in enumerate(mutations):
        changed=copy.deepcopy(result)
        if damage=='window':changed['window']=250
        elif damage=='context':changed['context']={**changed['context'],'source':'STALE'}
        else:
            score=Path(result['output'])/result['fits'][0]/'attempt-active/scores.parquet';original=score.read_bytes();score.unlink()
        p.atomic_json(Path(result['output'])/'REHEARSAL.json',changed);p.atomic_json(path,changed)
        with pytest.raises(ValueError):p.validate_rehearsal_result(path,request,common)
        assert before in source
        changed_source=source.replace(before,after)
        if damage!='artifact':changed_source=changed_source.replace("validate_rehearsal_schema(publication,rehearsal_schema('result',request))",'pass # bypass publication schema as well')
        ns=dict(p.__dict__);exec(compile(changed_source,f'in-memory-evidence-{i}','exec'),ns)
        with pytest.raises(AssertionError):
            try:ns['validate_rehearsal_result'](path,request,common)
            except ValueError:pass
            else:raise AssertionError('mutant accepted invalid evidence')
        if damage=='artifact':score.write_bytes(original)
        print(f'IN_MEMORY evidence-{damage}: KILLED')


def test_cli_publishes_gate_result_with_observed_context(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';(bundle/'source/environments').mkdir(parents=True)
    (bundle/'source/environments/p4a-colab-runtime.json').write_text('{"packages":["torch","numpy"]}')
    (bundle/'bundle-manifest.json').write_text('{"synthetic":true}')
    data=tmp_path/'data';data.mkdir();(data/'prepared.json').write_text('{"matrix":"synthetic"}')
    output=tmp_path/'rehearsal';output.mkdir()
    request=dict(binding={'identity':{'source':'EXPECTED'}},nonce='unique')
    request_file=tmp_path/'request.json';p.atomic_json(request_file,request)
    result_file=tmp_path/'result.json'
    monkeypatch.setenv('P4_REHEARSAL_REQUEST',str(request_file))
    monkeypatch.setenv('P4_REHEARSAL_RESULT',str(result_file))
    monkeypatch.setattr(p,'environment_snapshot',lambda packages:TEST_ENV)
    monkeypatch.setattr(p,'rehearse',lambda *a,**k:dict(output=str(output),evidence_class=p.REHEARSAL_EVIDENCE))
    p.main(['--rehearse','--bundle',str(bundle),'--data-dir',str(data),'--queue','Q3','--window','120','--arm','mamba2'])
    saved=json.loads(result_file.read_text())
    assert saved==json.loads((output/'REHEARSAL.json').read_text())
    assert saved['request']==request
    assert saved['context']==dict(source=p.digest(bundle/'bundle-manifest.json'),data={'matrix':'synthetic'},data_realpath=str(data.resolve()),environment=TEST_ENV)
    assert saved['context']!=request['binding']['identity'] # recomputed, never blindly copied


def test_marker_rejects_nested_identity_numeric_type_change(tmp_path):
    identity=dict(source={'bytes':1,'sha256':'a'*64})
    launch(tmp_path,identity=identity)
    path=tmp_path/'rehearsal_passed.json';saved=json.loads(path.read_text())
    saved['identity']['identity']['source']['bytes']=True
    for stage in saved['stages']:stage['identity']['binding']['identity']['source']['bytes']=True
    saved['digest']=p.hashed({k:v for k,v in saved.items() if k!='digest'})
    binding=json.loads(path.read_text())['identity']
    assert not p.validate_rehearsal_marker(saved,binding)


BAD_SCHEMA_VALUES=['true', [], {}, None, True, float('nan'), float('inf'), 10**400]
Q5_PATHS=[('R',),('label_support',),('label_support','input_rows'),('label_support','used_rows'),
    ('label_support','excluded_nan_rows'),('label_support','excluded_dates'),('label_support','excluded_dates',0),
    ('daily',),('daily',0),*[('daily',0,k) for k in ('permutation','date','score_spearman','ic','ic_change','top50_overlap')],
    ('permutation_mean_ic',),('permutation_mean_ic',0),('ic_std',),('mamba_ic_gap',),('order_sensitive',),('rule',),('evidence_class',)]


@pytest.mark.parametrize('path',Q5_PATHS,ids=lambda v:'.'.join(map(str,v)))
@pytest.mark.parametrize('bad',BAD_SCHEMA_VALUES,ids=['string','list','dict','null','bool','nan','inf','huge'])
def test_q5_shared_schema_rejects_each_malformed_field(tmp_path,path,bad):
    import copy
    split=dict(R=10,label_support=dict(input_rows=12,used_rows=10,excluded_nan_rows=2,excluded_dates=['2020-12-31']),
        daily=[dict(permutation=r,date='2021-01-01',score_spearman=1.,ic=.2,ic_change=0.,top50_overlap=1.) for r in range(10)],
        permutation_mean_ic=[.2]*10,ic_std=0.,mamba_ic_gap=0.,order_sensitive=False,
        rule='IC std >= absolute crossstock minus mamba2 IC gap',evidence_class='DESCRIPTIVE_ONLY')
    changed=copy.deepcopy(split);target=changed
    for key in path[:-1]:target=target[key]
    target[path[-1]]=bad
    # Empty excluded dates and either boolean order_sensitive are valid output values.
    valid=(path==('label_support','excluded_dates') and bad==[] or path==('order_sensitive',) and type(bad) is bool)
    if valid:p.validate_rehearsal_schema(changed,p.rehearsal_schema('q5'))
    else:
        with pytest.raises(ValueError):p.validate_rehearsal_schema(changed,p.rehearsal_schema('q5'))


@pytest.mark.parametrize('damage',['excluded_dates','rule','date','permutation','complete_link','publication_link','result_link'])
def test_gate3_real_publication_fail_closed(tmp_path,damage):
    import sys,os
    seen=[]
    def stream(command,env):
        seen.append(command)
        if '--rehearse' not in command:pytest.fail('formal launched with invalid publication')
        request=json.loads(Path(env['P4_REHEARSAL_REQUEST']).read_text())
        result=evidence(tmp_path,request,command);root=Path(result['output'])
        if damage.endswith('_link'):
            target={'complete_link':root/result['fits'][0]/'REHEARSAL-complete.json','publication_link':root/'REHEARSAL.json','result_link':Path(env['P4_REHEARSAL_RESULT'])}[damage]
        elif request['queue']=='Q5':
            split=result['diagnostic']['validation']
            if damage=='excluded_dates':split['label_support']['excluded_dates']='true'
            elif damage=='rule':split['rule']=[]
            elif damage=='date':split['daily'][0]['date']={}
            else:split['daily'][1]['permutation']=True
        p.atomic_json(root/'REHEARSAL.json',result)
        subprocess.run([sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])',env['P4_REHEARSAL_RESULT'],json.dumps(result)],check=True,env=env)
        if damage.endswith('_link'):
            moved=target.with_suffix('.original');target.rename(moved);target.symlink_to(moved)
    common=[sys.executable,'--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data'),'--drive-root','budget']
    with pytest.raises(ValueError,match='REHEARSAL'):p.launch_with_rehearsal(common,dict(os.environ),tmp_path,{'environment':TEST_ENV},stream)
    assert seen and not (tmp_path/'rehearsal_passed.json').exists()


@pytest.mark.parametrize('damage',['huge','recursive_json','symlink'])
def test_gate3_invalid_marker_reruns_actionably(tmp_path,damage,capsys):
    launch(tmp_path);capsys.readouterr()
    marker=tmp_path/'rehearsal_passed.json'
    if damage=='huge':
        saved=json.loads(marker.read_text());saved['stages'][0]['seconds']=10**400
        saved['digest']=p.hashed({k:v for k,v in saved.items() if k!='digest'});marker.write_text(json.dumps(saved))
    elif damage=='recursive_json':marker.write_text('['*2000+'0'+']'*2000)
    else:
        moved=tmp_path/'old-marker.json';marker.rename(moved);marker.symlink_to(moved)
    assert len(launch(tmp_path))==4
    output=capsys.readouterr().out
    for word in ('標記','期望','實際','下一步'):assert word in output
    assert marker.is_file() and not marker.is_symlink()


@pytest.mark.parametrize('damage',['date_invalid','excluded_invalid','R_count','daily_duplicate','daily_missing','mean_count','unknown_split','unknown_support','unknown_daily','missing_rule','numeric_range'])
def test_gate3_q5_closed_schema_and_support(tmp_path,damage):
    common=['python','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data')]
    request=dict(binding={'identity':{'environment':TEST_ENV}},queue='Q5',window=None,arm=None,steps=1,nonce='unique')
    result=evidence(tmp_path,request,common);split=result['diagnostic']['validation']
    if damage=='date_invalid':
        for row in split['daily']:row['date']='2021-02-30'
    elif damage=='excluded_invalid':split['label_support']['excluded_dates']=['not-a-date']
    elif damage=='R_count':split['R']=9
    elif damage=='daily_duplicate':split['daily'].append(dict(split['daily'][0]))
    elif damage=='daily_missing':split['daily'].pop()
    elif damage=='mean_count':split['permutation_mean_ic'].pop()
    elif damage=='unknown_split':split['unknown']=1
    elif damage=='unknown_support':split['label_support']['unknown']=1
    elif damage=='unknown_daily':split['daily'][0]['unknown']=1
    elif damage=='missing_rule':split.pop('rule')
    else:split['daily'][0]['top50_overlap']=2
    path=tmp_path/'result.json';p.atomic_json(Path(result['output'])/'REHEARSAL.json',result);p.atomic_json(path,result)
    with pytest.raises(ValueError,match='REHEARSAL'):p.validate_rehearsal_result(path,request,common)


def test_gate3_three_new_schema_mutations_are_killed(tmp_path,monkeypatch):
    # Execute the actual new regression tests against in-memory source mutations.
    source=inspect.getsource(p.validate_rehearsal_schema)
    mutations=[("set(value)==set(args[0])", "set(args[0])<=set(value)", 'object'),
        ("if valid:datetime.fromisoformat(value.replace('Z','+00:00'))", 'if False:pass', 'date'),
        ("type(value) is int if kind=='int' else type(value) in (int,float)", "isinstance(value,int) if kind=='int' else isinstance(value,(int,float))", 'bool')]
    for before,after,name in mutations:
        assert before in source
        changed=source.replace(before,after)
        ns=dict(p.__dict__);exec(compile(changed,'in-memory-schema-'+name,'exec'),ns)
        with monkeypatch.context() as patch:
            patch.setattr(p,'validate_rehearsal_schema',ns['validate_rehearsal_schema'])
            with pytest.raises(pytest.fail.Exception,match='DID NOT RAISE'):
                if name=='object':test_gate3_q5_closed_schema_and_support(tmp_path,'unknown_support')
                elif name=='date':test_gate3_q5_closed_schema_and_support(tmp_path,'date_invalid')
                else:test_q5_shared_schema_rejects_each_malformed_field(tmp_path,('daily',0,'permutation'),True)
        print(f'IN_MEMORY gate3-schema-{name}: KILLED (new regression DID NOT RAISE)')


MARKER_PATHS=[('identity',),('passed',),('stages',),('stages',0),('stages',0,'stage'),('stages',0,'seconds'),
    ('stages',0,'identity'),*[('stages',0,'identity',k) for k in ('binding','queue','window','arm','steps','result')],
    ('stages',0,'identity','result','bytes'),('stages',0,'identity','result','sha256'),('stages',0,'evidence_class'),('stages',0,'fit_count'),('evidence_class',),('digest',)]


@pytest.mark.parametrize('path',MARKER_PATHS,ids=lambda v:'.'.join(map(str,v)))
@pytest.mark.parametrize('bad',BAD_SCHEMA_VALUES,ids=['string','list','dict','null','bool','nan','inf','huge'])
def test_gate3_marker_each_field_types(path,bad):
    import copy
    binding={'identity':{'environment':TEST_ENV},'command':['python'],'policy':'test'}
    stages=[dict(stage=name,seconds=1.,identity=dict(binding=binding,queue=queue,window=window,arm=arm,steps=1,result=dict(bytes=1,sha256='a'*64)),evidence_class=p.REHEARSAL_EVIDENCE,fit_count=count) for name,queue,window,arm,count in p.REHEARSAL_STAGES]
    saved=dict(identity=binding,passed=True,stages=stages,evidence_class=p.REHEARSAL_EVIDENCE)
    saved['digest']=p.hashed(saved);changed=copy.deepcopy(saved);target=changed
    for key in path[:-1]:target=target[key]
    target[path[-1]]=bad
    if path!=('digest',):changed['digest']=p.hashed({k:v for k,v in changed.items() if k!='digest'}) if not (type(bad) is float and not __import__('math').isfinite(bad)) else 'a'*64
    assert p.validate_rehearsal_marker(changed,binding) is (path==('passed',) and bad is True)


def test_gate3_publication_preserves_json_types(tmp_path):
    import copy
    common=['python','--bundle',str(tmp_path/'bundle'),'--data-dir',str(tmp_path/'data')]
    request=dict(binding={'identity':{'environment':TEST_ENV}},queue='Q3',window=120,arm='mamba2',steps=1,nonce='unique')
    result=evidence(tmp_path,request,common);path=tmp_path/'result.json';p.atomic_json(path,result)
    publication=copy.deepcopy(result);publication['steps']=True
    p.atomic_json(Path(result['output'])/'REHEARSAL.json',publication)
    with pytest.raises(ValueError,match='REHEARSAL'):p.validate_rehearsal_result(path,request,common)


def test_gate3_q5_real_dates_preserve_numeric_diagnostic():
    import numpy as np
    from research_lab.p4_eval import permutation_summary
    dates=np.repeat(np.arange(3),12);calendar=np.array(['2021-01-01','2021-01-04','2021-01-05'],dtype='datetime64[D]')
    codes=np.tile(np.arange(12),3);labels=codes.astype(float);labels[-12:]=np.nan
    args=(codes,labels,codes,-codes,[codes.copy() for _ in range(10)])
    index=permutation_summary(dates,*args);actual=permutation_summary(calendar[dates],*args)
    p.validate_rehearsal_schema(actual,p.rehearsal_schema('q5'))
    assert actual['label_support']['excluded_dates']==['2021-01-05']
    for old,new in zip(index['daily'],actual['daily']):
        assert new['date']==str(calendar[int(old['date'])])
        assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in new.items() if k!='date'}
    assert {k:v for k,v in index.items() if k not in ('daily','label_support')}=={k:v for k,v in actual.items() if k not in ('daily','label_support')}
