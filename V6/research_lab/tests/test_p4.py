"""P4 CPU boundary tests: synthetic artifacts only, no research evidence."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from research_lab import p4_bundle as bundle
from research_lab import p4_colab as p4
from research_lab.seq_models import SeqRanker


def test_gru_shape_gradient():
    m = SeqRanker('gru', n_features=48, d_model=64, n_layers=2)
    out = m(torch.randn(3, 60, 48))
    assert all(v.shape == (3,) and torch.isfinite(v).all() for v in out.values())
    sum(v.square().mean() for v in out.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert 40000 < sum(p.numel() for p in m.parameters()) < 100000


@pytest.mark.parametrize('name,kind', [('../escape','file'),('/absolute','file'),('ok','symlink'),('ok','hardlink'),('a/../../escape','file'),('a\\escape','file')])
def test_extract_rejects_hostile(tmp_path, name, kind):
    archive = tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        member = tarfile.TarInfo(name)
        if kind != 'file':
            member.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            member.linkname = '/etc/passwd'
        tf.addfile(member, io.BytesIO(b''))
    with pytest.raises(ValueError): bundle.extract(archive, tmp_path/'out')
    assert not (tmp_path/'escape').exists()


def test_archive_hash_and_roundtrip(tmp_path):
    source=tmp_path/'a'; source.write_bytes(b'original')
    archive=tmp_path/'bundle.tar.gz'
    bundle.write_archive(archive, {'source/V6/research_lab/a.py':source}, {'source_commit':'abc'})
    bundle.extract(archive,tmp_path/'out')
    assert (tmp_path/'out/source/V6/research_lab/a.py').read_bytes()==b'original'
    (tmp_path/'out/source/V6/research_lab/a.py').write_bytes(b'mutated!')
    with pytest.raises(ValueError,match='hash'): bundle.verify(tmp_path/'out')
    with pytest.raises(FileExistsError): bundle.extract(archive,tmp_path/'out')


def test_budget_reserve_wait_and_unknown():
    assert p4.budget_charge(3600,600,2)==dict(wall_hours=1.,active_hours=5/6,compute_units=5/3,guard_wait_seconds=600.)
    p4.check_budget(10,20,2,4,6,12,.25)
    with pytest.raises(p4.BudgetStop): p4.check_budget(10,20,2,4,6.5,12,.25)
    for value in (float('nan'),-1):
        with pytest.raises(ValueError): p4.budget_charge(10,value,2)
    with pytest.raises(ValueError): p4.check_budget(10,20,0,0,1,2,.24)


def test_no_cuda_refuses_before_writing(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'): p4.require_cuda()
    assert list(tmp_path.iterdir())==[]


def test_publication_resume_and_mutation(tmp_path):
    contract={'matrix':'abc','seed':17}
    (tmp_path/'speed.json').write_text(json.dumps(p4.speed_report('gru',[{'seconds':1}]*2,[{'gpu':'fake'}],2,[.01]*200)))
    contract['speed_test']=True
    p4.complete_fit(tmp_path,contract,{'epochs':2},['speed.json'])
    assert p4.resume_fit(tmp_path,contract)['job']['epochs']==2
    with pytest.raises(ValueError,match='identity'): p4.resume_fit(tmp_path,{'matrix':'abc','seed':29})
    (tmp_path/'speed.json').write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash'): p4.resume_fit(tmp_path,contract)


def test_speed_schema():
    report=p4.speed_report('fake',[{'seconds':4},{'seconds':6}], [dict(gpu='fake',temperature_c=70,clock_mhz=1000,power_w=50)], 12, [0.1]*200)
    assert report['evidence_class']=='ENGINEERING_SPEED_ONLY'
    assert report['epochs']==2 and report['steps']==200
    assert report['epoch_seconds']==[4,6] and report['step_seconds_mean']==pytest.approx(.1)
    assert 'research_conclusion' not in report
    with pytest.raises(ValueError): p4.speed_report('fake',[],[],0,[])


def test_pack_synthetic_selfcontained(tmp_path):
    import pandas as pd
    folds=tmp_path/'folds';d=folds/'fold_2021';d.mkdir(parents=True)
    for name in bundle.MATRIX_FILES:
        (d/name).write_bytes(b'{}' if name.endswith('.json') else b'raw synthetic matrix')
    (d/'X_rankgauss.npy').write_bytes(b'never pack derived cache')
    ref=tmp_path/'p1';ref.mkdir()
    pd.DataFrame({'system':['c0plus'],'score':[.1]}).to_parquet(ref/'rank_average.parquet')
    (ref/'manifest.json').write_text(json.dumps({'outputs':{'rank_average.parquet':bundle.digest(ref/'rank_average.parquet')}}))
    lock=bundle.ROOT/'environments/p4a-colab.requirements.lock.txt'
    archive=tmp_path/'p4.tar.gz'
    mf=bundle.pack(folds,[2021],ref,lock,archive)
    assert not any('X_rankgauss' in n for n in mf['files'])
    assert mf['metadata']['source_commit']
    out=tmp_path/'unpacked';bundle.extract(archive,out,bundle.digest(archive)['sha256'])
    assert (out/'requirements.lock.txt').read_bytes()==lock.read_bytes()
    assert set(n for n in mf['files'] if n.startswith('folds/'))=={f'folds/fold_2021/{n}' for n in bundle.MATRIX_FILES}
    with pytest.raises(ValueError,match='hash'):bundle.extract(archive,tmp_path/'wrong','0'*64)


def budget_contract():
    return dict(max_hours=10,max_units=20,units_per_hour=2,reserve=.25)


def test_ledger_resume_contract_and_truncated_head(tmp_path):
    root=tmp_path/'budget';c=budget_contract();binding={'seed':17,'matrix':'abc'}
    with p4.Ledger(root,c) as ledger:
        ledger.start(binding,.1);ledger.end(binding,True)
    with p4.Ledger(root,c) as ledger:
        assert binding in ledger.state['completed']
        assert ledger.state['spent_hours']>=0
    with pytest.raises(ValueError,match='identity'): 
        with p4.Ledger(root,dict(c,max_units=21)):pass
    # Real mutation: revert head to a valid earlier prefix, leaving END generation.
    p4.atomic_json(root/'head.json',{'sequence':1,'identity':bundle.digest(root/'event-000001.json')})
    with pytest.raises(ValueError,match='head'):
        with p4.Ledger(root,c):pass


def test_crash_restart_latches_stop_and_charges_unknown_interval(tmp_path):
    import os
    import subprocess
    import sys
    code='''from research_lab.p4_colab import Ledger
import os
with Ledger(os.environ['P4_TEST_BUDGET'],dict(max_hours=10,max_units=20,units_per_hour=2,reserve=.25)) as ledger:
    ledger.start({'seed':17},.1)
    os.kill(os.getpid(),9)
'''
    child=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,P4_TEST_BUDGET=str(tmp_path)))
    assert child.returncode==-9
    with pytest.raises(p4.BudgetStop,match='STOP'):
        with p4.Ledger(tmp_path,budget_contract()):pass
    row=json.loads(sorted(tmp_path.glob('event-*.json'))[-1].read_text())
    assert row['state']['spent_hours']>0 and row['state']['stopped']
    with pytest.raises(p4.BudgetStop):
        with p4.Ledger(tmp_path,budget_contract()):pass


@pytest.mark.parametrize('loss',['head','generation','all'])
def test_missing_ledger_fail_closed(tmp_path,loss):
    with p4.Ledger(tmp_path,budget_contract()) as ledger:
        ledger.start({'seed':17},.1);ledger.end({'seed':17},True)
    (tmp_path/'run-synthetic').mkdir()
    if loss=='head':(tmp_path/'head.json').unlink()
    elif loss=='generation':(tmp_path/'event-000001.json').unlink()
    else:
        for p in tmp_path.glob('*.json'):p.unlink()
    with pytest.raises(ValueError):
        with p4.Ledger(tmp_path,budget_contract()):pass


def test_rankgauss_reuse_and_tag_mutation(tmp_path):
    from test_seq_runner import build
    from research_lab.seq_runner import stock_index
    build(tmp_path);stock_index(tmp_path)
    before={n:bundle.digest(tmp_path/n) for n in bundle.MATRIX_FILES}
    a=p4.rankgauss(tmp_path);assert p4.rankgauss(tmp_path)==a
    assert before=={n:bundle.digest(tmp_path/n) for n in bundle.MATRIX_FILES}
    (tmp_path/'X_rankgauss.tag').write_text('tamper')
    with pytest.raises(ValueError,match='tag'):p4.rankgauss(tmp_path)


def test_observer_cpu_checkpoint_best_previous_roundtrip(tmp_path,monkeypatch):
    from test_seq_runner import build, CFG
    from research_lab import seq_runner as sr
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(p4,'telemetry',lambda:dict(gpu='CPU_TEST_DOUBLE',temperature_c=70,clock_mhz=0,power_w=0))
    build(tmp_path)
    data=sr.SeqData(tmp_path);tr,va,_=data.split_rows(15)
    c={'matrix':'synthetic','seed':17}
    with p4.Ledger(tmp_path/'budget',budget_contract()) as ledger:
        ledger.start(c,.1)
        observer=p4.Observer(tmp_path/'checkpoints',c,ledger,device='cpu')
        cfg=dict(CFG,epochs=2,steps_per_epoch=2,d_model=64,n_layers=2)
        model,state=sr.train_one('gru',17,data,tr,va,cfg,'cpu',log=lambda *_:None,observer=observer)
        pointer=json.loads((tmp_path/'checkpoints/checkpoint-pointer.json').read_text())
        assert pointer['previous']['epoch']==1 and pointer['latest']['epoch']==2
        best=pointer['best'];path=tmp_path/'checkpoints'/best['file']
        assert bundle.digest(path)==best['identity']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['contract']==c and 'rng_state' in payload
        assert all(torch.equal(v,payload['best_sd'][k]) for k,v in model.state_dict().items())
        ledger.end(c,True)
    assert len(observer.steps)==4 and len(state['history'])==2


def test_thermal_stop_without_wait(tmp_path,monkeypatch):
    monkeypatch.setattr(p4,'telemetry',lambda:dict(gpu='fake',temperature_c=85,clock_mhz=1000,power_w=50))
    with p4.Ledger(tmp_path/'budget',budget_contract()) as ledger:
        ledger.start({'seed':17},.1)
        observer=p4.Observer(tmp_path/'ckpt',{},ledger)
        with pytest.raises(RuntimeError,match='85C'):observer.check()
        ledger.end({'seed':17},False)


def test_sha_validator_mutation_detected(tmp_path,monkeypatch):
    # A disabled digest guard must make our corruption regression fail to raise.
    original=tmp_path/'original';original.write_bytes(b'good')
    bundle.write_archive(tmp_path/'bundle.tar.gz',{'source/V6/research_lab/a.py':original},{})
    bundle.extract(tmp_path/'bundle.tar.gz',tmp_path/'out')
    expected=bundle.digest(tmp_path/'out/source/V6/research_lab/a.py')
    (tmp_path/'out/source/V6/research_lab/a.py').write_bytes(b'evil')
    with pytest.raises(ValueError):bundle.verify(tmp_path/'out')
    monkeypatch.setattr(bundle,'digest',lambda p:expected)
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(ValueError):bundle.verify(tmp_path/'out')


def test_run_entry_no_cuda_refuses_without_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):
        p4.run(tmp_path/'missing-bundle',tmp_path/'out',False,10,20,2,.1)
    assert not (tmp_path/'out').exists()


def test_p4_contract_rejects_draft_formal_and_recipe_drift():
    c=json.loads((bundle.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    p4.validate_contract(c,[2021],True)
    with pytest.raises(RuntimeError,match='DRAFT'):p4.validate_contract(dict(c,status='DRAFT'),list(range(2021,2027)),False)
    frozen=dict(c,status='FROZEN')
    p4.validate_contract(frozen,list(range(2021,2027)),False)
    with pytest.raises(ValueError,match='fold'):p4.validate_contract(frozen,[2021],False)
    with pytest.raises(ValueError,match='recipe'):p4.validate_contract(dict(frozen,recipe=dict(c['recipe'],window=59)),[2021],True)


def test_whole_plan_requires_reserve():
    # Single fit fits; the 30-fit grid does not. No partial-grid budget admission.
    p4.check_budget(10,20,0,0,.5,1,.25)
    with pytest.raises(p4.BudgetStop): p4.check_plan_budget(budget_contract(),0,0,30,.5)


@pytest.mark.parametrize('rate,planned',[(0,.1),(2,0),(-1,.1),(float('nan'),.1)])
def test_unknown_or_zero_budget_estimate_refused(rate,planned):
    with pytest.raises(ValueError):p4.validate_budget_plan(10,20,rate,planned,.25)


def test_cpu_orchestration_charges_setup_and_resumes_completed_fits(tmp_path,monkeypatch,capsys):
    """GPU/mount/fit boundary doubles; real packing, cache, ledger and publication."""
    import pandas as pd
    from test_seq_runner import build
    from research_lab.seq_runner import stock_index
    folds=tmp_path/'folds';fold=folds/'fold_2021';fold.mkdir(parents=True)
    build(fold);stock_index(fold)
    ref=tmp_path/'p1';ref.mkdir()
    pd.DataFrame({'fold':[2021],'system':['c0plus'],'score':[.1]}).to_parquet(ref/'rank_average.parquet')
    (ref/'manifest.json').write_text(json.dumps({'outputs':{'rank_average.parquet':bundle.digest(ref/'rank_average.parquet')}}))
    archive=tmp_path/'bundle.tar.gz'
    bundle.pack(folds,[2021],ref,bundle.ROOT/'environments/p4a-colab.requirements.lock.txt',archive)
    unpacked=tmp_path/'unpacked';bundle.extract(archive,unpacked)
    original_path=Path
    monkeypatch.setattr(p4,'Path',lambda x:tmp_path if x=='/content/drive/MyDrive' else original_path(x))
    monkeypatch.setattr(p4.os.path,'ismount',lambda _:True)
    monkeypatch.setattr(p4,'require_cuda',lambda:None)
    monkeypatch.setattr(p4,'environment',lambda _,**kwargs:dict(device='CPU_BOUNDARY_DOUBLE'))
    calls=[]
    def fake_train(directory,year,arm,seed,cfg,reference,observer):
        calls.append((year,arm,seed));observer.steps=[.001]*200
        observer.records=[dict(gpu='CPU_BOUNDARY_DOUBLE',temperature_c=60,clock_mhz=0,power_w=0)]
        return pd.DataFrame({'score':[.987654321]}),dict(best_val_ic5=.987654321,history=[dict(seconds=1,val_ic5=.987654321)]*2)
    monkeypatch.setattr(p4,'train_fit',fake_train)
    root=tmp_path/'drive-budget'
    first=p4.run(unpacked,root,True,10,20,2,.1)
    assert len(calls)==3 and len(first['jobs'])==3
    events=[json.loads(p.read_text()) for p in sorted(root.glob('event-*.json'))]
    assert events[0]['state']['active']['binding']['stage']=='verify_bundle_and_runtime'
    assert events[2]['state']['active']['binding']['stage']=='rankgauss_verify_or_build'
    assert first['budget']['spent_hours']>0
    second=p4.run(unpacked,root,True,10,20,2,.1)
    assert len(calls)==3 and len(second['jobs'])==3  # no re-fit
    assert second['budget']['spent_hours']>first['budget']['spent_hours']
    import re
    for path in root.rglob('*'):
        if path.is_file():
            text=path.read_text()
            assert not re.search(r'best_val_ic|val_ic|history|score|performance|research_conclusion|0\.987654321',text,re.I), path
    assert not re.search(r'ic|score|performance|0\.987654321',capsys.readouterr().out,re.I)



def test_bundle_runtime_bytecode_cache_refused(tmp_path):
    source=tmp_path/'a.py';source.write_text('x=1\n')
    bundle.write_archive(tmp_path/'b.tar.gz',{'source/V6/research_lab/a.py':source},{})
    root=tmp_path/'unpacked';bundle.extract(tmp_path/'b.tar.gz',root)
    cache=root/'source/V6/research_lab/__pycache__';cache.mkdir()
    (cache/'a.cpython-312.pyc').write_bytes(b'local generated bytecode')
    with pytest.raises(ValueError): bundle.verify(root,allow_derived=True)
    (cache/'extra.py').write_text('raise RuntimeError()')
    with pytest.raises(ValueError):bundle.verify(root,allow_derived=True)


def test_empty_completed_fit_rejected(tmp_path):
    with pytest.raises(ValueError): p4.complete_fit(tmp_path,{}, {}, [])
    assert not (tmp_path/'complete.json').exists()


@pytest.mark.parametrize('kind',['huge','sparse','file','total','count'])
def test_archive_resource_preflight_before_destination(tmp_path,monkeypatch,kind):
    import gzip
    # Huge header with no actual payload: interception proves rejection precedes writes.
    size=2**40 if kind=='huge' else 4
    member=tarfile.TarInfo('payload');member.size=size
    if kind in ('huge','sparse'):
        member.size=0
        member.pax_headers={'GNU.sparse.map':'0,0','GNU.sparse.size':str(size)}
    mf=json.dumps({'version':'p4-bundle-v1','files':{'payload':{'bytes':size,'sha256':'0'*64}}}).encode()
    manifest=tarfile.TarInfo('bundle-manifest.json');manifest.size=len(mf)
    # Put manifest first so a huge final header can be inspected without allocating 1 TiB.
    raw=manifest.tobuf()+mf+b'\0'*((-len(mf))%512)+member.tobuf(format=tarfile.PAX_FORMAT)+b'\0'*1024
    archive=tmp_path/'bomb.tar.gz';archive.write_bytes(gzip.compress(raw)+b'\0'*max(0,291-len(gzip.compress(raw))))
    if kind=='file': monkeypatch.setattr(bundle,'MAX_FILE_BYTES',3,raising=False)
    if kind=='total': monkeypatch.setattr(bundle,'MAX_TOTAL_BYTES',3,raising=False)
    if kind=='count': monkeypatch.setattr(bundle,'MAX_MEMBERS',1,raising=False)
    import shutil
    def abort(*a,**kw):raise RuntimeError('TEST_ABORT: extract reached payload copy')
    monkeypatch.setattr(shutil,'copyfileobj',abort)
    with pytest.raises(ValueError): bundle.extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_forged_timestamp_pyc_removed_before_import(tmp_path):
    import importlib.util, marshal, os, struct, sys
    root=tmp_path/'bundle';source=root/'source';source.mkdir(parents=True)
    target=source/'victim.py';target.write_text('value=1\n')
    cache=Path(importlib.util.cache_from_source(str(target)));cache.parent.mkdir()
    stat=target.stat()
    cache.write_bytes(importlib.util.MAGIC_NUMBER+struct.pack('<III',0,int(stat.st_mtime),stat.st_size)+marshal.dumps(compile('value=999','victim.py','exec')))
    bundle.prepare_source(root)
    spec=importlib.util.spec_from_file_location('p4_victim',target)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.value==1 and not cache.exists()
    assert sys.dont_write_bytecode and os.environ['PYTHONDONTWRITEBYTECODE']=='1'


def test_bundle_excludes_and_extract_rejects_executable_cache(tmp_path):
    source=tmp_path/'payload';source.write_bytes(b'evil')
    mf=bundle.write_archive(tmp_path/'a.tar.gz',{'source/__pycache__/a.pyc':source,'source/V6/research_lab/a.py':source},{})
    assert set(mf['files'])=={'source/V6/research_lab/a.py'}
    archive=tmp_path/'evil.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        m=tarfile.TarInfo('source/V6/research_lab/a.pyc');m.size=4;tf.addfile(m,io.BytesIO(b'evil'))
    with pytest.raises(ValueError):bundle.extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def formal_fixture(root, values=(.3,.2,.1,0.), job=None):
    import pandas as pd
    frozen=json.loads((bundle.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    job=job or p4.fit_queue(frozen)[0]
    cfg=dict(frozen['recipe'],window=job['window'],lr=job['lr'])
    c=dict(run_hash='synthetic',rankgauss='synthetic',speed_test=False,recipe=cfg,**job)
    table=pd.DataFrame(dict(fold=[job['year']],date=pd.to_datetime([f"{job['year']}-01-04"]),code=['A'],label_y5=[.1],score=[.2],
                            seed=[job['seed']],system=[job['arm']],window=[job['window']],lr=[job['lr']]))
    for name in ('scores','reference'):table.to_parquet(root/f'{name}.parquet',index=False)
    history=[dict(epoch=i,val_ic5=v) for i,v in enumerate(values,1)]
    (root/'metrics.json').write_text(json.dumps(dict(history=history,best_val_ic5=max(values),eval_rows=1)))
    entries=[];best=-9.;bad=0;best_entry=None
    for epoch,value in enumerate(values,1):
        improved=value>best
        if improved:best=value;bad=0
        else:bad+=1
        name=f'epoch-{epoch:03d}.pt'
        torch.save(dict(contract=c,state=dict(epoch=epoch,best=best,bad=bad,history=history[:epoch]),best_sd={'weight':torch.tensor([1.])},model={'weight':torch.tensor([1.])},opt={},sched={},rng_state={},torch_rng=torch.get_rng_state()),root/name)
        entry=dict(file=name,identity=bundle.digest(root/name),epoch=epoch);entries.append(entry)
        if improved:best_entry=entry
    pointer=dict(previous=entries[-2] if len(entries)>1 else None,best=best_entry,latest=entries[-1])
    (root/'checkpoint-pointer.json').write_text(json.dumps(pointer))
    torch.save(dict(contract=c,model={'weight':torch.tensor([1.])},best_val_ic5=max(values)),root/'best.pt')
    return c,[p.name for p in root.iterdir()]


@pytest.mark.parametrize('damage',['none','empty','pointer','missing','nan','rows','best','metrics','checkpoint','order'])
def test_completed_fit_required_schema(tmp_path,damage):
    import pandas as pd
    c,files=formal_fixture(tmp_path)
    if damage=='empty':files=[]
    if damage=='pointer':(tmp_path/'checkpoint-pointer.json').write_text('{}')
    if damage=='missing': files.remove('epoch-001.pt')
    if damage in ('nan','rows'):
        frame=pd.read_parquet(tmp_path/'scores.parquet')
        if damage=='nan':frame['score']=np.nan
        else:frame['code']='OTHER'
        frame.to_parquet(tmp_path/'scores.parquet',index=False)
    if damage=='best':torch.save(dict(contract=c,model={'weight':torch.tensor([99.])},best_val_ic5=.3),tmp_path/'best.pt')
    if damage=='checkpoint':
        path=tmp_path/'epoch-004.pt';payload=torch.load(path,weights_only=False);del payload['opt'];torch.save(payload,path)
        pointer=tmp_path/'checkpoint-pointer.json';data=json.loads(pointer.read_text());data['latest']['identity']=bundle.digest(path);pointer.write_text(json.dumps(data))
    if damage=='order':
        pointer=tmp_path/'checkpoint-pointer.json';data=json.loads(pointer.read_text());data['previous'],data['latest']=data['latest'],data['previous'];pointer.write_text(json.dumps(data))
    if damage=='metrics':(tmp_path/'metrics.json').write_text('{}')
    if damage=='none':
        p4.complete_fit(tmp_path,c,{},files)
        assert p4.resume_fit(tmp_path,c)
        # Rehashing a pointer change in the self-reported inventory must still fail.
        pointer=tmp_path/'checkpoint-pointer.json';pointer.write_text('{}')
        marker=tmp_path/'complete.json';mf=json.loads(marker.read_text());mf['outputs'][pointer.name]=bundle.digest(pointer);marker.write_text(json.dumps(mf))
        with pytest.raises(ValueError):p4.resume_fit(tmp_path,c)
    else:
        with pytest.raises((ValueError,KeyError)):p4.complete_fit(tmp_path,c,{},files)
        assert not (tmp_path/'complete.json').exists()


def test_train_fit_speed_suppresses_logger_predictions_and_checkpoint(tmp_path,monkeypatch,capsys):
    import pandas as pd
    from research_lab import seq_runner as sr
    from test_seq_runner import build,CFG
    build(tmp_path)
    data=sr.SeqData(tmp_path,window=60,normalize='rankgauss')
    # Isolate training/validation split boundary; execute the actual P4 train wrapper.
    rows=np.arange(3)
    monkeypatch.setattr(p4,'checked_splits',lambda *_:(rows,rows,rows))
    monkeypatch.setattr(p4,'require_same_rows',lambda *_:None)
    def fake_train(*args,log=print,observer=None,**kw):
        log('val_ic5 0.987654321 train_probe_ic5 0.987654321')
        observer('epoch',{'state':{'epoch':1,'best':.987654321}})
        return None,dict(best=.987654321,history=[dict(seconds=1,val_ic5=.987654321)]*2)
    monkeypatch.setattr(sr,'train_one',fake_train)
    monkeypatch.setattr(sr,'predict',lambda *_:pytest.fail('speed must not predict evaluation rows'))
    observer=p4.Observer(tmp_path/'output',{},None,speed=True)
    frame,job=p4.train_fit(tmp_path,2021,'gru',17,dict(CFG,val_max_rows=0),pd.DataFrame(),observer)
    assert frame is None and job==dict(history=[dict(seconds=1)]*2)
    assert 'ic5' not in capsys.readouterr().out
    assert not observer.root.exists()



def test_draft_documents_disclose_open_decisions_and_exposure():
    text=(bundle.ROOT/'tasks/spec-p4a-colab-20261121.md').read_text()
    for term in ('53,250','60,706','pre-norm','gated','dropout','Holm m=3','階層式','P3 結果已知','可確認的路徑','只能描述性的路徑'):
        assert term in text,term
    assert text.count('| OPEN |')==1 and 'FROZEN — 2026-11-22' in text


def test_operations_require_abi_stop_and_preimport_cleanup():
    text=(bundle.ROOT/'docs/operations/colab-p4.md').read_text()
    for term in ('ABI／版本衝突處理決策樹','python -c','torch.version.cuda','_GLIBCXX_USE_CXX11_ABI','不得自行升降級','p4.prepare_source(root)','UNVERIFIED'):
        assert term in text,term


def test_p4_knowledge_preserves_candidate_authority_status():
    for name in ('Current_State.md','Authority_Map.md','Worktree_Register.md'):
        text=(bundle.ROOT/'knowledge/00_Project_Map'/name).read_text().split('## P4a',1)[1].split('\n## ',1)[0]
        assert 'UNMERGED' in text and 'NOT_ACCEPTED' in text
        assert 'NOT_ACCEPTED' in text and 'NOT_DEPLOYED' in text


def _review_archive(tmp_path, files):
    import hashlib
    mf = {'version': 'p4-bundle-v1', 'files': {
        n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n, b in files.items()}}
    archive = tmp_path/'review.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for name, data in {**files, 'bundle-manifest.json': json.dumps(mf).encode()}.items():
            member = tarfile.TarInfo(name); member.size = len(data)
            tf.addfile(member, io.BytesIO(data))
    return archive


def test_review_pax_payload_bounded_before_parser(tmp_path, monkeypatch):
    import gzip
    header = tarfile.TarInfo('pax'); header.type = tarfile.XHDTYPE; header.size = 2**40
    archive = tmp_path/'pax.tar.gz'
    compressed = gzip.compress(header.tobuf(format=tarfile.GNU_FORMAT), mtime=0)
    archive.write_bytes(compressed + b'\0' * (81 - len(compressed)))
    assert archive.stat().st_size == 81
    requests = []
    original = gzip.GzipFile.read
    def guarded(stream, size=-1):
        requests.append(size)
        if size > 1024*1024 or size < 0:
            raise RuntimeError('REVIEW_SENTINEL oversized decompression read')
        return original(stream, size)
    monkeypatch.setattr(gzip.GzipFile, 'read', guarded)
    with pytest.raises(ValueError): bundle.extract(archive, tmp_path/'out')
    assert max(requests) <= 1024*1024 and not (tmp_path/'out').exists()
    print(f'PAX archive_bytes={archive.stat().st_size} requested_sizes={requests} destination_exists=False')


@pytest.mark.parametrize('relative', ['', 'source', 'source/V6/research_lab'])
def test_review_directory_swap_writes_no_outside_bytes(tmp_path, monkeypatch, relative):
    import os
    archive = _review_archive(tmp_path, {'source/V6/research_lab/a.py': b'payload'})
    outside = tmp_path/'outside'; outside.mkdir()
    victim = tmp_path/'out'/relative
    mkdir = os.mkdir
    swapped = []
    def swap(path, mode=0o777, *, dir_fd=None):
        result = mkdir(path, mode, dir_fd=dir_fd)
        if not swapped and victim.is_dir():
            victim.rmdir(); victim.symlink_to(outside, target_is_directory=True); swapped.append(True)
        return result
    monkeypatch.setattr(os, 'mkdir', swap)
    with pytest.raises((ValueError, OSError)): bundle.extract(archive, tmp_path/'out')
    assert swapped and not list(outside.rglob('*'))
    print('RACE outside_written=False')


@pytest.mark.parametrize('name', ['source/sitecustomize.py', 'source/V6/sitecustomize.py',
    'source/V6/research_lab/usercustomize.py', 'source/V6/research_lab/conftest.py',
    'source/V6/research_lab/setup.py', 'source/V6/research_lab/evil.so',
    'source/V6/research_lab/evil.pyd', 'source/V6/research_lab/evil.pth',
    'source/other/a.py', 'reference/sitecustomize.py', 'folds/fold_2021/evil.npy'])
def test_review_rejects_matching_manifest_startup_paths(tmp_path, name):
    archive = _review_archive(tmp_path, {name: b'raise RuntimeError("startup")'})
    with pytest.raises(ValueError): bundle.extract(archive, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_review_manifest_budget_applies_during_raw_preflight(tmp_path, monkeypatch):
    import gzip
    # Manifest first: a following physical payload cannot consume bytes beyond its
    # declared sum plus fixed metadata allowance, even before tarfile is invoked.
    manifest = json.dumps({'version': 'p4-bundle-v1', 'files': {}}).encode()
    m = tarfile.TarInfo('bundle-manifest.json'); m.size = len(manifest)
    extra = tarfile.TarInfo('source/V6/research_lab/a.py'); extra.size = 8192
    archive = tmp_path/'budget.tar.gz'
    archive.write_bytes(gzip.compress(m.tobuf()+manifest+b'\0'*((-len(manifest))%512)+extra.tobuf()+b'x'*8192+b'\0'*1024))
    monkeypatch.setattr(bundle, 'MAX_ARCHIVE_OVERHEAD', 2048)
    monkeypatch.setattr(bundle, 'MAX_MANIFEST_BYTES', 512)
    reads = []
    original = gzip.GzipFile.read
    def capture(stream, size=-1):
        data = original(stream, size); reads.append(len(data)); return data
    monkeypatch.setattr(gzip.GzipFile, 'read', capture)
    with pytest.raises(ValueError): bundle.extract(archive, tmp_path/'out')
    assert sum(reads) <= 2560
    assert not (tmp_path/'out').exists()


def test_extract_cli_optional_sha_records_and_checks(tmp_path, capsys):
    source=tmp_path/'a';source.write_bytes(b'original')
    archive=tmp_path/'a.tar.gz'
    bundle.write_archive(archive,{'source/V6/research_lab/a.py':source},{})
    bundle.main(['extract',str(archive),str(tmp_path/'out')])
    assert bundle.digest(archive)['sha256'] in capsys.readouterr().out
    with pytest.raises(ValueError,match='archive hash'):
        bundle.main(['extract',str(archive),str(tmp_path/'bad'),'--sha256','0'*64])
    (tmp_path/'out/source/V6/research_lab/a.py').write_bytes(b'changed!')
    with pytest.raises(ValueError,match='hash'):bundle.verify(tmp_path/'out')


def drive_wheels(tmp_path):
    d=tmp_path/'wheels';d.mkdir()
    for name in ('mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl',
                 'causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl'):
        (d/name).write_bytes(b'synthetic wheel identity only')
    return d


def test_drive_wheel_plan(tmp_path,monkeypatch,capsys):
    from packaging.tags import Tag
    import packaging.tags
    monkeypatch.setattr(bundle,'verify',lambda _:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(packaging.tags,'sys_tags',lambda:iter([Tag('cp312','cp312','linux_x86_64')]))
    d=drive_wheels(tmp_path)
    commands=bundle.installation_commands(tmp_path,wheel_dir=d)
    assert len(commands)==1 and all('pip check' not in command for command in commands)
    assert all(s in commands[0] for s in ('--no-deps','--no-index','--find-links',str(d)))
    assert 'torch' not in commands[0] and '-r ' not in commands[0]
    out=capsys.readouterr().out
    for wheel in d.iterdir():
        assert wheel.name in out and bundle.digest(wheel)['sha256'] in out
    assert 'cp312-cp312-linux_x86_64' in out


@pytest.mark.parametrize('failure',['python','platform','missing','ambiguous'])
def test_drive_wheel_plan_rejects(tmp_path,monkeypatch,failure):
    from packaging.tags import Tag
    import packaging.tags
    monkeypatch.setattr(bundle,'verify',lambda _:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    tag=Tag('cp311','cp311','linux_x86_64') if failure=='python' else Tag('cp312','cp312','win_amd64')
    if failure in ('missing','ambiguous'):tag=Tag('cp312','cp312','linux_x86_64')
    monkeypatch.setattr(packaging.tags,'sys_tags',lambda:iter([tag]))
    d=drive_wheels(tmp_path)
    if failure=='missing':next(d.glob('mamba*')).unlink()
    if failure=='ambiguous':(d/'mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl').write_bytes(b'other')
    with pytest.raises((RuntimeError,ValueError),match='wheel|Python|platform'):
        bundle.installation_commands(tmp_path,wheel_dir=d)


def runtime_fixture(tmp_path,monkeypatch):
    lock=tmp_path/'requirements.lock.txt';lock.write_text('torch==2.10.0+cu128\nmamba-ssm==2.3.2.post1\n')
    actual=dict(python='3.12.9',cuda='12.6',cxx11abi=True,platform='linux_x86_64',gpu='FAKE',
                packages={'torch':'2.6.0+cu126','mamba-ssm':'2.3.0',**{n:'1.0' for n in ('causal-conv1d','numpy','pandas','pyarrow','scipy','scikit-learn','einops','packaging')}})
    monkeypatch.setattr(p4,'require_cuda',lambda:None)
    monkeypatch.setattr(p4,'environment_snapshot',lambda names:actual.copy())
    contract=dict(version='p4-colab-runtime-v1',status='OBSERVED',observed_at='2026-11-21T00:00:00Z',
                  runtime={k:actual[k] for k in ('python','cuda','cxx11abi','platform')},packages=dict(actual['packages']))
    path=tmp_path/'runtime.json';path.write_text(json.dumps(contract))
    return lock,path,contract


def test_speed_environment_mismatch_and_formal_colab_identity(tmp_path,monkeypatch):
    lock,path,contract=runtime_fixture(tmp_path,monkeypatch)
    env=p4.environment(lock,runtime_contract=path,speed=True)
    assert {d['item'] for d in env['environment_mismatch']} >= {'lock:torch','lock:mamba-ssm'}
    assert '不可與正式環境比較' in env['comparability']
    formal=p4.environment(lock,runtime_contract=path)
    assert formal['environment_mismatch']==[] and len(formal['local_lock_mismatch'])==2
    contract['packages']['mamba-ssm']='9.9';path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='environment mismatch'):p4.environment(lock,runtime_contract=path)
    env=p4.environment(lock,runtime_contract=path,speed=True)
    assert 'contract:mamba-ssm' in {d['item'] for d in env['environment_mismatch']}


def test_tbd_contract_formal_stop_speed_records(tmp_path,monkeypatch):
    lock,path,contract=runtime_fixture(tmp_path,monkeypatch)
    contract.update(status='TBD',observed_at=None);path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='environment mismatch'):p4.environment(lock,runtime_contract=path)
    env=p4.environment(lock,runtime_contract=path,speed=True)
    assert 'contract:status' in {d['item'] for d in env['environment_mismatch']}
    monkeypatch.setattr(p4,'require_cuda',lambda:(_ for _ in ()).throw(RuntimeError('CUDA required')))
    with pytest.raises(RuntimeError,match='CUDA'):p4.environment(lock,runtime_contract=path,speed=True)


def test_speed_report_discloses_environment():
    env={'environment_mismatch':[{'item':'lock:mamba-ssm','expected':'2.3.2.post1','actual':'2.3.0'}],
         'comparability':'結果不可與正式環境比較'}
    report=p4.speed_report('gru',[{'seconds':1}]*2,[{'gpu':'fake'}],2,[.01]*200,environment=env)
    assert report['environment_mismatch']==env['environment_mismatch']
    assert report['comparability']==env['comparability']


def test_extract_without_external_sha_still_checks_internal_hash(tmp_path):
    source=tmp_path/'a';source.write_bytes(b'original')
    archive=tmp_path/'good.tar.gz'
    bundle.write_archive(archive,{'source/V6/research_lab/a.py':source},{})
    damaged=tmp_path/'damaged.tar.gz'
    with tarfile.open(archive,'r:gz') as old,tarfile.open(damaged,'w:gz') as new:
        for member in old:
            data=old.extractfile(member).read()
            if member.name.endswith('/a.py'):data=b'mutation'
            new.addfile(member,io.BytesIO(data))
    with pytest.raises(ValueError,match='bundle hash mismatch'):
        bundle.extract(damaged,tmp_path/'out')


@pytest.mark.parametrize('change',['python','cuda','cxx11abi','platform','missing_package','incomplete_contract'])
def test_formal_environment_rejects_runtime_and_package_drift(tmp_path,monkeypatch,change):
    lock,path,contract=runtime_fixture(tmp_path,monkeypatch)
    if change=='missing_package':contract['packages']['not-installed']='1.0'
    elif change=='incomplete_contract':contract['packages'].pop('causal-conv1d')
    else:contract['runtime'][change]='different'
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='environment mismatch'):
        p4.environment(lock,runtime_contract=path)
    assert p4.environment(lock,runtime_contract=path,speed=True)['environment_mismatch']


def test_drive_install_requires_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(bundle,'verify',lambda _:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):
        bundle.installation_commands(tmp_path,wheel_dir=drive_wheels(tmp_path))


def test_zip_dedup_symlink_roundtrip(tmp_path):
    import zipfile
    source=tmp_path/'X.npy';source.write_bytes(b'shared raw matrix\0'*100)
    link=tmp_path/'link.npy';link.symlink_to(source)
    names={f'folds/fold_{y}/X.npy':link for y in (2021,2022)}
    archive=tmp_path/'bundle.zip'
    mf=bundle.write_archive(archive,names,{})
    assert zipfile.is_zipfile(archive)
    with zipfile.ZipFile(archive) as z:
        assert sum(n.endswith('/X.npy') for n in z.namelist())==1
    assert mf['aliases']=={'folds/fold_2022/X.npy':'folds/fold_2021/X.npy'}
    bundle.extract(archive,tmp_path/'out')
    for n in names:
        assert (tmp_path/'out'/n).read_bytes()==source.read_bytes()
        assert not (tmp_path/'out'/n).is_symlink()


@pytest.mark.parametrize('name,mode',[('../escape',0o100600),('/absolute',0o100600),
    ('source/V6/research_lab/a.py',0o120777),('source/V6/research_lab/a.pyc',0o100600),
    ('unlisted',0o100600)])
def test_zip_rejects_hostile_before_destination(tmp_path,name,mode):
    import zipfile
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        info=zipfile.ZipInfo(name);info.external_attr=mode<<16
        z.writestr(info,b'bad')
    with pytest.raises(ValueError):bundle.extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_delivery_notebook_cpu(tmp_path):
    notebook=json.loads(json.dumps(bundle.colab_notebook()))
    assert notebook['nbformat']==4 and notebook['nbformat_minor']==5
    assert len({c['id'] for c in notebook['cells']})==len(notebook['cells'])
    assert notebook['cells'][0]['cell_type']=='code'
    settings={};exec(notebook['cells'][0]['source'],settings)
    assert not {'MAX_HOURS','MAX_UNITS','RESERVE'} & settings.keys()
    assert settings['MODE']=='FORMAL'
    import ast
    options=set()
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(cell['source'],'notebook','exec')
            for node in ast.walk(ast.parse(cell['source'])):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('--'):
                    options.add(node.value)
    assert options >= {'--bundle','--drive-root','--speed-test','--data-dir'}
    # Actual CLI parsing, with runner isolated from GPU/filesystem.
    from unittest.mock import patch
    with patch.object(p4,'run',return_value={'version':'test','jobs':[],'evidence_class':'ENGINEERING_SPEED_ONLY','binding':{'environment':{}}}):
        p4.main(['--bundle','x','--drive-root','y','--speed-test','--max-hours','6','--max-units','100','--units-per-hour','10','--planned-hours-per-fit','.5','--reserve','.25'])


def test_cp313_drive_wheel_positive_and_cp312_stop(tmp_path,monkeypatch):
    from packaging.tags import Tag
    import packaging.tags
    monkeypatch.setattr(bundle,'verify',lambda _:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(packaging.tags,'sys_tags',lambda:iter([Tag('cp313','cp313','linux_x86_64')]))
    directory=drive_wheels(tmp_path)
    with pytest.raises(RuntimeError,match='mismatch'):bundle.installation_commands(tmp_path,directory)
    for path in list(directory.iterdir()):path.rename(directory/path.name.replace('cp312','cp313'))
    assert '--no-index' in bundle.installation_commands(tmp_path,directory)[0]


def test_delivery_nbformat_parser():
    nbformat=pytest.importorskip('nbformat',reason='offline CPU runtime has no nbformat; schema/compile checks run separately')
    nbformat.validate(nbformat.from_dict(bundle.colab_notebook()))


@pytest.mark.parametrize('change',['alias_hash','alias_chain','missing','duplicate','oversize','count','hash'])
def test_zip_inventory_and_payload_gates(tmp_path,monkeypatch,change):
    import zipfile
    source=tmp_path/'X';source.write_bytes(b'original')
    archive=tmp_path/'good.zip'
    mf=bundle.write_archive(archive,{f'folds/fold_{y}/X.npy':source for y in (2021,2022)},{})
    if change=='alias_hash':mf['files']['folds/fold_2022/X.npy']['sha256']='0'*64
    if change=='alias_chain':mf['aliases']['folds/fold_2021/X.npy']='folds/fold_2022/X.npy'
    if change=='oversize':monkeypatch.setattr(bundle,'MAX_FILE_BYTES',4)
    if change=='count':monkeypatch.setattr(bundle,'MAX_MEMBERS',1)
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive) as old,zipfile.ZipFile(bad,'w') as new:
        for member in old.infolist():
            data=old.read(member)
            if member.filename=='bundle-manifest.json': data=json.dumps(mf).encode()
            elif change=='missing':continue
            elif change=='hash':data=b'mutation'
            new.writestr(member,data)
            if change=='duplicate' and member.filename!='bundle-manifest.json':new.writestr(member,data)
    with pytest.raises(ValueError):bundle.extract(bad,tmp_path/'out')
    if change!='hash':assert not (tmp_path/'out').exists()


def test_deliver_cli_synthetic_two_files(tmp_path):
    import zipfile
    folds=tmp_path/'folds'
    folds.mkdir()
    shared=folds/'shared.npy';shared.write_bytes(b'synthetic shared raw X')
    for year in (2021,2022):
        fold=folds/f'fold_{year}';fold.mkdir(parents=True)
        for name in bundle.MATRIX_FILES:
            if name=='X.npy':(fold/name).symlink_to(shared)
            else:(fold/name).write_bytes(b'{}' if name.endswith('.json') else str(year).encode())
    ref=tmp_path/'p1';ref.mkdir();(ref/'rank_average.parquet').write_bytes(b'synthetic P1 identity only')
    (ref/'manifest.json').write_text(json.dumps({'outputs':{'rank_average.parquet':bundle.digest(ref/'rank_average.parquet')}}))
    output=tmp_path/'delivery'
    args=['deliver','--folds-root',str(folds),'--years','2021','2022','--p1',str(ref),'--output-dir',str(output),'--run-name','02_trial','--drive-root','/custom/MyDrive/MarketMamba']
    bundle.main(args)
    assert {p.name for p in output.iterdir()}=={'p4a-bundle.zip','p4a-colab.ipynb','README.txt'}
    with pytest.raises(FileExistsError):bundle.main(args)
    with zipfile.ZipFile(output/'p4a-bundle.zip') as z:
        assert len([n for n in z.namelist() if n.endswith('/X.npy')])==1
        assert not any(n.endswith('.pyc') for n in z.namelist())
    bundle.extract(output/'p4a-bundle.zip',tmp_path/'out')
    for fold in folds.glob('fold_*'):
        for name in bundle.MATRIX_FILES:
            assert (tmp_path/'out/folds'/fold.name/name).read_bytes()==(fold/name).read_bytes()
    notebook=json.loads((output/'p4a-colab.ipynb').read_text())
    settings={};exec(notebook['cells'][0]['source'],settings)
    assert settings['RUN_DIR']=='/custom/MyDrive/MarketMamba/p4a/02_trial'
    assert settings['EXPERIMENT_DIR']=='/custom/MyDrive/MarketMamba/p4a'
    assert (output/'README.txt').read_text().splitlines()==[
        '上傳到 Drive 資料夾：/custom/MyDrive/MarketMamba/p4a/02_trial',
        '開啟檔案：p4a-colab.ipynb']
    assert settings['EXPECTED_ARCHIVE']==bundle.digest(output/'p4a-bundle.zip')
    # Execute only the stdlib bootstrap/extract cell with a local fake Colab mount.
    import sys,types
    fake=types.ModuleType('google.colab');fake.drive=types.SimpleNamespace(mount=lambda _:None)
    from unittest.mock import patch
    with patch.dict(sys.modules,{'google.colab':fake}), patch.object(sys,'path',list(sys.path)):
        namespace=dict(settings,RUN_DIR=str(output),EXPERIMENT_DIR=str(tmp_path),LOCAL_ROOT=str(tmp_path/'from-notebook'))
        exec(notebook['cells'][2]['source'],namespace)
        exec(notebook['cells'][2]['source'],namespace)  # same-root reconnect identity
    assert bundle.verify(tmp_path/'from-notebook')['metadata']['years']==[2021,2022]


def test_zip_symlink_with_valid_manifest_rejected(tmp_path):
    import zipfile
    data=b'/etc/passwd'
    name='source/V6/research_lab/a.py'
    import hashlib
    mf={'version':'p4-bundle-v1','files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
    archive=tmp_path/'linked.zip'
    with zipfile.ZipFile(archive,'w') as z:
        info=zipfile.ZipInfo(name);info.create_system=3;info.external_attr=0o120777<<16
        z.writestr(info,data);z.writestr('bundle-manifest.json',json.dumps(mf))
    with pytest.raises(ValueError,match='regular'):bundle.extract(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('scenario',['installed','missing_both','no_cuda'])
def test_notebook_environment_cpu_boundaries(tmp_path,monkeypatch,scenario):
    import importlib.metadata,platform,subprocess
    contract=json.loads((bundle.ROOT/'environments/p4a-colab-runtime.json').read_text())
    root=tmp_path/'bundle';path=root/'source/environments/p4a-colab-runtime.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(contract))
    (root/'bundle-manifest.json').write_text(json.dumps({'metadata':{}}))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:scenario!='no_cuda')
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'CPU_TEST_DOUBLE')
    monkeypatch.setattr(torch.version,'cuda',contract['runtime']['cuda'])
    monkeypatch.setattr(platform,'python_version',lambda:contract['runtime']['python'])
    monkeypatch.setattr(importlib.metadata,'version',lambda name: contract['packages'][name] if scenario=='installed' else 'missing')
    calls=[];monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append(args))
    namespace=dict(Path=Path,root=root,json=json,subprocess=subprocess,p4=bundle,WHEEL_DIR=str(tmp_path/'missing'),LEGACY_WHEEL_DIR=str(tmp_path/'legacy'),resolve_wheel_dir=notebook_helpers()['resolve_wheel_dir'])
    cell=bundle.colab_notebook()['cells'][3]['source']
    if scenario=='no_cuda':
        with pytest.raises(RuntimeError,match='CUDA'):exec(cell,namespace)
        assert calls==[]
    else:
        exec(cell,namespace)
        assert namespace['needs_compile']==(scenario=='missing_both')
        assert calls==[['nvidia-smi']]  # matching installed runtime never invokes pip


def notebook_helpers():
    """Execute generated stdlib helper definitions without mounting Drive."""
    import ast, hashlib, zipfile
    tree=ast.parse(bundle.colab_notebook()['cells'][2]['source'])
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
    namespace=dict(Path=Path,hashlib=hashlib,json=json,zipfile=zipfile)
    exec(compile(functions,'notebook helpers','exec'),namespace)
    return namespace


def test_drive_settings_shared_budget_and_formal_stop():
    nb=bundle.colab_notebook()
    settings={};exec(nb['cells'][0]['source'],settings)
    assert settings['MODE']=='FORMAL' and not {'MAX_HOURS','MAX_UNITS','RESERVE'} & settings.keys()
    assert Path(settings['RUN_DIR'])==Path('/content/drive/MyDrive/MarketMamba/p4a/02_formal')
    assert Path(settings['EXPERIMENT_DIR'])==Path(settings['RUN_DIR']).parent
    assert settings['WHEEL_DIR']=='/content/drive/MyDrive/MarketMamba/wheels'
    formal=nb['cells'][0]['source']
    exec(formal,{})
    with pytest.raises(ValueError,match='STOP'):exec(formal.replace("MODE = 'FORMAL'","MODE = 'INVALID'"),{})
    commands=[]
    for run in ('01_speedtest','02_formal'):
        namespace=dict(settings,MODE='SPEED_TEST' if run=='01_speedtest' else 'FORMAL',
                       RUN_DIR='/drive/p4a/'+run,EXPERIMENT_DIR='/drive/p4a',
                       root=Path('/bundle'),p4=type('P4',(),{'prepare_source':lambda _:None,'verify':lambda *a,**k:None}),
                       os=__import__('os'),sys=__import__('sys'),subprocess=type('S',(),{'run':lambda args,**kw:commands.append(args)}))
        exec(nb['cells'][6]['source'],namespace)
    assert [c[c.index('--drive-root')+1] for c in commands]==['/drive/p4a/budget']*2


def test_drive_archive_search_and_stale_content(tmp_path,capsys):
    helpers=notebook_helpers();select=helpers['find_archive']
    run=tmp_path/'p4a/01_speedtest';run.mkdir(parents=True)
    inputs=run.parent/'inputs';inputs.mkdir();content=tmp_path/'content';content.mkdir()
    paths=[run/'p4a-bundle.zip',inputs/'p4a-bundle.zip',content/'p4a-bundle.zip']
    for p in paths:p.write_bytes(b'expected')
    expected=bundle.digest(paths[0])
    assert select(run,run.parent,content,paths[0].name,expected)==paths[0]
    paths[2].write_bytes(b'incomplete')
    assert select(run,run.parent,content,paths[0].name,expected)==paths[0]
    select(run,run.parent,content,paths[0].name,expected)
    warning=capsys.readouterr().out
    assert 'WARNING' in warning and str(paths[2]) in warning
    paths[0].unlink()
    assert select(run,run.parent,content,paths[0].name,expected)==paths[1]
    paths[1].unlink()
    with pytest.raises(FileNotFoundError,match='STOP') as error:select(run,run.parent,content,paths[0].name,expected)
    assert all(str(p) in str(error.value) for p in paths)
    paths[2].write_bytes(b'expected')
    assert select(run,run.parent,content,paths[0].name,expected)==paths[2]


def test_drive_wheel_fallback(tmp_path):
    resolve=notebook_helpers()['resolve_wheel_dir']
    preferred=tmp_path/'MarketMamba/wheels';legacy=tmp_path/'Mamba_Wheels_T4_new'
    assert resolve(preferred,legacy)==preferred  # compile destination
    preferred.mkdir(parents=True);legacy.mkdir()
    (legacy/'mamba_ssm-test.whl').touch()
    assert resolve(preferred,legacy)==legacy
    (preferred/'mamba_ssm-test.whl').touch()
    assert resolve(preferred,legacy)==preferred  # partial preferred stays STOP at installer


@pytest.mark.parametrize('change',['truncated','manifest_total','member_size'])
def test_drive_zip_size_stop_before_extraction(tmp_path,change):
    import zipfile
    helpers=notebook_helpers()
    source=tmp_path/'x';source.write_bytes(b'hello')
    archive=tmp_path/'bundle.zip'
    mf=bundle.write_archive(archive,{'folds/fold_2021/X.npy':source},{})
    expected=bundle.digest(archive)
    helpers['validate_archive_size'](archive,expected)
    assert mf['total_bytes']==5
    if change=='truncated':archive.write_bytes(archive.read_bytes()[:-10])
    else:
        with zipfile.ZipFile(archive) as z:items={m.filename:z.read(m) for m in z.infolist()}
        if change=='manifest_total':mf['total_bytes']+=1
        else:mf['files']['folds/fold_2021/X.npy']['bytes']+=1
        items['bundle-manifest.json']=json.dumps(mf).encode()
        with zipfile.ZipFile(archive,'w') as z:
            for name,data in items.items():z.writestr(name,data)
        expected=bundle.digest(archive)  # isolate manifest check from outer identity
    with pytest.raises(ValueError,match='STOP.*(大小|size).*上傳不完整'):
        helpers['validate_archive_size'](archive,expected)
