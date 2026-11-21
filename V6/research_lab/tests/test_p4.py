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
    bundle.write_archive(archive, {'source/a':source}, {'source_commit':'abc'})
    bundle.extract(archive,tmp_path/'out')
    assert (tmp_path/'out/source/a').read_bytes()==b'original'
    (tmp_path/'out/source/a').write_bytes(b'mutated!')
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
    p4.publish_bytes(tmp_path/'weights.pt',b'weights')
    p4.complete_fit(tmp_path,contract,{'epochs':2},['weights.pt'])
    assert p4.resume_fit(tmp_path,contract)['job']['epochs']==2
    with pytest.raises(ValueError,match='identity'): p4.resume_fit(tmp_path,{'matrix':'abc','seed':29})
    (tmp_path/'weights.pt').write_bytes(b'bad')
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
    bundle.write_archive(tmp_path/'bundle.tar.gz',{'source/a':original},{})
    bundle.extract(tmp_path/'bundle.tar.gz',tmp_path/'out')
    expected=bundle.digest(tmp_path/'out/source/a')
    (tmp_path/'out/source/a').write_bytes(b'evil')
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
    with pytest.raises(RuntimeError,match='DRAFT'):p4.validate_contract(c,[2021,2022,2023,2024,2025],False)
    frozen=dict(c,status='FROZEN')
    p4.validate_contract(frozen,[2021,2022,2023,2024,2025],False)
    with pytest.raises(ValueError,match='fold'):p4.validate_contract(frozen,[2021],False)
    with pytest.raises(ValueError,match='recipe'):p4.validate_contract(dict(frozen,recipe=dict(c['recipe'],window=59)),[2021],True)


def test_whole_plan_requires_reserve():
    # Single fit fits; the 30-fit grid does not. No partial-grid budget admission.
    p4.check_budget(10,20,0,0,.5,1,.25)
    with pytest.raises(p4.BudgetStop): p4.check_plan_budget(budget_contract(),0,0,30,.5)


@pytest.mark.parametrize('rate,planned',[(0,.1),(2,0),(-1,.1),(float('nan'),.1)])
def test_unknown_or_zero_budget_estimate_refused(rate,planned):
    with pytest.raises(ValueError):p4.validate_budget_plan(10,20,rate,planned,.25)


def test_cpu_orchestration_charges_setup_and_resumes_completed_fits(tmp_path,monkeypatch):
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
    monkeypatch.setattr(p4,'environment',lambda _:dict(device='CPU_BOUNDARY_DOUBLE'))
    calls=[]
    def fake_train(directory,year,arm,seed,cfg,reference,observer):
        calls.append((year,arm,seed));observer.steps=[.001]*200
        observer.records=[dict(gpu='CPU_BOUNDARY_DOUBLE',temperature_c=60,clock_mhz=0,power_w=0)]
        p4.publish_bytes(observer.root/'best.pt',b'CPU_TEST_ONLY')
        return pd.DataFrame({'score':[.1]}),dict(history=[dict(seconds=1),dict(seconds=1)])
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


def test_bundle_runtime_bytecode_cache_allowed_but_extra_source_refused(tmp_path):
    source=tmp_path/'a.py';source.write_text('x=1\n')
    bundle.write_archive(tmp_path/'b.tar.gz',{'source/V6/research_lab/a.py':source},{})
    root=tmp_path/'unpacked';bundle.extract(tmp_path/'b.tar.gz',root)
    cache=root/'source/V6/research_lab/__pycache__';cache.mkdir()
    (cache/'a.cpython-312.pyc').write_bytes(b'local generated bytecode')
    bundle.verify(root,allow_derived=True)
    (cache/'extra.py').write_text('raise RuntimeError()')
    with pytest.raises(ValueError,match='inventory'):bundle.verify(root,allow_derived=True)
