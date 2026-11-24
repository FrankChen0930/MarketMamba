"""Source upgrade evidence: offline git objects and CPU only."""
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
import torch
from research_lab import p4_bundle as b, p4_colab as p, seq_runner as sr

OLD='4e56e9d'

def old_module(tmp, name):
    path=tmp/(name+'.py')
    path.write_bytes(subprocess.check_output(['git','show',f'{OLD}:V6/research_lab/{name}.py'],cwd=b.ROOT))
    spec=importlib.util.spec_from_file_location('research_lab._old_'+name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def setup_upgrade(tmp):
    from research_lab import p4_source_compat as c
    contract=json.loads((b.ROOT/c.CONTRACT).read_text())
    binding=dict(bundle={'bytes':1,'sha256':'new'},matrix={'bytes':2,'sha256':'matrix'},environment={'cpu':'test'},contract=contract,queue=p.fit_queue(contract),speed_test=False)
    files=c.current_sources()
    mf=dict(files={**files,'requirements.lock.txt':b.digest(b.ROOT/c.LOCK)},metadata={})
    mf['files']['DATASET_MANIFEST.json']=binding['matrix']
    decl=c.declaration(OLD,files,binding['matrix'])
    bundle=tmp/'bundle';bundle.mkdir();(bundle/'source/tasks').mkdir(parents=True)
    (bundle/('source/'+c.CONTRACT)).write_bytes((b.ROOT/c.CONTRACT).read_bytes())
    name='source/tasks/p4a-source-compat-test.json'
    path=bundle/name;path.write_text(json.dumps(decl));mf['files'][name]=b.digest(path)
    old_files=decl['from']['sources']
    old_mf=dict(files={**old_files,'requirements.lock.txt':mf['files']['requirements.lock.txt'],'DATASET_MANIFEST.json':binding['matrix']})
    payload=json.dumps(old_mf).encode()
    old_binding=dict(binding,bundle=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    root=tmp/'budget';out=root/('run-'+p.hashed(old_binding));out.mkdir(parents=True)
    (out/'run_identity.json').write_text(json.dumps(old_binding))
    from test_p4 import formal_fixture
    for job in binding['queue'][:36]:
        fit=out/f"{job['queue']}-{job['year']}-{job['arm']}-{job['seed']}-w{job['window']}-lr{job['lr']}"
        attempt=fit/'attempt-active';attempt.mkdir(parents=True)
        fc,files=formal_fixture(attempt,job=job)
        fc['run_hash']=p.hashed(old_binding)
        for checkpoint in attempt.glob('*.pt'):
            checkpoint_payload=torch.load(checkpoint,map_location='cpu',weights_only=False);checkpoint_payload['contract']=fc;torch.save(checkpoint_payload,checkpoint)
        pointer=json.loads((attempt/'checkpoint-pointer.json').read_text())
        for entry in pointer.values():
            if entry:entry['identity']=b.digest(attempt/entry['file'])
        (attempt/'checkpoint-pointer.json').write_text(json.dumps(pointer))
        p.complete_fit(fit,fc,{},['attempt-active/'+f for f in files])
    delivery=tmp/'02_formal';delivery.mkdir()
    with zipfile.ZipFile(delivery/'p4a-bundle.zip','w') as z:z.writestr('bundle-manifest.json',payload)
    return c,bundle,mf,binding,root,out,decl,name


def test_adopt_and_audit_identity(tmp_path,monkeypatch):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    checked=[];original_resume=p.resume_fit
    def resume(*args):
        checked.append(args[0]);return original_resume(*args)
    monkeypatch.setattr(p,'resume_fit',resume)
    chosen,event=c.resolve(bundle,mf,binding,root,dict(policy='record_only',units_per_hour=0,thermal_stop_c=85))
    assert len(checked)==36 and len(set(checked))==36
    assert chosen==json.loads((out/'run_identity.json').read_text())
    assert p.hashed(chosen)==out.name[4:]
    assert event['declaration']==b.digest(bundle/name)
    assert event['from_sources_hash']==p.hashed(decl['from']['sources'])
    assert event['to_sources_hash']==p.hashed(decl['to']['sources'])
    assert event['environment']==binding['environment'] and event['observed_at']


@pytest.mark.parametrize('damage',['missing','hash','extra','contract','queue','matrix','environment','budget','old_source'])
def test_upgrade_rejects_mismatch(tmp_path,damage):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    if damage=='missing':del mf['files'][name]
    elif damage=='hash':(bundle/name).write_text(json.dumps(dict(decl,evidence_class='tampered')))
    elif damage=='extra':mf['files']['source/V6/research_lab/extra.py']={'bytes':1,'sha256':'x'}
    elif damage in ('contract','queue','matrix','environment'):binding[damage]={}
    elif damage=='budget':decl['budget']['policy']='capped'
    elif damage=='old_source':decl['from']['sources']['source/V6/research_lab/seq_runner.py']['sha256']='bad'
    if damage in ('budget','old_source'):
        (bundle/name).write_text(json.dumps(decl));mf['files'][name]=b.digest(bundle/name)
    with pytest.raises(ValueError,match='STOP.*下一步'):
        c.resolve(bundle,mf,binding,root,dict(policy='record_only',units_per_hour=0,thermal_stop_c=85))


def test_no_old_run_is_new_and_explicit_override(tmp_path):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    assert c.resolve(bundle,dict(files={}),binding,root,{},allow_new=True)==(binding,None)
    import shutil
    shutil.rmtree(out)
    assert c.resolve(bundle,dict(files={}),binding,root,{})==(binding,None)


@pytest.mark.parametrize('kind',['gru','crossstock'])
def test_old_head_training_rng_and_epoch_resume(tmp_path,monkeypatch,kind):
    from test_seq_runner import build,CFG,CausalConvMixer
    old=old_module(tmp_path,'seq_runner');op=old_module(tmp_path,'p4_colab')
    torch.set_num_threads(1);matrix=tmp_path/'matrix';matrix.mkdir();build(matrix)
    data=sr.SeqData(matrix);tr,va,ev=data.split_rows(15)
    cfg=dict(CFG,epochs=4,steps_per_epoch=51,d_model=8,n_layers=2,patience=9,dropout=.2)
    kw=dict(device='cpu',mixer_factory=CausalConvMixer,log=lambda *_:None)
    def training(mod,**extra):
        np.random.seed(91);random.seed(91);python_rng=random.getstate()
        captures=[]
        def observe(event,payload):
            if event=='epoch':captures.append(copy.deepcopy(payload))
            if extra.get('observer'):extra['observer'](event,payload)
        model,state=mod.train_one(kind,17,data,tr,va,cfg,observer=observe,resume_state=extra.get('resume_state'),**kw)
        assert random.getstate()==python_rng
        return model,copy.deepcopy(captures[-1]),torch.get_rng_state().clone(),np.random.get_state()
    full,a,torch_a,np_a=training(old)
    head,z,torch_z,np_z=training(sr)
    def equal(x,y):
        if isinstance(x,torch.Tensor):assert x.numpy().tobytes()==y.numpy().tobytes()
        elif isinstance(x,np.ndarray):assert x.tobytes()==y.tobytes()
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            for v,w in zip(x,y,strict=True):equal(v,w)
        else:assert x==y
    equal(full.state_dict(),head.state_dict());equal(sr.predict(full,data,ev,'cpu'),sr.predict(head,data,ev,'cpu'))
    for key in ('rng_state','numpy_rng','torch_rng'):equal(a[key],z[key])
    equal(torch_a,torch_z);equal(np_a,np_z)
    class Ledger:
        def check(self):pass
    monkeypatch.setattr(op,'telemetry',lambda:dict(gpu='CPU',temperature_c=40,clock_mhz=0,power_w=0))
    contract=dict(seed=17,cfg=cfg,run_hash='old')
    obs=op.Observer(tmp_path/'ckpt',contract,Ledger(),device='cpu')
    def kill(event,payload):
        obs(event,payload)
        if event=='epoch' and payload['state']['epoch']==2:raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):training(old,observer=kill)
    pointer=json.loads((tmp_path/'ckpt/checkpoint-pointer.json').read_text())
    assert all(pointer[k] for k in ('latest','previous','best'))
    restored=p.Observer(tmp_path/'ckpt',contract,Ledger(),device='cpu').load_resume()
    assert restored['state']['epoch']==2 and restored['cuda_rng'] is None
    resumed,r,trng,nrng=training(sr,resume_state=restored)
    equal(full.state_dict(),resumed.state_dict());equal(sr.predict(full,data,ev,'cpu'),sr.predict(resumed,data,ev,'cpu'))
    for key in ('rng_state','numpy_rng','torch_rng'):equal(a[key],r[key])
    equal(torch_a,trng);equal(np_a,nrng)


def test_old_completed_fit_accepted_without_training(tmp_path,monkeypatch):
    from test_p4 import formal_fixture
    op=old_module(tmp_path,'p4_colab')
    monkeypatch.setattr(op,'__file__',p.__file__)  # unchanged frozen contract location
    fit=tmp_path/'fit';attempt=fit/'attempt';attempt.mkdir(parents=True)
    contract,files=formal_fixture(attempt)
    op.complete_fit(fit,contract,{},['attempt/'+f for f in files])
    monkeypatch.setattr(p,'train_fit',lambda *a,**k:pytest.fail('completed fit retrained'))
    before={str(f):b.digest(f) for f in fit.rglob('*') if f.is_file()}
    assert p.resume_fit(fit,contract)==op.resume_fit(fit,contract)
    assert before=={str(f):b.digest(f) for f in fit.rglob('*') if f.is_file()}


def test_delivery_includes_generated_declaration(tmp_path,monkeypatch):
    import pandas as pd
    from test_p4_full import dataset
    from research_lab import p4_source_compat as c
    matrix=dataset(tmp_path/'matrix');b.dataset_manifest(matrix,enforce_x=False)
    monkeypatch.setattr(b,'DATASET_X',b.digest(matrix/'shared/X.npy'))
    ref=tmp_path/'ref';ref.mkdir()
    pd.DataFrame(dict(fold=[2021],date=pd.to_datetime(['2021-01-01']),code=['A'],label_y5=[.1],score=[.2],system=['c0plus'])).to_parquet(ref/'rank_average.parquet')
    (ref/'manifest.json').write_text(json.dumps(dict(outputs={'rank_average.parquet':b.digest(ref/'rank_average.parquet')})))
    output=tmp_path/'delivery'
    b.main(['deliver','--folds-root',str(matrix),'--p1',str(ref),'--output-dir',str(output),'--resume-from-commit',OLD])
    with zipfile.ZipFile(output/'p4a-bundle.zip') as z:
        name=next(n for n in z.namelist() if n.startswith('source/tasks/p4a-source-compat-'))
        decl=json.loads(z.read(name));mf=json.loads(z.read('bundle-manifest.json'))
        assert b.allowed_path(name) is None
        assert c.identity(z.read(name))==mf['files'][name]
        assert decl['from']['sources']['source/V6/research_lab/seq_runner.py']==c.identity(c.git_bytes(OLD,'V6/research_lab/seq_runner.py'))
        assert decl['to']['sources']=={n:i for n,i in mf['files'].items() if c.source_name(n)}
        assert decl['queue_sha256']==c.QUEUE_SHA and decl['matrix']==b.digest(matrix/'DATASET_MANIFEST.json')
        assert {x['path'] for x in decl['differences']}=={n for n in decl['from']['sources'].keys()|decl['to']['sources'].keys() if decl['from']['sources'].get(n)!=decl['to']['sources'].get(n)}
    nb=json.loads((output/'p4a-colab.ipynb').read_text())
    assert '03_formal_resume' in nb['cells'][0]['source'] and 'EXPERIMENT_DIR = str(Path(RUN_DIR).parent)' in nb['cells'][0]['source']
    assert 'ALLOW_NEW_RUN = False' in nb['cells'][0]['source']
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'resume notebook','exec')
    nbformat = pytest.importorskip('nbformat', reason='offline CPU runtime has no nbformat; notebook schema checked separately')
    nbformat.validate(nbformat.read(output/'p4a-colab.ipynb',as_version=4))


def test_run_old_36_fits_then_upgrade_starts_only_first_q3(tmp_path,monkeypatch):
    """Real old run/ledger/publication, CPU artifact double; no GPU claims."""
    import io
    import pandas as pd
    from research_lab import p4_source_compat as c
    op=old_module(tmp_path,'p4_colab')
    to=c.current_sources();decl=c.declaration(OLD,to,None)
    ref=tmp_path/'reference.parquet'
    frame=pd.DataFrame([dict(fold=y,date=pd.Timestamp(f'{y}-02-01'),code=code,label_y5=np.nan if i==1 else float(i),score=float(i),system='c0plus') for y in range(2021,2027) for i,code in enumerate(('A','B'))])
    frame.to_parquet(ref)
    refmanifest=tmp_path/'refmanifest.json';refmanifest.write_text(json.dumps(dict(outputs={'rank_average.parquet':b.digest(ref)})))
    common={'reference/rank_average.parquet':ref,'reference/manifest.json':refmanifest,'requirements.lock.txt':b.ROOT/c.LOCK}
    def make_bundle(label,sources,upgrade=False):
        staging=tmp_path/(label+'-staging');staging.mkdir()
        files=dict(common)
        for n in sources:
            path=staging/n;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((b.ROOT/n.removeprefix('source/')).read_bytes() if upgrade else c.git_bytes(OLD,n.removeprefix('source/')))
            files[n]=path
        if upgrade:
            name='source/tasks/p4a-source-compat-test.json';path=staging/name;path.write_text(json.dumps(decl));files[name]=path
        delivery=tmp_path/label;delivery.mkdir()
        b.write_archive(delivery/'p4a-bundle.zip',files,dict(years=list(range(2021,2027)),source_commit=OLD if not upgrade else 'HEAD'))
        bundle=tmp_path/(label+'-bundle');b.extract(delivery/'p4a-bundle.zip',bundle)
        return bundle
    oldbundle=make_bundle('02_formal',decl['from']['sources'])
    newbundle=make_bundle('03_formal_resume',to,True)
    actual_path=Path
    for mod,bundle in ((op,oldbundle),(p,newbundle)):
        monkeypatch.setattr(mod,'__file__',str(bundle/'source/V6/research_lab/p4_colab.py'))
        monkeypatch.setattr(mod,'Path',lambda x:tmp_path if x=='/content/drive/MyDrive' else actual_path(x))
        monkeypatch.setattr(mod,'require_cuda',lambda:None)
        monkeypatch.setattr(mod,'environment',lambda *a,**k:dict(device='CPU_TEST_DOUBLE'))
        monkeypatch.setattr(mod,'rankgauss',lambda *a:dict(bytes=1,sha256='synthetic'))
    monkeypatch.setattr(p.os.path,'ismount',lambda _:True)
    calls=[];disconnect=[True]
    def fake_train(directory,year,arm,seed,cfg,reference,observer):
        calls.append((year,arm,seed,cfg['window'],cfg['lr']))
        if disconnect[0] and len(calls)==37:
            disconnect[0]=False
            raise RuntimeError('synthetic disconnection after 36 completed fits')
        observer.device='cpu';weights={'weight':torch.tensor([1.])}
        history=[dict(epoch=i,val_ic5=.3 if i==1 else .3-i*.1) for i in range(1,5)]
        for epoch in range(1,5):
            observer('epoch',dict(model=weights,opt={},sched={},state=dict(epoch=epoch,best=.3,bad=epoch-1,history=history[:epoch]),best_sd=weights,rng_state={},numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()))
        stream=io.BytesIO();torch.save(dict(contract=observer.contract,model=weights,best_val_ic5=.3),stream);p.publish_bytes(observer.root/'best.pt',stream.getvalue())
        return reference.assign(score=1.,seed=seed,system=arm,window=cfg['window'],lr=cfg['lr']),dict(history=history,best_val_ic5=.3,eval_rows=len(reference))
    monkeypatch.setattr(op,'train_fit',fake_train)
    root=tmp_path/'budget'
    with pytest.raises(RuntimeError,match='disconnection'):op.run(oldbundle,root)
    out=next(root.glob('run-*'))
    oldidentity=(out/'run_identity.json').read_bytes()
    originals={str(f):b.digest(f) for marker in out.glob('*/complete.json') for f in marker.parent.rglob('*') if f.is_file()}
    assert len(list(out.glob('*/complete.json')))==36
    completed_calls=calls[:36];calls.clear()
    monkeypatch.setattr(p,'train_fit',fake_train)
    monkeypatch.setattr(p,'permutation_fit',lambda *a,**k:dict(CPU_TEST_DOUBLE=True))
    # New run reaches Q3 with a tail NaN, then stops before its second fit.
    # Both immutable bundles already carry tail NaNs (see construction).
    class FirstQ3Done(RuntimeError):pass
    def first_q3(*args):
        if calls:raise FirstQ3Done('only Q3 first fit requested')
        assert args[4]['window']==120
        from research_lab.p4_eval import common_rows
        common_rows({'model':args[5],'reference':args[5]})
        return fake_train(*args)
    monkeypatch.setattr(p,'train_fit',first_q3)
    with pytest.raises(FirstQ3Done):p.run(newbundle,root)
    assert len(calls)==1 and calls[0]==(2021,'mamba2',17,120,3e-4)
    assert len(list(out.glob('*/complete.json')))==37
    assert all(b.digest(Path(f))==ident for f,ident in originals.items())
    assert (out/'run_identity.json').read_bytes()==oldidentity
    # Complete remaining jobs, verifying cached first Q3 is not retrained.
    monkeypatch.setattr(p,'train_fit',fake_train)
    result=p.run(newbundle,root)
    assert len(calls)==28 and not set(completed_calls)&set(calls)
    assert len(result['jobs'])==64 and (out/'run_identity.json').read_bytes()==oldidentity
    assert all(b.digest(Path(f))==ident for f,ident in originals.items())
    assert list(root.glob('run-*'))==[out]
    audits=sorted(out.glob('source-upgrade-*.json'));assert len(audits)==2
    event=json.loads(audits[0].read_text());assert event['event']=='SOURCE_UPGRADE'
    ledger_events=[json.loads(f.read_text()) for f in root.glob('event-*.json')]
    recorded=next(e for e in ledger_events if e['event']=='SOURCE_UPGRADE')
    assert recorded['state']['source_upgrade']['identity'] in [b.digest(a) for a in audits]
    assert recorded['state']['source_upgrade']['from_sources_hash']==event['from_sources_hash']
    assert result['binding']==json.loads(oldidentity)
    monkeypatch.setattr(p,'train_fit',lambda *a,**k:pytest.fail('cache retrained on second upgrade invocation'))
    assert len(p.run(newbundle,root)['jobs'])==64
    monkeypatch.setattr(op,'train_fit',lambda *a,**k:pytest.fail('old fallback notebook retrained'))
    assert len(op.run(oldbundle,root)['jobs'])==64  # original notebook remains usable


@pytest.mark.parametrize('damage',['identity_missing','declaration_missing_on_disk','multiple_lineages','difference_extra','difference_missing','unknown_category','invariant_hash','non_source_input','old_manifest_missing'])
def test_additional_upgrade_fail_closed(tmp_path,damage):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    if damage=='identity_missing':(out/'run_identity.json').unlink()
    elif damage=='declaration_missing_on_disk':(bundle/name).unlink()
    elif damage=='multiple_lineages':
        other=json.loads((out/'run_identity.json').read_text());other['environment']={'changed':True}
        extra=root/('run-'+p.hashed(other));extra.mkdir();(extra/'run_identity.json').write_text(json.dumps(other))
    elif damage=='difference_extra':decl['differences'].append(dict(decl['differences'][0],path='source/extra.py'))
    elif damage=='difference_missing':decl['differences'].pop()
    elif damage=='unknown_category':decl['differences'][0]['categories']=['optimizer change']
    elif damage=='invariant_hash':decl['invariants'][c.RUNTIME]['sha256']='changed'
    elif damage=='non_source_input':mf['files']['reference/keys.parquet']={'sha256':'new','bytes':1}
    elif damage=='old_manifest_missing':(root.parent/'02_formal/p4a-bundle.zip').unlink()
    if damage in ('difference_extra','difference_missing','unknown_category','invariant_hash'):
        (bundle/name).write_text(json.dumps(decl));mf['files'][name]=b.digest(bundle/name)
    with pytest.raises(ValueError,match='STOP.*下一步'):c.resolve(bundle,mf,binding,root,c.BUDGET)


def test_generator_rejects_non_neutral_changes(tmp_path,monkeypatch):
    from research_lab import p4_source_compat as c
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'git_bytes',lambda commit,path:subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=b.ROOT))
    folder=tmp_path/'V6/research_lab';folder.mkdir(parents=True)
    source=(b.ROOT/'V6/research_lab/p4_colab.py').read_text().replace("if mf['contract']!=contract or mf['contract_hash']!=hashed(contract)","if False")
    (folder/'p4_colab.py').write_text(source)
    with pytest.raises(ValueError,match='semantics changed'):c.neutral_surface(OLD)


def test_historical_source_hash_rejected_even_with_consistent_manifest_binding(tmp_path):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    archive=root.parent/'02_formal/p4a-bundle.zip'
    with zipfile.ZipFile(archive) as z:old_mf=json.loads(z.read('bundle-manifest.json'))
    old_mf['files']['source/V6/research_lab/seq_runner.py']['sha256']='unexpected old source'
    payload=json.dumps(old_mf).encode()
    old=json.loads((out/'run_identity.json').read_text());old['bundle']=c.identity(payload)
    target=out.with_name('run-'+p.hashed(old));out.rename(target)
    (target/'run_identity.json').write_text(json.dumps(old))
    with zipfile.ZipFile(archive,'w') as z:z.writestr('bundle-manifest.json',payload)
    with pytest.raises(ValueError,match='STOP: from source/V6/research_lab/seq_runner.py'):
        c.resolve(bundle,mf,binding,root,c.BUDGET)


def test_declaration_budget_is_detached(tmp_path):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    decl['budget']['policy']='mutated'
    assert c.BUDGET['policy']=='record_only'


def test_budget_requires_exact_numeric_representation(tmp_path):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    decl['budget']['units_per_hour']=0.0
    (bundle/name).write_text(json.dumps(decl));mf['files'][name]=b.digest(bundle/name)
    with pytest.raises(ValueError,match='STOP: budget'):c.resolve(bundle,mf,binding,root,c.BUDGET)


@pytest.mark.parametrize('damage',['missing_36th','corrupt_output','wrong_run_hash'])
def test_adoption_requires_all_36_valid_completed_fits(tmp_path,monkeypatch,damage):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    job=binding['queue'][35]
    fit=out/f"{job['queue']}-{job['year']}-{job['arm']}-{job['seed']}-w{job['window']}-lr{job['lr']}"
    if damage=='missing_36th':(fit/'complete.json').unlink()
    elif damage=='corrupt_output':(fit/'attempt-active/scores.parquet').write_bytes(b'corrupt')
    else:
        marker=json.loads((fit/'complete.json').read_text());marker['contract']['run_hash']='wrong'
        (fit/'complete.json').write_text(json.dumps(marker))
    monkeypatch.setattr(p,'train_fit',lambda *a,**k:pytest.fail('adoption retrained a completed fit'))
    with pytest.raises(ValueError,match='STOP'):c.resolve(bundle,mf,binding,root,c.BUDGET)


def test_local_only_source_boundary():
    from research_lab import p4_source_compat as c
    local='source/V6/research_lab/p4b_blend.py'
    assert local not in c.current_sources()
    with pytest.raises(ValueError,match='local-only'):
        b.allowed_path(local)
    stored=json.loads((b.ROOT/'tasks/p4a-source-compat-20e6b54.json').read_text())
    decl=c.declaration(OLD,c.current_sources(),stored['matrix'])
    # generation-time HEAD may differ after a commit; source bytes must agree.
    assert stored['to']['sources']==decl['to']['sources']
    assert stored['differences']==decl['differences']
    assert stored['local_only_sources']==decl['local_only_sources']
    assert decl['local_only_sources']=={
        'V6/research_lab/p4b_blend.py':'local CPU evaluation only; never executed in Colab'}
    assert decl['queue_sha256']=='4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23'
    assert decl['adoption']==dict(required_completed_prefix=36,queues=['Q1','Q2'],validator='resume_fit',retrain=False)


@pytest.mark.parametrize('required',['p4_colab.py','seq_runner.py','p4_eval.py','p4_bundle.py'])
def test_local_only_policy_cannot_hide_training_sources(monkeypatch,required):
    from research_lab import p4_source_compat as c
    monkeypatch.setattr(b,'LOCAL_ONLY_SOURCES',dict(b.LOCAL_ONLY_SOURCES,**{
        'V6/research_lab/'+required:'local CPU evaluation only; never executed in Colab'}))
    with pytest.raises(ValueError,match='unreviewed local-only'):
        c.current_sources()
    with pytest.raises(ValueError,match='unreviewed local-only'):
        b.allowed_path('source/V6/research_lab/'+required)


def test_undeclared_new_source_still_rejected(tmp_path,monkeypatch):
    from research_lab import p4_source_compat as c
    original=c.current_sources()
    monkeypatch.setattr(c,'current_sources',lambda:dict(original,**{
        'source/V6/research_lab/new_local_eval.py':c.identity(b'# undeclared new source\n')}))
    with pytest.raises(ValueError,match='unreviewed source difference: V6/research_lab/new_local_eval.py'):
        c.declaration(OLD,c.current_sources(),None)
    # Omitting any necessary source from the caller's inventory also fails.
    monkeypatch.undo()
    sources=c.current_sources();del sources['source/V6/research_lab/seq_runner.py']
    with pytest.raises(ValueError,match='current source inventory'):
        c.declaration(OLD,sources,None)


@pytest.mark.parametrize('statement',[
    'from research_lab import p4b_blend',
    'from .p4b_blend import main',
    'import research_lab.p4b_blend',
    "importlib.import_module('research_lab.p4b_blend')",
    "__import__('research_lab.p4b_blend')",
    "import_module('research_lab.p4b_blend')",
    "importlib.import_module(name='research_lab.p4b_blend')",
])
def test_colab_dependency_on_local_only_source_rejected(tmp_path,statement):
    folder=tmp_path/'V6/research_lab';folder.mkdir(parents=True)
    (folder/'p4_colab.py').write_text(statement+'\n')
    with pytest.raises(ValueError,match='imports local-only'):
        b.source_paths(tmp_path)


@pytest.mark.parametrize('damage',['missing','expanded','injected','misclassified'])
def test_resolve_rejects_local_only_policy_tampering(tmp_path,damage):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    if damage=='missing':del decl['local_only_sources']
    elif damage=='expanded':decl['local_only_sources']['V6/research_lab/seq_runner.py']='not allowed'
    elif damage=='misclassified':
        next(d for d in decl['differences'] if d['path'].endswith('/seq_runner.py'))['categories']=['本機專用來源排除']
    else:
        local='source/V6/research_lab/p4b_blend.py'
        mf['files'][local]=decl['to']['sources'][local]=c.identity(b'injected')
    (bundle/name).write_text(json.dumps(decl));mf['files'][name]=b.digest(bundle/name)
    with pytest.raises(ValueError,match='STOP: local-only'):
        c.resolve(bundle,mf,binding,root,c.BUDGET)


def test_excluded_content_cannot_change_any_bundle_file(tmp_path,monkeypatch):
    """Real pack/declaration/extract twice, changing only excluded bytes."""
    import shutil
    import pandas as pd
    from test_p4_full import dataset
    from research_lab import p4_source_compat as c
    original_root=b.ROOT
    checkout=tmp_path/'checkout';checkout.mkdir()
    # Retain the same read-only git object identity; no git writes in this test.
    git_dir=subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=original_root,text=True).strip()
    (checkout/'.git').write_text('gitdir: '+git_dir+'\n')
    for name in [n.removeprefix('source/') for n in c.current_sources()]+list(b.local_only_sources())+[c.LOCK]:
        target=checkout/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original_root/name,target)
    monkeypatch.setattr(b,'ROOT',checkout);monkeypatch.setattr(c,'ROOT',checkout)
    matrix=dataset(tmp_path/'matrix');b.dataset_manifest(matrix,enforce_x=False)
    monkeypatch.setattr(b,'DATASET_X',b.digest(matrix/'shared/X.npy'))
    ref=tmp_path/'ref';ref.mkdir()
    pd.DataFrame(dict(fold=[2021],date=pd.to_datetime(['2021-01-01']),code=['A'],label_y5=[.1],score=[.2],system=['c0plus'])).to_parquet(ref/'rank_average.parquet')
    (ref/'manifest.json').write_text(json.dumps(dict(outputs={'rank_average.parquet':b.digest(ref/'rank_average.parquet')})))
    snapshots=[]
    for index,payload in enumerate((b'raise RuntimeError("must never run")\n',b'not even valid Python: changed local bytes\x00')):
        (checkout/'V6/research_lab/p4b_blend.py').write_bytes(payload)
        archive=tmp_path/f'bundle-{index}.zip'
        mf=b.pack(matrix,[2021],ref,checkout/c.LOCK,archive,resume_from_commit=OLD)
        with zipfile.ZipFile(archive) as z:
            snapshots.append({name:z.read(name) for name in z.namelist()})
            assert 'source/V6/research_lab/p4b_blend.py' not in z.namelist()
        extracted=tmp_path/f'extracted-{index}'
        b.extract(archive,extracted);assert b.verify(extracted)==mf
        result=subprocess.run([sys.executable,'-I','-c',
            'import sys, importlib.util; sys.dont_write_bytecode=True; sys.path.insert(0,sys.argv[1]); '
            'assert importlib.util.find_spec("research_lab.p4b_blend") is None',
            str(extracted/'source/V6')],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    assert snapshots[0]==snapshots[1]  # Includes declaration AND bundle-manifest bytes.
    assert {n:c.identity(v) for n,v in snapshots[0].items()}=={n:c.identity(v) for n,v in snapshots[1].items()}
    print('excluded-content isolation:',len(snapshots[0]),'bundle members byte/hash identical; isolated Colab import absent')
    # A real new file with a similar prefix is not silently excluded.
    (checkout/'V6/research_lab/p4b_other_local.py').write_text('# undeclared local evaluation\n')
    assert 'source/V6/research_lab/p4b_other_local.py' in c.current_sources()
    with pytest.raises(ValueError,match='unreviewed source difference: V6/research_lab/p4b_other_local.py'):
        c.declaration(OLD,c.current_sources(),None)


def test_local_only_in_memory_mutations_are_killed(monkeypatch):
    from research_lab import p4_source_compat as c
    local='source/V6/research_lab/p4b_blend.py'
    def assert_excluded():
        assert local not in c.current_sources()
    def assert_archive_rejects():
        with pytest.raises(ValueError,match='local-only'):b.allowed_path(local)
    def assert_expansion_rejects():
        b.LOCAL_ONLY_SOURCES['V6/research_lab/seq_runner.py']='hidden training'
        with pytest.raises(ValueError,match='unreviewed local-only'):b.local_only_sources()
    mutations=[
        ('collection_filter',c,'source_paths',lambda root:list(b.source_paths(root))+[root/'V6/research_lab/p4b_blend.py'],assert_excluded),
        ('archive_boundary',b,'allowed_path',lambda name:None,assert_archive_rejects),
        ('policy_pin',b,'local_only_sources',lambda:dict(b.LOCAL_ONLY_SOURCES),assert_expansion_rejects),
    ]
    for label,module,name,mutant,check in mutations:
        with monkeypatch.context() as patch:
            patch.setattr(b,'LOCAL_ONLY_SOURCES',dict(b.LOCAL_ONLY_SOURCES))
            patch.setattr(module,name,mutant)
            with pytest.raises((AssertionError,pytest.fail.Exception)):
                check()
        print('assertion-killed:',label)


@pytest.mark.parametrize('stage',['write','extract','verify'])
def test_manual_local_only_archive_injection_rejected(tmp_path,stage):
    name='source/V6/research_lab/p4b_blend.py'
    payload=b'raise RuntimeError("local only")\n'
    file=tmp_path/'local.py';file.write_bytes(payload)
    if stage=='write':
        with pytest.raises(ValueError,match='local-only'):
            b.write_archive(tmp_path/'rejected.zip',{name:file},{})
    else:
        mf=dict(version='p4-bundle-v1',files={name:b.digest(file)},metadata={})
        archive=tmp_path/'injected.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(name,payload);z.writestr('bundle-manifest.json',json.dumps(mf))
        if stage=='extract':
            with pytest.raises(ValueError,match='local-only'):
                b.extract(archive,tmp_path/'output')
            assert not (tmp_path/'output').exists()
        else:
            root=tmp_path/'bundle';target=root/name;target.parent.mkdir(parents=True)
            target.write_bytes(payload);(root/'bundle-manifest.json').write_text(json.dumps(mf))
            with pytest.raises(ValueError,match='local-only'):b.verify(root)
