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
    delivery=tmp/'02_formal';delivery.mkdir()
    with zipfile.ZipFile(delivery/'p4a-bundle.zip','w') as z:z.writestr('bundle-manifest.json',payload)
    return c,bundle,mf,binding,root,out,decl,name


def test_adopt_and_audit_identity(tmp_path):
    c,bundle,mf,binding,root,out,decl,name=setup_upgrade(tmp_path)
    chosen,event=c.resolve(bundle,mf,binding,root,dict(policy='record_only',units_per_hour=0,thermal_stop_c=85))
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
    (out/'run_identity.json').unlink();out.rmdir()
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


def test_run_old_twelve_fits_then_upgrade_skips_and_audits(tmp_path,monkeypatch):
    """Real old run/ledger/publication, CPU artifact double; no GPU claims."""
    import io
    import pandas as pd
    from research_lab import p4_source_compat as c
    op=old_module(tmp_path,'p4_colab')
    to=c.current_sources();decl=c.declaration(OLD,to,None)
    ref=tmp_path/'reference.parquet'
    frame=pd.DataFrame([dict(fold=y,date=pd.Timestamp(f'{y}-02-01'),code=code,label_y5=float(i),score=float(i),system='c0plus') for y in range(2021,2027) for i,code in enumerate(('A','B'))])
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
        if disconnect[0] and len(calls)==13:
            disconnect[0]=False
            raise RuntimeError('synthetic disconnection after twelve completed fits')
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
    assert len(list(out.glob('*/complete.json')))==12
    completed_calls=calls[:12];calls.clear()
    monkeypatch.setattr(p,'train_fit',fake_train)
    monkeypatch.setattr(p,'permutation_fit',lambda *a,**k:dict(CPU_TEST_DOUBLE=True))
    result=p.run(newbundle,root)
    assert len(calls)==52 and not set(completed_calls)&set(calls)
    assert len(result['jobs'])==64 and (out/'run_identity.json').read_bytes()==oldidentity
    assert all(b.digest(Path(f))==ident for f,ident in originals.items())
    assert list(root.glob('run-*'))==[out]
    audits=list(out.glob('source-upgrade-*.json'));assert len(audits)==1
    event=json.loads(audits[0].read_text());assert event['event']=='SOURCE_UPGRADE'
    ledger_events=[json.loads(f.read_text()) for f in root.glob('event-*.json')]
    recorded=next(e for e in ledger_events if e['event']=='SOURCE_UPGRADE')
    assert recorded['state']['source_upgrade']['identity']==b.digest(audits[0])
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
