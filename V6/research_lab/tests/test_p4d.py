"""P4d stage1 executable boundary tests; CPU synthetic evidence only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from tools import p4d


def synthetic(n=1800, names=64):
    dates=pd.bdate_range('2015-01-01', periods=n)
    t=np.repeat(np.arange(n),names)
    data=SimpleNamespace(date_idx=t, dates=dates, eligible=t>=59,
        y={k:np.ones(len(t),dtype='float32') for k in ('y5','y10')},
        splits=np.where(t<1500,1,np.where(t>=1521,2,4)),
        codes=np.tile(np.arange(names).astype(str),n))
    return data


def test_strict_split_and_year_rounding():
    d=synthetic(); s=p4d.split(d, origin=1521)
    assert d.date_idx[s['validation']].max()==1500
    assert len(np.unique(d.date_idx[s['validation']]))==126
    assert d.date_idx[s['fit']].max()==1354
    assert len(np.unique(d.date_idx[s['fit']]))==5*252
    assert d.date_idx[s['fit']].max()<d.date_idx[s['validation']].min()-20
    assert s['manifest']['label_end_max']['validation']==1511


def test_full_outer_and_maturity_before_filter():
    d=synthetic();d.splits[d.date_idx==1501]=1
    with pytest.raises(ValueError,match='outer'):p4d.split(d,origin=1521)
    d=synthetic();d.y['y10'][d.date_idx==1500]=np.nan
    with pytest.raises(ValueError,match='matur'):p4d.split(d,origin=1521)


def test_missing_session_and_new_listing_and_tail_nan():
    d=synthetic(names=65); d.eligible[(d.date_idx<1600)&(d.codes=='0')]=False
    d.y['y5'][d.date_idx>1788]=np.nan
    s=p4d.split(d,origin=1521)
    assert np.isnan(d.y['y5'][s['evaluation']]).any()
    assert not np.any((d.codes[s['fit']]=='0'))
    d=synthetic();keep=d.date_idx!=1450
    for name in ('date_idx','eligible','splits','codes'):setattr(d,name,getattr(d,name)[keep])
    d.y={k:v[keep] for k,v in d.y.items()}
    with pytest.raises(ValueError,match='session'):p4d.split(d,origin=1521)


def test_rankgauss_future_bytes_and_schema_cache(tmp_path):
    rng=np.random.default_rng(17);x=rng.normal(size=(20,48)).astype('float32');ends=np.arange(2,21,2)
    a,ma=p4d.rankgauss(x,ends,{'fit':'abc'},tmp_path)
    changed=x.copy();changed[12:]=np.nan;changed[12:,0]=100
    b,mb=p4d.rankgauss(changed,ends,{'fit':'abc'},tmp_path)
    assert np.array_equal(a[:12],b[:12]);assert np.array_equal(a[:,36:],np.zeros((20,12)))
    assert ma['identity']!=mb['identity']
    c,mc=p4d.rankgauss(x,ends,{'fit':'different'},tmp_path)
    assert ma['identity']!=mc['identity'];assert np.array_equal(a,c)
    a2,m2=p4d.rankgauss(x,ends,{'fit':'abc'},tmp_path)
    assert m2==ma;assert np.array_equal(a,a2)


def test_queue_exact_and_separate():
    q=p4d.queue();assert len(q)==140
    assert len({p4d.hashed(j) for j in q})==140
    assert {j['arm'] for j in q}=={'gru','c0plus'}
    assert [j['seed'] for j in q if j['phase']=='anchor' and j['arm']=='gru' and (j['width'],j['layers'])==(64,2)]==list(p4d.SEEDS)
    assert p4d.hashed(q)==p4d.QUEUE_SHA
    assert p4d.anchor_seed(113)==29 and p4d.anchor_seed(101)==17
    assert p4d.CONTRACT['experiment']=='p4d_stage1'


def tiny_data():
    from research_lab.seq_models import SeqRanker
    rng=np.random.default_rng(4);x=rng.normal(size=(36,3,48)).astype('float32')
    y=x[:,-1,0].copy()
    return SimpleNamespace(feature_names=[str(i) for i in range(48)],
        date_idx=np.repeat(np.arange(3),12),dates=pd.bdate_range('2019-01-01',periods=3),
        y={'y5':y,'y10':y.copy()},windows=lambda rows:x[rows])


def test_fixed_update_resume_bytes_and_no_outer_validation(tmp_path):
    import torch
    from tools import p4d_train as train
    torch.set_num_threads(1)
    d=tiny_data();cfg=dict(p4d.CONTRACT['recipe'],d_model=4,n_layers=1,epochs=3,
        steps_per_epoch=2,dates_per_batch=1,stocks_per_date=8,min_names=8)
    tr=np.arange(24);va=np.arange(24,36)
    full=train.fit_gru(d,tr,va,cfg,17,'cpu',tmp_path/'full',{'seed':17},fixed_epochs=2)
    class Interrupt(Exception):pass
    def stop(state):
        if state['epoch']==1:raise Interrupt()
    with pytest.raises(Interrupt):train.fit_gru(d,tr,va,cfg,17,'cpu',tmp_path/'resume',{'seed':17},fixed_epochs=2,after_epoch=stop)
    resumed=train.fit_gru(d,tr,va,cfg,17,'cpu',tmp_path/'resume',{'seed':17},fixed_epochs=2)
    assert full[1]['epoch']==resumed[1]['epoch']==2
    assert full[1]['history']==resumed[1]['history']
    assert all(torch.equal(v,resumed[0].state_dict()[k]) for k,v in full[0].state_dict().items())
    # Outer gets no validation rows and cannot use outer labels to choose endpoint.
    outer=train.fit_gru(d,tr,np.array([],int),cfg,17,'cpu',tmp_path/'outer',{'seed':17},fixed_epochs=2)
    assert all(h['val_ic5'] is None for h in outer[1]['history'])
    with pytest.raises(ValueError,match='identity'):train.fit_gru(d,tr,va,cfg,17,'cpu',tmp_path/'resume',{'seed':29},fixed_epochs=2)


def test_checkpoint_corrupt_missing_orphan(tmp_path):
    from tools import p4d_train as train
    d=tiny_data();cfg=dict(p4d.CONTRACT['recipe'],d_model=4,n_layers=1,epochs=1,
        steps_per_epoch=1,dates_per_batch=1,stocks_per_date=8,min_names=8)
    train.fit_gru(d,np.arange(24),np.arange(24,36),cfg,17,'cpu',tmp_path,{'seed':17},fixed_epochs=1)
    pointer=tmp_path/'checkpoint-pointer.json';saved=pointer.read_bytes()
    pointer.unlink()
    with pytest.raises(ValueError,match='orphan'):train.fit_gru(d,np.arange(24),np.arange(24,36),cfg,17,'cpu',tmp_path,{'seed':17},fixed_epochs=1)
    pointer.write_bytes(saved);p=json.loads(saved);(tmp_path/p['latest']['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):train.fit_gru(d,np.arange(24),np.arange(24,36),cfg,17,'cpu',tmp_path,{'seed':17},fixed_epochs=1)


def test_future_labels_do_not_enter_tuning():
    d=synthetic();before=p4d.split(d,1521)
    for k in d.y:d.y[k][d.date_idx>=1521]=np.inf
    after=p4d.split(d,1521)
    assert np.array_equal(before['fit'],after['fit'])
    assert np.array_equal(before['validation'],after['validation'])


def test_rehearsal_negative_does_not_launch():
    from tools import p4d_bundle as bundle
    actions=[]
    def fail():
        actions.append('gate');raise ValueError('failed rehearsal')
    with pytest.raises(ValueError):bundle.guarded_launch(fail,lambda:actions.append('formal'))
    assert actions==['gate']


def test_two_file_delivery_and_safe_extract(tmp_path):
    from tools import p4d_bundle as bundle
    from research_lab import p4_bundle as old
    output=tmp_path/'delivery';output.mkdir()
    bundle.delivery(output)
    assert {p.suffix for p in output.iterdir()}=={'.zip','.ipynb','.txt'}
    with pytest.raises(ValueError):old.extract(output/'p4d-bundle.zip',tmp_path/'legacy-extract')
    mf=bundle.pb.extract(output/'p4d-bundle.zip',tmp_path/'extract')
    assert mf['metadata']['experiment']=='p4d_stage1'
    assert 'source/V6/research_lab/p4b_blend.py' not in mf['files']
    assert not any('p4d_evaluate' in n for n in mf['files'])
    notebook=json.loads((output/'p4d_stage1.ipynb').read_text())
    sources='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'from tools.p4d_bundle import launch' in sources and 'PROGRESS.txt' in sources
    assert 'expected_sha256=' not in sources


def test_ensemble_tail_nan_and_incomplete_support():
    frames={}
    for seed in p4d.SEEDS:
        frames[seed]=pd.DataFrame(dict(fold=2021,date=np.repeat(pd.bdate_range('2021-01-01',periods=3),12),
            code=np.tile(np.arange(12).astype(str),3),label_y5=np.r_[np.tile(np.arange(12),2),np.full(12,np.nan)],
            score=np.tile(np.arange(12),3)+seed/100))
    avg,_=p4d.ensemble(frames,p4d.SEEDS)
    report=p4d.descriptive_pair(avg,avg)
    assert report['daily'][-1]['ic5'] is None
    assert report['daily'][-1]['excluded_nan_rows']==12
    bad=copy.deepcopy(frames);bad[113]=bad[113].iloc[1:]
    with pytest.raises(ValueError,match='INCOMPLETE'):p4d.ensemble(bad,p4d.SEEDS)
    with pytest.raises(ValueError,match='INCOMPLETE'):p4d.ensemble({17:frames[17]},p4d.SEEDS)


def test_future_mutation_fresh_process_and_legacy_counterexample(tmp_path):
    import subprocess,sys,os
    from research_lab.seq_runner import macro_columns
    # A formerly constant column becomes cross-sectionally variable only later.
    x=np.zeros((240,48),dtype='float32');changed=x.copy()
    changed[120:,0]=np.tile([1.,-1.],60)
    ends=np.arange(2,241,2)
    old=macro_columns(x,ends);new=macro_columns(changed,ends)
    assert 0 in old and 0 not in new
    for value,name in ((x,'before'),(changed,'after')):
        np.save(tmp_path/(name+'.npy'),value)
        script="""import numpy as np,sys
from tools.p4d import rankgauss
x=np.load(sys.argv[1]);out,_=rankgauss(x,np.arange(2,241,2),{'fit':'fixed'},sys.argv[2],rebuild=True)
np.save(sys.argv[3],out[:120])
"""
        subprocess.run([sys.executable,'-c',script,str(tmp_path/(name+'.npy')),str(tmp_path/'cache'),str(tmp_path/(name+'-out.npy'))],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    assert (tmp_path/'before-out.npy').read_bytes()==(tmp_path/'after-out.npy').read_bytes()


def test_cpu_rehearsal_real_seq_path_and_evaluator(tmp_path):
    from tools import p4d_train as train
    import torch
    torch.set_num_threads(1)
    # Real SeqData bad shapes with a full 48-feature schema; synthetic only.
    from test_seq_runner import build
    matrix=tmp_path/'matrix';matrix.mkdir()
    _,day,_=build(matrix,days=1800,names=65,train_end=1500,eval_start=1521,gap=(0,0,210))
    real_names=json.loads((p4d.ROOT/'research/v7/corrected-baseline-v1/feature-manifest.json').read_text())['feature_order']
    manifest=json.loads((matrix/'manifest.json').read_text());manifest['feature_order']=real_names
    (matrix/'manifest.json').write_text(json.dumps(manifest))
    raw=np.load(matrix/'X.npy')
    # test_seq_runner builder has fewer features: extend using causal synthetic values.
    rng=np.random.default_rng(2);extended=rng.normal(size=(len(raw),48)).astype('float32');extended[:,:raw.shape[1]]=raw
    np.save(matrix/'X.npy',extended)
    for label,h in (('y5',5),('y10',10)):
        y=np.load(matrix/(label+'.npy'));y[day>=1800-h]=np.nan;np.save(matrix/(label+'.npy'),y)
    data=p4d.load_data(matrix);support=p4d.split(data,1521)
    data.Xn,_=p4d.rankgauss(data.X,data.ends,support['manifest'],tmp_path/'cache',feature_names=real_names)
    # CPU route smoke uses two fit days, one validation day and ten eval days.
    # Full 126-session / full-history support was verified above, not trained.
    support['fit']=support['fit'][-130:]
    support['validation']=support['validation'][-65:]
    def smoke_support(value):
        value=copy.deepcopy(value)
        for role in ('fit','validation','evaluation'):value['manifest']['support'][role]['rows']=len(value[role])
        return value
    support=smoke_support(support)
    selected={'gru:17':1,'c0plus:17':1}
    frames={};fit_evidence={}
    from tools import p4d_bundle as bundle
    def record(name,root,mf):
        fit_evidence[name]=dict(path=str(root),complete=bundle.pb.digest(root/'complete.json'),binding=mf['binding'],seconds=1.,peak_allocated=0,peak_reserved=0)
    for width,layers in ((256,2),(64,4),(32,2)):
        job=dict(phase='anchor',origin='anchor',arm='gru',seed=17,width=width,layers=layers)
        root=tmp_path/f'd{width}x{layers}'
        mf=train.execute_job(data,support,job,root,{'synthetic':True},selected,device='cpu',rehearsal=True)
        assert train.validate_complete(root,mf['binding'])==mf
        record(f'gru:{width}:{layers}:anchor',root,mf)
    cjob=dict(phase='anchor',origin='anchor',arm='c0plus',seed=17,width=64,layers=2)
    csupport=dict(support,fit=p4d.split(data,1521)['fit'][-650:])
    csupport=smoke_support(csupport)
    croot=tmp_path/'c0anchor'
    cmf=train.execute_job(data,csupport,cjob,croot,{'synthetic':True},selected,device='cpu',rehearsal=True)
    record('c0plus:64:2:anchor',croot,cmf)
    outer=p4d.split(data,1521,anchor=False)
    outer['fit']=outer['fit'][-130:]
    outer['evaluation']=outer['evaluation'][-650:]
    for arm in ('gru','c0plus'):
        job=dict(phase='outer',origin=2021,arm=arm,seed=17,width=32 if arm=='gru' else 64,layers=2)
        if arm=='c0plus':outer['fit']=p4d.split(data,1521,anchor=False)['fit'][-650:]
        outer=smoke_support(outer)
        root=tmp_path/arm
        mf=train.execute_job(data,outer,job,root,{'synthetic':True},selected,device='cpu',rehearsal=True)
        frames[arm]=pd.read_parquet(root/'scores.parquet')
        record(f"{arm}:{job['width']}:2:outer",root,mf)
    result=p4d.descriptive_pair(frames['gru'],frames['c0plus'])
    assert result['support']['common_rows']==len(frames['gru'])
    assert result['daily'][-1]['excluded_nan_rows']>0
    target=tmp_path/'evaluation.json';target.write_text(json.dumps(result))
    evidence=dict(binding={'synthetic':True},evidence_class='REHEARSAL / 不是研究結果',fits=fit_evidence,
        evaluation=dict(path=str(target),identity=bundle.pb.digest(target)),elapsed_seconds=1.)
    bundle.validate_gate(evidence,{'synthetic':True})
    for mode in ('missing','request','elapsed','publication','evaluation'):
        bad=copy.deepcopy(evidence)
        if mode=='missing':bad['fits'].pop('gru:256:2:anchor')
        elif mode=='request':bad['fits']['gru:256:2:anchor']['binding']['job']['width']=128
        elif mode=='elapsed':bad['elapsed_seconds']=True
        elif mode=='publication':bad['fits']['gru:256:2:anchor']['complete']['sha256']='0'*64
        else:bad['evaluation']['identity']['sha256']='0'*64
        with pytest.raises(ValueError):bundle.validate_gate(bad,{'synthetic':True})


    # A reconnect revalidates durable publications without the old /tmp fits.
    import shutil
    progress=tmp_path/'progress'
    durable=bundle.ensure_rehearsal(progress,{'synthetic':True},lambda:evidence)
    for fit in evidence['fits'].values():shutil.rmtree(fit['path'])
    target.unlink()
    def must_not_rehearse():raise AssertionError('valid marker was not reused')
    assert bundle.ensure_rehearsal(progress,{'synthetic':True},must_not_rehearse)==durable
    Path(durable['evaluation']['path']).write_text('{}')
    with pytest.raises(AssertionError,match='valid marker'):
        bundle.ensure_rehearsal(progress,{'synthetic':True},must_not_rehearse)
    assert not (progress/'rehearsal_passed.json').exists()


def test_five_in_memory_mutations(monkeypatch,tmp_path):
    """Real assertions kill five semantic regressions without editing source."""
    original=p4d.queue
    def assert_queue():
        assert len(p4d.queue())==140 and p4d.hashed(p4d.queue())==p4d.QUEUE_SHA
    mutants=[lambda:original()[:-1],lambda:original()[::-1],
        lambda:[dict(j,seed=999) if i==0 else j for i,j in enumerate(original())]]
    for mutation in mutants:
        with monkeypatch.context() as m:
            m.setattr(p4d,'queue',mutation)
            with pytest.raises(AssertionError):assert_queue()
    from tools import p4d_bundle as bundle
    # Bypass gate mutation is killed by the same invariant as the real negative test.
    with monkeypatch.context() as m:
        m.setattr(bundle,'guarded_launch',lambda gate,formal:formal())
        events=[]
        def gate():raise ValueError('gate failed')
        def invariant():
            with pytest.raises(ValueError):bundle.guarded_launch(gate,lambda:events.append('formal'))
            assert not events
        with pytest.raises(pytest.fail.Exception):invariant()
    # Future-dependent schema mutation violates content invariance, not just hash.
    x=np.zeros((240,48),dtype='float32');x[120:,0]=np.tile([1.,-1.],60);x[:120,0]=np.tile([1.,-1.],60)
    out,_=p4d.rankgauss(x,np.arange(2,241,2),{'fit':'x'})
    with monkeypatch.context() as m:
        schema=copy.deepcopy(p4d.SCHEMA);schema['zero_columns']=[]
        # A transform that reintroduces the frozen macro columns is observable.
        real=p4d.rankgauss
        def wrong(*args,**kwargs):
            a,b=real(*args,**kwargs);a[:,36]=1.;return a,b
        m.setattr(p4d,'rankgauss',wrong)
        with pytest.raises(AssertionError):
            a,_=p4d.rankgauss(x,np.arange(2,241,2),{'fit':'x'});assert np.all(a[:,36:]==0)


def test_raw_prepare_never_uses_legacy_mask_or_writes_source(tmp_path,monkeypatch):
    from test_p4_full import dataset
    from research_lab import p4_bundle as b,seq_runner
    from tools import p4d_bundle as bundle
    source=dataset(tmp_path/'drive');b.dataset_manifest(source,enforce_x=False)
    before={p.relative_to(source).as_posix():b.digest(p) for p in source.rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('legacy transform invoked')
    monkeypatch.setattr(seq_runner,'ensure_rankgauss',forbidden)
    bundle.prepare_raw_dataset(source,tmp_path/'ssd')
    bundle.prepare_raw_dataset(source,tmp_path/'ssd')
    after={p.relative_to(source).as_posix():b.digest(p) for p in source.rglob('*') if p.is_file()}
    assert before==after
    assert not (source/'rankgauss_cache').exists()


def test_local_evaluator_all_curves_and_missing_family():
    import importlib.util
    path=p4d.ROOT/'tools/p4d_evaluate.py';spec=importlib.util.spec_from_file_location('p4d_local_eval',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames={}
    for job in p4d.queue():
        frames[p4d.hashed(job)]=pd.DataFrame(dict(fold=job['origin'],
            date=np.repeat(pd.bdate_range('2020-01-01' if job['origin']=='anchor' else f"{job['origin']}-01-01",periods=2),12),
            code=np.tile(np.arange(12).astype(str),2),label_y5=np.tile(np.arange(12),2),score=np.tile(np.arange(12),2)))
    outputs,report=module.curves(frames)
    assert len(outputs)==217 and len(report)==210
    assert all('Holm m=2 deferred' in row['registered_family'] for row in report.values())
    frames.pop(p4d.hashed(p4d.queue()[-1]))
    with pytest.raises(KeyError):module.curves(frames)


def test_preprocessing_matches_frozen_nonfinite_policy():
    from scipy.special import ndtri
    from scipy.stats import rankdata
    x=np.zeros((4,48),dtype='float32');x[:,0]=[np.nan,np.inf,-3,2]
    out,_=p4d.rankgauss(x,np.array([4]),{'fit':'same'})
    expected=ndtri((rankdata([0,0,-3,2],method='average')-.5)/4).astype('float32')
    assert np.array_equal(out[:,0],expected)


def test_future_mutation_baseline_selection_and_weights(tmp_path):
    from tools.p4d_train import fit_gru
    import torch
    d=tiny_data();cfg=dict(p4d.CONTRACT['recipe'],d_model=4,n_layers=1,epochs=2,steps_per_epoch=2,
        min_names=10,dates_per_batch=1,stocks_per_date=12)
    first,state=fit_gru(d,np.arange(24),np.arange(24,36),cfg,17,'cpu',tmp_path/'before',{'fixed':'tuning'})
    changed=copy.copy(d);changed.y={k:np.r_[v,np.full(12,np.nan,dtype='float32')] for k,v in d.y.items()}
    changed.date_idx=np.r_[d.date_idx,np.repeat(3,12)];changed.dates=pd.bdate_range('2019-01-01',periods=4)
    windows=d.windows
    def mutated_windows(rows):
        assert np.max(rows)<36
        return windows(rows)
    changed.windows=mutated_windows
    second,after=fit_gru(changed,np.arange(24),np.arange(24,36),cfg,17,'cpu',tmp_path/'after',{'fixed':'tuning'})
    assert state['history']==after['history'] and state['selected_epoch']==after['selected_epoch']
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())


def test_new_run_ledger_order_and_restart(tmp_path,monkeypatch):
    """Exercise actual durable ledger, queue orchestration and selected freeze.

    CUDA/data/fits are boundary doubles; no research training occurs.
    """
    from tools import p4d_train as train,p4d_bundle as bundle
    binding={'experiment':'p4d_stage1','queue':p4d.QUEUE_SHA,'contract':p4d.hashed(p4d.CONTRACT)}
    monkeypatch.setattr(train,'require_cuda',lambda:None)
    monkeypatch.setattr(bundle.pb,'verify',lambda p:{})
    monkeypatch.setattr(bundle,'run_binding',lambda *a:binding)
    monkeypatch.setattr(bundle,'validate_gate',lambda *a:None)
    monkeypatch.setattr(bundle,'require_frozen_split',lambda *a:None)
    monkeypatch.setattr(train,'prepare',lambda *a,**k:(None,{'manifest':{'smoke':True}},{'identity':'smoke'}))
    outer=[]
    def fit(data,support,job,path,run,selected,**kw):
        if job['phase']=='outer':
            assert len(selected)==4;outer.append(job)
        path.mkdir(parents=True,exist_ok=True)
        metrics={'selected_epoch':1,'selected_iteration':25}
        (path/'metrics.json').write_text(json.dumps(metrics))
        mf={'binding':{'job':job,'run':run},'outputs':{}}
        (path/'complete.json').write_text(json.dumps(mf))
        return mf
    monkeypatch.setattr(train,'execute_job',fit)
    monkeypatch.setattr(train,'validate_complete',lambda path,b:json.loads((path/'complete.json').read_text()))
    # verify(bundle) is patched; the rest of orchestration uses an actual directory.
    root=train.run(tmp_path/'data',tmp_path/'experiment',tmp_path/'bundle',rehearsal_evidence={})
    completed=json.loads((root/'completed.json').read_text());assert len(completed['outputs'])==140
    assert len(outer)==120
    assert json.loads((root/'selected_recipe.json').read_text())['selected']=={'gru:17':1,'gru:29':1,'c0plus:17':25,'c0plus:29':25}
    assert train.run(tmp_path/'data',tmp_path/'experiment',tmp_path/'bundle',rehearsal_evidence={})==root
    first=json.loads((root.parent/'event-000001.json').read_text());assert first['event']=='RUN_CREATED'


def test_process_kill_restart_matches_uninterrupted(tmp_path):
    import subprocess,sys,os
    script=r'''
import sys,os,signal,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from tools import p4d
from tools.p4d_train import fit_gru
torch.set_num_threads(1)
x=np.random.default_rng(4).normal(size=(36,3,48)).astype('float32');y=x[:,-1,0].copy()
d=SimpleNamespace(feature_names=list(map(str,range(48))),date_idx=np.repeat(np.arange(3),12),
 dates=pd.bdate_range('2019-01-01',periods=3),y={'y5':y,'y10':y.copy()},windows=lambda r:x[r])
cfg=dict(p4d.CONTRACT['recipe'],d_model=4,n_layers=1,epochs=2,steps_per_epoch=2,dates_per_batch=1,stocks_per_date=12,min_names=10)
def crash(state):
 if state['epoch']==1:os.kill(os.getpid(),signal.SIGKILL)
m,state=fit_gru(d,np.arange(24),np.arange(24,36),cfg,17,'cpu',Path(sys.argv[1]),{'seed':17},fixed_epochs=2,after_epoch=crash if sys.argv[2]=='kill' else None)
h=hashlib.sha256()
for key,value in sorted(m.state_dict().items()):h.update(key.encode());h.update(value.numpy().tobytes())
Path(sys.argv[3]).write_text(json.dumps(dict(weights=h.hexdigest(),rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),history=state['history'])))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'full'),'full',str(tmp_path/'full.json')],check=True,env=env)
    killed=subprocess.run([sys.executable,'-c',script,str(tmp_path/'restart'),'kill',str(tmp_path/'partial.json')],env=env)
    assert killed.returncode==-9
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'restart'),'resume',str(tmp_path/'resume.json')],check=True,env=env)
    assert json.loads((tmp_path/'full.json').read_text())==json.loads((tmp_path/'resume.json').read_text())


def test_loader_stock_index_is_verified_and_never_written(tmp_path):
    from test_seq_runner import build
    build(tmp_path,days=100,names=12)
    d=p4d.load_data(tmp_path)
    assert not (tmp_path/'stock_idx.npy').exists()
    np.save(tmp_path/'stock_idx.npy',d.stock_idx)
    assert np.array_equal(p4d.load_data(tmp_path).stock_idx,d.stock_idx)
    bad=d.stock_idx.copy();bad[-1]=0;np.save(tmp_path/'stock_idx.npy',bad)
    with pytest.raises(ValueError,match='stock index cache'):p4d.load_data(tmp_path)


def test_stage1b_delivery_cells_compile_and_rehearsal_resume(tmp_path):
    from tools import p4d_bundle as bundle
    bundle.delivery(tmp_path)
    nb=json.loads((tmp_path/'p4d_stage1.ipynb').read_text())
    for index,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'p4d-cell-{index}','exec')
    text='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'ensure_rehearsal' in text
    assert 'progress/formal' in text and 'rehearsal_passed.json' in text
    assert 'zipfile.ZipFile' in nb['cells'][-1]['source']


def test_stage1b_invalid_marker_never_launches(tmp_path):
    from tools import p4d_bundle as bundle
    events=[]
    def fail():
        events.append('gate')
        raise ValueError('missing evidence')
    with pytest.raises(ValueError):bundle.ensure_rehearsal(tmp_path,{'run':'cpu'},fail)
    assert events==['gate']
    marker=tmp_path/'rehearsal_passed.json'
    assert not marker.exists()
    marker.write_text('{}')
    with pytest.raises(ValueError):bundle.ensure_rehearsal(tmp_path,{'run':'cpu'},fail)
    assert events==['gate','gate'] and not marker.exists()


def test_stage1b_five_named_in_memory_mutations(monkeypatch):
    import inspect
    from tools import p4d_bundle as bundle
    # Mutation is in a temporary namespace; repository bytes remain unchanged.
    namespace=dict(p4d.__dict__)
    source=inspect.getsource(p4d.split).replace('t[outer]>=origin-20','t[outer]>origin-20')
    assert source!=inspect.getsource(p4d.split)
    exec(compile(source,'split-boundary-mutant','exec'),namespace)
    d=synthetic();d.splits[d.date_idx==1501]=1
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(ValueError,match='outer'):namespace['split'](d,1521)
    print('MUTATION split_boundary KILLED')
    with monkeypatch.context() as m:
        m.setitem(p4d.SCHEMA,'zero_columns',[])
        with pytest.raises(ValueError,match='contract'):p4d.verify_contract()
    print('MUTATION macro_schema KILLED')
    with monkeypatch.context() as m:
        m.setattr(p4d,'SEEDS',(29,17,43,59,71,89,101,113))
        with pytest.raises(ValueError,match='queue SHA'):p4d.verify_contract()
    print('MUTATION seed_sequence KILLED')
    with monkeypatch.context() as m:
        real=p4d.queue
        m.setattr(p4d,'queue',lambda:real()[::-1])
        with pytest.raises(ValueError,match='queue SHA'):p4d.verify_contract()
    print('MUTATION queue_SHA KILLED')
    with monkeypatch.context() as m:
        m.setattr(bundle,'guarded_launch',lambda gate,formal:formal())
        launched=[]
        def fail():raise ValueError('STOP')
        with pytest.raises(pytest.fail.Exception):
            with pytest.raises(ValueError):bundle.guarded_launch(fail,lambda:launched.append(True))
        assert launched==[True]  # The oracle catches actual unauthorized launch.
    print('MUTATION rehearsal_bypass KILLED')


def test_stage1b_empty_gate_result_never_launches():
    from tools import p4d_bundle as bundle
    events=[]
    for empty in (None,{}):
        with pytest.raises(ValueError):
            bundle.guarded_launch(lambda:empty,lambda *args:events.append('formal'))
    assert not events


def test_rehearsal_heartbeat_and_formal_continuity(tmp_path):
    from tools.p4d_train import Progress
    clock=[0.]
    p=Progress(tmp_path,clock=lambda:clock[0])
    p.stage('preprocessing',mode='REHEARSAL')
    snapshots=[]
    original=p.write
    def record():
        original();snapshots.append(json.loads((tmp_path/'progress.json').read_text()))
    p.write=record
    class FakeStop:
        def wait(self,interval):
            assert interval==60.
            clock[0]+=interval
            return clock[0]>180
    p.stop=FakeStop()
    p._heartbeat()
    assert len(snapshots)==3
    assert all(s['mode']=='REHEARSAL' and s['stage']=='preprocessing' for s in snapshots)
    assert 'REHEARSAL' in (tmp_path/'PROGRESS.txt').read_text()
    p.stage('formal_setup',mode='FORMAL')
    p.start_fit(dict(queue='P4d',arm='gru',year=2026,seed=17),1,140,dict(epochs=1,steps_per_epoch=1))
    s=p.snapshot()
    assert s['mode']=='FORMAL' and s['stage']=='formal_setup' and s['fit_index']==1
    assert p.clock()==240. and p.durations==[]


def test_rehearsal_progress_write_failure_is_nonfatal(tmp_path,monkeypatch,capsys):
    from tools.p4d_train import Progress
    p=Progress(tmp_path)
    p.stage('c0plus_fit',mode='REHEARSAL')
    monkeypatch.setattr('os.replace',lambda *a:(_ for _ in ()).throw(OSError('disk failure')))
    with p:
        p.stage('inference')
        p.write()
    assert '訓練繼續' in capsys.readouterr().out
    assert p.snapshot()['mode']=='REHEARSAL'


def test_launch_progress_precedes_gate_and_is_shared(tmp_path,monkeypatch):
    from tools import p4d_bundle as b,p4d_train as t
    events=[];seen=[]
    monkeypatch.setattr(b,'run_binding',lambda *a:events.append('binding') or {'run':'tiny'})
    def gate(root,binding,callback):
        events.append('gate');assert (root/'PROGRESS.txt').exists()
        return callback()
    monkeypatch.setattr(b,'ensure_rehearsal',gate)
    def rehearsal(*args,progress):
        assert progress.snapshot()['mode']=='REHEARSAL'
        seen.append(progress);return {'ok':True}
    monkeypatch.setattr(b,'rehearse',rehearsal)
    def formal(*args,rehearsal_evidence,progress):
        assert progress is seen[0] and progress.snapshot()['mode']=='FORMAL'
        assert progress.thread.is_alive();return 'done'
    monkeypatch.setattr(t,'run',formal)
    assert b.launch('data',tmp_path,'bundle')=='done'
    assert events==['binding','gate'] and not seen[0].thread.is_alive()


def test_measure_c0plus_cli_tiny_full_rows(tmp_path,monkeypatch):
    from tools import p4d_train as t
    from research_lab.c0plus import CORE
    d=tiny_data();d.X=np.random.default_rng(3).normal(size=(36,48)).astype('float32')
    d.feature_names=CORE+[f'extra{i}' for i in range(48-len(CORE))]
    d.ends=np.arange(12,37,12);d.stock_idx=np.tile(np.arange(12),3);d.n_stocks=12
    rows=np.arange(36)
    support={'fit':rows,'validation':np.array([],int),'manifest':{'support':'tiny'}}
    monkeypatch.setattr(t,'prepare',lambda *a,**kw:(d,support,{'identity':'tiny'}))
    monkeypatch.setattr(t,'measurement_inputs',lambda directory:{'synthetic':True})
    root=tmp_path/'.artifacts'/'p4d';root.mkdir(parents=True)
    monkeypatch.setattr(t,'measurement_root',lambda:root)
    assert t.main(['--measure-c0plus','--origin','2026','--data-dir',str(tmp_path/'data')])==0
    created=[p for p in root.iterdir() if p.name.startswith('measure-')]
    assert len(created)==1
    output=created[0]
    report=json.loads((output/'measurement.json').read_text())
    assert report['fit_rows']==36 and report['matrix_shape']==[36,185]
    assert report['matrix_bytes']==36*185*4
    assert report['peak_rss_bytes']>0 and report['elapsed_seconds']>0
    assert report['iterations']==200 and report['origin']==2026
    assert report['evidence_class']=='CPU_RESOURCE_MEASUREMENT / 不是研究結果'
    assert report['preprocess_identity']=='tiny' and report['input_identity']=={'synthetic':True}
    with pytest.raises(SystemExit):t.main(['--measure-c0plus','--origin','2025','--data-dir','unused'])


def test_measure_c0plus_output_namespace_enforced(tmp_path,monkeypatch):
    from tools import p4d_train as t
    root=tmp_path/'.artifacts'/'p4d';root.mkdir(parents=True)
    monkeypatch.setattr(t,'measurement_root',lambda:root)
    # The CLI no longer accepts an output override.
    with pytest.raises(SystemExit):
        t.main(['--measure-c0plus','--origin','2026','--data-dir','unused','--output',str(tmp_path/'elsewhere')])
    # Direct calls outside the namespace are rejected before any directory is created.
    for bad in (tmp_path/'elsewhere', root/'not-measure', root/'measure-x'/'nested', root.parent/'measure-y'):
        with pytest.raises(ValueError,match='STOP: measurement output'):
            t.checked_measurement_output(bad)
        assert not bad.exists()
    assert t.checked_measurement_output(root/'measure-ok')==(root/'measure-ok').resolve()


def test_rehearsal_every_path_gets_publisher(tmp_path,monkeypatch):
    from tools import p4d_train as t,p4d_bundle as b
    import torch
    p=t.Progress(tmp_path/'progress');seen=[];prepared=[]
    monkeypatch.setattr(b,'run_binding',lambda *a:{'run':'tiny'})
    def prepare(*args,anchor,progress):
        assert progress is p and progress.snapshot()['mode']=='REHEARSAL'
        prepared.append(anchor);return None,{'manifest':{}},{}
    monkeypatch.setattr(t,'prepare',prepare)
    monkeypatch.setattr(b,'require_frozen_split',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(b,'validate_gate',lambda *a:None)
    monkeypatch.setattr(p4d,'descriptive_pair',lambda *a:{'synthetic':True})
    monkeypatch.setattr(pd,'read_parquet',lambda *a:pd.DataFrame())
    def execute(data,support,job,root,binding,selected,*,rehearsal,progress,ledger):
        assert rehearsal and progress is p and ledger is not None
        seen.append((job['phase'],job['arm'],job['width'],job['layers']))
        root.mkdir();(root/'complete.json').write_text('{}')
        return {'binding':{'synthetic':True}}
    monkeypatch.setattr(t,'execute_job',execute)
    result=b.rehearse('data','bundle',tmp_path/'cache',progress=p)
    assert prepared==[True,False] and len(result['fits'])==6
    assert seen==[('anchor','gru',256,2),('anchor','gru',64,4),('anchor','gru',32,2),
                  ('anchor','c0plus',64,2),('outer','gru',32,2),('outer','c0plus',64,2)]


def test_progress_two_in_memory_mutations_killed(tmp_path,monkeypatch):
    from tools import p4d_train as t
    def heartbeat_oracle():
        test_rehearsal_heartbeat_and_formal_continuity(tmp_path/'heartbeat')
    with monkeypatch.context() as m:
        m.setattr(t.Progress,'_heartbeat',lambda self:None)
        with pytest.raises(AssertionError):heartbeat_oracle()
    print('MUTATION rehearsal_heartbeat_disabled KILLED')
    with monkeypatch.context() as m:
        real=t.Progress.snapshot
        def wrong_mode(self):
            snapshot=real(self);snapshot['mode']='FORMAL';return snapshot
        m.setattr(t.Progress,'snapshot',wrong_mode)
        with pytest.raises(AssertionError):heartbeat_oracle()
    print('MUTATION rehearsal_mode_erased KILLED')
