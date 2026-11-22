"""P4a full queue: offline synthetic engineering evidence only."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from research_lab import p4_bundle as b, p4_colab as p, seq_runner as sr


def dataset(root):
    from test_seq_runner import build
    shared=root/'shared';shared.mkdir(parents=True);build(shared);sr.stock_index(shared)
    for year in range(2021,2027):
        fold=root/f'folds/fold_{year}';fold.mkdir(parents=True)
        shutil.copy2(shared/'splits.npy',fold/'splits.npy')
    (shared/'splits.npy').unlink()
    return root


def test_dataset_manifest_copy_cache_and_mutations(tmp_path):
    root=dataset(tmp_path/'drive')
    mf=b.dataset_manifest(root, enforce_x=False)
    assert len(mf['files'])==14
    local=tmp_path/'ssd'
    result=b.prepare_dataset(root,local)
    assert result['matrix_identity']==b.digest(root/'DATASET_MANIFEST.json')
    assert (local/'folds/fold_2021/X.npy').is_symlink()
    assert (root/'rankgauss_cache/X_rankgauss.npy').exists()
    first=b.digest(local/'shared/X_rankgauss.npy')
    b.prepare_dataset(root,local)
    assert b.digest(local/'shared/X_rankgauss.npy')==first
    for mutation in ('content','missing','link','inventory'):
        target=root/'shared/y5.npy';original=target.read_bytes()
        if mutation=='content': target.write_bytes(b'corrupt')
        elif mutation=='missing':target.unlink()
        elif mutation=='link':target.unlink();target.symlink_to(local/'shared/y5.npy')
        else:(root/'shared/unexpected.npy').write_bytes(b'x')
        with pytest.raises(ValueError):b.validate_dataset(root)
        if target.is_symlink() or target.exists():target.unlink()
        target.write_bytes(original)
        (root/'shared/unexpected.npy').unlink(missing_ok=True)


def test_queue_and_frozen_contract():
    c=json.loads((b.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    p.validate_contract(c,list(range(2021,2027)),False)
    jobs=p.fit_queue(c)
    assert len(jobs)==64
    assert [sum(j['queue']==q for j in jobs) for q in ('Q1','Q2','Q3','Q4')]==[18,18,24,4]
    assert len({p.hashed(j) for j in jobs})==64
    c['status']='DRAFT'
    with pytest.raises(RuntimeError,match='DRAFT'):p.validate_contract(c,list(range(2021,2027)),False)


def test_warmup_values():
    assert [sr.warmup_cosine(i,100) for i in (0,14,15,100)]==pytest.approx([1/15,1,1,0])
    assert sr.warmup_cosine(57,100)==pytest.approx(.5*(1+np.cos(np.pi*42/85)))


def test_resume_bitwise_and_identity(tmp_path,monkeypatch):
    from test_seq_runner import build,CFG
    torch.set_num_threads(1)
    build(tmp_path);data=sr.SeqData(tmp_path);tr,va,ev=data.split_rows(15)
    cfg=dict(CFG,epochs=4,steps_per_epoch=2,d_model=8,n_layers=2,patience=9)
    contract=dict(matrix='synthetic',source='synthetic',seed=17,cfg=cfg)
    class Ledger:
        def check(self):pass
    monkeypatch.setattr(p,'telemetry',lambda:dict(gpu='CPU',temperature_c=60,clock_mhz=0,power_w=0))
    full,_=sr.train_one('gru',17,data,tr,va,cfg,'cpu',log=lambda *_:None)
    observer=p.Observer(tmp_path/'checkpoint',contract,Ledger(),device='cpu')
    def kill(event,payload):
        observer(event,payload)
        if event=='epoch' and payload['state']['epoch']==2:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        sr.train_one('gru',17,data,tr,va,cfg,'cpu',observer=kill,log=lambda *_:None)
    resumed_observer=p.Observer(tmp_path/'checkpoint',contract,Ledger(),device='cpu')
    restored=resumed_observer.load_resume()
    assert restored['state']['epoch']==2
    resumed,state=sr.train_one('gru',17,data,tr,va,cfg,'cpu',resume_state=restored,observer=resumed_observer,log=lambda *_:None)
    assert state['epoch']==4
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items())
    assert np.array_equal(sr.predict(full,data,ev,'cpu'),sr.predict(resumed,data,ev,'cpu'))
    for key in ('matrix','source','seed'):
        bad=dict(contract,**{key:'changed'})
        with pytest.raises(ValueError,match='identity'):
            p.Observer(tmp_path/'checkpoint',bad,Ledger(),device='cpu').load_resume()
    pointer=tmp_path/'checkpoint/checkpoint-pointer.json';saved=pointer.read_bytes()
    pointer.write_bytes(b'broken')
    with pytest.raises(ValueError):resumed_observer.load_resume()
    pointer.write_bytes(saved)
    entry=json.loads(saved)['latest'];(pointer.parent/entry['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):resumed_observer.load_resume()


def test_record_only_ledger_recovers_interruption(tmp_path):
    c=dict(policy='record_only',units_per_hour=0)
    with p.Ledger(tmp_path,c) as ledger:ledger.start({'fit':1},0)
    with p.Ledger(tmp_path,c) as ledger:
        assert not ledger.state['stopped'] and ledger.state['spent_hours']>=0
        ledger.start({'fit':1},0);ledger.check();ledger.end({'fit':1},False)
    with p.Ledger(tmp_path,c) as ledger:assert not ledger.state['stopped']


def test_pack_fold_symlink_escape(tmp_path):
    source=tmp_path/'outside';source.write_bytes(b'secret')
    root=tmp_path/'folds';root.mkdir();link=root/'X.npy';link.symlink_to(source)
    with pytest.raises(ValueError,match='root'):
        b.write_archive(tmp_path/'bundle.zip',{'folds/fold_2021/X.npy':link},{'folds_realpath':str(root)})


def test_common_rows_reports_and_label_fail_closed():
    from research_lab.p4_eval import common_rows
    a=pd.DataFrame(dict(fold=[2021]*3,date=['2021-01-01']*3,code=['a','b','c'],label_y5=[1.,2.,3.]))
    aligned,report=common_rows({'c0plus':a,'w60':a,'w250':a.iloc[1:]})
    assert all(len(x)==2 for x in aligned.values())
    assert report['systems']['c0plus']['excluded_fraction']==pytest.approx(1/3)
    bad=a.copy();bad.loc[1,'label_y5']=9
    with pytest.raises(ValueError,match='label'):common_rows({'a':a,'b':bad})
    with pytest.raises(ValueError,match='duplicate'):common_rows({'a':pd.concat([a,a])})


def test_permutation_summary_order_sensitive():
    from research_lab.p4_eval import permutation_summary
    dates=np.repeat(np.arange(3),60);codes=np.tile(np.arange(60).astype(str),3)
    y=np.tile(np.arange(60,dtype=float),3);base=y.copy();mamba=y.copy()
    out=permutation_summary(dates,codes,y,base,mamba,[y,-y]*5)
    assert out['R']==10 and out['order_sensitive']
    assert out['ic_std']>0 and len(out['daily'])==30
    assert all(0<=r['top50_overlap']<=1 for r in out['daily'])


def test_pax_header_bound_isolated(tmp_path,monkeypatch):
    import gzip,tarfile,io
    # Smaller than single-file and total limits; only PAX-specific bound can reject.
    monkeypatch.setattr(b,'MAX_EXTENDED_BYTES',64)
    hdr=tarfile.TarInfo('pax');hdr.type=tarfile.XHDTYPE;hdr.size=65
    archive=tmp_path/'pax.tar.gz';archive.write_bytes(gzip.compress(hdr.tobuf()+b'x'*512+b'\0'*1024))
    with archive.open('rb') as f:
        with pytest.raises(ValueError,match='header size limit'):b._raw_preflight(f)


def test_external_delivery_keys_only(tmp_path,monkeypatch):
    import zipfile
    root=dataset(tmp_path/'matrix_v1');b.dataset_manifest(root,enforce_x=False)
    monkeypatch.setattr(b,'DATASET_X',b.digest(root/'shared/X.npy'))
    ref=tmp_path/'reference';ref.mkdir()
    frame=pd.DataFrame(dict(fold=[2021,2021],date=pd.to_datetime(['2021-01-01']*2),code=['a','b'],label_y5=[.1,.2],score=[9.,8.],system=['c0plus']*2))
    frame.to_parquet(ref/'rank_average.parquet')
    (ref/'manifest.json').write_text(json.dumps({'outputs':{'rank_average.parquet':b.digest(ref/'rank_average.parquet')}}))
    out=tmp_path/'delivery'
    mf=b.delivery(root,list(range(2021,2027)),ref,b.ROOT/'environments/p4a-colab.requirements.lock.txt',out)
    with zipfile.ZipFile(out/'p4a-bundle.zip') as z:
        assert not any(n.startswith('folds/') or n.endswith('rank_average.parquet') for n in z.namelist())
        keys=pd.read_parquet(__import__('io').BytesIO(z.read('reference/keys.parquet')))
        assert list(keys)==['fold','date','code','label_y5'] and len(keys)==2
    assert mf['metadata']['external_dataset']==b.digest(root/'DATASET_MANIFEST.json')
    nb=json.loads((out/'p4a-colab.ipynb').read_text());settings={};exec(nb['cells'][0]['source'],settings)
    assert settings['EXPECTED_ARCHIVE']==b.digest(out/'p4a-bundle.zip')
    assert settings['MODE']=='FORMAL' and settings['DATA_DIR'].endswith('data/matrix_v1')
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'notebook','exec')
    assert not any("'pip','check'" in c['source'] for c in nb['cells'])


def test_orphan_and_partial_checkpoint_writes(tmp_path,monkeypatch):
    c={'seed':17,'matrix':'fixture'}
    class Ledger:
        def check(self):pass
    observer=p.Observer(tmp_path,c,Ledger(),device='cpu')
    (tmp_path/'epoch-orphan.pt').write_bytes(b'incomplete copy')
    with pytest.raises(ValueError,match='orphan'):observer.load_resume()
    (tmp_path/'epoch-orphan.pt').unlink()
    assert observer.load_resume() is None
    payload=dict(model={},opt={},sched={},state=dict(epoch=1,best=.1),best_sd={},rng_state={},numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())
    original=p.publish_bytes
    def crash(path,data):
        Path(path).parent.mkdir(exist_ok=True);Path(path).write_bytes(data[:10]);raise OSError('simulated Drive failure')
    monkeypatch.setattr(p,'publish_bytes',crash)
    with pytest.raises(OSError):observer('epoch',payload)
    assert not (tmp_path/'checkpoint-pointer.json').exists()
    monkeypatch.setattr(p,'publish_bytes',original)
    with pytest.raises(ValueError,match='orphan'):observer.load_resume()


def test_zip_ratio_limit_and_legal_matrix(tmp_path,monkeypatch):
    source=tmp_path/'zeros';source.write_bytes(b'\0'*(2*1024**2))
    archive=tmp_path/'zeros.zip'
    b.write_archive(archive,{'folds/fold_2021/X.npy':source},{'folds_realpath':str(tmp_path)})
    b.extract(archive,tmp_path/'valid')
    monkeypatch.setattr(b,'MAX_ZIP_COMPRESSION_RATIO',1)
    with pytest.raises(ValueError,match='compression ratio'):b.extract(archive,tmp_path/'invalid')


def test_resume_dropout_rng_bitwise(tmp_path,monkeypatch):
    from test_seq_runner import build,CFG,CausalConvMixer
    build(tmp_path);data=sr.SeqData(tmp_path);tr,va,ev=data.split_rows(15)
    cfg=dict(CFG,d_model=8,n_layers=1,n_cross=1,dropout=.4,epochs=3,steps_per_epoch=2,patience=9)
    captures=[]
    def stop(event,payload):
        if event=='epoch' and payload['state']['epoch']==1:
            import copy
            captures.append(copy.deepcopy(dict(payload,cuda_rng=None)))
            raise KeyboardInterrupt()
    full,_=sr.train_one('crossstock',17,data,tr,va,cfg,'cpu',mixer_factory=CausalConvMixer,log=lambda *_:None)
    with pytest.raises(KeyboardInterrupt):
        sr.train_one('crossstock',17,data,tr,va,cfg,'cpu',mixer_factory=CausalConvMixer,observer=stop,log=lambda *_:None)
    torch.rand(100);np.random.rand(100)
    resumed,_=sr.train_one('crossstock',17,data,tr,va,cfg,'cpu',mixer_factory=CausalConvMixer,resume_state=captures[0],log=lambda *_:None)
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items())
    assert np.array_equal(sr.predict(full,data,ev,'cpu'),sr.predict(resumed,data,ev,'cpu'))


def test_cpu_full_queue_and_validated_skip(tmp_path,monkeypatch):
    import io
    root=dataset(tmp_path/'matrix');b.dataset_manifest(root,enforce_x=False)
    monkeypatch.setattr(b,'DATASET_X',b.digest(root/'shared/X.npy'))
    ref=tmp_path/'ref';ref.mkdir()
    frame=pd.DataFrame([dict(fold=y,date=pd.Timestamp(f'{y}-02-01'),code=c,label_y5=float(i),score=float(i),system='c0plus') for y in range(2021,2027) for i,c in enumerate(['a','b'])])
    frame.to_parquet(ref/'rank_average.parquet');(ref/'manifest.json').write_text(json.dumps({'outputs':{'rank_average.parquet':b.digest(ref/'rank_average.parquet')}}))
    b.pack(root,list(range(2021,2027)),ref,b.ROOT/'environments/p4a-colab.requirements.lock.txt',tmp_path/'bundle.zip')
    b.extract(tmp_path/'bundle.zip',tmp_path/'bundle');b.prepare_dataset(root,tmp_path/'ssd')
    actual_path=Path
    monkeypatch.setattr(p,'Path',lambda x:tmp_path if x=='/content/drive/MyDrive' else actual_path(x))
    monkeypatch.setattr(p.os.path,'ismount',lambda _:True)
    monkeypatch.setattr(p,'require_cuda',lambda:None)
    monkeypatch.setattr(p,'environment',lambda *a,**k:dict(device='CPU_BOUNDARY_DOUBLE'))
    calls=[];permutations=[]
    def fake_train(directory,year,arm,seed,cfg,reference,observer):
        calls.append((year,arm,seed,cfg['window'],cfg['lr']));observer.device='cpu'
        weights={'weight':torch.tensor([1.])}
        for epoch in (1,2):
            observer('epoch',dict(model=weights,opt={},sched={},state=dict(epoch=epoch,best=.2,bad=0,history=[{}]*epoch),best_sd=weights,rng_state={},numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()))
        stream=io.BytesIO();torch.save(dict(contract=observer.contract,model=weights,best_val_ic5=.2),stream);p.publish_bytes(observer.root/'best.pt',stream.getvalue())
        f=reference.copy()
        if cfg['window']>60:f=f.iloc[1:]
        f=f.assign(score=1.,seed=seed,system=arm,window=cfg['window'],lr=cfg['lr'])
        return f,dict(history=[{},{}],best_val_ic5=.2,eval_rows=len(f))
    monkeypatch.setattr(p,'train_fit',fake_train)
    def fake_permutation(directory,year,*args):permutations.append(year);return {'CPU_BOUNDARY_DOUBLE':True}
    monkeypatch.setattr(p,'permutation_fit',fake_permutation)
    drive=tmp_path/'drive'
    first=p.run(tmp_path/'bundle',drive,data_dir=tmp_path/'ssd')
    assert len(calls)==64 and permutations==[2021,2022,2023] and len(first['jobs'])==64
    second=p.run(tmp_path/'bundle',drive,data_dir=tmp_path/'ssd')
    assert len(calls)==64 and len(second['jobs'])==64 and len(permutations)==3
    from research_lab.p4_eval import load_run
    scores,binding=load_run(next(drive.glob('run-*')))
    assert len(scores)==104 and len(binding['queue'])==64


def test_eval_families_holm_and_shared_support(monkeypatch):
    from research_lab import p4_eval as ev
    c=json.loads((b.ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    rows=[];base=[]
    for year in (2021,2022,2023):
        for i,code in enumerate(['a','b','c','d']):
            key=dict(fold=year,date=pd.Timestamp(f'{year}-02-01'),code=code,label_y5=float(i))
            base.append(dict(key,system='c0plus',score=float(i)))
            for arm in ('mamba2','crossstock','gru'):
                for w in ((60,) if arm=='crossstock' else (60,120,250)):
                    if w>60 and code=='a':continue
                    for seed in (17,29):rows.append(dict(key,system=arm,score=float(i),seed=seed,window=w,lr=3e-4))
    supports=[]
    def controlled_portfolios(frames,prices,samples):
        supports.append({n:len(f) for n,f in frames.items()});out={}
        for name,f in frames.items():
            value=0. if name=='c0plus' else .001 if name=='gru' else .002 if name=='mamba2' else .003
            for year in f.fold.unique():out[(int(year),name)]=dict(status='COMPUTED',daily_returns=[dict(date=str(i),net=value,benchmark=0.) for i in range(40)])
        return out
    monkeypatch.setattr(ev,'portfolio_rows',controlled_portfolios)
    result=ev.evaluate(pd.DataFrame(rows),pd.DataFrame(base),pd.DataFrame(),c)
    assert [result['hypotheses'][f]['m'] for f in ('A','B','C')]==[3,2,4]
    assert result['hypotheses']['A']['tests']['H-A1']['registered_threshold_met']
    assert not result['independent_confirmation']
    assert result['intersections']['C']['common_rows']==9
    assert all(n==9 for n in supports[2].values())
    assert result['intersections']['C']['systems']['c0plus']['excluded_fraction']==.25
    assert result['hypotheses']['A']['tests']['H-A2']['replication']
    partial=ev.evaluate(pd.DataFrame(rows),pd.DataFrame(base),pd.DataFrame(),c,samples=5)
    assert not partial['hypotheses']['A']['tests']['H-A1']['registered_threshold_met']


def test_p4_portfolio_reuses_p2_engine_exactly(tmp_path):
    from research_lab import p4_eval as ev,portfolio_eval as pe
    days=pd.bdate_range('2023-01-02',periods=65)
    prices=pd.DataFrame([dict(date=d,code=code,open=100.+i,close=100.+i,volume=1e8) for i,d in enumerate(days) for code in ('A','B')])
    frames={name:pd.DataFrame([dict(fold=2023,system=name,date=d,code=code,label_y5=v,score=v) for d in days[:-1] for code,v in [('A',1.),('B',2.)]]) for name in ('mamba2','crossstock','c0plus')}
    expected=pe.evaluate_p3(pd.concat(frames.values()),prices,pd.DataFrame(),{},bootstrap_samples=5,part_dir=tmp_path/'p3')
    actual=ev.portfolio_rows(frames,prices,5)
    assert all(r['status']=='COMPUTED' for r in actual.values())
    for row in expected['results']:
        assert actual[(row['fold'],row['system'])]['daily_returns']==row['daily_returns']


def test_record_ledger_preserves_attempt_telemetry_and_seconds(tmp_path):
    with p.Ledger(tmp_path,dict(policy='record_only',units_per_hour=0)) as ledger:
        ledger.start({'fit':1},0)
        ledger.observe(dict(gpu='CPU_DOUBLE',temperature_c=70.,clock_mhz=1000.,power_w=50.,observed_at=1.))
        ledger.observe(dict(gpu='CPU_DOUBLE',temperature_c=75.,clock_mhz=900.,power_w=55.,observed_at=2.))
        ledger.end({'fit':1},False)
        attempt=ledger.state['last_attempt']
        assert attempt['wall_seconds']>=0
        assert attempt['telemetry']['temperature_c_max']==75.
        assert attempt['telemetry']['clock_mhz_min']==900.
    events=[json.loads(f.read_text()) for f in sorted(tmp_path.glob('event-*.json'))]
    assert events[-1]['state']['last_attempt']==attempt


def test_thermal_stop_survives_process_loss_before_end(tmp_path):
    c=dict(policy='record_only',units_per_hour=0)
    with p.Ledger(tmp_path,c) as ledger:
        ledger.start({'fit':1},0)
        ledger.observe(dict(gpu='CPU_DOUBLE',temperature_c=85.,clock_mhz=900.,power_w=50.,observed_at=1.))
        # Simulate process loss before end(); record-only recovery must not clear thermal protection.
    with pytest.raises(p.BudgetStop,match='STOP'):
        with p.Ledger(tmp_path,c):pass
