"""Synthetic contract checks only; no research performance evaluation."""
import numpy as np
import pandas as pd
import pytest
from tools import step3 as s


def events():
    p=pd.date_range('2013-01-01',periods=28,freq='MS')
    return pd.DataFrame(dict(stock_id='X',period=p,available_date=p+pd.offsets.MonthBegin(1),
        revenue=np.arange(100.,128.),scope='C',currency='TWD',unit='yuan',
        comparability_id='verified-fixture',version=0,source_row=np.arange(28)))


def test_release_day_excluded_and_features_reconstructed():
    f=events(); days=pd.bdate_range('2013-01-01','2015-06-01')
    out=s.monthly_events(f,days)
    assert out.session.iloc[0]==days.searchsorted(f.available_date.iloc[0],side='right')
    r=out.iloc[-1]
    assert r.monthly_yoy==pytest.approx(127/115-1)
    assert r.monthly_cumulative_yoy==pytest.approx(sum(range(124,128))/sum(range(112,116))-1)
    assert r.monthly_high12==1
    assert r.monthly_surprise==pytest.approx((127-115*(1+np.median([126/114-1,125/113-1,124/112-1])))/(115*(1+np.median([126/114-1,125/113-1,124/112-1]))))


def test_same_session_shuffle_and_late_old_month():
    f=events(); f.loc[0,'available_date']=pd.Timestamp('2015-05-02')
    f.loc[27,'available_date']=pd.Timestamp('2015-05-03')
    days=pd.bdate_range('2013-01-01','2015-06-01')
    a=s.monthly_events(f,days); b=s.monthly_events(f.sample(frac=1,random_state=3),days)
    pd.testing.assert_frame_equal(a,b)
    assert not a.duplicated(['stock_id','session']).any()
    assert a.iloc[-1].monthly_yoy==pytest.approx(127/115-1)
    # Late old month cannot change previously materialized events.
    pd.testing.assert_frame_equal(a.iloc[:-1].reset_index(drop=True),s.monthly_events(f.iloc[1:27],days).reset_index(drop=True))


def test_version_collision_stop_and_identical_dedup():
    f=events(); days=pd.bdate_range('2013-01-01','2015-06-01')
    assert len(s.monthly_events(pd.concat([f,f.iloc[:1]]),days))==len(f)
    bad=f.iloc[:1].copy();bad.revenue+=1
    with pytest.raises(ValueError,match='STOP.*collision'):s.monthly_events(pd.concat([f,bad]),days)


def test_missing_month_unknown_scope_and_year_start():
    f=events(); days=pd.bdate_range('2013-01-01','2015-06-01')
    a=s.monthly_events(f[f.period!=pd.Timestamp('2015-02-01')],days).iloc[-1]
    assert np.isnan(a.monthly_cumulative_yoy) and np.isnan(a.monthly_high12)
    f.comparability_id='UNKNOWN';a=s.monthly_events(f,days)
    assert a[s.FEATURES].isna().all().all()
    a=s.monthly_events(events(),days)
    assert np.isnan(a.iloc[0].monthly_cumulative_yoy)
    assert a.iloc[24].monthly_cumulative_yoy==pytest.approx(124/112-1)


def test_nonpositive_denominator():
    f=events();f.loc[15,'revenue']=0
    r=s.monthly_events(f,pd.bdate_range('2013-01-01','2015-06-01')).iloc[-1]
    assert np.isnan(r.monthly_yoy) and np.isnan(r.monthly_surprise)


def test_label_exact_h20_and_missing_tail():
    o=np.arange(1.,46)[:,None]; c=o+10;valid=np.ones_like(o,bool)
    y=s.h20_labels(o,c,valid)
    assert y[0,0]==pytest.approx(c[20,0]/o[1,0]-1)
    assert np.isnan(y[-20:]).all()
    valid[10]=False;assert np.isnan(s.h20_labels(o,c,valid)[0,0])
    o[1]=0;assert np.isnan(s.h20_labels(o,c,np.ones_like(valid))[0,0])


def test_fold_strict_gap_and_warmup():
    d=pd.bdate_range('2013-01-01','2015-03-01').to_numpy();base=np.ones(len(d),int);base[:60]=0
    tr,ev=s.fold_rows(d,base,2015,np.ones(len(d),bool))
    origin=np.flatnonzero(d.astype('datetime64[Y]')==np.datetime64('2015'))[0]
    assert tr.max()==origin-22 and (tr+21<origin).all() and tr.min()==60
    assert ev.min()==origin


def test_rank_full_day_then_shared_sampling():
    y=np.array([1.,1.,4.,2.,5.,9.]);t=np.array([0,0,0,1,1,1]);idx=np.arange(6)
    rows,target=s.sample_training(idx,y,t,17,cap=4)
    assert len(rows)==4 and len(set(rows))==4 and np.all(np.diff(rows)>0)
    expected=np.array([-1/6,-1/6,1/3,-1/3,0,1/3])
    np.testing.assert_allclose(target,expected[rows],atol=1e-7)


def test_w60_gap_and_future_invariance():
    present=np.ones((70,2),bool);present[15,1]=False
    a=s.w60_eligible(present)
    assert not a[:59].any() and a[59:,0].all() and not a[:,1].any()
    present[65:]=False;np.testing.assert_array_equal(s.w60_eligible(present)[:65],a[:65])


def test_base_equal_nan_and_key_alignment():
    a=np.zeros((3,185));a[0,0]=np.nan;b=np.column_stack([a,np.ones((3,5))]);keys=np.array([[1,1],[1,2],[1,3]])
    s.validate_arms(keys,a,keys,b)
    b[1,0]=1
    with pytest.raises(ValueError,match='STOP'):s.validate_arms(keys,a,keys,b)
    with pytest.raises(ValueError,match='STOP'):s.validate_arms(keys,a,keys[::-1],np.column_stack([a,np.ones((3,5))]))


def test_gate_fail_closed_before_fit(tmp_path):
    with pytest.raises(ValueError,match='STOP'):s.require_admission({'status':'STOP'})
    p=tmp_path/'x';p.write_text('x')
    with pytest.raises(ValueError,match='STOP'):s.verify_file(p,'0'*64,1)


def test_selection_unknown_is_not_suspension_or_delisting():
    d=pd.bdate_range('2020-01-01',periods=70);m=np.ones((70,2),bool);v=m.copy();v[65,1]=False
    r=s.selection_table(d,m,v)
    assert r[0]['members']==140 and r[0]['w60']==22
    assert r[0]['suspension']==None and r[0]['delisted']==None
    assert sum(r[0]['primary_reasons'].values())==140


def test_mutation_release_inclusive_is_killed(monkeypatch):
    real=s.effective_session
    monkeypatch.setattr(s,'effective_session',lambda release,days:int(days.searchsorted(release,side='left')))
    with pytest.raises(AssertionError):test_release_day_excluded_and_features_reconstructed()
    monkeypatch.setattr(s,'effective_session',real)


def test_mutation_exit_off_by_one_is_killed(monkeypatch):
    real=s.h20_labels
    def bad(o,c,v):
        r=real(o,c,v);r[0]=c[21]/o[1]-1;return r
    monkeypatch.setattr(s,'h20_labels',bad)
    with pytest.raises(AssertionError):test_label_exact_h20_and_missing_tail()


def test_admission_cannot_be_claimed_by_status_only():
    with pytest.raises(ValueError,match='STOP'):s.require_admission({'status':'PASS'})


def test_atomic_artifact_resume_corruption_and_identity(tmp_path,monkeypatch):
    from tools import step3_runner as r
    monkeypatch.setattr(r,'private_dir',lambda p:Path(p))
    root=tmp_path/'jobs';root.mkdir(); request={'contract':s.SPEC_SHA,'seed':17,'matrix':'fixture'}
    files={'keys.npy':np.array([[1,1],[1,2]]),'sample.npy':np.array([2,5]),'prediction.npy':np.zeros(2)}
    p=r.publish(root,'job',request,files,b'fixture-model-not-fitted')
    assert r.resume(p,request)['state']=='COMPLETE'
    with pytest.raises(ValueError,match='STOP'):r.resume(p,{**request,'matrix':'different'})
    (p/'keys.npy').write_bytes(b'broken')
    with pytest.raises(ValueError,match='STOP'):r.resume(p,request)


def test_incomplete_orphan_is_never_complete(tmp_path,monkeypatch):
    from tools import step3_runner as r
    monkeypatch.setattr(r,'private_dir',lambda p:Path(p))
    p=tmp_path/'partial';p.mkdir();(p/'keys.npy').write_bytes(b'partial')
    with pytest.raises(ValueError,match='STOP'):r.resume(p,{'seed':17})


def test_six_output_keys_reject_missing_seed_duplicate_or_nonfinite():
    from tools.step3_runner import validate_outputs
    keys=np.array([[1,1],[1,2]])
    # No prediction values are needed to test key-only family validation.
    outputs={(a,seed):keys.copy() for a in 'AB' for seed in s.SEEDS}
    validate_outputs(outputs)
    del outputs['B',43]
    with pytest.raises(ValueError,match='STOP'):validate_outputs(outputs)


def test_feature_support_constant_or_all_missing_stop():
    from tools.step3_runner import feature_support
    with pytest.raises(ValueError,match='STOP'):feature_support(np.full((10,5),np.nan))
    with pytest.raises(ValueError,match='STOP'):feature_support(np.ones((10,5)))

from pathlib import Path


def test_fold_uses_frozen_calendar_origin_even_if_first_member_session_absent():
    calendar=pd.bdate_range('2013-01-01','2015-03-01').to_numpy()
    first=np.flatnonzero(calendar.astype('datetime64[Y]')==np.datetime64('2015'))[0]
    dates=np.delete(calendar,first)
    train,ev=s.fold_rows(dates,np.ones(len(dates),int),2015,np.ones(len(dates),bool),calendar=calendar)
    assert calendar.searchsorted(dates[train[-1]])==first-22


def test_missing_doc_or_tampered_raw_stays_stop(tmp_path,monkeypatch):
    # Does not require licensed inputs; no synthetic PASS can slip through.
    monkeypatch.setattr(s,'private_dir',lambda p:Path(p))
    output=tmp_path/'out';output.mkdir()
    result=s.admission(tmp_path,output)
    assert result['status']=='STOP' and not result['documentation_clauses']
    assert any('identity' in v for v in result['stops'])


def test_partial_orphan_recovery_and_seed_isolation(tmp_path,monkeypatch):
    from tools import step3_runner as r
    monkeypatch.setattr(r,'private_dir',lambda p:Path(p))
    arrays={'keys.npy':np.array([[1,1]]),'sample.npy':np.array([2]),'prediction.npy':np.zeros(1)}
    req={'seed':17,'contract':s.SPEC_SHA}
    p=r.publish(tmp_path,'fixture',req,arrays,b'not-a-trained-model')
    stage=tmp_path/'.partial-fixture';p.rename(stage)
    with pytest.raises(ValueError,match='STOP'):r.resume(stage,{**req,'seed':29})
    final=r.recover_orphan(stage,tmp_path/'fixture',req)
    assert r.resume(final,req)['state']=='COMPLETE'


def matrix_fixture():
    from types import SimpleNamespace
    from research_lab.c0plus import CORE
    names=list(CORE)+[f'fixture_{i}' for i in range(48-len(CORE))]
    n=70*3;rng=np.random.default_rng(17)
    return SimpleNamespace(X=rng.normal(size=(n,48)).astype(np.float32),
        date_idx=np.repeat(np.arange(70),3),stock_idx=np.tile(np.arange(3),70),
        ends=np.arange(3,n+1,3),n_stocks=3,feature_names=names)


def test_c0plus_base_exact_and_future_feature_mutation():
    from tools.step3_runner import build_arms
    from research_lab.c0plus import build_features
    data=matrix_fixture();rows=np.arange(180,195);monthly=np.ones((len(data.X),5),np.float32)
    a,b,names=build_arms(data,rows,monthly)
    expected,fn=build_features(data,rows,data.feature_names)
    np.testing.assert_array_equal(a,expected);assert names[:185]==fn
    assert a.shape==(15,185) and b.shape==(15,190)
    data.X[195:]=1e8
    changed=build_arms(data,rows,monthly)[0]
    np.testing.assert_array_equal(a,changed)


def test_publish_kill_restart_recovers_complete_orphan(tmp_path,monkeypatch):
    from tools import step3_runner as r
    import subprocess,sys,textwrap
    monkeypatch.setattr(r,'private_dir',lambda p:Path(p))
    # Kill after durable complete manifest but before atomic rename. No model fitting.
    code=textwrap.dedent('''
        import os,signal,sys,numpy as np
        from pathlib import Path
        from tools import step3_runner as r
        r.private_dir=lambda p:Path(p)
        real=r._sync_dir
        def kill(p):
            real(p)
            os.kill(os.getpid(),signal.SIGKILL)
        r._sync_dir=kill
        r.publish(Path(sys.argv[1]),'job',{'seed':17},
            {'keys.npy':np.array([[1,1]]),'sample.npy':np.array([4]),'prediction.npy':np.zeros(1)},b'fixture')
    ''')
    process=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True)
    assert process.returncode==-9
    assert not (tmp_path/'job').exists()
    stage=list(tmp_path.glob('.partial-*'));assert len(stage)==1
    r.recover_orphan(stage[0],tmp_path/'job',{'seed':17})
    assert r.resume(tmp_path/'job',{'seed':17})['state']=='COMPLETE'


def test_selection_uses_2b_price_validation_start():
    from tools.step3_audit import validate_selection_prices
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2012-12-28','2013-01-02']),code='X',
        open=[10.,20.],high=[10.,20.],low=[10.,20.],close=[10.,20.],volume=100.,adj_close=np.nan))
    result=validate_selection_prices(frame)
    assert len(result)==1 and result.observation_valid.all()


def test_queue_fixed_and_runner_has_no_evaluation_calls():
    import ast,inspect
    from tools import step3_runner as r
    jobs=r.queue()
    assert len(jobs)==72 and len(set(jobs))==72
    assert {seed for _,seed,_ in jobs}=={17,29,43}
    tree=ast.parse(inspect.getsource(r))
    assert not any(isinstance(n,ast.ImportFrom) and n.module and ('evaluation' in n.module or 'tej_diag' in n.module) for n in ast.walk(tree))
    with pytest.raises(ValueError,match='STOP'):
        r.run_fold({'status':'STOP'},None,None,None,None,None,None,None,None,None,None)


def test_missing_reason_audit_and_dedup_lineage_separate_from_features():
    f=events();duplicate=f.iloc[:1].copy();duplicate.source_row=999
    out=s.monthly_events(pd.concat([f,duplicate]),pd.bdate_range('2013-01-01','2015-06-01'))
    assert list(out)==['stock_id','session']+s.FEATURES
    assert out.attrs['deduplication_lineage'][0]['source_rows']==['0','999']
    assert 'monthly_yoy' in out.attrs['missing_reasons'][0]['reasons']


def test_unknown_boolean_evidence_cannot_pass_via_truthiness():
    with pytest.raises(ValueError,match='STOP'):
        s.h20_labels(np.ones((30,1)),np.ones((30,1)),np.full((30,1),np.nan))
    with pytest.raises(ValueError,match='STOP'):
        s.w60_eligible(np.full((70,1),np.nan))
    with pytest.raises(ValueError,match='STOP'):
        s.selection_table(pd.bdate_range('2020-01-01',periods=30),np.ones((30,1),bool),np.full((30,1),None))


def test_pre2013_comparisons_are_nan_even_with_matching_scope():
    f=events();old=f.iloc[:12].copy();old.period-=pd.DateOffset(years=1);old.available_date-=pd.DateOffset(years=1);old.source_row+=100
    out=s.monthly_events(pd.concat([old,f]),pd.bdate_range('2012-01-01','2015-06-01'))
    first=out.iloc[0]
    assert out.session.min()>250
    assert first[s.FEATURES].isna().all()
    assert out.iloc[11][['monthly_yoy','monthly_cumulative_yoy','monthly_high12','monthly_surprise']].isna().all()


def test_tej_exported_comparators_cannot_affect_features_mutation(monkeypatch):
    f=events();f['去年單月營收(千元)']=1.;f['累計營收(千元)']=1.
    days=pd.bdate_range('2013-01-01','2015-06-01')
    expected=s.monthly_events(f,days)
    f['去年單月營收(千元)']=999999.;f['累計營收(千元)']=-10.
    pd.testing.assert_frame_equal(expected,s.monthly_events(f,days))
    real=s.monthly_events
    def mutant(frame,calendar):
        out=real(frame,calendar);out['monthly_yoy']=frame['去年單月營收(千元)'].iloc[0];return out
    monkeypatch.setattr(s,'monthly_events',mutant)
    with pytest.raises(AssertionError):pd.testing.assert_frame_equal(expected,s.monthly_events(f,days))


def test_bundle_gate_rechecks_all_identities(tmp_path):
    import json
    from tools.step3 import BUNDLE_CATEGORIES
    files={}
    for category in BUNDLE_CATEGORIES:
        p=tmp_path/category;p.write_text(category)
        files[category]={'path':str(p),'sha256':s.sha256(p),'bytes':p.stat().st_size}
    m=tmp_path/'bundle.json';m.write_text(json.dumps({'contract':s.SPEC_SHA,'files':files,'jobs':{}}))
    gate={'status':'PASS','contract_sha256':s.SPEC_SHA,'bundle':{'path':str(m),'sha256':s.sha256(m),'bytes':m.stat().st_size}}
    with pytest.raises(ValueError,match='STOP'):s.require_admission(gate)
    (tmp_path/'labels').write_text('mutated')
    with pytest.raises(ValueError,match='STOP'):s.require_admission(gate)
    with pytest.raises(ValueError,match='STOP'):s.require_admission({'status':'PASS'})


def test_real_adapter_reconstructs_first_month_and_rejects_currency():
    from tools.step3_prepare import admitted_events
    f=events().rename(columns={'source_row':'input_row_number'})
    f['單月營收(千元)']=np.arange(100.,128.)
    f['去年單月營收(千元)']=1.;f['累計營收(千元)']=-999.
    f['currency']='TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING'
    pairs=pd.DataFrame(dict(input_row_number=f.input_row_number,agreement=True))
    a=admitted_events(f,pairs)
    np.testing.assert_array_equal(a.revenue,f['單月營收(千元)']*1000)
    f['去年單月營收(千元)']=np.nan;f['累計營收(千元)']=1e100
    pd.testing.assert_frame_equal(a,admitted_events(f,pairs))
    f.loc[0,'currency']='USD'
    assert np.isnan(admitted_events(f,pairs).revenue.iloc[0])
    f.loc[0,'currency']='EUR'
    assert np.isnan(admitted_events(f,pairs).revenue.iloc[0])


def test_fit_rejects_array_mutation_before_lightgbm(tmp_path,monkeypatch):
    from tools import step3_runner as r
    monkeypatch.setattr(r,'require_admission',lambda gate:{'year':2026,'seed':17,'jobs':{}})
    with pytest.raises(ValueError,match='STOP job arrays'):
        r.fit_job({'bundle':{'sha256':'frozen'}},tmp_path,2026,17,'A',np.array([[0,0],[0,1]]),
            np.array([1,2]),np.array([-.2,.2]),np.zeros((2,185)),np.zeros((2,185)),'frozen')


def test_review_minimal_bundle_rejected(tmp_path):
    import json
    files={}
    for category in s.BUNDLE_CATEGORIES:
        p=tmp_path/category;p.write_text(category)
        files[category]=dict(path=str(p),sha256=s.sha256(p),bytes=p.stat().st_size)
    p=tmp_path/'bundle.json';p.write_text(json.dumps(dict(contract=s.SPEC_SHA,files=files,jobs={})))
    gate=dict(status='PASS',contract_sha256=s.SPEC_SHA,bundle=dict(path=str(p),sha256=s.sha256(p),bytes=p.stat().st_size))
    with pytest.raises(ValueError,match='STOP'):s.require_admission(gate)


def test_review_resume_missing_prediction(tmp_path,monkeypatch):
    import json
    from tools import step3_runner as r
    monkeypatch.setattr(r,'private_dir',lambda p:Path(p))
    p=r.publish(tmp_path,'job',{},dict(**{'keys.npy':np.array([[0,0]]),'sample.npy':np.array([0]),'prediction.npy':np.array([0.])}),b'fixture')
    m=json.loads((p/'manifest.json').read_text());del m['files']['prediction.npy']
    (p/'prediction.npy').unlink();(p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='STOP'):r.resume(p,{})


@pytest.mark.parametrize('bad', ['labels','session','inf','-inf'])
def test_review_evaluation_rejects_before_any_scoring(bad,monkeypatch):
    from tools import step3_evaluation as e
    def forbidden(*args,**kwargs):raise AssertionError('scoring must never run')
    monkeypatch.setattr(e,'_spearman',forbidden)
    monkeypatch.setattr(e,'_circular_paired_interval',forbidden)
    keys=np.array([[0,0],[0,1]])
    family={(a,seed):keys.copy() for a in 'AB' for seed in s.SEEDS}
    y=np.array([1.,2.]);t=np.array([0,0]);label_keys=keys.copy()
    if bad=='labels':label_keys=label_keys[::-1]
    if bad=='session':t[0]=1
    if bad in ('inf','-inf'):y[0]=float(bad)
    with pytest.raises(ValueError,match='STOP'):
        e.evaluate(family,{k:np.zeros(2) for k in family},y,t,['2015-01-01','2015-01-02'],
                   label_keys=label_keys,frozen_labels=y.copy(),independent_execution_authorized=True)


def test_review_cross_source_adapter():
    from tools.step3_prepare import admitted_events,cross_source_evidence
    f=events().rename(columns={'source_row':'input_row_number'});f['單月營收(千元)']=np.arange(100.,128.)
    f['證券代碼']='1234 fixture';f.currency='TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING'
    raw=pd.DataFrame(dict(stock_id='X',revenue_year=f.period.dt.year,revenue_month=f.period.dt.month,revenue=f['單月營收(千元)']*1000))
    raw.loc[0,'revenue']*=2
    pairs,summary=cross_source_evidence(f,raw)
    a=admitted_events(f,pairs)
    assert a.currency.iloc[0]=='UNIT_UNVERIFIED' and np.isnan(a.revenue.iloc[0])
    assert (a.currency.iloc[1:]=='NTD_BY_CROSS_SOURCE_AGREEMENT').all()
    assert summary['overall']['gt_1pct']==1


def test_review_human_decision_record_validation(tmp_path):
    from tools.step3 import human_source_acceptance
    import json
    p=tmp_path/'decision.json'
    for record in [{},{'decision_date':'bad','decision_text':'accept','decision':'ACCEPT_HISTORICAL_SOURCE_GATE'},
                   {'decision_date':'2026-11-26','decision_text':'','decision':'ACCEPT_HISTORICAL_SOURCE_GATE'}]:
        p.write_text(json.dumps(record))
        with pytest.raises(ValueError,match='STOP'):human_source_acceptance(p)
    p.write_text(json.dumps(dict(decision_date='2026-11-26',decision_text='Human 接受所列歷史來源限制',decision='ACCEPT_HISTORICAL_SOURCE_GATE')))
    assert human_source_acceptance(p)['sha256']==s.sha256(p)
    with pytest.raises(ValueError,match='STOP'):s.require_admission(dict(status='HUMAN_DECISION_REQUIRED',contract_sha256=s.SPEC_SHA))


def complete_bundle_fixture(tmp_path):
    import json
    p=tmp_path/'data';p.write_text('fixture')
    info=dict(path=str(p),sha256=s.sha256(p),bytes=p.stat().st_size)
    names=s.BUNDLE_CATEGORIES|{'matrix_dates','matrix_ends','stock_index','price_manifest','finmind_revenue','source_manifest'}
    names|={'code_'+p.name for p in (s.ROOT/'tools').glob('step3*.py') if p.name!='step3.py'}
    files={k:info.copy() for k in names}
    jobs={s.job_id(y,seed,a):dict(year=y,seed=seed,arm=a,
        **{role+'_file':'matrix' for role in ('sample','target','keys','fit','eval','splits','label_keys','eval_labels')},
        **{role+'_sha256':'0'*64 for role in ('sample','target','keys','fit','eval')})
        for y in s.YEARS for seed in s.SEEDS for a in 'AB'}
    bundle=dict(contract=s.SPEC_SHA,source_gate='HUMAN_DECISION_REQUIRED',source_commit='fixture',source_identities=files.copy(),files=files,jobs=jobs)
    return bundle


@pytest.mark.parametrize('missing',['job','target','fit','eval','source','label_keys'])
def test_review_full_schema_missing_roles_stop(tmp_path,missing):
    bundle=complete_bundle_fixture(tmp_path);s.validate_bundle_schema(bundle)
    job=bundle['jobs'][s.job_id(2015,17,'A')]
    if missing=='job':del bundle['jobs'][s.job_id(2015,17,'A')]
    elif missing=='source':del bundle['source_identities']['prices']
    else:del job[missing+'_file']
    with pytest.raises((ValueError,KeyError)):s.validate_bundle_schema(bundle)


def test_review_pending_gate_acceptance_never_overrides_schema_or_stop(tmp_path,monkeypatch):
    import json
    bundle=complete_bundle_fixture(tmp_path)
    path=tmp_path/'bundle.json';path.write_text(json.dumps(bundle))
    gate=dict(status='HUMAN_DECISION_REQUIRED',contract_sha256=s.SPEC_SHA,
              bundle=dict(path=str(path),sha256=s.sha256(path),bytes=path.stat().st_size))
    decision=tmp_path/'decision.json';decision.write_text(json.dumps(dict(decision_date='2026-11-26',
        decision_text='Synthetic Human acceptance fixture only',decision='ACCEPT_HISTORICAL_SOURCE_GATE')))
    monkeypatch.setattr(s,'verify_source_provenance',lambda b:None)
    with pytest.raises(ValueError,match='STOP'):s.require_admission(gate)
    assert len(s.require_admission(gate,decision)['jobs'])==72
    with pytest.raises(ValueError,match='STOP'):s.require_admission({**gate,'status':'STOP'},decision)
    del bundle['jobs'][s.job_id(2015,17,'A')];path.write_text(json.dumps(bundle))
    gate['bundle'].update(sha256=s.sha256(path),bytes=path.stat().st_size)
    with pytest.raises(ValueError,match='STOP'):s.require_admission(gate,decision)


def test_review_label_values_must_match_frozen_labels(monkeypatch):
    from tools import step3_evaluation as e
    monkeypatch.setattr(e,'_spearman',lambda *a:pytest.fail('scoring forbidden'))
    keys=np.array([[0,0],[0,1]]);family={(a,seed):keys for a in 'AB' for seed in s.SEEDS}
    with pytest.raises(ValueError,match='STOP'):
        e.evaluate(family,{k:np.zeros(2) for k in family},[2.,1.],[0,0],['2015-01-01'],
            label_keys=keys,frozen_labels=[1.,2.],independent_execution_authorized=True)
