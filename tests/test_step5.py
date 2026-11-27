"""Synthetic Step5 formula and causal replay oracles; no real labels."""
import numpy as np
import pandas as pd
import pytest
from tools import step5 as s
from tools.step5_features import replay, align_events


def fixture():
    periods=pd.date_range('2013-01-01','2017-05-01',freq='MS')
    f=pd.DataFrame(dict(stock_id='X',period=periods,available_date=periods+pd.offsets.MonthEnd(0)+pd.Timedelta(days=10),revenue=100.,scope='C',currency='NTD',unit='yuan',comparability_id='known',version=0,source_row=np.arange(len(periods))))
    return f,pd.bdate_range('2013-01-01','2017-08-01')


def latest(f,days):
    return replay(f,days).iloc[-1]


def test_contract():
    assert [len(s.FEATURE_ORDER[a]) for a in s.ARMS]==[190,194]
    assert len(s.YEARS)*len(s.SEEDS)*len(s.ARMS)==72
    s.verify_file(s.SPEC,s.SPEC_SHA)
    s.frozen_decision()


def test_formulas_and_seasonal_support():
    f,d=fixture()
    f.loc[f.period==pd.Timestamp('2017-02-01'),'revenue']=110
    f.loc[f.period==pd.Timestamp('2017-03-01'),'revenue']=120
    f.loc[f.period==pd.Timestamp('2017-04-01'),'revenue']=130
    f.loc[f.period==pd.Timestamp('2017-05-01'),'revenue']=140
    r=latest(f,d)
    assert r.revenue_age_sessions==0
    assert r.revenue_yoy_acceleration==pytest.approx(.4-(.3+.2+.1)/3)
    assert r.revenue_yoy_signed_streak==4
    assert r.revenue_seasonal_mom==pytest.approx(140/130-1)
    assert r.seasonal_support==3
    f.loc[f.period==pd.Timestamp('2016-05-01'),'revenue']=np.nan
    assert np.isnan(latest(f,d).revenue_yoy_acceleration)
    assert latest(f,d).seasonal_support==2


def test_age_late_old_month_and_future_invariance():
    f,d=fixture()
    old=f.period==pd.Timestamp('2014-01-01')
    f.loc[old,'available_date']=pd.Timestamp('2017-06-20')
    ev=replay(f,d)
    a=ev.iloc[-2];b=ev.iloc[-1]
    assert b.revenue_age_sessions==b.session-a.session
    keys=pd.MultiIndex.from_arrays([d[[int(a.session),int(b.session)]],['X','X']])
    x=align_events(ev,d,keys)
    assert x[1,0]==b.session-a.session
    future=f.iloc[-1:].copy();future['period']=pd.Timestamp('2017-06-01');future['available_date']=pd.Timestamp('2017-07-10');future['revenue']=999
    np.testing.assert_allclose(x,align_events(replay(pd.concat([f,future]),d),d,keys),equal_nan=True)


def test_streak_zero_negative_and_left_censor():
    f,d=fixture();assert latest(f,d).revenue_yoy_signed_streak==0
    f.loc[f.period>=pd.Timestamp('2017-03-01'),'revenue']=90
    assert latest(f,d).revenue_yoy_signed_streak==-3
    f.loc[f.period==pd.Timestamp('2017-02-01'),'revenue']=np.nan
    r=latest(f,d);assert np.isnan(r.revenue_yoy_signed_streak) and r.streak_left_censored
    f,d=fixture();f['revenue']=np.arange(len(f))+100
    assert np.isnan(latest(f,d).revenue_yoy_signed_streak)


def test_one_season_zero_support_unknown_and_version_collision():
    f,d=fixture();f=f[f.period<=pd.Timestamp('2014-05-01')]
    assert latest(f,d).seasonal_support==1
    f.loc[f.period==pd.Timestamp('2013-04-01'),'currency']='UNKNOWN'
    assert latest(f,d).seasonal_support==0 and np.isnan(latest(f,d).revenue_seasonal_mom)
    bad=f.iloc[-1:].copy();bad['revenue']=999
    with pytest.raises(ValueError,match='collision'): replay(pd.concat([f,bad]),d)


def test_mutation_age_reset_killed():
    f,d=fixture();ev=replay(f,d);keys=pd.MultiIndex.from_arrays([d[[int(ev.iloc[-1].session)+3]],['X']])
    def oracle(aligner): assert aligner(ev,d,keys)[0,0]==3
    oracle(align_events)
    with pytest.raises(AssertionError): oracle(lambda e,d,k: np.zeros((1,4)))


def test_mutation_left_censor_zero_killed():
    f,d=fixture();f['revenue']=np.arange(len(f))+100
    def oracle(value): assert np.isnan(value)
    oracle(latest(f,d).revenue_yoy_signed_streak)
    with pytest.raises(AssertionError): oracle(0.)


def test_seasonal_median_and_fixed_three_year_window():
    f,d=fixture()
    for year,rev in [(2013,900),(2014,110),(2015,130),(2016,120),(2017,150)]:
        f.loc[f.period==pd.Timestamp(f'{year}-05-01'),'revenue']=rev
    r=latest(f,d)
    assert r.revenue_seasonal_mom==pytest.approx(.5-.2)
    assert r.seasonal_support==3
    f=f[~f.period.isin(pd.to_datetime(['2014-04-01','2015-04-01']))]
    assert latest(f,d).revenue_seasonal_mom==pytest.approx(.5-.2)
    assert latest(f,d).seasonal_support==1


def test_late_denominator_never_resets_age_and_unknown_latest_no_fallback():
    f,d=fixture();f.loc[f.period==pd.Timestamp('2016-05-01'),'available_date']=pd.Timestamp('2017-06-20')
    ev=replay(f,d);a,b=ev.iloc[-2],ev.iloc[-1]
    assert np.isnan(a.revenue_yoy_acceleration)
    assert b.revenue_age_sessions==b.session-a.session
    f.loc[f.period==pd.Timestamp('2017-05-01'),'revenue']=np.nan
    r=latest(f,d)
    assert np.isnan(r.revenue_age_sessions) and np.isnan(r.revenue_seasonal_mom)


def test_release_day_weekend_and_same_session_permutation():
    f,d=fixture();f=f.iloc[:2].copy();f['available_date']=pd.Timestamp('2013-03-09')
    ev=replay(f,d)
    assert len(ev)==1 and d[int(ev.iloc[0].session)]==pd.Timestamp('2013-03-11')
    pd.testing.assert_frame_equal(ev,replay(f.iloc[::-1],d))
    f['available_date']=pd.Timestamp('2013-03-11')
    assert d[int(replay(f,d).iloc[0].session)]==pd.Timestamp('2013-03-12')


def test_disk_limit_and_unsafe_paths():
    from tools.step5_prepare import check_disk_budget
    assert check_disk_budget(40_000_000_000,50_000_000_000)==40_000_000_000
    with pytest.raises(ValueError,match='40 GB'):check_disk_budget(40_000_000_001,100_000_000_000)
    with pytest.raises(ValueError,match='namespace'):s.new_output('/tmp/step5-forbidden')


def test_component_matrix_and_missing_support_stop(tmp_path):
    from tools.step5_audit import feature_support
    base=np.zeros((3,190),np.float32);extra=np.arange(12,dtype=np.float32).reshape(3,4)
    np.save(tmp_path/'base.npy',base);np.save(tmp_path/'extra.npy',extra)
    bundle={'files':{'base':{'path':'base.npy'},'extra':{'path':'extra.npy'}}}
    job=dict(arm='B',fit_base='base',fit_extra='extra')
    x=s.matrix(tmp_path,bundle,job,'fit')
    assert x.shape==(3,194) and np.array_equal(x[:,:190],base) and np.array_equal(x[:,190:],extra)
    job['arm']='A';assert s.matrix(tmp_path,bundle,job,'fit').shape==(3,190)
    assert len(feature_support(extra,s.EXTRA_FEATURES))==4
    extra[:,2]=np.nan
    with pytest.raises(ValueError,match='support'):feature_support(extra,s.EXTRA_FEATURES)


def test_bad_calendar_and_duplicate_versions_fail_closed():
    f,d=fixture()
    with pytest.raises(ValueError,match='calendar'):replay(f,d[::-1])
    bad=f.iloc[-1:].copy();bad['version']=1
    with pytest.raises(ValueError,match='multiple versions'):replay(pd.concat([f,bad]),d)


@pytest.mark.parametrize('case,reason', [('current','CURRENT_MISSING'),('gap','GAP_OR_NONCOMPARABLE'),('identity','GAP_OR_NONCOMPARABLE'),('start','LEFT_CENSORED_HISTORY_START'),('ok','OK')])
def test_streak_reason_complete(case, reason):
    f,d=fixture()
    if case=='current': f.loc[f.index[-1],'revenue']=np.nan
    elif case in ('gap','identity'):
        f.loc[f.period>=pd.Timestamp('2017-03-01'),'revenue']=90.
        if case=='gap': f.loc[f.period==pd.Timestamp('2017-02-01'),'revenue']=np.nan
        else: f.loc[f.period==pd.Timestamp('2017-02-01'),'currency']='USD'
    elif case=='start': f['revenue']=np.arange(len(f))+100.
    r=latest(f,d)
    assert r.streak_nan_reason==reason
    assert bool(r.streak_left_censored)==(reason!='OK')
    assert np.isnan(r.revenue_yoy_signed_streak)==(reason!='OK')


def test_task_paths_are_main_checkout():
    assert str(s.SPEC)=='/home/frank/projects/MarketMamba/tasks/spec-step5-revenue-refinement-h20-20261127.md'
    assert str(s.DECISION)=='/home/frank/projects/MarketMamba/tasks/decision-step5-freeze-20261127.md'
    assert '.worktrees' not in str(s.frozen_decision()['realpath'])


def test_diagnostic_reason_encoding_and_model_column_isolation():
    from tools.step5_features import STREAK_REASONS
    f,d=fixture();f.loc[f.index[-1],'revenue']=np.nan
    events=replay(f,d);last=events.iloc[-1]
    keys=pd.MultiIndex.from_arrays([d[[int(last.session)]],['X']])
    diagnostics=align_events(events,d,keys,diagnostics=True)
    model=align_events(events,d,keys)
    assert diagnostics[0,4]==STREAK_REASONS['CURRENT_MISSING']
    assert diagnostics[:,:4].tobytes()==model.tobytes()
    assert not any(p.name=='step5_runner.py' for p in s.engineering_paths())
    assert {p.name for p in s.engineering_paths()}=={'step5.py','step5_features.py','step5_prepare.py','step5_audit.py'}
