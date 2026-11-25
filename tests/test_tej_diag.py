"""Synthetic-only TEJ descriptive diagnostics; never opens licensed inputs."""
import numpy as np
import pandas as pd
import pytest
from tools import tej_diag as d


def test_publication_and_delay():
    days = pd.bdate_range('2020-01-02', periods=30)
    assert d.available_session(days[0], days) == 1
    assert d.available_session(days[0], days, 5) == 6
    assert d.available_session(pd.NaT, days) is None
    assert d.available_session(days[-1], days) is None
    assert d.available_session(pd.Timestamp('2020-01-04'), days) == 2


def test_asof_no_future_and_unknown_replaces_old():
    events = pd.DataFrame({'session':[1,4], 'x':[3.,np.nan]})
    values, age = d.asof_values(events, ['x'], 7, 0)
    assert np.isnan(values[0,0])
    assert values[1:4,0].tolist() == [3.,3.,3.]
    assert np.isnan(values[4:,0]).all()
    assert age[1:4].tolist() == [0,1,2]
    delayed, _ = d.asof_values(events, ['x'], 7, 2)
    assert np.isnan(delayed[:3]).all()
    assert delayed[3:6,0].tolist() == [3.,3.,3.]


def test_ic_ties_nan_constant_and_residual():
    assert d.rank_ic([1,2,3,4], [4,3,2,1], minimum=3) == pytest.approx(-1)
    assert np.isnan(d.rank_ic([1,1,1], [1,2,3], minimum=3))
    assert d.rank_ic([1,2,3,np.nan], [1,2,3,7], minimum=3) == pytest.approx(1)
    z=np.arange(20.)[:,None]; x=z[:,0]*3+np.tile([1.,-1.],10)
    r=d.residual(x,z)
    assert np.dot(r,z[:,0]) == pytest.approx(0,abs=1e-9)


def test_labels_calendar_gap_no_roll():
    o=np.ones((10,2))*10; c=np.arange(10.)[:,None]+o
    valid=np.ones_like(o,dtype=bool); valid[3,1]=False
    y=d.forward_returns(o,c,valid,5)
    assert y[0,0] == pytest.approx(.5)
    assert np.isnan(y[0,1])
    assert np.isnan(y[-5:]).all()


def test_monthly_history_is_pit_and_late_old_does_not_replace_latest():
    days=pd.bdate_range('2019-01-01','2021-12-31')
    rows=[]
    for i,p in enumerate(pd.date_range('2019-01-01',periods=25,freq='MS')):
        rows.append(dict(stock_id='X', period=p, available_date=p+pd.offsets.MonthEnd()+pd.Timedelta(days=10),
                         revenue=100.+i, publication_date_usable=True,
                         **{'累計營收(千元)':100.+i,'去年累計營收(千元)':100.,'合併營收(Y/N)':'Y'}))
    f=pd.DataFrame(rows)
    e=d.monthly_events(f,days)
    cut=e.iloc[-2].session
    a=e[e.session<=cut].reset_index(drop=True)
    b=d.monthly_events(f.iloc[:-1],days).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    assert e.iloc[-1].monthly_yoy == pytest.approx(124/112-1)
    assert e.iloc[-1].monthly_high12 == 1
    future=f.copy();future.loc[24,'revenue']=1e9
    pd.testing.assert_frame_equal(e.iloc[:-1],d.monthly_events(future,days).iloc[:-1])


def test_bootstrap_deterministic_and_missing_unknown():
    a=d.summarize(np.linspace(-.1,.2,50))
    assert a == d.summarize(np.linspace(-.1,.2,50))
    assert a['days']==50 and len(a['ci95'])==2
    assert d.summarize([np.nan]*50)['mean_ic'] is None


def test_mutations_are_killed(monkeypatch):
    original=d.available_session
    def same_day(release,days,delay=0):
        if pd.isna(release): return None
        return int(days.searchsorted(release,side='left'))+delay
    monkeypatch.setattr(d,'available_session',same_day)
    with pytest.raises(AssertionError): test_publication_and_delay()
    monkeypatch.setattr(d,'available_session',original)
    original_asof=d.asof_values
    def future_asof(events,cols,n,delay=0):
        values,age=original_asof(events,cols,n,delay)
        values[0]=values[1]
        return values,age
    monkeypatch.setattr(d,'asof_values',future_asof)
    with pytest.raises(AssertionError): test_asof_no_future_and_unknown_replaces_old()


def test_label_parity_with_existing_contract():
    from research_lab.data import validate_prices
    from research_lab.labels import compute_labels
    days=pd.bdate_range('2020-01-01',periods=70)
    rows=[]
    for code in ['A','B']:
        for i,date in enumerate(days):
            if code=='B' and i==13:continue
            rows.append(dict(code=code,date=date,open=100+i,close=101+i,high=102+i,low=99+i,adj_close=101+i,volume=100))
    frame=validate_prices(pd.DataFrame(rows))
    labels=compute_labels(frame,horizons=(5,20,60))
    panel=frame.set_index(['date','code']).reindex(pd.MultiIndex.from_product([days,['A','B']],names=['date','code']))
    o=panel.open.to_numpy().reshape(70,2);c=panel.close.to_numpy().reshape(70,2)
    valid=panel.observation_valid.eq(True).to_numpy().reshape(70,2)
    for h in [5,20,60]:
        actual=d.forward_returns(o,c,valid,h)
        expected=labels.pivot(index='date',columns='code',values=f'fwd_return_{h}d').reindex(days).to_numpy()
        np.testing.assert_allclose(actual,expected,equal_nan=True)


def test_unknown_monthly_scope_is_explicit_not_boolean_error():
    days=pd.bdate_range('2020-01-01','2021-12-31')
    f=pd.DataFrame([dict(stock_id='X',period=pd.Timestamp('2020-01-01'),available_date=pd.Timestamp('2020-02-10'),
        revenue=100.,publication_date_usable=True,**{'合併營收(Y/N)':pd.NA,'累計營收(千元)':100.,'去年累計營收(千元)':80.})])
    events=d.monthly_events(f,days)
    assert len(events)==1 and np.isnan(events.monthly_yoy.iloc[0])
    assert events.monthly_cumulative_yoy.iloc[0]==pytest.approx(.25)


def test_quarterly_filters_and_same_scope_eps():
    days=pd.bdate_range('2019-01-01','2022-12-31')
    def row(year,**kwargs):
        r=dict(stock_id='X',period=pd.Timestamp(f'{year}-03-01'),period_start=pd.Timestamp(f'{year}-01-01'),
            period_end=pd.Timestamp(f'{year}-03-31'),available_date=pd.Timestamp(f'{year}-05-15'),publication_date_usable=True,
            current_financial_sector=False,eps=year-2018.,**{'單季(Q)/單半年(H)':'Q','幣別':'TWD','合併(Y/N)':'Y',
            '財報類別（1個別2個體3合併）':'3','季別':'1','營業收入淨額':100.,'營業毛利':20.,'營業利益':10.})
        r.update(kwargs);return r
    f=pd.DataFrame([row(2019),row(2020)])
    events,audit=d.quarterly_events(f,days)
    assert audit['eligible_rows']==2
    assert events.iloc[-1].quarterly_eps_yoy_difference==1
    f.loc[1,'合併(Y/N)']='N'
    assert np.isnan(d.quarterly_events(f,days)[0].iloc[-1].quarterly_eps_yoy_difference)
    for field,value in [('current_financial_sector',True),('period_start',pd.NaT),
                        ('available_date',pd.Timestamp('2022-05-15')),('單季(Q)/單半年(H)','H')]:
        f=pd.DataFrame([row(2020,**{field:value})])
        assert d.quarterly_events(f,days)[1]['eligible_rows']==0


def test_late_older_month_only_changes_signals_after_its_release():
    days=pd.bdate_range('2020-01-01','2020-05-01')
    rows=[]
    for period,release,revenue in [('2020-01-01','2020-03-20',100.),('2020-02-01','2020-03-10',120.)]:
        rows.append(dict(stock_id='X',period=pd.Timestamp(period),available_date=pd.Timestamp(release),revenue=revenue,
                         publication_date_usable=True,**{'合併營收(Y/N)':'Y','累計營收(千元)':100.,'去年累計營收(千元)':100.}))
    events=d.monthly_events(pd.DataFrame(rows),days)
    assert len(events)==2 and np.isnan(events.iloc[0].monthly_mom)
    assert events.iloc[1].monthly_mom==pytest.approx(.2)
    values,_=d.asof_values(events,['monthly_mom'],len(days))
    assert np.isnan(values[:days.get_loc('2020-03-23')]).all()
    assert values[days.get_loc('2020-03-23'),0]==pytest.approx(.2)


def test_coverage_empty_universe_is_unknown():
    members=np.array([[True,True],[False,False],[True,True]])
    x=np.array([[1.,np.nan],[np.nan,np.nan],[np.nan,np.nan]])
    out=d.coverage_fraction(x,members)
    assert out[0]==.5 and np.isnan(out[1]) and out[2]==0.
