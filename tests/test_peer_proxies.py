"""Synthetic only: no licensed rows, labels or forward outcomes."""
import numpy as np
import pandas as pd
import pytest
from tools import peer_proxies as p


def members():
    return pd.DataFrame({'stock_id':list('ABCDE'),'industry':['I']*5,'eligible':[True]*5})


def test_excludes_self_minimum_and_market():
    m=members(); r=pd.Series([.9,.1,.2,.3,np.nan],index=list('ABCDE'))
    out=p.proxy_day(m,r,pd.Series([1.,2.,3.,4.,np.nan],index=list('ABCDE')),
                    pd.Series([False,True,True,True,False],index=list('ABCDE')))
    a=out.set_index('stock_id').loc['A']
    assert a.peer_return == pytest.approx(.2)
    assert a.return_peer_count == 3
    assert a.revenue_spillover == 3
    assert a.market_return == pytest.approx(.375)
    out=p.proxy_day(m.iloc[:3],r,pd.Series(dtype=float),pd.Series(dtype=bool))
    assert out.peer_return.isna().all()
    assert out.return_reason.eq('INSUFFICIENT_VALID_PEERS').all()


def test_release_same_day_target_announcement_and_late_denominator():
    days=pd.to_datetime(['2014-02-05','2014-02-06','2014-02-07','2014-02-10'])
    rows=[]
    for c in 'ABCDE':
        rows.extend([(c,'2013-01-01','2013-02-01',100.),(c,'2014-01-01','2014-02-05' if c!='E' else '2014-02-07',200.)])
    f=pd.DataFrame(rows,columns=['stock_id','period','release','revenue'])
    e=p.revenue_events(f,days)
    y,a=p.revenue_day(e,days[0],list('ABCDE'))
    assert not a.any() and y.isna().all()
    y,a=p.revenue_day(e,days[1],list('ABCDE'))
    assert a.loc['A'] and not a.loc['E']
    o=p.proxy_day(members(),pd.Series(dtype=float),y,a).set_index('stock_id')
    assert np.isnan(o.loc['A','revenue_spillover'])
    assert o.loc['E','revenue_spillover']==1
    f.loc[(f.stock_id=='A') & (f.period=='2013-01-01'),'release']='2014-02-07'
    y,a=p.revenue_day(p.revenue_events(f,days),days[1],list('ABCDE'))
    assert a.loc['A'] and np.isnan(y.loc['A'])


def test_2013_break_and_invalid_release():
    f=pd.DataFrame([('A','2012-01-01','2012-02-01',100.),('A','2013-01-01','2013-02-01',200.),('B','2014-01-01','2014-01-01',20.)],columns=['stock_id','period','release','revenue'])
    e=p.revenue_events(f,pd.to_datetime(['2013-02-04','2014-02-04']))
    assert e.yoy.isna().all()
    assert pd.isna(e.loc[e.stock_id=='B','available'].iloc[0])


def test_industry_effective_date_and_unknown_direction():
    row={'stock_id':'A','TSE產業_代碼':'M2','首次掛牌TSE產業':'M1 old','前一次TSE產業變更日':'20200103','前一次TSE產業變更':'M2 new'}
    assert p.industry_asof(row,'2020-01-02','industry')[0]=='M1'
    assert p.industry_asof(row,'2020-01-03','industry')[0]=='M2'
    assert p.industry_asof(row,'2020-01-03','industry')[1]=='KNOWN_AT_UNKNOWN'


def example_6161():
    return {'stock_id':'6161','首次掛牌日期':'20020102',
            '首次掛牌TEJ子產業':'M23B1 主機板','TEJ子產業_代碼':'M23B4A',
            '前一次TEJ子產業變更':'M23B4A 繪圖卡','前一次TEJ子產業變更日':'20070730',
            '前二次TEJ子產業變更':'M23B4 顯示卡','前二次TEJ子產業變更日':'20050816',
            '前三次TEJ子產業變更':'M23B5 泛工業電腦','前三次TEJ子產業變更日':'20050309'}


def test_documented_6161_segments_and_future_noninterference():
    row=example_6161()
    for day,expected in [('2002-01-02','M23B1'),('2005-03-08','M23B1'),
                         ('2005-03-09','M23B5'),('2005-08-15','M23B5'),
                         ('2005-08-16','M23B4'),('2007-07-29','M23B4'),
                         ('2007-07-30','M23B4A')]:
        assert p.industry_asof(row,day,'tej_subindustry')==(expected,'KNOWN_AT_UNKNOWN')
    # Valid future event and matching snapshot must not rewrite prior segments.
    row['前一次TEJ子產業變更']='M23ZZ';row['TEJ子產業_代碼']='M23ZZ'
    assert p.industry_asof(row,'2006-01-01','tej_subindustry')[0]=='M23B4'


@pytest.mark.parametrize('mutation,reason',[
    ({'前一次TEJ子產業變更':'M23B4'},'LATEST_CURRENT_MISMATCH'),
    ({'前二次TEJ子產業變更日':'20080101'},'CHANGE_DATES_NOT_DESCENDING'),
    ({'前二次TEJ子產業變更日':None},'INCOMPLETE_HISTORY'),
    ({'前二次TEJ子產業變更日':'bad'},'INVALID_HISTORY_DATE'),
    ({'前四次TEJ子產業變更':'M23B0','前四次TEJ子產業變更日':'20010101'},'CHANGE_BEFORE_FIRST_LISTING'),
])
def test_history_direction_and_date_mutations_fail_closed(mutation,reason):
    row=example_6161();row.update(mutation)
    assert p.industry_asof(row,'2006-01-01','tej_subindustry')==(None,reason)


def test_delisting_unknown_terminal_and_market_change():
    row={'stock_id':'A','首次掛牌日期':'20100101','首次掛牌市場':'TSE','上市別':'TSE','TSE上市日':'20100101','下市日期':'20200103'}
    assert p.market_asof(row,'2020-01-02')==(True,'SNAPSHOT_DERIVED')
    assert p.market_asof(row,'2020-01-03')==(False,'DELISTING_ENDPOINT_UNKNOWN')
    assert not p.market_asof(row,'2020-01-06')[0]
    assert not p.market_asof(row,'2009-12-31')[0]
    row['前一次變更市場日期']='20190101';row['前一次變更市場']='TSE'
    assert p.market_asof(row,'2020-01-02')==(False,'MARKET_HISTORY_DIRECTION_UNKNOWN')


def test_listing_requires_explicit_tse_otc_date():
    row={'首次掛牌日期':'20020102','首次掛牌市場':'TSE','上市別':'TSE'}
    assert not p.market_asof(row,'2003-01-01')[0]
    row['OTC上市日']='20020301'
    assert not p.market_asof(row,'2002-02-28')[0]
    assert p.market_asof(row,'2002-03-01')[0]
    del row['首次掛牌日期']
    assert p.market_asof(row,'2002-03-01')[0]


def test_price_lag_missing_session_and_future_mutation():
    d=pd.date_range('2020-01-01',periods=6)
    c=pd.DataFrame({'A':[10,11,12,13,14,15],'B':[10,10,np.nan,12,12,12]},index=d,dtype=float)
    v=c.notna(); r=p.lagged_returns(c,v)
    assert r.loc[d[2],'A']==pytest.approx(.1)
    assert pd.isna(r.loc[d[3],'B'])
    changed=c.copy();changed.loc[d[4]:]=999
    pd.testing.assert_frame_equal(r.loc[:d[4]],p.lagged_returns(changed,v).loc[:d[4]])
    assert p.lagged_returns(changed,v).loc[d[5],'A'] != r.loc[d[5],'A']


def test_future_revenue_mutation_does_not_rewrite_past():
    days=pd.date_range('2014-02-01','2014-02-12')
    f=pd.DataFrame([('A','2013-01-01','2013-02-01',100.),('A','2014-01-01','2014-02-05',200.),('B','2013-01-01','2013-02-01',100.),('B','2014-01-01','2014-02-10',200.)],columns=['stock_id','period','release','revenue'])
    e=p.revenue_events(f,days)
    changed=f.copy();changed.loc[3,'revenue']=99999
    y,a=p.revenue_day(e,days[6],list('AB'))
    yy,aa=p.revenue_day(p.revenue_events(changed,days),days[6],list('AB'))
    pd.testing.assert_series_equal(y,yy);pd.testing.assert_series_equal(a,aa)
    future,_=p.revenue_day(e,days[-1],list('AB'))
    changed_future,_=p.revenue_day(p.revenue_events(changed,days),days[-1],list('AB'))
    assert future.loc['B'] != changed_future.loc['B']


def test_nonfinite_duplicates_and_output_boundary(tmp_path):
    with pytest.raises(ValueError,match='duplicate'):
        p.revenue_events(pd.DataFrame([('A','2014-01-01','2014-02-05',1.)]*2,columns=['stock_id','period','release','revenue']),pd.to_datetime(['2014-02-06']))
    with pytest.raises(ValueError,match='peer'):
        p.private_output(tmp_path)


def test_missing_release_with_microsecond_dates_is_unknown():
    f=pd.DataFrame({'stock_id':['A','B'],'period':pd.to_datetime(['2014-01-01']*2).as_unit('us'),
                    'release':pd.to_datetime(['2014-02-05',None]).as_unit('us'),'revenue':[10.,20.]})
    e=p.revenue_events(f,pd.to_datetime(['2014-02-05','2014-02-06']).as_unit('us'))
    assert pd.isna(e.loc[e.stock_id.eq('B'),'available'].iloc[0])
    assert e.loc[e.stock_id.eq('A'),'available'].iloc[0]==pd.Timestamp('2014-02-06')


def test_unknown_classification_and_unlisted_do_not_join_peer_group():
    m=members();m.loc[3,'industry']=None;m.loc[4,'eligible']=False
    r=pd.Series(1.,index=list('ABCDE'))
    out=p.proxy_day(m,r,r,pd.Series(False,index=list('ABCDE'))).set_index('stock_id')
    assert out.loc['A','industry_peer_count']==2
    assert out.loc['A','return_peer_count']==2
    assert np.isnan(out.loc['A','peer_return'])
    assert out.loc['D','return_reason']=='INDUSTRY_UNKNOWN'
    assert out.loc['E','return_reason']=='MARKET_OR_LIFECYCLE_UNKNOWN'
    assert out.loc['A','market_valid_count']==4
    row={'TSE產業_代碼':'M2','首次掛牌TSE產業':'M1 old'}
    assert p.industry_asof(row,'2020-01-02','industry')[0] is None
