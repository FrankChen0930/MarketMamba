"""Offline descriptive TEJ diagnostics. No training/admission or confirmatory tests.

All row-level outputs stay under .artifacts/tej/diag; reports are aggregates only.
Observed price-date union is a research calendar proxy, not execution evidence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tools.tej_pit import sha256

MONTHLY = ['monthly_yoy','monthly_mom','monthly_cumulative_yoy','monthly_high12','monthly_surprise']
QUARTERLY = ['quarterly_gross_margin','quarterly_operating_margin','quarterly_eps_yoy_difference']


def available_session(release, days, delay=0):
    if pd.isna(release):
        return None
    i = int(days.searchsorted(release, side='right')) + delay
    return i if i < len(days) else None


def ratio(a,b):
    return a/b-1 if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def monthly_events(frame, days):
    from itertools import groupby
    rows=[]
    f=frame.loc[frame.publication_date_usable].copy()
    # Unknown scope is retained and disclosed, never interpreted as verified same scope.
    f['scope']=f['合併營收(Y/N)'].fillna('UNKNOWN')
    f['period_key']=f.period.dt.year*12+f.period.dt.month
    cols=['period_key','available_date','revenue','累計營收(千元)','去年累計營收(千元)','scope']
    for code,g in f.groupby('stock_id',sort=True):
        history={}
        records=g.sort_values(['available_date','period_key'])[cols].itertuples(index=False,name=None)
        for release,batch in groupby(records,key=lambda row:row[1]):
            session=available_session(release,days)
            if session is None: continue
            for row in batch: history[row[0]]=row
            period=max(history); r=history[period];rev=r[2]
            def past(k):
                other=history.get(period-k)
                return other[2] if other is not None and other[5]==r[5] else np.nan
            yoy=ratio(rev,past(12));mom=ratio(rev,past(1))
            previous=np.array([past(k) for k in range(1,13)],float)
            high=float(rev>previous.max()) if np.isfinite(previous).all() and np.isfinite(rev) else np.nan
            known=[]
            for p in sorted(history,reverse=True):
                if p>=period:continue
                a=history[p];b=history.get(p-12)
                if b is not None and a[5]==b[5]==r[5]:
                    y=ratio(a[2],b[2])
                    if np.isfinite(y):known.append(y)
                if len(known)==3:break
            expected=past(12)*(1+np.median(known)) if len(known)==3 else np.nan
            surprise=(rev-expected)/abs(expected) if np.isfinite(expected) and expected!=0 else np.nan
            rows.append(dict(stock_id=code,session=session,monthly_yoy=yoy,monthly_mom=mom,
                monthly_cumulative_yoy=ratio(r[3],r[4]),monthly_high12=high,monthly_surprise=surprise))
    return pd.DataFrame(rows,columns=['stock_id','session']+MONTHLY)


def quarterly_events(frame, days):
    f=frame.copy()
    lag=(f.available_date-f.period_end).dt.days
    calendar=(f.period_start.dt.month.eq(1) & f.period_end.dt.month.isin([3,6,9,12]) &
              f.period_start.dt.year.eq(f.period_end.dt.year) & f.period_end.eq(f.period+pd.offsets.MonthEnd(0)))
    eligible=(f.publication_date_usable & lag.between(0,365) & calendar &
              f['單季(Q)/單半年(H)'].eq('Q') & f.current_financial_sector.eq(False))
    rows=[]
    for code,g in f[eligible].groupby('stock_id',sort=True):
        history={}
        for release,batch in g.sort_values(['available_date','period']).groupby('available_date',sort=True):
            session=available_session(release,days)
            if session is None:continue
            for _,row in batch.iterrows():history[row.period]=row
            p=max(history);r=history[p];old=history.get(p-pd.DateOffset(years=1))
            same=old is not None and all(r[c]==old[c] for c in ['幣別','合併(Y/N)','財報類別（1個別2個體3合併）','季別','單季(Q)/單半年(H)'])
            denom=r['營業收入淨額']
            rows.append(dict(stock_id=code,session=session,
                quarterly_gross_margin=r['營業毛利']/denom if np.isfinite(denom) and denom>0 else np.nan,
                quarterly_operating_margin=r['營業利益']/denom if np.isfinite(denom) and denom>0 else np.nan,
                quarterly_eps_yoy_difference=r.eps-old.eps if same else np.nan))
    audit={'input_rows':len(f),'eligible_rows':int(eligible.sum()),'excluded_rows':int((~eligible).sum()),
           'unknown_sector_excluded':int(f.current_financial_sector.isna().sum()),
           'over365':int((lag>365).sum()),'not_calendar_or_unknown':int((~calendar).sum()),
           'not_Q':int((~f['單季(Q)/單半年(H)'].eq('Q')).sum()),
           'current_financial_sector':int(f.current_financial_sector.eq(True).sum())}
    return pd.DataFrame(rows,columns=['stock_id','session']+QUARTERLY),audit


def asof_values(events, columns, n, delay=0):
    out=np.full((n,len(columns)),np.nan); age=np.full(n,np.nan)
    if events.empty:return out,age
    e=events.sort_values('session').drop_duplicates('session',keep='last')
    indices=e.session.to_numpy(int)+delay
    loc=np.searchsorted(indices,np.arange(n),side='right')-1
    valid=loc>=0
    out[valid]=e[columns].to_numpy(float)[loc[valid]]
    age[valid]=np.arange(n)[valid]-indices[loc[valid]]
    return out,age


def forward_returns(opens, closes, valid, h):
    y=np.full_like(closes,np.nan,dtype=float)
    if len(y)<=h:return y
    invalid=np.concatenate([np.zeros((1,valid.shape[1]),int),np.cumsum(~valid,axis=0)])
    complete=(invalid[h+1:]-invalid[:-(h+1)])==0
    with np.errstate(divide='ignore',invalid='ignore'):
        values=closes[h:]/opens[1:len(opens)-h+1]-1
    y[:-h]=np.where(complete,values,np.nan)
    return y


def rank_ic(x,y,minimum=20):
    x=np.asarray(x,float);y=np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<minimum:return np.nan
    a=rankdata(x[valid]);b=rankdata(y[valid]);a-=a.mean();b-=b.mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>0 else np.nan


def residual(x,z):
    a=np.column_stack([np.ones(len(x)),z])
    # Rank deficiency is an unknown control adjustment, never silently dropped.
    if len(x)<=a.shape[1]+2 or np.linalg.matrix_rank(a)<a.shape[1]:return np.full(len(x),np.nan)
    return x-a@np.linalg.lstsq(a,x,rcond=None)[0]


def summarize(values):
    x=np.asarray(values,float); finite=np.isfinite(x)
    if not finite.any():return {'days':0,'mean_ic':None,'positive_fraction':None,'ci95':None}
    # Keep missing sessions in the time axis rather than compressing gaps before blocks.
    rng=np.random.default_rng(17); n=len(x); blocks=(n+19)//20
    starts=rng.integers(n,size=(1000,blocks))
    samples=x[((starts[:,:,None]+np.arange(20))%n).reshape(1000,-1)[:,:n]]
    counts=np.isfinite(samples).sum(axis=1)
    means=np.divide(np.nansum(samples,axis=1),counts,out=np.full(1000,np.nan),where=counts>0)
    return {'days':int(finite.sum()),'mean_ic':float(x[finite].mean()),
            'positive_fraction':float((x[finite]>0).mean()),'ci95':np.nanquantile(means,[.025,.975]).tolist()}


def coverage_fraction(values, membership):
    denominator=membership.sum(axis=1)
    numerator=(np.isfinite(values)&membership).sum(axis=1)
    return np.divide(numerator,denominator,out=np.full(len(values),np.nan),where=denominator>0)


def arrays_from_events(events,cols,days,codes,delay):
    values=np.full((len(days),len(codes),len(cols)),np.nan,dtype=np.float32)
    age=np.full((len(days),len(codes)),np.nan,dtype=np.float32); lookup={c:i for i,c in enumerate(codes)}
    for code,g in events.groupby('stock_id',sort=False):
        if code not in lookup:continue
        values[:,lookup[code]],age[:,lookup[code]]=asof_values(g,cols,len(days),delay)
    return values,age


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('/home/frank/projects/MarketMamba'))
    p.add_argument('--shared',type=Path,default=Path('/home/frank/projects/p4-bundles/prepared/shared'))
    p.add_argument('--output',type=Path,default=Path('.artifacts/tej/diag'))
    a=p.parse_args(argv);start=time.monotonic()
    expected=Path('.artifacts/tej/diag').resolve()
    if a.output.resolve()!=expected:raise ValueError('STOP row outputs must be .artifacts/tej/diag')
    if any(p.is_file() for p in a.output.iterdir()):raise ValueError('STOP output namespace already populated')
    root=a.source_root/'.artifacts'; identities={}; script_hash=sha256(__file__)
    def identity(path):
        path=Path(path);identities[str(path.resolve())]={'sha256':sha256(path),'bytes':path.stat().st_size}
    pit=root/'tej/pit'; identity(pit/'manifest.json'); manifest=json.loads((pit/'manifest.json').read_text())
    for name in ['monthly_revenue','ifrs_quarterly']:
        path=pit/(name+'.parquet');identity(path)
        if sha256(path)!=manifest['outputs'][name]['sha256']:raise ValueError('STOP TEJ manifest mismatch')
    monthly=pd.read_parquet(pit/'monthly_revenue.parquet');quarterly=pd.read_parquet(pit/'ifrs_quarterly.parquet')
    for name in ['stock_ids.npy','dates.npy','manifest.json']:identity(a.shared/name)
    ids=np.load(a.shared/'stock_ids.npy',mmap_mode='r');dates=np.load(a.shared/'dates.npy',mmap_mode='r')
    codes=np.sort(np.unique(ids)); lookup={c:i for i,c in enumerate(codes)}
    price_path=root/'p4b/snapshot-prices-20261125/prices_raw.parquet';identity(price_path)
    frozen_path=root/'p4b/snapshot-manifest-20261125.json';identity(frozen_path)
    frozen=json.loads(frozen_path.read_text())
    if identities[str(price_path.resolve())]['sha256']!=frozen['files']['prices_raw.parquet']['sha256']:
        raise ValueError('STOP frozen price SHA mismatch')
    raw=pd.read_parquet(price_path);raw['Date']=pd.to_datetime(raw.Date)
    # Calendar retains every source session; prices before 2013 are unnecessary for fixed features.
    days=pd.DatetimeIndex(raw.loc[raw.Date>='2013-01-01','Date'].unique()).sort_values()
    raw=raw[raw.stock_id.isin(codes)&raw.Date.isin(days)]
    from research_lab.data import validate_prices
    converted=raw.rename(columns={'Date':'date','stock_id':'code',**{c:c.lower() for c in ['Open','High','Low','Close','Volume']}})
    converted['adj_close']=np.nan  # provenance unavailable; not an invented adjusted price
    validated=validate_prices(converted)
    price_quality=validated.attrs['quality']
    panels={name:validated.pivot(index='date',columns='code',values=name.lower()).reindex(index=days,columns=codes).to_numpy(float)
            for name in ['Open','High','Low','Close','Volume']}
    valid=validated.pivot(index='date',columns='code',values='observation_valid').reindex(index=days,columns=codes).eq(True).to_numpy()
    del raw,converted,validated
    o,c,v=panels['Open'],panels['Close'],panels['Volume']
    labels={h:forward_returns(o,c,valid,h) for h in [5,20,60]}
    membership=np.zeros(c.shape,bool)
    positions=days.get_indexer(pd.DatetimeIndex(dates)); stockpos=pd.Index(codes).get_indexer(ids)
    ok=positions>=0;membership[positions[ok],stockpos[ok]]=True
    del ids,dates,positions,stockpos
    with np.errstate(divide='ignore',invalid='ignore'):
        mom=np.full_like(c,np.nan);mom[20:]=c[20:]/c[:-20]-1
        rev=np.full_like(c,np.nan);rev[5:]=-(c[5:]/c[:-5]-1)
        volume=pd.DataFrame(np.where(valid,v,np.nan)).rolling(20,min_periods=20).mean().to_numpy()
    # All endpoints and intervening observations required for PIT price controls.
    window20=pd.DataFrame(valid.astype(int)).rolling(21,min_periods=21).sum().to_numpy()==21
    window5=pd.DataFrame(valid.astype(int)).rolling(6,min_periods=6).sum().to_numpy()==6
    mom[~window20]=np.nan;rev[~window5]=np.nan
    controls=np.stack([mom,rev,volume],axis=2)
    score_path=root/'p1/full-scores-ace2eef/rank_average.parquet';identity(score_path)
    identity(score_path.parent/'manifest.json')
    scores=pd.read_parquet(score_path,columns=['system','date','code','score']);scores=scores[scores.system.eq('c0plus')]
    if scores.duplicated(['date','code']).any():raise ValueError('STOP duplicate C0+ scores')
    c0=scores.pivot(index='date',columns='code',values='score').reindex(index=days,columns=codes).to_numpy(float)
    del scores
    print('prices/membership/labels ready',c.shape,int(membership.sum()),flush=True)
    me=monthly_events(monthly,days);qe,qa=quarterly_events(quarterly,days)
    me.to_parquet(a.output/'monthly_events.parquet',index=False);qe.to_parquet(a.output/'quarterly_events.parquet',index=False)
    print('events ready',len(me),len(qe),flush=True)
    result={'status':'DESCRIPTIVE_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'script_sha256':script_hash,'identities':identities,'ifrs_filter':qa,
        'config':{'primary_horizon':20,'descriptive_horizons':[5,60],'periods':['2014-2020','2021-2026'],
                  'bootstrap':{'type':'circular','block':20,'samples':1000,'seed':17},'minimum_cross_section':20},
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                   'data_root_env':os.getenv('MARKETMAMBA_DATA_ROOT'),'data_realpath':str(Path('Data').resolve())},
        'full_pit_four_control_ic':'STOP: historical PIT market capitalization unavailable; known snapshot is non-PIT; no substitution',
        'metrics':{},'coverage':{},'delay':{},'support':{'members':len(codes),'matrix_rows':int(membership.sum()),
          'calendar_first':str(days.min().date()),'calendar_last':str(days.max().date()),'calendar_sessions':len(days),'price_quality':price_quality},
        'limitations':['EXPOSED_DATA_DESCRIPTIVE_NOT_HYPOTHESIS_TEST','RAW_PRICE_NO_DIVIDEND_ADJUSTMENT',
         'CURRENT_INDUSTRY_APPROXIMATION','IFRS_VALUE_VERSION_UNKNOWN_INCLUDING_EPS','MONTHLY_FIRST_VALUE_DOCUMENTATION_NOT_INDEPENDENTLY_VERIFIED',
         'MONTHLY_SCOPE_INDICATOR_UNKNOWN_COMPARABILITY_NOT_VERIFIED','CUMULATIVE_PRIOR_YEAR_REPORTED_COMPARATOR_VERSION_UNKNOWN','NON_PIT_MARKET_CAP_NOT_USED',
         'OBSERVED_CALENDAR_NOT_OFFICIAL','MEMBERSHIP_NOT_LIFECYCLE_OR_TRADABILITY','OVERLAPPING_LABELS_60D_CI_BLOCK20_MAY_UNDERSTATE_DEPENDENCE',
         'C0PLUS_TUNED_ON_EXPOSED_LATER_DATA_2021_ONWARD_ONLY','NO_INDEPENDENT_REVIEW']}
    period_masks={'2014-2020':(days.year>=2014)&(days.year<=2020),'2021-2026':(days.year>=2021)&(days.year<=2026)}
    # Persist daily IC/coverage separately; summary JSON contains no stock/date rows.
    daily=[]
    for block,events,cols in [('monthly',me,MONTHLY),('quarterly',qe,QUARTERLY)]:
        for delay in [0,5,10,20]:
            values,age=arrays_from_events(events,cols,days,codes,delay)
            values[~membership]=np.nan
            ranks=rankdata(values,axis=1,nan_policy='omit')
            count=np.isfinite(values).sum(axis=1)
            ranks=np.divide(ranks,count[:,None,:],out=np.full_like(ranks,np.nan),where=count[:,None,:]>0)
            # Fixed equal weight, require every fixed feature; never select available winners.
            composite=np.mean(ranks,axis=2);features=np.concatenate([values,composite[:,:,None]],axis=2)
            names=cols+[block+'_equal_weight']
            if delay:features=features[:,:,-1:];names=[names[-1]]
            for j,name in enumerate(names):
                x=features[:,:,j]; raw_ic={h:np.full(len(days),np.nan) for h in ([20] if delay else [5,20,60])}
                partial=np.full(len(days),np.nan);c0ic=np.full(len(days),np.nan);corr=np.full(len(days),np.nan)
                event_ic={b:np.full(len(days),np.nan) for b in ['1-5','6-20','21-60','>60']}
                coverage=coverage_fraction(x,membership)
                for t in np.flatnonzero((days.year>=2014)&(membership.sum(axis=1)>=20)):
                    for h in raw_ic:raw_ic[h][t]=rank_ic(x[t],labels[h][t])
                    if delay:continue
                    mask=np.isfinite(x[t])&np.isfinite(labels[20][t])&np.isfinite(controls[t]).all(axis=1)
                    if mask.sum()>=20:
                        z=controls[t,mask];z=(z-z.mean(axis=0))/np.where(z.std(axis=0)>0,z.std(axis=0),1)
                        partial[t]=rank_ic(residual(x[t,mask],z),residual(labels[20][t,mask],z))
                    corr[t]=rank_ic(x[t],c0[t])
                    mask=np.isfinite(x[t])&np.isfinite(labels[20][t])&np.isfinite(c0[t])
                    if mask.sum()>=20:
                        z=rankdata(c0[t,mask])[:,None]
                        c0ic[t]=rank_ic(residual(x[t,mask],z),residual(labels[20][t,mask],z))
                    # Age is sessions since usable close: zero is retained separately, not mislabeled as day one.
                    for b,lo,hi in [('1-5',1,5),('6-20',6,20),('21-60',21,60),('>60',61,np.inf)]:
                        select=(age[t]>=lo)&(age[t]<=hi)
                        event_ic[b][t]=rank_ic(x[t,select],labels[20][t,select])
                for period,pm in period_masks.items():
                    key=f'{name}/{period}'
                    if delay:
                        result['delay'].setdefault(key,{})[str(delay)]=summarize(raw_ic[20][pm]);continue
                    result['metrics'][key]={'raw':{str(h):summarize(ic[pm]) for h,ic in raw_ic.items()},
                        'three_price_controls_partial':summarize(partial[pm]),'c0_rank_correlation':summarize(corr[pm]),
                        'c0_residual_ic':summarize(c0ic[pm]),'event_h20':{b:summarize(ic[pm]) for b,ic in event_ic.items()}}
                    if name.endswith('equal_weight'):result['delay'].setdefault(key,{})['0']=summarize(raw_ic[20][pm])
                if not delay:
                    result['coverage'][name]={str(y):{'sessions':int(((days.year==y)&(membership.sum(axis=1)>0)).sum()),
                        'mean_fraction':float(coverage[(days.year==y)&(membership.sum(axis=1)>0)].mean())}
                        for y in sorted(set(days.year)) if y>=2014 and ((days.year==y)&(membership.sum(axis=1)>0)).any()}
                    daily.append(pd.DataFrame({'date':days,'feature':name,'coverage':coverage,'ic5':raw_ic[5],
                        'ic20':raw_ic[20],'ic60':raw_ic[60],'three_controls_ic20':partial,'c0_corr':corr,'c0_residual_ic20':c0ic,
                        **{'event_'+b:ic for b,ic in event_ic.items()}}))
            if delay==0:
                # one private row table per block, including age and fixed composite
                ti,si=np.where(membership & (np.asarray(days.year)[:,None]>=2014))
                row=pd.DataFrame({'date':days[ti],'stock_id':codes[si],'age_sessions':age[ti,si]})
                for j,name in enumerate(cols+[block+'_equal_weight']):row[name]=features[ti,si,j]
                row.to_parquet(a.output/(block+'_daily.parquet'),index=False)
                del row
            print(block,'delay',delay,'done',round(time.monotonic()-start,1),flush=True)
    pd.concat(daily,ignore_index=True).to_parquet(a.output/'daily_diagnostics.parquet',index=False)
    for key,curve in result['delay'].items():
        # Descriptive flag, not a test or a feature-selection rule.
        m=[curve[str(i)]['mean_ic'] for i in [0,5,10,20]]
        curve['timing_warning_heuristic']=bool(all(v is not None for v in m) and m[0]>0 and m[0]-m[1]>.01 and m[1]<m[0]/2)
    if sha256(__file__)!=script_hash:raise ValueError('STOP script changed during run')
    for path,info in identities.items():
        if sha256(path)!=info['sha256']:raise ValueError('STOP input changed during diagnosis')
    result['outputs']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in a.output.iterdir() if p.is_file()}
    result['elapsed_seconds']=time.monotonic()-start
    Path('reports/tej_diag_20261125.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.output/'manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['metrics','coverage','delay']},ensure_ascii=False,indent=2)+'\n')
    print('DONE',result['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
