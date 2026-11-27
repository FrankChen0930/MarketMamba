"""Step5 causal first-month replay. Historical source tolerance is not strict PIT certification."""
from __future__ import annotations
import argparse
from itertools import groupby
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tools import step3
from tools import step5 as s
from tools.step3 import effective_session, _known
from tools.step3_prepare import admitted_events


def replay(frame, days):
    days=pd.DatetimeIndex(days)
    if not days.is_unique or not days.is_monotonic_increasing:raise ValueError('STOP calendar')
    required=['stock_id','period','available_date','revenue','scope','currency','unit','comparability_id','version','source_row']
    if not set(required)<=set(frame):raise ValueError('STOP event schema')
    f=frame[required].copy()
    f['period']=pd.to_datetime(f.period);f['available_date']=pd.to_datetime(f.available_date)
    if f[['stock_id','period','version','source_row']].isna().any().any():raise ValueError('STOP missing event key')
    f=f[f.available_date.notna() & (f.period>=pd.Timestamp('2013-01-01'))].copy()
    if (f.available_date<f.period+pd.offsets.MonthEnd(0)).any():raise ValueError('STOP release before period end')
    f['month']=f.period.dt.year*12+f.period.dt.month-1
    f['session']=[effective_session(d,days) for d in f.available_date]
    # Multiple conflicting values for an unorderable version are never row-order resolved.
    value_cols=['available_date','revenue','scope','currency','unit','comparability_id']
    logical=['stock_id','month','version']
    # Equivalent to per-key drop_duplicates, without constructing 380k frames.
    distinct=f.drop_duplicates(logical+value_cols)
    if distinct.duplicated(logical).any():raise ValueError('STOP version collision')
    # This first-announcement implementation does not silently replace earlier values.
    if (f.groupby(['stock_id','month']).version.nunique()>1).any():
        raise ValueError('STOP multiple versions require first-announcement evidence')
    f=f.sort_values(['stock_id','session','available_date','month','version','source_row'],kind='stable')
    lineage=[]
    duplicates=f[f.duplicated(['stock_id','month','version'],keep=False)]
    for (code,month,version),g in duplicates.groupby(['stock_id','month','version'],sort=True):
        lineage.append(dict(stock_id=str(code),month=int(month),version=str(version),source_rows=sorted(set(g.source_row.astype(str)))))
    f=f.drop_duplicates(['stock_id','month','version'])
    rows=[]
    for code,g in f.groupby('stock_id',sort=True):
        history={}; first={}
        for session,batch in groupby(g.itertuples(index=False),key=lambda r:r.session):
            if session>=len(days):continue
            for r in batch:
                identity=tuple(getattr(r,k) for k in ('scope','currency','unit','comparability_id'))
                identity=identity if all(_known(v) for v in identity) else None
                history[r.month]=(float(r.revenue),identity)
                if np.isfinite(r.revenue) and identity is not None:first.setdefault(r.month,session)
            month=max(history); identity=history[month][1]
            def value(m):
                item=history.get(m)
                return item[0] if item is not None and identity is not None and item[1]==identity else np.nan
            def ratio(a,b):return a/b-1 if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan
            def yoy(m):return ratio(value(m),value(m-12))
            def mom(m):return ratio(value(m),value(m-1))
            age=float(session-first[month]) if month in first and np.isfinite(value(month)) else np.nan
            ys=np.array([yoy(month-k) for k in range(4)])
            acc=ys[0]-ys[1:].mean() if np.isfinite(ys).all() else np.nan
            streak=np.nan;left=False
            if np.isfinite(ys[0]):
                sign=np.sign(ys[0])
                if sign==0:streak=0.
                else:
                    k=1
                    while True:
                        prior=yoy(month-k)
                        if not np.isfinite(prior):left=True;break
                        if np.sign(prior)!=sign:streak=float(sign*k);break
                        k+=1
            past=np.array([mom(month-12*j) for j in (1,2,3)])
            valid=past[np.isfinite(past)];support=len(valid)
            seasonal=mom(month)-np.median(valid) if support and np.isfinite(mom(month)) else np.nan
            rows.append(dict(stock_id=str(code),session=int(session),
                **dict(zip(s.EXTRA_FEATURES,(age,acc,streak,seasonal))),
                streak_left_censored=left, seasonal_support=support))
    return pd.DataFrame(rows,columns=['stock_id','session',*s.EXTRA_FEATURES,'streak_left_censored','seasonal_support'])


def align_events(events, days, keys, diagnostics=False):
    days=pd.DatetimeIndex(days)
    sessions=days.get_indexer(pd.DatetimeIndex(keys.get_level_values(0)))
    if (sessions<0).any():raise ValueError('STOP unknown session')
    n=6 if diagnostics else 4
    result=np.full((len(keys),n),np.nan,np.float32)
    stocks=np.asarray(keys.get_level_values(1).astype(str))
    columns=[*s.EXTRA_FEATURES,'streak_left_censored','seasonal_support'][:n]
    groups=pd.Series(stocks).groupby(stocks,sort=False).indices
    for code,g in events.groupby('stock_id',sort=False):
        rows=groups.get(str(code),np.empty(0,dtype=np.int64))
        ix=np.searchsorted(g.session.to_numpy(),sessions[rows],side='right')-1
        rows=rows[ix>=0];ix=ix[ix>=0]
        result[rows]=g[columns].to_numpy(np.float32)[ix]
        age=np.isfinite(result[rows,0])
        result[rows[age],0]+=sessions[rows[age]]-g.session.to_numpy()[ix[age]]
    return result


def build(out, resume_verified_partial=False):
    from tools.step4_prepare import axes
    source=Path('/home/frank/projects/MarketMamba/.artifacts/step3/bundle-72')
    s.verify_file(s.SPEC,s.SPEC_SHA);s.frozen_decision()
    s.verify_file(source/'bundle.json',s.STEP3_BUNDLE_SHA)
    b=json.loads((source/'bundle.json').read_text());step3.validate_bundle_schema(b);step3.verify_source_provenance(b)
    raw=Path('/home/frank/projects/MarketMamba/.artifacts/tej/pit/monthly_revenue.parquet')
    # Frozen source manifest binds normalized input, never a new raw acquisition.
    manifest=b['files']['source_manifest'];s.verify_file(manifest['path'],manifest['sha256'],manifest['bytes'])
    m=json.loads(Path(manifest['path']).read_text())
    info=m['outputs']['monthly_revenue']
    s.verify_file(raw,info['sha256'],info['bytes'])
    for ref in ('calendar','cross_source_pairs'):
        i=b['files'][ref];s.verify_file(i['path'],i['sha256'],i['bytes'])
    days=pd.DatetimeIndex(np.load(b['files']['calendar']['path'],allow_pickle=False))
    f=admitted_events(pd.read_parquet(raw),pd.read_parquet(b['files']['cross_source_pairs']['path']))
    events=replay(f,days)
    # Read matrix axes only (never labels).
    files={k:dict(v,path=str(Path(v['path']).relative_to('/'))) for k,v in b['files'].items()}
    dates,di,stock,codes,ordered,sorter=axes(dict(files=files),Path('/'))
    estimate=len(dates)*6*4+int(f.memory_usage(deep=True).sum())*3
    if estimate>s.LIMIT_BYTES:raise ValueError('STOP features estimate >40 GB')
    if resume_verified_partial:
        out=Path(out).absolute()
        if not out.is_relative_to(s.OUTPUT_ROOT) or out.resolve()!=out or not out.is_dir():raise ValueError('STOP partial namespace')
        if set(p.name for p in out.iterdir())!={'source_events.parquet','events.parquet','block.npy','calendar.npy'}:raise ValueError('STOP partial inventory')
        if not f.equals(pd.read_parquet(out/'source_events.parquet')) or not events.equals(pd.read_parquet(out/'events.parquet')):raise ValueError('STOP partial source replay')
        block=np.load(out/'block.npy',mmap_mode='r',allow_pickle=False)
        if block.shape!=(len(dates),6) or block.dtype!=np.float32:raise ValueError('STOP partial block schema')
        if not np.array_equal(np.load(out/'calendar.npy',allow_pickle=False),days.to_numpy()):raise ValueError('STOP partial calendar')
    else:
        out=s.new_output(out)
        f.to_parquet(out/'source_events.parquet',index=False)
        events.to_parquet(out/'events.parquet',index=False)
        block=np.lib.format.open_memmap(out/'block.npy',mode='w+',dtype=np.float32,shape=(len(dates),6))
    for start in range(0,len(dates),500000):
        end=min(start+500000,len(dates))
        keys=pd.MultiIndex.from_arrays([pd.to_datetime(dates[start:end]),codes[stock[start:end]].astype(str)])
        values=align_events(events,days,keys,True)
        if resume_verified_partial:
            if not np.array_equal(block[start:end],values,equal_nan=True):raise ValueError('STOP partial block replay')
        else:block[start:end]=values
    if not resume_verified_partial:
        block.flush()
        np.save(out/'calendar.npy',days.to_numpy(),allow_pickle=False)
    outputs={p.name:dict(sha256=s.sha256(p),bytes=p.stat().st_size) for p in out.iterdir()}
    s.write_json(out/'manifest.json',dict(schema='step5-features-v1',contract=s.contract(),
        source_bundle_sha256=s.STEP3_BUNDLE_SHA, source_input={**info,'path':str(raw)},
        outputs=outputs, source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(), environment=dict(python=__import__('platform').python_version(),executable=__import__('sys').executable,data_root_env=__import__('os').getenv('MARKETMAMBA_DATA_ROOT'),data_realpath=str(Path('/home/frank/projects/MarketMamba/Data').resolve()),gpu_used=False), recovered_partial=resume_verified_partial, rows=len(dates),events=len(events),estimated_bytes=estimate,
        strict_pit='STOP', evidence_class='HISTORICAL_RESEARCH_ONLY',implementation={str(p.relative_to(s.ROOT)):s.sha256(p) for p in (s.ROOT/'tools').glob('step5*.py')}))
    print(json.dumps(dict(rows=len(dates),events=len(events),estimated_bytes=estimate,manifest_sha256=s.sha256(out/'manifest.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=s.OUTPUT_ROOT/'features');p.add_argument('--resume-verified-partial',action='store_true');a=p.parse_args();build(a.out,a.resume_verified_partial)

if __name__=='__main__':main()
