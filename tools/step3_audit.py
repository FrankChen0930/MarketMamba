"""Step3 selection/resource accounting; never loads historical model scores."""
from __future__ import annotations
import argparse
import json
import resource
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tools.step3 import ROOT, SPEC_SHA, private_dir, sha256, verify_file, selection_table, w60_eligible


def validate_selection_prices(frame):
    """Match 2b's exact validation boundary, including first-observation jump flags."""
    from research_lab.data import validate_prices
    return validate_prices(frame[frame.date>=pd.Timestamp('2013-01-01')])


def selection_audit(out,source=Path('/home/frank/projects/MarketMamba'),shared=Path('/home/frank/projects/p4-bundles/prepared/shared')):
    start=time.monotonic();out=private_dir(out)
    # Only identity metadata is accessed; no diagnostic metrics or scores are read.
    frozen=json.loads((ROOT/'reports/tej_diag_20261125.json').read_text())['identities']
    inputs={}
    for p in [shared/'dates.npy',shared/'stock_ids.npy',shared/'manifest.json',
              source/'.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet',
              source/'.artifacts/p4b/snapshot-manifest-20261125.json']:
        info=frozen[str(p)];inputs[str(p)]=verify_file(p,info['sha256'],info['bytes'])
    dates=np.load(shared/'dates.npy',mmap_mode='r',allow_pickle=False)
    ids=np.load(shared/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
    codes=np.unique(ids)
    raw=pd.read_parquet(source/'.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet',
        columns=['Date','stock_id','Open','High','Low','Close','Volume'])
    raw['Date']=pd.to_datetime(raw.Date)
    # Full observed union retained to calculate W60, including pre-2013 history.
    days=pd.DatetimeIndex(raw.Date.unique()).sort_values()
    raw=raw[raw.stock_id.isin(codes)].copy()
    raw=raw.rename(columns={'Date':'date','stock_id':'code',**{k:k.lower() for k in ['Open','High','Low','Close','Volume']}})
    raw['adj_close']=np.nan
    checked=validate_selection_prices(raw);del raw
    quality=checked.attrs['quality']
    valid=checked.pivot(index='date',columns='code',values='observation_valid').reindex(index=days,columns=codes).eq(True).to_numpy()
    dollars=checked.pivot(index='date',columns='code',values='close').reindex(index=days,columns=codes)
    dollars*=checked.pivot(index='date',columns='code',values='volume').reindex(index=days,columns=codes)
    dollars=dollars.where(valid)
    # Strictly prior information; classification is engineering disclosure, not a trading gate.
    liq=dollars.rolling(20,min_periods=20).mean().shift(1).rank(axis=1,pct=True).to_numpy(np.float32)
    del checked,dollars
    member=np.zeros(valid.shape,bool)
    ti=days.get_indexer(pd.DatetimeIndex(dates));si=pd.Index(codes).get_indexer(ids)
    if (si<0).any():raise ValueError('STOP matrix stock mapping')
    absent=int((ti<0).sum());good=ti>=0
    member[ti[good],si[good]]=True
    if int(member.sum())!=int(good.sum()):raise ValueError('STOP duplicate matrix key')
    del ti,si,good,ids,dates
    result=selection_table(days,member,valid)
    result=[r for r in result if 2013<=r['year']<=2026]
    w=w60_eligible(member)
    # Preserve rows privately, including overlapping flags and unknown lifecycle.
    t,s=np.where(member & (np.asarray(days.year)>=2013)[:,None])
    rows=pd.DataFrame(dict(date=days[t],stock_id=codes[s],w60=w[t,s],valid_signal_price=valid[t,s],
        liquidity_prior20_pct=liq[t,s],lifecycle_status='UNKNOWN'))
    rows.to_parquet(out/'selection_flags.parquet',index=False)
    strata=[]
    for year in range(2013,2027):
        ym=np.asarray(days.year==year)[:,None]
        for name,mask in [('low',liq<=1/3),('middle',(liq>1/3)&(liq<=2/3)),('high',liq>2/3),('unknown',~np.isfinite(liq))]:
            n=int((member&ym&mask).sum());kept=int((member&ym&mask&w).sum())
            strata.append(dict(year=year,lifecycle='UNKNOWN',liquidity=name,members=n,w60=kept,
                               w60_fraction=kept/n if n else None))
    summary=dict(status='STOP_STRICT_ADMISSION',evidence_class='COMPLETE_PRICE_RESEARCH_SELECTION_ONLY',
        contract_sha256=SPEC_SHA,inputs=inputs,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        implementation={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'tools/step3.py',Path(__file__)]},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            executable=os.sys.executable,MARKETMAMBA_DATA_ROOT=os.getenv('MARKETMAMBA_DATA_ROOT'),
            data_realpath=str((ROOT/'Data').resolve()),address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0]),calendar_type='OBSERVED_PRICE_UNION_NOT_OFFICIAL',price_validation_start='2013-01-01_MATCH_2B',
        calendar_first=str(days.min().date()),calendar_last=str(days.max().date()),sessions=len(days),
        matrix_dates_absent_from_price_union=absent,
        matrix_signal_first=str(rows.date.min().date()),matrix_signal_last=str(rows.date.max().date()),price_quality=quality,by_year=result,strata=strata,
        liquidity_definition='previous-session 20-observed-session mean raw close*volume; daily percentile tertiles',
        lifecycle_limitation='All lifecycle unknown; no suspension/delisting/action attribution inferred from prices',
        resources=dict(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        outputs={'selection_flags.parquet':dict(sha256=sha256(out/'selection_flags.parquet'),bytes=(out/'selection_flags.parquet').stat().st_size)})
    # Reverify inputs after the read; identity changes fail closed.
    for path,info in inputs.items():verify_file(path,info['sha256'],info['bytes'])
    with (out/'selection_summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=summary['status'],years=len(result),member_rows=sum(r['members'] for r in result),
        absent_matrix_dates=absent,resources=summary['resources']),indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();selection_audit(a.out)

if __name__=='__main__':main()
