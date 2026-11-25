"""Prepare frozen Step3 historical resource bundle; no evaluation or score reads.

Coordinator A-D ruling pending independent review. All generated rows private.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tools import step3 as s
from tools.step3_audit import validate_selection_prices
from tools.step3_runner import build_arms, feature_support, array_digest

SOURCE=Path('/home/frank/projects/MarketMamba')
SHARED=Path('/home/frank/projects/p4-bundles/prepared/shared')


def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def admitted_events(frame):
    """Only first single-month source column, never exported prior/YTD values.

    Currency/unit mapping is licensed by the pinned TEJ original field document,
    not by equality of missing consolidated flags. Invalid units remain missing.
    """
    f=frame[['stock_id','period','available_date','單月營收(千元)','currency','input_row_number']].copy()
    valid=f.currency.isin(['TWD','NTD','TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING'])
    f['revenue']=f['單月營收(千元)']*1000
    f.loc[~valid,'revenue']=np.nan
    f['currency']=np.where(valid,'NTD','UNKNOWN')
    f['unit']='yuan';f['scope']='TEJ_201301_CONSOLIDATED_PREFERRED_OR_INDIVIDUAL_NO_SUBSIDIARY'
    f['comparability_id']='TEJ_DOC_2e0e9b5e_201301_ONWARD_FIRST_MONTH'
    f['version']=0;f['source_row']=f.input_row_number
    return f


def prepare(out):
    out=s.private_dir(out)
    # Reuse existing canonical artifacts; do not regenerate/modify shared matrices.
    pre=s.admission(SOURCE,out/'source-check')
    if any('bundle missing' not in reason for reason in pre['stops']):raise ValueError('STOP source checks')
    files={}
    def record(key,path):
        path=Path(path).resolve();files[key]=dict(path=str(path),sha256=s.sha256(path),bytes=path.stat().st_size)
    for key,name in [('matrix','X.npy'),('membership','stock_ids.npy'),('matrix_dates','dates.npy'),('matrix_ends','end_indices.npy'),('stock_index','stock_idx.npy')]:record(key,SHARED/name)
    record('feature_order',SHARED/'manifest.json')
    fold=SHARED.parent/'folds/fold_2026/splits.npy';record('splits',fold)
    prices=SOURCE/'.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet'
    record('prices',prices);record('price_manifest',SOURCE/'.artifacts/p4b/snapshot-manifest-20261125.json')
    pm=json.loads(Path(files['price_manifest']['path']).read_text())
    if files['prices']['sha256']!=pm['files']['prices_raw.parquet']['sha256']:raise ValueError('STOP price snapshot')
    for i,(path,info) in enumerate(pre['identities'].items()):
        record('source_'+str(i),path)
    record('version_evidence',SOURCE/'.artifacts/tej/raw/docs/monthly_revenue_field_doc.html')
    for path in [*sorted((s.ROOT/'tools').glob('step3*.py')),s.ROOT/'V6/research_lab/c0plus.py',s.ROOT/'V6/research_lab/data.py']:
        record('implementation' if path.name=='step3.py' else 'code_'+path.name,path)
    dates=np.load(SHARED/'dates.npy',mmap_mode='r');ids=np.load(SHARED/'stock_ids.npy',mmap_mode='r')
    codes=np.unique(ids);ends=np.load(SHARED/'end_indices.npy');di=np.repeat(np.arange(len(ends),dtype=np.int32),np.diff(ends,prepend=0))
    si=np.load(SHARED/'stock_idx.npy',mmap_mode='r')
    if not np.array_equal(codes[si],ids):raise ValueError('STOP stock mapping')
    raw=pd.read_parquet(prices,columns=['Date','stock_id','Open','High','Low','Close','Volume'])
    raw['Date']=pd.to_datetime(raw.Date);days=pd.DatetimeIndex(raw.Date.unique()).sort_values()
    raw=raw[raw.stock_id.isin(codes)].rename(columns={'Date':'date','stock_id':'code',**{k:k.lower() for k in ['Open','High','Low','Close','Volume']}})
    raw['adj_close']=np.nan;checked=validate_selection_prices(raw);del raw
    def panel(col):return checked.pivot(index='date',columns='code',values=col).reindex(index=days,columns=codes)
    valid=panel('observation_valid').eq(True).to_numpy();op=panel('open').to_numpy();cl=panel('close').to_numpy()
    ti=days.get_indexer(pd.DatetimeIndex(dates))
    if (ti<0).any():raise ValueError('STOP calendar coverage')
    labels=s.h20_labels(op,cl,valid)[ti,si];del checked,op,cl,valid
    member=np.zeros((len(ends),len(codes)),bool);member[di,si]=True
    eligible=s.w60_eligible(member)[di,si];del member
    np.save(out/'calendar.npy',days.to_numpy());record('calendar',out/'calendar.npy')
    np.save(out/'h20.npy',labels);record('labels',out/'h20.npy')
    f=pd.read_parquet(SOURCE/'.artifacts/tej/pit/monthly_revenue.parquet')
    invalid_currency=int((~f.currency.isin(['TWD','NTD','TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING'])).sum())
    ev=s.monthly_events(admitted_events(f),days);del f
    write_json(out/'event_reasons.json',ev.attrs);record('event_reasons',out/'event_reasons.json')
    # pandas 3 deep-copies attrs into every group; audit metadata is durable above.
    ev.attrs={}
    ev.to_parquet(out/'monthly_events.parquet',index=False);record('events',out/'monthly_events.parquet')
    monthly=np.full((len(dates),5),np.nan,np.float32)
    for code,g in ev.groupby('stock_id',sort=False):
        if code not in codes:continue
        j=int(np.searchsorted(codes,code));rows=np.flatnonzero(si==j)
        ix=np.searchsorted(g.session.to_numpy(),ti[rows],side='right')-1;ok=ix>=0
        monthly[rows[ok]]=g[s.FEATURES].to_numpy(np.float32)[ix[ok]]
    np.save(out/'monthly.npy',monthly);record('features',out/'monthly.npy')
    coverage={}
    for year in [2013,2014,2026]:
        a=monthly[dates.astype('datetime64[Y]')==np.datetime64(str(year),'Y')]
        coverage[str(year)]={'rows':len(a),'finite':dict(zip(s.FEATURES,np.isfinite(a).sum(axis=0).tolist()))}
    write_json(out/'coverage.json',dict(by_year=coverage,invalid_currency_rows=invalid_currency,invalid_currency_reason='NON_NTD_OR_UNKNOWN_UNIT_NAN',scope='201301_ONWARD_ONLY'))
    splits=np.load(fold,mmap_mode='r');tr,er=s.fold_rows(dates,splits,2026,np.isfinite(labels),calendar=days)
    er=er[eligible[er]];feature_support(monthly[tr]);feature_support(monthly[er])
    rows,target=s.sample_training(tr,labels,di,17)
    np.save(out/'sample.npy',rows);record('sampling',out/'sample.npy')
    np.save(out/'target.npy',target);record('target',out/'target.npy')
    keys=np.column_stack([di[er],si[er]]);np.save(out/'keys.npy',keys);record('keys',out/'keys.npy')
    data=SimpleNamespace(X=np.load(SHARED/'X.npy',mmap_mode='r'),date_idx=di,stock_idx=si,ends=ends,n_stocks=len(codes),feature_names=json.loads((SHARED/'manifest.json').read_text())['feature_order'])
    # Persist requested rows only; release price and event intermediates first.
    del ev,ids,labels,eligible,tr;gc.collect()
    jobs={}
    for kind,idx in [('fit',rows),('eval',er)]:
        a,b,names=build_arms(data,idx,monthly)
        for arm,x in [('A',a),('B',b)]:
            path=out/f'{kind}_{arm}.npy';np.save(path,x);record(f'{kind}_{arm}',path)
            jobs.setdefault(arm,{})[kind+'_sha256']=array_digest(x)
        del a,b;gc.collect()
    for arm in 'AB':jobs[arm].update(sample_sha256=array_digest(rows),target_sha256=array_digest(target),keys_sha256=array_digest(keys))
    bundle=dict(contract=s.SPEC_SHA,source_commit=pre['source_commit'],evidence_class='HISTORICAL_COMPLETE_PRICE_RESEARCH_ONLY',coordinator_ruling='A-D; PENDING_INDEPENDENT_REVIEW',files=files,jobs=jobs,year=2026,seed=17,coverage=coverage)
    # Detect changes during preparation, including raw sizes paired with SHA.
    for info in files.values():s.verify_file(info['path'],info['sha256'],info['bytes'])
    write_json(out/'bundle.json',bundle)
    result=s.admission(SOURCE,out/'admission',out/'bundle.json')
    print(json.dumps(dict(status=result['status'],stops=result['stops'],bundle_sha256=s.sha256(out/'bundle.json'),identities=len(files),coverage=coverage),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
