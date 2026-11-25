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


def cross_source_evidence(frame,raw):
    """Exact stock/month pairing; FinMind is a consistency reference, not PIT proof."""
    f=frame.loc[pd.to_datetime(frame.period)>=pd.Timestamp('2013-01-01'),
                ['stock_id','period','單月營收(千元)','input_row_number','證券代碼']].copy()
    r=raw[['stock_id','revenue_year','revenue_month','revenue']].copy()
    r['period']=pd.to_datetime(dict(year=r.revenue_year,month=r.revenue_month,day=1))
    r['stock_id']=r.stock_id.astype(str);f['stock_id']=f.stock_id.astype(str)
    if r.duplicated(['stock_id','period']).any():raise ValueError('STOP duplicate FinMind stock/month')
    f=f.merge(r[['stock_id','period','revenue']].rename(columns={'revenue':'reference_ntd'}),on=['stock_id','period'],how='left',validate='many_to_one')
    x=f['單月營收(千元)'].to_numpy(float)*1000;y=f.reference_ntd.to_numpy(float)
    paired=np.isfinite(x)&np.isfinite(y)
    relative=np.full(len(f),np.nan);nonzero=paired&(y!=0)
    relative[nonzero]=np.abs(x[nonzero]-y[nonzero])/np.abs(y[nonzero])
    relative[paired&(x==0)&(y==0)]=0
    relative[paired&(x!=0)&(y==0)]=np.inf
    f['relative_difference']=relative
    f['agreement']=paired&(relative<.001)
    f['classification']=np.select([~paired,relative<.001,relative>.01],
        ['UNPAIRED_OR_NONFINITE','AGREEMENT_LT_0_1PCT','DIFFERENCE_GT_1PCT'],default='BETWEEN_0_1_AND_1PCT_INCLUSIVE')
    # A name/code proxy only: absence of a marker does not establish domestic domicile.
    foreign=f['證券代碼'].astype(str).str.contains(r'KY|DR|外國',case=False,regex=True)|f.stock_id.str.startswith('91')
    f['company_proxy']=np.where(foreign,'KY_FOREIGN_IDENTIFIABLE','NO_FOREIGN_MARKER_NOT_DOMICILE_PROOF')
    f['year']=pd.to_datetime(f.period).dt.year
    def stats(g):
        n=int(g.reference_ntd.notna().sum());p=np.isfinite(g.relative_difference)|np.isinf(g.relative_difference)
        denominator=int(p.sum())
        return dict(rows=len(g),paired=denominator,lt_0_1pct=int(g.agreement.sum()),
            gt_1pct=int((g.relative_difference>.01).sum()),
            agreement_rate=float(g.agreement.sum()/denominator) if denominator else None,
            gt_1pct_rate=float((g.relative_difference>.01).sum()/denominator) if denominator else None,
            classifications={str(k):int(v) for k,v in g.classification.value_counts().items()})
    summary=dict(overall=stats(f),by_year={str(k):stats(g) for k,g in f.groupby('year')},
        by_company_proxy={str(k):stats(g) for k,g in f.groupby('company_proxy')},
        by_year_company_proxy={str(y)+'/'+str(c):stats(g) for (y,c),g in f.groupby(['year','company_proxy'])},
        interpretation='UNIT_AND_VALUE_CONSISTENCY_ONLY_NOT_FIRST_VERSION_OR_PUBLICATION_PIT_PROOF',
        relative_difference='abs(TEJ*1000-FinMind)/abs(FinMind); both zero=0; nonzero/zero=infinity')
    return f,summary


def admitted_events(frame,pairs=None):
    """Cross-source unit evidence only; version zero is replay order, not first-version certification."""
    f=frame[['stock_id','period','available_date','單月營收(千元)','currency','input_row_number']].copy()
    if pairs is None:valid=np.zeros(len(f),bool)
    else:
        if pairs.input_row_number.duplicated().any():raise ValueError('STOP duplicate cross-source identity')
        valid=f.input_row_number.map(pairs.set_index('input_row_number').agreement).fillna(False).to_numpy(bool)
    valid = valid & f.currency.isin(['TWD','NTD','TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING']).to_numpy()
    f['revenue']=np.where(valid,f['單月營收(千元)']*1000,np.nan)
    f['currency']=np.where(valid,'NTD_BY_CROSS_SOURCE_AGREEMENT','UNIT_UNVERIFIED')
    f['unit']=np.where(valid,'yuan','UNKNOWN');f['scope']='TEJ_201301_CONSOLIDATED_PREFERRED_OR_INDIVIDUAL_NO_SUBSIDIARY'
    f['comparability_id']=np.where(valid,'TEJ_201301_SCOPE_ASSUMPTION_PENDING_HUMAN','UNKNOWN')
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
    record('source_manifest',SOURCE/'.artifacts/tej/pit/manifest.json')
    comparison=json.loads((SOURCE/'.artifacts/tej/pit/manifest.json').read_text())['comparisons']['revenue_raw']
    s.verify_file(comparison['path'],comparison['sha256']);record('finmind_revenue',comparison['path'])
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
    pairs,summary=cross_source_evidence(f,pd.read_parquet(comparison['path']))
    pairs.to_parquet(out/'cross_source_pairs.parquet',index=False);record('cross_source_pairs',out/'cross_source_pairs.parquet')
    write_json(out/'cross_source_summary.json',summary);record('cross_source_summary',out/'cross_source_summary.json')
    invalid_currency=int((~pairs.agreement).sum())
    ev=s.monthly_events(admitted_events(f,pairs),days);del f,pairs
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
    data=SimpleNamespace(X=np.load(SHARED/'X.npy',mmap_mode='r'),date_idx=di,stock_idx=si,ends=ends,n_stocks=len(codes),feature_names=json.loads((SHARED/'manifest.json').read_text())['feature_order'])
    jobs={}
    # Only the original warmup mask (code zero) is reused. fold_rows computes
    # every annual split against the frozen full calendar, not 2026 train/eval codes.
    base_split=SHARED.parent/'folds/fold_2026/splits.npy';record('base_warmup',base_split)
    base_splits=np.load(base_split,mmap_mode='r')
    def save(name,value):
        path=out/(name+'.npy');np.save(path,value);record(name,path)
        return name,array_digest(value)
    for year in s.YEARS:
        split_name='base_warmup'
        tr,er=s.fold_rows(dates,base_splits,year,np.isfinite(labels),calendar=days)
        er=er[eligible[er]];feature_support(monthly[tr]);feature_support(monthly[er])
        key_file,key_hash=save(f'keys_{year}',np.column_stack([di[er],si[er]]))
        label_keys,_=save(f'label_keys_{year}',np.column_stack([di[er],si[er]]))
        eval_labels,_=save(f'eval_labels_{year}',labels[er])
        a,b,_=build_arms(data,er,monthly)
        evals={arm:save(f'eval_{year}_{arm}',x) for arm,x in [('A',a),('B',b)]}
        del a,b;gc.collect()
        for seed in s.SEEDS:
            rows,target=s.sample_training(tr,labels,di,seed)
            sample_file,sample_hash=save(f'sample_{year}_{seed}',rows)
            target_file,target_hash=save(f'target_{year}_{seed}',target)
            a,b,_=build_arms(data,rows,monthly)
            for arm,x in [('A',a),('B',b)]:
                fit_file,fit_hash=save(f'fit_{year}_{seed}_{arm}',x)
                eval_file,eval_hash=evals[arm]
                jobs[s.job_id(year,seed,arm)]=dict(year=year,seed=seed,arm=arm,
                    sample_file=sample_file,sample_sha256=sample_hash,target_file=target_file,target_sha256=target_hash,
                    keys_file=key_file,keys_sha256=key_hash,fit_file=fit_file,fit_sha256=fit_hash,
                    eval_file=eval_file,eval_sha256=eval_hash,splits_file=split_name,label_keys_file=label_keys,eval_labels_file=eval_labels)
            del a,b;gc.collect()
        print(json.dumps(dict(prepared_year=year,jobs=len(jobs))),flush=True)
    for category,ref in [('sampling','sample_file'),('target','target_file'),('keys','keys_file'),('splits','splits_file')]:
        files[category]=files[next(iter(jobs.values()))[ref]]
    sources={k:v for k,v in files.items() if not any(k.startswith(prefix) for prefix in ('sample_','target_','keys_','fit_','eval_','splits_','label_keys_'))}
    sources['splits']=files['splits']
    bundle=dict(contract=s.SPEC_SHA,source_commit=pre['source_commit'],source_identities=sources,
        evidence_class='HISTORICAL_COMPLETE_PRICE_RESEARCH_ONLY',source_gate='HUMAN_DECISION_REQUIRED',files=files,jobs=jobs,coverage=coverage)
    # Detect changes during preparation, including raw sizes paired with SHA.
    for info in files.values():s.verify_file(info['path'],info['sha256'],info['bytes'])
    write_json(out/'bundle.json',bundle)
    result=s.admission(SOURCE,out/'admission',out/'bundle.json')
    print(json.dumps(dict(status=result['status'],stops=result['stops'],bundle_sha256=s.sha256(out/'bundle.json'),identities=len(files),coverage=coverage),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
