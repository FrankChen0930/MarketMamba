"""Offline quarterly version audit; public aggregates, private rows only.

No market outcomes, network, training, ledger writes, or historical PIT admission.
Numerical agreement is diagnostic and never proves matching accounting scope.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tools.tej_pit import read_zip, dates, number, sha256, counts, distribution, BASIC_COLUMNS
from tools import pit_ledger

SELF_COLUMNS = ['證券代碼','年月','合併(Y/N)','發佈日期','營業收入淨額','營業成本','營業毛利','營業費用','營業利益','繼續營業部門稅前純益','本期稅後淨利','每股盈餘(元)','每股稅前淨利(元)']
METRICS = {'Revenue':('revenue','營業收入淨額_元'), 'GrossProfit':('gross_profit','營業毛利_元'),
           'OperatingIncome':('operating_income','營業利益_元'), 'EPS':('eps','eps')}
ROOT = Path(__file__).resolve().parents[1]


def load_self(path, sums):
    d = read_zip(path, sums, SELF_COLUMNS, exact=True)
    d['stock_id'] = d['證券代碼'].str.split().str[0]
    d['period'] = dates(d['年月'], '%Y%m')
    d['available_date'] = dates(d['發佈日期'], '%Y%m%d')
    if d.stock_id.isna().any() or d.period.isna().any() or not d.period.dt.month.isin([3,6,9,12]).all():
        raise ValueError('STOP invalid self business key')
    if not d['合併(Y/N)'].isin(['Y','N']).all():
        raise ValueError('STOP unknown self scope')
    if d.duplicated(['stock_id','period','合併(Y/N)']).any():
        raise ValueError('STOP duplicate self scope key; cannot select a version')
    for col in SELF_COLUMNS[4:]:
        d[col] = number(d[col])
    for out,col in [('revenue','營業收入淨額'),('gross_profit','營業毛利'),('operating_income','營業利益')]:
        d[out] = d[col]*1000
    d['eps'] = d['每股盈餘(元)']
    d['period_end'] = d.period + pd.offsets.MonthEnd(0)
    d['basis_status'] = 'UNKNOWN_SINGLE_OR_YTD'
    d['amount_unit_status'] = 'THOUSAND_TO_NTD_DIAGNOSTIC_HYPOTHESIS_NOT_DOCUMENT_VERIFIED'
    d['currency'] = 'UNKNOWN'
    d['version_semantics'] = 'SELF_REPORTED_SNAPSHOT_NOT_VERIFIED_FIRST_VERSION'
    d['strict_pit_eligible'] = False
    d['publication_date_usable'] = d.available_date.notna() & (d.available_date >= d.period_end)
    d['input_row_number'] = np.arange(2,len(d)+2)
    return d


def duplicate_audit(d):
    dup=d[d.duplicated(['stock_id','period'],keep=False)]
    groups=dup.groupby(['stock_id','period'])
    return {'company_period_duplicate_rows':len(dup),'company_period_groups':groups.ngroups,
            'excess_rows':int(d.duplicated(['stock_id','period']).sum()),
            'same_scope_duplicate_rows':int(d.duplicated(['stock_id','period','合併(Y/N)'],keep=False).sum()),
            'groups_with_two_scopes':int((groups['合併(Y/N)'].nunique()==2).sum()),
            'groups_with_different_dates':int((groups.available_date.nunique(dropna=False)>1).sum()),
            'groups_with_same_date':int((groups.available_date.nunique(dropna=False)==1).sum()),
            'different_values_by_metric':{v:int((groups[v].nunique(dropna=False)>1).sum()) for v,_ in METRICS.values()},
            'interpretation':'SCOPE_PAIRS_NOT_SAME_SCOPE_REVISION_HISTORY'}


def difference_class(a,b,metric):
    a,b=a.astype(float),b.astype(float)
    finite=np.isfinite(a)&np.isfinite(b)
    delta=(a-b).abs()
    relative=delta/b.abs().replace(0,np.nan)
    out=pd.Series('MISSING_OR_NONFINITE',index=a.index)
    out.loc[finite]='GT_1PCT'
    out.loc[finite & (relative<=.01)]='LE_1PCT'
    out.loc[finite & (relative<=.001)]='LE_0_1PCT'
    # Source display precision is a diagnostic tolerance, not an admission gate.
    out.loc[finite & (delta <= (.005 if metric=='eps' else 500))]='WITHIN_PRECISION'
    out.loc[finite & ((a*b)<0)]='SIGN'
    out.loc[finite & (b==0) & (a!=0)]='REFERENCE_ZERO'
    out.loc[finite & (a==b)]='EXACT'
    return out


def revision_class(old,new):
    if old.get('value') != new.get('value'):
        a,b=old.get('value'),new.get('value')
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<=1e-12:
            return 'FLOAT_REPRESENTATION_ONLY'
        return 'NUMERIC_CHANGE'
    return 'NAME_ONLY' if old.get('origin_name') != new.get('origin_name') else 'OTHER_METADATA_ONLY'


def observed_asof(d,cutoff):
    """Local observation time only; never backdate to reporting period."""
    return d[d.observed_at<=cutoff].sort_values('observed_at',kind='stable').drop_duplicates(['stock_id','period']+(['metric'] if 'metric' in d else []),keep='last')


def check_output(out):
    allowed=ROOT/'.artifacts/tej/quarterly'
    lexical=Path(os.path.abspath(out))
    if not lexical.resolve().is_relative_to(allowed) or any(p.is_symlink() for p in [lexical,*lexical.parents]):
        raise ValueError('STOP row output must remain in checkout .artifacts/tej/quarterly without symlinks')
    if lexical.exists() and any(lexical.iterdir()):
        raise ValueError('STOP output namespace populated')
    return lexical


def formal_table(f):
    if f.duplicated(['stock_id','period']).any():
        raise ValueError('STOP formal duplicate keys')
    d=f[['stock_id','period','period_start','period_end','available_date','合併(Y/N)','幣別','單季(Q)/單半年(H)','財報類別（1個別2個體3合併）','current_financial_sector','current_industry']].copy()
    for metric,(v,col) in METRICS.items():
        d[v]=f[col]
    d['nominal_end']=d.period+pd.offsets.MonthEnd(0)
    d['lag_days']=(d.available_date-d.period_end).dt.days
    d['structure_candidate']=(d['單季(Q)/單半年(H)']=='Q') & (d.period_end==d.nominal_end) & (d.period_start.dt.month==1) & d.period_end.notna() & d['合併(Y/N)'].isin(['Y','N']) & d['幣別'].isin(['NTD','TWD'])
    d['date_candidate']=d.available_date.notna() & d.lag_days.between(0,365)
    d['strict_pit_eligible']=False
    # Explicit UNKNOWN is not a negative financial-sector classification.
    d['historical_industry_status']='UNKNOWN_CURRENT_SNAPSHOT_ONLY'
    return d


def reconcile(left,right,pair,keys,metrics):
    """Full outer join keeps both unpaired denominators; rejects ambiguous keys."""
    for d in [left,right]:
        if d.duplicated(keys).any():
            raise ValueError('STOP reconciliation duplicate key')
    j=left.merge(right,on=keys,how='outer',suffixes=('_left','_right'),indicator=True,validate='one_to_one')
    parts=[]
    for metric in metrics:
        a=j[metric+'_left']; b=j[metric+'_right']
        x=j[keys].copy()
        x['pair']=pair; x['metric']=metric
        x['left_value']=a; x['right_value']=b
        x['key_coverage']=j['_merge'].astype(str)
        x['difference_class']=difference_class(a,b,metric)
        x['absolute_difference']=(a-b).abs()
        x['relative_difference']=(a-b).abs()/b.abs().replace(0,np.nan)
        x['comparison_status']='DIAGNOSTIC_SCOPE_CURRENCY_BASIS_VERSION_NOT_FULLY_VERIFIED'
        parts.append(x)
    return pd.concat(parts,ignore_index=True)


def cumulative_diagnostic(f):
    """Sum only observed Q amounts, consecutive within same year/scope/currency.

    This is an as-exported basis diagnostic; publication versions may differ.
    Never sum EPS or treat these values as a historical known-at series.
    """
    d=f.copy().sort_values(['stock_id','period'])
    d['year']=d.period.dt.year
    d['quarter']=d.period.dt.quarter
    group=['stock_id','year','合併(Y/N)','幣別']
    d['n_prior']=d.groupby(group).cumcount()+1
    supported=d.structure_candidate & (d.n_prior==d.quarter)
    # Any non-Q/invalid prior row invalidates the whole prefix.
    supported &= d.groupby(group).structure_candidate.cummin()
    out=d[['stock_id','period','合併(Y/N)']].copy()
    for metric in ['revenue','gross_profit','operating_income']:
        values=d.groupby(group)[metric].cumsum()
        finite_count=d[metric].notna().groupby([d[c] for c in group]).cumsum()
        out[metric]=values.where(supported & (finite_count==d.quarter))
    return out


def ledger_evidence(ledger,f):
    db=ledger/'ledger.sqlite3'
    # immutable prevents journal/WAL/SHM creation; reject live sidecars rather than ignore them.
    if any(Path(str(db)+s).exists() for s in ['-wal','-journal']):
        raise ValueError('STOP live ledger sidecar; immutable snapshot unavailable')
    with sqlite3.connect(f'file:{db}?mode=ro&immutable=1',uri=True) as c:
        head,last,initial=pit_ledger.validate(c,ledger)
        runs={rid:json.loads(s) for rid,s in c.execute('SELECT run_id,summary_json FROM runs')}
        result=[]
        for eid,rid,kind,key,new,old in c.execute("SELECT id,run_id,kind,key,value_json,previous_json FROM events WHERE table_name='financials_raw' ORDER BY id"):
            k=json.loads(key); n=json.loads(new) if new else {}; o=json.loads(old) if old else {}
            metric=k['type']; period=pd.Timestamp(k['Date']).to_period('M').to_timestamp()
            match=f[(f.stock_id==k['stock_id']) & (f.period==period)]
            row={'event_id':eid,'run_id':rid,'stock_id':k['stock_id'],'period':period,'metric':metric,'kind':kind,
                 'observed_at':runs[rid]['observed_at'],'old_value':o.get('value'),'new_value':n.get('value'),
                 'revision_class':revision_class(o,n) if kind=='REVISED' else kind,
                 'old_name':o.get('origin_name'),'new_name':n.get('origin_name'),
                 'formal_match':'NOT_TARGET_METRIC_OR_MISSING','strict_historical_replay':False}
            if metric in METRICS and len(match)==1:
                value=match.iloc[0][METRICS[metric][0]]
                tol=0  # exact numeric version evidence; precision bands belong to diagnostics
                old_match=o.get('value') is not None and pd.notna(value) and abs(value-o['value'])<=tol
                new_match=n.get('value') is not None and pd.notna(value) and abs(value-n['value'])<=tol
                row['formal_match']='BOTH' if old_match and new_match else 'NEW_ONLY' if new_match else 'OLD_ONLY' if old_match else 'NEITHER'
            result.append(row)
        events=pd.DataFrame(result)
        base=initial['tables']['financials_raw']
        # Verify actual per-key transitions against immutable base, independently of
        # cross-source matches. Keep all event keys, including non-target FIRST_SEEN.
        keys={pit_ledger.canonical(json.loads(k)) for (k,) in c.execute(
            "SELECT key FROM events WHERE table_name='financials_raw'")}
        stocks=sorted({json.loads(k)['stock_id'] for k in keys})
        base_path=ledger/'blobs'/(base['base']['sha256']+'.parquet')
        base_rows=pd.read_parquet(base_path,filters=[('stock_id','in',stocks)])
        state={}
        for record in base_rows.to_dict('records'):
            key=pit_ledger.canonical({k:record[k] for k in pit_ledger.KEYS['financials_raw']})
            if key in keys:
                state[key]=pit_ledger.canonical(record)
        replay_rows=[]
        initial_id=initial['run_id']
        for key,value in state.items():
            replay_rows.append({'run_id':initial_id,'observed_at':initial['observed_at'],
                                'key':key,'value_json':value,'kind':'INITIAL_STOCK'})
        transition_count=0
        for rid,run in runs.items():
            for key,kind,new,old in c.execute(
                    "SELECT key,kind,value_json,previous_json FROM events WHERE table_name='financials_raw' AND run_id=? ORDER BY id",(rid,)):
                if state.get(key)!=old:
                    raise ValueError('STOP ledger previous value does not replay from base')
                transition_count+=1
                if kind=='DISAPPEARED':
                    state.pop(key,None)
                else:
                    state[key]=new
                replay_rows.append({'run_id':rid,'observed_at':run['observed_at'],'key':key,
                                    'value_json':new,'kind':kind})
        summary={'validated_run_count':len(runs),'head_digest':head,'first_observed_at':initial['observed_at'],
                 'last_observed_at':last['observed_at'],'initial_stock_rows':base['counts']['INITIAL_STOCK'],
                 'event_rows':len(events),'events':counts(events,['kind','metric','revision_class','formal_match']),
                 'local_observation_replay':'VALIDATED_BASE_PLUS_SEALED_EVENT_OVERLAY_AT_RUN_CUTOFF',
                 'historical_first_publication_replay':'UNKNOWN_NOT_INFERRED_FROM_LOCAL_OBSERVATION',
                 'base_identity':base['base'],'index_identity':base['index'],
                 'replayed_event_transitions':transition_count,'replayed_business_keys':len(keys),
                 'target_first_seen_events':int(((events.kind=='FIRST_SEEN') & events.metric.isin(METRICS)).sum())}
    return summary,events,pd.DataFrame(replay_rows)


def run(raw_dir,ifrs_path,financials_path,ledger,out,report_json):
    out=check_output(out)
    if Path(report_json).resolve()!=ROOT/'reports/tej_quarterly_versions_20261126.json':
        raise ValueError('STOP public summary destination must be fixed report JSON')
    raw_dir=Path(raw_dir); ifrs_path=Path(ifrs_path); financials_path=Path(financials_path); ledger=Path(ledger)
    paths={'self_zip':raw_dir/'自結財務資訊2013_2026_export20261126.zip','new_sums':raw_dir/'SHA256SUMS.20261126',
           'original_sums':raw_dir/'SHA256SUMS','ifrs':ifrs_path,'financials':financials_path,
           'ledger_db':ledger/'ledger.sqlite3','basic_zip':raw_dir/'monthly_revenue_full_basicinfo_202511_202610_export20261125.zip',
           'self_doc':raw_dir/'docs/self_reported_quarterly_field_doc.html','docs_sums':raw_dir/'docs/SHA256SUMS','ifrs_manifest':ifrs_path.parent/'manifest.json'}
    identities={k:{'path':str(p.resolve()),'sha256':sha256(p),'bytes':p.stat().st_size} for k,p in paths.items()}
    prior=json.loads(paths['ifrs_manifest'].read_text())
    expected=prior['outputs']['ifrs_quarterly']['sha256']
    if identities['ifrs']['sha256']!=expected:
        raise ValueError('STOP formal parquet differs from producer manifest')
    # Documentation is local-only and must match its supplied checksum too.
    from tools.tej_pit import verify_zip
    verify_zip(paths['self_doc'],paths['docs_sums'])
    d=load_self(paths['self_zip'],paths['new_sums'])
    f=formal_table(pd.read_parquet(ifrs_path))
    basic=read_zip(paths['basic_zip'],paths['original_sums'],BASIC_COLUMNS)
    basic['stock_id']=basic['證券代碼'].str.split().str[0]
    basic=basic.sort_values('年月').drop_duplicates('stock_id',keep='last').set_index('stock_id')
    for col in ['上市別','TSE產業_名稱']:
        d[col]=d.stock_id.map(basic[col])
    d['classification_evidence']='CURRENT_SNAPSHOT_NOT_HISTORICAL'
    d['year']=d.period.dt.year; d['quarter']=d.period.dt.quarter
    dup=duplicate_audit(d)
    paired=d.merge(f[['stock_id','period','available_date','period_end','合併(Y/N)']],on=['stock_id','period','合併(Y/N)'],how='left',suffixes=('','_formal'),validate='one_to_one')
    paired['publication_minus_end_days']=(paired.available_date-paired.period_end).dt.days
    paired['publication_minus_formal_days']=(paired.available_date-paired.available_date_formal).dt.days
    paired['publication_minus_actual_formal_end_days']=(paired.available_date-paired.period_end_formal).dt.days
    metrics=[v for v,_ in METRICS.values()]
    rows=[reconcile(d,f,'SELF_VS_FORMAL',['stock_id','period','合併(Y/N)'],metrics),
          reconcile(d,cumulative_diagnostic(f),'SELF_VS_FORMAL_YTD_DIAGNOSTIC',['stock_id','period','合併(Y/N)'],metrics[:3])]
    raw=pd.read_parquet(financials_path,columns=['stock_id','Date','type','value','origin_name'])
    raw=raw[raw.type.isin(METRICS)].copy()
    raw['period']=pd.to_datetime(raw.Date).dt.to_period('M').dt.to_timestamp()
    raw_dups=int(raw.duplicated(['stock_id','period','type'],keep=False).sum())
    raw=raw[~raw.duplicated(['stock_id','period','type'],keep=False)]
    # Raw lacks scope/currency/basis/publication columns; never pretend they match.
    wide=raw.pivot(index=['stock_id','period'],columns='type',values='value').rename(columns={k:v[0] for k,v in METRICS.items()}).reset_index()
    rows.append(reconcile(f,wide,'FORMAL_VS_FINMIND',['stock_id','period'],metrics))
    # Compare only self scope selected by formal scope; don't select Y over N arbitrarily.
    self_selected=d.merge(f[['stock_id','period','合併(Y/N)']],on=['stock_id','period','合併(Y/N)'],how='inner',validate='one_to_one')
    rows.append(reconcile(self_selected,wide,'SELF_FORMAL_SCOPE_VS_FINMIND',['stock_id','period'],metrics))
    comparisons=pd.concat(rows,ignore_index=True)
    comparisons['year']=comparisons.period.dt.year
    structure=f.set_index(['stock_id','period']).structure_candidate
    comparisons['formal_structure_candidate']=[structure.get((a,b),False) for a,b in zip(comparisons.stock_id,comparisons.period)]
    triples=self_selected[['stock_id','period']+metrics].merge(
        f[['stock_id','period']+metrics],on=['stock_id','period'],suffixes=('_self','_formal'),validate='one_to_one').merge(
        wide,on=['stock_id','period'],how='outer',validate='one_to_one')
    triples['year']=triples.period.dt.year
    triple_summary=[]
    for year,g in triples.groupby('year'):
        for metric in metrics:
            finite=np.isfinite(g[[metric+'_self',metric+'_formal',metric]]).all(axis=1)
            exact=(g[metric+'_self']==g[metric+'_formal']) & (g[metric+'_formal']==g[metric])
            triple_summary.append({'year':int(year),'metric':metric,'outer_keys':len(g),'finite_three_way':int(finite.sum()),'exact_three_way':int((finite&exact).sum()),'verified_comparable':0})
    ledger_summary,events,replay_versions=ledger_evidence(ledger,f)
    # Cross-reference local EPS changes with the exported self scope, privately.
    if len(events):
        events=events.merge(self_selected[['stock_id','period','eps','available_date']],on=['stock_id','period'],how='left',validate='many_to_one')
        events['self_matches_old_eps']=(events.metric=='EPS') & ((events.eps-events.old_value).abs()<=.005)
        events['self_matches_new_eps']=(events.metric=='EPS') & ((events.eps-events.new_value).abs()<=.005)
        ledger_summary['self_eps_support']=counts(events[events.metric=='EPS'],['revision_class','self_matches_old_eps','self_matches_new_eps'])
    coverage=[]
    for (year,quarter),g in d.groupby(['year','quarter']):
        coverage.append({'year':int(year),'quarter':int(quarter),'rows':len(g),'companies':g.stock_id.nunique(),
                         'finite_by_metric':{v:int(g[v].notna().sum()) for v in metrics}})
    summary={'status':'RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED','evidence_class':'DESCRIPTIVE_SOURCE_VERSION_AUDIT',
             'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'implementation_sha256':sha256(__file__),'inputs':identities,
             'runtime':{'python':platform.python_version(),'pandas':pd.__version__,'executable':os.sys.executable,
                        'data_root_env':os.environ.get('MARKETMAMBA_DATA_ROOT'),'checkout_data_realpath':str((ROOT/'Data').resolve()),'device':'CPU'},
             'split_support':'NO_RETURNS_NO_LABELS_DATA_ONLY','model_identity':'N/A',
             'self':{'rows':len(d),'companies':d.stock_id.nunique(),'duplicates':dup,'coverage_by_quarter':coverage,
                     'coverage_by_year_market_current_proxy':counts(d.drop_duplicates(['stock_id','year']),['year','上市別']),
                     'coverage_by_year_industry_current_proxy':counts(d.drop_duplicates(['stock_id','year']),['year','TSE產業_名稱']),
                     'missing_by_year_metric':{v:counts(d[d[v].isna()],['year']) for v in metrics},
                     'release_lag_nominal_end':distribution(paired.publication_minus_end_days),
                     'release_minus_formal':distribution(paired.publication_minus_formal_days),
                     'release_minus_actual_formal_end':distribution(paired.publication_minus_actual_formal_end_days),
                     'timing_by_year':{str(y):{'end':distribution(g.publication_minus_end_days),'formal':distribution(g.publication_minus_formal_days)} for y,g in paired.groupby('year')},
                     'missing_release':int(d.available_date.isna().sum()),'missing_formal_same_scope_release':int(paired.available_date_formal.isna().sum()),
                     'same_scope_multiple_versions':0,'strict_replayable_versions':0,'historical_first_version_unknown_fraction':1.0},
             'formal':{'rows':len(f),'structure_candidate':int(f.structure_candidate.sum()),'structure_and_date_candidate':int((f.structure_candidate&f.date_candidate).sum()),
                       'strict_historical_eligible':0,'first_vs_revised_unknown_rows':len(f),'first_vs_revised_unknown_fraction':1.0},
             'comparisons':counts(comparisons,['pair','year','metric','key_coverage','difference_class']),
             'triple_summary':triple_summary,
             'structural_comparison_counts':counts(comparisons,['pair','metric','formal_structure_candidate','difference_class']),
             'comparison_totals':counts(comparisons,['pair','metric','key_coverage','difference_class']),
             'raw_duplicate_key_rows_excluded':raw_dups,'ledger':ledger_summary,
             'limitations':['Self export lacks currency, start date, Q/H and first/revision identity; amount x1000 is diagnostic hypothesis only.',
                            'FinMind lacks scope/currency/basis/release metadata; all cross-source comparisons are diagnostic, not verified comparable denominators.',
                            'YTD diagnostic sums Q amounts only with complete prefixes; never sums EPS; mixed export versions do not establish historical YTD.',
                            'Self scope pairs are not same-scope revision events. Current market/industry snapshots cannot explain historical mandatory coverage causally.',
                            'Local FIRST_SEEN/INITIAL_STOCK are not first company publications. REVISED includes name-only changes.',
                            'No independent review or Human acceptance; no historical strict PIT admission.'],
             'outputs':{}}
    for k,p in paths.items():
        if sha256(p)!=identities[k]['sha256']:
            raise ValueError('STOP source changed during audit: '+k)
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('self_reported',d),('timing',paired),('comparisons',comparisons),('ledger_events',events),('replay_versions',replay_versions),('formal_admission',f)]:
        p=out/(name+'.parquet'); frame.to_parquet(p,index=False)
        summary['outputs'][name]={'path':str(p),'rows':len(frame),'bytes':p.stat().st_size,'sha256':sha256(p)}
    manifest=out/'manifest.json'
    manifest.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary['manifest_sha256']=sha256(manifest)
    Path(report_json).write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--ifrs',type=Path,required=True)
    p.add_argument('--financials',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--report-json',type=Path,default=ROOT/'reports/tej_quarterly_versions_20261126.json')
    a=p.parse_args()
    s=run(a.raw_dir,a.ifrs,a.financials,a.ledger,a.out_dir,a.report_json)
    print(json.dumps({'self_rows':s['self']['rows'],'duplicates':s['self']['duplicates'],'formal':s['formal'],'manifest_sha256':s['manifest_sha256']},ensure_ascii=False))


if __name__=='__main__':
    main()
