"""Offline licensed TEJ data audit. Row outputs belong exclusively in .artifacts/tej.

Publication dates describe source metadata, not proof of historical value versions.
No prices, returns, labels, network, ledger writes or production imports.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np
import pandas as pd

MONTHLY_COLUMNS = ['證券代碼','年月','營收發布日','單月營收(千元)','去年單月營收(千元)','累計營收(千元)','去年累計營收(千元)','盈餘發布日','合併營收(Y/N)','以合併為主單月營收(千元)']
IFRS_COLUMNS = ['證券代碼','年月','財報發布日','季別','單季(Q)/單半年(H)','月份','財報年月起日','財報年月迄日','合併(Y/N)','財報類別（1個別2個體3合併）','幣別','TEJ主產業代碼','營業收入淨額','營業成本','營業毛利','營業利益','稅前淨利','歸屬母公司淨利（損）','每股盈餘','資產總額','負債總額','股東權益總額','營業費用']
BASIC_COLUMNS = ['證券代碼','年月','TSE產業_名稱','TEJ產業_名稱','會計月份','目前狀態','上市別']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(path, sums):
    path = Path(path)
    matches = []
    for line in Path(sums).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if Path(name.lstrip('*')).name == path.name:
            matches.append(digest)
    actual = sha256(path)
    if len(matches) != 1 or matches[0] != actual:
        raise ValueError('STOP SHA256 mismatch or ambiguous checksum: '+path.name)
    return actual


def read_zip(path, sums, columns, exact=False):
    digest = verify_zip(path,sums)
    with zipfile.ZipFile(path) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if len(members) != 1 or members[0].file_size > 512*1024*1024:
            raise ValueError('STOP unexpected ZIP members/size')
        # Stream without extraction; do not decode unused basic-info values (e.g. prices).
        with z.open(members[0]) as f:
            header = io.TextIOWrapper(f,encoding='utf-16').readline().rstrip('\r\n').split('\t')
        names = [c.strip() for c in header]
        if len(set(names)) != len(names) or not set(columns) <= set(names) or (exact and names != columns):
            raise ValueError('STOP schema mismatch')
        with z.open(members[0]) as f:
            d = pd.read_csv(f,sep='\t',encoding='utf-16',dtype=str,keep_default_na=False,
                            usecols=[i for i,c in enumerate(names) if c in columns])
        d.columns = d.columns.str.strip()
        for c in d:
            d[c] = d[c].str.strip().replace({'':pd.NA,'.':pd.NA})
    if sha256(path) != digest:
        raise ValueError('STOP SHA changed during read')
    d['input_sha256'] = digest
    return d


def dates(values, fmt):
    parsed = pd.to_datetime(values,format=fmt,errors='coerce')
    if (values.notna() & parsed.isna()).any():
        raise ValueError('STOP invalid date')
    return parsed


def number(values):
    try:
        out = pd.to_numeric(values.str.replace(',','',regex=False),errors='raise').astype(float)
    except (ValueError,TypeError) as exc:
        raise ValueError('STOP invalid numeric') from exc
    if np.isinf(out).any():
        raise ValueError('STOP nonfinite numeric')
    return out


def load_export(path, sums, kind):
    cols = MONTHLY_COLUMNS if kind == 'monthly' else IFRS_COLUMNS
    if kind not in ('monthly','ifrs'):
        raise ValueError('STOP unknown export kind')
    d = read_zip(path,sums,cols,exact=True)
    d['stock_id'] = d['證券代碼'].str.split().str[0]
    d['period'] = dates(d['年月'], '%Y%m')
    if d.stock_id.isna().any() or d.period.isna().any() or d.duplicated(['stock_id','period']).any():
        raise ValueError('STOP missing or duplicate business key')
    d['available_date'] = dates(d['營收發布日' if kind=='monthly' else '財報發布日'],
                                '%Y%m%d' if kind=='monthly' else '%Y/%m/%d')
    d['release_status'] = np.where(d.available_date.isna(),'NO_RELEASE_DATE','RELEASE_DATE_PRESENT')
    if kind == 'monthly':
        for c in [c for c in cols if '(千元)' in c]:
            d[c] = number(d[c])
        d['revenue'] = d['單月營收(千元)']*1000
        d['consolidated_preferred_revenue'] = d['以合併為主單月營收(千元)']*1000
        d['period_end'] = d.period + pd.offsets.MonthEnd(0)
        d['version_semantics'] = 'FIRST_ANNOUNCED_REVENUE_PER_TEJ_DOCUMENTATION_NOT_INDEPENDENTLY_VERIFIED'
        d['currency'] = 'TWD_ASSUMED_UNIT_DOCUMENTATION_PENDING'
    else:
        for c in cols[12:]:
            d[c] = number(d[c])
        d['revenue'] = d['營業收入淨額']*1000
        d['eps'] = d['每股盈餘']  # yuan/share; never scale EPS by 1000
        for c in cols[12:]:
            if c != '每股盈餘':
                d[c+'_元'] = d[c]*1000
        d['period_start'] = dates(d['財報年月起日'],'%Y/%m')
        d['period_end'] = dates(d['財報年月迄日'],'%Y/%m') + pd.offsets.MonthEnd(0)
        d['version_semantics'] = 'UNKNOWN'
        d['currency'] = d['幣別']
    d.loc[d.available_date.notna() & d.period_end.isna(),'release_status'] = 'NO_PERIOD_END'
    d.loc[d.available_date < d.period_end,'release_status'] = 'BEFORE_PERIOD_END'
    # Intentionally fail closed for strict source-version PIT until accepted separately.
    d['strict_pit_eligible'] = False
    d['publication_date_usable'] = (d.release_status == 'RELEASE_DATE_PRESENT') & d.period_end.notna()
    d['nominal_period_end'] = d.period + pd.offsets.MonthEnd(0)
    d['before_nominal_period_end'] = d.available_date < d.nominal_period_end
    d['input_row_number'] = np.arange(2,len(d)+2)
    return d


def distribution(s):
    s = s.dropna()
    return {'n':len(s),'min':None if s.empty else float(s.min()),
            'quantiles':{str(q):float(s.quantile(q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]} if len(s) else {},
            'negative':int((s<0).sum()),'zero':int((s==0).sum()),'positive':int((s>0).sum())}


def counts(d, cols):
    return [{**{c:str(k) for c,k in zip(cols,key if isinstance(key,tuple) else (key,))},'rows':int(n)}
            for key,n in d.groupby(cols,dropna=False,sort=True).size().items()]


def timing_audit(m,f):
    f = f.copy()
    f['year'] = f.period.dt.year
    f['lag_days'] = (f.available_date-f.period_end).dt.days
    # Calendar quarter membership determined from start/end, independently of nominal season.
    span = (f.period_end.dt.year-f.period_start.dt.year)*12+f.period_end.dt.month-f.period_start.dt.month+1
    f['non_calendar_period'] = (~((f.period_start.dt.month == 1) &
                                  f.period_end.dt.month.isin([3,6,9,12]) & span.isin([3,6,9,12]))).astype('boolean')
    f.loc[f.period_start.isna() | f.period_end.isna(),'non_calendar_period'] = pd.NA
    f['nominal_lag_days'] = (f.available_date - (f.period + pd.offsets.MonthEnd(0))).dt.days
    f['period_span_months'] = span
    dims = ['季別','單季(Q)/單半年(H)','合併(Y/N)','財報類別（1個別2個體3合併）','year','non_calendar_period','TEJ主產業代碼']
    anomalies = {}
    for name,mask in [('before_end',f.lag_days<0),('before_nominal_end',f.nominal_lag_days<0),('over_365',f.lag_days>365),('no_release',f.available_date.isna())]:
        a = f[mask]
        anomalies[name] = {'rows':len(a),'by_dimension':{c:counts(a,[c]) for c in dims},
                           'joint':counts(a,dims)}
    # Row examples remain PRIVATE: public report contains aggregate counts only.
    private_examples = {name:f.loc[mask,['stock_id','period','period_start','period_end','available_date']].head(30)
                        for name,mask in [('before_end',f.lag_days<0),('before_nominal_end',f.nominal_lag_days<0),('over_365',f.lag_days>365)]}
    m=m.copy()
    m['year']=m.period.dt.year
    m['month']=m.period.dt.month
    m['reference_10th']=m.period+pd.offsets.MonthBegin(1)+pd.Timedelta(days=9)
    m['offset_days']=(m.available_date-m.reference_10th).dt.days
    monthly={'rows':len(m),'companies':m.stock_id.nunique(),'missing_release':int(m.available_date.isna().sum()),
             'before_month_end':int((m.available_date<m.period_end).sum()),'offset_days':distribution(m.offset_days),
             'by_year':{str(y):{'rows':len(g),'missing_release':int(g.available_date.isna().sum()),'offset_days':distribution(g.offset_days)} for y,g in m.groupby('year')},
             'jan_feb_reporting_month_proxy':{str(k):distribution(g.offset_days) for k,g in m.groupby(m.available_date.dt.month.isin([1,2]))},
             'missing_by_year_month_scope':counts(m[m.available_date.isna()],['year','month','合併營收(Y/N)']),
             'missing_features':{'companies':m.loc[m.available_date.isna(),'stock_id'].nunique(),
                'missing_revenue':int(m.loc[m.available_date.isna(),'revenue'].isna().sum()),
                'by_current_industry':counts(m[m.available_date.isna()],['current_industry']) if 'current_industry' in m else [],
                'by_scope':counts(m[m.available_date.isna()],['合併營收(Y/N)'])},
             'by_reference_reporting_month':{str(k):distribution(g.offset_days) for k,g in m.groupby((m.period+pd.offsets.MonthBegin(1)).dt.month)},
             'over_14_reference_days':int((m.offset_days>4).sum()),
             'coverage':counts(m[m.available_date.notna()].drop_duplicates(['year','month','stock_id']),['year','month'])}
    return {'ifrs':{'rows':len(f),'companies':f.stock_id.nunique(),'lag_days':distribution(f.lag_days),'anomalies':anomalies,
                    'nominal_lag_days':distribution(f.nominal_lag_days),
                    'lag_by_quarter':{str(k):distribution(g.lag_days) for k,g in f.groupby('季別')},
                    'nominal_minus_actual_end_days':distribution(((f.period+pd.offsets.MonthEnd(0))-f.period_end).dt.days),
                    'negative_nominal_resolved_by_actual_end':int(((f.nominal_lag_days<0)&(f.lag_days>=0)).sum()),
                    'non_calendar_period_unknown':int(f.non_calendar_period.isna().sum()),
                    'non_calendar_period_rows':int(f.non_calendar_period.sum()),'period_spans':counts(f,['period_span_months'])},'monthly':monthly},private_examples


def reconciliation(tej, raw, value, extra_dims):
    keys=['stock_id','period']
    duplicate_raw=int(raw.duplicated(keys,keep=False).sum())
    raw=raw[~raw.duplicated(keys,keep=False)]
    joined=tej.merge(raw,on=keys,how='inner',validate='one_to_one')
    valid=joined[value].notna() & joined.raw_value.notna()
    j=joined[valid].copy()
    # Reference raw absolute denominator. Both zero -> 0; raw zero/TEJ nonzero -> undefined, separately counted.
    j['relative_difference']=(j[value]-j.raw_value).abs()/j.raw_value.abs().replace(0,np.nan)
    both_zero=(j[value]==0)&(j.raw_value==0)
    j.loc[both_zero,'relative_difference']=0
    different_zero=(j.raw_value==0)&(j[value]!=0)
    bad=j[(j.relative_difference>.01)|different_zero].copy()
    bad['year']=bad.period.dt.year
    ratios=bad[value]/bad.raw_value.replace(0,np.nan)
    bad['unit_ratio_class']='OTHER_OR_UNKNOWN'
    for factor in [1.,1000.,.001]:
        bad.loc[(ratios/factor-1).abs()<=.01,'unit_ratio_class']='NEAR_'+str(factor)
    bad.loc[bad.raw_value==0,'unit_ratio_class']='RAW_ZERO'
    bad.loc[(bad[value]*bad.raw_value)<0,'unit_ratio_class']='SIGN_DIFFERENCE'
    return {'tej_rows':len(tej),'raw_unique_rows':len(raw),'raw_duplicate_key_rows_excluded':duplicate_raw,
            'matched_keys':len(joined),'paired_finite_values':len(j),
            'exact_values':int((j[value]==j.raw_value).sum()),
            'relative_difference_lt_0_1pct':int((j.relative_difference<.001).sum()),
            'relative_difference_le_1pct':int((j.relative_difference<=.01).sum()),'match_rate_tej':len(joined)/len(tej) if len(tej) else None,
            'match_rate_raw':len(joined)/len(raw) if len(raw) else None,'relative_difference':distribution(j.relative_difference),
            'raw_zero_tej_nonzero':int(different_zero.sum()),'over_1pct':len(bad),
            'over_1pct_by_dimension':{c:counts(bad,[c]) for c in ['year','unit_ratio_class']+extra_dims},
            'ratio_tej_over_raw':distribution(j[value]/j.raw_value.replace(0,np.nan))},bad


def universe_coverage(monthly, directory):
    """Only read membership/date arrays, never instantiate SeqData or open X/y."""
    directory=Path(directory)
    identities={}
    for name in ['stock_ids.npy','dates.npy']:
        p=directory/name
        identities[name]={'path':str(p.resolve()),'sha256':sha256(p)}
    ids=np.load(directory/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
    dates_array=np.load(directory/'dates.npy',mmap_mode='r',allow_pickle=False)
    if ids.ndim!=1 or dates_array.ndim!=1 or len(ids)!=len(dates_array):
        raise ValueError('STOP universe identity arrays shape mismatch')
    members={}
    for start in range(0,len(ids),100000):
        frame=pd.DataFrame({'stock_id':np.asarray(ids[start:start+100000]),
                            'month':np.asarray(dates_array[start:start+100000]).astype('datetime64[M]')})
        for month,g in frame.groupby('month'):
            members.setdefault(month,set()).update(g.stock_id.unique())
    rows=[]
    for month,g in monthly.groupby('period'):
        expected=members.get(month)
        released=set(g.loc[g.available_date.notna(),'stock_id'])
        rows.append({'year':month.year,'month':month.month,'tej_released_companies':len(released),
                     'research_matrix_members':None if expected is None else len(expected),
                     'released_intersection':None if expected is None else len(released & expected),
                     'matrix_members_missing_release':None if expected is None else len(expected-released)})
    for name,identity in identities.items():
        if sha256(directory/name)!=identity['sha256']:
            raise ValueError('STOP universe arrays changed during read')
    return {'identity':identities,'definition':'Monthly union of C0+ prepared research matrix date/stock identities; source-period coverage, not as-of availability or lifecycle/tradability proof',
            'rows':rows,'historical_lifecycle_evidence':'UNKNOWN'}


def run(raw_dir, comparison_dir, out_dir, report_json, universe_dir=None):
    raw_dir,comparison_dir,out_dir=map(Path,[raw_dir,comparison_dir,out_dir])
    root=Path(__file__).resolve().parents[1]
    allowed=root.resolve()/'.artifacts/tej'
    if not out_dir.resolve().is_relative_to(allowed) or out_dir.is_symlink():
        raise ValueError('STOP row outputs must remain inside checkout .artifacts/tej')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('STOP output namespace already populated; do not overwrite audit evidence')
    sums=raw_dir/'SHA256SUMS'
    names={'monthly':'monthly_revenue_2013_2026_export20261125.zip','ifrs':'IFRS_單季財報2013_2026_export20261125.zip','basic':'monthly_revenue_full_basicinfo_202511_202610_export20261125.zip'}
    # Verify all inputs before generating any output.
    inputs={k:{'path':str((raw_dir/n).resolve()),'sha256':verify_zip(raw_dir/n,sums)} for k,n in names.items()}
    m=load_export(raw_dir/names['monthly'],sums,'monthly')
    f=load_export(raw_dir/names['ifrs'],sums,'ifrs')
    basic=read_zip(raw_dir/names['basic'],sums,BASIC_COLUMNS)
    basic['stock_id']=basic['證券代碼'].str.split().str[0]
    b=basic.sort_values('年月').drop_duplicates('stock_id',keep='last')
    for d in [m,f]:
        d['current_industry']=d.stock_id.map(b.set_index('stock_id')['TSE產業_名稱'])
        d['current_financial_sector']=d.current_industry.str.contains('金融|保險|銀行|證券').astype('boolean')
        d['industry_evidence']='CURRENT_SNAPSHOT_NOT_HISTORICAL_PIT'
    audit,examples=timing_audit(m,f)
    audit['reconciliation']={}
    comparisons={}
    rpath=comparison_dir/'revenue_raw.parquet'
    revenue_sha=sha256(rpath)
    r=pd.read_parquet(rpath,columns=['stock_id','revenue_year','revenue_month','revenue'])
    r['period']=pd.to_datetime(dict(year=r.revenue_year,month=r.revenue_month,day=1))
    r=r.rename(columns={'revenue':'raw_value'})[['stock_id','period','raw_value']]
    audit['reconciliation']['monthly'],bad_m=reconciliation(m,r,'revenue',['合併營收(Y/N)','current_financial_sector'])
    if sha256(rpath)!=revenue_sha:
        raise ValueError('STOP reference revenue changed during read')
    comparisons['revenue_raw']={'path':str(rpath.resolve()),'sha256':revenue_sha}
    fpath=comparison_dir/'financials_raw.parquet'
    financial_sha=sha256(fpath)
    raw=pd.read_parquet(fpath,columns=['stock_id','Date','type','value'])
    for typ,col in [('Revenue','revenue'),('EPS','eps')]:
        r=raw[raw.type==typ].copy()
        r['period']=pd.to_datetime(r.Date).dt.to_period('M').dt.to_timestamp()
        r=r.rename(columns={'value':'raw_value'})[['stock_id','period','raw_value']]
        audit['reconciliation'][typ],bad=reconciliation(f,r,col,['current_financial_sector','合併(Y/N)','幣別','單季(Q)/單半年(H)'])
        examples['reconciliation_'+typ]=bad[['stock_id','period','available_date']]
    examples['reconciliation_monthly']=bad_m[['stock_id','period','available_date']]
    if sha256(fpath)!=financial_sha:
        raise ValueError('STOP reference financials changed during read')
    comparisons['financials_raw']={'path':str(fpath.resolve()),'sha256':financial_sha}
    audit['universe_coverage']=universe_coverage(m,universe_dir) if universe_dir else {'status':'UNKNOWN','reason':'No isolated membership/date arrays supplied'}
    audit['limitations']=[
        'IFRS 首次／重編版本 UNKNOWN；即使發布日存在也禁止當作已驗 strict PIT。',
        '單位：TEJ 金額千元乘1000、EPS 元/股不縮放；幣別保留，不作匯率換算。',
        '基本資料是現況快照；金融分類僅描述切片，不回填歷史產業身分。',
        '一、二月為春節可能影響的發布月份代理，未驗逐年春節與法定順延，不能稱法律遲報。',
        '未讀實際帳本，因此21筆 REVISED 無法對應；整體差異不能識別首次或重編值。',
        'Universe 僅讀 C0+ prepared 的 dates/stock_ids；研究矩陣月內聯集非歷史 lifecycle，2013 年之前／之後缺支援的分母 UNKNOWN。',
        '差異分類是描述，不能僅靠金額比率判定單位／合併／重編成因。',
        'available_date 為日粒度，未證明盤中可得；同日決策另需保守時點規則。']
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    identity={'source_commit':source_commit,'loader_sha256':sha256(__file__),'sha256sums_sha256':sha256(sums),
              'inputs':inputs,'comparisons':comparisons,'runtime':{'python':platform.python_version(),'pandas':pd.__version__,
              'executable':os.sys.executable,'data_root_env':os.environ.get('MARKETMAMBA_DATA_ROOT'),'checkout_data_realpath':str((root/'Data').resolve())},
              'status':'RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
              'artifact_class':'GENERATED_LICENSED_PRIVATE','split_support':'DATA_ONLY_NO_SPLIT_NO_LABELS','model_identity':None,
              'units':'financial_amounts_yuan_eps_yuan_per_share','strict_pit_eligible':False}
    out_dir.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name,d in [('monthly_revenue',m),('ifrs_quarterly',f)]:
        p=out_dir/(name+'.parquet')
        d.to_parquet(p,index=False)
        outputs[name]={'path':str(p.resolve()),'sha256':sha256(p),'rows':len(d),'bytes':p.stat().st_size}
    # Even dates/code-only examples remain licensed row derivatives and never enter public reports.
    for name,d in examples.items():
        p=out_dir/(name+'_private_examples.parquet')
        d.to_parquet(p,index=False)
        outputs[name]={'path':str(p.resolve()),'sha256':sha256(p),'rows':len(d),'bytes':p.stat().st_size}
    manifest={**identity,'outputs':outputs}
    (out_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    audit['identity']=identity
    audit['outputs']=outputs
    audit['manifest_sha256']=sha256(out_dir/'manifest.json')
    Path(report_json).write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'monthly_rows':len(m),'ifrs_rows':len(f),'manifest_sha256':audit['manifest_sha256'],'report':str(report_json)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--comparison-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--report-json',type=Path,required=True)
    p.add_argument('--universe-dir',type=Path)
    a=p.parse_args()
    run(a.raw_dir,a.comparison_dir,a.out_dir,a.report_json,a.universe_dir)
