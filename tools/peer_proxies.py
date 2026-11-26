"""Offline WP3 peer data replay; no labels, forward returns or IC.

Licensed rows ONLY under .artifacts/peer/. Documented industry effective-date
histories are research classifications, never publication-time or strict PIT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from tools import tej_pit
from tools.forward_wp1 import history, field, tej_date, load_lifecycle, Stop

EVIDENCE = 'DOCUMENTED_EFFECTIVE_DATE_HISTORICAL_RESEARCH_REPLAY'
MIN_PEERS = 3
PRIMARY = 'tej_subindustry'
CHOICE = ('TEJ子產業較細，且2958檔中1934檔無變更紀錄，優於TSE的1622檔；'
          '依分類粒度與已記錄穩定性選擇，不依價格或報酬。無紀錄不證明完整歷史。')


def private_output(path):
    path = Path(path).resolve()
    root = (Path(__file__).resolve().parents[1]/'.artifacts/peer').resolve()
    if path == root or root not in path.parents:
        raise ValueError('STOP private rows require a fresh .artifacts/peer/ namespace')
    if path.exists():
        raise ValueError('STOP peer output already exists')
    return path


def code(value):
    if not value or pd.isna(value):
        return None
    token = str(value).split()[0]
    return token if token.startswith('M') else 'M'+token


def validated_history(row, dimension):
    """Newest-first raw history. Effective dates are never publication dates.

    Industry NEW_VALUE semantics are Human-confirmed against the pinned TEJ
    document (whose descriptions are displaced one HTML row). Market semantics
    have no such authority: this function only audits their candidate direction.
    """
    current_key, first_key = (('TSE產業_代碼','首次掛牌TSE產業') if dimension=='industry'
                              else ('TEJ子產業_代碼','首次掛牌TEJ子產業'))
    normalize=code
    if dimension=='market':
        current_key,first_key='上市別','首次掛牌市場'
        normalize=lambda v:v.split()[0] if v else None
    first,current=normalize(field(row,first_key)),normalize(field(row,current_key))
    if row.get('snapshot_conflict'):
        return None, [], 'SNAPSHOT_CONFLICT'
    try:
        changes=history(row,dimension)
        start=tej_date(row,'首次掛牌日期')
    except (ValueError, Stop):
        return None, [], 'INVALID_HISTORY_DATE'
    if not changes:
        return first, [], 'OK' if first and first==current else 'INITIAL_CURRENT_CONFLICT_OR_MISSING'
    if dimension!='market':
        for event in changes:event['value_direction']='NEW_VALUE'
    if not first or not current or any(not c['effective_date'] or not normalize(c['reported_value']) for c in changes):
        return None, changes, 'INCOMPLETE_HISTORY'
    dates=[c['effective_date'] for c in changes]
    if any(dates[i]<=dates[i+1] for i in range(len(dates)-1)):
        return None, changes, 'CHANGE_DATES_NOT_DESCENDING'
    if start and min(dates)<start:
        return None, changes, 'CHANGE_BEFORE_FIRST_LISTING'
    if normalize(changes[0]['reported_value'])!=current:
        return None, changes, 'LATEST_CURRENT_MISMATCH'
    return first,changes,'OK'


def audit_histories(book, output):
    """Aggregate public diagnostics; licensed per-stock evidence stays private."""
    records=[];summary={}
    for dim in ['industry','tej_subindustry','market']:
        counts=Counter();reasons=Counter()
        for stock,row in book.rows.items():
            first,h,reason=validated_history(row,dim)
            reasons[reason]+=1
            if h:
                normalize=(lambda v:v.split()[0] if v else None) if dim=='market' else code
                current=field(row,{'industry':'TSE產業_代碼','tej_subindustry':'TEJ子產業_代碼','market':'上市別'}[dim])
                dates=[c['effective_date'] for c in h]
                counts['changed_stocks']+=1
                counts['latest_current_mismatches']+=int(normalize(h[0]['reported_value'])!=normalize(current))
                counts['date_order_failures']+=int(any(not d for d in dates) or any(dates[i]<=dates[i+1] for i in range(len(dates)-1)))
            records.append({'stock_id':stock,'dimension':dim,'initial':first,'history':h,'validation':reason})
        n=counts['changed_stocks']
        summary[dim]={**counts,'validation_reasons':dict(reasons),
                      'latest_current_mismatch_fraction':counts['latest_current_mismatches']/n if n else None,
                      'date_order_failure_fraction':counts['date_order_failures']/n if n else None}
    # Pin the document's 6161 example as an oracle, never repair raw snapshots.
    row=book.rows.get('6161');checks=[]
    if row:
        checks=[code(field(row,'首次掛牌TEJ子產業'))=='M23B1',
                tej_date(row,'首次掛牌日期')=='2002-01-02']
        expected=[('2007-07-30','M23B4A'),('2005-08-16','M23B4'),('2005-03-09','M23B5')]
        h=history(row,'tej_subindustry')
        for i,(day,value) in enumerate(expected):
            checks.extend([len(h)>i and h[i]['effective_date']==day,
                           len(h)>i and code(h[i]['reported_value'])==value])
        checks.append(len(h)==len(expected))
        if not all(checks):row['_tej_subindustry_document_conflict']=True
        (output/'document-example-6161.json').write_text(json.dumps(
            {'expected':expected,'actual_row':row,'checks':checks,'known_at':'UNKNOWN'},ensure_ascii=False,indent=2)+'\n')
    summary['document_example_6161']={'present':row is not None,'checks':len(checks),
        'passed':sum(checks),'failed':len(checks)-sum(checks),'policy':'EXCLUDE_ON_DOCUMENT_MISMATCH'}
    # No undocumented global direction inferred from a partially passing sample.
    summary['market']['value_direction']='UNKNOWN'
    summary['market']['usage']='NO_CHANGED_MARKET_HISTORY_USED'
    (output/'history-validation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    return summary


def industry_asof(row, day, dimension):
    first,changes,reason=validated_history(row,dimension)
    if row.get('_'+dimension+'_document_conflict'):
        return None,'DOCUMENT_EXAMPLE_MISMATCH'
    if reason!='OK':
        return None,reason
    result=first
    for c in sorted(changes,key=lambda x:x['effective_date']):
        if c['effective_date']<=str(day):
            result=code(c['reported_value'])
    return result,'KNOWN_AT_UNKNOWN'


def market_asof(row, day):
    """Conservative subset of WP1 lifecycle semantics; terminal day is unknown."""
    day=str(day)
    dates=[tej_date(row,k) for k in ['TSE上市日','OTC上市日'] if field(row,k)]
    start,end=min(dates) if dates else None,tej_date(row,'下市日期')
    if row.get('snapshot_conflict'):
        return False,'LIFECYCLE_UNKNOWN'
    if not start:
        return False,'MARKET_OR_LISTING_UNKNOWN'
    if day<start:
        return False,'NOT_LISTED'
    if end and day>=end:
        return False,'DELISTING_ENDPOINT_UNKNOWN' if day==end else 'DELISTED'
    if history(row,'market'):
        return False,'MARKET_HISTORY_DIRECTION_UNKNOWN'
    known=dates and min(dates)<=day
    return (True,'SNAPSHOT_DERIVED') if known else (False,'MARKET_OR_LISTING_UNKNOWN')


def lagged_returns(closes, valid):
    """At t use only c[t-1]/c[t-2]-1, on complete calendar endpoints."""
    usable=valid & np.isfinite(closes) & closes.gt(0)
    c=closes.where(usable)
    r=c.shift(1)/c.shift(2)-1
    return r.where(np.isfinite(r))


def revenue_events(frame, days):
    f=frame[['stock_id','period','release','revenue']].copy()
    for c in ['period','release']:
        f[c]=pd.to_datetime(f[c],errors='raise')
    if f.duplicated(['stock_id','period']).any():
        raise ValueError('STOP duplicate monthly key')
    if (f.period.isna() | f.stock_id.isna()).any() or not f.period.dt.is_month_start.all():
        raise ValueError('STOP invalid monthly key')
    f['revenue']=pd.to_numeric(f.revenue,errors='raise')
    if np.isinf(f.revenue).any():
        raise ValueError('STOP nonfinite monthly value')
    days=pd.DatetimeIndex(days)
    if days.empty or not days.is_monotonic_increasing or days.has_duplicates:
        raise ValueError('STOP invalid sessions')
    # Date-only publication: strictly later session, no same-day use.
    good=f.release.notna() & (f.release>=f.period+pd.offsets.MonthEnd(0))
    pos=days.searchsorted(f.release.fillna(days[-1]),side='right')
    f['available']=pd.NaT
    ok=good & (pos<len(days))
    f.loc[ok,'available']=days[pos[ok]].to_numpy()
    prior=f[['stock_id','period','revenue','available']].copy()
    prior['period']=prior.period+pd.DateOffset(years=1)
    prior=prior.rename(columns={'revenue':'prior_revenue','available':'prior_available'})
    f=f.merge(prior,on=['stock_id','period'],how='left',validate='one_to_one')
    comparable=(f.period>=pd.Timestamp('2014-01-01')) & f.prior_revenue.gt(0) & f.revenue.ge(0)
    with np.errstate(divide='ignore',invalid='ignore'):
        f['yoy']=(f.revenue/f.prior_revenue-1).where(comparable)
    f['yoy']=f.yoy.where(np.isfinite(f.yoy))
    f['yoy_available']=f[['available','prior_available']].max(axis=1).where(f.available.notna() & f.prior_available.notna())
    return f


def revenue_day(events, day, codes):
    day=pd.Timestamp(day)
    month=day.to_period('M').start_time-pd.DateOffset(months=1)
    f=events[events.period.eq(month)].set_index('stock_id').reindex(codes)
    announced=f.available.notna() & f.available.le(day)
    valid=announced & f.yoy_available.notna() & f.yoy_available.le(day)
    return f.yoy.where(valid),announced


def proxy_day(members, returns, revenue_yoy, announced):
    """Members retain unknown rows; denominators only count qualified members.

    All inputs already refer to the same decision session. No cross-date joins.
    """
    m=members.copy().set_index('stock_id')
    r=returns.reindex(m.index).where(lambda s:np.isfinite(s))
    y=revenue_yoy.reindex(m.index).where(lambda s:np.isfinite(s))
    a=announced.reindex(m.index).fillna(False).astype(bool)
    eligible=m.eligible.astype(bool)
    classified=eligible & m.industry.notna()
    group=m.industry.where(classified)
    rr=r.where(classified)
    count=rr.groupby(group).transform('count').fillna(0).astype(int)-rr.notna().astype(int)
    total=rr.groupby(group).transform('sum')-rr.fillna(0)
    m['return_peer_count']=count
    m['industry_peer_count']=group.groupby(group).transform('size').fillna(0).astype(int)-classified.astype(int)
    m['peer_return']=(total/count.replace(0,np.nan)).where(count>=MIN_PEERS)
    m['return_reason']=np.where(count>=MIN_PEERS,'OK','INSUFFICIENT_VALID_PEERS')
    # Fixed market control, includes target; excludes unresolved market members.
    m['market_return']=r.where(eligible).mean()
    m['market_valid_count']=int(r.where(eligible).notna().sum())
    yy=y.where(classified & a)
    ny=yy.groupby(group).transform('count').fillna(0).astype(int)-yy.notna().astype(int)
    na=a.where(classified,False).astype(int)
    m['announced_peer_count']=na.groupby(group).transform('sum').fillna(0).astype(int)-na
    m['unannounced_peer_count']=m.industry_peer_count-m.announced_peer_count
    # Only unannounced targets are scored: own value cannot enter their median.
    med=yy.groupby(group).transform('median')
    m['revenue_peer_count']=ny
    m['revenue_spillover']=med.where(~a & (ny>=MIN_PEERS))
    m['revenue_reason']=np.where(a,'TARGET_REVENUE_AVAILABLE',np.where(ny>=MIN_PEERS,'OK','INSUFFICIENT_VALID_PEERS'))
    for name in ['return','revenue']:
        m.loc[~classified,name+'_reason']='INDUSTRY_UNKNOWN'
        m.loc[~eligible,name+'_reason']='MARKET_OR_LIFECYCLE_UNKNOWN'
        m.loc[~classified,'peer_return' if name=='return' else 'revenue_spillover']=np.nan
    m['missing_price_peer_count']=m.industry_peer_count-m.return_peer_count
    m['announced_but_invalid_yoy_peer_count']=m.announced_peer_count-m.revenue_peer_count
    m['target_revenue_available']=a
    return m.reset_index()


def dist(series):
    s=pd.Series(series).dropna()
    return {'n':len(s),'min':float(s.min()) if len(s) else None,'median':float(s.median()) if len(s) else None,
            'p10':float(s.quantile(.1)) if len(s) else None,'p90':float(s.quantile(.9)) if len(s) else None,
            'max':float(s.max()) if len(s) else None}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path('/home/frank/projects/MarketMamba'))
    parser.add_argument('--output',type=Path,default=Path('.artifacts/peer/replay-v2-20261127'))
    args=parser.parse_args(argv); start=time.monotonic(); output=private_output(args.output)
    source=args.source_root.resolve(); raw=source/'.artifacts/tej/raw'
    inputs={}
    def identity(path):
        path=Path(path);inputs[str(path.resolve())]={'sha256':tej_pit.sha256(path),'bytes':path.stat().st_size}
    # Supplement is authorized by the separate checksum file; NEVER rewrite originals.
    basic=raw/'月營收盈餘_基本資料補充_2013_2026_export20261126.zip'
    sums=raw/'SHA256SUMS.20261126'; original=raw/'SHA256SUMS'
    doc=raw/'docs/company_basic_field_doc.html';doc_sums=raw/'docs/SHA256SUMS'
    for p in [basic,sums,original,doc,doc_sums]:identity(p)
    if tej_pit.sha256(doc)!='e49283d66edab4470d683dfe134bb04363cede05776fb3321c77d8aab484af92':
        raise ValueError('STOP company basic field document identity mismatch')
    if f'{tej_pit.sha256(doc)}  {doc.name}' not in doc_sums.read_text().splitlines():
        raise ValueError('STOP field document checksum record mismatch')
    book,lifecycle_summary=load_lifecycle(basic,sums)
    monthly_path=raw/'monthly_revenue_2013_2026_export20261125.zip';identity(monthly_path)
    # Read only first single-month value and its date; never last-year/export YoY columns.
    cols=['證券代碼','年月','營收發布日','單月營收(千元)']
    monthly=tej_pit.read_zip(monthly_path,original,cols)
    monthly=pd.DataFrame({'stock_id':monthly['證券代碼'].str.split().str[0],
        'period':tej_pit.dates(monthly['年月'],'%Y%m'),
        'release':tej_pit.dates(monthly['營收發布日'],'%Y%m%d'),
        'revenue':tej_pit.number(monthly['單月營收(千元)'])})
    price_path=source/'.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet'
    frozen_path=source/'.artifacts/p4b/snapshot-manifest-20261125.json'
    for p in [price_path,frozen_path]:identity(p)
    frozen=json.loads(frozen_path.read_text())
    if inputs[str(price_path.resolve())]['sha256']!=frozen['files']['prices_raw.parquet']['sha256']:
        raise ValueError('STOP frozen price identity mismatch')
    # This isolated read imports no evaluator and never opens any label matrix.
    prices=pd.read_parquet(price_path,columns=['Date','stock_id','Open','High','Low','Close','Volume'])
    prices['Date']=pd.to_datetime(prices.Date)
    days=pd.DatetimeIndex(prices.loc[prices.Date.ge('2013-01-01') & prices.Date.le('2026-11-26'),'Date'].unique()).sort_values()
    prices=prices[prices.Date.isin(days)]
    from V6.research_lab.data import validate_prices
    prices=prices.rename(columns={'Date':'date','stock_id':'code',**{c:c.lower() for c in ['Open','High','Low','Close','Volume']}})
    prices['adj_close']=np.nan
    validated=validate_prices(prices);quality=validated.attrs['quality'];del prices
    codes=sorted(book.rows)
    closes=validated.pivot(index='date',columns='code',values='close').reindex(index=days,columns=codes)
    valid=validated.pivot(index='date',columns='code',values='observation_valid').reindex(index=days,columns=codes).eq(True)
    returns=lagged_returns(closes,valid);del closes,valid,validated
    events=revenue_events(monthly,days);del monthly
    output.mkdir(parents=True)
    events.to_parquet(output/'revenue-events.parquet',index=False)
    (output/'lifecycle-history.json').write_text(json.dumps([book.describe(c) for c in codes],ensure_ascii=False,sort_keys=True)+'\n')
    history_audit=audit_histories(book,output)
    # Build piecewise historical classification once, then slice by session.
    base=pd.DataFrame({'stock_id':codes}).set_index('stock_id')
    historical={}
    for dim in ['industry','tej_subindustry']:
        values=np.full((len(days),len(codes)),None,object);reasons=[]
        for j,c in enumerate(codes):
            row=book.rows[c];first,changes,reason=validated_history(row,dim)
            if row.get('_'+dim+'_document_conflict'):reason='DOCUMENT_EXAMPLE_MISMATCH'
            reasons.append('KNOWN_AT_UNKNOWN' if reason=='OK' else reason)
            if reason!='OK':continue
            values[:,j]=first
            for event in reversed(changes):
                values[days>=event['effective_date'],j]=code(event['reported_value'])
        historical[dim]=values;base[dim+'_reason']=reasons
    static={c:book.rows[c] for c in codes}
    # As-of lifecycle mask via stock intervals: reproduce conservative WP1 rules.
    eligible=np.zeros((len(days),len(codes)),bool);candidate=np.zeros_like(eligible)
    market_reasons=np.full((len(days),len(codes)),'MARKET_OR_LISTING_UNKNOWN',object)
    for j,c in enumerate(codes):
        row=static[c]; first=tej_date(row,'首次掛牌日期');end=tej_date(row,'下市日期')
        entries=[tej_date(row,k) for k in ['TSE上市日','OTC上市日'] if field(row,k)]
        if row.get('snapshot_conflict'):continue
        entry=min(entries) if entries else None
        if not entry:continue
        candidate[:,j]=(days>=entry) & ((days<end) if end else True)
        if history(row,'market'):
            market_reasons[:,j]='MARKET_HISTORY_DIRECTION_UNKNOWN'
        else:
            known_entry=bool(entries)
            if known_entry:
                eligible[:,j]=candidate[:,j];market_reasons[candidate[:,j],j]='SNAPSHOT_DERIVED'
    counts={k:Counter() for k in ['return_reason','revenue_reason','return_peer_count','revenue_peer_count']}
    valid_counts=Counter();annual=[];year_rows=[];stats=[];monthly_stats=[];year=None
    first_codes={dim:{c:code(field(static[c],'TSE產業_代碼' if dim=='industry' else 'TEJ子產業_代碼')) for c in codes} for dim in ['industry','tej_subindustry']}
    for i,day in enumerate(days):
        if year is not None and day.year!=year:
            pd.concat(year_rows,ignore_index=True).to_parquet(output/f'peer-daily-{year}.parquet',index=False)
            year_rows=[]
        year=day.year
        m=base.copy();m['eligible']=eligible[i];m['market_reason']=market_reasons[i]
        for dim in ['industry','tej_subindustry']:
            m[dim]=historical[dim][i]
            g=m.loc[m.eligible,dim].dropna();sizes=g.value_counts()
            known=len(g);n=int(m.eligible.sum())
            different=sum(v!=first_codes[dim][c] for c,v in g.items())
            stats.append({'date':day,'classification':dim,'eligible_members':n,'classified_members':known,
                          'unknown_members':n-known,'industries':len(sizes),'singleton_industries':int(sizes.eq(1).sum()),
                          'singleton_fraction':float(sizes.eq(1).mean()) if len(sizes) else None,
                          'different_from_current':different,'different_fraction_known':different/known if known else None,
                          'different_fraction_all_lower':different/n if n else None,
                          'different_fraction_all_upper':(different+n-known)/n if n else None,
                          **{'size_'+k:v for k,v in dist(sizes).items()}})
            # Per-day per-industry sizes remain private.
            for group,size in sizes.items():
                monthly_stats.append({'date':day,'classification':dim,'industry':group,'members':int(size)})
        m=m.rename(columns={'industry':'tse_industry','industry_reason':'tse_industry_reason','tej_subindustry':'industry'})
        y,a=revenue_day(events,day,codes)
        out=proxy_day(m.reset_index(),returns.iloc[i],y,a)
        out=out.loc[candidate[i]].copy()
        out.insert(0,'date',day);out['revenue_month']=(day.to_period('M')-1).start_time
        out['available_at']=day.strftime('%Y-%m-%dT13:30:00+08:00');out['classification_known_at']='UNKNOWN'
        out['strict_pit_eligible']=False;out['evidence_class']=EVIDENCE
        # Separate return/revenue numerical availability from strict source admission.
        out['price_return_session']=days[i-1] if i>=1 else pd.NaT
        for k in counts:counts[k].update(out[k].value_counts().to_dict())
        valid_counts['rows']+=len(out);valid_counts['eligible']+=int(out.eligible.sum())
        for k in ['peer_return','revenue_spillover','market_return']:valid_counts[k]+=int(out[k].notna().sum())
        year_rows.append(out)
        if i%250==0:print(json.dumps({'sessions_done':i+1,'rows':valid_counts['rows']}),flush=True)
    if year_rows:pd.concat(year_rows,ignore_index=True).to_parquet(output/f'peer-daily-{year}.parquet',index=False)
    stats=pd.DataFrame(stats);stats.to_parquet(output/'daily-industry-distribution.parquet',index=False)
    sizes=pd.DataFrame(monthly_stats);sizes.to_parquet(output/'daily-industry-counts.parquet',index=False)
    # Monthly early-announcer proportions, all classified listed peers in denominator.
    aggregates=[]
    for path in sorted(output.glob('peer-daily-*.parquet')):
        rows=pd.read_parquet(path,columns=['date','stock_id','industry','eligible','industry_peer_count','announced_peer_count','target_revenue_available'])
        rows=rows[rows.eligible & rows.industry.notna()]
        for month,g in rows.groupby(rows.date.dt.to_period('M')):
            denom=(g.industry_peer_count).sum();num=g.announced_peer_count.sum()
            aggregates.append({'signal_month':str(month),'member_sessions':len(g),
                'peer_opportunities':int(denom),'announced_peer_opportunities':int(num),
                'early_announcer_peer_fraction':float(num/denom) if denom else None,
                'unannounced_target_member_sessions':int((~g.target_revenue_available).sum())})
    limitations=[
        'STRICT_PIT_STOP：known_at UNKNOWN；啟用日不等於公告日。市場變更方向未通過全體驗證，維持UNKNOWN；下市當日保守排除。',
        '產業依Human確認的變更後新分類語義及啟用日分段重建；最新值／現況、日期或文件範例不一致者留缺。僅研究歷史分類，非完整歷史成員或公告PIT認證。',
        '歷史TEJ首次單月值依原資料說明作受限代理；公司逐期首次版本、合併可比性及實收時間未獨立驗證。',
        '價格為frozen raw價差、非含息；公司行動未完整驗證，沿用OHLCV有效性與>35%異常遮罩，不授交易資格。',
        '日曆為observed price-date union歷史研究代理，非官方前瞻日曆；缺價與未入價格快照者仍保留分母。',
        '全體分母限可重建TSE/OTC分類成員；未明市場／分類數另列，不能冒稱全市場。',
        '2013比較跨2013/01口徑NaN，禁止使用匯出去年／累計營收；分母公告晚到則等兩期都可用。',
        '未讀標籤、未來報酬、blind或IC；作者自查非獨立審查，完成非Human acceptance。']
    classifications={}
    for dim,g in stats.groupby('classification'):
        classifications[dim]={'daily_member_size':dist(sizes.loc[sizes.classification.eq(dim),'members']),
            'daily_singleton_industry_fraction':dist(g.singleton_fraction),
            'classified_member_sessions':int(g.classified_members.sum()),'unknown_member_sessions':int(g.unknown_members.sum()),
            'different_from_current_known':int(g.different_from_current.sum()),
            'different_fraction_known':float(g.different_from_current.sum()/g.classified_members.sum()) if g.classified_members.sum() else None,
            'different_fraction_all_upper':float((g.different_from_current.sum()+g.unknown_members.sum())/g.eligible_members.sum()) if g.eligible_members.sum() else None}
    for path in inputs:
        if tej_pit.sha256(path)!=inputs[path]['sha256']:raise ValueError('STOP input changed during replay')
    outputs={p.name:{'sha256':tej_pit.sha256(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    before=json.loads(Path('reports/peer_proxies_data_20261126.json').read_text())['coverage']
    result={'schema':'peer-proxies-data-v2','history_audit':history_audit,'coverage_before':before,
        'coverage_change':{k:int(valid_counts[k]-before[k]) for k in valid_counts},
        'status':'PARTIAL_REPLAY_COMPLETE_STRICT_PIT_STOP',
        'state':['RESEARCH_ONLY','UNMERGED','NOT_DEPLOYED','NOT_ACCEPTED'],'evidence_class':EVIDENCE,
        'primary_classification':PRIMARY,'classification_choice_reason':CHOICE,'minimum_peers':MIN_PEERS,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'lifecycle':lifecycle_summary,'sessions':len(days),'first_session':str(days.min().date()),'last_session':str(days.max().date()),
        'coverage':dict(valid_counts),'distributions':{k:{str(a):int(b) for a,b in v.items()} for k,v in counts.items()},
        'classifications':classifications,'monthly_early_announcers':aggregates,'price_quality':quality,
        'limitations':limitations,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'executable':os.sys.executable,'data_root_env':os.environ.get('MARKETMAMBA_DATA_ROOT','UNSET'),
        'data_realpath':str(Path('Data').resolve()),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},
        'inputs':inputs,'implementation':{str(p):tej_pit.sha256(p) for p in [Path(__file__),Path('tests/test_peer_proxies.py'),Path('tools/forward_wp1.py'),Path('tools/tej_pit.py'),Path('V6/research_lab/data.py'),Path('tasks/plan-fundamentals-supplychain-integration-20261126.md')]},
        'outputs':outputs,'artifact_class':'GENERATED','model_identity':'N/A','split_support':'N/A: no fitting or labels',
        'elapsed_seconds':round(time.monotonic()-start,3)}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    public={k:v for k,v in result.items() if k not in ['inputs','outputs','implementation','monthly_early_announcers']}
    public['private_manifest_sha256']=tej_pit.sha256(output/'manifest.json')
    public['monthly_early_announcers']=aggregates
    report=Path('reports/peer_proxies_data_v2_20261127.json')
    if report.exists():raise ValueError('STOP report exists')
    report.write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:public[k] for k in ['status','sessions','coverage','elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
