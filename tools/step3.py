"""Frozen Step3 CPU engineering primitives. No evaluation is called by this module.

RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED.
Licensed rows and predictions must stay below this checkout's .artifacts/step3.
"""
from __future__ import annotations
import argparse
import hashlib
from itertools import groupby
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'tasks/spec-step3-monthly-revenue-h20-20261126.md'
SPEC_SHA='9cd7cb8681e3d881193efa3c6fa3c72bd74fba84a96f6a16e6da3a85398eaf34'
FEATURES=['monthly_yoy','monthly_mom','monthly_cumulative_yoy','monthly_high12','monthly_surprise']
SEEDS=(17,29,43)
YEARS=tuple(range(2015,2027))
PARAMS=dict(n_estimators=200,num_leaves=31,learning_rate=.05,min_child_samples=200,
            subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=10.,n_jobs=4,verbosity=-1)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_file(path,digest,size=None):
    p=Path(path)
    if not p.is_file() or (size is not None and p.stat().st_size!=size) or sha256(p)!=digest:
        raise ValueError('STOP file identity: '+str(p))
    return dict(sha256=digest,bytes=p.stat().st_size,realpath=str(p.resolve()))


def private_dir(path):
    p=Path(path).absolute(); root=ROOT/'.artifacts/step3'
    if p==root or not p.is_relative_to(root) or p.resolve()!=p:
        raise ValueError('STOP private output namespace/symlink')
    p.mkdir(parents=True,exist_ok=True)
    return p


def effective_session(release,days):
    return int(days.searchsorted(pd.Timestamp(release),side='right'))


def _known(value):
    return pd.notna(value) and str(value).strip().upper() not in ('','UNKNOWN','NONE','NAN')


def monthly_events(frame,days):
    """Replay verified first-month values; no reported cumulative comparators used.

    comparability_id is an upstream evidence identity, never inferred from two
    missing scope flags. A new semantic version must be admitted independently.
    """
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
    rows=[];missing=[]
    for code,g in f.groupby('stock_id',sort=True):
        history={};known_values={}
        for session,batch in groupby(g.itertuples(index=False),key=lambda r:r.session):
            if session>=len(days):continue
            for r in batch:
                history[r.month]=r
                identity=tuple(getattr(r,k) for k in ['scope','currency','unit','comparability_id'])
                known_values[r.month]=(float(r.revenue) if np.isfinite(r.revenue) else np.nan,
                    identity if all(_known(v) for v in identity) else None)
            month=max(history);identity=known_values[month][1]
            def value(m):
                item=known_values.get(m)
                return item[0] if item is not None and identity is not None and item[1]==identity else np.nan
            rev=value(month)
            def ratio(a,b):return a/b-1 if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan
            yoy=ratio(rev,value(month-12));mom=ratio(rev,value(month-1))
            january=month-month%12
            cur=np.array([value(m) for m in range(january,month+1)])
            prev=np.array([value(m-12) for m in range(january,month+1)])
            cumulative=ratio(cur.sum(),prev.sum())
            past=np.array([value(month-k) for k in range(1,13)])
            high=float(rev>past.max()) if np.isfinite(rev) and np.isfinite(past).all() else np.nan
            growth=[]
            for m in sorted(history,reverse=True):
                if m>=month:continue
                y=ratio(value(m),value(m-12))
                if np.isfinite(y):growth.append(y)
                if len(growth)==3:break
            expected=value(month-12)*(1+np.median(growth)) if len(growth)==3 else np.nan
            surprise=(rev-expected)/abs(expected) if np.isfinite(rev) and np.isfinite(expected) and expected>0 else np.nan
            vals=[yoy,mom,cumulative,high,surprise]
            rows.append(dict(stock_id=code,session=int(session),**dict(zip(FEATURES,vals))))
            codes=['MISSING_OR_INCOMPARABLE_MONTH_OR_NONPOSITIVE_PRIOR_YEAR',
                   'MISSING_OR_INCOMPARABLE_ADJACENT_MONTH_OR_NONPOSITIVE_PRIOR_MONTH',
                   'INCOMPLETE_OR_INCOMPARABLE_YTD_OR_NONPOSITIVE_PRIOR_YTD',
                   'INCOMPLETE_OR_INCOMPARABLE_PREVIOUS_12_MONTHS',
                   'FEWER_THAN_3_VALID_PRIOR_YOY_OR_INVALID_EXPECTATION']
            reasons={name:reason for name,reason,v in zip(FEATURES,codes,vals) if not np.isfinite(v)}
            if reasons:missing.append(dict(stock_id=str(code),session=int(session),reasons=reasons))
    result=pd.DataFrame(rows,columns=['stock_id','session']+FEATURES)
    result.attrs['deduplication_lineage']=lineage
    result.attrs['missing_reasons']=missing
    return result


def boolean_evidence(value):
    result=np.asarray(value)
    if result.dtype.kind!='b':raise ValueError('STOP unknown/nonboolean evidence')
    return result


def h20_labels(opens,closes,valid):
    o,c,v=np.asarray(opens,float),np.asarray(closes,float),boolean_evidence(valid)
    if o.shape!=c.shape or o.shape!=v.shape or o.ndim!=2:raise ValueError('STOP price shape')
    v=v & np.isfinite(o)&np.isfinite(c)&(o>0)&(c>0)
    out=np.full(c.shape,np.nan)
    if len(c)>20:
        bad=np.vstack([np.zeros((1,c.shape[1]),int),np.cumsum(~v,axis=0)])
        complete=(bad[21:]-bad[:-21])==0
        with np.errstate(divide='ignore',invalid='ignore'):
            out[:-20]=np.where(complete,c[20:]/o[1:len(o)-19]-1,np.nan)
    return out


def fold_rows(dates,base_splits,year,mature,calendar=None):
    if year not in YEARS:raise ValueError('STOP outer year')
    d=np.asarray(dates,dtype='datetime64[D]')
    sessions=np.unique(d) if calendar is None else np.asarray(calendar,dtype='datetime64[D]')
    if not len(sessions) or np.isnat(sessions).any() or (sessions[1:]<=sessions[:-1]).any():raise ValueError('STOP frozen calendar')
    t=np.searchsorted(sessions,d)
    if (t>=len(sessions)).any() or not np.array_equal(sessions[t],d):raise ValueError('STOP dates outside calendar')
    evaluation=d.astype('datetime64[Y]')==np.datetime64(str(year),'Y')
    if not evaluation.any():raise ValueError('STOP empty fold')
    o=np.flatnonzero(sessions.astype('datetime64[Y]')==np.datetime64(str(year),'Y'))[0]
    train=(t<o-21)&(np.asarray(base_splits)!=0)&(d>=np.datetime64('2013-01-01'))&boolean_evidence(mature)
    tr=np.flatnonzero(train);ev=np.flatnonzero(evaluation)
    if not len(tr) or not (t[tr]+21<o).all():raise ValueError('STOP split boundary')
    return tr,ev


def sample_training(rows,y,session,seed,cap=500_000):
    rows=np.asarray(rows);y=np.asarray(y);session=np.asarray(session)
    if seed not in SEEDS or not 0<cap<=500_000:raise ValueError('STOP sampling contract')
    if not np.isfinite(y[rows]).all() or (np.diff(rows)<=0).any():raise ValueError('STOP training support')
    target=pd.Series(y[rows]);g=target.groupby(session[rows])
    ranked=(g.rank(method='average',pct=True)-(1+1/g.transform('count'))/2).to_numpy(np.float32)
    pick=np.arange(len(rows)) if len(rows)<=cap else np.sort(np.random.default_rng(seed).choice(len(rows),cap,replace=False))
    return rows[pick],ranked[pick]


def w60_eligible(present):
    p=boolean_evidence(present);c=np.vstack([np.zeros((1,p.shape[1]),np.int64),np.cumsum(p,axis=0)])
    out=np.zeros_like(p)
    if len(p)>=60:out[59:]=(c[60:]-c[:-60])==60
    return out


def validate_arms(keys_a,a,keys_b,b):
    ka,kb=np.asarray(keys_a),np.asarray(keys_b)
    if a.shape!=(len(ka),185) or b.shape!=(len(kb),190) or not np.array_equal(ka,kb) or len(np.unique(ka,axis=0))!=len(ka):
        raise ValueError('STOP arm shape/keys')
    if not np.array_equal(a,b[:,:185],equal_nan=True):raise ValueError('STOP base 185 differ')
    if np.isinf(a).any() or np.isinf(b).any():raise ValueError('STOP infinite features')


def selection_table(days,membership,valid):
    """Accounting only, no returns. Lifecycle reasons remain null without evidence."""
    days=pd.DatetimeIndex(days);m=boolean_evidence(membership);v=boolean_evidence(valid)
    w=w60_eligible(m);complete=np.zeros_like(v);mature=np.zeros(len(days),bool)
    if len(v)>20:
        bad=np.vstack([np.zeros((1,v.shape[1]),int),np.cumsum(~v,axis=0)])
        complete[:-20]=(bad[21:]-bad[:-21])==0;mature[:-20]=True
    rows=[]
    for year in sorted(set(days.year)):
        ym=np.asarray(days.year==year)[:,None];total=int((m&ym).sum())
        reasons={
            'no_w60':int((m&ym&~w).sum()),
            'immature':int((m&ym&w&~mature[:,None]).sum()),
            'price_invalid_cause_unknown':int((m&ym&w&mature[:,None]&~complete).sum()),
            'complete_price_research_support':int((m&ym&w&complete).sum())}
        assert sum(reasons.values())==total
        rows.append(dict(year=int(year),members=total,w60=int((m&ym&w).sum()),scorable_expected=int((m&ym&w).sum()),
            mature=int((m&ym&mature[:,None]).sum()),missing_or_invalid_price=int((m&ym&mature[:,None]&~complete).sum()),
            suspension=None,delisted=None,corporate_action=None,lifecycle_reason_unknown=total,
            primary_reasons=reasons,primary_fractions={k:n/total if total else None for k,n in reasons.items()}))
    return rows


class FieldText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,text):
        if text.strip():self.parts.append(text.strip())


def admission(source_root,out,bundle=None):
    """Hash all frozen raw/PIT files; extract documentation without granting scope."""
    out=private_dir(out);verify_file(SPEC,SPEC_SHA)
    root=Path(source_root)/'.artifacts/tej';identities={};stops=[]
    def check(p,digest,size=None):
        try:identities[str(p)]=verify_file(p,digest,size)
        except (ValueError,OSError) as exc:stops.append(str(exc))
    manifest=root/'pit/manifest.json'
    check(manifest,'f3532527b51ed05ad2b0360aad0c14de0531d18f230cf00c7ef456d79fa254f1')
    # Do not parse a manifest after failed identity validation.
    if str(manifest) in identities:
        mf=json.loads(manifest.read_text())
        for info in mf['inputs'].values():
            p=root/'raw'/Path(info['path']).name
            # Frozen audit supplies size as well as digest (if available).
            check(p,info['sha256'])
        for info in mf['outputs'].values():
            check(root/'pit'/Path(info['path']).name,info['sha256'],info['bytes'])
        check(root/'raw/SHA256SUMS',mf['sha256sums_sha256'])
    docs=root/'raw/docs'; sums=docs/'SHA256SUMS';clauses={}
    if sums.is_file():
        for line in sums.read_text().splitlines():
            digest,name=line.split(maxsplit=1);p=docs/Path(name.lstrip('*')).name;check(p,digest)
            if p.name=='monthly_revenue_field_doc.html' and digest!='2e0e9b5e6897e2eaf7af3565a1344f34cb4b475e1e1da786ee1c58a07dc62dfa':
                stops.append('STOP original monthly documentation identity changed')
                continue
            if str(p) in identities and p.name=='monthly_revenue_field_doc.html':
                parser=FieldText();parser.feed(p.read_text(encoding='utf-8-sig'))
                text='\n'.join(parser.parts)+'\n';target=out/'monthly_field_doc_extracted.txt'
                with target.open('x',encoding='utf-8') as f:f.write(text)
                identities[str(target)]=dict(sha256=sha256(target),bytes=target.stat().st_size)
                clauses={
                    'earliest_disclosure_date':'最早揭露營收相關資訊' in text,
                    'single_month_not_updated':'本項目不會隨之更新' in text,
                    'ifrs_201301_transition':'2013/01' in text and '合併' in text,
                    'calendar_ytd':'1月累計起' in text}
    else:stops.append('STOP missing original field documentation checksums')
    if not clauses or not all(clauses.values()):stops.append('STOP required documentation clauses missing')
    # Coordinator A-D ruling, pending independent review; historical research only.
    bundle_info=None
    if bundle is None:stops.append('STOP identity bundle missing')
    else:
        try:
            bundle_info=dict(path=str(Path(bundle).resolve()),sha256=sha256(bundle),bytes=Path(bundle).stat().st_size)
            candidate=json.loads(Path(bundle).read_text())
            validate_bundle_schema(candidate);verify_source_provenance(candidate)
            for info in candidate['files'].values():verify_file(info['path'],info['sha256'],info['bytes'])
        except (ValueError,OSError,KeyError,TypeError) as exc:stops.append('STOP bundle: '+str(exc))
    result=dict(status='STOP' if stops else 'HUMAN_DECISION_REQUIRED',bundle=bundle_info,strict_forward_status='STOP',
        strict_forward_reasons=['qualified exchange calendar/lifecycle/action evidence unavailable'],
        source_gate='HUMAN_DECISION_REQUIRED',residual_risks=['Company corrections / first-version identity unproven','Publication date means earliest related information, not proven first knowledge of this value','Cross-source agreement does not prove historical PIT or scope'],
        evidence_class='CPU_ENGINEERING_ONLY',contract_sha256=SPEC_SHA,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        identities=identities,documentation_clauses=clauses,stops=stops,fit_count=0,
        implementation={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'tools').glob('step3*.py'))},
        runtime=dict(python=platform.python_version(),executable=os.sys.executable,
                     data_root_env=os.getenv('MARKETMAMBA_DATA_ROOT'),data_realpath=str((ROOT/'Data').resolve())))
    with (out/'admission.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


BUNDLE_CATEGORIES={'matrix','feature_order','calendar','prices','membership','labels','features',
                   'sampling','version_evidence','splits','keys','implementation'}


def human_source_acceptance(path):
    """Explicit Human record; never generated or inferred by the runner."""
    from datetime import date
    try:
        p=Path(path);record=json.loads(p.read_text())
        date.fromisoformat(record['decision_date'])
        if record['decision']!='ACCEPT_HISTORICAL_SOURCE_GATE' or not record['decision_text'].strip():
            raise ValueError('acceptance text/decision')
        return dict(path=str(p.resolve()),sha256=sha256(p),bytes=p.stat().st_size,record=record)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('STOP invalid Human source-gate decision record') from exc


def job_id(year,seed,arm):return f'fold{year}-{arm}-seed{seed}'


def validate_bundle_schema(bundle):
    files=bundle['files'];jobs=bundle['jobs']
    if bundle['source_gate'] not in ('PASS','HUMAN_DECISION_REQUIRED'):raise ValueError('source gate')
    if bundle['contract']!=SPEC_SHA or not BUNDLE_CATEGORIES<=set(files):raise ValueError('incomplete bundle')
    if not bundle['source_commit'] or not bundle['source_identities']:raise ValueError('missing source identities')
    for name,info in bundle['source_identities'].items():
        if name not in files or files[name]!=info:raise ValueError('source identity mismatch')
    if set(jobs)!={job_id(y,s,a) for y in YEARS for s in SEEDS for a in 'AB'}:raise ValueError('72 jobs required')
    for y in YEARS:
        for seed in SEEDS:
            for arm in 'AB':
                job=jobs[job_id(y,seed,arm)]
                if (job['year'],job['seed'],job['arm'])!=(y,seed,arm):raise ValueError('job identity')
                for role in ('sample','target','keys','fit','eval','splits','label_keys','eval_labels'):
                    ref=job[role+'_file']
                    if ref not in files:raise ValueError('missing job file '+role)
                    if role in ('sample','target','keys','fit','eval'):
                        digest=job[role+'_sha256']
                        if not isinstance(digest,str) or len(digest)!=64:raise ValueError('missing array identity')
    required_sources={'matrix','feature_order','calendar','prices','membership','labels','features','version_evidence',
                      'matrix_dates','matrix_ends','stock_index','price_manifest','finmind_revenue','source_manifest'}
    required_sources.update('code_'+p.name for p in (ROOT/'tools').glob('step3*.py') if p.name!='step3.py')
    required_sources.add('implementation')
    if not required_sources<=set(bundle['source_identities']):raise ValueError('incomplete source identities')
    for y in YEARS:
        for seed in SEEDS:
            a,b=(jobs[job_id(y,seed,arm)] for arm in 'AB')
            if any(a[k+'_sha256']!=b[k+'_sha256'] for k in ('sample','target','keys')):raise ValueError('unpaired jobs')


def verify_source_provenance(bundle):
    files=bundle['files']
    verify_file(files['source_manifest']['path'],'f3532527b51ed05ad2b0360aad0c14de0531d18f230cf00c7ef456d79fa254f1')
    manifest=json.loads(Path(files['source_manifest']['path']).read_text())
    digests={v['sha256'] for v in bundle['source_identities'].values()}
    required={v['sha256'] for group in ('inputs','outputs') for v in manifest[group].values()}
    required.add(manifest['sha256sums_sha256'])
    required.add(manifest['comparisons']['revenue_raw']['sha256'])
    required.add('2e0e9b5e6897e2eaf7af3565a1344f34cb4b475e1e1da786ee1c58a07dc62dfa')
    if not required<=digests:raise ValueError('incomplete original source provenance')
    for path in (ROOT/'tools').glob('step3*.py'):
        key='implementation' if path.name=='step3.py' else 'code_'+path.name
        if files[key]['sha256']!=sha256(path):raise ValueError('implementation changed since preparation')


def require_admission(gate,human_accepted_source_gate=None):
    """Reverify schema and bytes; Human acceptance only overrides pending source decision."""
    try:
        if gate['contract_sha256']!=SPEC_SHA:raise ValueError('contract')
        if gate['status']=='HUMAN_DECISION_REQUIRED':
            if human_accepted_source_gate is None:raise ValueError('Human source decision required')
            human_source_acceptance(human_accepted_source_gate)
        elif gate['status']!='PASS':raise ValueError('status')
        info=gate['bundle'];verify_file(info['path'],info['sha256'],info['bytes'])
        bundle=json.loads(Path(info['path']).read_text())
        validate_bundle_schema(bundle)
        if bundle['source_gate']=='HUMAN_DECISION_REQUIRED':
            if human_accepted_source_gate is None:raise ValueError('Human source decision required by bundle')
            human_source_acceptance(human_accepted_source_gate)
        verify_source_provenance(bundle)
        for info in bundle['files'].values():verify_file(info['path'],info['sha256'],info['bytes'])
        verify_file(SPEC,SPEC_SHA)
        return bundle
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('STOP admission/bundle identity: '+str(exc)) from exc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('/home/frank/projects/MarketMamba'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bundle',type=Path)
    a=p.parse_args(argv);r=admission(a.source_root,a.out,a.bundle)
    print(json.dumps({k:r[k] for k in ['status','documentation_clauses','stops','fit_count']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
