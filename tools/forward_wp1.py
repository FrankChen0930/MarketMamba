"""Offline shared WP1 primitives. RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

No research admission is conferred by a development calendar or snapshot-derived
lifecycle. All row outputs must be written under checkout .artifacts/forward/.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, time
import json
from pathlib import Path
import sqlite3
import zipfile
import io
from zoneinfo import ZoneInfo

from tools import pit_ledger, tej_pit

Stop = pit_ledger.Stop
TAIPEI = ZoneInfo('Asia/Taipei')
W34_SHA = 'f7fb12e8aa3a14f4ecd0809408eb189bd14808eb917f8e5a2d4a5bbba8c373a8'


def civil(value):
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise Stop('missing/invalid ISO date') from exc


def timestamp(value):
    try:
        d = datetime.fromisoformat(value)
        if d.tzinfo is None or d.utcoffset() is None:
            raise ValueError('timezone required')
        return d.astimezone(TAIPEI)
    except (TypeError, ValueError) as exc:
        raise Stop('missing/invalid aware timestamp') from exc


def cutoff(session):
    return datetime.combine(date.fromisoformat(civil(session)), time(18), TAIPEI).isoformat()


class Calendar:
    """Pinned positive session list; never infer weekdays or closure reasons."""
    def __init__(self, sessions, coverage_start, coverage_end, identity):
        self.sessions = tuple(civil(s) for s in sessions)
        self.start, self.end = civil(coverage_start), civil(coverage_end)
        if (not self.sessions or self.sessions != tuple(sorted(set(self.sessions))) or
                self.start > self.sessions[0] or self.end < self.sessions[-1]):
            raise Stop('invalid calendar coverage/order/duplicates')
        self.identity = dict(identity)
        self._positions = {s:i for i,s in enumerate(self.sessions)}

    def covered(self, day):
        day = civil(day)
        if not self.start <= day <= self.end:
            raise Stop('calendar coverage UNKNOWN')
        return day

    def require_session(self, day):
        day = self.covered(day)
        if day not in self._positions:
            raise Stop('not a qualified session')
        return day

    def after(self, day, n=1):
        day = self.covered(day)
        if type(n) is not int or n < 1:
            raise Stop('positive session count required')
        i = bisect_right(self.sessions, day) + n - 1
        if i >= len(self.sessions):
            raise Stop('calendar endpoint unavailable')
        return self.sessions[i]

    def shift(self, session, n):
        session = self.require_session(session)
        if type(n) is not int:
            raise Stop('integer shift required')
        i = self._positions[session] + n
        if not 0 <= i < len(self.sessions):
            raise Stop('calendar endpoint unavailable')
        return self.sessions[i]

    def start_after(self, *prerequisite_dates):
        """T0/S0 strict maximum-date exclusion; caller supplies ALL prerequisites."""
        if not prerequisite_dates:
            raise Stop('missing prerequisites')
        return self.after(max(self.covered(d) for d in prerequisite_dates))

    def t1(self, annual_origin, signed_date, accepted_date):
        """First session >= annual origin, strictly after signing/acceptance dates."""
        self.require_session(annual_origin)
        return max(annual_origin, self.start_after(signed_date, accepted_date))

    def origin_minus21(self, origin):
        """Inclusive last train session: train <= origin-21 (also < origin-20)."""
        return self.shift(origin, -21)

    def p2_endpoint(self, t1):
        """T1 signal; first return T1+1, 252nd return T1+252."""
        return self.shift(t1, 252)

    def step3_endpoint(self, s0):
        """252 signals inclusive of S0; final maturity is this endpoint +20."""
        return self.shift(s0, 251)

    def h20_maturity(self, signal):
        return self.shift(signal, 20)

    def require_forward_admission(self):
        if not self.identity.get('forward_admitted', False):
            raise Stop('development calendar is NOT forward-admitted')


def load_w34(path, expected_sha=W34_SHA):
    path = Path(path)
    actual = tej_pit.sha256(path)
    if actual != expected_sha:
        raise Stop('calendar SHA mismatch')
    d = json.loads(path.read_text())
    if d['schema'] != 'w34-development-twse-calendar-v1' or d['research_admitted'] is not False:
        raise Stop('unexpected W34 authority/schema')
    # Official 34-month positive-date qualification is only 2024-01-01..2026-10-22.
    # Exclude older mixed W23/W29 reference sessions, even though present in the file.
    sessions = [r['session'] for r in d['sessions'] if '2024-01-01' <= r['session'] <= '2026-10-22']
    return Calendar(sessions, '2024-01-01', '2026-10-22',
                    {'version':d['schema'], 'path':str(path.resolve()), 'sha256':actual,
                     'source_sha256':d['source_sha256'], 'vintage_id':d['vintage_id'],
                     'sources':d['market_date_refs'], 'forward_admitted':False,
                     'qualification':d['qualification_ref']})


def available_at(calendar, publication_date, observed_at):
    """Date-only publication: strictly later first session CLOSE; observed time too."""
    session = calendar.after(publication_date)
    close = timestamp(session+'T13:30:00+08:00')
    return max(close, timestamp(observed_at)).isoformat()


def source_usable(calendar, decision_session, publication_date, observed_at):
    calendar.require_session(decision_session)
    return timestamp(available_at(calendar, publication_date, observed_at)) <= timestamp(cutoff(decision_session))


def buffered_ready_at(calendar, publication_date, observed_at):
    """P2 A+2 rule, A assigned by Taipei18:00; returns ready decision cutoff.

    Caller compares this against origin-1 cutoff. This does not establish source
    vintage, legal-deadline classification or lifecycle acceptance.
    """
    available = timestamp(available_at(calendar, publication_date, observed_at))
    day = available.date().isoformat()
    calendar.covered(day)
    if day not in calendar._positions or available > timestamp(cutoff(day)):
        day = calendar.after(day)
    return cutoff(calendar.shift(day, 2))


def select_run(runs, as_of):
    end = timestamp(as_of)
    eligible = [(timestamp(info['observed_at']), seq, rid) for seq,rid,info in runs
                if timestamp(info['observed_at']) <= end]
    if not eligible:
        raise Stop('no observed version before cutoff')
    return max(eligible)[2]


def ledger_latest(ledger, as_of, table_name, key):
    """Read-only validated replay for one business key. INITIAL_STOCK != first release.

    Returns None only if a known key is absent in the selected observed snapshot;
    this is missing source evidence, never a zero or an eligible input.
    """
    ledger = Path(ledger).resolve()
    if table_name not in pit_ledger.KEYS or set(key) != set(pit_ledger.KEYS[table_name]):
        raise Stop('invalid table/business key')
    if any(v is None or (isinstance(v,str) and not v.strip()) for v in key.values()):
        raise Stop('missing business key')
    try:
        with sqlite3.connect(f'file:{ledger / "ledger.sqlite3"}?mode=ro', uri=True) as conn:
            head, _, initial = pit_ledger.validate(conn, ledger)
            runs = [(seq,rid,json.loads(payload)) for seq,rid,payload in
                    conn.execute('SELECT seq,run_id,summary_json FROM runs ORDER BY seq')]
            rid = select_run(runs, as_of)
            seq = next(seq for seq,r,info in runs if r == rid)
            rows = list(pit_ledger._replay(conn, ledger, initial, table_name, seq,
                                         {pit_ledger.canonical(key)}))
            if len(rows) > 1:
                raise Stop('duplicate replay key')
            return {'run_id':rid, 'ledger_head':head,
                    'observed_at':next(info['observed_at'] for _,r,info in runs if r==rid),
                    'value':json.loads(rows[0][1]) if rows else None,
                    'evidence_class':'LOCAL_OBSERVATION_NOT_FIRST_PUBLICATION'}
    except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
        raise Stop(f'ledger as-of STOP: {exc}') from exc


def field(row, name):
    v = row.get(name)
    return None if v is None or str(v).strip() in ('', '.', '<NA>') else str(v).strip()


def tej_date(row, name):
    value = field(row, name)
    if value is None:
        return None
    try:
        return datetime.strptime(value,'%Y%m%d').date().isoformat()
    except ValueError as exc:
        raise Stop('invalid lifecycle date') from exc


def history(row, dimension):
    if dimension == 'market':
        pairs = [(f'前{x}次變更市場日期', f'前{x}次變更市場') for x in ['一','二','三']]
    else:
        label = 'TSE產業' if dimension == 'industry' else 'TEJ子產業'
        pairs = [(f'前{x}次{label}變更日', f'前{x}次{label}變更')
                 for x in (['一','二','三'] if dimension=='industry' else ['一','二','三','四'])]
    return [{'effective_date':tej_date(row,d), 'reported_value':field(row,v),
             'source_field':v, 'known_at':'UNKNOWN', 'value_direction':'UNKNOWN'}
            for d,v in pairs if field(row,d) or field(row,v)]


class Lifecycle:
    def __init__(self, rows, snapshot_date='2026-11-26'):
        self.snapshot_date = civil(snapshot_date)
        self.rows = {}
        for row in rows:
            code = row['stock_id']
            if code in self.rows:
                raise Stop('duplicate lifecycle stock')
            self.rows[code] = row

    def describe(self, stock_id):
        """Normalize supplied dates; retain ambiguous change direction explicitly."""
        if stock_id not in self.rows:
            raise Stop('lifecycle missing')
        row = self.rows[stock_id]
        intervals = []
        for x in ['一','二','三']:
            start = tej_date(row,f'全額交割起日({x})')
            end = tej_date(row,f'全額交割迄日({x})')
            if start and end and start > end:
                raise Stop('inverted full-delivery interval')
            if start or end:
                intervals.append({'start':start,'end':end,'end_inclusive':'UNKNOWN',
                                  'known_at':'UNKNOWN'})
        return {'stock_id':stock_id, 'snapshot_date':self.snapshot_date,
                'listing_dates':{k:tej_date(row,k) for k in ['首次掛牌日期','TSE上市日','OTC上市日']},
                'delisting_date':tej_date(row,'下市日期'),
                'delisting_endpoint_convention':'UNKNOWN',
                'first_market':field(row,'首次掛牌市場'),'snapshot_market':field(row,'上市別'),
                'snapshot_industry':field(row,'TSE產業_代碼'),
                'full_delivery_intervals':intervals,
                'changes':{dim:history(row,dim) for dim in ['market','industry','tej_subindustry']},
                'evidence_class':'SNAPSHOT_DERIVED_HISTORY', 'known_at':'UNKNOWN',
                'strict_pit_eligible':False, 'snapshot_conflict':row.get('snapshot_conflict',False)}

    def asof(self, stock_id, day, strict=False):
        day = civil(day)
        if day > self.snapshot_date:
            raise Stop('lifecycle beyond snapshot coverage')
        if stock_id not in self.rows:
            raise Stop('lifecycle missing')
        row = self.rows[stock_id]
        self.describe(stock_id)  # Validate interval/date semantics before answering.
        start, end = tej_date(row,'首次掛牌日期'), tej_date(row,'下市日期')
        market_history, industry_history = history(row,'market'), history(row,'industry')
        market = field(row,'首次掛牌市場') if not market_history and field(row,'首次掛牌市場') == field(row,'上市別') else None
        initial_industry = (field(row,'首次掛牌TSE產業') or '').split()[:1]
        industry = field(row,'TSE產業_代碼') if not industry_history and initial_industry == [field(row,'TSE產業_代碼')] else None
        # First listing may be ROTC/PUB: existence is NOT TSE/OTC market eligibility.
        exists = None if start is None else start <= day
        if end and day >= end:
            exists = None if day == end else False  # source terminal-day convention unverified
        in_market = False if exists is False else (True if exists and market in ('TSE','OTC') else None)
        # Explicit TSE/OTC listing dates establish at least historical venue entry,
        # but do not resolve an ambiguous subsequent transfer/removal.
        listed = [tej_date(row,k) for k in ('TSE上市日','OTC上市日') if field(row,k)]
        if exists and not market_history and listed and min(listed) <= day:
            in_market = True
        full = False
        for x in ['一','二','三']:
            a,b = tej_date(row,f'全額交割起日({x})'),tej_date(row,f'全額交割迄日({x})')
            if b and not a:
                full = None
            if a and (not b or a <= day <= b):
                if day >= a:
                    full = True if b is None or day < b else None
        result = {'stock_id':stock_id,'date':day,'in_market':in_market,'listed_entity_exists':exists,
                  'market':market if exists else None,'industry':industry if exists else None,
                  'full_delivery':full,'strict_pit_eligible':False,'known_at':'UNKNOWN',
                  'evidence_class':'SNAPSHOT_DERIVED_HISTORY',
                  'market_history':market_history,'industry_history':industry_history,
                  'tej_subindustry_history':history(row,'tej_subindustry')}
        if strict:
            raise Stop('snapshot lifecycle known_at UNKNOWN; strict PIT STOP')
        return result


def load_lifecycle(path, sums):
    import pandas as pd
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            columns = io.TextIOWrapper(f,encoding='utf-16').readline().strip().split('\t')
    d = tej_pit.read_zip(path,sums,columns,exact=True)
    d['stock_id'] = d['證券代碼'].str.split().str[0]
    if d.duplicated(['stock_id','年月']).any():
        raise Stop('duplicate stock/month')
    # Preserve conflicting snapshots as UNKNOWN; never choose the latest month silently.
    metadata = [c for c in columns if c not in ('證券代碼','年月','營收發布日')]
    conflicts = d.groupby('stock_id')[metadata].nunique(dropna=False).max(axis=1)
    records = []
    for code,g in d.groupby('stock_id',sort=True):
        if conflicts[code] > 1:
            records.append({'stock_id':code,'snapshot_conflict':True})
        else:
            records.append({k:None if pd.isna(v) else str(v) for k,v in g.iloc[0].items()})
    return Lifecycle(records), {'input_rows':len(d),'stocks':len(records),
                               'conflicting_stocks':int((conflicts>1).sum())}


def audit(source_root, shared, output):
    """Reproducible private derivation; no prices, scores, labels or model reads."""
    import numpy as np
    import platform
    import subprocess
    import time as clock
    started=clock.monotonic()
    root=Path(source_root).resolve()/'.artifacts'
    output=Path(output).resolve()
    allowed=Path(__file__).resolve().parents[1]/'.artifacts/forward'
    if not output.is_relative_to(allowed.resolve()) or output.exists():
        raise Stop('output must be NEW namespace under .artifacts/forward')
    raw=root/'tej/raw/月營收盈餘_基本資料補充_2013_2026_export20261126.zip'
    sums=root/'tej/raw/SHA256SUMS.20261126'
    inputs={}
    def pin(path):
        path=Path(path).resolve()
        inputs[str(path)]={'sha256':tej_pit.sha256(path),'bytes':path.stat().st_size}
    for path in [raw,sums,root/'tej/raw/SHA256SUMS',Path(__file__),
                 Path(__file__).resolve().parents[1]/'tasks/spec-forward-p2-primary-20261124.md',
                 Path(__file__).resolve().parents[1]/'tasks/spec-step3-monthly-revenue-h20-20261126.md']:
        pin(path)
    cal=load_w34(root/'window34-calendar-model/calendar/combined-calendar.json')
    for ref in cal.identity['sources']:
        pin(ref['path'])
        if inputs[str(Path(ref['path']).resolve())]['sha256']!=ref['sha256']:
            raise Stop('official calendar source changed')
    book,summary=load_lifecycle(raw,sums)
    shared=Path(shared)
    for name in ['stock_ids.npy','dates.npy','manifest.json']:
        pin(shared/name)
    ids=np.load(shared/'stock_ids.npy',mmap_mode='r')
    dates=np.load(shared/'dates.npy',mmap_mode='r')
    if len(ids)!=len(dates):raise Stop('membership length mismatch')
    members=set(map(str,np.unique(ids)))
    counts={'member_stocks':len(members),'member_rows':len(ids),
            'covered_stocks':len(members & book.rows.keys()),
            'missing_stocks':len(members-book.rows.keys()),'strict_pit_eligible_stocks':0}
    ledger=root/'pit-ledger-v2'
    pin(ledger/'ledger.sqlite3')
    with sqlite3.connect(f'file:{ledger/"ledger.sqlite3"}?mode=ro',uri=True) as conn:
        head,last,initial=pit_ledger.validate(conn,ledger)
        ledger_summary={'head':head,'runs':conn.execute('SELECT count(*) FROM runs').fetchone()[0],
                        'latest_observed_at':last['observed_at'],
                        'initial_observed_at':initial['observed_at']}
    # No output until all input identities/schema/ledger validation pass.
    output.mkdir(parents=True)
    lifecycle_path=output/'lifecycle-snapshot.json'
    lifecycle_path.write_text(json.dumps(list(book.rows.values()),ensure_ascii=False,sort_keys=True)+'\n')
    history_path=output/'lifecycle-history.json'
    history_path.write_text(json.dumps([book.describe(c) for c in sorted(book.rows)],ensure_ascii=False,sort_keys=True)+'\n')
    calpath=output/'calendar-identity.json'
    calpath.write_text(json.dumps({**cal.identity,'coverage':[cal.start,cal.end],
                                  'sessions':list(cal.sessions)},ensure_ascii=False,sort_keys=True)+'\n')
    result={'schema':'forward-wp1-audit-v1','evidence_class':'CPU_ENGINEERING_ONLY',
            'status':'STOP_FORWARD_ADMISSION','source_commit':subprocess.check_output(
                ['git','rev-parse','HEAD'],text=True).strip(),
            'environment':{'python':platform.python_version(),'runtime':'WSL_CPU',
                           'data_root_env':__import__('os').environ.get('MARKETMAMBA_DATA_ROOT'),
                           'data_realpath':str(Path('Data').resolve())},
            'inputs':inputs,'calendar':{'sha256':W34_SHA,'sessions':len(cal.sessions),
                                     'coverage':[cal.start,cal.end],'forward_admitted':False},
            'lifecycle':summary,'coverage':counts,'ledger':ledger_summary,
            'model_identity':None,'split_support':'membership only; no labels/scores/performance',
            'outputs':{p.name:{'sha256':tej_pit.sha256(p),'bytes':p.stat().st_size}
                       for p in [lifecycle_path,history_path,calpath]},
            'elapsed_seconds':clock.monotonic()-started}
    # Detect concurrent changes to all pinned inputs before claiming a stable audit.
    if any(tej_pit.sha256(p)!=v['sha256'] for p,v in inputs.items()):
        raise Stop('input changed during audit; outputs incomplete')
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','calendar','lifecycle','coverage','ledger','elapsed_seconds']},ensure_ascii=False))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--shared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source_root,args.shared,args.output)
