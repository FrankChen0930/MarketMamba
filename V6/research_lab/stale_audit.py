"""Read-only raw-cache diagnostic; UNMERGED / NOT_DEPLOYED / RESEARCH_ONLY.

Run with the research Python: python -m V6.research_lab.stale_audit --fetch-public
No production imports, credentials, dotenv, apply operation, or data writes.
Existing hygiene checks do not compare historical cross-sectional fingerprints;
this isolated audit preserves their semantics. tests/ is the user-specified new
location. Generated evidence stays in the existing .local-data namespace.

Contract: exact equality of ALL value columns on common stock IDs, paired nulls
count as equal but are separately disclosed. Overlap denominator is max of both
cross-section sizes (>=80% of BOTH). Candidate >=99% is diagnostic, not proof of
staleness. Segments include the initial reference trading day; stale_days excludes
it. Missing sessions break chains. Nontrading observations are compared with the
immediately preceding prices session, never used as trading-day predecessors.
"""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORMAL = Path('/mnt/d/Desktop/work/ProjectForMe/MarketMamba')
DATES = ('2026-10-12', '2026-10-13', '2026-10-14')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to((ROOT / '.local-data').resolve()):
        raise ValueError('outputs must stay under repository .local-data')
    return resolved


def write_json(path, obj):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def normalize(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df.Date, errors='raise').dt.strftime('%Y-%m-%d')
    if df.Date.isna().any() or df.stock_id.isna().any():
        raise ValueError('null date/stock_id')
    df['stock_id'] = df.stock_id.astype(str)
    if df.duplicated(['Date', 'stock_id']).any():
        raise ValueError('duplicate Date/stock_id keys')
    if set(df.columns) == {'Date', 'stock_id'}:
        raise ValueError('no value columns')
    return df


def compare(previous, current):
    common = previous.index.intersection(current.index)
    left, right = previous.loc[common], current.loc[common]
    equal_rows = int((left.eq(right) | (left.isna() & right.isna())).all(axis=1).sum())
    overlap = len(common)
    equal_ratio = equal_rows / overlap if overlap else None
    coverage = overlap / max(len(previous), len(current)) if max(len(previous), len(current)) else 0.
    per_field = {c: int((left[c].eq(right[c]) | (left[c].isna() & right[c].isna())).sum())
                 for c in previous.columns}
    return {'overlap': overlap, 'previous_rows': len(previous), 'current_rows': len(current),
            'overlap_ratio': coverage, 'equal_rows': equal_rows, 'equal_ratio': equal_ratio,
            'equal_rows_by_field': per_field,
            'fully_observed_overlap': int((left.notna().all(axis=1) & right.notna().all(axis=1)).sum()),
            'stale_candidate': bool(overlap and equal_ratio >= 0.99 and coverage >= 0.8)}


def audit_frame(df, trading_dates):
    df = normalize(df)
    daily = {d: g.drop(columns='Date').set_index('stock_id').sort_index()
             for d, g in df.groupby('Date', sort=True)}
    sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))
    if not sessions or not daily:
        raise ValueError('empty input/calendar')
    # Limit missing-day claims to this dataset's observed range; disclose tails separately.
    sessions = [d for d in sessions if min(daily) <= d <= max(daily)]
    empty = df.iloc[:0].drop(columns='Date').set_index('stock_id')
    comparisons, segments = [], []
    active = None
    for previous, current in zip(sessions, sessions[1:]):
        c = {'previous': previous, 'date': current,
             **compare(daily.get(previous, empty), daily.get(current, empty))}
        comparisons.append(c)
        if c['stale_candidate']:
            if active is None:
                active = {'start': previous, 'end': current, 'stale_dates': []}
                segments.append(active)
            active['end'] = current
            active['stale_dates'].append(current)
        else:
            active = None
    session_set = set(pd.to_datetime(trading_dates).strftime('%Y-%m-%d'))
    nontrading = []
    all_sessions = sorted(session_set)
    for d in daily:
        if d not in session_set:
            i = bisect.bisect_left(all_sessions, d)
            previous = all_sessions[i - 1] if i else None
            nontrading.append({'date': d, 'previous': previous,
                               **compare(daily.get(previous, empty), daily[d])})
    for s in segments:
        dates = [d for d in daily if s['start'] <= d <= s['end']]
        s.update(trading_days=len(s['stale_dates']) + 1, stale_days=len(s['stale_dates']),
                 rows=sum(len(daily[d]) for d in dates),
                 stale_trading_rows=sum(len(daily[d]) for d in s['stale_dates']),
                 contains_nontrading_rows=any(d not in session_set for d in dates),
                 nontrading_rows=sum(len(daily[d]) for d in dates if d not in session_set))
    # Supplement session-based findings with observed-date chains: this exposes
    # holiday rows equal to a LATER session (e.g. Oct 9 -> Oct 12), not just prior.
    observed_segments, active = [], None
    for previous, current in zip(sorted(daily), sorted(daily)[1:]):
        c = compare(daily[previous], daily[current])
        if c['stale_candidate']:
            if active is None:
                active = {'start': previous, 'end': current, 'dates': [previous]}
                observed_segments.append(active)
            active['end'] = current
            active['dates'].append(current)
        else:
            active = None
    for s in observed_segments:
        s.update(days=len(s['dates']), rows=sum(len(daily[d]) for d in s['dates']),
                 contains_nontrading_rows=any(d not in session_set for d in s['dates']),
                 nontrading_dates=[d for d in s['dates'] if d not in session_set])
    return {'range': [min(daily), max(daily)], 'rows': len(df),
            'daily_counts': {d: len(v) for d, v in daily.items()},
            'missing_trading_days': [d for d in all_sessions if d not in daily],
            'missing_within_observed_range': [d for d in sessions if d not in daily],
            'segments': segments, 'comparisons': comparisons,
            'observed_date_segments': observed_segments,
            'nontrading_comparisons': nontrading,
            'nontrading_rows': sum(len(v) for d, v in daily.items() if d not in session_set)}


def parse_public(payload, source, day):
    """Date/schema gate. Refuse latest-only payloads; never relabel their date."""
    table = payload['tables'][0] if source.startswith('tpex') else payload
    actual = table.get('date', payload.get('date', ''))
    roc = f'{int(day[:4])-1911}/{day[5:7]}/{day[8:10]}'
    if actual not in (day.replace('-', ''), day, roc):
        raise ValueError(f'wrong/unknown response date: {actual!r}, requested {day}')
    fields = table.get('fields', [])
    rows = table.get('data', [])
    if not rows:
        raise ValueError('empty public response')
    sid_name = ('代號' if source == 'twse_sbl_balance' else
                '證券代號' if '證券代號' in fields else '股票代號')
    mapping = {'stock_id': sid_name}
    if source.endswith('per'):
        mapping.update(PER='本益比', PBR='股價淨值比', DY='殖利率(%)')
    elif source == 'tpex_sbl':
        # This is NEXT-session quota, NOT current Securities_Lending semantics.
        mapping['next_session_short_sale_quota'] = '次一營業日可借券賣出限額'
    elif source == 'twse_sbl_balance':
        mapping['next_session_short_sale_quota'] = '次一營業日可限額'
    else:
        raise ValueError('historical semantic mapping not established for ' + source)
    for name in mapping.values():
        if fields.count(name) != 1:
            raise ValueError('missing/ambiguous public field ' + name)
    df = pd.DataFrame({k: [row[fields.index(v)] for row in rows] for k, v in mapping.items()})
    df.stock_id = df.stock_id.astype(str).str.strip()
    df = df[df.stock_id.str.fullmatch(r'\d{4}')].copy()
    for c in df.columns.difference(['stock_id']):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    df['Date'] = day
    return normalize(df)


def repair_counts(local, public):
    """Exact-key counts only; call only for a validated SAME-semantic source."""
    local, public = normalize(local), normalize(public)
    cols = list(public.columns.difference(['Date', 'stock_id']))
    old = local.set_index(['Date', 'stock_id'])[cols]
    new = public.set_index(['Date', 'stock_id'])[cols]
    common = old.index.intersection(new.index)
    equal = (old.loc[common].eq(new.loc[common]) |
             (old.loc[common].isna() & new.loc[common].isna())).all(axis=1)
    return {'insert': len(new.index.difference(old.index)), 'overwrite': int((~equal).sum()),
            'unchanged': int(equal.sum()), 'local_only_unresolved': len(old.index.difference(new.index))}


def public_urls(day):
    compact, slash = day.replace('-', ''), day.replace('-', '/')
    return {
        'twse_per': f'https://www.twse.com.tw/exchangeReport/BWIBBU_d?date={compact}&selectType=ALL&response=json',
        'twse_per_all': f'https://www.twse.com.tw/exchangeReport/BWIBBU_ALL?date={compact}&response=json',
        'twse_sbl': f'https://www.twse.com.tw/SBL/TWT96U?date={compact}&response=json',
        'twse_sbl_balance': f'https://www.twse.com.tw/exchangeReport/TWT93U?date={compact}&selectType=ALL&response=json',
        'tpex_per': f'https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate?date={slash}&id=&response=json',
        'tpex_sbl': f'https://www.tpex.org.tw/www/zh-tw/margin/sbl?date={slash}&id=&response=json',
    }


def collect_public(directory):
    import requests  # No key, dotenv, provider clients, or environment credential lookup.
    records = []
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for day in DATES:
        for source, url in public_urls(day).items():
            record = {'source': source, 'requested_date': day, 'url': url,
                      'observed_at': datetime.now(timezone.utc).isoformat()}
            try:
                response = requests.get(url, timeout=30)
                path = directory / f'{source}_{day}.body'
                with path.open('xb') as f:
                    f.write(response.content)
                record.update(path=str(path), sha256=sha256(path), bytes=len(response.content),
                              http_status=response.status_code)
            except requests.RequestException as e:
                record['transport_error'] = str(e)
            records.append(record)
            write_json(directory / f'{source}_{day}.meta.json', record)
    write_json(directory / 'index.json', records)
    return records


def public_analysis(records, datasets):
    parsed, observations, repairs = {}, [], []
    for record in records:
        r = dict(record)
        try:
            if r.get('http_status') != 200 or sha256(r['path']) != r['sha256']:
                raise ValueError('HTTP/integrity failure')
            payload = json.loads(Path(r['path']).read_bytes())
            table = payload.get('tables', [payload])[0]
            r.update(response_date=table.get('date'), response_title=table.get('title'),
                     fields=table.get('fields'), raw_rows=len(table.get('data', [])))
            frame = parse_public(payload, r['source'], r['requested_date'])
            parsed.setdefault(r['source'], []).append(frame)
            r.update(validation='DATE_SCHEMA_VALIDATED', parsed_rows=len(frame))
            if r['source'].endswith('per'):
                counts = repair_counts(datasets['per_raw'][datasets['per_raw'].Date == r['requested_date']], frame)
                repairs.append({'date': r['requested_date'], 'source': r['source'],
                                'scope': 'existing_TWSE_values' if r['source'] == 'twse_per'
                                else 'additional_TPEx_coverage_not_stale_repair', **counts})
        except (ValueError, KeyError, IndexError, TypeError) as e:
            r.update(validation='UNRESOLVED', reason=str(e))
        observations.append(r)
    comparisons = {}
    for source, frames in parsed.items():
        combined = pd.concat(frames, ignore_index=True)
        comparisons[source] = audit_frame(combined, DATES)
    return {'observations': observations, 'historical_comparisons': comparisons,
            'per_repair_rows_exact_public_keys': repairs,
            'securities_repair_rows': None,
            'securities_limit': 'TWT96U ignores historical date and exposes available-to-sell shares; '
            'TWT93U is balance/short-sale flow, TPEx margin/sbl is next-session quota. '
            'No verified same-date same-semantic historical replacement for Securities_Lending; '
            'do not infer repair counts or copy these different fields.',
            'daily_change_claim': 'PER uses price and earnings; unchanged price/rounding/NA can repeat. '
            'Neither PER nor lending figures MUST change every day. Cross-sectional equality alone is not proof.'}


def source_review(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    patterns = ['def fetch_per_twse_direct', 'def fetch_securities_lending_twse_direct',
                'def _catch_up_generic', 'date_compact =', 'BWIBBU_ALL', 'SBL/TWT96U',
                'start = pd.to_datetime(max(existing))', 'if d.dayofweek < 5',
                '"Date":              date_str', '"Date": date_str',
                '("per_raw",', '("securities_raw",']
    downstream = path.parent / 'feature_engineer.py'
    downstream_lines = downstream.read_text(encoding='utf-8').splitlines()
    return {'path': str(path.resolve()), 'sha256': sha256(path),
            'downstream_ffill': {'path': str(downstream.resolve()), 'sha256': sha256(downstream),
                'locations': [{'line': i, 'text': line.strip()} for i, line in enumerate(downstream_lines, 1)
                              if ('def _merge_per_pbr' in line or 'def _merge_securities' in line
                                  or ('ffill' in line and (400 <= i <= 435 or 1050 <= i <= 1090)))]},
            'locations': [{'line': i, 'text': line.strip()} for i, line in enumerate(lines, 1)
                          if any(p in line for p in patterns)],
            'verified_code': ['Both fetchers assign requested date without validating returned date/title.',
                              '_catch_up_generic walks weekdays, not prices sessions; starts after max existing date, '
                              'does not repair existing wrong-date rows or older holes.',
                              'No ffill or previous-day-copy in these fetcher function bodies; '
                              'schema padding uses NA, not copied values.'],
            'hypothesis': 'Repeated latest-only responses relabeled as requested dates can explain '
            'holiday copies and multi-session flat batches. Specific historical runs NOT VERIFIED '
            'without contemporaneous raw responses/logs. Downstream feature ffill cannot by itself '
            'explain already-present duplicate-valued raw rows.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=FORMAL / 'Data/processed_v6')
    parser.add_argument('--fetcher', type=Path, default=FORMAL / 'V6/marketmamba/data/fetcher.py')
    parser.add_argument('--out', type=Path, default=ROOT / '.local-data/stale_audit.json')
    parser.add_argument('--public-dir', type=Path, default=ROOT / '.local-data/public_fill/stale_audit')
    parser.add_argument('--fetch-public', action='store_true')
    args = parser.parse_args()
    out = output_path(args.out)
    if out.exists():
        raise FileExistsError(out)
    paths = {n: (args.data_root / (n + '.parquet')).resolve()
             for n in ('prices_raw', 'per_raw', 'securities_raw')}
    inputs = {n: {'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for n, p in paths.items()}
    prices = pd.read_parquet(paths['prices_raw'], columns=['Date'])
    calendar = sorted(pd.to_datetime(prices.Date).dt.strftime('%Y-%m-%d').unique())
    del prices
    datasets = {n: normalize(pd.read_parquet(paths[n])) for n in ('per_raw', 'securities_raw')}
    report = {'schema': 'stale-audit-v1', 'classification': 'GENERATED',
              'evidence_class': 'RESEARCH_ONLY_READ_ONLY_DIAGNOSTIC',
              'status': 'UNMERGED / NOT_DEPLOYED', 'observed_at': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                                (Path(__file__).resolve(), ROOT / 'tests/test_stale_audit.py')},
              'runtime': {'python': sys.version, 'executable': sys.executable, 'pandas': pd.__version__,
                          'platform': platform.platform(), 'MARKETMAMBA_DATA_ROOT': os.getenv('MARKETMAMBA_DATA_ROOT')},
              'inputs': inputs, 'calendar': {'source': 'prices_raw Date union, user-specified proxy, not official calendar',
                                           'range': [calendar[0], calendar[-1]], 'sessions': len(calendar)},
              'contract': {'equal_ratio': .99, 'overlap_ratio': .8, 'overlap_denominator': 'max(previous_rows,current_rows)',
                           'equality': 'all value fields exact, paired nulls equal; observed support separately reported',
                           'segments': 'initial reference + consecutive candidate sessions; missing sessions break',
                           'scope': 'full available history; missing dates include outside dataset observed range'},
              'datasets': {n: audit_frame(df, calendar) for n, df in datasets.items()},
              'fetcher_review': source_review(args.fetcher)}
    if args.fetch_public:
        records = collect_public(args.public_dir)
    else:
        records = json.loads((args.public_dir / 'index.json').read_text())
    report['public'] = public_analysis(records, datasets)
    for n, path in paths.items():
        if sha256(path) != inputs[n]['sha256']:
            raise RuntimeError('input changed during audit: ' + str(path))
    report['inputs_reverified_unchanged'] = True
    write_json(out, report)
    print(json.dumps({'out': str(out), 'sha256': sha256(out),
                      'datasets': {n: {'rows': len(df), 'segments': len(report['datasets'][n]['segments'])}
                                   for n, df in datasets.items()}}, indent=2))


if __name__ == '__main__':
    main()
