"""Offline financial long-table audit; E1 / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Date is period-end, never publication/availability time. No features, alias
selection, PIT qualification, financial-sector deadlines, or admission decisions.
Coverage is presence of any row, not completeness of the three future ratios.
The standalone audit belongs in research_lab; existing feature pipelines do not
preserve unresolved aliases or provide this read-only evidence contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ['Date', 'stock_id', 'type', 'value', 'origin_name']
KEY = ['Date', 'stock_id', 'type']
# Deliberately broad hypotheses, including qualified/component line items.
# A match is evidence for human inspection, never an adopted synonym.
KEYWORDS = {
    'revenue': ['revenue', '營收', '營業收入'],
    'gross_profit': ['grossprofit', '營業毛利', '毛利'],
    'operating_income': ['operatingincome', 'operatingprofit', '營業利益', '營業淨利'],
    'net_income': ['incomeaftertax', 'netincome', '稅後淨利', '稅後純益', '本期淨利',
                   '淨利淨損歸屬', '淨利損歸屬', '本期純益', 'incomefromcontinuingoperations'],
    'eps': ['eps', 'earningspershare', '每股盈餘', '每股稅後盈餘'],
}


def _frame(df):
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    d = df[COLUMNS].copy()
    d['Date'] = pd.to_datetime(d['Date'], errors='raise')
    d['stock_id'] = d['stock_id'].astype('string')
    return d


def _date(value):
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _names(values):
    return sorted({str(x) for x in values if pd.notna(x)})


def audit_types(df):
    """All type rows, descending count then lexical type; missing_ratio is value NA."""
    d = _frame(df)
    result = []
    for name, g in d.groupby('type', dropna=False, observed=True, sort=False):
        result.append(dict(type=None if pd.isna(name) else str(name), row_count=len(g),
                           stock_count=int(g.stock_id.nunique()), period_count=int(g.Date.nunique()),
                           first_period=_date(g.Date.min()), last_period=_date(g.Date.max()),
                           missing_ratio=float(g.value.isna().mean()), origin_names=_names(g.origin_name)))
    return sorted(result, key=lambda r: (-r['row_count'], r['type'] or ''))


def _normalize(text):
    return re.sub(r'[\W_]+', '', text.casefold())


def candidate_types(df):
    """All broad name matches, with exact matched source strings and coverage."""
    result = {category: [] for category in KEYWORDS}
    for row in audit_types(df):
        texts = ([row['type']] if row['type'] is not None else []) + row['origin_names']
        for category, words in KEYWORDS.items():
            matched = sorted({t for t in texts if any(w in _normalize(t) for w in words)})
            if matched:
                result[category].append({**row, 'matched_names': matched})
    return result


def _four_digit_ids(values):
    s = pd.Series(values, dtype='string')
    return set(s[s.str.fullmatch(r'[0-9]{4}', na=False)].tolist())


def _deadline(period):
    parts = {(3, 31): (period.year, 5, 15), (6, 30): (period.year, 8, 14),
             (9, 30): (period.year, 11, 14), (12, 31): (period.year + 1, 3, 31)}
    values = parts.get((period.month, period.day))
    return pd.Timestamp(year=values[0], month=values[1], day=values[2]).date() if values else None


def period_coverage(df, universe_ids=None, *, as_of=None):
    """Quarter grid through latest overdue quarter; 100% of explicit denominator required.

    Without a supplied universe the all-period observed union is a diagnostic
    proxy, with survivorship/IPO/delisting bias, not an expected PIT population.
    Empty denominator -> unknown (null), never zero-risk. General deadlines are
    the task-specified rule, not independently verified legal/PIT evidence.
    """
    d = _frame(df)
    valid = d['stock_id'].str.fullmatch(r'[0-9]{4}', na=False)
    eligible = d.loc[valid]
    expected = _four_digit_ids(universe_ids) if universe_ids is not None else set(eligible.stock_id)
    today = pd.Timestamp(as_of if as_of is not None else datetime.now(ZoneInfo('Asia/Taipei')).date()).date()
    dates = d.Date.dropna()
    if dates.empty:
        return []
    latest = pd.Period(today, freq='Q') - 1
    if today <= _deadline(latest.end_time):
        latest -= 1
    end = max(dates.max(), latest.end_time.normalize())
    periods = set(pd.period_range(dates.min(), end, freq='Q').to_timestamp(how='end').normalize())
    # Preserve nonstandard period ends for inspection, without inventing deadlines.
    periods.update(dates.unique())
    groups = {pd.Timestamp(k): set(g.stock_id) for k, g in eligible.groupby('Date', observed=True)}
    result = []
    for period in sorted(pd.Timestamp(p) for p in periods):
        ids = groups.get(period, set())
        deadline = _deadline(period)
        count = len(ids & expected)
        ratio = count / len(expected) if expected else None
        overdue = today > deadline if deadline else None
        result.append(dict(period_end=_date(period), stock_count=len(ids),
                           expected_stock_count=len(expected),
                           denominator_basis='supplied_universe' if universe_ids is not None else 'observed_union_non_PIT',
                           universe_stock_count=count if universe_ids is not None else None,
                           universe_coverage_ratio=ratio if universe_ids is not None else None,
                           coverage_ratio=ratio, deadline=deadline.isoformat() if deadline else None,
                           deadline_passed=overdue,
                           overdue_insufficient=(overdue and count < len(expected))
                           if expected and overdue is not None else None))
    return result


def duplicate_and_sign_checks(df):
    """Compare every candidate-type pair; exclude ALL duplicate keys from pairs.

    No summing, first/last selection, aggregation across aliases or units inferred.
    Zero has sign 0; finite zero/nonzero pairs count as sign disagreements.
    Nonpositive revenue counts are raw rows (including duplicates); pair counts
    are unique Date/stock_id matches. No pair support -> ratio null.
    """
    d = _frame(df)
    candidates = candidate_types(d)
    duplicated = d.duplicated(KEY, keep=False)
    counts = d.groupby(KEY, dropna=False, observed=True).size()
    numeric = pd.to_numeric(d.value, errors='coerce')
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    usable = (~duplicated) & finite & d[KEY].notna().all(axis=1)
    clean = d.loc[usable, KEY].copy()
    clean['value'] = numeric.loc[usable].astype(float)
    by_type = {k: g[['Date', 'stock_id', 'value']] for k, g in clean.groupby('type', observed=True)}

    def pairs(left, right, sign=False):
        out = []
        for a in sorted(r['type'] for r in candidates[left] if r['type'] is not None):
            for b in sorted(r['type'] for r in candidates[right] if r['type'] is not None):
                if a == b:
                    continue
                x = by_type.get(a, clean.iloc[:0][['Date', 'stock_id', 'value']])
                y = by_type.get(b, clean.iloc[:0][['Date', 'stock_id', 'value']])
                joined = x.merge(y, on=['Date', 'stock_id'], suffixes=('_left', '_right'), validate='one_to_one')
                flag = (np.sign(joined.value_left) != np.sign(joined.value_right)) if sign else (joined.value_left > joined.value_right)
                n, bad = len(joined), int(flag.sum())
                item = dict(left_type=a, right_type=b, compared_count=n, anomaly_count=bad)
                if sign:
                    item['mismatch_ratio'] = bad / n if n else None
                out.append(item)
        return out

    revenues = d.type.isin([r['type'] for r in candidates['revenue']])
    return dict(duplicate_extra_rows=int(d.duplicated(KEY).sum()),
                duplicate_involved_rows=int(duplicated.sum()), duplicate_key_count=int((counts > 1).sum()),
                invalid_numeric_rows=int((~finite).sum()),
                pair_excluded_duplicate_rows=int(duplicated.sum()),
                revenue_nonpositive_rows=int((revenues & finite & (numeric <= 0)).sum()),
                revenue_negative_rows=int((revenues & finite & (numeric < 0)).sum()),
                revenue_zero_rows=int((revenues & finite & (numeric == 0)).sum()),
                gross_profit_vs_revenue=pairs('gross_profit', 'revenue'),
                eps_vs_net_income=pairs('eps', 'net_income', sign=True))


def financial_sector_flag(stock_info_path):
    """Industry-name hypotheses, distinct IDs across all snapshots, no code list."""
    d = pd.read_parquet(stock_info_path)
    required = {'industry_category', 'stock_id'}
    if not required.issubset(d.columns):
        raise ValueError(f'Stock info requires {sorted(required)}')
    mask = d.industry_category.astype('string').str.contains(
        r'金融|保險|銀行|證券|金控|financial|insurance|bank|securit', case=False, na=False)
    return [dict(industry_name=str(name), stock_count=int(g.stock_id.nunique()))
            for name, g in d.loc[mask].groupby('industry_category', sort=True, observed=True)]


def validate_output_path(out, repo_root=None):
    """Resolve traversal and symlinks; only this checkout's .local-data is allowed."""
    root = Path(repo_root if repo_root is not None else REPO_ROOT).resolve()
    local = root / '.local-data'
    if local.resolve() != local:
        raise ValueError('.local-data must not redirect outside the checkout')
    path = Path(out).resolve()
    if not path.is_relative_to(local) or path == local or path.suffix.lower() != '.json':
        raise ValueError('out must be a JSON file under this checkout .local-data/')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    return path


def _identity(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(chunk)
    return dict(realpath=str(path.resolve()), size_bytes=path.stat().st_size, sha256=sha.hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--as-of', help='ISO date; defaults to current Asia/Taipei date')
    args = parser.parse_args(argv)
    try:
        out = validate_output_path(args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    as_of = _date(pd.Timestamp(args.as_of)) if args.as_of else datetime.now(ZoneInfo('Asia/Taipei')).date().isoformat()
    report = dict(schema_version='fin-audit-v1', evidence='E1', status='RESEARCH_ONLY',
                  implementation_status='UNMERGED / NOT_DEPLOYED', as_of=as_of,
                  source_commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip(),
                  implementation_sha256=_identity(Path(__file__))['sha256'],
                  environment=dict(python=sys.version, executable=sys.executable, pandas=pd.__version__, numpy=np.__version__,
                                   MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT')),
                  limitations=['No announcement/ingestion dates: UNVERIFIED PIT.',
                               'General deadlines supplied by task; financial-sector deadlines unverified.',
                               'All aliases/components are hypotheses; no type selected, no features built.',
                               'Presence coverage is not ratio completeness; observed-union denominator is non-PIT.',
                               'Industry counts union all snapshots; names may overlap or change.',
                               'Pair checks exclude duplicate keys and nonfinite values; units not verified.'],
                  inputs={}, tables={})
    for name in ['financials_raw', 'balance_sheet_raw', 'cashflow_raw', 'stock_info']:
        path = args.data_dir / (name + '.parquet')
        before = _identity(path)
        if name == 'stock_info':
            report['financial_sector'] = financial_sector_flag(path)
        else:
            d = pd.read_parquet(path)
            report['tables'][name] = dict(row_count=len(d), audit_types=audit_types(d),
                                         candidate_types=candidate_types(d),
                                         period_coverage=period_coverage(d, as_of=as_of),
                                         checks=duplicate_and_sign_checks(d))
            del d
        if _identity(path) != before:
            raise RuntimeError(f'Input changed during audit: {path}')
        report['inputs'][name] = before
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir, and use exclusive creation to preserve existing evidence.
    out = validate_output_path(out)
    with out.open('x', encoding='utf-8') as handle:
        handle.write(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
