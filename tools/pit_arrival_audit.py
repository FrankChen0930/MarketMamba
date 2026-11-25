"""Offline descriptive arrival audit. Reads time metadata, never returns/labels."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tools.pit_ledger import SOURCE, arrival_audit, atomic_json, file_hash

CALENDAR = Path('/home/frank/projects/MarketMamba/.artifacts/window34-calendar-model/calendar/combined-calendar.json')
EXPECTED_CALENDAR_SHA = 'f7fb12e8aa3a14f4ecd0809408eb189bd14808eb917f8e5a2d4a5bbba8c373a8'


def audit(source=SOURCE, calendar=CALENDAR):
    source, calendar = Path(source).resolve(), Path(calendar).resolve()
    if file_hash(calendar) != EXPECTED_CALENDAR_SHA:
        raise ValueError('STOP: calendar differs from research/v7/window34-calendar.json authority')
    cal = json.loads(calendar.read_text())
    # W34 contains both inherited diagnostic and official dates. Use official only.
    sessions = [r['session'] for r in cal['sessions'] if r['evidence_class'] == 'MARKET_SESSION_OBSERVED_OFFICIAL']
    revenue = source / 'revenue_raw.parquet'
    revenue_sha = file_hash(revenue)
    df = pd.read_parquet(revenue, columns=['stock_id', 'revenue_year', 'revenue_month', 'create_time'])
    official = arrival_audit(df, sessions, min(sessions), max(sessions))
    # Mirror existing fetcher._trading_calendar using Date projection only: not qualified exchange evidence.
    prices = source / 'prices_raw.parquet'
    prices_sha = file_hash(prices)
    dates = pd.to_datetime(pd.read_parquet(prices, columns=['Date']).Date).dropna().dt.normalize().unique()
    observed = arrival_audit(df, dates, min(dates), max(dates))
    if file_hash(revenue) != revenue_sha or file_hash(prices) != prices_sha:
        raise ValueError('STOP: input changed during audit')
    return dict(schema='pit-arrival-audit-v1', generated_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                evidence_class='DESCRIPTIVE_LOCAL_METADATA_ONLY',
                source=dict(path=str(revenue), sha256=revenue_sha),
                calendar=dict(path=str(calendar), sha256=file_hash(calendar),
                              session_count=len(sessions), evidence_class='W34_OFFICIAL_DEVELOPMENT_DATES_ONLY'),
                official_calendar_proxy=official,
                observed_calendar=dict(path=str(prices), sha256=prices_sha, columns_read=['Date'],
                                       method='same Date union as fetcher._trading_calendar; NOT qualified exchange calendar'),
                observed_price_calendar_proxy=observed,
                parser_sha256=file_hash(__file__), source_ledger_parser_sha256=file_hash(Path(__file__).with_name('pit_ledger.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--calendar', type=Path, default=CALENDAR)
    parser.add_argument('--output', type=Path, default=Path('reports/pit_arrival_audit_20261125.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose an independent name to preserve evidence')
    result = audit(args.source, args.calendar)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k]['general_10th_proxy'] for k in ['official_calendar_proxy', 'observed_price_calendar_proxy']}, indent=2))


if __name__ == '__main__':
    main()
