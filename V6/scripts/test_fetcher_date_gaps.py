"""Offline date-binding and calendar-gap regressions; all writes are temporary.

Run with the V6 data environment: python -m unittest discover -s V6/scripts
-p test_fetcher_date_gaps.py. No real config/.env or network is loaded.
"""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / 'marketmamba/data/fetcher.py'


def load_fetcher(root, source=None):
    config = types.ModuleType('marketmamba.config')
    for name in ('DATA_END_DATE', 'DATA_SOURCE_PRIORITY', 'DATA_START_DATE',
                 'FINMIND_TOKEN', 'MARGIN_FORWARD_FILL', 'TPEX_INSTITUTIONAL_URL',
                 'TWSE_INSTITUTIONAL_URL'):
        setattr(config, name, '')
    config.DATA_DIR = root
    config.PROCESSED_DIR = root
    requests = types.ModuleType('requests')
    requests.get = Mock(side_effect=AssertionError('unmocked network'))
    with patch.dict(sys.modules, {'marketmamba.config': config, 'requests': requests,
                                 'yfinance': types.ModuleType('yfinance')}):
        spec = importlib.util.spec_from_file_location('gap_fetcher', SOURCE)
        module = importlib.util.module_from_spec(spec)
        exec(compile(source if source is not None else SOURCE.read_text(), str(SOURCE), 'exec'),
             module.__dict__)
    module.time = Mock()
    return module


class FetcherDateGapTest(unittest.TestCase):
    source = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = load_fetcher(self.root, self.source)

    def payloads(self):
        return [
            ('fetch_per_twse_direct', {'data': [['2330', '台積電', '20', '2', '5']]}),
            ('fetch_securities_lending_twse_direct', {'data': [['2330', '100', '2317', '50']]}),
            ('fetch_institutional_twse', {'stat': 'OK', 'data': [['2330', '台積電'] + ['1'] * 17]}),
            ('fetch_institutional_tpex', {'tables': [{'data': [['2330', '台積電'] + ['1'] * 22]}]}),
        ]

    def fetch(self, name, payload):
        self.f.requests.get = Mock(return_value=Mock(json=Mock(return_value=payload)))
        return getattr(self.f, name)('2026-10-12')

    def test_wrong_response_date_rejected(self):
        for name, payload in self.payloads():
            with self.subTest(endpoint=name), self.assertLogs(self.f.logger, 'WARNING'):
                self.assertIsNone(self.fetch(name, dict(payload, date='20261117')))

    def test_missing_or_invalid_response_date_rejected(self):
        for name, payload in self.payloads():
            for meta in ({}, {'date': '20260230'}, {'date': '20261012~20261013'},
                         {'title': '沒有日期'}):
                with self.subTest(endpoint=name, meta=meta):
                    self.assertIsNone(self.fetch(name, dict(payload, **meta)))

    def test_matching_response_dates_accepted(self):
        for name, payload in self.payloads():
            for meta in ({'date': '20261012'}, {'date': '2026/10/12'},
                         {'title': '115年10月12日 當日報表'},
                         {'title': '115/10/12 報表'}):
                with self.subTest(endpoint=name, meta=meta):
                    result = self.fetch(name, dict(payload, **meta))
                    self.assertIsNotNone(result)
                    self.assertIn('2330', set(result.stock_id))

    def test_conflicting_title_date_rejected(self):
        for name, payload in self.payloads():
            with self.subTest(endpoint=name):
                self.assertIsNone(self.fetch(name, dict(payload, date='20261012',
                                                       title='115/10/13 報表')))

    def test_missing_trading_days_is_pure_sorted_set_difference(self):
        existing = ['2026-10-12', pd.Timestamp('2026-10-14')]
        calendar = ['2026-10-14', '2026-10-13', '2026-10-13', '2026-10-12']
        self.assertEqual(self.f.missing_trading_days(existing, calendar), ['2026-10-13'])
        self.assertEqual(self.f.missing_trading_days([], []), [])
        self.assertEqual(len(existing), 2)
        self.assertEqual(len(calendar), 4)

    def prepare(self, name, dates, calendar):
        def frame(days):
            return pd.DataFrame({'Date': pd.to_datetime(days), 'stock_id': ['2330'] * len(days),
                                 'value': [1.] * len(days)})
        old = frame(dates)
        path = self.root / (name + '.parquet')
        old.to_parquet(path, index=False)
        prices = frame(calendar).rename(columns={'value': 'Close'})
        prices.to_parquet(self.root / 'prices_raw.parquet', index=False)
        self.f.fetch_margin_direct = lambda d: frame([d])
        self.f.fetch_daytrade_direct = lambda d: frame([d])
        self.f.daytrade_shares_to_ratio = lambda df: df.assign(Day_Trade_Volume=.5)
        if name == 'daytrade_raw':
            old = old.assign(Day_Trade_Volume=.5)
            old.to_parquet(path, index=False)
        self.f.fetch_shares_outstanding_mops = Mock(return_value=pd.DataFrame(
            {'stock_id': ['2330'], 'shares_outstanding': [10.]}))
        if name == 'market_value_raw':
            old = old.rename(columns={'value': 'market_value'})
            old.to_parquet(path, index=False)
        return path, lambda d: frame([d])

    def catch_up(self, name, today, fetch, **kwargs):
        if name == 'per_raw':
            return self.f._catch_up_generic(name, fetch, today, **kwargs)
        return getattr(self.f, '_catch_up_' + name.removesuffix('_raw'))(today, **kwargs)

    def test_middle_gap_filled_and_existing_rows_preserved(self):
        for name in ('margin_raw', 'daytrade_raw', 'per_raw', 'market_value_raw'):
            with self.subTest(source=name):
                path, fetch = self.prepare(name, ['2026-10-12', '2026-10-14'],
                                            ['2026-10-12', '2026-10-13', '2026-10-14'])
                n, days = self.catch_up(name, '2026-10-14', fetch)
                self.assertEqual((n, days), (1, ['2026-10-13']))
                self.assertEqual(len(pd.read_parquet(path)), 3)

    def test_limit_and_default_ten(self):
        calendar = pd.bdate_range('2026-09-01', periods=15).strftime('%Y-%m-%d').tolist()
        for name in ('margin_raw', 'daytrade_raw', 'per_raw', 'market_value_raw'):
            for limit in (None, 2, 0):
                with self.subTest(source=name, limit=limit):
                    path, fetch = self.prepare(name, [calendar[0]], calendar)
                    kwargs = {} if limit is None else {'max_days': limit}
                    n, days = self.catch_up(name, calendar[-1], fetch, **kwargs)
                    count = 10 if limit is None else limit
                    self.assertEqual(n, count)
                    self.assertEqual(days, calendar[-count:] if count else [])
                    self.assertEqual(len(pd.read_parquet(path)), 1 + count)

    def test_no_gap_does_not_write_or_fetch(self):
        for name in ('margin_raw', 'daytrade_raw', 'per_raw', 'market_value_raw'):
            with self.subTest(source=name):
                path, _ = self.prepare(name, ['2026-10-12'], ['2026-10-12'])
                before = path.read_bytes()
                fail = Mock(side_effect=AssertionError('no gap must not fetch'))
                self.f.fetch_margin_direct = self.f.fetch_daytrade_direct = fail
                self.f.fetch_shares_outstanding_mops = fail
                self.assertEqual(self.catch_up(name, '2026-10-12', fail), (0, []))
                self.assertEqual(path.read_bytes(), before)

    def test_no_calendar_or_future_days_do_not_trigger_fetch(self):
        for name in ('margin_raw', 'daytrade_raw', 'per_raw', 'market_value_raw'):
            with self.subTest(source=name):
                _, fetch = self.prepare(name, ['2026-10-12'], ['2026-10-12', '2026-10-14'])
                self.assertEqual(self.catch_up(name, '2026-10-12', fetch), (0, []))
                (self.root / 'prices_raw.parquet').unlink()
                self.assertEqual(self.catch_up(name, '2026-10-14', fetch), (0, []))


def run_mutations():
    """Mutate in memory, never overwrite source or touch non-fixture data."""
    original = SOURCE.read_text()
    mutations = [
        ('A_bypass_date_binding',
         '    def parse(value):', '    return True\n\n    def parse(value):',
         'test_wrong_response_date_rejected'),
        ('A_accept_missing_date',
         'if invalid_date or not dates or any(d != requested for d in dates):',
         'if dates and any(d != requested for d in dates):',
         'test_missing_or_invalid_response_date_rejected'),
        ('B_latest_plus_one',
         'start = min(existing) if existing else today',
         'start = (pd.Timestamp(max(existing)) + pd.Timedelta(days=1)).isoformat() if existing else today',
         'test_middle_gap_filled_and_existing_rows_preserved'),
        ('B_remove_cap', 'return days[-max_days:]', 'return days',
         'test_limit_and_default_ten'),
        ('B_refetch_existing',
         'return sorted(normalized(calendar_days) - normalized(existing_dates))',
         'return sorted(normalized(calendar_days))',
         'test_no_gap_does_not_write_or_fetch'),
    ]
    reports = []
    for name, before, after, test in mutations:
        if original.count(before) != 1:
            raise AssertionError(f'{name}: mutation target must be unique')
        FetcherDateGapTest.source = original.replace(before, after, 1)
        suite = unittest.TestSuite([FetcherDateGapTest(test)])
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output).run(suite)
        reports.append({'mutation': name, 'killed': bool(result.failures) and not result.errors,
                        'assertion_failures': len(result.failures), 'errors': len(result.errors)})
    FetcherDateGapTest.source = None
    print(json.dumps({'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                      'mutations': reports}, indent=2))
    return 0 if all(r['killed'] for r in reports) else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--mutations']:
        sys.exit(run_mutations())
    unittest.main()
