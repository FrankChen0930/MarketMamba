"""Offline regression of the real fetcher module with isolated configuration.

No operational config import, cache access, provider request or financial policy
change. Only HTTP/HTML and quarterly wide-table acquisition are fixture supplied.
"""
import importlib.util
import logging
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


class OfficialParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        config = ModuleType('marketmamba.config')
        for name in ('DATA_DIR', 'PROCESSED_DIR'):
            setattr(config, name, Path(cls.temp.name) / name)
        for name in ('DATA_END_DATE', 'DATA_SOURCE_PRIORITY', 'DATA_START_DATE',
                     'FINMIND_TOKEN', 'MARGIN_FORWARD_FILL',
                     'TPEX_INSTITUTIONAL_URL', 'TWSE_INSTITUTIONAL_URL'):
            setattr(config, name, None)
        path = Path(__file__).with_name('fetcher.py')
        spec = importlib.util.spec_from_file_location('offline_official_fetcher', path)
        cls.fetcher = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'marketmamba.config': config,
                                       'yfinance': SimpleNamespace(),
                                       'requests': SimpleNamespace(get=Mock(side_effect=AssertionError('network disabled')))}):
            spec.loader.exec_module(cls.fetcher)

    def setUp(self):
        self.get = Mock(side_effect=AssertionError('unmocked network call'))
        self.patcher = patch.object(self.fetcher.requests, 'get', self.get)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.sleep = patch.object(self.fetcher.time, 'sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def response(self, payload):
        self.get.side_effect = None
        self.get.return_value = SimpleNamespace(status_code=200, json=lambda: payload,
                                               raise_for_status=lambda: None)

    def twse(self, **updates):
        payload = {'stat': 'OK', 'date': '20260716', 'tables': [{
            'fields': ['證券代號', '開盤價', '最高價', '最低價', '收盤價', '成交股數'],
            'data': [[str(1100+i), '10', '12', '9', '11', '1,000'] for i in range(101)]}]}
        payload.update(updates)
        self.response(payload)
        return self.fetcher.fetch_prices_twse_direct('2026-07-16')

    def test_twse_valid_identity_and_values(self):
        for date in ('20260716', '2026/07/16', '2026-07-16'):
            with self.subTest(date=date):
                frame = self.twse(date=date)
                self.assertEqual(len(frame), 101)
                self.assertEqual(frame.Date.unique().tolist(), ['2026-07-16'])
                self.assertEqual(frame.iloc[0][['Open','High','Low','Close','Volume']].tolist(),
                                 [10.,12.,9.,11.,1000.])
        self.assertEqual(self.get.call_args.kwargs['params']['date'], '20260716')

    def test_twse_rejects_missing_mismatched_response_identity(self):
        for date in ('20260717', '', None, ['20260716']):
            with self.subTest(date=date):
                self.assertIsNone(self.twse(date=date))
        self.assertIsNone(self.twse(stat='no data'))
        payload = {'stat':'OK','tables':self.get.return_value.json().get('tables',[])}
        self.response(payload)
        self.assertIsNone(self.fetcher.fetch_prices_twse_direct('2026-07-16'))

    def tpex(self, fields=None, row=None, **updates):
        payload = {'stat':'ok','date':'20260716','tables':[{
            'fields': fields if fields is not None else
                      ['代號','名稱','收盤','漲跌','開盤','最高','最低','成交股數'],
            'data':[row if row is not None else ['1101','fixture','11','1','10','12','9','1,000']]}]}
        payload.update(updates)
        self.response(payload)
        return self.fetcher.fetch_prices_tpex_direct('2026-07-16')

    def test_tpex_header_binding_preserves_values_under_permutations(self):
        fields = ['代號','名稱','收盤','漲跌','開盤','最高','最低','成交股數']
        row = ['1101','fixture','11','1','10','12','9','1,000']
        baseline = self.tpex()
        for order in (list(range(8)), [0,1,4,3,2,5,6,7], list(reversed(range(8))), [7,2,0,6,4,1,3,5]):
            with self.subTest(order=order):
                frame=self.tpex([fields[i] for i in order], [row[i] for i in order])
                pd.testing.assert_frame_equal(frame, baseline)
        pd.testing.assert_frame_equal(self.tpex(['extra']+fields,['unused']+row), baseline)

    def test_tpex_unknown_duplicate_or_missing_required_headers_rejected(self):
        fields = ['代號','名稱','收盤','漲跌','開盤','最高','最低','成交股數']
        for required in ('代號','收盤','開盤','最高','最低','成交股數'):
            bad = [f if f != required else 'UNKNOWN' for f in fields]
            with self.subTest(missing=required): self.assertIsNone(self.tpex(bad))
        for bad in (fields+['收盤'], fields[:4]+['收盤']+fields[5:], fields[:4]+[None]+fields[5:]):
            with self.subTest(fields=bad): self.assertIsNone(self.tpex(bad))

    def test_tpex_date_and_suspended_skeleton_controls(self):
        self.assertIsNone(self.tpex(date='20260717'))
        self.assertIsNone(self.tpex(row=['1101']))
        self.assertIsNone(self.tpex(row=['1101','fixture','---','0','---','---','---','0']))

    def test_mops_quarterly_existing_missing_market_guard(self):
        wide = pd.DataFrame({'公司代號':['1101']})
        for missing in (None,pd.DataFrame()):
            with self.subTest(missing=type(missing).__name__), patch.object(
                self.fetcher,'fetch_mops_statement_wide',side_effect=[wide,missing]):
                self.assertIsNone(self.fetcher.fetch_financial_statement_mops_direct('financials',2026,2))

    def test_mops_quarterly_previous_market_failure_rejects_current(self):
        wide = pd.DataFrame({'公司代號':['1101']})
        long = pd.DataFrame({'stock_id':['1101'],'type':['Revenue'],'value':[10.]})
        with patch.object(self.fetcher,'fetch_mops_statement_wide',side_effect=[wide,wide,wide,None]), \
             patch.object(self.fetcher,'_mops_wide_to_long',return_value=long):
            self.assertIsNone(self.fetcher.fetch_financial_statement_mops_direct('financials',2026,2))

    def test_mops_monthly_preserves_existing_partial_policy_and_units(self):
        for partial in (False,True):
            def get(url, **kwargs):
                if partial and '/otc/' in url: raise OSError('fixture failure')
                return SimpleNamespace(text='otc' if '/otc/' in url else 'sii',raise_for_status=lambda:None)
            def html(stream):
                return [pd.DataFrame({'公司代號':['2201' if stream.getvalue()=='otc' else '1101'],
                                      '營業收入|當月營收':['100']})]
            self.get.side_effect=get
            with patch.object(pd,'read_html',side_effect=html):
                frame=self.fetcher.fetch_revenue_mops_direct(2026,5)
            self.assertEqual(len(frame),1 if partial else 2)
            self.assertEqual(frame.revenue.unique().tolist(),[100000])
            self.assertEqual(frame.Date.astype(str).unique().tolist(),['2026-06-01'])
            self.assertEqual(frame.create_time.unique().tolist(),[''])


if __name__ == '__main__':
    unittest.main()
