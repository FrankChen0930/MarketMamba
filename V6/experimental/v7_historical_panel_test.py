"""Synthetic eligibility checks: no research returns, training or network."""
import unittest
from V6.experimental.v7_historical_panel import label_diagnostic
class LabelDiagnosticTest(unittest.TestCase):
 def setUp(self):
  self.days=['2023-12-'+str(x).zfill(2) for x in [14,15,18,19,20,21,22,25,26,27,28,29]]
  self.rows={d:{'open':10,'close':11} for d in self.days}
 def call(self,**kw):
  args=dict(signal=self.days[0],horizon=5,sessions=self.days,prices=self.rows,asof='2024-01-02T00:00:00+00:00',calendar_verified=True,membership_verified=True,actions_verified=True,feature_available_at='2023-12-14T10:00:00+00:00',decision_at='2023-12-15T00:00:00+00:00',label_available_at='2023-12-29T08:00:00+00:00');args.update(kw);return label_diagnostic(**args)
 def test_session_offsets_not_stock_rows(self):
  q=self.call();self.assertEqual(q['entry'],'2023-12-15');self.assertEqual(q['exit'],'2023-12-21');self.assertTrue(q['structural_ready']);self.assertFalse(q['research_admitted'])
 def test_missing_entry_exit_and_delisting(self):
  r=dict(self.rows);del r['2023-12-15'];self.assertIn('MISSING_ENTRY_PRICE',self.call(prices=r)['reasons'])
  self.assertIn('DELISTING_BOUNDARY',self.call(delisted_at='2023-12-21')['reasons'])
 def test_unknown_calendar_membership_actions_feature_failclosed(self):
  for key,val,reason in [('calendar_verified',False,'CALENDAR_UNKNOWN'),('membership_verified',False,'MEMBERSHIP_UNKNOWN'),('actions_verified',False,'CORPORATE_ACTIONS_UNKNOWN'),('feature_available_at',None,'FEATURE_AVAILABILITY_UNKNOWN')]:
   q=self.call(**{key:val});self.assertFalse(q['eligible']);self.assertIn(reason,q['reasons'])
 def test_future_and_immature(self):
  self.assertIn('FUTURE_FEATURE',self.call(feature_available_at='2023-12-16T00:00:00+00:00')['reasons'])
  self.assertIn('IMMATURE_LABEL',self.call(asof='2023-12-20T00:00:00+00:00')['reasons'])
 def test_ten_day_missing_tail_and_zero_not_imputed(self):
  self.assertEqual(self.call(horizon=10)['exit'],'2023-12-28')
  self.assertIn('EXIT_OUTSIDE_PANEL',self.call(signal=self.days[-1])['reasons'])
  r={**self.rows,'2023-12-15':{'open':0,'close':0}};self.assertIn('INVALID_ENTRY_PRICE',self.call(prices=r)['reasons'])

 def test_label_publication_unknown_is_not_maturity(self):
  self.assertIn('LABEL_MATURITY_EVIDENCE_UNKNOWN',self.call(label_available_at=None)['reasons'])
  self.assertIn('LABEL_AVAILABILITY_BEFORE_EXIT',self.call(label_available_at='2023-12-14T00:00:00+00:00')['reasons'])
