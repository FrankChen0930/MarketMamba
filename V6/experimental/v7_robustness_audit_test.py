import unittest,numpy as np
from V6.experimental.v7_robustness_audit import describe,circular_bounds,leave_one
class RobustnessTest(unittest.TestCase):
 def test_zero_is_data_not_unknown_and_nonfinite_rejected(self):
  d=describe([-1.,0.,1.]);self.assertEqual(d['mean'],0);self.assertEqual(d['zero_fraction'],1/3)
  with self.assertRaises(ValueError):describe([1.,float('nan')])
 def test_bootstrap_constant_deterministic_and_no_gap_compression(self):
  a=circular_bounds(np.ones(30)*.25,list(range(30)),10,20,2501);self.assertEqual(a['lower'],.25);self.assertEqual(a['upper'],.25)
  self.assertEqual(circular_bounds(np.arange(30)/30,list(range(30)),10,20,2501),circular_bounds(np.arange(30)/30,list(range(30)),10,20,2501))
  with self.assertRaises(ValueError):circular_bounds(np.ones(3),[0,2,3],2,20,1)
 def test_leave_one_recomputes_average_ties_common_support_not_refit(self):
  p={'min_valid_n':2,'aggregation_weighting':'EQUAL_SESSION'};rows=[dict(stock_id=str(i),signal_session='2023-01-03',horizon=5,score=x,label=y,label_id=str(i),label_metadata_sha256='a'*64) for i,(x,y) in enumerate([(1.,1.),(1.,2.),(3.,3.)])]
  a=leave_one(rows,rows,'0','2023-01-03',5,p);self.assertEqual(a['Ridge']['valid_n'],2);self.assertEqual(a['delta'],0.);self.assertAlmostEqual(a['L0']['rank_ic'],1.)
  p['min_valid_n']=3;a=leave_one(rows,rows,'0','2023-01-03',5,p);self.assertIsNone(a['delta']);self.assertIn('INSUFFICIENT_SUPPORT',a['L0']['reasons'])
 def test_paired_label_mismatch_and_nan_bootstrap_refuse(self):
  rows=[dict(stock_id=str(i),signal_session='2023-01-03',horizon=5,score=float(i),label=float(i),label_id=str(i),label_metadata_sha256='a'*64) for i in range(3)]
  bad=[dict(r) for r in rows];bad[1]['label']=10.
  with self.assertRaises(ValueError):leave_one(rows,bad,'0','2023-01-03',5,{'min_valid_n':2})
  with self.assertRaises(ValueError):circular_bounds([1.,float('nan'),2.],[0,1,2],2,20,1)
