import unittest
import pandas as pd
import numpy as np
from V6.experimental.v7_window05_signals import candidates

class SignalCandidateTest(unittest.TestCase):
 def setUp(self):
  self.days=pd.bdate_range('2020-01-01',periods=23).strftime('%Y-%m-%d').tolist()
  self.frame=pd.DataFrame([dict(stock_id='A',session=d,close=100+i,volume=10 if i<20 else 20,foreign_net=5,benchmark=100,
    price_available_at=d+'T14:00:00+08:00',flow_available_at=d+'T18:00:00+08:00',benchmark_available_at=d+'T14:00:00+08:00') for i,d in enumerate(self.days)])
 def get(self,frame=None,at=None):
  return candidates(self.frame if frame is None else frame,self.days,'A',self.days[20],at or self.days[20]+'T20:00:00+08:00')
 def test_equivalent_cutoff_instants_use_taipei_session_date(self):
  # Both instants are before same-day availability: NaN, not UTC-date rejection.
  for cut in [self.days[20]+'T00:30:00+08:00',self.days[19]+'T16:30:00+00:00']:
   result=self.get(at=cut)
   self.assertTrue(all(np.isnan(v) for v in result.values()))

 def test_hand_calculated_values_and_trailing_baseline_excludes_current(self):
  x=self.get();self.assertEqual(x['abnormal_volume'],1.0);self.assertAlmostEqual(x['market_relative_momentum'],.2)
  self.assertAlmostEqual(x['short_reversal'],-(120/119-1));self.assertEqual(x['foreign_flow_intensity'],.25)
 def test_future_mutation_cannot_change_current_features(self):
  x=self.frame.copy();x.loc[x.session>self.days[20],['close','volume','foreign_net','benchmark']]=999999
  self.assertEqual(self.get(),self.get(x))
 def test_missing_session_is_not_rolled_or_filled(self):
  x=self.frame[self.frame.session!=self.days[3]]
  self.assertTrue(np.isnan(self.get(x)['abnormal_volume']))
 def test_unknown_or_late_availability_is_missing(self):
  x=self.frame.copy();x.loc[x.session==self.days[20],'flow_available_at']=None
  self.assertTrue(np.isnan(self.get(x)['foreign_flow_intensity']))
  self.assertTrue(np.isnan(self.get(at=self.days[20]+'T13:30:00+08:00')['abnormal_volume']))
 def test_zero_denominator_and_unit_mismatch_fail_closed(self):
  x=self.frame.copy();x['volume']=0
  self.assertTrue(np.isnan(self.get(x)['foreign_flow_intensity']))
  self.assertTrue(np.isnan(self.get(x)['abnormal_volume']))
  with self.assertRaises(ValueError):candidates(self.frame,self.days,'A',self.days[20],self.days[20]+'T20:00:00+08:00',flow_unit='lots')
 def test_duplicates_naive_cutoff_and_out_of_calendar_rejected(self):
  for x,cut in [(pd.concat([self.frame,self.frame.iloc[:1]]),self.days[20]+'T20:00:00+08:00'),(self.frame,self.days[20])]:
   with self.assertRaises(ValueError):self.get(x,cut)
 def test_volume_unit_rescale_does_not_change_ratios(self):
  x=self.frame.copy();x[['volume','foreign_net']]*=1000
  self.assertEqual(self.get(),self.get(x))
 def test_finite_positivity_and_stock_isolation(self):
  x=self.frame.copy();x.loc[x.session==self.days[20],'close']=-1
  self.assertTrue(np.isnan(self.get(x)['short_reversal']))
  y=self.frame.copy();y.stock_id='B';y.close=1e8
  self.assertEqual(self.get(),self.get(pd.concat([self.frame,y])))
