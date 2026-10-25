import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('w27',Path(__file__).with_name('window27-safety.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LabelSafety(unittest.TestCase):
 def setUp(self):
  self.days=['2023-01-%02d'%i for i in range(2,16)]
  self.prices={d:dict(open=10.,close=10.,max=11.,min=9.) for d in self.days}
  self.row=dict(stock_id='X',signal_session=self.days[1],entry=self.days[2],exit=self.days[6],horizon=5,split='FORWARD_DIAGNOSTIC')
 def flags(self,events=()):return m.classify(self.row,self.days,self.prices,list(events))
 def test_unknown_does_not_become_clean(self):
  q=self.flags();self.assertTrue(q['eligible']);self.assertEqual(q['risk_status'],'UNKNOWN_EVENT_SUSPENSION_ASOF')
 def test_confirmed_event_crossing_excludes(self):
  q=self.flags([self.days[3]]);self.assertEqual(q['primary'],'CONFIRMED_UNIT_EVENT')
 def test_event_on_entry_same_unit_is_not_a_crossing(self):self.assertTrue(self.flags([self.days[2]])['eligible'])
 def test_feature_interval_unit_change_excludes(self):self.assertFalse(self.flags([self.days[1]])['eligible'])
 def test_holding_gap_not_compressed(self):
  del self.prices[self.days[3]];self.assertEqual(self.flags()['primary'],'HOLDING_PATH_MISSING')
 def test_zero_path_not_filled(self):
  self.prices[self.days[3]]['close']=0.;self.assertEqual(self.flags()['primary'],'HOLDING_PATH_INVALID_OR_ZERO')
 def test_missing_entry_separate_from_path_and_disjoint_primary(self):
  del self.prices[self.days[2]];q=self.flags();self.assertEqual(q['primary'],'ENTRY_EXIT_MISSING');self.assertIn('HOLDING_PATH_MISSING',q['reasons'])
 def test_nonfinite_feature_rejected(self):
  self.prices[self.days[0]]['close']=float('nan');self.assertEqual(self.flags()['primary'],'FEATURE_INVALID_OR_ZERO')
 def test_changed_shared_session_geometry_stops(self):
  self.row['exit']=self.days[7]
  with self.assertRaises(ValueError):self.flags()
 def test_future_fold_label_stops(self):
  self.row['split']='TRAIN'
  with self.assertRaises(ValueError):self.flags()
if __name__=='__main__':unittest.main()
