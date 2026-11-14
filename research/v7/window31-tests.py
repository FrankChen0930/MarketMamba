import importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('w31',Path(__file__).with_name('window31-folds.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TemporalBoundaries(unittest.TestCase):
 def test_exit_at_boundary_refused_and_before_allowed(self):
  q={'key':['2022-03-01','X',5],'exit':'2022-04-01','eligible':True}
  self.assertFalse(m.mature(q,'2022-04-01'));q['exit']='2022-03-31';self.assertTrue(m.mature(q,'2022-04-01'))
 def test_unknown_numeric_or_exit_cannot_be_mature(self):
  self.assertFalse(m.mature({'eligible':False,'exit':'2022-01-03'},'2023-01-01'));self.assertFalse(m.mature({'eligible':True,'exit':None},'2023-01-01'))
 def test_refit_includes_newer_mature_but_not_forward(self):
  rows=[{'key':[d,'X',5],'exit':e,'eligible':True} for d,e in [('2021-01-01','2021-01-08'),('2022-06-01','2022-06-08'),('2023-01-03','2023-01-10')]]
  self.assertEqual(m.view_ids(rows,5,'2021-01-01','2022-01-03','2022-01-03'),[0]);self.assertEqual(m.view_ids(rows,5,'2021-01-01','2023-01-03','2023-01-03'),[0,1])
 def test_shared_session_horizon_holiday_not_stock_row_count(self):
  ds=['2023-03-30','2023-03-31','2023-04-06','2023-04-07','2023-04-10','2023-04-11','2023-04-12','2023-04-13'];p={d:{'open':10.,'close':10.,'min':9.,'max':11.} for d in ds};f={'signal_session':ds[1],'stock_id':'X','features':[.1,.2],'known_membership_end':None}
  q=m.numeric_label(f,5,ds,p,[]);self.assertEqual(q['entry'],'2023-04-06');self.assertEqual(q['exit'],'2023-04-12')
 def test_interior_missing_zero_fail_closed(self):
  ds=['2023-01-%02d'%i for i in range(2,15)];p={d:{'open':10.,'close':10.,'min':9.,'max':11.} for d in ds};f={'signal_session':ds[1],'stock_id':'X','features':[.1,.2],'known_membership_end':None};del p[ds[3]];self.assertIn('HOLDING_PATH_MISSING',m.numeric_label(f,5,ds,p,[])['reasons'])
 def test_future_event_outside_holding_interval_does_not_select_sample(self):
  ds=['2023-01-%02d'%i for i in range(2,15)];p={d:{'open':10.,'close':10.,'min':9.,'max':11.} for d in ds};f={'signal_session':ds[1],'stock_id':'X','features':[.1,.2],'known_membership_end':None};self.assertEqual(m.numeric_label(f,5,ds,p,[]),m.numeric_label(f,5,ds,p,[ds[-1]]))
if __name__=='__main__':unittest.main()
