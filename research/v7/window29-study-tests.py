"""Synthetic regression: outcome changes cannot edit feature-only predictions. Zero fits."""
import unittest,runpy,copy
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('window29-study.py')))
class FeaturePredictionBoundary(unittest.TestCase):
 def setUp(self):
  self.features=[{'stock_id':s,'signal_session':d,'split':r,'features':[i/100.,.02]} for i,s in enumerate(['X','Y']) for d,r in [('2022-06-01','VALIDATION'),('2023-06-01','EXPOSED_DIAGNOSTIC')]]
  state={'scaling':{'mean':[0.,0.],'std':[1.,1.]},'coef':[1.,-1.],'intercept':0.}
  self.models={str(h):{'state':state} for h in [5,10]};self.expected=M['H'](sorted((f['signal_session'],f['stock_id'],h) for f in self.features for h in [5,10]));self.pred=M['predict'](self.features,self.models,self.expected)
  self.labels=[{'key':[f['signal_session'],f['stock_id'],h],'eligible':True,'label':i/10.,'label_id':'label'+str(i),'label_metadata_sha256':'meta'+str(i),'reasons':[]} for i,f in enumerate(self.features) for h in [5,10]]
 def test_changed_removed_future_labels_leave_predictions_identical(self):
  before=copy.deepcopy(self.pred)
  for r in self.labels:r.update(eligible=False,label=None,reasons=['TEST_LABEL_REMOVED'])
  self.assertEqual(before,M['predict'](self.features,self.models,self.expected))
 def test_evaluation_join_changes_coverage_not_saved_predictions(self):
  policy={'min_valid_n':40};before=copy.deepcopy(self.pred);days=['2022-06-01','2023-06-01']
  a=M['evaluate'](self.pred,self.labels,days,policy,'contract')
  for r in self.labels:r.update(eligible=False,label=None,reasons=['TEST_MISSING'])
  b=M['evaluate'](self.pred,self.labels,days,policy,'contract')
  self.assertEqual(before,self.pred);self.assertTrue(all(r['valid_n']==2 for r in a[0]));self.assertTrue(all(r['valid_n']==0 for r in b[0]));self.assertTrue(all(r['rank_ic'] is None for r in b[0]))
 def test_changed_expected_prediction_universe_refuses(self):
  with self.assertRaises(AssertionError):M['predict'](self.features,self.models,'wrong')
if __name__=='__main__':unittest.main()
