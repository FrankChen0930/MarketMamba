import importlib.util,unittest,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('w28',Path(__file__).with_name('window28-study.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TrainingAndCausality(unittest.TestCase):
 def test_clean_training_filters_mask_and_horizon_only(self):
  rows=[dict(stock_id='X',signal_session='2021-06-01',entry='2021-06-02',exit='2021-06-08',horizon=5,split='TRAIN',features=[1.,2.],diagnostic_label=.1)]
  key=m.key(rows[0]);self.assertEqual(m.clean_training(rows,{key:{'eligible':True}},5),rows)
  self.assertEqual(m.clean_training(rows,{key:{'eligible':False}},5),[])
  self.assertEqual(m.clean_training(rows,{key:{'eligible':True}},10),[])
 def test_future_train_label_fails(self):
  r=dict(stock_id='X',signal_session='2021-12-30',entry='2021-12-31',exit='2022-01-05',horizon=5,split='TRAIN')
  with self.assertRaises(ValueError):m.clean_training([r],{m.key(r):{'eligible':True}},5)
 def test_wrong_label_identity_fails_support(self):
  a=[dict(stock_id='X',signal_session='2022-01-03',horizon=5,label_id='a',label_metadata_sha256='m',label=.1,score=1.)]
  with self.assertRaises(ValueError):m.same_support(a,[{**a[0],'label':.2}])
 def test_exclusive_fit_claim_no_second_dispatch(self):
  with tempfile.TemporaryDirectory() as d:
   m.reserve_fit(Path(d),5)
   with self.assertRaises(FileExistsError):m.reserve_fit(Path(d),5)
 def test_poison_changes_future_only_without_mutating_source(self):
  rows={'2021-01-01':{'open':10.,'close':10.,'min':9.,'max':11.,'Trading_Volume':3},'2021-01-04':{'open':10.,'close':10.,'min':9.,'max':11.,'Trading_Volume':3}}
  changed,n=m.poison_future(rows,'2021-01-01');self.assertEqual(n,1);self.assertEqual(changed['2021-01-01'],rows['2021-01-01']);self.assertEqual(rows['2021-01-04']['close'],10.);self.assertNotEqual(changed['2021-01-04'],rows['2021-01-04'])
 def test_actual_feature_pipeline_is_future_invariant_and_positive_control_sensitive(self):
  days=['2021-01-%02d'%i for i in range(1,15)];p={d:dict(open=10.,close=10.,min=9.,max=11.) for d in days}
  before=m.features(days[2],5,days,p);poisoned,_=m.poison_future(p,days[2]);self.assertEqual(before,m.features(days[2],5,days,poisoned))
  p[days[2]]={k:v*2 for k,v in p[days[2]].items()};self.assertNotEqual(before,m.features(days[2],5,days,p))
if __name__=='__main__':unittest.main()
