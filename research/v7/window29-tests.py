import importlib.util,unittest,copy
from pathlib import Path
spec=importlib.util.spec_from_file_location('w29panel',Path(__file__).with_name('window29-panel.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Isolation(unittest.TestCase):
 def setUp(self):
  self.days=['2022-01-%02d'%i for i in range(2,17)];self.p={d:dict(open=10.,max=11.,min=9.,close=10.) for d in self.days};self.c={'reported_twse_start':'2020-01-01','reported_end':None};self.t=self.days[2]
 def f(self,p=None,events=()):return m.feature_at('X',self.t,self.days,self.p if p is None else p,self.c,list(events))
 def test_future_label_removal_does_not_change_feature_universe(self):
  a=self.f();p={d:v for d,v in self.p.items() if d<=self.t};self.assertEqual(a,self.f(p));self.assertTrue(a['eligible'])
 def test_future_poison_does_not_change_features(self):
  p=copy.deepcopy(self.p)
  for d in p:
   if d>self.t:p[d]={k:v*123 for k,v in p[d].items()}
  self.assertEqual(self.f(),self.f(p))
 def test_label_coverage_changes_but_saved_prediction_does_not(self):
  f=self.f();prediction={'key':[self.t,'X',5],'score':f['features'][0]};saved=copy.deepcopy(prediction);a=m.label_at(f,5,self.days,self.p,[])
  self.assertTrue(a['eligible']);p=copy.deepcopy(self.p);p[self.days[7]]['close']=0.;b=m.label_at(f,5,self.days,p,[])
  self.assertFalse(b['eligible']);self.assertEqual(saved,prediction);self.assertEqual(f,self.f(p))
 def test_tail_has_prediction_but_no_label(self):
  f=m.feature_at('X',self.days[-1],self.days,self.p,self.c,[]);self.assertTrue(f['eligible']);self.assertFalse(m.label_at(f,5,self.days,self.p,[])['eligible'])
 def test_future_event_only_changes_label_join(self):
  event=self.days[5];self.assertEqual(self.f(),self.f(events=[event]));self.assertIn('SUSPECTED_UNIT_DISCONTINUITY',m.label_at(self.f(),5,self.days,self.p,[event])['reasons'])
 def test_current_feature_zero_refuses_without_fill(self):
  self.p[self.t]['close']=0.;self.assertFalse(self.f()['eligible'])
 def test_unknown_membership_refuses_predictions(self):
  self.c['reported_twse_start']=None;self.assertIn('LISTING_START_UNKNOWN',self.f()['reasons'])
 def test_training_maturity_and_minimum40(self):
  f=self.f();f['split']='TRAIN';self.assertIn('LABEL_NOT_MATURE_BEFORE_SPLIT',m.label_at(f,5,self.days,self.p,[])['reasons']);self.assertFalse(m.coverage_gate([39,60]));self.assertTrue(m.coverage_gate([40,60]))
 def test_known_delisting_end_only_restricts_future_label_not_past_feature(self):
  self.c['reported_end']=self.days[6];f=self.f();self.assertTrue(f['eligible']);self.assertIn('DELISTING_BOUNDARY',m.label_at(f,5,self.days,self.p,[])['reasons'])
 def test_discontinuity_threshold_is_data_only(self):
  self.p[self.days[3]]={k:v*3 for k,v in self.p[self.days[3]].items()};self.assertIn(self.days[3],m.suspect_events(self.p))
if __name__=='__main__':unittest.main()
