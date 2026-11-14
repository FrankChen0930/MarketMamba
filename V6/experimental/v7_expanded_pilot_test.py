import copy,unittest,tempfile,json,runpy
from pathlib import Path
from V6.experimental.v7_expanded_pilot import exploratory_mask,coverage_gate
class ExpandedMaskTest(unittest.TestCase):
 def setUp(self):
  self.days=['2023-01-'+str(i).zfill(2) for i in range(1,15)];self.prices={d:dict(open=10,close=11,max=12,min=9) for d in self.days};self.c={'reported_twse_start':'2023-01-01','reported_end':None}
 def call(self,**kw):
  a=dict(signal=self.days[1],horizon=5,sessions=self.days,prices=self.prices,cohort=self.c,next_split='2024-01-01',known_action_dates=[]);a.update(kw);return exploratory_mask(**a)
 def test_three_masks_are_not_admission(self):
  x=self.call();self.assertTrue(x['engineering_eligible']);self.assertTrue(x['exploratory_eligible']);self.assertFalse(x['strict_eligible']);self.assertFalse(x['formal_research_admitted']);self.assertEqual(x['label_class'],'RAW_PRICE_DIAGNOSTIC_PROXY_CORPORATE_ACTIONS_UNVERIFIED')
 def test_lifecycle_spans_feature_history_and_label(self):
  for c in [{'reported_twse_start':None,'reported_end':None},{'reported_twse_start':self.days[1],'reported_end':None},{'reported_twse_start':self.days[0],'reported_end':self.days[6]}]:self.assertFalse(self.call(cohort=c)['exploratory_eligible'])
 def test_known_action_zero_and_split_overlap_refuse(self):
  self.assertFalse(self.call(known_action_dates=[self.days[3]])['exploratory_eligible'])
  self.assertFalse(self.call(next_split=self.days[6])['exploratory_eligible'])
  p=copy.deepcopy(self.prices);p[self.days[2]]['open']=0;self.assertFalse(self.call(prices=p)['engineering_eligible'])
 def test_shared_calendar_not_missing_stock_rows(self):
  p=copy.deepcopy(self.prices);del p[self.days[2]];q=self.call(prices=p);self.assertFalse(q['exploratory_eligible']);self.assertEqual(q['entry'],self.days[2])
 def test_minimum_n_cannot_be_lowered_after_coverage(self):
  self.assertFalse(coverage_gate({'2022-01-03:5':19},20));self.assertTrue(coverage_gate({'2022-01-03:5':20},20));self.assertFalse(coverage_gate({},20))

class ExpandedProfileTest(unittest.TestCase):
 def test_only_explicit_W24_scope_no_formal_admission(self):
  import hashlib
  from V6.experimental.v7_research_artifacts_test import bundle_fixture,encoded
  from V6.experimental.v7_research_artifacts import publish_bundle
  from V6.experimental.v7_corrected_e5_replication import canonical_hash
  m,f=bundle_fixture();m.update(schema='v7-study-run-bundle-v2',profile='RETROSPECTIVE_EXPLORATORY_LIMITED',training_status='CPU_EXPLORATORY_PILOT_COMPLETED',evidence_class='RETROSPECTIVE_EXPLORATORY_LIMITED',authority_scope='HumanWindow24_LIMITED_EXPLORATORY_PILOT');m['identity']['checkpoint_status']='NOT_GENERATED_TINY_CLASSICAL_NO_RESUME'
  b=encoded({'schema':'logs'});f['logs']=b;m['required_artifacts'].append('logs');m['artifacts'].append(dict(name='logs',kind='TRAINING_LOGS',storage='BUNDLE',path='logs.json',sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),format='JSON',schema_id='logs'))
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(publish_bundle(Path(d)/'ok',m,canonical_hash(m),f)['status'],'ARTIFACTS_VERIFIED')
   bad=copy.deepcopy(m);bad['research_admitted']=True;self.assertEqual(publish_bundle(Path(d)/'bad',bad,canonical_hash(bad),f)['status'],'INCOMPLETE')
