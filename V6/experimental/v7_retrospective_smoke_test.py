import unittest,tempfile,copy
from pathlib import Path
from V6.experimental.v7_retrospective_smoke import schedule_row_role,engineering_sample
from V6.experimental.v7_classical_baseline import ridge_fit_state
class SmokeGuardsTest(unittest.TestCase):
 def test_last_trading_day_note_does_not_close_current_day(self):
  self.assertEqual(schedule_row_role(['2023-01-17','農曆春節前最後交易日','農曆春節前最後交易。1月18日及1月19日市場無交易']),'OFFICIAL_SESSION_VERIFIED')
  self.assertEqual(schedule_row_role(['2021-02-08','農曆春節前最後交易日','2月8日市場無交易，僅辦理結算交割作業。']),'OFFICIAL_CLOSURE')
 def test_ridge_scaling_is_training_only(self):
  import numpy as np
  m=ridge_fit_state(np.array([[1.,2.],[3.,4.]]),np.array([0.,1.]),1.)
  self.assertEqual(m['scaling']['mean'],[2.,3.])
 def test_session_geometry_zero_actions_and_membership(self):
  days=['2023-01-'+str(n).zfill(2) for n in range(1,15)]
  rows={d:dict(open=10,close=11,max=12,min=9) for d in days}
  kw=dict(signal=days[1],horizon=5,sessions=days,prices=rows,market='TWSE_LISTED',calendar_ok=True,known_action_dates=[])
  q=engineering_sample(**kw);self.assertTrue(q['engineering_eligible']);self.assertAlmostEqual(q['diagnostic_label'],.1);self.assertEqual(q['exit'],days[6]);self.assertFalse(q['strict_eligible'])
  self.assertFalse(engineering_sample(**{**kw,'market':'UNKNOWN'})['engineering_eligible'])
  self.assertFalse(engineering_sample(**{**kw,'calendar_ok':False})['engineering_eligible'])
  self.assertFalse(engineering_sample(**{**kw,'known_action_dates':[days[3]]})['engineering_eligible'])
  changed=copy.deepcopy(rows);changed[days[2]]['open']=0
  self.assertFalse(engineering_sample(**{**kw,'prices':changed})['engineering_eligible'])
 def test_future_edits_cannot_change_features(self):
  days=['2023-01-'+str(n).zfill(2) for n in range(1,15)];r={d:dict(open=10,close=11,max=12,min=9) for d in days}
  kw=dict(signal=days[1],horizon=5,sessions=days,prices=r,market='TWSE_LISTED',calendar_ok=True,known_action_dates=[])
  a=engineering_sample(**kw);r[days[6]]['close']=10;b=engineering_sample(**kw)
  self.assertEqual(a['features'],b['features']);self.assertNotEqual(a['diagnostic_label'],b['diagnostic_label'])

class EngineeringBundleScopeTest(unittest.TestCase):
 def test_new_profile_is_explicit_and_never_admitted(self):
  from V6.experimental.v7_research_artifacts_test import bundle_fixture,encoded
  from V6.experimental.v7_research_artifacts import publish_bundle
  from V6.experimental.v7_corrected_e5_replication import canonical_hash
  import hashlib
  m,f=bundle_fixture();m.update(schema='v7-study-run-bundle-v2',profile='RETROSPECTIVE_ENGINEERING_SMOKE',training_status='CPU_ENGINEERING_SMOKE_COMPLETED',evidence_class='RETROSPECTIVE_ENGINEERING_SMOKE',authority_scope='HumanWindow23_LIMITED_ENGINEERING_SMOKE');m['identity'].update(checkpoint_status='NOT_GENERATED_TINY_CLASSICAL_NO_RESUME')
  b=encoded({'schema':'logs-v1'});f['logs']=b;m['artifacts'].append(dict(name='logs',kind='TRAINING_LOGS',storage='BUNDLE',path='logs.json',sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),format='JSON',schema_id='logs-v1'));m['required_artifacts'].append('logs')
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(publish_bundle(Path(d)/'valid',m,canonical_hash(m),f)['status'],'ARTIFACTS_VERIFIED')
   bad=copy.deepcopy(m);bad['research_admitted']=True;self.assertEqual(publish_bundle(Path(d)/'bad',bad,canonical_hash(bad),f)['status'],'INCOMPLETE')
   bad=copy.deepcopy(m);bad['authority_scope']='other';self.assertEqual(publish_bundle(Path(d)/'scope',bad,canonical_hash(bad),f)['status'],'INCOMPLETE')

class ReplayOrderTest(unittest.TestCase):
 def test_json_map_order_is_not_metric_identity_and_duplicates_refused(self):
  import runpy
  f=runpy.run_path(str(Path(__file__).resolve().parents[2]/'research/v7/window23-pipeline.py'))['daily_identity']
  a=[dict(model='L0',signal_session='2023-01-03',horizon=5,rank_ic=.5),dict(model='Ridge',signal_session='2023-01-03',horizon=5,rank_ic=-.5)]
  self.assertEqual(f(a),f(list(reversed(a))))
  b=copy.deepcopy(a);b[0]['rank_ic']=.6;self.assertNotEqual(f(a),f(b))
  with self.assertRaises(ValueError):f([a[0],a[0]])
