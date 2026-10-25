import importlib.util,unittest,tempfile,copy
from pathlib import Path
s=importlib.util.spec_from_file_location('w30',Path(__file__).with_name('window30-lgbm.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class BoundedLGBM(unittest.TestCase):
 def test_claim_cannot_fit_twice(self):
  with tempfile.TemporaryDirectory() as d:
   m.reserve(Path(d),5)
   with self.assertRaises(FileExistsError):m.reserve(Path(d),5)
 def test_training_rows_cannot_contain_future_maturity(self):
  f={('2021-12-30','X'):{'features':[1.,2.]}};q=[{'key':['2021-12-30','X',5],'eligible':True,'entry':'2021-12-31','exit':'2022-01-05','label':.1}]
  with self.assertRaises(ValueError):m.training(f,q,5)
 def test_mock_fit_receives_train_arrays_only_and_fixed_recipe(self):
  calls=[]
  class FakeBooster:
   def model_to_string(self,**kw):return 'trusted-mock-tree'
   def current_iteration(self):return 64
  class Fake:
   def __init__(self,**kw):calls.append(('config',kw));self.booster_=FakeBooster()
   def fit(self,x,y,**kw):calls.append(('fit',len(x),len(y),kw));return self
  recipe=m.recipe();x=m.np.asarray([[1.,2.],[3.,4.]]);y=m.np.asarray([.1,.2]);model=m.fit_arrays(x,y,recipe,Fake)
  self.assertEqual(calls[1],('fit',2,2,{'feature_name':recipe['feature_order']}));self.assertEqual(calls[0][1],recipe['config']);self.assertEqual(model['actual_iterations'],64)
 def test_source_label_identity_mismatch_refuses(self):
  a={'stock_id':'X','signal_session':'2022-01-03','horizon':5,'label_id':'a','label_metadata_sha256':'meta','label':.1}
  with self.assertRaises(ValueError):m.assert_support([a],[{**a,'label':.2}])
 def test_feature_only_scoring_has_no_future_label_dependency(self):
  from unittest.mock import patch
  class FakeBooster:
   def __init__(self,**kw):pass
   def feature_name(self):return m.recipe()['feature_order']
   def predict(self,x,**kw):return x[:,0]
  fs=[{'stock_id':'X','signal_session':d,'split':role,'features':[.1,.2]} for d,role in [('2022-06-01','VALIDATION'),('2023-06-01','EXPOSED_DIAGNOSTIC')]]
  models={}
  for h in [5,10]:
   q={'model_id':'mock'+str(h),'spec_sha256':'spec','recipe_sha256':'r','booster_text':'mock-only','actual_iterations':64,'feature_order':m.recipe()['feature_order'],'train_view_sha256':'t'};q['model_sha256']=m.model_sha(q);models[str(h)]=q
  p={'expected_prediction_keys_sha256':m.H(sorted((f['signal_session'],'X',h) for f in fs for h in [5,10])),'feature_id':'f','dataset_id':'d','source_evidence_id':'s'}
  with patch.object(m.lgb,'Booster',FakeBooster):
   a,_=m.predict(models,fs,p)
   for f in fs:f['unused_future_label']=None
   b,_=m.predict(models,fs,p)
  self.assertEqual(a,b);self.assertEqual(sum(map(len,a.values())),4);self.assertTrue(all('model_id' in r and 'source_evidence_ref' in r for rows in a.values() for r in rows))
 def test_mutable_counter_uses_existing_atomic_json(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'counter.json';m.atomic_json(p,{'n':1});m.atomic_json(p,{'n':2});self.assertEqual(m.json.loads(p.read_text()),{'n':2})
if __name__=='__main__':unittest.main()
