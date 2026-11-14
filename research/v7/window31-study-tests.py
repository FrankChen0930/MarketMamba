"""Zero real Fit tests for authorized views, fresh identity and feature-only scoring."""
import runpy,unittest,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
M=runpy.run_path(str(Path(__file__).with_name('window31-study.py')))
class Isolation(unittest.TestCase):
 def test_future_label_in_pinned_view_is_refused(self):
  qs=[{'key':['2022-06-01','X',5],'eligible':True,'exit':'2023-01-03'}]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'views.json';p.write_text(json.dumps({'5':{'REFIT':[0]}}));o={'origin':'2023-01-03','validation_start':'2022-01-03','view_ref':{'path':str(p),'sha256':M['sha'](p)},'stats':{'5':{'REFIT':{'view_sha256':M['H'](qs)}}}}
   with self.assertRaises(ValueError):M['authorized']({},o,qs,5,'REFIT')
 def test_refit_new_run_and_own_scaler_not_old_weights(self):
  rows=[{'key':['2021-06-01','X',5],'label':.1},{'key':['2022-06-01','X',5],'label':.2}];idx={(q['key'][0],'X'):{'features':[i+1.,i+2.]} for i,q in enumerate(rows)}
  def fake(x,y,a):return {'scaling':{'mean':x.mean(axis=0).tolist(),'std':M['np'].where(x.std(axis=0)==0,1.,x.std(axis=0)).tolist()},'coef':[0.,0.],'intercept':float(y.mean())}
  with patch.dict(M['fit_scoped'].__globals__,ridge_fit_state=fake):
   a=M['fit_scoped']({'run_id':'validation','family':'Ridge','phase':'VALIDATION_FIT','horizon':5},rows[:1],idx);b=M['fit_scoped']({'run_id':'fresh-refit','family':'Ridge','phase':'REFIT','horizon':5},rows,idx)
  self.assertNotEqual(a['run_id'],b['run_id']);self.assertNotEqual(a['view_sha256'],b['view_sha256']);self.assertNotEqual(a['state']['scaling']['mean'],b['state']['scaling']['mean']);self.assertIsNone(b['resume_checkpoint'])
 def test_feature_only_packets_ignore_future_label_field(self):
  class FakeBooster:
   def __init__(self,**kw):pass
   def predict(self,x,**kw):return x[:,0]+.1
  fs=[{'stock_id':'X','signal_session':'2023-01-03','features':[.1,.2]}];models={};specs={}
  for h in [5,10]:
   for fam in ['Ridge','LightGBM']:
    rid=f'W31-O1-H{h}-{fam}-REFIT';state={'scaling':{'mean':[0.,0.],'std':[1.,1.]},'coef':[1.,0.],'intercept':0.} if fam=='Ridge' else {'booster_text':'mock','actual_iterations':64};models[rid]={'state':state};specs[rid]={'recipe_id':'fixed'}
  o={'origin_id':'O1','fold':{'forward':['2023-01-03','2023-03-31']},'model_specs':specs};p={'source_dataset_id':'d','feature_id':'f','label_contract_id':'l'}
  with patch.object(M['lgb'],'Booster',FakeBooster):
   a=M['prediction_packets'](o,'FORWARD',models,fs,p);fs[0]['unused_future_label']=None;b=M['prediction_packets'](o,'FORWARD',models,fs,p)
  self.assertEqual(a,b);self.assertEqual(len(a),8)
 def test_prepared_refit_lineage_is_new_and_not_resume(self):
  p=json.loads(Path(__file__).with_name('window31-prepared.json').read_text())
  for o in p['origins']:
   ids=[s['run_id'] for s in o['model_specs'].values()];self.assertEqual(len(ids),len(set(ids)))
   for s in o['model_specs'].values():
    if s['phase']=='REFIT':self.assertIsNone(s['resume_checkpoint']);self.assertTrue(all(v=='NEW' for v in s['declared_lineage']['initialization'].values()));self.assertNotEqual(s['run_id'],s['declared_lineage']['selection']['run_id'])
if __name__=='__main__':unittest.main()
