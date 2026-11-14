"""Offline acceptance: synthetic model bytes/scores only; no fit/network/inference."""
import copy,hashlib,json,runpy,tempfile,unittest
from datetime import date,timedelta
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('window32-shadow.py')))
class ShadowAcceptance(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
  days=[(date(2030,1,2)+timedelta(days=i)).isoformat() for i in range(12)] # explicitly synthetic sessions, not real market claims
  self.ctx={'evidence_class':'SYNTHETIC','study_id':'W32_OFFLINE_ONLY','dataset_id':'synthetic-data','feature_id':'synthetic-two-features','universe_id':'synthetic-U','label_policy_id':'synthetic-label','recipe_set_id':'synthetic-recipe','code_commit':'a'*40,'signal_session':days[0],'entry_session':days[1],'calendar':[{'session':d,'open_at':d+'T09:00:00+08:00','close_at':d+'T13:30:00+08:00'} for d in days],'calendar_status':'SYNTHETIC_QUALIFIED','qc_status':'PASS','observation_at':days[0]+'T13:30:00+08:00','retrieved_at':days[0]+'T19:00:00+08:00','qc_at':days[0]+'T19:01:00+08:00','publication_at':None,'capture_class':'ORIGINAL_FORWARD_CAPTURE'}
  raw=self.root/'raw.json';raw.write_text('{"synthetic":true}');self.ctx['raw_refs']=[{'path':str(raw),'sha256':M['sha'](raw)}]
  self.models=[];self.rows=[];self.features=[{'stock_id':s,'signal_session':days[0],'values':[.1,.2]} for s in ['SYNTH_A','SYNTH_B']]
  for fam in M['FAMILIES']:
   for h in [5,10]:
    model=self.root/(fam+str(h)+'.json');model.write_text(json.dumps({'synthetic':fam,'horizon':h}));self.models.append({'family':fam,'horizon':h,'model_id':fam+str(h),'recipe_id':'fixed-'+fam,'feature_id':self.ctx['feature_id'],'feature_order':M['FEATURE_ORDER'],'path':str(model),'sha256':M['sha'](model)})
    self.rows.extend({'stock_id':s,'signal_session':days[0],'family':fam,'horizon':h,'score':.1 if s=='SYNTH_A' else .2} for s in ['SYNTH_A','SYNTH_B'])
  self.exec=days[1]+'T07:30:00+08:00';self.pub=days[1]+'T07:31:00+08:00'
 def publish(self,**kw):return M['publish_synthetic'](self.root/'predictions',self.ctx,self.models,self.features,self.rows,executed_at=kw.get('executed_at',self.exec),published_at=kw.get('published_at',self.pub))
 def test_preopen_immutable_publication(self):
  r=self.publish();p=M['read_prediction'](r);self.assertEqual(p['payload']['timing'],'PREOPEN');self.assertEqual(p['payload']['label_status'],'PENDING_NOT_READ');self.assertFalse(p['research_admitted'])
 def test_recovery_before_open_and_after_open(self):
  r=self.publish(executed_at='2030-01-03T08:50:00+08:00',published_at='2030-01-03T08:59:59+08:00');self.assertEqual(M['read_prediction'](r)['payload']['timing'],'PREOPEN')
  with tempfile.TemporaryDirectory() as d:r=M['publish_synthetic'](d,self.ctx,self.models,self.features,self.rows,executed_at=self.exec,published_at='2030-01-03T09:00:00+08:00');self.assertEqual(M['read_prediction'](r)['payload']['timing'],'MISSED_PREOPEN');self.assertEqual(M['read_prediction'](r)['payload']['prediction_role'],'RESEARCH_ONLY')
 def test_qc_failure_and_unknown_calendar(self):
  for key,value in [('qc_status','FAIL'),('calendar_status','UNKNOWN')]:
   c=copy.deepcopy(self.ctx);c[key]=value
   with self.assertRaises(ValueError):M['publish_synthetic'](self.root/'bad',c,self.models,self.features,self.rows,executed_at=self.exec,published_at=self.pub)
 def test_missing_model_and_changed_feature_order(self):
  Path(self.models[0]['path']).unlink()
  with self.assertRaises(ValueError):self.publish()
 def test_changed_feature_order_refused(self):
  self.models[0]['feature_order']=list(reversed(M['FEATURE_ORDER']))
  with self.assertRaises(ValueError):self.publish()
 def test_mislabeled_calendar_open_refused(self):
  self.ctx['calendar'][1]['open_at']='2030-01-04T09:00:00+08:00'
  with self.assertRaises(ValueError):self.publish()
 def test_feature_frame_cannot_carry_labels(self):
  self.features[0]['future_label']=.99
  with self.assertRaises(ValueError):self.publish()
 def test_same_content_and_conflicting_duplicate(self):
  a=self.publish();before=Path(a['path']).read_bytes();self.assertEqual(a,self.publish());self.rows[0]['score']+=1
  with self.assertRaises(ValueError):self.publish()
  self.assertEqual(before,Path(a['path']).read_bytes())
 def test_labels_pending_then_mature_separately(self):
  r=self.publish();payload=M['read_prediction'](r)['payload'];label={'stock_id':'SYNTH_A','signal_session':self.ctx['signal_session'],'horizon':5,'label_policy_id':self.ctx['label_policy_id'],'prediction_sha256':r['sha256'],'exit_at':payload['planned_exits']['5'],'available_at':'2030-01-07T14:00:00+08:00','eligible':True,'value':.1}
  self.assertEqual(M['evaluation_gate'](r,label,now=self.pub)['status'],'PENDING_MATURITY');self.assertEqual(M['evaluation_gate'](r,label,now='2030-01-07T14:01:00+08:00')['status'],'ELIGIBLE_SYNTHETIC_JOIN')
  label['horizon']=10;self.assertEqual(M['evaluation_gate'](r,label,now='2030-01-07T14:01:00+08:00')['status'],'REJECTED')
  label['exit_at']=payload['planned_exits']['10'];label['available_at']='2030-01-12T14:00:00+08:00'
  self.assertEqual(M['evaluation_gate'](r,label,now='2030-01-07T14:01:00+08:00')['status'],'PENDING_MATURITY');self.assertEqual(M['evaluation_gate'](r,label,now='2030-01-12T14:01:00+08:00')['status'],'ELIGIBLE_SYNTHETIC_JOIN')
 def test_evaluation_cannot_precede_publication(self):
  r=self.publish(published_at='2030-01-08T09:01:00+08:00');p=M['read_prediction'](r)['payload'];label={'stock_id':'SYNTH_A','signal_session':self.ctx['signal_session'],'horizon':5,'label_policy_id':self.ctx['label_policy_id'],'prediction_sha256':r['sha256'],'exit_at':p['planned_exits']['5'],'available_at':'2030-01-07T14:00:00+08:00','eligible':True,'value':.1}
  self.assertEqual(M['evaluation_gate'](r,label,now='2030-01-07T14:01:00+08:00')['status'],'REJECTED')
 def test_backfill_never_claims_forward_vintage(self):
  self.ctx['capture_class']='HISTORICAL_BACKFILL';p=M['read_prediction'](self.publish())['payload'];self.assertEqual(p['capture_class'],'HISTORICAL_BACKFILL');self.assertFalse(p['verified_forward_vintage']);self.assertIsNone(p['source_publication_at'])
 def test_future_labels_do_not_change_prediction_keys(self):
  future={'SYNTH_A':.2};a=self.publish();future.clear();self.assertEqual(a,self.publish());self.assertEqual(len(M['read_prediction'](a)['payload']['predictions']),16)
 def test_future_data_and_real_mode_refused(self):
  self.ctx['qc_at']='2030-01-03T10:00:00+08:00'
  with self.assertRaises(ValueError):self.publish()
  self.ctx['qc_at']='2030-01-02T19:01:00+08:00';self.ctx['evidence_class']='REAL'
  with self.assertRaises(ValueError):self.publish()
if __name__=='__main__':unittest.main()
