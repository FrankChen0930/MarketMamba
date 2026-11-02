"""Offline v0.4 acceptance; zero GET/scheduler/model operations."""
import sys,tempfile,json,csv,io,copy,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental import v7_data_recovery as M
from V6.experimental.v7_manual_collector import source,archive
class Recovery(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.binding=source('11549');self.scope={'market':'TWSE','source':self.binding,'session':'2030-01-02'};self.row={'market':'TWSE','session':'2030-01-02','security_id':'SYNTH','basic_valid':False,'ohlc_valid':False,'missing_fields':['close']};self.universe={('TWSE','SYNTH')};self.policy={'status':'OFFLINE_REFERENCE_NOT_ACTIVATED','content_limit':1,'network_retry_limit':2,'physical_limit':4};self.ledger=M.AttemptLedger(self.root/'ledger',self.policy)
 def ev(self,kind):return {'kind':kind,'market':'TWSE','security_id':'SYNTH','session':'2030-01-02','evidence_ref':'SYNTHETIC_FIXTURE_ONLY:known-case','evidence_class':'SYNTHETIC'}
 def missing(self):return M.classify(self.row,self.universe)
 def plan(self):return M.decide(self.missing(),scope=self.scope,update_evidence='SYNTHETIC_REVISION_EXPECTATION',source_permitted=True,quota_available=True,ledger=self.ledger)
 def test_only_outside_universe_no_recovery(self):
  c=M.classify(self.row,set());self.assertEqual(c['state'],'OUT_OF_RESEARCH_UNIVERSE');self.assertFalse(M.decide(c,scope=self.scope,update_evidence='fixture',source_permitted=True,quota_available=True,ledger=self.ledger)['recovery_candidate'])
 def test_known_no_price_and_suspension_need_bound_evidence(self):
  for kind in ['LEGITIMATE_NO_PRICE','KNOWN_SUSPENSION']:
   self.assertEqual(M.classify(self.row,self.universe,self.ev(kind))['state'],kind)
  bad=self.ev('KNOWN_SUSPENSION');bad['session']='2030-01-03';self.assertEqual(M.classify(self.row,self.universe,bad)['state'],'UNKNOWN_MISSING')
 def test_important_sporadic_missing_candidate_requires_update_evidence(self):
  self.assertTrue(self.plan()['recovery_candidate']);self.assertFalse(M.decide(self.missing(),scope=self.scope,update_evidence=None,source_permitted=True,quota_available=True,ledger=self.ledger)['recovery_candidate'])
 def test_persistent_unknown_never_creates_unbounded_attempts(self):
  ref=self.ledger.reserve(self.scope,'CONTENT_RECOVERY','first',decision=self.plan());self.ledger.finish(ref,{'status':'STILL_MISSING_STOP'})
  self.assertEqual(M.decide(self.missing(),scope=self.scope,update_evidence='fixture',source_permitted=True,quota_available=True,ledger=self.ledger)['state'],'UNRECOVERABLE_AFTER_POLICY_LIMIT')
  with self.assertRaises(ValueError):self.ledger.reserve(self.scope,'CONTENT_RECOVERY','second',decision=self.plan())
 def test_content_and_network_attempts_separate_and_idempotent(self):
  first=self.ledger.reserve(self.scope,'NORMAL_CAPTURE','normal');self.assertEqual(first,self.ledger.reserve(self.scope,'NORMAL_CAPTURE','normal'));self.ledger.finish(first,{'status':'NETWORK_FAILURE'})
  retry=self.ledger.reserve(self.scope,'NETWORK_RETRY','retry',retry_of=first);self.ledger.finish(retry,{'status':'NETWORK_FAILURE'});retry2=self.ledger.reserve(self.scope,'NETWORK_RETRY','retry2',retry_of=retry);self.ledger.finish(retry2,{'status':'SUCCESS'})
  recovery=self.ledger.reserve(self.scope,'CONTENT_RECOVERY','content',decision=self.plan());self.assertEqual(self.ledger.counts(self.scope),{'NORMAL_CAPTURE':1,'CONTENT_RECOVERY':1,'NETWORK_RETRY':2})
  with self.assertRaises(ValueError):self.ledger.reserve(self.scope,'NETWORK_RETRY','extra',retry_of=retry)
 def test_failure_parent_has_one_retry_child_even_pending_or_success(self):
  for status in ['PENDING','SUCCESS']:
   with self.subTest(status=status):
    ledger=M.AttemptLedger(self.root/('parent-'+status),self.policy);parent=ledger.reserve(self.scope,'NORMAL_CAPTURE','normal');ledger.finish(parent,{'status':'NETWORK_FAILURE'});child=ledger.reserve(self.scope,'NETWORK_RETRY','child',retry_of=parent)
    if status=='SUCCESS':ledger.finish(child,{'status':'SUCCESS'})
    self.assertEqual(child,ledger.reserve(self.scope,'NETWORK_RETRY','child',retry_of=parent))
    altered_ref={**parent,'irrelevant_annotation':'must_not_allow_second_child'}
    with self.assertRaises(ValueError):ledger.reserve(self.scope,'NETWORK_RETRY','second-child',retry_of=altered_ref)
 def test_rate_or_source_stop_and_unknown_provider_halt(self):
  for code in [402,403,429]:
   d=M.decide(self.missing(),scope=self.scope,update_evidence='fixture',source_permitted=True,quota_available=True,ledger=self.ledger,http_status=code);self.assertFalse(d['recovery_candidate']);self.assertEqual(d['state'],'SOURCE_LEVEL_FAILURE')
  d=M.decide(self.missing(),scope=self.scope,update_evidence='fixture',source_permitted=True,quota_available=False,ledger=self.ledger);self.assertFalse(d['recovery_candidate'])
  other=copy.deepcopy(self.scope);other['source']['resource_url']='https://unapproved.invalid'
  with self.assertRaises(ValueError):self.ledger.reserve(other,'NORMAL_CAPTURE','bad')
 def test_recovery_decision_cannot_use_other_session(self):
  scope={**self.scope,'session':'2030-01-03'}
  with self.assertRaises(ValueError):M.decide(self.missing(),scope=scope,update_evidence='fixture',source_permitted=True,quota_available=True,ledger=self.ledger)
 def test_source_halt_persists_across_sessions(self):
  claim=self.ledger.reserve(self.scope,'NORMAL_CAPTURE','halted');self.ledger.finish(claim,{'status':'SOURCE_HALT','http_status':429})
  other={**self.scope,'session':'2030-01-03'}
  with self.assertRaises(ValueError):self.ledger.reserve(other,'NORMAL_CAPTURE','new-day')
 def snapshots(self,change=False,session='2030-01-02'):
  b={'market':'TWSE','source':self.binding,'session':'2030-01-02','raw_sha256':'a'*64,'missing_required':['SYNTH'],'qc_fatal':False};a={**b,'session':session,'raw_sha256':'b'*64 if change else 'a'*64,'missing_required':[] if change else ['SYNTH']};return b,a
 def test_same_content_not_repaired(self):self.assertEqual(M.recovery_outcome(*self.snapshots())['status'],'SAME_CONTENT_NO_REPAIR_STOP')
 def test_changed_content_repaired_or_still_missing_stops(self):
  b,a=self.snapshots(True);self.assertEqual(M.recovery_outcome(b,a)['repaired'],['SYNTH']);a['missing_required']=['SYNTH'];self.assertEqual(M.recovery_outcome(b,a)['status'],'STILL_MISSING_STOP')
 def test_different_session_rejected(self):self.assertEqual(M.recovery_outcome(*self.snapshots(True,'2030-01-03'))['status'],'SESSION_MISMATCH_STOP')
 def test_volume_only_not_required_for_two_features(self):
  r={**self.row,'ohlc_valid':True,'missing_fields':['volume']};c=M.classify(r,self.universe);self.assertFalse(c['blocks_features']);self.assertFalse(M.decide(c,scope=self.scope,update_evidence='fixture',source_permitted=True,quota_available=True,ledger=self.ledger)['recovery_candidate'])
 def test_immutable_new_vintage_offline_not_forward(self):
  body=b'\xef\xbb\xbf'+','.join(M.CSV_HEADERS['11549']).encode()+b'\n1190102,SYNTH,synthetic,1000,10000,10,11,9,--,0,1\n';r1=archive(self.root/'snap','11549',body,retrieved_at='2030-01-02T19:00:00+08:00',http_status=200,headers={'content-type':'text/csv'},mode='OFFLINE_REPLAY');r2=archive(self.root/'snap','11549',body.replace(b'9,--',b'9,10'),retrieved_at='2030-01-03T07:00:00+08:00',http_status=200,headers={'content-type':'text/csv'},mode='OFFLINE_REPLAY');self.assertNotEqual(r1['raw_sha256'],r2['raw_sha256']);self.assertTrue((self.root/'snap'/r1['raw_path']).exists());self.assertFalse(r2['verified_forward_vintage'])
 def test_full_raw_missing_not_exception_cap(self):
  h=','.join(M.CSV_HEADERS['11549'])+'\n';body=h+''.join(f'1190102,S{i},synthetic,0,0,--,--,--,--,0,0\n' for i in range(25));rows,q=M.parse_csv(body.encode(),'11549');self.assertEqual(sum(not r['basic_valid'] for r in rows),25);self.assertEqual(len(q['exceptions']),20)
if __name__=='__main__':unittest.main()
