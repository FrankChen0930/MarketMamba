"""Synthetic-only persistence tests; all writes are owned temporary namespaces."""
from pathlib import Path
import copy,json,unittest
from V6.experimental.v7_forward_vintage_test import ForwardVintagePrototypeTest
from V6.experimental.v7_source_snapshot import RequestSpec
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_vintage_dryrun import publish_capture,verify_capture

class VintageDryrunTest(unittest.TestCase):
    def setUp(self):
        self.f=ForwardVintagePrototypeTest('test_timely_only_prototype_and_zero_bytes_preserved');self.f.setUp();self.addCleanup(self.f.tearDown)
        self.root=self.f.root/'dryrun';self.intent=copy.deepcopy(self.f.intent);self.obs=copy.deepcopy(self.f.record);self.obs['stale_status']='FRESH'
        self.request=RequestSpec('GET','https://example.org/synthetic',params={'date':'20240102'})
        self.binding={'schema':'v7-vintage-binding-synthetic-v1','profile':'SYNTHETIC_ONLY','track':'B_FORWARD_VINTAGE','source_id':'SYNTHETIC','dataset_id':'synthetic-dataset','endpoint':self.request.url,'license_id':'synthetic-license-v1','license_evidence_sha256':'a'*64,'collector_enabled':False}
        self.body=b'{"date":"20240102","volume":0}'
    def publish(self,body='default',**kw):
        return publish_capture(self.root,self.binding,canonical_hash(self.binding),self.intent,canonical_hash(self.intent),self.obs,self.request,self.body if body=='default' else body,**kw)
    def verify(self,r):return verify_capture(self.root,r['key'],r['record_sha256'])
    def test_publish_retry_dedup_and_zero_preserved(self):
        first=self.publish();self.assertEqual(first['status'],'ARCHIVED_SYNTHETIC');self.assertEqual(first['capture_class'],'ORIGINAL_FORWARD_CAPTURE');again=self.publish();self.assertEqual(again['status'],'ALREADY_PUBLISHED');self.assertEqual(first['key'],again['key']);self.assertEqual(len(list((self.root/'records').rglob('*.json'))),1);self.assertEqual(self.verify(first)['status'],'BYTES_VERIFIED_SYNTHETIC');self.assertFalse(first['research_admitted'])
    def test_changed_content_creates_revision_retains_original(self):
        first=self.publish();self.obs.update(prior_vintage_sha256=first['raw_sha256'],request_started_at='2024-01-02T16:01:00+08:00',retrieved_at='2024-01-02T16:02:00+08:00',qc_at='2024-01-02T16:03:00+08:00');second=self.publish(body=b'{"date":"20240102","volume":1}')
        self.assertEqual(second['capture_class'],'REVISION_CAPTURE');self.assertNotEqual(first['key'],second['key']);self.assertEqual(self.verify(first)['status'],'BYTES_VERIFIED_SYNTHETIC');self.assertEqual(self.verify(second)['status'],'BYTES_VERIFIED_SYNTHETIC')
    def test_new_content_without_previous_link_refused(self):
        self.publish();self.assertEqual(self.publish(body=b'changed')['status'],'FAILED')
    def test_interruption_before_record_retry_recoverable(self):
        r=self.publish(interrupt_at='before_record');self.assertEqual(r['status'],'FAILED');self.assertFalse(list((self.root/'records').rglob('*.json')))
        self.assertEqual(self.publish()['status'],'ARCHIVED_SYNTHETIC')
    def test_qc_failure_never_success(self):
        self.obs['qc_status']='FAIL';self.assertEqual(self.publish()['status'],'FAILED');self.assertFalse(list((self.root/'records').rglob('*.json')))
    def test_missing_retained_then_backfill_has_own_identity(self):
        missing=self.publish(body=None);self.assertEqual(missing['capture_class'],'MISSING_CAPTURE');self.assertFalse(missing['research_admitted'])
        self.intent['acquisition_mode']='BACKFILL';self.obs['acquisition_mode']='BACKFILL';back=self.publish();self.assertEqual(back['capture_class'],'HISTORICAL_BACKFILL');self.assertGreaterEqual(len(list((self.root/'attempts').glob('*.json'))),2)
    def test_late_and_recovery_not_timely(self):
        self.obs['qc_at']='2024-01-03T07:31:00+08:00';self.assertEqual(self.publish()['capture_class'],'LATE_CAPTURE')
    def test_source_license_identity_change_cannot_reuse_archive(self):
        self.publish();self.binding['license_id']='different';self.assertEqual(self.publish()['status'],'FAILED')
    def test_missing_snapshot_not_restorable_and_no_overwrite_repair(self):
        r=self.publish();next((self.root/'snapshots'/'raw').iterdir()).unlink()
        self.assertEqual(self.verify(r)['status'],'NOT_RESTORABLE');self.assertEqual(self.publish()['status'],'FAILED')
    def test_corrupt_raw_not_repaired(self):
        r=self.publish();raw=next((self.root/'snapshots'/'raw').iterdir());raw.write_bytes(b'corrupt')
        self.assertEqual(self.verify(r)['status'],'NOT_RESTORABLE');self.assertEqual(self.publish()['status'],'FAILED');self.assertEqual(raw.read_bytes(),b'corrupt')
    def test_archive_creation_not_historical_time_and_publication_not_imputed(self):
        self.obs['published_at']=None;self.obs['publication_evidence_ref']=None;r=self.publish();record=json.loads((self.root/r['record_relpath']).read_text())
        self.assertIsNone(record['observation']['published_at']);self.assertGreater(record['archive_created_at'],'2025-01-01');self.assertNotEqual(record['archive_created_at'],self.obs['retrieved_at']);self.assertFalse(record['historical_asof_verified'])
    def test_real_profile_and_cross_track_rejected(self):
        for key,value in [('profile','LIVE'),('track','A_EXPLORATORY_HISTORICAL')]:
            self.binding[key]=value;self.assertEqual(self.publish()['status'],'FAILED');self.binding[key]='SYNTHETIC_ONLY' if key=='profile' else 'B_FORWARD_VINTAGE'
    def test_duplicate_qc_failure_never_reports_current_attempt_success(self):
        self.publish();self.obs['qc_status']='FAIL';self.assertEqual(self.publish()['status'],'FAILED')
    def test_late_duplicate_keeps_first_but_marks_current_attempt_late(self):
        first=self.publish();self.obs['qc_at']='2024-01-03T08:30:00+08:00';r=self.publish()
        self.assertEqual(r['status'],'ALREADY_PUBLISHED');self.assertEqual(r['attempt_capture_class'],'LATE_CAPTURE');self.assertEqual(r['first_capture_class'],'ORIGINAL_FORWARD_CAPTURE')
    def test_unknown_publication_never_claims_qualified_original_forward(self):
        self.obs['published_at']=None;self.obs['publication_evidence_ref']=None
        self.assertEqual(self.publish()['capture_class'],'UNQUALIFIED_CAPTURE')

    def test_stale_or_unknown_freshness_stored_but_not_qualified(self):
        for value in ('STALE','UNKNOWN'):
            self.obs['stale_status']=value;r=self.publish()
            self.assertIn(r.get('capture_class',r.get('attempt_capture_class')),('UNQUALIFIED_CAPTURE',))

    def test_missing_or_corrupt_intent_not_restorable_no_retry_repair(self):
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt):
                self.root=self.f.root/('intent-fault-'+str(corrupt));r=self.publish();p=next((self.root/'intents').glob('*.json'))
                if corrupt:p.write_bytes(b'{}')
                else:p.unlink()
                self.assertEqual(self.verify(r)['status'],'NOT_RESTORABLE');self.assertEqual(self.publish()['status'],'FAILED')
                if corrupt:self.assertEqual(p.read_bytes(),b'{}')
                else:self.assertFalse(p.exists())
    def test_missing_binding_no_retry_reconstruction(self):
        r=self.publish();p=self.root/'binding.json';p.unlink()
        self.assertEqual(self.verify(r)['status'],'NOT_RESTORABLE');self.assertEqual(self.publish()['status'],'FAILED');self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
