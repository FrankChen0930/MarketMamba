"""Synthetic timing/identity fixtures; no historical PIT or scheduler acceptance."""
import copy,json,tempfile,unittest
from pathlib import Path
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_forward_vintage import inspect_record

class ForwardVintagePrototypeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        request=RequestSpec('GET','https://example.org/synthetic',params={'date':'20240102'})
        path=snapshot_response(request,b'{"date":"20240102","volume":0}',self.root,retrieved_at='2024-01-02T16:00:00+08:00',parser_version='synthetic-v1',content_type='application/json')
        self.intent={'schema':'v7-forward-intent-proposal-v1','status':'PROPOSED_NOT_ADMITTED','source_id':'SYNTHETIC','session':'2024-01-02','request_id':request.request_id,'acquisition_mode':'FORWARD','planned_at':'2024-01-02T14:00:00+08:00','candidate_freeze_at':'2024-01-03T07:30:00+08:00','decision_deadline_at':'2024-01-03T08:00:00+08:00'}
        self.record={'schema':'v7-forward-observation-proposal-v1','status':'CAPTURED','source_id':'SYNTHETIC','session':'2024-01-02','acquisition_mode':'FORWARD','request_started_at':'2024-01-02T15:59:00+08:00','retrieved_at':'2024-01-02T16:00:00+08:00','observation_at':'2024-01-02T13:30:00+08:00','published_at':'2024-01-02T15:00:00+08:00','source_available_at':'2024-01-02T15:01:00+08:00','publication_evidence_ref':'synthetic-clock-fixture','qc_at':'2024-01-02T16:01:00+08:00','qc_status':'PASS','qc_evidence_sha256':'a'*64,'manifest_relpath':str(path.relative_to(self.root)),'manifest_sha256':file_sha256(path),'raw_sha256':json.loads(path.read_text())['response']['sha256'],'revision_id':'synthetic-original','prior_vintage_sha256':None}
        self.pin=canonical_hash(self.record);self.ipin=canonical_hash(self.intent)
    def tearDown(self):self.tmp.cleanup()
    def run_case(self,record=None,intent=None,repin=False):
        r=self.record if record is None else record;i=self.intent if intent is None else intent
        return inspect_record(i,r,self.root,expected_intent_sha256=canonical_hash(i) if repin else self.ipin,expected_record_sha256=canonical_hash(r) if repin else self.pin)
    def test_timely_only_prototype_and_zero_bytes_preserved(self):
        r=self.run_case();self.assertEqual(r['status'],'PROTOTYPE_TIMELY');self.assertFalse(r['research_admitted']);self.assertFalse(r['historical_asof_verified'])
        self.assertIn(b'"volume":0',next((self.root/'raw').iterdir()).read_bytes())
    def test_missing_source_stays_missing_without_payload(self):
        r={**self.record,'status':'MISSING','manifest_relpath':None,'retrieved_at':None,'qc_status':'UNKNOWN'}
        result=self.run_case(r,repin=True);self.assertEqual(result['status'],'MISSING');self.assertFalse(result['candidate_time_eligible'])
    def test_recovery_and_backfill_cannot_be_timely_even_repin(self):
        for mode in ('RECOVERY','BACKFILL'):
            r={**self.record,'acquisition_mode':mode};i={**self.intent,'acquisition_mode':mode}
            self.assertEqual(self.run_case(r,i,True)['status'],'RETROSPECTIVE_ONLY')
    def test_late_qc_never_rewrites_frozen_snapshot(self):
        r={**self.record,'qc_at':'2024-01-03T07:31:00+08:00'}
        self.assertEqual(self.run_case(r,repin=True)['status'],'LATE')
    def test_missing_publication_or_naive_clock_unknown(self):
        for patch in ({'published_at':None},{'publication_evidence_ref':None},{'observation_at':'2024-01-02T13:30:00'}):
            self.assertEqual(self.run_case({**self.record,**patch},repin=True)['status'],'UNKNOWN')
    def test_changed_vintage_or_mode_rejected_by_original_pin(self):
        for patch in ({'revision_id':'later-revision'},{'raw_sha256':'b'*64},{'acquisition_mode':'BACKFILL'}):
            self.assertEqual(self.run_case({**self.record,**patch})['status'],'REJECTED')
    def test_backdated_receipt_cannot_override_manifest_even_new_record_pin(self):
        r={**self.record,'retrieved_at':'2024-01-02T15:59:30+08:00'}
        self.assertIn('SNAPSHOT_IDENTITY_MISMATCH',self.run_case(r,repin=True)['reasons'])
    def test_corrupt_or_missing_raw_and_traversal_rejected(self):
        r={**self.record,'manifest_relpath':'../outside.json'}
        self.assertEqual(self.run_case(r,repin=True)['status'],'REJECTED')
        next((self.root/'raw').iterdir()).write_bytes(b'corrupt')
        self.assertEqual(self.run_case()['status'],'REJECTED')
    def test_failed_qc_and_late_created_intent_fail_closed(self):
        self.assertEqual(self.run_case({**self.record,'qc_status':'FAIL'},repin=True)['status'],'REJECTED')
        i={**self.intent,'planned_at':'2024-01-03T09:00:00+08:00'}
        self.assertEqual(self.run_case(intent=i,repin=True)['status'],'REJECTED')

    def test_malformed_manifest_containers_fail_closed(self):
        for value in (None,[],{'response':None},{'response':[]}):
            with self.subTest(value=value):
                path=self.root/self.record['manifest_relpath'];path.write_text(json.dumps(value))
                r={**self.record,'manifest_sha256':file_sha256(path)}
                self.assertEqual(self.run_case(r,repin=True)['status'],'REJECTED')

if __name__=='__main__':unittest.main()
