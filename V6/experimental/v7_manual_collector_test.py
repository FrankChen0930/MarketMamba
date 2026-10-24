"""Tiny offline fixtures only; no real GET budget consumed by tests."""
import json,tempfile,unittest
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import Mock,patch
from V6.experimental.v7_manual_collector import archive,run_one,restore,source,CSV_HEADERS

AT='2026-10-24T14:00:00+00:00'
def csv_bytes(dataset='11549',close='11'):
    rows={'11549':['1151022','2330','SYNTHETIC','1000','11000','10','12','9',close,'1','2'],
          '11371':['1151022','1234','SYNTHETIC',close,'1','10','12','9','1000','11000','2','10','11','100000','12','9']}
    return (','.join(CSV_HEADERS[dataset])+'\n'+','.join(rows[dataset])+'\n').encode()

class ManualCollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.store=self.root/'store';self.budget=self.root/'budget';self.receipts={}
    def save(self,body=None,**kw):
        body=csv_bytes() if body is None else body
        index=self.receipts.setdefault(body,len(self.receipts))
        receipt=(datetime.fromisoformat(AT)+timedelta(seconds=index)).isoformat()
        return archive(self.store,'11549',body,retrieved_at=receipt,http_status=200,headers={'Content-Type':'text/csv;charset=utf-8'},mode='OFFLINE_REPLAY',**kw)
    def test_both_source_schemas_and_no_research_promotion(self):
        for dataset in CSV_HEADERS:
            r=archive(self.store,dataset,csv_bytes(dataset),retrieved_at=AT,http_status=200,headers={'Content-Type':'text/csv'},mode='OFFLINE_REPLAY')
            self.assertEqual(r['status'],'RAW_CAPTURED');self.assertEqual(r['qc']['rows'],1);self.assertFalse(r['research_admitted']);self.assertFalse(r['daily_collector_enabled']);self.assertIsNone(r['published_at'])
    def test_idempotent_bytes_not_overwritten_and_revision_preserved(self):
        a=self.save();p=self.store/a['raw_path'];st=p.stat();b=self.save();self.assertEqual(a,b);self.assertEqual(p.stat().st_ino,st.st_ino);self.assertEqual(p.stat().st_mtime_ns,st.st_mtime_ns)
        c=self.save(csv_bytes(close='10'));self.assertNotEqual(a['raw_sha256'],c['raw_sha256']);self.assertTrue(p.exists());self.assertEqual(len(list((self.store/'snapshots/raw').glob('*'))),2)
    def test_independent_restore_preserves_bytes_and_row_counts(self):
        a=self.save();r=restore(self.store,a['record_path'],a['record_sha256'],self.root/'restore');self.assertEqual(r['status'],'RESTORED_BYTES_HASH_ROWS_VERIFIED');self.assertEqual(r['rows'],1)
    def test_missing_corrupt_hash_or_identity_not_restorable(self):
        a=self.save();p=self.store/a['raw_path'];p.unlink();self.assertEqual(restore(self.store,a['record_path'],a['record_sha256'],self.root/'restore')['status'],'NOT_RESTORABLE')
        self.assertEqual(restore(self.store,a['record_path'],'0'*64,self.root/'restore2')['status'],'NOT_RESTORABLE')
    def test_bad_qc_raw_saved_not_research_ready(self):
        for body in [b'<html>error</html>',csv_bytes(close='99'),csv_bytes()+csv_bytes().split(b'\n')[1]+b'\n']:
            with self.subTest(body=body[:10]):
                a=self.save(body);self.assertEqual(a['qc']['status'],'QC_FAILED');self.assertEqual(a['status'],'RAW_CAPTURED_WITH_QC_FAILURE');self.assertFalse(a['research_admitted'])
    def test_missing_no_zero_fill_and_nonfinite_rejected(self):
        a=self.save(csv_bytes(close='--'));self.assertEqual(a['qc']['counts']['MISSING_OHLCV_NO_FILL'],1)
        b=self.save(csv_bytes(close='NaN'));self.assertEqual(b['qc']['counts']['INVALID_OHLCV'],1)
    def test_budget_claimed_before_network_and_no_second_get(self):
        fetch=Mock(return_value=(csv_bytes(),200,{'Content-Type':'text/csv'},True))
        a=run_one('11549',self.store,self.budget,fetch=fetch);self.assertEqual(a['status'],'RAW_CAPTURED')
        b=run_one('11549',self.store,self.budget,fetch=fetch);self.assertEqual(b['status'],'BUDGET_EXHAUSTED');self.assertEqual(fetch.call_count,1)
    def test_403_429_redirect_timeout_never_retry_or_fallback(self):
        for i,status in enumerate([403,429,302]):
            fetch=Mock(return_value=(b'blocked',status,{'Content-Type':'text/html','Location':'https://unapproved.invalid/'},True))
            a=run_one('11549',self.root/str(i),self.root/('budget'+str(i)),fetch=fetch);self.assertEqual(a['status'],'HTTP_REJECTED_RAW_RETAINED');self.assertEqual(fetch.call_count,1)
        fetch=Mock(side_effect=TimeoutError('synthetic timeout'));a=run_one('11371',self.store,self.budget,fetch=fetch);self.assertEqual(a['status'],'FAILED');self.assertEqual(fetch.call_count,1)
        self.assertEqual(run_one('11371',self.store,self.budget,fetch=fetch)['status'],'BUDGET_EXHAUSTED')
    def test_unauthorized_source_refused_before_dispatch(self):
        fetch=Mock();a=run_one('OTHER',self.store,self.budget,fetch=fetch);self.assertEqual(a['status'],'FAILED');fetch.assert_not_called()
    def test_atomic_record_failure_then_offline_recovery(self):
        import V6.experimental.v7_manual_collector as m
        real=m._atomic_bytes
        def interrupt(path,body):
            if 'captures' in Path(path).parts:raise OSError('synthetic record publication interruption')
            return real(path,body)
        with patch.object(m,'_atomic_bytes',side_effect=interrupt):
            with self.assertRaises(OSError):self.save()
        self.assertFalse(list(self.store.glob('captures/**/*.json')))
        a=self.save();self.assertEqual(restore(self.store,a['record_path'],a['record_sha256'],self.root/'restore')['status'],'RESTORED_BYTES_HASH_ROWS_VERIFIED')
    def test_backfill_not_forward_and_calendar_wall_day_not_stale(self):
        a=archive(self.store,'11549',csv_bytes(),retrieved_at=AT,http_status=200,headers={'Content-Type':'text/csv;charset=utf-8'},mode='BACKFILL')
        self.assertEqual(a['capture_classification'],'HISTORICAL_BACKFILL_NOT_ORIGINAL_VINTAGE')
        b=self.save();self.assertEqual(b['freshness']['status'],'BOUNDED_LAST_ACTUAL_MATCH_NOT_ADMITTED');self.assertEqual(b['freshness']['calendar_verdict']['status'],'NON_TRADING_DAY');self.assertFalse(b['verified_forward_vintage'])
    def test_wrong_encoding_truncated_response_and_storage_failure(self):
        self.assertEqual(self.save(b'\xff')['qc']['status'],'QC_FAILED')
        a=archive(self.store,'11549',csv_bytes(),retrieved_at='2026-10-24T14:00:01+00:00',http_status=200,headers={'Content-Type':'text/csv'},mode='OFFLINE_REPLAY',complete=False)
        self.assertEqual(a['status'],'INCOMPLETE_RAW_RETAINED')
        self.store.mkdir(exist_ok=True);fetch=Mock(return_value=(csv_bytes(),200,{},True))
        with patch('V6.experimental.v7_manual_collector._atomic_bytes',side_effect=OSError('disk full')):
            self.assertEqual(run_one('11549',self.store,self.budget,fetch=fetch)['status'],'FAILED')
        fetch.assert_not_called()

class ReviewRegressionTest(unittest.TestCase):
    setUp=ManualCollectorTest.setUp
    save=ManualCollectorTest.save
    def test_review_missing_four_dash_retained_as_missing(self):
        a=self.save(csv_bytes(close='----'))
        self.assertEqual(a['qc']['counts'].get('MISSING_OHLCV_NO_FILL'),1)
    def test_review_content_length_short_read_incomplete(self):
        from unittest.mock import MagicMock
        import V6.experimental.v7_manual_collector as m
        response=MagicMock();response.__enter__.return_value=response;response.code=200
        response.headers={'Content-Type':'text/csv','Content-Length':str(len(csv_bytes())+100)};response.read.return_value=csv_bytes()
        with patch.object(m.urllib.request,'build_opener') as opener:
            opener.return_value.open.return_value=response
            self.assertFalse(m.fetch_once(m.URLS['11549'])[3])
    def test_review_absent_length_is_explicit_unknown(self):
        a=self.save()
        self.assertEqual(a['framing_evidence'],'UNKNOWN_NO_CONTENT_LENGTH')
        self.assertFalse(a['verified_forward_vintage'])

    def test_review_missing_committed_raw_never_silent_retry_repair(self):
        a=self.save();p=self.store/a['raw_path'];p.unlink()
        with self.assertRaises(ValueError):self.save()
        self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
