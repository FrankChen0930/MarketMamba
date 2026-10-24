"""Offline operational guards. Synthetic clocks/announcements are not real PIT."""
import copy
import unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_capture_readiness import calendar_status, capture_status


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        self.cal={'schema':'v7-capture-calendar-proposal-v1','market':'SYNTHETIC_TW',
                  'coverage_start':'2026-10-22','coverage_end':'2026-10-27',
                  'days':[]}
        for day,state in [('22','TRADED'),('23','CLOSED'),('24','CLOSED'),('25','CLOSED'),('26','CLOSED'),('27','SCHEDULED')]:
            self.cal['days'].append({'session':'2026-10-'+day,'state':state,'evidence_ref':'synthetic-announcement',
                                    'known_at':'2026-10-22T20:10:00+08:00','revision_id':'v1','supersedes':None})
    def check(self,day='2026-10-24',**kw):
        return calendar_status(self.cal,canonical_hash(self.cal),day,'2026-10-24T21:00:00+08:00',**kw)
    def test_required_closure_window(self):
        for day in range(23,27):
            r=self.check('2026-10-'+str(day));self.assertEqual(r['status'],'NON_TRADING_DAY');self.assertFalse(r['missing_session'])
        self.assertEqual(self.check('2026-10-22')['status'],'MISSING_TRADING_SESSION')
        self.assertEqual(self.check('2026-10-22',captured_sessions=['2026-10-22'])['status'],'CAPTURE_PRESENT')
        r=self.check();self.assertEqual(r['last_actual_session'],'2026-10-22');self.assertEqual(r['next_scheduled_session'],'2026-10-27');self.assertFalse(r['next_actual_verified'])
    def test_unknown_or_outside_calendar_blocks(self):
        for day in ['2026-10-21','2026-10-28']:
            self.assertEqual(self.check(day)['status'],'BLOCKED')
        self.cal['days'][1]['evidence_ref']=None
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_changed_calendar_pin_rejected(self):
        pin=canonical_hash(self.cal);self.cal['days'][1]['state']='TRADED'
        self.assertEqual(calendar_status(self.cal,pin,'2026-10-23','2026-10-24T21:00:00+08:00')['status'],'BLOCKED')
    def test_revisions_are_asof_and_must_link(self):
        rev={**self.cal['days'][-1],'state':'CLOSED','revision_id':'v2','supersedes':'v1','known_at':'2026-10-27T08:00:00+08:00'}
        self.cal['days'].append(rev)
        self.assertEqual(self.check()['next_scheduled_session'],'2026-10-27')
        r=calendar_status(self.cal,canonical_hash(self.cal),'2026-10-27','2026-10-27T08:01:00+08:00')
        self.assertEqual(r['status'],'NON_TRADING_DAY');self.assertIsNone(r['next_scheduled_session'])
        rev['supersedes']=None;self.assertEqual(self.check()['status'],'BLOCKED')
    def test_calendar_conflict_unknown_timezone_and_pending_session(self):
        self.assertEqual(self.check('2026-10-27')['status'],'SCHEDULED_NOT_YET_VERIFIED')
        self.cal['days'].append(copy.deepcopy(self.cal['days'][0]))
        self.assertEqual(self.check()['status'],'BLOCKED')
        self.cal['days'].pop();self.cal['days'][0]['known_at']='2026-10-22'
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_missing_day_and_other_market_do_not_inherit(self):
        self.cal['days'].pop(1)
        self.assertEqual(self.check()['status'],'BLOCKED')
        self.cal['market']=None
        self.assertEqual(self.check()['status'],'BLOCKED')

    def test_invalid_captured_identity_does_not_report_present(self):
        self.assertEqual(self.check('2026-10-22',captured_sessions='2026-10-22')['status'],'BLOCKED')

    def test_real_storage_write_failure_never_success(self):
        from unittest.mock import patch
        from V6.experimental.v7_vintage_dryrun_test import VintageDryrunTest
        fixture=VintageDryrunTest('test_qc_failure_never_success');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch('V6.experimental.v7_vintage_dryrun._atomic_bytes',side_effect=OSError('synthetic disk full')):
            with self.assertRaises(OSError):fixture.publish()
        self.assertEqual(list(fixture.root.glob('records/**/*.json')),[])

    def test_storage_and_backup_are_separate_no_success_on_failure(self):
        for storage,backup,want in [(True,'PENDING','STORED_BACKUP_PENDING'),(True,'FAILED','FAILED'),(False,'VERIFIED','FAILED'),(True,'VERIFIED','STORED_AND_BACKUP_VERIFIED')]:
            r=capture_status(storage_verified=storage,backup_status=backup,qc_status='PASS')
            self.assertEqual(r['status'],want);self.assertFalse(r['collector_activated']);self.assertFalse(r['research_admitted'])
        self.assertEqual(capture_status(storage_verified=True,backup_status='VERIFIED',qc_status='MISSING_OHLCV_NO_FILL')['status'],'QC_REJECTED')
        self.assertEqual(capture_status(storage_verified=1,backup_status='VERIFIED',qc_status='PASS')['status'],'FAILED')

if __name__=='__main__':unittest.main()
