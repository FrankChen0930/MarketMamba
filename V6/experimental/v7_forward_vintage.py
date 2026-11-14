"""Read-only PROPOSED forward-collection record checks, not a collector.

Reuses v1 immutable snapshots and W07 aware clocks. Caller pins/QC/publication
claims are consistency inputs, not authenticated historical facts. Cannot certify
clock honesty, original publication, calendar/universe or research admission.
No network, scheduler, migration or snapshot rewrite is performed.
"""
from pathlib import Path
import re
from V6.experimental.v7_source_snapshot import verify_snapshot
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_research_temporal import instant
from V6.experimental.v7_research_artifacts import strict_json


def inspect_record(intent,record,archive_root,*,expected_intent_sha256,expected_record_sha256):
    def result(status,*reasons):
        return {'schema':'v7-forward-prototype-verdict-v1','status':status,'reasons':list(reasons),
                'candidate_time_eligible':status=='PROTOTYPE_TIMELY','research_admitted':False,
                'historical_asof_verified':False,'evidence_scope':'CALLER_DECLARED_CLOCKS_PINS_AND_LOCAL_SNAPSHOT_INTEGRITY_ONLY'}
    try:
        if not isinstance(intent,dict) or not isinstance(record,dict):return result('REJECTED','MALFORMED_RECORD')
        if canonical_hash(intent)!=expected_intent_sha256 or canonical_hash(record)!=expected_record_sha256:return result('REJECTED','PIN_MISMATCH')
        if intent.get('schema')!='v7-forward-intent-proposal-v1' or intent.get('status')!='PROPOSED_NOT_ADMITTED' or record.get('schema')!='v7-forward-observation-proposal-v1':return result('REJECTED','UNSUPPORTED_PROPOSED_SCHEMA')
        for key in ('source_id','session','acquisition_mode'):
            if not isinstance(intent.get(key),str) or not intent[key].strip() or record.get(key)!=intent[key]:return result('REJECTED','SOURCE_SESSION_MODE_MISMATCH')
        mode=record['acquisition_mode']
        if mode not in ('FORWARD','RECOVERY','BACKFILL'):return result('REJECTED','UNKNOWN_ACQUISITION_MODE')
        planned=instant(intent.get('planned_at'));freeze=instant(intent.get('candidate_freeze_at'));deadline=instant(intent.get('decision_deadline_at'))
        if planned>freeze or freeze>deadline:return result('REJECTED','INVALID_INTENT_WINDOW')
        if record.get('status')=='MISSING':return result('MISSING','NO_SOURCE_BYTES_NO_ZERO_FILL')
        if record.get('status')!='CAPTURED':return result('REJECTED','INVALID_CAPTURE_STATUS')
        root=Path(archive_root).absolute();rel=Path(record['manifest_relpath'])
        if root.resolve()!=root or rel.is_absolute() or len(rel.parts)!=2 or rel.parts[0]!='manifests' or '..' in rel.parts:return result('REJECTED','UNSAFE_SNAPSHOT_REFERENCE')
        path=root/rel
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file() or path.stat().st_size>65536:return result('REJECTED','MISSING_OR_OVERSIZED_MANIFEST')
        manifest=strict_json(path.read_bytes())
        if not isinstance(manifest,dict) or not isinstance(manifest.get('response'),dict):return result('REJECTED','MALFORMED_MANIFEST_CONTAINER')
        response=manifest['response']
        if type(response.get('bytes')) is not int or not 0<=response['bytes']<=2097152:return result('REJECTED','RAW_SIZE_OUTSIDE_PROTOTYPE_BOUND')
        if file_sha256(path)!=record.get('manifest_sha256') or not verify_snapshot(path):return result('REJECTED','SNAPSHOT_INTEGRITY_FAILED')
        if manifest['request_id']!=intent.get('request_id') or response['sha256']!=record.get('raw_sha256') or manifest['retrieved_at']!=record.get('retrieved_at'):return result('REJECTED','SNAPSHOT_IDENTITY_MISMATCH')
        if response['status_code']!=200:return result('REJECTED','SOURCE_HTTP_FAILURE')
        if not isinstance(record.get('revision_id'),str) or not record['revision_id'].strip():return result('UNKNOWN','MISSING_VINTAGE_IDENTITY')
        prior=record.get('prior_vintage_sha256')
        if prior is not None and not re.fullmatch('[0-9a-f]{64}',str(prior)):return result('REJECTED','INVALID_PRIOR_VINTAGE_REFERENCE')
        started=instant(record.get('request_started_at'));received=instant(record.get('retrieved_at'))
        if not planned<=started<=received:return result('REJECTED','INTENT_OR_REQUEST_CHRONOLOGY_INVALID')
        if mode!='FORWARD':return result('RETROSPECTIVE_ONLY','DECLARED_RECOVERY_OR_BACKFILL_NOT_TIMELY_FORWARD')
        if record.get('qc_status') is None or record.get('qc_status')=='UNKNOWN':return result('UNKNOWN','MISSING_QC_RESULT')
        if record['qc_status']!='PASS':return result('REJECTED','QC_NOT_PASS')
        if not re.fullmatch('[0-9a-f]{64}',str(record.get('qc_evidence_sha256',''))):return result('UNKNOWN','MISSING_QC_EVIDENCE_REFERENCE')
        if not isinstance(record.get('publication_evidence_ref'),str) or not record['publication_evidence_ref'].strip():return result('UNKNOWN','MISSING_PUBLICATION_EVIDENCE')
        observed=instant(record.get('observation_at'));published=instant(record.get('published_at'));available=instant(record.get('source_available_at'));qc=instant(record.get('qc_at'))
        if not observed<=published<=available<=received<=qc:return result('REJECTED','SOURCE_CLOCK_ORDER_INVALID')
        if qc>freeze:return result('LATE','RECEIPT_OR_QC_AFTER_FREEZE_RETAIN_RECOVERY_ONLY')
        return result('PROTOTYPE_TIMELY','NO_REAL_CALENDAR_SOURCE_AUTHENTICITY_OR_ADMISSION_ASSERTED')
    except (KeyError,TypeError,ValueError,OverflowError,OSError,RuntimeError):
        return result('UNKNOWN','INVALID_OR_UNAVAILABLE_EVIDENCE')
