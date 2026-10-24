"""Tiny SYNTHETIC-only capture adapter over existing immutable snapshot storage.

No network/scheduler/registry/trainer. W09 locks/path guards/JSON/artifact verifier
and v1 no-clobber bytes publisher are reused. W09 study bundles require predictions
and metrics, so are not mislabelled as raw-vintage bundles. Trusted stable local
root only; caller-declared clocks/QC do not prove real forward or PIT capture.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,uuid
from V6.experimental.v7_source_snapshot import snapshot_response,verify_snapshot,_atomic_bytes
from V6.experimental.v7_research_artifacts import _safe,_lock,strict_json,verify_artifact,_relative
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
from V6.experimental.v7_forward_vintage import inspect_record

SCHEMA='v7-vintage-dryrun-record-v1'
def encoded(x):return (json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode()
def stamp():return datetime.now(timezone.utc).isoformat()
def result(status,**kw):return {'status':status,'research_admitted':False,'historical_asof_verified':False,'collector_enabled':False,**kw}

def verify_capture(root,key,record_sha256):
    try:
        root=_safe(root)
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{64}/[0-9a-f]{64}',key):raise ValueError('INVALID_VINTAGE_KEY')
        rel='records/'+key+'.json';path=_safe(root/rel)
        if not path.is_file():raise ValueError('MISSING_VINTAGE_RECORD')
        verify_artifact(root,{'name':'vintage','path':rel,'storage':'BUNDLE','sha256':record_sha256,'bytes':path.stat().st_size,'format':'JSON','schema_id':SCHEMA},())
        e=strict_json(path.read_bytes())
        if not isinstance(e,dict) or e.get('profile')!='SYNTHETIC_ONLY' or e.get('research_admitted') is not False:raise ValueError('SYNTHETIC_RECORD_REQUIRED')
        binding=strict_json(_safe(root/'binding.json').read_bytes())
        if not isinstance(binding,dict) or binding.get('profile')!='SYNTHETIC_ONLY' or canonical_hash(binding)!=e['binding_sha256']:raise ValueError('SOURCE_LICENSE_IDENTITY_MISMATCH')
        if not re.fullmatch('[0-9a-f]{64}',str(e.get('intent_sha256',''))):raise ValueError('INTENT_IDENTITY_INVALID')
        intent=strict_json(_safe(root/'intents'/(e['intent_sha256']+'.json')).read_bytes())
        if not isinstance(intent,dict) or canonical_hash(intent)!=e['intent_sha256']:raise ValueError('INTENT_BYTES_MISMATCH')
        observation=e.get('observation')
        if not isinstance(observation,dict) or intent.get('source_id')!=binding.get('source_id') or observation.get('source_id')!=intent.get('source_id') or observation.get('session')!=intent.get('session') or observation.get('acquisition_mode')!=intent.get('acquisition_mode'):raise ValueError('INTENT_RECORD_BINDING_MISMATCH')
        if key.split('/')[0]!=canonical_hash({'binding_sha256':e['binding_sha256'],'session':intent['session']}):raise ValueError('STREAM_SESSION_MISMATCH')
        relative=_relative(e['snapshot_manifest_relative'])
        if not relative.startswith('snapshots/manifests/'):raise ValueError('SNAPSHOT_LOCATION_MISMATCH')
        manifest=_safe(root/relative)
        if file_sha256(manifest)!=e['snapshot_manifest_sha256'] or not verify_snapshot(manifest):raise ValueError('MISSING_OR_CORRUPT_SNAPSHOT')
        m=strict_json(manifest.read_bytes())
        if m['response']['sha256']!=e['raw_sha256'] or e['raw_sha256']!=key.split('/')[1] or m['request_id']!=intent.get('request_id') or m['request']['url']!=binding['endpoint']:raise ValueError('RAW_REQUEST_IDENTITY_MISMATCH')
        if observation.get('manifest_sha256')!=e['snapshot_manifest_sha256'] or observation.get('manifest_relpath')!=str(manifest.relative_to(root/'snapshots')):raise ValueError('RECORD_SNAPSHOT_IDENTITY_MISMATCH')
        clocks=inspect_record(intent,observation,root/'snapshots',expected_intent_sha256=e['intent_sha256'],expected_record_sha256=canonical_hash(observation))
        if clocks['status'] in ('REJECTED','MISSING') or observation.get('qc_status')!='PASS':raise ValueError('RESTORED_INTENT_OR_QC_INVALID')
        return result('BYTES_VERIFIED_SYNTHETIC',record=e)
    except (OSError,KeyError,TypeError,ValueError):return result('NOT_RESTORABLE',reasons=['MISSING_CORRUPT_OR_MISMATCHED_VINTAGE_BYTES'])

def publish_capture(root,binding,binding_pin,intent,intent_pin,observation,request,body,*,interrupt_at=None):
    attempt_id=uuid.uuid4().hex;attempt_at=stamp();log=None;event=None
    try:
        if not isinstance(binding,dict) or canonical_hash(binding)!=binding_pin or binding.get('schema')!='v7-vintage-binding-synthetic-v1' or binding.get('profile')!='SYNTHETIC_ONLY' or binding.get('track')!='B_FORWARD_VINTAGE' or binding.get('collector_enabled') is not False:raise ValueError('SYNTHETIC_TRACK_BINDING_REQUIRED')
        for k in ('source_id','dataset_id','endpoint','license_id','license_evidence_sha256'):
            if not isinstance(binding.get(k),str) or not binding[k].strip():raise ValueError('SOURCE_LICENSE_IDENTITY_REQUIRED')
        if not isinstance(intent,dict) or canonical_hash(intent)!=intent_pin or intent.get('source_id')!=binding['source_id'] or request.url!=binding['endpoint'] or request.request_id!=intent.get('request_id'):raise ValueError('INTENT_SOURCE_REQUEST_MISMATCH')
        if not isinstance(observation,dict) or (body is not None and (not isinstance(body,bytes) or len(body)>65536)):raise ValueError('BOUNDED_SYNTHETIC_BYTES_REQUIRED')
        root=_safe(root);root.mkdir(parents=True,exist_ok=True)
        for folder in ('records','intents','attempts','snapshots'):_safe(root/folder).mkdir(exist_ok=True)
        log=root/'attempts'/(attempt_id+'.json')
        with _lock(root/'publication.lock'):
            if any((root/'records').rglob('*.json')) and not (root/'binding.json').is_file():raise ValueError('EXISTING_ARCHIVE_BINDING_MISSING_NO_REPAIR')
            _atomic_bytes(root/'binding.json',encoded(binding))
            event={'schema':'v7-vintage-dryrun-attempt-v1','profile':'SYNTHETIC_ONLY','archive_created_at':attempt_at,'intent_sha256':intent_pin,'binding_sha256':binding_pin,'session':intent.get('session'),'qc_status':observation.get('qc_status'),'stale_status':observation.get('stale_status','UNKNOWN'),'reported_retrieved_at':observation.get('retrieved_at'),'mode':intent.get('acquisition_mode'),'research_admitted':False}
            if body is None:
                _atomic_bytes(root/'intents'/(intent_pin+'.json'),encoded(intent))
                event.update(status='MISSING_CAPTURE');_atomic_bytes(log,encoded(event));return result('MISSING_CAPTURE',capture_class='MISSING_CAPTURE')
            digest=hashlib.sha256(body).hexdigest();stream=canonical_hash({'binding_sha256':binding_pin,'session':intent['session']});key=stream+'/'+digest;rel='records/'+key+'.json';dest=_safe(root/rel)
            existing=None
            if dest.exists():
                record_hash=file_sha256(dest);verified=verify_capture(root,key,record_hash)
                if verified['status']!='BYTES_VERIFIED_SYNTHETIC':raise ValueError('EXISTING_VINTAGE_NOT_RESTORABLE_NO_REPAIR')
                existing=verified['record']
            prior=observation.get('prior_vintage_sha256')
            if existing is None:
                if dest.parent.exists() and any(dest.parent.glob('*.json')) and prior is None:raise ValueError('NEW_CONTENT_REQUIRES_PREVIOUS_REFERENCE')
                if prior is not None:
                    if not re.fullmatch('[0-9a-f]{64}',str(prior)) or not (dest.parent/(prior+'.json')).is_file():raise ValueError('PRIOR_VINTAGE_NOT_PRESENT')
                    prior_key=stream+'/'+prior;prior_path=dest.parent/(prior+'.json')
                    if verify_capture(root,prior_key,file_sha256(prior_path))['status']!='BYTES_VERIFIED_SYNTHETIC':raise ValueError('PRIOR_VINTAGE_NOT_RESTORABLE')
            _atomic_bytes(root/'intents'/(intent_pin+'.json'),encoded(intent))
            manifest=snapshot_response(request,body,root/'snapshots',retrieved_at=observation['retrieved_at'],parser_version='window15-synthetic-dryrun-v1',content_type='application/json')
            record={**observation,'status':'CAPTURED','manifest_relpath':str(manifest.relative_to(root/'snapshots')),'manifest_sha256':file_sha256(manifest),'raw_sha256':digest}
            verdict=inspect_record(intent,record,root/'snapshots',expected_intent_sha256=intent_pin,expected_record_sha256=canonical_hash(record))
            if verdict['status'] in ('REJECTED','MISSING') or record.get('qc_status')!='PASS':raise ValueError('QC_OR_TIME_IDENTITY_FAILED:'+','.join(verdict['reasons']))
            if record.get('stale_status')!='FRESH':verdict={**verdict,'status':'UNKNOWN','candidate_time_eligible':False,'reasons':verdict['reasons']+['FRESHNESS_UNQUALIFIED']}
            mode=intent['acquisition_mode'];capture_class=('UNQUALIFIED_CAPTURE' if verdict['status']=='UNKNOWN' else 'HISTORICAL_BACKFILL' if mode=='BACKFILL' else 'LATE_CAPTURE' if mode=='RECOVERY' or verdict['status']=='LATE' else 'REVISION_CAPTURE' if prior is not None else 'ORIGINAL_FORWARD_CAPTURE')
            if existing is not None:
                event.update(status='ALREADY_PUBLISHED',raw_sha256=digest,attempt_capture_class=capture_class)
                _atomic_bytes(log,encoded(event))
                return result('ALREADY_PUBLISHED',key=key,record_sha256=record_hash,raw_sha256=digest,first_capture_class=existing['capture_class'],attempt_capture_class=capture_class,attempt_mode=mode)
            # UNKNOWN clocks still archive bytes, but never claim qualified timely data.
            status='ARCHIVED_SYNTHETIC' if verdict['status']!='UNKNOWN' else 'ARCHIVED_UNQUALIFIED_SYNTHETIC'
            envelope={'schema':SCHEMA,'profile':'SYNTHETIC_ONLY','binding_sha256':binding_pin,'intent_sha256':intent_pin,'capture_class':capture_class,'candidate_clock_verdict':verdict,'observation':record,'raw_sha256':digest,'snapshot_manifest_relative':str(manifest.relative_to(root)),'snapshot_manifest_sha256':file_sha256(manifest),'archive_created_at':stamp(),'research_admitted':False,'historical_asof_verified':False}
            if interrupt_at=='before_record':raise ValueError('INTERRUPTED_BEFORE_RECORD_PUBLICATION')
            _atomic_bytes(dest,encoded(envelope));record_hash=file_sha256(dest)
            if verify_capture(root,key,record_hash)['status']!='BYTES_VERIFIED_SYNTHETIC':raise ValueError('PUBLICATION_VERIFICATION_FAILED')
            event.update(status=status,raw_sha256=digest,capture_class=capture_class);_atomic_bytes(log,encoded(event))
            return result(status,key=key,record_relpath=rel,record_sha256=record_hash,raw_sha256=digest,capture_class=capture_class)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        if log is not None:_atomic_bytes(log,encoded({**(event or {}),'schema':'v7-vintage-dryrun-attempt-v1','status':'FAILED','binding_sha256':binding_pin,'intent_sha256':intent_pin,'archive_created_at':attempt_at,'reasons':[str(exc)],'research_admitted':False}))
        return result('FAILED',reasons=[str(exc)])
