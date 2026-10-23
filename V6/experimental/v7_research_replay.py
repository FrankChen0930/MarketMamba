"""Offline replay of W08 synthetic artifacts and JSON-only restore contracts.

Never trains, imports checkpoints through pickle, rewrites pointers, or admits
research. Reproducibility proofs are independent flags, not an ordinal ladder.
"""
from collections.abc import Mapping
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import argparse,json,platform,sys
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_e1_checkpoint import sha256,validate_state,POINTER_SCHEMA
from V6.experimental.v7_research_artifacts import verify_bundle,artifact_path,strict_json,_safe,_relative,BundleError
from V6.experimental.v7_research_evaluation import evaluate,prediction_hash
from V6.experimental.v7_research_comparison import compare
from V6.experimental.v7_research_temporal import make_pin,_sha,_text

ROOT=Path(__file__).resolve().parents[2]
REPLAY_SCHEMAS={'resolved_config':'v7-replay-config-v1','environment':'v7-replay-environment-v1',
 'data_split_eligibility':'v7-replay-split-v1','predictions':'v7-replay-predictions-v1',
 'daily_metrics':'v7-replay-expected-evaluations-v1','paired_comparison':'v7-ranking-comparison-v1',
 'rejection_ledger':'v7-replay-rejections-v1','exposure_ledger':'v7-replay-exposure-v1',
 'completion_evidence':'v7-replay-completion-v1','code_manifest':'v7-replay-code-manifest-v1',
 'dataset_snapshot':'v7-ranking-synthetic-input-evidence-v1'}


def runtime_identity():
    return {'python':platform.python_version(),'platform':sys.platform,
            'dependencies':{name:version(name) for name in ('torch','numpy','pandas')}}


def _capabilities():
    return {name:{'proven':False,'reasons':[reason]} for name,reason in [
       ('METADATA_VERIFIED','BUNDLE_NOT_VERIFIED'),('EVALUATION_REPLAYABLE','REPLAY_NOT_VERIFIED'),
       ('TRAINING_RESUMABLE','NO_REAL_RUNTIME_STATE_OR_TRAINING_VERIFIED'),
       ('FULL_REPRODUCTION_READY','NO_FULL_DATA_CODE_TRAINING_RUNTIME_REPRODUCTION_VERIFIED')]}


def replay_bundle(root,key,expected_manifest_sha256,*,external_roots=(),code_root=ROOT):
    out={'schema':'v7-offline-replay-evidence-v1','status':'REPLAY_REJECTED','research_admitted':False,
         'capabilities':_capabilities(),'reasons':[],'comparisons_match':False,
         'replayed_at':datetime.now(timezone.utc).isoformat(),'historical_live_prediction_proven':False,
         'manifest_sha256':expected_manifest_sha256,'run_key':key,'simulated_evaluation_times':[]}
    def fail(reason):return {**out,'reasons':[reason]}
    v=verify_bundle(root,key,expected_manifest_sha256,external_roots=external_roots)
    if v['status']!='ARTIFACTS_VERIFIED':return fail('BUNDLE_NOT_VERIFIED:'+','.join(v['reasons']))
    out['capabilities']['METADATA_VERIFIED']={'proven':True,'reasons':[]}
    out['publication_generated_at']=v['publication']['generated_at']
    try:
        m=v['manifest'];ident=m['identity'];descriptors={a['name']:a for a in m['artifacts']};payloads={}
        for name,schema in REPLAY_SCHEMAS.items():
            if name not in descriptors:raise BundleError('REPLAY_ARTIFACT_MISSING:'+name)
            a=descriptors[name]
            if a['format']!='JSON' or a['schema_id']!=schema:raise BundleError('REPLAY_SCHEMA_MISMATCH:'+name)
            payloads[name]=strict_json(artifact_path(v['bundle_path'],a,external_roots).read_bytes())
        raw=payloads['dataset_snapshot']
        if any(not isinstance(raw.get(k),Mapping) for k in ('scores','policy','fold','calendar','label_metadata','label_payloads')):raise BundleError('REPLAY_NESTED_CONTAINER_INVALID')
        if set(raw['scores'])!={'a','b'} or any(not isinstance(rows,list) or any(not isinstance(row,Mapping) for row in rows) for rows in raw['scores'].values()):raise BundleError('REPLAY_NESTED_CONTAINER_INVALID')
        policy=raw['policy'];fold=raw['fold'];cal=raw['calendar'];code=payloads['code_manifest']
        if payloads['resolved_config']['identity']!=ident:raise BundleError('CONFIG_RUN_IDENTITY_MISMATCH')
        if canonical_hash(raw)!=ident['data_sha256'] or canonical_hash(policy)!=ident['evaluation_policy_sha256']:raise BundleError('DATA_OR_POLICY_IDENTITY_MISMATCH')
        if canonical_hash(code)!=ident['code_manifest_sha256'] or code['commit']!=ident['code_commit']:raise BundleError('CODE_MANIFEST_IDENTITY_MISMATCH')
        if not isinstance(code['files'],Mapping) or not code['files']:raise BundleError('CODE_BYTES_MISSING')
        required_code={'V6/experimental/'+name+'.py' for name in ('v7_corrected_e5_train','v7_corrected_e5_replication','v7_research_temporal','v7_research_evaluation','v7_research_comparison','v7_temporal_contract')}
        if not required_code<=set(code['files']):raise BundleError('CODE_BYTES_MISSING')
        for path,h in code['files'].items():
            _relative(path)
            if not path.startswith('V6/experimental/') or not path.endswith('.py'):raise BundleError('CODE_PATH_NOT_ALLOWED')
            p=_safe(Path(code_root)/path)
            if not p.is_file() or not _sha(h) or sha256(p)!=h:raise BundleError('CODE_BYTES_MISMATCH')
        if payloads['environment']['runtime']!=runtime_identity():raise BundleError('ENVIRONMENT_IDENTITY_MISMATCH')
        split=payloads['data_split_eligibility']
        if any(split[name]!=canonical_hash(obj) for name,obj in [('fold_sha256',fold),('calendar_sha256',cal),('label_metadata_sha256',raw['label_metadata'])]):raise BundleError('SPLIT_OR_ELIGIBILITY_IDENTITY_MISMATCH')
        if any(ident[k]!=fold[k] for k in ('study_id','fold_id','origin_at')):raise BundleError('RUN_FOLD_IDENTITY_MISMATCH')
        if ident['feature_sha256']!=canonical_hash({'feature_contract_id':fold['feature_contract_id']}) or ident['label_sha256']!=canonical_hash({'label_contract_id':fold['label_contract_id'],'label_definition':fold['label_definition']}):raise BundleError('FEATURE_OR_LABEL_IDENTITY_MISMATCH')
        universe=sorted({r['stock_id'] for rows in raw['scores'].values() for r in rows})
        if ident['universe_sha256']!=canonical_hash(universe):raise BundleError('UNIVERSE_IDENTITY_MISMATCH')
        requests=payloads['predictions']['bundles'];expected=payloads['daily_metrics']['evaluations']
        if not isinstance(requests,Mapping) or not isinstance(expected,Mapping):raise BundleError('REPLAY_NESTED_CONTAINER_INVALID')
        if any(not isinstance(v,Mapping) or not isinstance(v.get('header'),Mapping) or not isinstance(v.get('rows'),list) for v in requests.values()):raise BundleError('REPLAY_NESTED_CONTAINER_INVALID')
        if any(not isinstance(v,Mapping) or not isinstance(v.get('records'),list) or any(not isinstance(r,Mapping) for r in v['records']) for v in expected.values()):raise BundleError('REPLAY_NESTED_CONTAINER_INVALID')
        if set(requests)!={'a','b'} or set(expected)!={'a','b'}:raise BundleError('PAIRED_RUN_REQUESTS_REQUIRED')
        primary=requests['a']['header']
        if any(primary[k]!=ident[k] for k in ('study_id','recipe_id','run_id','fold_id','origin_at','prediction_role','model_sha256')):raise BundleError('PREDICTION_RUN_IDENTITY_MISMATCH')
        evaluations={}
        for name,bundle in requests.items():
            # Original reference maps to retained exact bytes; no external source fetch.
            if bundle['header']['raw_predictions_ref']['sha256']!=descriptors['dataset_snapshot']['sha256']:raise BundleError('RAW_PREDICTION_REFERENCE_MISMATCH')
            if bundle['rows']!=raw['scores'][name]:raise BundleError('RAW_PREDICTION_ROWS_MISMATCH')
            ev=evaluate(bundle,prediction_hash(bundle),fold,cal,make_pin(fold,cal),policy,ident['evaluation_policy_sha256'],raw['label_metadata'],raw['label_payloads'])
            if ev['status']=='REJECTED':raise BundleError('ELIGIBILITY_RECHECK_FAILED:'+','.join(ev['reasons']))
            prior={(r['signal_session'],r['stock_id'],r['horizon']) for r in expected[name]['records']}
            now={(r['signal_session'],r['stock_id'],r['horizon']) for r in ev['records']}
            if not prior<=now:
                out['eligibility_rejections']=ev['rejections']
                raise BundleError('LABEL_ELIGIBILITY_RECHECK_MISMATCH:'+','.join(sorted({x for r in ev['rejections'] for x in r['reasons']})))
            evaluations[name]=ev
        out['simulated_evaluation_times']=[b['header']['evaluation_at'] for b in requests.values()]
        paired=compare(evaluations['a'],canonical_hash(evaluations['a']),evaluations['b'],canonical_hash(evaluations['b']))
        if paired['status']=='REJECTED':raise BundleError('PAIRED_ELIGIBILITY_RECHECK_FAILED:'+','.join(paired['reasons']))
        results={'daily_metrics':evaluations,'paired_comparison':paired,
                 'rejection_ledger':{'schema':'v7-replay-rejections-v1','evaluations':{k:v['rejections'] for k,v in evaluations.items()},'comparison':paired['exclusions']}}
        expected_results={'daily_metrics':expected,'paired_comparison':payloads['paired_comparison'],'rejection_ledger':payloads['rejection_ledger']}
        out['recomputed_hashes']={k:canonical_hash(v) for k,v in results.items()}
        out['expected_hashes']={k:canonical_hash(v) for k,v in expected_results.items()}
        for name in results:
            if out['recomputed_hashes'][name]!=out['expected_hashes'][name]:raise BundleError('REPLAY_RESULT_MISMATCH:'+name)
        # Exposure is retained only; replay is not permission to feed results into selection.
        if not isinstance(payloads['exposure_ledger']['events'],list):raise BundleError('EXPOSURE_LEDGER_MALFORMED')
        if payloads['completion_evidence'].get('training_executed') is not False or payloads['completion_evidence'].get('historical_live_prediction_proven') is not False:raise BundleError('UNSUPPORTED_TRAINING_OR_HISTORICAL_CLAIM')
        out['capabilities']['EVALUATION_REPLAYABLE']={'proven':True,'reasons':[]}
        return {**out,'status':'EVALUATION_REPLAYABLE','comparisons_match':True,
                'daily':{k:v['daily'] for k,v in evaluations.items()},'paired_daily':paired['daily'],
                'rejection_ledger':results['rejection_ledger'],'exposure_events_retained':len(payloads['exposure_ledger']['events'])}
    except (OSError,KeyError,TypeError,ValueError,IndexError) as exc:return fail(str(exc))


def _safe_generation(root,entry,identity):
    base={'status':'INVALID_GENERATION','file':entry.get('file') if isinstance(entry,Mapping) else None}
    def reject(reason):return {**base,'status':reason}
    try:
        if not isinstance(entry,Mapping):return reject('MISSING_GENERATION')
        name=entry['file']
        if entry.get('payload_format')!='SYNTHETIC_JSON_NOT_PYTORCH' or not isinstance(name,str) or not name.endswith('.json'):return reject('UNTRUSTED_CHECKPOINT_FORMAT')
        if Path(name).name!=name or '/' in name or '\\' in name:return reject('UNSAFE_GENERATION_PATH')
        p=_safe(Path(root)/name)
        if not p.is_file():return reject('MISSING_GENERATION')
        if type(entry.get('bytes')) is not int or entry['bytes']<=0 or not _sha(entry.get('sha256')) or p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:return reject('CORRUPT_GENERATION')
        state=strict_json(p.read_bytes())
        if not isinstance(state,Mapping):return reject('INVALID_STATE_CONTAINER')
        if state.get('payload_format')!='SYNTHETIC_JSON_NOT_PYTORCH':return reject('UNTRUSTED_CHECKPOINT_FORMAT')
        validate_state(state,(identity['contract_sha256'],identity['matrix_sha256'],identity['seed']))
        if any(state.get(k)!=identity[k] for k in ('run_id','model_sha256')):return reject('RUN_OR_MODEL_IDENTITY_MISMATCH')
        if any(not isinstance(state[k],Mapping) or not state[k] for k in ('model','optimizer','scheduler','rng')):return reject('MISSING_RUNTIME_STATE')
        return {**base,'status':'VERIFIED_SYNTHETIC_STATE','sha256':entry['sha256'],'runtime_proof':'MOCK_CONTAINERS_ONLY_NOT_REAL_TRAINING_STATE'}
    except (OSError,KeyError,TypeError,ValueError):return reject('STATE_OR_IDENTITY_INVALID')


def restore_eligibility(checkpoint_root,expected_pointer_sha256,identity,dataset_root,dataset_ref):
    out={'schema':'v7-safe-restore-evidence-v1','status':'NOT_RESTORABLE','research_admitted':False,
         'training_resumable':False,'reasons':[],'roles':{},'resume_candidate':None,'orphans_not_promoted':[],
         'limitations':['Synthetic JSON state only; no PyTorch deserialization or real optimizer/RNG/AMP/sampler roundtrip',
                        'No full reproduction or research admission; no pointer writes or orphan promotion']}
    def fail(reason):return {**out,'reasons':[reason]}
    try:
        if not isinstance(identity,Mapping) or not isinstance(dataset_ref,Mapping):raise BundleError('RESTORE_IDENTITY_INVALID')
        if not all(_sha(identity.get(k)) for k in ('contract_sha256','matrix_sha256','model_sha256')) or type(identity.get('seed')) is not int or identity['seed']<0 or not _text(identity.get('run_id')):raise BundleError('RESTORE_IDENTITY_INVALID')
        _relative(dataset_ref['path']);data=_safe(Path(dataset_root)/dataset_ref['path'])
        if not data.is_file():raise BundleError('MISSING_DATASET_BYTES')
        if type(dataset_ref['bytes']) is not int or not _sha(dataset_ref['sha256']) or data.stat().st_size!=dataset_ref['bytes'] or sha256(data)!=dataset_ref['sha256']:raise BundleError('CORRUPT_DATASET_BYTES')
        if dataset_ref['identity_sha256']!=identity['matrix_sha256']:raise BundleError('DATASET_IDENTITY_MISMATCH')
        pointer=_safe(Path(checkpoint_root)/'checkpoint.json')
        if not pointer.is_file():raise BundleError('MISSING_CHECKPOINT_POINTER')
        if not _sha(expected_pointer_sha256) or sha256(pointer)!=expected_pointer_sha256:raise BundleError('CHECKPOINT_POINTER_IDENTITY_MISMATCH')
        m=strict_json(pointer.read_bytes())
        if not isinstance(m,Mapping) or m.get('schema_version')!=POINTER_SCHEMA:raise BundleError('CHECKPOINT_POINTER_SCHEMA_MISMATCH')
        roles={role:_safe_generation(checkpoint_root,m.get(role),identity) for role in ('best','latest','previous')}
        out['roles']=roles
        linked={e['file'] for e in roles.values() if isinstance(e.get('file'),str)}
        out['orphans_not_promoted']=sorted(p.name for p in Path(checkpoint_root).glob('state-*.json') if p.name not in linked)
        candidate=next((r for r in ('latest','previous') if roles[r]['status']=='VERIFIED_SYNTHETIC_STATE'),None)
        out['resume_candidate']=candidate
        out['dataset_bytes_verified']=True
        if candidate is None:return fail('NO_VALID_LATEST_OR_PREVIOUS_GENERATION')
        out['status']='SYNTHETIC_RESTORE_CONTRACT_VERIFIED'
        if candidate=='previous':out['reasons']=['LATEST_INVALID_USING_VERIFIED_PREVIOUS']
        return out
    except (OSError,KeyError,TypeError,ValueError) as exc:return fail(str(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--run-key',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=_safe(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=replay_bundle(a.root,a.run_key,a.manifest_sha256)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return 0 if result['status']=='EVALUATION_REPLAYABLE' else 2

if __name__=='__main__':raise SystemExit(main())
