"""Offline synthetic replay and safe JSON checkpoint fixtures."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_artifacts import publish_bundle,run_key
from V6.experimental.v7_research_artifacts_test import bundle_fixture,encoded
from V6.experimental.v7_e1_checkpoint import FORMAT,POINTER_SCHEMA
from V6.experimental.v7_research_replay import replay_bundle,restore_eligibility

ROOT=Path(__file__).resolve().parents[2]
CODE_PATHS=['V6/experimental/v7_corrected_e5_train.py','V6/experimental/v7_corrected_e5_replication.py','V6/experimental/v7_research_temporal.py','V6/experimental/v7_research_evaluation.py','V6/experimental/v7_research_comparison.py','V6/experimental/v7_temporal_contract.py']


def replay_fixture():
    raw_bytes=(ROOT/'research/v7/window08-synthetic-inputs.json').read_bytes();raw=json.loads(raw_bytes)
    original=json.loads((ROOT/'research/v7/window08-synthetic-outputs.json').read_text())
    m,_=bundle_fixture();code={'schema':'v7-replay-code-manifest-v1','commit':'bdea30d877fa2c5d02308f343fa5591a35616aec','files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in CODE_PATHS}}
    from V6.experimental.v7_research_replay import runtime_identity
    header=original['evaluations']['a']['header'];fold=raw['fold']
    ident=m['identity'];ident.update(study_id=header['study_id'],recipe_id=header['recipe_id'],run_id=header['run_id'],fold_id=header['fold_id'],origin_id='synthetic-origin-A',origin_at=fold['origin_at'],
                                   model_sha256=header['model_sha256'],code_manifest_sha256=canonical_hash(code),data_sha256=canonical_hash(raw),feature_sha256=canonical_hash({'feature_contract_id':fold['feature_contract_id']}),
                                   label_sha256=canonical_hash({'label_contract_id':fold['label_contract_id'],'label_definition':fold['label_definition']}),universe_sha256=canonical_hash(sorted({r['stock_id'] for r in raw['scores']['a']+raw['scores']['b']})),evaluation_policy_sha256=canonical_hash(raw['policy']))
    data={
      'resolved_config':('RESOLVED_CONFIG',{'schema':'v7-replay-config-v1','identity':copy.deepcopy(ident)}),
      'environment':('ENVIRONMENT',{'schema':'v7-replay-environment-v1','runtime':runtime_identity()}),
      'data_split_eligibility':('DATA_SPLIT_ELIGIBILITY',{'schema':'v7-replay-split-v1','fold_sha256':canonical_hash(fold),'calendar_sha256':canonical_hash(raw['calendar']),'label_metadata_sha256':canonical_hash(raw['label_metadata'])}),
      'predictions':('PREDICTIONS',{'schema':'v7-replay-predictions-v1','bundles':original['prediction_bundles']}),
      'daily_metrics':('DAILY_METRICS',{'schema':'v7-replay-expected-evaluations-v1','evaluations':original['evaluations']}),
      'paired_comparison':('PAIRED_COMPARISON',original['comparison']),
      'rejection_ledger':('REJECTION_LEDGER',{'schema':'v7-replay-rejections-v1','evaluations':{k:v['rejections'] for k,v in original['evaluations'].items()},'comparison':original['comparison']['exclusions']}),
      'exposure_ledger':('EXPOSURE_LEDGER',{'schema':'v7-replay-exposure-v1','events':[]}),
      'completion_evidence':('COMPLETION_EVIDENCE',{'schema':'v7-replay-completion-v1','training_executed':False,'historical_live_prediction_proven':False}),
      'code_manifest':('CODE_MANIFEST',code)}
    files={};artifacts=[]
    for name,(kind,value) in data.items():
        b=encoded(value);files[name]=b;artifacts.append({'name':name,'kind':kind,'storage':'BUNDLE','path':name+'.json','sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'format':'JSON','schema_id':value['schema']})
    files['dataset_snapshot']=raw_bytes;artifacts.append({'name':'dataset_snapshot','kind':'DATASET_SNAPSHOT','storage':'BUNDLE','path':'dataset.json','sha256':hashlib.sha256(raw_bytes).hexdigest(),'bytes':len(raw_bytes),'format':'JSON','schema_id':raw['schema']})
    m.update(artifacts=artifacts,required_artifacts=[a['name'] for a in artifacts]);return m,files


def change_json(m,files,name,fn):
    value=json.loads(files[name]);fn(value);b=encoded(value);files[name]=b
    a=next(a for a in m['artifacts'] if a['name']==name);a.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def restore_fixture(base):
    root=base/'checkpoint';root.mkdir();dataset=base/'dataset.json';dataset.write_bytes(encoded({'schema':'synthetic-dataset','rows':[1,2,3]}));dsha=hashlib.sha256(dataset.read_bytes()).hexdigest()
    identity={'contract_sha256':'a'*64,'matrix_sha256':dsha,'seed':0,'run_id':'mock-run','model_sha256':'b'*64}
    dataset_ref={'path':'dataset.json','sha256':dsha,'bytes':dataset.stat().st_size,'identity_sha256':dsha}
    pointer={'schema_version':POINTER_SCHEMA}
    for role in ('best','latest','previous'):
        state={'format':FORMAT,'payload_format':'SYNTHETIC_JSON_NOT_PYTORCH',**identity,'epoch':2,'batch':0,'step':3,'phase':'train','terminal':False,
               'model':{'mock':1},'optimizer':{'mock':2},'scheduler':{'mock':3},'rng':{'mock':4}}
        data=encoded(state);name='state-'+role+'.json';(root/name).write_bytes(data)
        pointer[role]={'file':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'payload_format':'SYNTHETIC_JSON_NOT_PYTORCH'}
    (root/'checkpoint.json').write_bytes(encoded(pointer))
    return root,hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest(),identity,base,dataset_ref


class OfflineReplayTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'bundles';self.m,self.files=replay_fixture()
    def tearDown(self):self.temp.cleanup()
    def published(self):return publish_bundle(self.root,self.m,canonical_hash(self.m),self.files)
    def replay(self):return replay_bundle(self.root,run_key(self.m['identity']),canonical_hash(self.m))
    def test_replay_from_verified_bundle_exact_metrics_and_rejections(self):
        self.assertEqual(self.published()['status'],'ARTIFACTS_VERIFIED');self.files.clear()
        r=self.replay();self.assertEqual(r['status'],'EVALUATION_REPLAYABLE');self.assertTrue(r['comparisons_match'])
        self.assertTrue(r['capabilities']['METADATA_VERIFIED']['proven']);self.assertTrue(r['capabilities']['EVALUATION_REPLAYABLE']['proven'])
        self.assertFalse(r['capabilities']['TRAINING_RESUMABLE']['proven']);self.assertFalse(r['capabilities']['FULL_REPRODUCTION_READY']['proven'])
        self.assertFalse(r['historical_live_prediction_proven']);json.dumps(r,allow_nan=False)
    def test_independent_cpu_process_replay(self):
        self.published();output=self.base/'replay-evidence.json'
        command=[sys.executable,'-B','-m','V6.experimental.v7_research_replay','--root',str(self.root),'--run-key',run_key(self.m['identity']),'--manifest-sha256',canonical_hash(self.m),'--output',str(output)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(output.read_text())['status'],'EVALUATION_REPLAYABLE')
        second=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);self.assertNotEqual(second.returncode,0)
    def test_corrupt_artifact_missing_prediction_or_schema_identity(self):
        p=Path(self.published()['bundle_path']);(p/'predictions.json').write_bytes(b'bad')
        self.assertEqual(self.replay()['status'],'REPLAY_REJECTED');self.assertFalse(self.replay()['capabilities']['METADATA_VERIFIED']['proven'])
    def test_changed_evaluation_policy_rejected(self):
        change_json(self.m,self.files,'dataset_snapshot',lambda raw:raw['policy'].update(min_valid_n=3))
        self.published();self.assertIn('DATA_OR_POLICY_IDENTITY_MISMATCH',self.replay()['reasons'])
    def test_missing_label_or_maturity_not_recreated(self):
        for key in ('label_payloads','label_metadata'):
            m,files=replay_fixture()
            change_json(m,files,'dataset_snapshot',lambda raw:raw[key].clear())
            # Preserve matching data identity to exercise evaluator maturity/access recheck.
            raw=json.loads(files['dataset_snapshot']);m['identity']['data_sha256']=canonical_hash(raw)
            change_json(m,files,'resolved_config',lambda v:v.update(identity=m['identity']))
            if key=='label_metadata':change_json(m,files,'data_split_eligibility',lambda v:v.update(label_metadata_sha256=canonical_hash(raw['label_metadata'])))
            # Raw prediction artifact reference must also match changed bytes, not silently bypass it.
            root=self.base/key;publish_bundle(root,m,canonical_hash(m),files)
            r=replay_bundle(root,run_key(m['identity']),canonical_hash(m));self.assertEqual(r['status'],'REPLAY_REJECTED')
    def test_malformed_nested_snapshot_has_serializable_denial(self):
        change_json(self.m,self.files,'dataset_snapshot',lambda v:v.update(scores=None))
        self.m['identity']['data_sha256']=canonical_hash(json.loads(self.files['dataset_snapshot']))
        change_json(self.m,self.files,'resolved_config',lambda v:v.update(identity=self.m['identity']))
        self.assertEqual(self.published()['status'],'ARTIFACTS_VERIFIED')
        r=self.replay();self.assertEqual(r['status'],'REPLAY_REJECTED');self.assertIn('REPLAY_NESTED_CONTAINER_INVALID',r['reasons']);json.dumps(r)
    def test_missing_or_late_maturity_reaches_actual_label_gate(self):
        for mode in ('missing_available','late_available','missing_payload'):
            m,files=replay_fixture();raw=json.loads(files['dataset_snapshot']);lid=raw['scores']['a'][0]['label_id']
            if mode=='missing_available':del raw['label_metadata'][lid]['available_at']
            elif mode=='late_available':
                raw['label_metadata'][lid]['available_at']='2030-01-01T18:00:00+08:00'
            else:del raw['label_payloads'][lid]
            for rows in raw['scores'].values():
                for row in rows:
                    if row['label_id']==lid:row['label_metadata_sha256']=canonical_hash(raw['label_metadata'][lid])
            change_json(m,files,'dataset_snapshot',lambda v:v.update(raw))
            m['identity']['data_sha256']=canonical_hash(raw)
            change_json(m,files,'resolved_config',lambda v:v.update(identity=m['identity']))
            change_json(m,files,'data_split_eligibility',lambda v:v.update(label_metadata_sha256=canonical_hash(raw['label_metadata'])))
            datahash=hashlib.sha256(files['dataset_snapshot']).hexdigest()
            def align(v):
                for name,bundle in v['bundles'].items():
                    bundle['rows']=copy.deepcopy(raw['scores'][name]);bundle['header']['raw_predictions_ref']['sha256']=datahash
            change_json(m,files,'predictions',align)
            root=self.base/mode;publish_bundle(root,m,canonical_hash(m),files)
            result=replay_bundle(root,run_key(m['identity']),canonical_hash(m))
            self.assertEqual(result['status'],'REPLAY_REJECTED')
            self.assertTrue(result['reasons'][0].startswith('LABEL_ELIGIBILITY_RECHECK_MISMATCH:'),result['reasons'])
    def test_saved_metrics_mismatch_detected(self):
        change_json(self.m,self.files,'daily_metrics',lambda v:v['evaluations']['a']['daily'][0].update(rank_ic=0.123))
        self.published();self.assertIn('REPLAY_RESULT_MISMATCH:daily_metrics',self.replay()['reasons'])
    def test_saved_rejection_ledger_mismatch_detected(self):
        change_json(self.m,self.files,'rejection_ledger',lambda v:v.update(comparison=[]))
        self.published();self.assertIn('REPLAY_RESULT_MISMATCH:rejection_ledger',self.replay()['reasons'])
    def test_code_identity_changed_is_not_replayable(self):
        change_json(self.m,self.files,'code_manifest',lambda v:v['files'].update({CODE_PATHS[0]:'0'*64}))
        self.m['identity']['code_manifest_sha256']=canonical_hash(json.loads(self.files['code_manifest']))
        change_json(self.m,self.files,'resolved_config',lambda v:v.update(identity=self.m['identity']))
        self.published();self.assertIn('CODE_BYTES_MISMATCH',self.replay()['reasons'])
    def test_simulated_eval_time_not_actual_publication_time(self):
        self.published();r=self.replay();self.assertGreater(r['replayed_at'],'2025-01-01')
        self.assertTrue(all(t.startswith('2024-') for t in r['simulated_evaluation_times']))
        self.assertFalse(r['historical_live_prediction_proven'])


class RestoreEligibilityTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.args=restore_fixture(self.base)
    def tearDown(self):self.temp.cleanup()
    def restore(self):return restore_eligibility(*self.args)
    def test_best_latest_previous_safe_synthetic_state_only(self):
        with patch('torch.load',side_effect=AssertionError('no torch deserialization')):r=self.restore()
        self.assertEqual(r['status'],'SYNTHETIC_RESTORE_CONTRACT_VERIFIED');self.assertEqual(r['resume_candidate'],'latest')
        self.assertFalse(r['training_resumable']);self.assertFalse(r['research_admitted'])
        self.assertTrue(all(x['status']=='VERIFIED_SYNTHETIC_STATE' for x in r['roles'].values()))
    def test_missing_dataset_bytes_not_restorable(self):
        (self.base/'dataset.json').unlink();self.assertIn('MISSING_DATASET_BYTES',self.restore()['reasons'])
    def test_missing_checkpoint_generation_not_restorable(self):
        root=self.args[0]
        for role in ('latest','previous'):(root/('state-'+role+'.json')).unlink()
        r=self.restore();self.assertEqual(r['status'],'NOT_RESTORABLE');self.assertEqual(r['roles']['latest']['status'],'MISSING_GENERATION')
    def test_corrupt_latest_valid_previous_no_best_promotion(self):
        (self.args[0]/'state-latest.json').write_bytes(b'bad');r=self.restore()
        self.assertEqual(r['resume_candidate'],'previous');self.assertEqual(r['roles']['latest']['status'],'CORRUPT_GENERATION')
        self.assertEqual(r['roles']['best']['file'],'state-best.json')
    def test_orphan_not_promoted_to_best_or_resume(self):
        root=self.args[0];(root/'state-orphan.json').write_bytes((root/'state-best.json').read_bytes())
        for role in ('latest','previous'):(root/('state-'+role+'.json')).unlink()
        before=(root/'checkpoint.json').read_bytes();r=self.restore()
        self.assertIsNone(r['resume_candidate']);self.assertIn('state-orphan.json',r['orphans_not_promoted']);self.assertEqual((root/'checkpoint.json').read_bytes(),before)
    def test_untrusted_pytorch_payload_rejected_without_loading(self):
        root,pin,ident,base,ds=self.args;pointer=json.loads((root/'checkpoint.json').read_text())
        for role in ('best','latest','previous'):
            pointer[role].update(file='historical.pt',payload_format='UNKNOWN_PYTORCH')
        (root/'historical.pt').write_bytes(b'not even a pickle');(root/'checkpoint.json').write_bytes(encoded(pointer))
        args=(root,hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest(),ident,base,ds)
        with patch('torch.load',side_effect=AssertionError('unsafe load')):r=restore_eligibility(*args)
        self.assertEqual(r['status'],'NOT_RESTORABLE');self.assertTrue(all(x['status']=='UNTRUSTED_CHECKPOINT_FORMAT' for x in r['roles'].values()))
    def test_changed_run_model_contract_or_dataset_identity(self):
        for key,value in [('run_id','OTHER'),('model_sha256','f'*64),('contract_sha256','f'*64)]:
            args=list(self.args);args[2]={**args[2],key:value};r=restore_eligibility(*args)
            self.assertEqual(r['status'],'NOT_RESTORABLE')
    def test_unknown_restore_identity_container_is_rejected(self):
        args=list(self.args);args[2]=None;r=restore_eligibility(*args)
        self.assertIn('RESTORE_IDENTITY_INVALID',r['reasons']);json.dumps(r,allow_nan=False)
    def test_missing_runtime_state_and_pointer_pin(self):
        args=list(self.args);args[1]='0'*64;self.assertIn('CHECKPOINT_POINTER_IDENTITY_MISMATCH',restore_eligibility(*args)['reasons'])
        root=self.args[0];pointer=json.loads((root/'checkpoint.json').read_text())
        for role in ('latest','previous'):
            file=root/pointer[role]['file'];state=json.loads(file.read_text());state['rng']=None;data=encoded(state);file.write_bytes(data);pointer[role].update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        (root/'checkpoint.json').write_bytes(encoded(pointer));args=list(self.args);args[1]=hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest()
        self.assertEqual(restore_eligibility(*args)['status'],'NOT_RESTORABLE')

if __name__=='__main__':unittest.main()
