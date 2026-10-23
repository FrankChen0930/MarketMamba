"""Owned temporary synthetic bytes only; no historical artifacts touched."""
from pathlib import Path
import copy,hashlib,json,tempfile,unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_artifacts import publish_bundle,verify_bundle,run_key


def encoded(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def bundle_fixture():
    identity={'study_id':'synthetic-study','recipe_id':'synthetic-recipe','run_id':'synthetic-run','fold_id':'synthetic-fold',
              'origin_id':'synthetic-origin','origin_at':'2024-02-01T08:00:00+08:00','seed':0,
              'code_commit':'bdea30d877fa2c5d02308f343fa5591a35616aec','prediction_role':'FORWARD',
              **{k:'a'*64 for k in ('code_manifest_sha256','data_sha256','feature_sha256','label_sha256','universe_sha256','evaluation_policy_sha256','model_sha256','training_contract_sha256')},
              'checkpoint_status':'NOT_GENERATED_SYNTHETIC_NO_TRAINING','checkpoint_sha256':None,
              'source_vintages':[{'source_sha256':'b'*64,'vintage_id':'synthetic-original'}]}
    kinds=['RESOLVED_CONFIG','ENVIRONMENT','DATA_SPLIT_ELIGIBILITY','PREDICTIONS','DAILY_METRICS','REJECTION_LEDGER','COMPLETION_EVIDENCE']
    files={};artifacts=[]
    for kind in kinds:
        name=kind.lower();payload={'schema':'synthetic-'+name+'-v1','evidence_class':'SYNTHETIC'}
        data=encoded(payload);files[name]=data
        artifacts.append({'name':name,'kind':kind,'storage':'BUNDLE','path':name+'.json','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'format':'JSON','schema_id':payload['schema']})
    manifest={'schema':'v7-study-run-bundle-v1','evidence_class':'SYNTHETIC','identity':identity,
              'training_status':'NOT_RUN_SYNTHETIC','research_admitted':False,'profile':'SYNTHETIC_EVALUATION',
              'required_artifacts':[x['name'] for x in artifacts],'artifacts':artifacts,
              'logical_observation_period':['2024-01-01','2024-03-01']}
    return manifest,files


class ArtifactPublicationTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'bundles';self.m,self.files=bundle_fixture()
    def tearDown(self):self.temp.cleanup()
    def publish(self,**kwargs):return publish_bundle(self.root,self.m,canonical_hash(self.m),self.files,**kwargs)
    def verify(self):return verify_bundle(self.root,run_key(self.m['identity']),canonical_hash(self.m))
    def test_complete_atomic_publication_and_verify(self):
        r=self.publish();self.assertEqual(r['status'],'ARTIFACTS_VERIFIED');v=self.verify()
        self.assertEqual(v['status'],'ARTIFACTS_VERIFIED');self.assertFalse(v['research_admitted'])
        self.assertEqual(v['training_status'],'NOT_RUN_SYNTHETIC');json.dumps(v,allow_nan=False)
    def test_partial_publication_never_complete(self):
        r=self.publish(interrupt_at='after_artifact');self.assertEqual(r['status'],'INCOMPLETE')
        self.assertIn('INTERRUPTED_AFTER_ARTIFACT',r['reasons']);self.assertNotEqual(self.verify()['status'],'ARTIFACTS_VERIFIED')
        self.assertTrue(list((self.root/'staging').iterdir()));self.assertTrue(list((self.root/'attempts').glob('*.json')))
    def test_missing_required_prediction(self):
        del self.files['predictions'];r=self.publish();self.assertIn('MISSING_ARTIFACT_BYTES:predictions',r['reasons'])
        self.assertFalse((self.root/'completed'/run_key(self.m['identity'])).exists())
    def test_corrupt_bytes_rejected(self):
        self.files['predictions']=b'corrupt';self.assertIn('ARTIFACT_HASH_MISMATCH:predictions',self.publish()['reasons'])
    def test_schema_mismatch(self):
        data=encoded({'schema':'OTHER'});self.files['predictions']=data
        d=next(a for a in self.m['artifacts'] if a['name']=='predictions');d.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        self.assertIn('ARTIFACT_SCHEMA_MISMATCH:predictions',self.publish()['reasons'])
    def test_manifest_identity_pin_and_required_kinds(self):
        original=canonical_hash(self.m);self.m['identity']['run_id']='changed'
        r=publish_bundle(self.root,self.m,original,self.files);self.assertIn('MANIFEST_IDENTITY_MISMATCH',r['reasons'])
        self.m['required_artifacts']=[];self.assertIn('REQUIRED_PROFILE_ARTIFACTS_MISSING',self.publish()['reasons'])
    def test_duplicate_publication_is_idempotent_not_overwritten(self):
        first=self.publish();second=self.publish();self.assertEqual(second['status'],'ALREADY_PUBLISHED')
        self.assertEqual(first['bundle_path'],second['bundle_path'])
        self.m['identity']['recipe_id']='OTHER';r=self.publish();self.assertIn('IMMUTABLE_RUN_CONFLICT',r['reasons'])
    def test_interrupted_before_publish_can_retry(self):
        r=self.publish(interrupt_at='before_publish');self.assertIn('INTERRUPTED_BEFORE_PUBLISH',r['reasons'])
        self.assertEqual(self.publish()['status'],'ARTIFACTS_VERIFIED')
    def test_interrupted_after_publish_retry_recognizes_valid_bundle(self):
        self.publish(interrupt_at='after_publish');self.assertEqual(self.verify()['status'],'ARTIFACTS_VERIFIED')
        self.assertEqual(self.publish()['status'],'ALREADY_PUBLISHED')
    def test_corrupt_complete_never_silently_replaced(self):
        r=self.publish();p=Path(r['bundle_path'])/'predictions.json';p.write_bytes(b'bad')
        self.assertIn('ARTIFACT_HASH_MISMATCH:predictions',self.verify()['reasons'])
        self.assertTrue(any(x.startswith('EXISTING_BUNDLE_INVALID:') for x in self.publish()['reasons']));self.assertEqual(p.read_bytes(),b'bad')
    def test_paths_symlinks_duplicate_entries_fail_closed(self):
        for bad in ('../escape','/tmp/escape','nested/../../escape','\\escape'):
            m,f=bundle_fixture();m['artifacts'][0]['path']=bad
            self.assertEqual(publish_bundle(self.root,m,canonical_hash(m),f)['status'],'INCOMPLETE')
        self.m['artifacts'].append(copy.deepcopy(self.m['artifacts'][0]));self.assertIn('DUPLICATE_ARTIFACT_NAME_OR_PATH',self.publish()['reasons'])
    def test_hash_only_reference_not_restorable(self):
        a=self.m['artifacts'][0];a.update(storage='EXTERNAL_REFERENCE',path=str(Path(self.temp.name)/'missing.json'))
        self.assertIn('NOT_RESTORABLE:'+a['name'],self.publish(external_roots=[Path(self.temp.name)])['reasons'])
    def test_verified_shared_reference_and_later_loss(self):
        shared=Path(self.temp.name)/'shared.json';shared.write_bytes(self.files['predictions']);a=next(x for x in self.m['artifacts'] if x['name']=='predictions');a.update(storage='EXTERNAL_REFERENCE',path=str(shared))
        r=self.publish(external_roots=[Path(self.temp.name)]);self.assertEqual(r['status'],'ARTIFACTS_VERIFIED')
        shared.unlink() # test-owned synthetic file only
        v=verify_bundle(self.root,run_key(self.m['identity']),canonical_hash(self.m),external_roots=[Path(self.temp.name)])
        self.assertIn('NOT_RESTORABLE:predictions',v['reasons'])
    def test_generated_time_not_historical_observation_time(self):
        r=self.publish();stamp=json.loads((Path(r['bundle_path'])/'publication.json').read_text())
        self.assertGreater(stamp['generated_at'],'2025-01-01');self.assertNotEqual(stamp['generated_at'][:10],self.m['logical_observation_period'][0])
    def test_external_reference_parent_traversal_is_not_allowed(self):
        allowed=Path(self.temp.name)/'allowed';allowed.mkdir();outside=Path(self.temp.name)/'outside.json';outside.write_bytes(self.files['predictions'])
        a=next(x for x in self.m['artifacts'] if x['name']=='predictions')
        a.update(storage='EXTERNAL_REFERENCE',path=str(allowed/'..'/'outside.json'))
        self.assertNotEqual(self.publish(external_roots=[allowed])['status'],'ARTIFACTS_VERIFIED')
    def test_manifest_and_publication_leaf_symlinks_are_refused(self):
        for leaf in ('manifest.json','publication.json'):
            with self.subTest(leaf=leaf):
                # Separate run per owned fixture, preserving original completed data.
                self.m['identity']['run_id']=leaf
                p=Path(self.publish()['bundle_path'])/leaf;outside=Path(self.temp.name)/('outside-'+leaf)
                outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
                self.assertNotEqual(self.verify()['status'],'ARTIFACTS_VERIFIED')
                self.assertEqual(self.publish()['status'],'INCOMPLETE')
    def test_json_exponent_overflow_is_rejected_recursively(self):
        name='predictions';data=b'{"schema":"synthetic-predictions-v1","nested":[{"value":1e400}]}'
        self.files[name]=data;a=next(x for x in self.m['artifacts'] if x['name']==name)
        a.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        self.assertIn('NONFINITE_JSON',self.publish()['reasons'])
    def test_publication_namespace_symlink_refused(self):
        self.root.mkdir();target=Path(self.temp.name)/'elsewhere';target.mkdir();(self.root/'completed').symlink_to(target,target_is_directory=True)
        self.assertEqual(self.publish()['status'],'INCOMPLETE');self.assertEqual(list(target.iterdir()),[])

if __name__=='__main__':unittest.main()
