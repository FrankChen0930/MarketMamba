"""Bounded two-baseline CPU lifecycle. All panels are named toy fixtures."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,tempfile,unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_classical_baseline_test import panel,ids_for
from V6.experimental.v7_research_artifacts import verify_bundle,run_key,publish_bundle
from V6.experimental.v7_research_replay import replay_bundle
from V6.experimental.v7_classical_study import run_study


class ClassicalStudyTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'bundle'
    def tearDown(self):self.temp.cleanup()
    def test_end_to_end_fresh_runs_and_complete_bundle(self):
        r=run_study(self.root);self.assertEqual(r['status'],'STUDY_ARTIFACTS_VERIFIED',r.get('reasons'))
        self.assertEqual(set(r['validation_evaluations']),{'L0','L1'});self.assertEqual(set(r['forward_evaluations']),{'a','b'})
        for kind,refit in r['refit_fits'].items():
            self.assertNotEqual(refit['model']['fit_run_id'],r['selection_fits'][kind]['model']['fit_run_id'])
            self.assertIsNone(refit['model']['resume_checkpoint']);self.assertEqual(refit['model']['fit_phase'],'REFIT')
        self.assertFalse(r['research_admitted']);self.assertEqual(r['manifest']['training_status'],'SYNTHETIC_CPU_FIT_COMPLETED')
        self.assertEqual(r['selection_decision_before_sha256'],r['selection_decision_after_sha256']);json.dumps(r,allow_nan=False)
    def test_independent_process_uses_existing_replay(self):
        r=run_study(self.root);path=Path(self.temp.name)/'replay.json'
        result=subprocess.run([sys.executable,'-B','-m','V6.experimental.v7_research_replay','--root',str(self.root),'--run-key',r['run_key'],'--manifest-sha256',r['manifest_sha256'],'--output',str(path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);ev=json.loads(path.read_text());self.assertTrue(ev['comparisons_match']);self.assertTrue(ev['classical_validation_replay']['verified'])
        self.assertFalse(ev['capabilities']['TRAINING_RESUMABLE']['proven']);self.assertFalse(ev['capabilities']['FULL_REPRODUCTION_READY']['proven'])
    def test_selection_cannot_read_forward_labels(self):
        p=panel();sid=ids_for(p,(30,))[0]
        # Forward answers are unavailable until after scoring; changing them cannot select a model.
        a=run_study(self.root,inputs=copy.deepcopy(p));p[4][sid]['5']['value']=-999.
        b=run_study(Path(self.temp.name)/'other',inputs=p)
        self.assertEqual(a['selection_decision'],b['selection_decision'])
        self.assertEqual(a['selection_fits'],b['selection_fits']);self.assertEqual(a['refit_fits'],b['refit_fits'])
        self.assertNotEqual(a['forward_evaluations'],b['forward_evaluations'])
    def test_missing_immature_and_future_features_fail_before_publication(self):
        for mode in ('missing','immature','future'):
            p=panel();sid=ids_for(p,(1,))[0]
            if mode=='missing':del p[4][sid]['5']
            elif mode=='immature':p[4][sid]['10']['available_at']='2030-01-01T00:00:00+08:00'
            else:p[3][sid]['quality_checked_at']='2030-01-01T00:00:00+08:00'
            r=run_study(self.root/ mode,inputs=p);self.assertEqual(r['status'],'STUDY_REJECTED');self.assertFalse((self.root/mode/'completed').exists())
    def test_invalid_fold_or_changed_contract_fail_closed(self):
        p=panel();p[1]['unresolved_policies']=['unknown'];self.assertEqual(run_study(self.root,inputs=p)['status'],'STUDY_REJECTED')
        for key in ('feature_contract_id','label_contract_id'):
            p=panel();sid=ids_for(p,(1,))[0]
            if key=='feature_contract_id':p[3][sid][key]='OTHER'
            else:p[4][sid]['5'][key]='OTHER'
            self.assertEqual(run_study(self.root/key,inputs=p)['status'],'STUDY_REJECTED')
    def test_duplicate_publication_and_interruption_are_not_false_completion(self):
        a=run_study(self.root);b=run_study(self.root)
        self.assertEqual(a['manifest_sha256'],b['manifest_sha256']);self.assertEqual(b['publication']['status'],'ALREADY_PUBLISHED')
        root=Path(self.temp.name)/'interrupted';r=run_study(root,interrupt_at='after_artifact')
        self.assertEqual(r['status'],'STUDY_INCOMPLETE');self.assertEqual(verify_bundle(root,r['run_key'],r['manifest_sha256'])['status'],'NOT_VERIFIED')
    def test_coverage_mismatch_uses_common_support(self):
        r=run_study(self.root,drop_l0_forward_stock=True);self.assertEqual(r['status'],'STUDY_ARTIFACTS_VERIFIED')
        comparison=r['paired_comparison'];self.assertTrue(comparison['exclusions'])
        self.assertTrue(all(d['valid_n']==3 for d in comparison['daily']))
        ev=replay_bundle(self.root,r['run_key'],r['manifest_sha256']);self.assertEqual(ev['status'],'EVALUATION_REPLAYABLE')
    def test_profile_v1_not_relabelled_and_v2_has_fit_evidence(self):
        r=run_study(self.root);m=copy.deepcopy(r['manifest']);self.assertEqual(m['schema'],'v7-study-run-bundle-v2')
        m.update(schema='v7-study-run-bundle-v1',profile='SYNTHETIC_EVALUATION')
        result=publish_bundle(self.root/'wrong-profile',m,canonical_hash(m),{})
        self.assertEqual(result['status'],'INCOMPLETE')
        path=Path(r['publication']['bundle_path']);(path/'training_logs.json').write_text('{}')
        self.assertEqual(replay_bundle(self.root,r['run_key'],r['manifest_sha256'])['status'],'REPLAY_REJECTED')
    def test_no_fabricated_lightgbm_result_or_formal_admission(self):
        r=run_study(self.root)
        if r['dependency_probe']['status']=='DEPENDENCY_BLOCKED':self.assertNotIn('L2',r['refit_fits']);self.assertNotIn('L2',r['validation_evaluations'])
        self.assertEqual(r['manifest']['evidence_class'],'SYNTHETIC');self.assertFalse(r['manifest']['research_admitted'])
    def altered_bundle(self,result,name,mutate,suffix):
        manifest=copy.deepcopy(result['manifest']);base=Path(result['publication']['bundle_path'])
        sources={a['name']:(base/a['path']).read_bytes() for a in manifest['artifacts']}
        value=json.loads(sources[name]);mutate(value);data=(json.dumps(value,sort_keys=True)+'\n').encode();sources[name]=data
        artifact=next(a for a in manifest['artifacts'] if a['name']==name);artifact.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        root=self.root/suffix;pub=publish_bundle(root,manifest,canonical_hash(manifest),sources)
        self.assertEqual(pub['status'],'ARTIFACTS_VERIFIED')
        return replay_bundle(root,run_key(manifest['identity']),canonical_hash(manifest))
    def test_v2_missing_or_mismatched_fit_model_is_not_replayable(self):
        r=run_study(self.root)
        changes=[lambda log:log.update(refit_fits={'unrelated':{'status':'FIT_ENGINEERING_ONLY','research_admitted':False}}),
                 lambda log:log['refit_fits']['L1']['model'].update(fit_run_sha256='0'*64),
                 lambda log:log['forward_declarations']['L1'].update(model_sha256='0'*64)]
        for n,change in enumerate(changes):
            replay=self.altered_bundle(r,'training_logs',change,'changed-'+str(n))
            self.assertEqual(replay['status'],'REPLAY_REJECTED');self.assertTrue(replay['reasons'][0].startswith('CLASSICAL_'),replay['reasons'])
    def test_validation_metrics_are_independently_replayed_too(self):
        r=run_study(self.root)
        replay=self.altered_bundle(r,'validation_metrics',lambda v:v['evaluations']['L0']['daily'][0].update(rank_ic=-.123),'bad-validation-metric')
        self.assertEqual(replay['status'],'REPLAY_REJECTED');self.assertIn('CLASSICAL_VALIDATION_REPLAY_MISMATCH',replay['reasons'])
    def test_v2_training_configuration_hash_is_bound(self):
        r=run_study(self.root)
        replay=self.altered_bundle(r,'resolved_config',lambda conf:conf['study_configuration']['selection_policy'].update(metric='OTHER'),'changed-config')
        self.assertEqual(replay['status'],'REPLAY_REJECTED');self.assertIn('CLASSICAL_CONFIG_IDENTITY_MISMATCH',replay['reasons'])
    def test_synthetic_result_identities_are_persistent(self):
        r=run_study(self.root);v=verify_bundle(self.root,r['run_key'],r['manifest_sha256'])
        self.assertEqual(v['manifest']['identity'],r['manifest']['identity'])
        self.assertEqual(run_key(r['manifest']['identity']),r['run_key']);self.assertEqual(v['manifest']['identity']['model_sha256'],r['forward_evaluations']['a']['header']['model_sha256'])

if __name__=='__main__':unittest.main()
