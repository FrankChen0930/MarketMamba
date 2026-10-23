"""Requires only the dedicated pinned Window11 CPU overlay environment."""
import json,tempfile,unittest
from pathlib import Path
from V6.experimental.v7_classical_study import run_study
from V6.experimental.v7_classical_baseline import dependency_status
from V6.experimental.v7_research_replay import replay_bundle

class RealLightGBMTest(unittest.TestCase):
    def test_actual_l2_fit_refit_forward_and_replay(self):
        self.assertTrue(dependency_status()['available'],'run in the pinned W11 overlay, never silently skip')
        with tempfile.TemporaryDirectory() as folder:
            result=run_study(folder,comparison_model='L2')
            self.assertEqual(result['status'],'STUDY_ARTIFACTS_VERIFIED',result['reasons'])
            model=result['refit_fits']['L2']['model'];self.assertTrue(model['booster_text']);self.assertEqual(model['dependency']['installed_version'],'4.6.0')
            self.assertEqual(result['forward_evaluations']['a']['header']['recipe_id'],'toy-L2')
            proof=replay_bundle(folder,result['run_key'],result['manifest_sha256']);self.assertEqual(proof['status'],'EVALUATION_REPLAYABLE',proof['reasons'])
            self.assertTrue(proof['classical_validation_replay']['verified']);self.assertFalse(proof['capabilities']['TRAINING_RESUMABLE']['proven'])
    def test_same_support_and_coverage_exclusions(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_study(folder,comparison_model='L2',drop_l0_forward_stock=True)
            self.assertEqual(result['status'],'STUDY_ARTIFACTS_VERIFIED',result['reasons'])
            self.assertTrue(result['paired_comparison']['exclusions']);self.assertTrue(all(d['valid_n']==3 for d in result['paired_comparison']['daily']))
    def test_forward_label_change_does_not_select_or_refit(self):
        from V6.experimental.v7_classical_baseline_test import panel,ids_for
        import copy
        with tempfile.TemporaryDirectory() as folder:
            p=panel();a=run_study(Path(folder)/'a',inputs=copy.deepcopy(p),comparison_model='L2')
            p[4][ids_for(p,(30,))[0]]['5']['value']=-99
            b=run_study(Path(folder)/'b',inputs=p,comparison_model='L2')
            self.assertEqual(a['selection_decision'],b['selection_decision']);self.assertEqual(a['refit_fits'],b['refit_fits'])
            self.assertNotEqual(a['forward_evaluations'],b['forward_evaluations'])

    def test_explicit_code_snapshot_cannot_claim_unavailable_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_study(folder,comparison_model='L2',source_commit='0'*40)
            self.assertEqual(result['status'],'STUDY_REJECTED');self.assertIn('CODE_SNAPSHOT_UNAVAILABLE',result['reasons'])

    def test_explicit_matching_snapshot_success_path(self):
        from unittest.mock import patch
        from V6.experimental.v7_classical_study import ROOT
        def source_bytes(command,**kwargs):return (ROOT/command[2].split(':',1)[1]).read_bytes()
        with tempfile.TemporaryDirectory() as folder,patch('V6.experimental.v7_classical_study.subprocess.check_output',side_effect=source_bytes):
            result=run_study(folder,comparison_model='L2',source_commit='a'*40)
            self.assertEqual(result['status'],'STUDY_ARTIFACTS_VERIFIED',result['reasons'])
            self.assertEqual(result['manifest']['identity']['code_commit'],'a'*40)
            self.assertTrue(result['manifest']['identity']['run_id'].endswith('a'*12))

if __name__=='__main__':unittest.main()
