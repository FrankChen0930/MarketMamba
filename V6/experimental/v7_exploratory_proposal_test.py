"""CPU identity fixtures only; no scores, fits or historical labels."""
from pathlib import Path
import copy,json,unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_exploratory_proposal import inspect_proposal

ROOT=Path(__file__).resolve().parents[2]/'research/v7'
def fixture():
    return [json.loads((ROOT/n).read_text()) for n in ('window15-exploratory-study.json','window15-exploratory-dataset.json','window15-exploratory-feature-manifest.json')]
class ExploratoryProposalTest(unittest.TestCase):
    def test_separate_proposal_only_with_serializable_identity(self):
        x=fixture();r=inspect_proposal(*x,canonical_hash(x));self.assertEqual(r['status'],'PREPARATION_ONLY');self.assertFalse(r['DATA_READY']);self.assertFalse(r['execution_authorized']);json.dumps(r,allow_nan=False)
    def test_identity_change_not_rehashable_into_old_pin(self):
        x=fixture();pin=canonical_hash(x);x[1]['dataset_id']='changed';self.assertEqual(inspect_proposal(*x,pin)['status'],'REJECTED')
    def test_e5_identity_and_forward_track_cannot_be_repurposed(self):
        for field,value in [('study_id','E5-PIT-Clean-v1'),('track','B_FORWARD_VINTAGE')]:
            x=fixture();x[0][field]=value;self.assertEqual(inspect_proposal(*x,canonical_hash(x))['status'],'REJECTED')
    def test_no_admission_or_final_or_execution_status_flip(self):
        for field in ('human_research_authorized','execution_authorized','independent_final_test_claim','e5_e1_replication_claim'):
            x=fixture();x[0][field]=True;self.assertEqual(inspect_proposal(*x,canonical_hash(x))['status'],'REJECTED')
    def test_dataset_and_feature_pins_required_even_if_outer_repin(self):
        for idx,key in [(1,'requested_data_class'),(2,'ordered_candidates')]:
            x=fixture();x[idx][key]='changed';self.assertEqual(inspect_proposal(*x,canonical_hash(x))['status'],'REJECTED')
    def test_forward_capture_not_a_research_dataset_or_hash_restore(self):
        x=fixture();x[1]={'schema':'v7-forward-observation-proposal-v1'};self.assertEqual(inspect_proposal(*x,canonical_hash(x))['status'],'REJECTED')
    def test_no_automatic_conservative_lag(self):
        x=fixture();x[0]['conservative_lag_sessions']=1;self.assertEqual(inspect_proposal(*x,canonical_hash(x))['status'],'REJECTED')
if __name__=='__main__':unittest.main()
