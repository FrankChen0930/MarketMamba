"""Fair support and exposure integration, synthetic only."""
import copy
import json
import unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_evaluation_test import evaluation_fixture,run_eval
from V6.experimental.v7_research_evaluation import exposure_reference
from V6.experimental.v7_research_isolation import expose_forward_result
from V6.experimental.v7_research_temporal import make_pin
from V6.experimental.v7_research_comparison import compare


def pair_fixture():
    a=list(evaluation_fixture(scores=(1.,2.,3.,4.),values=(1.,2.,3.,4.)));b=copy.deepcopy(a)
    a[0]['rows']=[r for r in a[0]['rows'] if r['stock_id']!='stock-3']
    b[0]['rows']=[r for r in b[0]['rows'] if r['stock_id']!='stock-0']
    b[0]['header'].update(run_id='run-B',recipe_id='recipe-B',model_sha256='7'*64)
    for r in b[0]['rows']:r['score']=-r['score']
    return a,b


def compared(a=None,b=None):
    if a is None:a,b=pair_fixture();a,b=run_eval(a),run_eval(b)
    return compare(a,canonical_hash(a),b,canonical_hash(b))


class FairComparisonTest(unittest.TestCase):
    def test_partial_intersection_recomputes_identical_support(self):
        r=compared();self.assertEqual(r['status'],'COMPARED')
        self.assertEqual(len(r['daily']),2)
        for d in r['daily']:
            self.assertEqual(d['stock_ids'],['stock-1','stock-2']);self.assertEqual(d['valid_n'],2)
            self.assertAlmostEqual(d['rank_ic_a'],1.);self.assertAlmostEqual(d['rank_ic_b'],-1.);self.assertAlmostEqual(d['delta_b_minus_a'],-2.)
        self.assertEqual(len(r['exclusions']),4);self.assertEqual(len(r['paired_records']),4)
        self.assertTrue(all('MISSING_PREDICTION' in (e['reasons_a']+e['reasons_b']) for e in r['exclusions']))
        json.dumps(r,allow_nan=False)
    def test_model_identity_and_own_support_are_preserved(self):
        a,b=pair_fixture();ea,eb=run_eval(a),run_eval(b);r=compared(ea,eb)
        self.assertEqual(r['models']['a']['run_id'],'run-A');self.assertEqual(r['models']['b']['run_id'],'run-B')
        self.assertEqual(r['daily'][0]['requested_a'],3);self.assertEqual(r['daily'][0]['union_n'],4)
        self.assertEqual(ea['daily'][0]['valid_n'],3);self.assertEqual(r['daily'][0]['excluded_union_n'],2)
    def test_different_support_means_can_hide_real_paired_difference(self):
        a,b=pair_fixture()
        for row in a[0]['rows']:row['score']={'stock-0':3.,'stock-1':1.,'stock-2':2.}[row['stock_id']]
        for row in b[0]['rows']:row['score']={'stock-1':3.,'stock-2':1.,'stock-3':2.}[row['stock_id']]
        ea,eb=run_eval(a),run_eval(b)
        self.assertAlmostEqual(ea['daily'][0]['rank_ic'],-.5)
        self.assertAlmostEqual(eb['daily'][0]['rank_ic'],-.5)
        r=compared(ea,eb);self.assertAlmostEqual(r['daily'][0]['delta_b_minus_a'],-2.)
    def test_source_pin_or_inner_identity_mismatch(self):
        a,b=pair_fixture();ea,eb=run_eval(a),run_eval(b)
        self.assertIn('SOURCE_ARTIFACT_IDENTITY_MISMATCH',compare(ea,'0'*64,eb,canonical_hash(eb))['reasons'])
        ea['records'][0]['score']=99
        self.assertNotEqual(compared(ea,eb)['status'],'COMPARED')
    def test_fold_calendar_origin_horizon_session_policy_mismatch(self):
        for field,value in [('fold_id','OTHER'),('origin_at','2000-01-01T00:00:00+08:00'),('horizons',[5]),('sessions',[])]:
            a,b=pair_fixture();b[0]['header'][field]=value
            self.assertEqual(compared(run_eval(a),run_eval(b))['status'],'REJECTED')
        a,b=pair_fixture();b[3]['min_valid_n']=3
        self.assertIn('POLICY_IDENTITY_MISMATCH',compared(run_eval(a),run_eval(b))['reasons'])
    def test_label_identity_or_changed_vintage_cannot_pair(self):
        for field,value in [('vintage_id','revised'),('label_contract_id','OTHER'),('source_sha256','8'*64)]:
            a,b=pair_fixture();row=b[0]['rows'][0];m=b[4][row['label_id']];m[field]=value
            row['label_metadata_sha256']=canonical_hash(m)
            r=compared(run_eval(a),run_eval(b));self.assertIn('LABEL_IDENTITY_MISMATCH',r['reasons'])
    def test_stock_mismatch_creates_explicit_exclusion(self):
        a,b=pair_fixture();row=b[0]['rows'][0];row['stock_id']='OTHER'
        r=compared(run_eval(a),run_eval(b));self.assertEqual(r['status'],'COMPARED')
        self.assertTrue(any('LABEL_OWNER_OR_HORIZON_MISMATCH' in x['reasons_b'] for x in r['exclusions']))
    def test_invalid_scores_excluded_before_pairing(self):
        a,b=pair_fixture();b[0]['rows'][0]['score']=float('inf')
        r=compared(run_eval(a),run_eval(b));self.assertEqual(r['daily'][0]['valid_n'],1)
        self.assertIsNone(r['daily'][0]['delta_b_minus_a']);self.assertTrue(r['exclusions'])
    def test_no_intersection_or_no_eligible_sessions(self):
        a,b=pair_fixture();b[0]['rows']=[];r=compared(run_eval(a),run_eval(b))
        self.assertEqual(r['status'],'NO_ELIGIBLE_PAIRED_SESSIONS')
        self.assertTrue(all(d['delta_b_minus_a'] is None for d in r['daily']))
        self.assertTrue(all(x['mean_rank_ic'] is None for x in r['summary']['delta_by_horizon']))
    def test_constant_predictions_delta_remains_undefined(self):
        a,b=pair_fixture()
        for row in b[0]['rows']:row['score']=1.
        r=compared(run_eval(a),run_eval(b));self.assertIsNone(r['daily'][0]['delta_b_minus_a'])
        self.assertIn('B:CONSTANT_RANKS',r['daily'][0]['reasons'])
    def test_monthly_quarterly_use_actual_declared_sessions(self):
        a,b=pair_fixture();day=a[2]['sessions'][31]['session']
        for f in (a,b):f[0]['header']['sessions'].append(day)
        r=compared(run_eval(a),run_eval(b))
        self.assertEqual(len(r['daily']),4)
        for p in r['periods']:
            self.assertEqual(len(p['sessions']),2);self.assertEqual(p['requested_session_horizons'],2)
            self.assertEqual(p['defined_session_horizons'],1)
    def test_selection_forward_role_cannot_be_mixed(self):
        a,b=pair_fixture();r=compared(run_eval(a),run_eval(evaluation_fixture('SELECTION')))
        self.assertEqual(r['status'],'REJECTED')
    def test_no_implicit_horizon_pool_or_confidence_threshold(self):
        r=compared();self.assertNotIn('mean_rank_ic',r['summary']);self.assertNotIn('ci',r)
        self.assertEqual(r['statistical_inference'],'NOT_CONFIGURED_NOT_ADMITTED')
        self.assertEqual(len(r['summary']['delta_by_horizon']),2)
    def test_forward_exposure_requires_later_fold_and_ledger(self):
        a,b=pair_fixture();ev=run_eval(a);ref=exposure_reference(ev,canonical_hash(ev));fold=a[1];cal=a[2]
        denied=expose_forward_result(ref,canonical_hash(ref),fold,cal,make_pin(fold,cal),ev['header']['evaluation_at'],'synthetic-decision',[],canonical_hash([]))
        self.assertIn('SAME_FOLD_FORWARD_CANNOT_SELECT',denied['reasons'])
        # Synthetic prior-fold result usable at this target only with recorded exposure.
        ref.update(source_fold_id='prior-fold',source_origin_at=cal['sessions'][0]['close_at'],available_at=cal['sessions'][1]['close_at'])
        admitted=expose_forward_result(ref,canonical_hash(ref),fold,cal,make_pin(fold,cal),fold['selection_cutoff_at'],'synthetic-decision',[],canonical_hash([]))
        self.assertEqual(admitted['status'],'EXPOSURE_RECORDED')
        self.assertEqual(admitted['ledger'][0]['role'],'DEVELOPMENT_EXPOSED_NOT_FINAL')
    def test_selection_evaluation_cannot_be_forward_exposure(self):
        ev=run_eval(evaluation_fixture('SELECTION'))
        with self.assertRaises(ValueError):exposure_reference(ev,canonical_hash(ev))

if __name__=='__main__':unittest.main()
