"""Synthetic evaluator evidence only, no real labels or predictions."""
import copy
import json
import unittest
from V6.experimental.v7_research_temporal_test import fixtures,sample
from V6.experimental.v7_research_temporal import make_pin
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_evaluation import evaluate, prediction_hash


def evaluation_fixture(role='FORWARD',scores=(1.,2.,3.),values=(1.,2.,3.)):
    cal,fold,days=fixtures();idx=30 if role=='FORWARD' else 15;day=days[idx]
    policy={'schema':'v7-ranking-policy-prototype-v1','policy_id':'synthetic-average-equal-v1',
            'status':'SYNTHETIC_NOT_ADMITTED','tie_policy':'AVERAGE','min_valid_n':2,
            'aggregation_weighting':'EQUAL_SESSION','invalid_support':'EXCLUDE_WITH_LEDGER',
            'statistical_policy':'NOT_CONFIGURED_NO_INFERENCE'}
    header={'schema':'v7-ranking-input-v1','study_id':fold['study_id'],'recipe_id':'recipe-A','run_id':'run-A',
            'fold_id':fold['fold_id'],'origin_at':fold['origin_at'],'fold_sha256':canonical_hash(fold),
            'model_sha256':'1'*64,'source_sha256':'2'*64,'snapshot_sha256':'3'*64,'vintage_id':'fixture-v1',
            'evidence_class':'SYNTHETIC','prediction_role':role,
            'evaluation_role':'FORWARD_EVALUATION' if role=='FORWARD' else 'SELECTION_VALIDATION',
            'evaluation_at':days[42]+'T18:00:00+08:00' if role=='FORWARD' else fold['selection_cutoff_at'],
            'sessions':[day],'horizons':[5,10],
            'raw_predictions_ref':{'path':'synthetic://fixture/predictions','sha256':'4'*64}}
    rows=[];metadata={};store={}
    for n,(score,value) in enumerate(zip(scores,values)):
        feature,labels=sample(cal,fold,idx);stock='stock-'+str(n)
        for h in (5,10):
            lid=f'{stock}-{day}-{h}';lab=labels[str(h)];lab.update(stock_id=stock,sample_id=stock+'-'+day,value=value)
            meta={k:v for k,v in lab.items() if k!='value'}
            meta.update(horizon=h,snapshot_sha256=header['snapshot_sha256'],payload_sha256=canonical_hash(lab),eligibility_status='ELIGIBLE_SYNTHETIC',eligibility_ref='5'*64)
            metadata[lid]=meta;store[lid]=lab
            rows.append({'stock_id':stock,'signal_session':day,'horizon':h,'score':score,'label_id':lid,
                         'label_metadata_sha256':canonical_hash(meta),'prediction_at':day+'T18:01:00+08:00',
                         'feature_eligibility_ref':'6'*64})
    bundle={'header':header,'rows':rows}
    return bundle,fold,cal,policy,metadata,store


def run_eval(f):
    b,fold,cal,p,m,s=f
    return evaluate(b,prediction_hash(b),fold,cal,make_pin(fold,cal),p,canonical_hash(p),m,s)


class NoRead(dict):
    def __getitem__(self,key):raise AssertionError('ineligible label read')


class RankingEvaluationTest(unittest.TestCase):
    def test_perfect_positive_negative_and_serialization(self):
        for scores,expected in [((1.,2.,3.),1.),((3.,2.,1.),-1.)]:
            r=run_eval(evaluation_fixture(scores=scores));self.assertEqual(r['status'],'EVALUATED')
            self.assertTrue(all(abs(d['rank_ic']-expected)<1e-12 for d in r['daily']))
            self.assertFalse(r['research_admitted']);json.dumps(r,allow_nan=False)
    def test_ties_reuse_average_rank_semantics(self):
        r=run_eval(evaluation_fixture(scores=(1.,1.,3.),values=(1.,2.,2.)))
        self.assertAlmostEqual(r['daily'][0]['rank_ic'],.5)
    def test_constant_not_zero(self):
        for kw in ({'scores':(1.,1.,1.)},{'values':(1.,1.,1.)}):
            d=run_eval(evaluation_fixture(**kw))['daily'][0]
            self.assertIsNone(d['rank_ic']);self.assertIn('CONSTANT_RANKS',d['reasons'])
    def test_nan_inf_missing_scores_excluded_with_reasons(self):
        for x in (float('nan'),float('inf'),None):
            r=run_eval(evaluation_fixture(scores=(1.,2.,x)))
            self.assertEqual(r['daily'][0]['valid_n'],2);self.assertEqual(len(r['rejections']),2)
            self.assertIn('INVALID_PREDICTION',r['rejections'][0]['reasons']);json.dumps(r,allow_nan=False)
    def test_insufficient_and_no_eligible_support(self):
        f=evaluation_fixture(scores=(1.,),values=(1.,));d=run_eval(f)['daily'][0]
        self.assertIsNone(d['rank_ic']);self.assertIn('INSUFFICIENT_SUPPORT',d['reasons'])
        f=evaluation_fixture(scores=(None,None,None));r=run_eval(f)
        self.assertEqual(r['status'],'NO_ELIGIBLE_SESSIONS');self.assertTrue(all(x['mean_rank_ic'] is None for x in r['summary']['by_horizon']))
    def test_duplicate_rejected_before_reads(self):
        f=list(evaluation_fixture());f[0]['rows'].append(copy.deepcopy(f[0]['rows'][0]));f[5]=NoRead()
        self.assertIn('DUPLICATE_KEY',run_eval(f)['reasons'])
    def test_stock_session_horizon_identity(self):
        for field,value in [('stock_id','OTHER'),('signal_session','1999-01-01'),('horizon',7)]:
            f=evaluation_fixture();f[0]['rows'][0][field]=value;r=run_eval(f)
            self.assertNotEqual(r['status'],'EVALUATED') if field!='stock_id' else self.assertTrue(r['rejections'])
    def test_label_metadata_changed_vintage_and_contract(self):
        for field,value in [('vintage_id','revised'),('label_contract_id','OTHER')]:
            f=evaluation_fixture();lid=f[0]['rows'][0]['label_id'];f[4][lid][field]=value
            self.assertIn('LABEL_METADATA_IDENTITY_MISMATCH',run_eval(f)['rejections'][0]['reasons'])
    def test_label_payload_identity_and_nonfinite(self):
        f=evaluation_fixture();lid=f[0]['rows'][0]['label_id'];f[5][lid]['value']=99.
        self.assertIn('LABEL_PAYLOAD_IDENTITY_MISMATCH',run_eval(f)['rejections'][0]['reasons'])
        f=evaluation_fixture()
        lid=f[0]['rows'][0]['label_id'];f[5][lid]['value']=float('nan')
        self.assertTrue(run_eval(f)['rejections'])
    def test_distinct_5_10_maturity_before_store_access(self):
        f=list(evaluation_fixture());day=f[2]['sessions'][36]['session'];f[0]['header']['evaluation_at']=day+'T18:00:00+08:00'
        ten={r['label_id'] for r in f[0]['rows'] if r['horizon']==10}
        class Guard(dict):
            def __getitem__(self,k):
                if k in ten:raise AssertionError('immature 10d read')
                return super().__getitem__(k)
        f[5]=Guard(f[5]);r=run_eval(f)
        self.assertEqual(r['daily'][0]['valid_n'],3);self.assertEqual(r['daily'][1]['valid_n'],0)
        self.assertTrue(all('LABEL_NOT_MATURE_OR_AVAILABLE' in d['reasons'] for d in r['rejections']))
    def test_unknown_availability_no_read(self):
        f=list(evaluation_fixture())
        for row in f[0]['rows']:
            m=f[4][row['label_id']];del m['available_at'];row['label_metadata_sha256']=canonical_hash(m)
        f[5]=NoRead();r=run_eval(f);self.assertEqual(len(r['rejections']),6)
    def test_selection_cannot_read_forward_labels(self):
        f=list(evaluation_fixture());f[0]['header'].update(prediction_role='SELECTION',evaluation_role='SELECTION_VALIDATION');f[5]=NoRead()
        r=run_eval(f);self.assertIn('SELECTION_AFTER_CUTOFF',r['reasons']);self.assertEqual(r['records'],[])
    def test_selection_validation_and_after_cutoff_block(self):
        f=evaluation_fixture('SELECTION');self.assertEqual(run_eval(f)['status'],'EVALUATED')
        f[0]['header']['evaluation_at']=f[2]['sessions'][42]['session']+'T18:00:00+08:00'
        self.assertIn('SELECTION_AFTER_CUTOFF',run_eval(f)['reasons'])
    def test_unapproved_or_missing_policy_cannot_imply_statistics(self):
        for key,value in [('status','ADMITTED'),('min_valid_n',None),('tie_policy','ORDINAL'),('statistical_policy',{'ci':.95})]:
            f=evaluation_fixture();f[3][key]=value;self.assertEqual(run_eval(f)['status'],'REJECTED')
    def test_future_prediction_or_unknown_source(self):
        f=evaluation_fixture();f[0]['rows'][0]['prediction_at']=f[2]['sessions'][31]['close_at']
        self.assertIn('PREDICTION_TIME_OUTSIDE_ROLE',run_eval(f)['rejections'][0]['reasons'])
        f=evaluation_fixture();f[0]['header']['source_sha256']='UNKNOWN';self.assertEqual(run_eval(f)['status'],'REJECTED')
    def test_input_and_policy_pin_mismatch(self):
        b,fold,cal,p,m,s=evaluation_fixture()
        r=evaluate(b,'0'*64,fold,cal,make_pin(fold,cal),p,canonical_hash(p),m,NoRead())
        self.assertIn('INPUT_IDENTITY_MISMATCH',r['reasons'])
    def test_malformed_payload_and_prediction_containers_fail_closed(self):
        for value in (None,[],42):
            f=evaluation_fixture();f[5][f[0]['rows'][0]['label_id']]=value
            r=run_eval(f);self.assertIn('INVALID_LABEL_PAYLOAD_CONTAINER',r['rejections'][0]['reasons'])
        for rows in ([None],None,[[]]):
            b,fold,cal,p,m,store=evaluation_fixture();b['rows']=rows
            r=evaluate(b,'0'*64,fold,cal,make_pin(fold,cal),p,canonical_hash(p),m,NoRead())
            self.assertEqual(r['status'],'REJECTED');json.dumps(r,allow_nan=False)
    def test_summary_never_pools_5d_and_10d_without_policy(self):
        f=evaluation_fixture()
        for row in f[0]['rows']:
            if row['horizon']==10:row['score']=-row['score']
        r=run_eval(f);self.assertNotIn('mean_rank_ic',r['summary'])
        by={x['horizon']:x['mean_rank_ic'] for x in r['summary']['by_horizon']}
        self.assertAlmostEqual(by[5],1.);self.assertAlmostEqual(by[10],-1.)
    def test_empty_declared_sessions_and_weighting(self):
        f=evaluation_fixture();f[0]['rows']=[];r=run_eval(f)
        self.assertEqual(len(r['daily']),2);self.assertEqual(r['status'],'NO_ELIGIBLE_SESSIONS')
        f=evaluation_fixture();f[3]['aggregation_weighting']='VALID_STOCK_COUNT';self.assertEqual(run_eval(f)['status'],'EVALUATED')

if __name__=='__main__':unittest.main()
