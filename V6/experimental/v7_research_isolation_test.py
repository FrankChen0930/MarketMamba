import copy
import json
import unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal import make_pin
from V6.experimental.v7_research_temporal_test import fixtures,sample
from V6.experimental.v7_research_isolation import make_refit_run,make_forward_run,authorize_view,expose_forward_result

class NoLabelReads(dict):
    def __getitem__(self,key):raise AssertionError('Forward read label store')

class MockModel:
    def predict(self,view):
        assert all(set(row)=={'features'} for row in view)
        return [sum(row['features']['values'].values()) for row in view]

class ResearchIsolationTest(unittest.TestCase):
    def setUp(self):
        self.cal,self.fold,self.days=fixtures()
        self.selection={'schema':'v7-research-run-engineering-v1','kind':'SELECTION','study_id':self.fold['study_id'],'fold_id':self.fold['fold_id'],
            'fold_sha256':canonical_hash(self.fold),'run_id':'selection-A','recipe_id':'recipe-explicit','seed':17,'execution':'DECLARED_ONLY_NOT_EXECUTED'}
        self.ready={k:self.days[29]+v for k,v in {'model_built_at':'T07:10:00+08:00','artifact_verified_at':'T07:20:00+08:00','approval_at':'T07:30:00+08:00','ready_at':'T07:40:00+08:00'}.items()}
        self.refit=make_refit_run(self.selection,'refit-A',self.fold,self.ready)
        self.forward=make_forward_run(self.refit,'forward-A',self.fold)
    def access(self,role='SELECTION_TRAIN',index=2,run=None,labels_override=None,pin_override=None,model_pin=None):
        x,y=sample(self.cal,self.fold,index);sid=x['sample_id']
        runs={'SELECTION_TRAIN':self.selection,'SELECTION_VALIDATION':self.selection,'REFIT':self.refit,'FORWARD_FEATURES':self.forward}
        run=run or runs[role]
        return authorize_view(run,canonical_hash(run),role,[sid],{sid:x},labels_override if labels_override is not None else {sid:y},
            {sid:pin_override or make_pin(self.fold,self.cal,x,None if role=='FORWARD_FEATURES' else y)},self.fold,self.cal,make_pin(self.fold,self.cal),
            model_pin=model_pin or canonical_hash(self.refit))
    def test_selection_reads_only_training_or_validation_eligible_views(self):
        for role,index in [('SELECTION_TRAIN',2),('SELECTION_VALIDATION',16)]:
            r=self.access(role,index);self.assertEqual(r['status'],'ELIGIBLE');self.assertIn('labels',r['view'][0])
            self.assertEqual(json.loads(json.dumps(r)),r)
    def test_forward_labels_never_reach_selection_even_if_fake_mature(self):
        for role in ('SELECTION_TRAIN','SELECTION_VALIDATION','REFIT'):
            r=self.access(role,30,labels_override=NoLabelReads())
            self.assertEqual(r['status'],'REJECTED');self.assertEqual(r['view'],[])
            self.assertIn('OUTSIDE_ROLE_RANGE',r['decisions'][0]['reasons'])
    def test_forward_features_only_no_label_store_access_mock_predictions(self):
        r=self.access('FORWARD_FEATURES',30,labels_override=NoLabelReads())
        self.assertEqual(r['status'],'ELIGIBLE');self.assertEqual(MockModel().predict(r['view']),[0.])
        self.assertFalse(r['research_admitted'])
    def test_refit_has_new_id_fresh_state_and_mature_newer_samples(self):
        self.assertNotEqual(self.refit['run_id'],self.selection['run_id'])
        self.assertTrue(all(v=='NEW' for v in self.refit['initialization'].values()))
        self.assertEqual(self.access('REFIT',18)['status'],'ELIGIBLE')
        bad=make_refit_run(self.selection,self.selection['run_id'],self.fold,self.ready)
        self.assertIn('REFIT_ID_NOT_NEW',self.access('REFIT',18,bad)['reasons'])
        bad=copy.deepcopy(self.refit);bad['resume_checkpoint']='prior.pt'
        self.assertIn('REFIT_IS_NOT_RESUME',self.access('REFIT',18,bad)['reasons'])
    def test_forward_requires_refit_model_ready_before_deadline(self):
        bad=copy.deepcopy(self.forward);bad['model']=copy.deepcopy(self.selection)
        self.assertNotEqual(self.access('FORWARD_FEATURES',30,bad)['status'],'ELIGIBLE')
        late=copy.deepcopy(self.refit);late['readiness']['ready_at']=self.days[29]+'T08:01:00+08:00'
        f=make_forward_run(late,'late-forward',self.fold)
        self.assertIn('MODEL_READY_DEADLINE_MISSED',self.access('FORWARD_FEATURES',30,f,model_pin=canonical_hash(late))['reasons'])
    def test_changed_run_fold_or_vintage_is_closed(self):
        x,y=sample(self.cal,self.fold);sid=x['sample_id'];pin=make_pin(self.fold,self.cal,x,y)
        x['vintage_id']='changed'
        r=authorize_view(self.selection,'0'*64,'SELECTION_TRAIN',[sid],{sid:x},{sid:y},{sid:pin},self.fold,self.cal,make_pin(self.fold,self.cal))
        self.assertEqual(r['status'],'REJECTED');self.assertEqual(r['view'],[])
        changed=copy.deepcopy(self.refit);changed['fold_id']='another-fold'
        self.assertNotEqual(self.access('REFIT',18,changed)['status'],'ELIGIBLE')
    def test_batch_failure_is_atomic_no_partial_eligible_values(self):
        a,ya=sample(self.cal,self.fold,2);b,yb=sample(self.cal,self.fold,30);sa,sb=a['sample_id'],b['sample_id']
        r=authorize_view(self.selection,canonical_hash(self.selection),'SELECTION_TRAIN',[sa,sb],{sa:a,sb:b},{sa:ya,sb:yb},
            {sa:make_pin(self.fold,self.cal,a,ya),sb:make_pin(self.fold,self.cal,b,yb)},self.fold,self.cal,make_pin(self.fold,self.cal))
        self.assertEqual(r['status'],'REJECTED');self.assertEqual(r['view'],[]);self.assertEqual(len(r['decisions']),2)
    def test_same_sample_different_fold_role_and_exposure(self):
        from V6.experimental.v7_research_temporal import check_sample
        f=copy.deepcopy(self.fold);f.update(fold_id='fold-B',train=[self.days[0],self.days[18]],validation=[self.days[30],self.days[31]],refit=[self.days[0],self.days[31]],forward=[self.days[42],self.days[43]],
          selection_cutoff_at=self.days[41]+'T18:00:00+08:00',refit_cutoff_at=self.days[41]+'T19:00:00+08:00',model_ready_deadline_at=self.days[41]+'T20:00:00+08:00',origin_at=self.days[41]+'T21:00:00+08:00')
        x,y=sample(self.cal,f,30)
        self.assertEqual(check_sample(x,y,f,self.cal,make_pin(f,self.cal,x,y),'SELECTION_VALIDATION')['status'],'ELIGIBLE')
        result={'source_study_id':self.fold['study_id'],'source_fold_id':self.fold['fold_id'],'source_run_id':self.forward['run_id'],'source_origin_at':self.fold['origin_at'],
           'available_at':self.days[41]+'T15:00:00+08:00','source_sha256':'d'*64,'evidence_class':'SYNTHETIC','payload':{'mock_result':1}}
        ledger=[]
        r=expose_forward_result(result,canonical_hash(result),f,self.cal,make_pin(f,self.cal),self.days[41]+'T16:00:00+08:00','research-choice-1',ledger,canonical_hash(ledger))
        self.assertEqual(r['status'],'EXPOSURE_RECORDED');self.assertEqual(ledger,[])
        self.assertEqual(r['ledger'][-1]['role'],'DEVELOPMENT_EXPOSED_NOT_FINAL');self.assertEqual(len(r['ledger']),1)
        self.assertEqual(json.loads(json.dumps(r)),r)
        bad=expose_forward_result(result,canonical_hash(result),self.fold,self.cal,make_pin(self.fold,self.cal),self.days[41]+'T16:00:00+08:00','samefold',[],canonical_hash([]))
        self.assertEqual(bad['status'],'REJECTED')
    def test_unknown_or_malformed_forward_source_hash_is_closed(self):
        for source in ('UNKNOWN','a'*63,'g'*64,None,True):
            result={'source_study_id':'old-study','source_fold_id':'old-fold','source_run_id':'old-forward',
                'source_origin_at':self.days[0]+'T08:00:00+08:00','available_at':self.days[2]+'T14:00:00+08:00',
                'source_sha256':source,'evidence_class':'SYNTHETIC','payload':{'mock':1}}
            r=expose_forward_result(result,canonical_hash(result),self.fold,self.cal,make_pin(self.fold,self.cal),
                self.days[3]+'T18:00:00+08:00','choice',[],canonical_hash([]))
            self.assertNotEqual(r['status'],'EXPOSURE_RECORDED')
            self.assertIsNone(r['result']);self.assertIsNone(r['ledger'])

    def test_changed_label_vintage_and_exposure_roots_are_rejected(self):
        x,y=sample(self.cal,self.fold);sid=x['sample_id'];pin=make_pin(self.fold,self.cal,x,y)
        y['10']['vintage_id']='later-revision'
        r=authorize_view(self.selection,canonical_hash(self.selection),'SELECTION_TRAIN',[sid],{sid:x},{sid:y},{sid:pin},self.fold,self.cal,make_pin(self.fold,self.cal))
        self.assertEqual(r['view'],[]);self.assertIn('LABEL_IDENTITY_MISMATCH',r['decisions'][0]['reasons'])
        result={'source_study_id':'old-study','source_fold_id':'old-fold','source_run_id':'old-forward',
                'source_origin_at':self.days[0]+'T08:00:00+08:00','available_at':self.days[2]+'T14:00:00+08:00',
                'source_sha256':'d'*64,'evidence_class':'SYNTHETIC','payload':{'mock':1}}
        for result_pin,ledger,ledger_pin in [('0'*64,[],canonical_hash([])),(canonical_hash(result),[{'changed':1}],canonical_hash([]))]:
            r=expose_forward_result(result,result_pin,self.fold,self.cal,make_pin(self.fold,self.cal),self.days[3]+'T18:00:00+08:00','choice',ledger,ledger_pin)
            self.assertEqual(r['status'],'REJECTED');self.assertIsNone(r['result']);self.assertIsNone(r['ledger'])

    def test_missing_source_or_pin_never_yields_view(self):
        x,y=sample(self.cal,self.fold);sid=x['sample_id']
        r=authorize_view(self.selection,canonical_hash(self.selection),'SELECTION_TRAIN',[sid],{sid:x},{sid:y},{},self.fold,self.cal,make_pin(self.fold,self.cal))
        self.assertEqual(r['status'],'BLOCKED');self.assertEqual(r['view'],[])
