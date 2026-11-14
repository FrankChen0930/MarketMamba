"""Tiny CPU engineering fixtures, not financial research."""
import copy,json,unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal_test import fixtures,sample
from V6.experimental.v7_research_temporal import make_pin
from V6.experimental.v7_research_isolation import authorize_view,make_refit_run,make_forward_run
from V6.experimental.v7_classical_baseline import fit_baseline,score_view,dependency_status


def recipe(kind='L1'):
    config={'L0':{'factor':'x','direction':1.},'L1':{'alpha':1.,'scaling':'TRAIN_ONLY_ZSCORE','zero_variance':'UNIT_SCALE','fit_intercept':True},
            'L2':{'n_estimators':4,'num_leaves':3,'max_depth':2,'learning_rate':.1,'min_child_samples':1,'n_jobs':1}}[kind]
    return {'schema':'v7-classical-recipe-v1','status':'SYNTHETIC_ENGINEERING_ONLY','recipe_id':'toy-'+kind,'model_kind':kind,'features':['x','z'],'horizon':5,'seed':0,'max_samples':64,'max_features':4,'config':config}


def panel():
    cal,fold,days=fixtures();fold.update(feature_contract_id='window10-toy-x-z-v1',label_contract_id='window10-toy-label-v1');features={};labels={}
    for index in (1,2,3,15,16,18,30,31):
        for n in range(4):
            f,y=sample(cal,fold,index);stock='toy-'+str(n);sid=stock+'-'+f['signal_session']
            f.update(sample_id=sid,stock_id=stock,values={'x':float(n+index/10),'z':float(n*n-index/20)})
            for h,l in y.items():l.update(sample_id=sid,stock_id=stock,value=(2*f['values']['x']-.1*f['values']['z'])/100+int(h)/10000)
            features[sid]=f;labels[sid]=y
    return cal,fold,days,features,labels


def selection(p,rec=None):
    cal,fold,days,fs,ls=p;rec=rec or recipe()
    return {'schema':'v7-research-run-engineering-v1','kind':'SELECTION','study_id':fold['study_id'],'fold_id':fold['fold_id'],
            'fold_sha256':canonical_hash(fold),'run_id':'selection-'+rec['model_kind'],'recipe_id':rec['recipe_id'],'seed':rec['seed'],'execution':'DECLARED_ONLY_NOT_EXECUTED'}


def ids_for(p,indices):return [sid for sid,f in sorted(p[3].items()) if f['signal_session'] in {p[2][i] for i in indices}]
def pins_for(p,phase):
    return {sid:make_pin(p[1],p[0],f,None if phase=='FORWARD_FEATURES' else p[4][sid]) for sid,f in p[3].items()}
def fit(p,rec=None,run=None,phase='SELECTION_TRAIN',indices=(1,2,3),pins=None):
    rec=rec or recipe();run=run or selection(p,rec)
    return fit_baseline(rec,canonical_hash(rec),run,canonical_hash(run),phase,ids_for(p,indices),p[3],p[4],pins or pins_for(p,phase),p[1],p[0],make_pin(p[1],p[0]))
def view(p,run,phase,indices):
    return authorize_view(run,canonical_hash(run),phase,ids_for(p,indices),p[3],p[4] if phase!='FORWARD_FEATURES' else None,pins_for(p,phase),p[1],p[0],make_pin(p[1],p[0]),model_pin=run.get('model_sha256'))


class BaselineAdapterTest(unittest.TestCase):
    def setUp(self):self.p=panel()
    def test_ridge_fit_and_validation_scoring_serializable(self):
        r=fit(self.p);self.assertEqual(r['status'],'FIT_ENGINEERING_ONLY');run=selection(self.p);v=view(self.p,run,'SELECTION_VALIDATION',(15,16))
        out=score_view(r,canonical_hash(r),v,canonical_hash(v),run,canonical_hash(run));self.assertEqual(out['status'],'SCORED')
        self.assertEqual(len(out['scores']),8);self.assertFalse(out['research_admitted']);json.dumps(out,allow_nan=False)
    def test_l0_direction_explicit_no_financial_default(self):
        rec=recipe('L0');r=fit(self.p,rec);run=selection(self.p,rec);v=view(self.p,run,'SELECTION_VALIDATION',(15,))
        out=score_view(r,canonical_hash(r),v,canonical_hash(v),run,canonical_hash(run))
        self.assertEqual([row['score'] for row in out['scores']],[f['features']['values']['x'] for f in v['view']])
        del rec['config']['direction'];self.assertEqual(fit(self.p,rec)['status'],'REJECTED')
    def test_lightgbm_dependency_gap_is_explicit(self):
        status=dependency_status();r=fit(self.p,recipe('L2'))
        if status['available']:self.assertEqual(r['status'],'FIT_ENGINEERING_ONLY')
        else:self.assertEqual(r['status'],'DEPENDENCY_BLOCKED');self.assertIn('lightgbm',r['reasons'][0])
    def test_missing_and_immature_training_labels(self):
        for mutate in ('missing','late'):
            p=panel();sid=ids_for(p,(1,))[0]
            if mutate=='missing':p[4][sid]={}
            else:p[4][sid]['10']['available_at']='2030-01-01T00:00:00+08:00'
            self.assertEqual(fit(p)['status'],'REJECTED')
    def test_forward_samples_never_fit(self):
        r=fit(self.p,indices=(30,));self.assertEqual(r['status'],'REJECTED');self.assertIsNone(r['model'])
        r=fit(self.p,phase='FORWARD_FEATURES');self.assertIn('FIT_ROLE_FORBIDDEN',r['reasons'])
    def test_future_feature_is_rejected(self):
        sid=ids_for(self.p,(1,))[0];self.p[3][sid]['quality_checked_at']='2030-01-01T00:00:00+08:00'
        self.assertEqual(fit(self.p)['status'],'REJECTED')
    def test_changed_feature_or_label_pin_and_fold_identity(self):
        pins=pins_for(self.p,'SELECTION_TRAIN');sid=ids_for(self.p,(1,))[0]
        self.p[3][sid]['vintage_id']='changed';self.assertEqual(fit(self.p,pins=pins)['status'],'REJECTED')
        p=panel();pins=pins_for(p,'SELECTION_TRAIN');p[4][ids_for(p,(1,))[0]]['5']['value']=99
        self.assertEqual(fit(p,pins=pins)['status'],'REJECTED')
        p=panel();run=selection(p);run['fold_sha256']='0'*64;self.assertEqual(fit(p,run=run)['status'],'REJECTED')
    def test_scaling_uses_only_current_training_view(self):
        r=fit(self.p);mean=r['model']['scaling']['mean'];p=copy.deepcopy(self.p)
        for sid in ids_for(p,(15,16,18,30,31)):p[3][sid]['values']={'x':1e9,'z':-1e9}
        self.assertEqual(fit(p)['model']['scaling']['mean'],mean)
        train=[self.p[3][sid]['values']['x'] for sid in ids_for(self.p,(1,2,3))]
        self.assertAlmostEqual(mean[0],sum(train)/len(train))
    def test_refit_fresh_identity_and_scaling_not_checkpoint_resume(self):
        p=self.p;sel=selection(p);d=p[2][29];ready={k:d+v for k,v in {'model_built_at':'T07:10:00+08:00','artifact_verified_at':'T07:20:00+08:00','approval_at':'T07:30:00+08:00','ready_at':'T07:40:00+08:00'}.items()}
        refit=make_refit_run(sel,'fresh-refit',p[1],ready);r=fit(p,run=refit,phase='REFIT',indices=(1,2,3,15,16,18))
        self.assertEqual(r['status'],'FIT_ENGINEERING_ONLY');self.assertNotEqual(r['model']['scaling']['mean'],fit(p)['model']['scaling']['mean'])
        forward=make_forward_run(refit,'forward',p[1]);v=view(p,forward,'FORWARD_FEATURES',(30,31))
        out=score_view(r,canonical_hash(r),v,canonical_hash(v),forward,canonical_hash(forward));self.assertEqual(out['status'],'SCORED')
        refit['resume_checkpoint']='bad';self.assertEqual(fit(p,run=refit,phase='REFIT')['status'],'REJECTED')
    def test_view_model_and_run_pin_mismatch(self):
        r=fit(self.p);run=selection(self.p);v=view(self.p,run,'SELECTION_VALIDATION',(15,))
        self.assertEqual(score_view(r,'0'*64,v,canonical_hash(v),run,canonical_hash(run))['status'],'REJECTED')
        v['view'][0]['features']['values']['x']=99
        self.assertEqual(score_view(r,canonical_hash(r),v,'0'*64,run,canonical_hash(run))['status'],'REJECTED')
    def test_selection_manifest_same_id_changed_source_cannot_score(self):
        model=fit(self.p);run=selection(self.p);run['selection_snapshot']='different-source-declaration'
        authorized=view(self.p,run,'SELECTION_VALIDATION',(15,))
        r=score_view(model,canonical_hash(model),authorized,canonical_hash(authorized),run,canonical_hash(run))
        self.assertEqual(r['status'],'REJECTED');self.assertIn('SELECTION_MODEL_SCOPE_MISMATCH',r['reasons'])
    def test_invalid_calendar_fold_containers_return_serializable_denial(self):
        for value in (None,[]):
            p=self.p;r=recipe();run=selection(p)
            for cal,fold,fp in [(value,p[1],make_pin(p[1],p[0])),(p[0],value,make_pin(p[1],p[0])),(p[0],p[1],value)]:
                result=fit_baseline(r,canonical_hash(r),run,canonical_hash(run),'SELECTION_TRAIN',ids_for(p,(1,2,3)),p[3],p[4],pins_for(p,'SELECTION_TRAIN'),fold,cal,fp)
                self.assertEqual(result['status'],'REJECTED');json.dumps(result,allow_nan=False)
    def test_no_search_oversized_or_foreign_feature_contract(self):
        rec=recipe();rec['config']['alpha']=[1,2];self.assertEqual(fit(self.p,rec)['status'],'REJECTED')
        rec=recipe();rec['max_samples']=10000;self.assertEqual(fit(self.p,rec)['status'],'REJECTED')
        p=panel();sid=ids_for(p,(1,))[0];p[3][sid]['feature_contract_id']='OTHER';self.assertEqual(fit(p)['status'],'REJECTED')

if __name__=='__main__':unittest.main()
