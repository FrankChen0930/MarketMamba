"""Window07 synthetic calendar, not a certified historical TW calendar."""
import copy
from datetime import date,timedelta
import unittest
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal import check_fold,check_sample,make_pin


def fixtures():
    days=[];d=date(2024,1,2)
    while len(days)<45:
        if d.weekday()<5 and d.isoformat()!='2024-01-15':days.append(d.isoformat())
        d+=timedelta(days=1)
    cal={'sessions':[{'session':d,'open_at':d+'T09:00:00+08:00','close_at':d+'T13:30:00+08:00'} for d in days],
         'source_sha256':'c'*64,'vintage_id':'fixture-v1','qualification':'SYNTHETIC','qualification_ref':'window07-fixture-not-real-PIT'}
    fold={'schema':'v7-research-fold-engineering-v1','study_id':'study-fixture','fold_id':'fold-A','calendar_sha256':canonical_hash(cal),
          'feature_contract_id':'feature-fixture-v1','label_contract_id':'label-fixture-next-open-v1','label_definition':'NEXT_SESSION_OPEN_HOLD_H_CLOSE',
          'required_horizons':[5,10], 'train':[days[0],days[3]],'validation':[days[15],days[17]],'refit':[days[0],days[18]],'forward':[days[30],days[35]],
          'selection_cutoff_at':days[28]+'T18:00:00+08:00','refit_cutoff_at':days[29]+'T07:00:00+08:00',
          'model_ready_deadline_at':days[29]+'T08:00:00+08:00','origin_at':days[29]+'T08:30:00+08:00',
          'purge_sessions':0,'embargo_sessions':0,'unresolved_policies':[],'policy_status':'SYNTHETIC_PARAMETERS_NOT_APPROVED'}
    return cal,fold,days


def sample(cal,fold,index=2):
    ss=cal['sessions'];d=ss[index]['session'];entry=ss[index+1]['session']
    feature={'sample_id':'A-'+d,'stock_id':'A','signal_session':d,'entry_session':entry,'feature_contract_id':fold['feature_contract_id'],
             'source_sha256':'a'*64,'vintage_id':'feature-original','observation_at':d+'T13:30:00+08:00',
             'published_at':d+'T14:00:00+08:00','source_available_at':d+'T14:01:00+08:00','retrieved_at':d+'T14:02:00+08:00',
             'quality_checked_at':d+'T14:03:00+08:00','quality_status':'PASS','freeze_at':d+'T18:00:00+08:00',
             'decision_at':d+'T18:01:00+08:00','values':{'x':0.}}
    labels={}
    for h in fold['required_horizons']:
        exit=ss[index+h]['session']
        labels[str(h)]={'sample_id':feature['sample_id'],'stock_id':feature['stock_id'],'signal_session':d,'entry_session':entry,'entry_at':ss[index+1]['open_at'],'exit_session':exit,'exit_at':ss[index+h]['close_at'],'mature_at':ss[index+h]['close_at'],
           'available_at':exit+'T14:00:00+08:00','quality_status':'PASS','value':.01,'source_sha256':'b'*64,
           'vintage_id':'label-original','label_contract_id':fold['label_contract_id']}
    return feature,labels


class TemporalFoundationTest(unittest.TestCase):
    def setUp(self):self.cal,self.fold,self.days=fixtures()
    def checked(self,feature=None,labels=None,phase='SELECTION_TRAIN',fold=None,cal=None,pin=None):
        c=cal or self.cal;f=fold or self.fold
        if feature is None:feature,labels=sample(c,f)
        return check_sample(feature,labels,f,c,pin or make_pin(f,c,feature,labels),phase)
    def test_labels_are_bound_to_stock_sample_and_signal_per_horizon(self):
        for h in ('5','10'):
            for key in ('stock_id','sample_id','signal_session'):
                x,y=sample(self.cal,self.fold);y[h][key]='OTHER'
                self.assertIn('LABEL_SAMPLE_IDENTITY_MISMATCH',self.checked(x,y)['reasons'])
                x,y=sample(self.cal,self.fold);del y[h][key]
                self.assertEqual(self.checked(x,y)['status'],'BLOCKED')

    def test_valid_fold_sample_and_serializable_decision(self):
        import json
        self.assertEqual(check_fold(self.fold,self.cal,make_pin(self.fold,self.cal))['status'],'ELIGIBLE')
        r=self.checked();self.assertEqual(r['status'],'ELIGIBLE');self.assertFalse(r['research_admitted'])
        self.assertEqual(json.loads(json.dumps(r)),r)
    def test_each_unmatured_required_head_is_rejected(self):
        for h in (5,10):
            x,y=sample(self.cal,self.fold);y[str(h)]['available_at']=self.days[29]+'T14:00:00+08:00'
            self.assertIn('LABEL_NOT_AVAILABLE_AT_CUTOFF',self.checked(x,y)['reasons'])
    def test_missing_availability_or_qc_is_blocked_not_zero(self):
        for key in ('published_at','source_available_at','retrieved_at','quality_checked_at'):
            x,y=sample(self.cal,self.fold);x[key]=None
            self.assertEqual(self.checked(x,y)['status'],'BLOCKED')
        x,y=sample(self.cal,self.fold);x['quality_status']='FAIL'
        self.assertEqual(self.checked(x,y)['status'],'REJECTED')
    def test_session_calendar_skips_weekend_and_explicit_holiday(self):
        i=next(i for i,s in enumerate(self.cal['sessions']) if s['session']=='2024-01-12')
        x,y=sample(self.cal,self.fold,i);self.assertEqual(x['entry_session'],'2024-01-16')
        x['entry_session']='2024-01-15'
        self.assertIn('ENTRY_NOT_NEXT_SESSION',self.checked(x,y,phase='REFIT')['reasons'])
    def test_train_validation_label_information_overlap_is_excluded(self):
        f=copy.deepcopy(self.fold);f['train'][1]=self.days[6]
        x,y=sample(self.cal,f,6)
        self.assertIn('LABEL_INTERVAL_OVERLAP',self.checked(x,y,fold=f)['reasons'])
    def test_future_feature_and_future_label_values_are_rejected(self):
        x,y=sample(self.cal,self.fold);x['freeze_at']=self.days[29]+'T19:00:00+08:00';x['decision_at']=self.days[29]+'T19:01:00+08:00'
        self.assertNotEqual(self.checked(x,y)['status'],'ELIGIBLE')
        x,y=sample(self.cal,self.fold);pin=make_pin(self.fold,self.cal,x,y);y['10']['value']=float('inf')
        self.assertNotEqual(self.checked(x,y,pin=pin)['status'],'ELIGIBLE')
    def test_identity_and_changed_vintage_cannot_be_rehashed_into_old_pin(self):
        x,y=sample(self.cal,self.fold);pin=make_pin(self.fold,self.cal,x,y);x['vintage_id']='revised'
        self.assertIn('FEATURE_IDENTITY_MISMATCH',self.checked(x,y,pin=pin)['reasons'])
        f=copy.deepcopy(self.fold);f['feature_contract_id']='changed'
        self.assertEqual(check_fold(f,self.cal,pin)['status'],'REJECTED')
    def test_unknown_policy_and_calendar_fail_closed(self):
        f=copy.deepcopy(self.fold);f['purge_sessions']=None
        self.assertEqual(check_fold(f,self.cal,make_pin(f,self.cal))['status'],'BLOCKED')
        f=copy.deepcopy(self.fold);f['unresolved_policies']=['maturity']
        self.assertEqual(check_fold(f,self.cal,make_pin(f,self.cal))['status'],'BLOCKED')
        c=copy.deepcopy(self.cal);c['qualification']='UNKNOWN';f=copy.deepcopy(self.fold);f['calendar_sha256']=canonical_hash(c)
        self.assertEqual(check_fold(f,c,make_pin(f,c))['status'],'BLOCKED')
    def test_purge_embargo_are_explicit_session_gaps_not_panel_rows(self):
        f=copy.deepcopy(self.fold);f['purge_sessions']=2;f['embargo_sessions']=1
        self.assertIn('REQUIRED_SESSION_GAP',self.checked(fold=f)['reasons'])
        self.assertNotEqual(f['purge_sessions'],30)
    def test_legal_newer_refit_sample_and_maturity(self):
        x,y=sample(self.cal,self.fold,18)
        self.assertEqual(self.checked(x,y,'REFIT')['status'],'ELIGIBLE')
        self.assertIn('OUTSIDE_ROLE_RANGE',self.checked(x,y,'SELECTION_TRAIN')['reasons'])
    def test_forward_role_uses_features_only_and_after_origin(self):
        x,_=sample(self.cal,self.fold,30)
        self.assertEqual(self.checked(x,None,'FORWARD_FEATURES')['status'],'ELIGIBLE')
        self.assertIn('OUTSIDE_ROLE_RANGE',self.checked(x,None,'SELECTION_VALIDATION')['reasons'])
    def test_model_ready_deadline_and_naive_clock_are_rejected(self):
        f=copy.deepcopy(self.fold);f['model_ready_deadline_at']=self.days[30]+'T10:00:00+08:00'
        self.assertEqual(check_fold(f,self.cal,make_pin(f,self.cal))['status'],'REJECTED')
        x,y=sample(self.cal,self.fold);x['observation_at']=self.days[2]+'T13:30:00'
        self.assertEqual(self.checked(x,y)['status'],'BLOCKED')
