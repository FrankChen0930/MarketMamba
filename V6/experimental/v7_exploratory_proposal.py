"""Preparation-only Track A identity gate. Not a trainer/admission decision."""
from V6.experimental.v7_corrected_e5_replication import canonical_hash

def inspect_proposal(study,dataset,features,expected_sha256):
    base={'schema':'v7-exploratory-preparation-verdict-v1','DATA_READY':False,'execution_authorized':False,'research_admitted':False}
    def reject(reason):return {**base,'status':'REJECTED','reasons':[reason]}
    try:
        values=[study,dataset,features]
        if any(not isinstance(x,dict) for x in values) or canonical_hash(values)!=expected_sha256:return reject('PROPOSAL_PIN_MISMATCH')
        schemas=('v7-exploratory-study-proposal-v1','v7-exploratory-dataset-proposal-v1','v7-exploratory-feature-proposal-v1')
        if any(x.get('schema')!=s or x.get('status')!='PROPOSED' for x,s in zip(values,schemas)):return reject('PROPOSED_SCHEMAS_ONLY')
        if study.get('track')!='A_EXPLORATORY_HISTORICAL' or dataset.get('track')!=study['track']:return reject('TRACK_ISOLATION_MISMATCH')
        for x,key in ((study,'study_id'),(dataset,'dataset_id'),(features,'feature_manifest_id')):
            if not isinstance(x.get(key),str) or not x[key].startswith('v7-exploratory-'):return reject('INDEPENDENT_NAMESPACE_REQUIRED')
        if any(study.get(k) is not False for k in ('human_research_authorized','execution_authorized','independent_final_test_claim','e5_e1_replication_claim')) or dataset.get('admitted') is not False or features.get('admitted') is not False:return reject('NO_AUTHORITY_OR_FINAL_PROMOTION')
        if study.get('evidence_class')!='EXPLORATORY_PREPARATION_NOT_PIT_CONFIRMATION' or study.get('conservative_lag_sessions') is not None:return reject('NO_PIT_OR_LAG_APPROVAL')
        if study.get('dataset_id')!=dataset['dataset_id'] or study.get('dataset_spec_sha256')!=canonical_hash(dataset) or study.get('feature_manifest_id')!=features['feature_manifest_id'] or study.get('feature_spec_sha256')!=canonical_hash(features):return reject('DATASET_FEATURE_SPEC_IDENTITY_MISMATCH')
        if not isinstance(study.get('parameters'),dict) or any(v is not None for v in study['parameters'].values()):return reject('UNFROZEN_PARAMETERS_REQUIRED')
        return {**base,'status':'PREPARATION_ONLY','study_id':study['study_id'],'dataset_id':dataset['dataset_id'],'feature_spec_sha256':canonical_hash(features),'reasons':['NO_QUALIFIED_BYTES_POLICY_OR_REAL_RESEARCH_AUTHORIZATION']}
    except (KeyError,ValueError,TypeError):return reject('MALFORMED_PROPOSAL')
