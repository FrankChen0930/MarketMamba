"""Paired common-support adapter to the common ranking evaluator; synthetic only.

Reads pinned evaluation artifacts, never label stores. No statistical/selection
admission, independence claim, or comparison of different-support mean ICs.
"""
from copy import deepcopy
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_evaluation import (OUTPUT_SCHEMA,check_policy,key,daily_metric,aggregate)

COMPARISON_SCHEMA='v7-ranking-comparison-v1'


def _artifact_issue(value,expected):
    try:
        if canonical_hash(value)!=expected:return 'SOURCE_ARTIFACT_IDENTITY_MISMATCH'
        if value['schema']!=OUTPUT_SCHEMA or value['research_admitted'] is not False:return 'SOURCE_SCHEMA_OR_ADMISSION_MISMATCH'
        if value['status'] not in {'EVALUATED','NO_ELIGIBLE_SESSIONS'}:return 'SOURCE_EVALUATION_REJECTED'
        content={k:v for k,v in value.items() if k!='evaluation_identity'}
        if canonical_hash(content)!=value['evaluation_identity']:return 'SOURCE_INNER_IDENTITY_MISMATCH'
        if not check_policy(value['policy'],value['policy_sha256']):return 'SOURCE_POLICY_UNSUPPORTED'
        for name in ('requested','records'):
            keys=[key(r) for r in value[name]]
            if len(set(keys))!=len(keys):return 'SOURCE_DUPLICATE_KEY'
    except (KeyError,TypeError,ValueError):return 'SOURCE_MALFORMED'
    return None


def _paired_summary(daily,policy):
    """All model means and delta use sessions where BOTH metrics are defined."""
    groups=[]
    for h in sorted({d['horizon'] for d in daily}):
        ds=[{**d,'rank_ic':d['delta_b_minus_a']} for d in daily if d['horizon']==h]
        paired=[d for d in daily if d['horizon']==h and d['delta_b_minus_a'] is not None]
        result={'horizon':h,**aggregate(ds,policy)}
        for model in ('a','b'):
            model_ds=[{**d,'rank_ic':d['rank_ic_'+model]} for d in paired]
            result['mean_rank_ic_'+model]=aggregate(model_ds,policy)['mean_rank_ic']
        groups.append(result)
    return {'delta_by_horizon':groups}


def compare(a,expected_a_sha256,b,expected_b_sha256):
    base={'schema':COMPARISON_SCHEMA,'research_admitted':False,'daily':[],'paired_records':[],
          'exclusions':[],'periods':[],'summary':{'delta_by_horizon':[]},'statistical_inference':'NOT_CONFIGURED_NOT_ADMITTED'}
    def fail(reason):return {**base,'status':'REJECTED','reasons':[reason]}
    for value,expected in ((a,expected_a_sha256),(b,expected_b_sha256)):
        issue=_artifact_issue(value,expected)
        if issue:return fail(issue)
    try:
        if a['policy_sha256']!=b['policy_sha256']:return fail('POLICY_IDENTITY_MISMATCH')
        if any(a[k]!=b[k] for k in ('fold_sha256','calendar_sha256','label_contract_id')):return fail('FOLD_CALENDAR_OR_LABEL_CONTRACT_MISMATCH')
        ha,hb=a['header'],b['header']
        for field in ('study_id','fold_id','origin_at','sessions','horizons','snapshot_sha256','vintage_id','prediction_role','evaluation_role'):
            if ha[field]!=hb[field]:return fail('COMPARISON_CONTEXT_MISMATCH:'+field)
        if ha['run_id']==hb['run_id']:return fail('DISTINCT_RUN_IDENTITIES_REQUIRED')
        requested_a={key(r):r for r in a['requested']};requested_b={key(r):r for r in b['requested']}
        for k in requested_a.keys()&requested_b.keys():
            if any(requested_a[k][name]!=requested_b[k][name] for name in ('label_id','label_metadata_sha256')):return fail('LABEL_IDENTITY_MISMATCH')
        valid_a={key(r):r for r in a['records']};valid_b={key(r):r for r in b['records']}
        rejected_a={key(r):r['reasons'] for r in a['rejections']};rejected_b={key(r):r['reasons'] for r in b['rejections']}
        common=valid_a.keys()&valid_b.keys();union=requested_a.keys()|requested_b.keys()
        if not set(valid_a)<=set(requested_a) or not set(valid_b)<=set(requested_b):return fail('SOURCE_SUPPORT_INCONSISTENT')
        for k in common:
            if valid_a[k]['label']!=valid_b[k]['label'] or valid_a[k]['label_metadata']!=valid_b[k]['label_metadata']:return fail('LABEL_VALUE_OR_VINTAGE_MISMATCH')
        exclusions=[]
        for k in sorted(union-common):
            item={'signal_session':k[0],'stock_id':k[1],'horizon':k[2]}
            for model,requested,valid,rejected in [('a',requested_a,valid_a,rejected_a),('b',requested_b,valid_b,rejected_b)]:
                item['reasons_'+model]=['MISSING_PREDICTION'] if k not in requested else ([] if k in valid else rejected.get(k,['NO_ELIGIBLE_LABEL']))
                item['label_reference_'+model]=deepcopy(requested.get(k))
            exclusions.append(item)
        policy=a['policy'];daily=[];paired=[]
        records_a=[valid_a[k] for k in sorted(common)];records_b=[valid_b[k] for k in sorted(common)]
        for k in sorted(common):
            paired.append({**{name:valid_a[k][name] for name in ('stock_id','signal_session','horizon','label_id','label_metadata_sha256')},
                           'score_a':valid_a[k]['score'],'score_b':valid_b[k]['score'],'label':valid_a[k]['label']})
        for session in ha['sessions']:
            for h in ha['horizons']:
                da,db=(daily_metric(records,session,h,policy) for records in (records_a,records_b))
                if da['support_sha256']!=db['support_sha256']:return fail('INTERNAL_PAIRED_SUPPORT_MISMATCH')
                va,vb=da['rank_ic'],db['rank_ic']
                du={'signal_session':session,'horizon':h,'valid_n':da['valid_n'],'stock_ids':da['stock_ids'],
                    'support_sha256':da['support_sha256'],'rank_ic_a':va,'rank_ic_b':vb,
                    'delta_b_minus_a':vb-va if va is not None and vb is not None else None,
                    'reasons':['A:'+x for x in da['reasons']]+['B:'+x for x in db['reasons']]}
                for name,keys in [('requested_a',requested_a),('requested_b',requested_b),('union_n',union)]:
                    du[name]=sum(k[0]==session and k[2]==h for k in keys)
                du['excluded_union_n']=du['union_n']-du['valid_n']
                daily.append(du)
        period_groups={}
        for d in daily:
            month=d['signal_session'][:7];quarter=month[:4]+'-Q'+str((int(month[5:])-1)//3+1)
            for kind,period in [('MONTH',month),('QUARTER',quarter)]:period_groups.setdefault((kind,period,d['horizon']),[]).append(d)
        periods=[]
        for (kind,period,h),ds in sorted(period_groups.items()):
            periods.append({'kind':kind,'period':period,'horizon':h,'sessions':[d['signal_session'] for d in ds],
                            **_paired_summary(ds,policy)['delta_by_horizon'][0]})
        out={**base,'status':'COMPARED' if any(d['delta_b_minus_a'] is not None for d in daily) else 'NO_ELIGIBLE_PAIRED_SESSIONS',
             'reasons':[],'daily':daily,'paired_records':paired,'exclusions':exclusions,'periods':periods,'summary':_paired_summary(daily,policy),
             'models':{'a':deepcopy(ha),'b':deepcopy(hb)},'source_artifact_sha256':{'a':expected_a_sha256,'b':expected_b_sha256},
             'source_evaluation_identities':{'a':a['evaluation_identity'],'b':b['evaluation_identity']},
             'policy':deepcopy(policy),'policy_sha256':a['policy_sha256'],'fold_sha256':a['fold_sha256'],
             'calendar_sha256':a['calendar_sha256'],'label_contract_id':a['label_contract_id']}
        out['comparison_identity']=canonical_hash(out)
        return out
    except (KeyError,TypeError,ValueError):return fail('MALFORMED_COMPARISON_INPUT')
