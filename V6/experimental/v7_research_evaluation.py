"""Common CPU ranking adapter over the frozen E5 average-tie RankIC primitive.

Synthetic engineering only. Caller-pinned claims do not authenticate provenance.
Prediction feature eligibility references must come from the W07 guarded workflow;
this evaluator validates label access, not upstream model execution or data admission.
Selection callers receive only validation-scoped labels through this API. Raw-store
access outside this API remains outside the Python library's security boundary.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
import torch
from V6.experimental.v7_corrected_e5_train import _rank_ic
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal import check_fold, instant, role_check, _sha, _text, _number

INPUT_SCHEMA='v7-ranking-input-v1'
OUTPUT_SCHEMA='v7-ranking-evaluation-v1'
POLICY_SCHEMA='v7-ranking-policy-prototype-v1'
ROW_FIELDS={'stock_id','signal_session','horizon','score','label_id','label_metadata_sha256','prediction_at','feature_eligibility_ref'}


def prediction_hash(bundle):
    """Canonical semantic input hash. Nonfinite scores have explicit typed tags.

    Not a raw file checksum: raw_predictions_ref retains that separate identity.
    Other non-JSON values are rejected, never coerced into valid observations.
    """
    if not isinstance(bundle,Mapping) or not isinstance(bundle.get('rows'),list):raise ValueError('prediction container')
    safe=deepcopy(bundle)
    for r in safe['rows']:
        if not isinstance(r,Mapping):raise ValueError('prediction row container')
        v=r.get('score')
        if type(v) is float and not math.isfinite(v):
            r['score']={'invalid_float':repr(v)}
    return canonical_hash(safe)


def check_policy(policy,expected_sha256):
    try:
        return (canonical_hash(policy)==expected_sha256 and policy['schema']==POLICY_SCHEMA
                and _text(policy['policy_id']) and policy['status']=='SYNTHETIC_NOT_ADMITTED'
                and policy['tie_policy']=='AVERAGE' and type(policy['min_valid_n']) is int
                and policy['min_valid_n']>=2 and policy['aggregation_weighting'] in {'EQUAL_SESSION','VALID_STOCK_COUNT'}
                and policy['invalid_support']=='EXCLUDE_WITH_LEDGER'
                and policy['statistical_policy']=='NOT_CONFIGURED_NO_INFERENCE')
    except (KeyError,TypeError,ValueError):return False


def key(row):return (row['signal_session'],row['stock_id'],row['horizon'])


def daily_metric(records,session,horizon,policy):
    support=sorted((r for r in records if r['signal_session']==session and r['horizon']==horizon),key=lambda r:r['stock_id'])
    n=len(support);value=None;reasons=[]
    if n<policy['min_valid_n']:reasons=['INSUFFICIENT_SUPPORT']
    else:
        score=torch.tensor([r['score'] for r in support],dtype=torch.float64)
        label=torch.tensor([r['label'] for r in support],dtype=torch.float64)
        ic=_rank_ic(score,label)
        if math.isfinite(ic):value=ic
        else:reasons=['CONSTANT_RANKS']
    return {'signal_session':session,'horizon':horizon,'valid_n':n,'stock_ids':[r['stock_id'] for r in support],
            'rank_ic':value,'reasons':reasons,'support_sha256':canonical_hash([
                {'stock_id':r['stock_id'],'label_id':r['label_id'],'label_metadata_sha256':r['label_metadata_sha256']} for r in support])}


def aggregate(daily,policy):
    valid=[d for d in daily if d['rank_ic'] is not None]
    weights=[d['valid_n'] if policy['aggregation_weighting']=='VALID_STOCK_COUNT' else 1 for d in valid]
    mean=sum(d['rank_ic']*w for d,w in zip(valid,weights))/sum(weights) if weights else None
    return {'mean_rank_ic':mean,'defined_session_horizons':len(valid),'requested_session_horizons':len(daily),
            'weighting':policy['aggregation_weighting'],'statistical_inference':'NOT_CONFIGURED_NOT_ADMITTED'}


def period_summaries(daily,policy):
    # Groups actual declared trading sessions, never calendar-day denominators.
    groups={}
    for d in daily:
        month=d['signal_session'][:7];quarter=month[:4]+'-Q'+str((int(month[5:])-1)//3+1)
        for kind,period in [('MONTH',month),('QUARTER',quarter)]:
            groups.setdefault((kind,period,d['horizon']),[]).append(d)
    return [{'kind':kind,'period':period,'horizon':h,'sessions':[d['signal_session'] for d in ds],**aggregate(ds,policy)}
            for (kind,period,h),ds in sorted(groups.items())]


def _label_issue(row,meta,header,fold,cal):
    """Validate metadata/role/maturity before any label payload access."""
    days=[s['session'] for s in cal['sessions']]
    try:
        if canonical_hash(meta)!=row['label_metadata_sha256']:return 'LABEL_METADATA_IDENTITY_MISMATCH'
        if any(meta[k]!=row[k] for k in ('stock_id','signal_session','horizon')):return 'LABEL_OWNER_OR_HORIZON_MISMATCH'
        if meta['label_contract_id']!=fold['label_contract_id']:return 'LABEL_CONTRACT_MISMATCH'
        if meta['snapshot_sha256']!=header['snapshot_sha256']:return 'LABEL_SNAPSHOT_MISMATCH'
        if not all(_sha(meta[k]) for k in ('source_sha256','payload_sha256','eligibility_ref')) or not _text(meta['vintage_id']):return 'LABEL_SOURCE_UNKNOWN'
        if meta['eligibility_status']!='ELIGIBLE_SYNTHETIC' or meta['quality_status']!='PASS':return 'LABEL_ELIGIBILITY_UNRESOLVED'
        i=days.index(row['signal_session']);h=row['horizon']
        if i+h>=len(days):return 'LABEL_CALENDAR_TAIL_UNAVAILABLE'
        if meta['entry_session']!=days[i+1] or meta['exit_session']!=days[i+h]:return 'LABEL_SESSION_MISMATCH'
        entry,exit,mature,available=(instant(meta[k]) for k in ('entry_at','exit_at','mature_at','available_at'))
        if entry!=instant(cal['sessions'][i+1]['open_at']) or exit!=instant(cal['sessions'][i+h]['close_at']):return 'LABEL_CALENDAR_CLOCK_MISMATCH'
        if not entry<=exit<=mature<=available:return 'INCONSISTENT_LABEL_CLOCKS'
        cutoff=instant(header['evaluation_at'])
        if available>cutoff:return 'LABEL_NOT_MATURE_OR_AVAILABLE'
        if header['prediction_role']=='SELECTION':
            b=days.index(fold['forward'][0])
            if exit>=instant(cal['sessions'][b]['open_at']):return 'LABEL_INTERVAL_OVERLAP'
            if b-(i+h)-1<fold['purge_sessions']+fold['embargo_sessions']:return 'REQUIRED_SESSION_GAP'
    except (KeyError,TypeError,ValueError):return 'LABEL_METADATA_UNKNOWN'
    return None


def evaluate(bundle,expected_input_sha256,fold,calendar,fold_pin,policy,expected_policy_sha256,label_metadata,label_store):
    """Metadata-first scoped evaluation. No label_store iteration or bulk reads.

    Rows are (signal, stock, horizon) keyed. Rejections retain all requested keys.
    Missing/nonfinite scores or labels are exclusions, never zero-valued metrics.
    Only synthetic calendars and named synthetic policies are supported in v1.
    """
    base={'schema':OUTPUT_SCHEMA,'research_admitted':False,'daily':[],'records':[],'rejections':[],
          'requested':[],'summary':{'by_horizon':[]},'periods':[]}
    def fail(reason):return {**base,'status':'REJECTED','reasons':[reason]}
    f=check_fold(fold,calendar,fold_pin)
    if f['status']!='ELIGIBLE':return fail('FOLD_INELIGIBLE:'+','.join(f['reasons']))
    if not check_policy(policy,expected_policy_sha256):return fail('POLICY_UNAPPROVED_OR_UNSUPPORTED')
    try:
        if prediction_hash(bundle)!=expected_input_sha256:return fail('INPUT_IDENTITY_MISMATCH')
        h=bundle['header'];rows=bundle['rows']
        if h['schema']!=INPUT_SCHEMA or calendar['qualification']!='SYNTHETIC' or h['evidence_class']!='SYNTHETIC':return fail('SYNTHETIC_SCOPE_REQUIRED')
        if any(h[k]!=fold[k] for k in ('study_id','fold_id','origin_at')) or h['fold_sha256']!=canonical_hash(fold):return fail('FOLD_IDENTITY_MISMATCH')
        if not all(_text(h[k]) for k in ('recipe_id','run_id','vintage_id')):return fail('RUN_IDENTITY_UNKNOWN')
        if not all(_sha(h[k]) for k in ('source_sha256','snapshot_sha256','model_sha256')):return fail('SOURCE_IDENTITY_UNKNOWN')
        if not _sha(h['raw_predictions_ref']['sha256']) or not _text(h['raw_predictions_ref']['path']):return fail('RAW_REFERENCE_UNKNOWN')
        role=h['prediction_role'];evaluation=h['evaluation_role']
        if (role,evaluation) not in {('SELECTION','SELECTION_VALIDATION'),('FORWARD','FORWARD_EVALUATION')}:return fail('ROLE_MISMATCH')
        evaluated_at=instant(h['evaluation_at'])
        if role=='SELECTION' and evaluated_at>instant(fold['selection_cutoff_at']):return fail('SELECTION_AFTER_CUTOFF')
        if role=='FORWARD' and evaluated_at<instant(fold['origin_at']):return fail('EVALUATION_BEFORE_ORIGIN')
        sessions=h['sessions'];horizons=h['horizons'];days=[s['session'] for s in calendar['sessions']]
        if not isinstance(sessions,list) or not sessions or sessions!=sorted(set(sessions)) or any(s not in days for s in sessions):return fail('INVALID_SESSION_SUPPORT')
        if not isinstance(horizons,list) or not horizons or any(type(x) is not int for x in horizons) or len(set(horizons))!=len(horizons) or not set(horizons)<=set(fold['required_horizons']):return fail('HORIZON_MISMATCH')
        seen=set()
        for row in rows:
            if set(row)!=ROW_FIELDS or not _text(row['stock_id']) or not _text(row['label_id']):return fail('INVALID_ROW_IDENTITY')
            if type(row['horizon']) is not int or row['horizon'] not in horizons or row['signal_session'] not in sessions:return fail('ROW_OUTSIDE_DECLARED_SUPPORT')
            if not _sha(row['label_metadata_sha256']) or not _sha(row['feature_eligibility_ref']):return fail('ROW_EVIDENCE_UNKNOWN')
            k=key(row)
            if k in seen:return fail('DUPLICATE_KEY')
            seen.add(k)
        # All structural validation precedes label access.
        accepted=[];rejected=[];requested=[]
        for row in sorted(rows,key=key):
            ref={k:row[k] for k in ('stock_id','signal_session','horizon','label_id','label_metadata_sha256')}
            requested.append(ref)
            def exclude(reason):rejected.append({**ref,'reasons':[reason]})
            phase='FORWARD_FEATURES' if role=='FORWARD' else 'SELECTION_VALIDATION'
            r=role_check({'signal_session':row['signal_session'],'entry_session':days[days.index(row['signal_session'])+1] if days.index(row['signal_session'])+1<len(days) else None},fold,calendar,phase)
            if r:exclude(r);continue
            if not _number(row['score']):exclude('INVALID_PREDICTION');continue
            try:
                predicted_at=instant(row['prediction_at']);entry=instant(calendar['sessions'][days.index(row['signal_session'])+1]['open_at'])
                signal_close=instant(calendar['sessions'][days.index(row['signal_session'])]['close_at'])
                cutoff=instant(fold['selection_cutoff_at']) if role=='SELECTION' else evaluated_at
                if not signal_close<=predicted_at<entry or predicted_at>cutoff or (role=='FORWARD' and predicted_at<instant(fold['origin_at'])):
                    exclude('PREDICTION_TIME_OUTSIDE_ROLE');continue
            except (KeyError,TypeError,ValueError,IndexError):exclude('PREDICTION_TIME_UNKNOWN');continue
            try:meta=label_metadata[row['label_id']]
            except KeyError:exclude('LABEL_METADATA_MISSING');continue
            issue=_label_issue(row,meta,h,fold,calendar)
            if issue:exclude(issue);continue
            try:
                payload=label_store[row['label_id']]
                if not isinstance(payload,Mapping):exclude('INVALID_LABEL_PAYLOAD_CONTAINER');continue
                if not _number(payload.get('value')):exclude('INVALID_LABEL_VALUE');continue
                if canonical_hash(payload)!=meta['payload_sha256']:exclude('LABEL_PAYLOAD_IDENTITY_MISMATCH');continue
                if any(payload.get(k)!=v for k,v in meta.items() if k not in {'horizon','snapshot_sha256','payload_sha256','eligibility_status','eligibility_ref'}):
                    exclude('LABEL_PAYLOAD_METADATA_MISMATCH');continue
            except (KeyError,TypeError,ValueError):exclude('LABEL_PAYLOAD_MISSING_OR_INVALID');continue
            accepted.append({**ref,'score':row['score'],'label':payload['value'],'label_metadata':deepcopy(meta)})
        daily=[daily_metric(accepted,s,horizon,policy) for s in sessions for horizon in horizons]
        for d in daily:
            d['requested_n']=sum(r['signal_session']==d['signal_session'] and r['horizon']==d['horizon'] for r in requested)
            d['excluded_n']=d['requested_n']-d['valid_n']
            d['evaluation_identity']=canonical_hash({'input':expected_input_sha256,'policy':expected_policy_sha256,'fold':fold_pin['fold_sha256'],'session':d['signal_session'],'horizon':d['horizon']})
        out={**base,'status':'EVALUATED' if any(d['rank_ic'] is not None for d in daily) else 'NO_ELIGIBLE_SESSIONS',
             'reasons':[],'header':deepcopy(h),'policy':deepcopy(policy),'input_sha256':expected_input_sha256,'policy_sha256':expected_policy_sha256,
             'fold_sha256':fold_pin['fold_sha256'],'calendar_sha256':fold_pin['calendar_sha256'],'label_contract_id':fold['label_contract_id'],
             'requested':requested,'records':accepted,'rejections':rejected,'daily':daily,'summary':{'by_horizon':[{'horizon':horizon,**aggregate([d for d in daily if d['horizon']==horizon],policy)} for horizon in horizons]},'periods':period_summaries(daily,policy)}
        out['evaluation_identity']=canonical_hash(out)
        return out
    except (KeyError,TypeError,ValueError,IndexError):return fail('MALFORMED_INPUT')


def exposure_reference(evaluation,expected_evaluation_sha256):
    """Bridge to W07 expose_forward_result; caller must persist its returned ledger.

    This does not authorize selection use. W07 checks later-fold/time/root identity.
    """
    if canonical_hash(evaluation)!=expected_evaluation_sha256 or evaluation.get('status')!='EVALUATED' or evaluation['header']['evaluation_role']!='FORWARD_EVALUATION':
        raise ValueError('eligible forward evaluation identity required')
    h=evaluation['header']
    return {'source_study_id':h['study_id'],'source_fold_id':h['fold_id'],'source_run_id':h['run_id'],
            'source_origin_at':h['origin_at'],'available_at':h['evaluation_at'],'source_sha256':expected_evaluation_sha256,'evidence_class':'SYNTHETIC'}
