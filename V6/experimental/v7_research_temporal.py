"""Parameterized research guards; no study, data or trading admission.

Caller pins/qualification references establish a trust boundary, not authenticity.
Only explicit parameters are checked: no months/origins/purge/maturity defaults.
"""
from collections.abc import Mapping
from datetime import datetime,timezone
import math
import re
from zoneinfo import ZoneInfo
from V6.experimental.v7_temporal_contract import normalize_calendar,ContractError
from V6.experimental.v7_corrected_e5_replication import canonical_hash

SCHEMA='v7-research-fold-engineering-v1'
PHASES={'SELECTION_TRAIN':'train','SELECTION_VALIDATION':'validation','REFIT':'refit','FORWARD_FEATURES':'forward'}
CLOCKS=('observation_at','published_at','source_available_at','retrieved_at','quality_checked_at')


def instant(value):
    if not isinstance(value,str):raise ValueError('aware timestamp required')
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:raise ValueError('aware timestamp required')
    return t.astimezone(timezone.utc)


def _text(x):return isinstance(x,str) and bool(x.strip()) and x==x.strip()
def _sha(x):return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None
def _number(x):return type(x) in (int,float) and math.isfinite(x)


def verdict(status,reasons,fold,**identities):
    return {'status':status,'reasons':list(reasons),'fold_id':fold.get('fold_id') if isinstance(fold,Mapping) else None,
            'schema':SCHEMA,'research_admitted':False,'identities':identities}


def make_pin(fold,calendar,features=None,labels=None):
    """Build a synthetic/explicit manifest pin once; validation never refreshes it."""
    return {'fold_sha256':canonical_hash(fold),'calendar_sha256':canonical_hash(calendar),
            'feature_sha256':canonical_hash(features),'labels_sha256':canonical_hash(labels)}


def _calendar(cal):
    sessions=cal['sessions'];days=normalize_calendar([s['session'] for s in sessions])
    if not _sha(cal.get('source_sha256')) or not _text(cal.get('vintage_id')) or not _text(cal.get('qualification_ref')):
        return None,None,'CALENDAR_SOURCE_UNKNOWN'
    if cal.get('qualification') not in {'SYNTHETIC','QUALIFIED_REFERENCE'}:return None,None,'CALENDAR_QUALIFICATION_UNKNOWN'
    prior=None
    for s,d in zip(sessions,days):
        op,cl=instant(s['open_at']),instant(s['close_at'])
        if not op<cl or (prior is not None and op<=prior):raise ValueError('calendar clock order')
        if any(t.astimezone(ZoneInfo('Asia/Taipei')).date().isoformat()!=d for t in (op,cl)):raise ValueError('calendar session timezone')
        prior=cl
    return days,sessions,None


def check_fold(fold,calendar,pin):
    if not isinstance(fold,Mapping) or not isinstance(calendar,Mapping) or not isinstance(pin,Mapping):
        return verdict('REJECTED',['INVALID_CONTAINER'],fold)
    try:
        if canonical_hash(fold)!=pin.get('fold_sha256') or canonical_hash(calendar)!=pin.get('calendar_sha256') or fold.get('calendar_sha256')!=pin.get('calendar_sha256'):
            return verdict('REJECTED',['FOLD_OR_CALENDAR_IDENTITY_MISMATCH'],fold)
        if fold.get('schema')!=SCHEMA:return verdict('REJECTED',['UNSUPPORTED_SCHEMA'],fold)
        days,sessions,unknown=_calendar(calendar)
        if unknown:return verdict('BLOCKED',[unknown],fold)
        for k in ('study_id','fold_id','feature_contract_id','label_contract_id'):
            if not _text(fold.get(k)):return verdict('BLOCKED',['IDENTITY_UNRESOLVED'],fold)
        if fold.get('unresolved_policies')!=[] or fold.get('policy_status') not in {'SYNTHETIC_PARAMETERS_NOT_APPROVED','EXPLICIT_CANDIDATE_NOT_ADMITTED'}:
            return verdict('BLOCKED',['POLICY_UNRESOLVED'],fold)
        for k in ('purge_sessions','embargo_sessions'):
            if fold.get(k) is None:return verdict('BLOCKED',['GAP_POLICY_UNRESOLVED'],fold)
            if type(fold[k]) is not int or fold[k]<0:raise ValueError('gap parameter')
        hs=fold.get('required_horizons')
        if not isinstance(hs,list) or not hs:return verdict('BLOCKED',['HORIZON_POLICY_UNRESOLVED'],fold)
        if any(type(h) is not int or h<=0 for h in hs) or len(hs)!=len(set(hs)):raise ValueError('horizons')
        if fold.get('label_definition')!='NEXT_SESSION_OPEN_HOLD_H_CLOSE':return verdict('BLOCKED',['LABEL_DEFINITION_UNRESOLVED'],fold)
        ranges={}
        for k in ('train','validation','refit','forward'):
            r=fold[k]
            if not isinstance(r,list) or len(r)!=2:raise ValueError('range shape')
            a,b=map(days.index,r)
            if a>b:raise ValueError('range order')
            ranges[k]=(a,b)
        t,v,r,f=(ranges[k] for k in ('train','validation','refit','forward'))
        if not(t[1]<v[0] and v[1]<f[0] and r[0]==t[0] and r[1]>=v[1] and r[1]<f[0]):raise ValueError('range isolation')
        select,refit,deadline,origin=(instant(fold[k]) for k in ('selection_cutoff_at','refit_cutoff_at','model_ready_deadline_at','origin_at'))
        if not select<=refit<=deadline<=origin<=instant(sessions[f[0]]['close_at']):raise ValueError('fold clocks')
    except (KeyError,TypeError,ValueError,IndexError,ContractError):return verdict('REJECTED',['MALFORMED_FOLD_OR_CALENDAR'],fold)
    return verdict('ELIGIBLE',[],fold,fold_sha256=pin['fold_sha256'],calendar_sha256=pin['calendar_sha256'],calendar_evidence=calendar['qualification'])


def role_check(feature,fold,calendar,phase):
    if phase not in PHASES:return 'INVALID_PHASE'
    try:
        d=feature['signal_session'];a,b=fold[PHASES[phase]]
        if not a<=d<=b:return 'OUTSIDE_ROLE_RANGE'
        days=normalize_calendar([s['session'] for s in calendar['sessions']]);i=days.index(d)
        if i+1>=len(days):return 'CALENDAR_TAIL_UNAVAILABLE'
        if feature['entry_session']!=days[i+1]:return 'ENTRY_NOT_NEXT_SESSION'
    except (KeyError,ValueError,TypeError,ContractError):return 'INVALID_SAMPLE_SESSION'
    return None


def check_sample(features,labels,fold,calendar,pin,phase):
    valid=check_fold(fold,calendar,pin)
    if valid['status']!='ELIGIBLE':return valid
    if not isinstance(features,Mapping):return verdict('REJECTED',['INVALID_FEATURE_RECORD'],fold)
    role=role_check(features,fold,calendar,phase)
    if role:return verdict('REJECTED',[role],fold,sample_id=features.get('sample_id'))
    ids={**valid['identities'],'sample_id':features.get('sample_id'),'stock_id':features.get('stock_id'),'signal_session':features.get('signal_session'),'phase':phase,'feature_sha256':pin.get('feature_sha256'),'labels_sha256':pin.get('labels_sha256') if phase!='FORWARD_FEATURES' else None}
    def out(s,r):return verdict(s,[r],fold,**ids)
    try:
        if canonical_hash(features)!=pin.get('feature_sha256'):return out('REJECTED','FEATURE_IDENTITY_MISMATCH')
        if features.get('feature_contract_id')!=fold['feature_contract_id']:return out('REJECTED','FEATURE_CONTRACT_MISMATCH')
        if not all(_text(features.get(k)) for k in ('sample_id','stock_id','vintage_id')) or not _sha(features.get('source_sha256')):
            return out('BLOCKED','FEATURE_SOURCE_UNKNOWN')
        # A forward feature record cannot smuggle labels to a model through extra fields.
        allowed={'sample_id','stock_id','signal_session','entry_session','feature_contract_id','source_sha256','vintage_id',*CLOCKS,'quality_status','freeze_at','decision_at','values'}
        if set(features)!=allowed:return out('REJECTED','UNEXPECTED_FEATURE_FIELDS')
        if features.get('quality_status') is None:return out('BLOCKED','QUALITY_UNKNOWN')
        if features['quality_status']!='PASS':return out('REJECTED','QUALITY_FAILED')
        if not isinstance(features['values'],dict) or not features['values']:return out('BLOCKED','FEATURE_VALUES_MISSING')
        if any(v is None for v in features['values'].values()):return out('BLOCKED','FEATURE_VALUES_MISSING')
        if not all(_number(v) for v in features['values'].values()):return out('REJECTED','INVALID_FEATURE_VALUE')
        times=[instant(features[k]) for k in CLOCKS]
        freeze,decision=instant(features['freeze_at']),instant(features['decision_at'])
    except (KeyError,TypeError,ValueError):return out('BLOCKED','FEATURE_CLOCK_OR_SERIALIZATION_UNKNOWN')
    if times!=sorted(times):return out('REJECTED','INCONSISTENT_FEATURE_CLOCKS')
    days=[s['session'] for s in calendar['sessions']];i=days.index(features['signal_session'])
    if not times[-1]<=freeze<=decision<instant(calendar['sessions'][i+1]['open_at']):return out('REJECTED','FUTURE_FEATURE_OR_LATE_DECISION')
    cutoff=(decision if phase=='FORWARD_FEATURES' else instant(fold['refit_cutoff_at'] if phase=='REFIT' else fold['selection_cutoff_at']))
    if decision>cutoff:return out('REJECTED','FEATURE_AFTER_STAGE_CUTOFF')
    if phase=='FORWARD_FEATURES':
        if freeze<instant(fold['origin_at']):return out('REJECTED','FEATURE_FREEZE_BEFORE_ORIGIN')
        return verdict('ELIGIBLE',[],fold,**ids)
    try:
        if not isinstance(labels,Mapping):return out('BLOCKED','LABELS_MISSING')
        if set(labels)!={str(h) for h in fold['required_horizons']}:return out('BLOCKED','REQUIRED_HORIZONS_MISSING')
        # Finite serialization required, pin never silently refreshed on revised data.
        if canonical_hash(labels)!=pin.get('labels_sha256'):return out('REJECTED','LABEL_IDENTITY_MISMATCH')
        boundary=fold['validation'][0] if phase=='SELECTION_TRAIN' else fold['forward'][0]
        b=days.index(boundary);start=instant(calendar['sessions'][b]['open_at'])
        for h in fold['required_horizons']:
            lab=labels[str(h)]
            if not isinstance(lab,Mapping):return out('REJECTED','INVALID_LABEL_RECORD')
            for key in ('sample_id','stock_id','signal_session'):
                if not _text(lab.get(key)):return out('BLOCKED','LABEL_SAMPLE_IDENTITY_UNKNOWN')
                if lab[key]!=features[key]:return out('REJECTED','LABEL_SAMPLE_IDENTITY_MISMATCH')
            if i+h>=len(days):return out('BLOCKED','LABEL_CALENDAR_TAIL_UNAVAILABLE')
            exit_session=days[i+h]
            if lab.get('entry_session')!=features['entry_session'] or lab.get('exit_session')!=exit_session:return out('REJECTED','LABEL_SESSION_MISMATCH')
            if lab.get('label_contract_id')!=fold['label_contract_id']:return out('REJECTED','LABEL_CONTRACT_MISMATCH')
            if lab.get('value') is None:return out('BLOCKED','LABEL_MISSING')
            if not _number(lab['value']):return out('REJECTED','INVALID_LABEL_VALUE')
            if not _sha(lab.get('source_sha256')) or not _text(lab.get('vintage_id')):return out('BLOCKED','LABEL_SOURCE_UNKNOWN')
            if lab.get('quality_status') is None:return out('BLOCKED','LABEL_QUALITY_UNKNOWN')
            if lab['quality_status']!='PASS':return out('REJECTED','LABEL_QUALITY_FAILED')
            entry,exit,mature,available=(instant(lab[k]) for k in ('entry_at','exit_at','mature_at','available_at'))
            if entry!=instant(calendar['sessions'][i+1]['open_at']) or exit!=instant(calendar['sessions'][i+h]['close_at']):return out('REJECTED','LABEL_CALENDAR_CLOCK_MISMATCH')
            if not entry<=exit<=mature<=available:return out('REJECTED','INCONSISTENT_LABEL_CLOCKS')
            if available>cutoff:return out('REJECTED','LABEL_NOT_AVAILABLE_AT_CUTOFF')
            if exit>=start:return out('REJECTED','LABEL_INTERVAL_OVERLAP')
            # Explicit additive gap beyond the supervised label interval, in sessions.
            if b-(i+h)-1<fold['purge_sessions']+fold['embargo_sessions']:return out('REJECTED','REQUIRED_SESSION_GAP')
    except (KeyError,TypeError,ValueError):return out('BLOCKED','LABEL_CLOCK_OR_SERIALIZATION_UNKNOWN')
    return verdict('ELIGIBLE',[],fold,**ids)
