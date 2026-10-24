"""Read-only historical proof diagnostics; not a label formula or admission policy.

Session indices are diagnostic next-open / h-th-session-close geometry matching
W07. Unknown calendar, membership, actions or availability always block eligibility.
"""
import math
from V6.experimental.v7_temporal_contract import normalize_calendar
from V6.experimental.v7_research_temporal import instant

def label_diagnostic(*,signal,horizon,sessions,prices,asof,calendar_verified=False,
                     membership_verified=False,actions_verified=False,
                     feature_available_at=None,decision_at=None,delisted_at=None,label_available_at=None):
    days=normalize_calendar(sessions)
    if horizon not in (5,10):raise ValueError('UNAPPROVED_DIAGNOSTIC_HORIZON')
    reasons=[];entry=exit=None
    if signal not in days:reasons.append('SIGNAL_OUTSIDE_CALENDAR')
    else:
        i=days.index(signal)
        if i+1<len(days):entry=days[i+1]
        else:reasons.append('ENTRY_OUTSIDE_PANEL')
        if i+horizon<len(days):exit=days[i+horizon]
        else:reasons.append('EXIT_OUTSIDE_PANEL')
    for d,field,role in [(entry,'open','ENTRY'),(exit,'close','EXIT')]:
        if d:
            row=prices.get(d)
            if row is None:reasons.append('MISSING_'+role+'_PRICE')
            elif type(row.get(field)) not in (int,float) or not math.isfinite(row[field]) or row[field]<=0:reasons.append('INVALID_'+role+'_PRICE')
    if delisted_at and ((entry and entry>=delisted_at) or (exit and exit>=delisted_at)):reasons.append('DELISTING_BOUNDARY')
    # Exchange close is a geometry check, never proof of label publication/maturity.
    if exit and instant(asof)<instant(exit+'T13:30:00+08:00'):reasons.append('IMMATURE_LABEL')
    structural=not reasons
    if label_available_at is None:reasons.append("LABEL_MATURITY_EVIDENCE_UNKNOWN")
    elif exit:
        available=instant(label_available_at)
        if available<instant(exit+"T13:30:00+08:00"):reasons.append("LABEL_AVAILABILITY_BEFORE_EXIT")
        if available>instant(asof):reasons.append("IMMATURE_LABEL")
    for ok,reason in [(calendar_verified,'CALENDAR_UNKNOWN'),(membership_verified,'MEMBERSHIP_UNKNOWN'),(actions_verified,'CORPORATE_ACTIONS_UNKNOWN')]:
        if ok is not True:reasons.append(reason)
    if feature_available_at is None or decision_at is None:reasons.append('FEATURE_AVAILABILITY_UNKNOWN')
    elif instant(feature_available_at)>instant(decision_at):reasons.append('FUTURE_FEATURE')
    if decision_at and entry and instant(decision_at)>=instant(entry+'T09:00:00+08:00'):reasons.append('DECISION_NOT_BEFORE_ENTRY')
    return {'signal':signal,'horizon':horizon,'entry':entry,'exit':exit,
            'structural_ready':structural,'eligible':not reasons,'reasons':reasons,
            'geometry':'DIAGNOSTIC_W07_SIGNAL_PLUS_H_EXIT_NOT_APPROVED_LABEL_FORMULA',
            'label_maturity':'UNKNOWN' if label_available_at is None else 'EXPLICIT_AVAILABILITY_CHECK_ONLY',
            'research_admitted':False,'return_value':None}
