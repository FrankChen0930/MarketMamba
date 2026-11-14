"""W24 bounded raw-price proxy masks; no strict gate or frozen contract changes."""
from V6.experimental.v7_retrospective_smoke import engineering_sample

def exploratory_mask(*,signal,horizon,sessions,prices,cohort,next_split,known_action_dates):
    # Numeric diagnostics remain separate from research-role permission.
    q=engineering_sample(signal=signal,horizon=horizon,sessions=sessions,prices=prices,
                         market='TWSE_LISTED',calendar_ok=True,known_action_dates=known_action_dates)
    reasons=list(q['reasons']);start=cohort.get('reported_twse_start');end=cohort.get('reported_end')
    if signal in sessions:
        i=sessions.index(signal);prev=sessions[i-1] if i else None
    else:prev=None
    if not start:reasons.append('HISTORICAL_LISTING_START_UNKNOWN')
    elif prev is None or prev<start:reasons.append('FEATURE_LOOKBACK_PRE_LISTING')
    if end and (signal>=end or q['entry'] and q['entry']>=end or q['exit'] and q['exit']>=end):reasons.append('DELISTING_BOUNDARY')
    if q['exit'] and q['exit']>=next_split:reasons.append('LABEL_NOT_MATURE_BEFORE_NEXT_SPLIT')
    return {**q,'exploratory_eligible':not reasons,'exploratory_reasons':sorted(set(reasons)),
            'strict_eligible':False,'formal_research_admitted':False,
            'label_class':'RAW_PRICE_DIAGNOSTIC_PROXY_CORPORATE_ACTIONS_UNVERIFIED',
            'strict_reasons':['HISTORICAL_AVAILABILITY_UNKNOWN','COMPLETE_ACTION_HISTORY_UNKNOWN','DAILY_EXECUTABLE_UNIVERSE_UNKNOWN']}

def coverage_gate(counts,minimum):
    if minimum!=20:raise ValueError('W24_FIXED_MINIMUM_20_ONLY')
    return bool(counts) and all(type(n)is int and n>=minimum for n in counts.values())
