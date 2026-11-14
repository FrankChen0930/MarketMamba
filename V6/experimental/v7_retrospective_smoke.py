"""Window23 bounded retrospective diagnostics, not strict PIT or a new trainer.

Only numerical Ridge/L0 core, W08 daily metric and W09 publisher are reused.
Public synthetic-only W07/W08/W10 admission paths remain unchanged.
"""
import math
from V6.experimental.v7_temporal_contract import normalize_calendar

def schedule_row_role(row):
    day,title,note=row
    current=f'{int(day[5:7])}月{int(day[8:])}日'
    if ('無交易' in title) or note.startswith(current+'市場無交易'):return 'OFFICIAL_CLOSURE'
    if '開始交易' in title or '最後交易' in title:return 'OFFICIAL_SESSION_VERIFIED'
    return 'OFFICIAL_CLOSURE'

def engineering_sample(*,signal,horizon,sessions,prices,market,calendar_ok,known_action_dates):
    days=normalize_calendar(sessions);reasons=[];entry=exit=None;values=None;label=None
    if horizon not in (5,10):raise ValueError('HORIZON_NOT_AUTHORIZED')
    if signal not in days:reasons.append('DATE_NOT_IN_SHARED_CALENDAR')
    else:
        i=days.index(signal)
        if i<1 or i+horizon>=len(days):reasons.append('LOOKBACK_OR_LABEL_TAIL')
        else:
            prev,entry,exit=days[i-1],days[i+1],days[i+horizon]
            involved=[prev,signal,entry,exit]
            if any(d not in prices for d in involved):reasons.append('MISSING_PRICE')
            else:
                for d in involved:
                    p=prices[d];v=[p.get(k) for k in ('open','max','min','close')]
                    if any(type(x) not in (float,int) or not math.isfinite(x) or x<=0 for x in v):reasons.append('ZERO_OR_INVALID_PRICE');break
                    if not p['min']<=min(p['open'],p['close'])<=max(p['open'],p['close'])<=p['max']:reasons.append('UNVERIFIED_MARKET_PRICE_SEMANTICS');break
                if not reasons:
                    p=prices[signal];values=[p['close']/prices[prev]['close']-1,(p['max']-p['min'])/p['close']]
                    label=prices[exit]['close']/prices[entry]['open']-1
            if any(prev<=d<=exit for d in known_action_dates):reasons.append('KNOWN_CORPORATE_ACTION')
    if not calendar_ok:reasons.append('CALENDAR_EVIDENCE_INSUFFICIENT')
    if market!='TWSE_LISTED':reasons.append('MARKET_MEMBERSHIP_NOT_ELIGIBLE_FOR_THIS_SMOKE')
    ok=not reasons
    return {'signal_session':signal,'horizon':horizon,'entry':entry,'exit':exit,'features':values if ok else None,'diagnostic_label':label if ok else None,'engineering_eligible':ok,'strict_eligible':False,'reasons':sorted(set(reasons)),'limitations':['RETROSPECTIVE_NOT_PIT','CORPORATE_ACTIONS_NOT_FULLY_VERIFIED','NO_EXECUTABLE_RETURN_CLAIM','NO_RESEARCH_ADMISSION']}
