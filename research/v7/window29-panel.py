"""Minimal as-of feature/retrospective label separation missing in W24 wrapper.

Two unchanged formula expressions only; no new feature framework or admission.
"""
import sys,math,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
S=runpy.run_path(str(Path(__file__).with_name('window27-safety.py')))
def split(day):return 'TRAIN' if day<'2022-01-01' else 'VALIDATION' if day<'2023-01-01' else 'EXPOSED_DIAGNOSTIC'
def valid(p):return S['valid_ohlc'](p)
def suspect_events(prices):
 last=None;events=[]
 for d,p in sorted(prices.items()):
  if valid(p):
   if last is not None and (p['close']/last>=2 or p['close']/last<=.5):events.append(d)
   last=p['close']
 return events

def feature_at(sec,day,sessions,prices,cohort,events):
 # No label/horizon/entry/exit argument; only prices <= signal are accessed.
 i=sessions.index(day);reasons=[];values=None;start=cohort.get('reported_twse_start');end=cohort.get('reported_end');prev=sessions[i-1] if i else None
 if not start:reasons.append('LISTING_START_UNKNOWN')
 elif prev is None or prev<start:reasons.append('FEATURE_PRE_LISTING_OR_LOOKBACK_TAIL')
 if end and day>=end:reasons.append('DELISTED_AT_DECISION')
 if prev is None or prev not in prices or day not in prices:reasons.append('FEATURE_MISSING')
 elif not valid(prices[prev]) or not valid(prices[day]):reasons.append('FEATURE_ZERO_OR_INVALID')
 if prev and any(prev<e<=day for e in events):reasons.append('FEATURE_OBSERVED_UNIT_DISCONTINUITY')
 if not reasons:
  p=prices[day];values=[p['close']/prices[prev]['close']-1,(p['max']-p['min'])/p['close']]
 return {'stock_id':sec,'signal_session':day,'split':split(day),'features':values,'known_membership_end':end,'eligible':not reasons,'reasons':reasons}

def label_at(feature,h,sessions,prices,events):
 day=feature['signal_session'];i=sessions.index(day);boundary={'TRAIN':'2022-01-01','VALIDATION':'2023-01-01','EXPOSED_DIAGNOSTIC':'2024-01-01'}[feature['split']]
 base={'key':[day,feature['stock_id'],h],'eligible':False,'entry':None,'exit':None,'label':None,'reasons':[],'risk':'UNKNOWN_ACTION_SUSPENSION_ASOF'}
 if h not in (5,10):raise ValueError('HORIZON_NOT_AUTHORIZED')
 if not feature['eligible']:return {**base,'reasons':['FEATURE_NOT_ELIGIBLE']}
 if i+h>=len(sessions):return {**base,'reasons':['LABEL_CALENDAR_TAIL']}
 entry,exit=sessions[i+1],sessions[i+h];base.update(entry=entry,exit=exit)
 if feature.get('known_membership_end') and exit>=feature['known_membership_end']:return {**base,'reasons':['DELISTING_BOUNDARY']}
 if exit>=boundary:return {**base,'reasons':['LABEL_NOT_MATURE_BEFORE_SPLIT']}
 row={'stock_id':feature['stock_id'],'signal_session':day,'horizon':h,'entry':entry,'exit':exit,'split':'FORWARD_DIAGNOSTIC' if feature['split']=='EXPOSED_DIAGNOSTIC' else feature['split']}
 q=S['classify'](row,sessions,prices,[]);reasons=list(q['reasons'])
 if any(sessions[i-1]<e<=day or entry<e<=exit for e in events):reasons.insert(0,'SUSPECTED_UNIT_DISCONTINUITY')
 # W29 cohort excludes2321; no other confirmed unit events are in existing evidence.
 reasons=list(dict.fromkeys(reasons));value=prices[exit]['close']/prices[entry]['open']-1 if not reasons else None
 return {**base,'eligible':not reasons,'label':value,'reasons':reasons,'primary':reasons[0] if reasons else None}

def coverage_gate(ns):return bool(ns) and all(type(n)is int and n>=40 for n in ns)
