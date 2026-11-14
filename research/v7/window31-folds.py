"""Role-neutral W27 numerical safety plus origin-specific maturity. No model/metric code."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_temporal_contract import normalize_calendar
from V6.experimental.v7_research_temporal import instant,check_fold,make_pin,SCHEMA
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H
P=runpy.run_path(str(Path(__file__).with_name('window29-panel.py')))
def numeric_label(f,h,sessions,prices,events):
 i=sessions.index(f['signal_session']);base={'key':[f['signal_session'],f['stock_id'],h],'eligible':False,'entry':None,'exit':None,'label':None,'reasons':[],'risk':'UNKNOWN_ACTION_SUSPENSION_ASOF'}
 if h not in (5,10):raise ValueError('HORIZON')
 if i<1 or i+h>=len(sessions):return {**base,'reasons':['LABEL_CALENDAR_TAIL']}
 entry,exit=sessions[i+1],sessions[i+h];base.update(entry=entry,exit=exit);reasons=[]
 end=f.get('known_membership_end')
 if end and exit>=end:reasons.append('DELISTING_BOUNDARY')
 if any(sessions[i-1]<e<=f['signal_session'] or entry<e<=exit for e in events):reasons.append('SUSPECTED_UNIT_DISCONTINUITY')
 for day,field in [(entry,'open'),(exit,'close')]:
  if day not in prices:reasons.append('ENTRY_EXIT_MISSING')
  elif not P['S']['finite_positive'](prices[day].get(field)):reasons.append('ENTRY_EXIT_INVALID_OR_ZERO')
 for day in sessions[i+1:i+h+1]:
  if day not in prices:reasons.append('HOLDING_PATH_MISSING')
  elif not P['valid'](prices[day]):reasons.append('HOLDING_PATH_INVALID_OR_ZERO')
 reasons=list(dict.fromkeys(reasons));value=prices[exit]['close']/prices[entry]['open']-1 if not reasons else None
 return {**base,'eligible':not reasons,'label':value,'reasons':reasons}
def mature(q,cutoff):return bool(q.get('eligible') and q.get('exit') and instant(q['exit']+'T13:30:00+08:00')<instant(cutoff+'T00:00:00+08:00'))
def view_ids(labels,h,start,end,cutoff):return [i for i,q in enumerate(labels) if q['key'][2]==h and start<=q['key'][0]<end and mature(q,cutoff)]
def resolve(contract,days,source_sha,feature_id,label_id):
 days=list(normalize_calendar(days));cal={'sessions':[{'session':d,'open_at':d+'T09:00:00+08:00','close_at':d+'T13:30:00+08:00'} for d in days],'source_sha256':source_sha,'vintage_id':'W29_CURRENT_RETROSPECTIVE_CALENDAR','qualification':'QUALIFIED_REFERENCE','qualification_ref':'W23/W29 pinned development session reference; not original publication/PIT admission'};origins=[]
 for k,(anchor,nextq) in enumerate(zip(contract['origin_anchors'],contract['next_quarter_anchors']),1):
  origin=next(d for d in days if d>=anchor);vanchor=str(int(anchor[:4])-1)+anchor[4:];vs=[d for d in days if vanchor<=d<anchor];train=[d for d in days if d<vs[0]];fw=[d for d in days if origin<=d<nextq]
  fold={'schema':SCHEMA,'study_id':contract['study_id'],'fold_id':'W31_O'+str(k),'feature_contract_id':feature_id,'label_contract_id':label_id,'calendar_sha256':H(cal),'unresolved_policies':[],'policy_status':'EXPLICIT_CANDIDATE_NOT_ADMITTED','purge_sessions':0,'embargo_sessions':0,'required_horizons':[5,10],'label_definition':'NEXT_SESSION_OPEN_HOLD_H_CLOSE','train':[train[0],train[-1]],'validation':[vs[0],vs[-1]],'refit':[days[0],vs[-1]],'forward':[fw[0],fw[-1]],'selection_cutoff_at':vs[0]+'T00:00:00+08:00','refit_cutoff_at':origin+'T00:00:00+08:00','model_ready_deadline_at':origin+'T09:00:00+08:00','origin_at':origin+'T09:00:00+08:00'}
  pin=make_pin(fold,cal);gate=check_fold(fold,cal,pin);assert gate['status']=='ELIGIBLE',gate
  origins.append({'origin_id':'O'+str(k),'origin':origin,'validation_start':vs[0],'next_quarter_anchor':nextq,'fold':fold,'fold_pin':pin,'geometry_gate':gate,'proxy_only':'maturity geometry, not historical publication/vintage correctness'})
 return cal,origins
