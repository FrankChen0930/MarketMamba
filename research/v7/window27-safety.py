"""W27 development-only mask. No model scores, model fit, network or ranking imports."""
import json,math,sys,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'research/v7'
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window27-label-safety')
W24=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot')
PRIORITY=['CONFIRMED_UNIT_EVENT','ENTRY_EXIT_MISSING','ENTRY_EXIT_INVALID_OR_ZERO','FEATURE_MISSING','FEATURE_INVALID_OR_ZERO','HOLDING_PATH_MISSING','HOLDING_PATH_INVALID_OR_ZERO']
def immutable(path,value):
 data=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('IMMUTABLE_OUTPUT_CONFLICT:'+str(path))
 else:
  with path.open('xb') as f:f.write(data);f.flush()
 return {'path':str(path),'sha256':file_sha256(path),'bytes':len(data)}
def checked(ref):
 p=Path(ref['path'])
 if not p.is_file() or file_sha256(p)!=ref['sha256']:raise ValueError('SOURCE_BYTES_MISMATCH')
 return json.loads(p.read_text())
def finite_positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0

def valid_ohlc(p):
 return all(finite_positive(p.get(k)) for k in ('open','close','max','min')) and p['min']<=min(p['open'],p['close'])<=max(p['open'],p['close'])<=p['max']

def classify(row,sessions,prices,event_dates):
 signal=row['signal_session'];h=row['horizon'];i=sessions.index(signal)
 boundary={'TRAIN':'2022-01-01','VALIDATION':'2023-01-01','FORWARD_DIAGNOSTIC':'2024-01-01'}[row['split']]
 if h not in (5,10) or i<1 or i+h>=len(sessions) or row['entry']!=sessions[i+1] or row['exit']!=sessions[i+h] or row['exit']>=boundary:raise ValueError('TEMPORAL_OR_MATURITY_GEOMETRY_CHANGED')
 prev=sessions[i-1];entry=row['entry'];exit=row['exit'];flags=set()
 if any(prev<d<=signal or entry<d<=exit for d in event_dates):flags.add('CONFIRMED_UNIT_EVENT')
 for day,field in ((entry,'open'),(exit,'close')):
  if day not in prices:flags.add('ENTRY_EXIT_MISSING')
  elif not finite_positive(prices[day].get(field)):flags.add('ENTRY_EXIT_INVALID_OR_ZERO')
 for day in (prev,signal):
  if day not in prices:flags.add('FEATURE_MISSING')
  elif not valid_ohlc(prices[day]):flags.add('FEATURE_INVALID_OR_ZERO')
 for day in sessions[i+1:i+h+1]:
  if day not in prices:flags.add('HOLDING_PATH_MISSING')
  elif not valid_ohlc(prices[day]):flags.add('HOLDING_PATH_INVALID_OR_ZERO')
 reasons=[r for r in PRIORITY if r in flags]
 return {'eligible':not reasons,'primary':reasons[0] if reasons else None,'reasons':reasons,'risk_status':'UNKNOWN_EVENT_SUSPENSION_ASOF'}

def input_pins():
 counts={}
 for name in ('window24-local-artifacts.json','window25-local-artifacts.json'):
  inv=json.loads((OUT/name).read_text())['files']
  for r in inv:
   p=Path(r['path']);assert p.stat().st_size==r['bytes'] and file_sha256(p)==r['sha256']
  counts[name]=len(inv)
 for r in json.loads((OUT/'window26-independent-restore.json').read_text())['files']:
  p=Path(r['path']);assert p.stat().st_size==r['bytes'] and file_sha256(p)==r['sha256']
 counts['window26-private']=15
 return counts

def source_views():
 run=json.loads((W24/'PILOT_CLOSED.json').read_text());bundle=Path(run['bundle_path'])
 coverage=json.loads((bundle/'data_split_eligibility.json').read_text())['value']
 refs=coverage['references'];data=checked(refs['dataset']);calendar=checked(refs['calendar']);sources=checked(refs['sources'])
 days=[r['date'] for r in calendar if r['engineering_calendar_ok'] and r['price_observed']]
 assert len(days)==len(set(days)) and days==sorted(days)
 raw={}
 for r in sources:
  p=Path(r['raw_path']);assert file_sha256(p)==r['raw_sha256'];rows=json.loads(p.read_text())['data'];prices={r['date']:r for r in rows};assert len(prices)==len(rows);raw[r['security']]=prices
 return run,bundle,data,days,raw

def build_mask():
 prior=input_pins();policy=json.loads((OUT/'window27-policy.json').read_text());assert policy['primary_exclusion_priority']==PRIORITY
 old_policy=json.loads((OUT/'window24-label-recipe.json').read_text())['evaluation']
 assert all(old_policy[k]==v for k,v in policy['evaluation'].items())
 event=json.loads((OUT/'window26-event-matrix.json').read_text());effective=next(e['effective_reference_date'] for e in event['events'] if e['id']=='2321-market-reference-change')
 assert effective=='2023-02-17'
 run,bundle,data,days,raw=source_views();masks=[];groups={}
 for r in data:
  i=days.index(r['signal_session']);p=raw[r['stock_id']]
  # Source dependencies and arithmetic are byte-preserved; no new Label definition.
  expected=[p[r['signal_session']]['close']/p[days[i-1]]['close']-1,(p[r['signal_session']]['max']-p[r['signal_session']]['min'])/p[r['signal_session']]['close']]
  assert r['features']==expected and r['diagnostic_label']==p[r['exit']]['close']/p[r['entry']]['open']-1
  q=classify(r,days,p,[effective] if r['stock_id']=='2321' else [])
  item={'key':[r['signal_session'],r['stock_id'],r['horizon']],'split':r['split'],'source_sample_sha256':canonical_hash(r),**q};masks.append(item)
  g=groups.setdefault(r['split']+'_'+str(r['horizon']),{'base':0,'eligible':0,'excluded':0,'primary':Counter(),'all_reason_flags':Counter(),'unknown_event_risk':0})
  g['base']+=1;g['eligible']+=q['eligible'];g['excluded']+=not q['eligible'];g['unknown_event_risk']+=1;g['all_reason_flags'].update(q['reasons'])
  if q['primary']:g['primary'].update([q['primary']])
 assert len({tuple(r['key']) for r in masks})==len(data)
 for g in groups.values():assert g['base']==g['eligible']+g['excluded'] and sum(g['primary'].values())==g['excluded']
 ref=immutable(ROOT/'mask.json',masks)
 frozen={'schema':'w27-mask-manifest-v1','status':'FROZEN_BEFORE_METRICS','policy_sha256':canonical_hash(policy),'policy_file_sha256':file_sha256(OUT/'window27-policy.json'),'predicate_sha256':file_sha256(Path(__file__)),'mask':ref,'groups':groups,'expected_support':'new group keys exactly original W24 frozen prediction keys intersect eligible mask; no additions; identical labels/models; all original dates retained even empty','W24_bundle':{'path':str(bundle),'key':run['bundle_key'],'manifest_sha256':run['manifest_sha256']},'source_refs':{str(p.relative_to(REPO)):file_sha256(p) for p in [OUT/'window24-label-recipe.json',OUT/'window26-event-matrix.json',OUT/'window26-exposure-ledger.json']},'prior_bytes_verified':prior,'no_prediction_or_metric_values_read_for_mask':True,'historical_unknown_is_not_clean':True,'training_influence_not_removed_by_evaluation_filter':True,'original_masks_unchanged':True}
 immutable(ROOT/'FROZEN_MASK.json',frozen);immutable(OUT/'window27-mask-manifest.json',frozen)
 print(json.dumps({'groups':groups,'mask_sha256':ref['sha256'],'status':frozen['status']}))
if __name__=='__main__':build_mask()
