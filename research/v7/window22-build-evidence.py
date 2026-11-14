"""Offline only: rebuild private panel/diagnostic mask from pinned W22 records.
No model fit, financial return, RankIC or external request is implemented here.
"""
import csv,io,json,hashlib,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_finmind_proof import verified_record,put
from V6.experimental.v7_corrected_e5_replication import file_sha256,canonical_hash
from V6.experimental.v7_historical_panel import label_diagnostic
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window22-historical-panel')
OUT=Path('research/v7');F=ROOT/'finmind'
records=[];panel=[];summaries=[];lifecycle=[];labels=[]
listed={};delisted={}
for ident in ['18419','25036','11543']:
 r=json.loads((ROOT/'official-resources'/f'{ident}.record.json').read_text());p=Path(r['raw']);assert file_sha256(p)==r['sha256']
 rows=list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8-sig'))))
 for x in rows:
  if ident=='11543':
   v=x['終止上市日期'];delisted[x['上市編號']]=f'{int(v[:3])+1911:04d}-{v[3:5]}-{v[5:7]}'
  else:
   d=x['上市日期' if ident=='18419' else '上櫃日期'];listed[x['公司代號']]={'market':'TWSE' if ident=='18419' else 'TPEx','start':d[:4]+'-'+d[4:6]+'-'+d[6:8],'resource':ident,'source_sha256':r['sha256']}
for p in sorted((F/'records').glob('*.json')):
 r=verified_record(F,str(p.relative_to(F)),file_sha256(p));records.append(r)
 rows=json.loads((F/r['raw_path']).read_bytes())['data']
 for row in rows:panel.append({**row,'snapshot_sha256':r['raw_sha256'],'retrieved_at':r['retrieved_at'],'source':'FinMindTaiwanStockPrice','version':'CURRENT_RETROSPECTIVE_NOT_ASOF'})
base=next(r['qc']['sessions'] for r in records if r['params']['data_id']=='2330');assert base==next(r['qc']['sessions'] for r in records if r['params']['data_id']=='8069')
for r in records:
 sec=r['params']['data_id'];rows=[x for x in panel if x['stock_id']==sec];byday={x['date']:x for x in rows};days=sorted(byday);member=listed.get(sec);end=delisted.get(sec)
 diagnostic=[]
 for d in days:
  for h in [5,10]:
   q=label_diagnostic(signal=d,horizon=h,sessions=base,prices=byday,asof=r['retrieved_at'],delisted_at=end)
   diagnostic.append(q);labels.append({'stock_id':sec,**q})
 prelisting=[d for d in days if member and d<member['start']]
 expected=[d for d in base if (not member or d>=member['start']) and (not end or d<end)]
 zero=[x['date'] for x in rows if x['open']==x['max']==x['min']==x['close']==0]
 exceptions=[x['date'] for x in rows if x['open']==0 and x['max']>0 or x['open']>x['max'] or 0<x['open']<x['min']]
 lifecycle.append({'security':sec,'reported_current_market_start':member,'reported_delisting':end,'pre_current_market_price_rows':len(prelisting),'pre_current_market_role':'UNKNOWN_EXCEPT_W21_EXACT_EMERGING_EXAMPLES' if prelisting else 'N/A','suspension_history':'UNKNOWN','effective_daily_tradability':'UNKNOWN','membership_not_inferred_from_price':True,'evidence':'official-resources18419/25036/11543 current retrospective CSV; W21 vendor metadata is secondary'})
 summaries.append({'security':sec,'rows':len(rows),'earliest':days[0] if days else None,'latest':days[-1] if days else None,'year_counts':dict(Counter(x[:4] for x in days)),'qc':r['qc']['counts'],'reference_missing_dates':sorted(set(base)-set(days)),'reference_extra_dates':sorted(set(days)-set(base)),'expected_in_reported_market_interval':len(expected),'missing_within_reported_interval':sorted(set(expected)-set(days)),'unknown_membership_reference':member is None,'pre_current_market_rows':len(prelisting),'zero_price_dates':zero,'open_semantic_exception_dates':exceptions,'label_geometry':{str(h):{'structural_ready':sum(x['structural_ready'] for x in diagnostic if x['horizon']==h),'eligible':sum(x['eligible'] for x in diagnostic if x['horizon']==h),'reasons':dict(Counter(y for x in diagnostic if x['horizon']==h for y in x['reasons']))} for h in [5,10]},'raw_bytes':r['raw_bytes'],'raw_sha256':r['raw_sha256'],'record':r['record_path'] if 'record_path' in r else 'records/'+str(r['sequence']).zfill(2)+'.json','query_id':r['query_id']})
# These private generated outputs contain licensed vendor data: never add to Git.
def immutable_csv(path,rows):
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=sorted(set(k for x in rows for k in x)));w.writeheader();w.writerows(rows);body=buf.getvalue().encode()
 if path.exists():assert path.read_bytes()==body,'GENERATED_OUTPUT_CHANGED_NO_OVERWRITE'
 else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
 return {'path':str(path),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'rows':len(rows)}
panel_id=immutable_csv(ROOT/'derived'/'continuous-panel.csv',panel)
mask_id=immutable_csv(ROOT/'derived'/'label-geometry.csv',[{**x,'reasons':'|'.join(x['reasons'])} for x in labels])
summary={'schema':'w22-continuous-proof-v1','sample':'HINDSIGHT_SELECTED_SAMPLE_NOT_HISTORICAL_UNIVERSE','period':['2019-01-01','2023-12-31'],'finmind_requests':len(records),'price_requests':len(records),'metadata_requests':0,'diagnostic_requests':0,'rows':len(panel),'reference_sessions':len(base),'reference_year_counts':dict(Counter(x[:4] for x in base)),'reference':'2330/8069 actual vendor date equality; annual-count official corroboration does not verify each date or TPExcalendar','calendar_qualification':'UNKNOWN_FULL_DAILY_OFFICIAL_HISTORY','panel':panel_id,'mask':mask_id,'securities':summaries,'dataset_class':'RETROSPECTIVE_ONLY','status':['TECHNICALLY_USABLE_FOR_DIAGNOSTICS','EXPLORATORY_POLICY_REQUIRED','PIT_NOT_VERIFIED'],'raw_price_adjustment':'TaiwanStockPrice RAW; no mixing Adj, corporate actions unverified','formal_dataset_admission':False,'research_admitted':False,'backup_status':'BACKUP_PENDING','raw_bytes_total':sum(r['raw_bytes'] for r in records)}
put(OUT/'window22-panel-summary.json',summary);put(OUT/'window22-lifecycle.json',{'schema':'w22-lifecycle-evidence-v1','rows':lifecycle,'not_complete_historical_universe':True})
print(json.dumps({'requests':len(records),'rows':len(panel),'bytes':summary['raw_bytes_total'],'sessions':len(base),'qc':dict(sum((Counter(r['qc']['counts']) for r in records),Counter())),'panel_sha256':panel_id['sha256'],'eligible_labels':sum(x['eligible'] for x in labels)}))
