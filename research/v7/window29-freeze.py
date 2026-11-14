"""Pre-data deterministic development cohort over existing official source bytes."""
import csv,io,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
OUT=Path(__file__).parent;ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window29-expanded-development');W22=ROOT.parent/'window22-historical-panel'
def save(p,x):
 b=(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n').encode();p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b)
def official(dataset):
 record=json.loads((W22/'official-resources'/f'{dataset}.record.json').read_text());p=Path(record['raw']);assert file_sha256(p)==record['sha256'];return list(csv.DictReader(io.StringIO(p.read_bytes().decode('utf-8-sig')))),record
rows,listed=official('18419');retired,delisted=official('11543');old={r['security_id'] for r in json.loads((OUT/'window24-cohort.json').read_text())['cohort']}
def order(sec):return hashlib.sha256(('W29-development-v1:'+sec).encode()).hexdigest()
def valid(r):
 sec=r['公司代號'];return len(sec)==4 and sec.isdigit() and not sec.startswith('91') and not any(x in r['公司簡稱'] for x in ('KY','DR','-創','-新','特')) and len(r['上市日期'])==8 and r['上市日期'].isdigit() and r['公司代號'] not in old and bool(r['普通股每股面額'])
byindustry=defaultdict(list)
for r in rows:
 if valid(r) and r['上市日期']<'20210101':byindustry[r['產業別']].append(r)
for group in byindustry.values():group.sort(key=lambda r:order(r['公司代號']))
selected=[]
while len(selected)<60:
 for industry in sorted(byindustry):
  if byindustry[industry] and len(selected)<60:selected.append((byindustry[industry].pop(0),'INDUSTRY_ROUND_ROBIN_PRE2021_CURRENT_SURVIVOR'))
new=sorted([r for r in rows if valid(r) and '20210101'<=r['上市日期']<='20231231'],key=lambda r:order(r['公司代號']))[:4]
cohort=[]
for r,role in selected+[(r,'DURING_PERIOD_LISTING_OR_TRANSFER_UNRESOLVED') for r in new]:
 d=r['上市日期'];cohort.append({'security_id':r['公司代號'],'name':r['公司簡稱'],'industry':r['產業別'],'reported_twse_start':d[:4]+'-'+d[4:6]+'-'+d[6:],'reported_end':None,'role':role,'ordinary_identity':'CONDITIONAL_OFFICIAL_PRIMARY_4DIGIT_ORDINARY_SHARE_FACE_FIELD_NOT_FULL_HISTORICAL_SECURITY_MASTER','historical_market':'TWSE_REPORTED_START_INTERVAL_CONDITIONAL; prior_market/temporary_suspension UNKNOWN','source_dataset':'18419','source_sha256':listed['sha256']})
retired=[r for r in retired if len(r['上市編號'])==4 and r['上市編號'].isdigit() and r['上市編號'] not in old and not any(x in r['公司名稱'] for x in ('KY','DR')) and '1100101'<=r['終止上市日期']<='1121231']
for r in sorted(retired,key=lambda r:order(r['上市編號']))[:4]:
 d=r['終止上市日期'];cohort.append({'security_id':r['上市編號'],'name':r['公司名稱'],'industry':'UNKNOWN','reported_twse_start':None,'reported_end':f'{int(d[:3])+1911:04d}-{d[3:5]}-{d[5:]}','role':'DELISTED_DIAGNOSTIC_MEMBERSHIP_START_UNKNOWN_NO_PREDICTION_ADMISSION','ordinary_identity':'UNKNOWN_HISTORICAL_COMMON_SHARE_TYPE','historical_market':'TWSE_DELISTED_END_CONFIRMED; start/previous_market UNKNOWN','source_dataset':'11543','source_sha256':delisted['sha256']})
assert len(cohort)==68 and len({r['security_id'] for r in cohort})==68 and not old&{r['security_id'] for r in cohort}
# Cache selection uses request metadata only, never prices/returns/label completeness.
caches={}
for name in ['window22-historical-panel','window24-expanded-pilot','window21-finmind-proof']:
 base=ROOT.parent/name;folder=base/'finmind' if (base/'finmind').is_dir() else base
 for p in sorted((folder/'records').glob('*.json')):
  r=json.loads(p.read_text());q=r.get('params',{});sec=q.get('data_id')
  if sec in {c['security_id'] for c in cohort} and r.get('status')=='CAPTURED' and q.get('dataset')=='TaiwanStockPrice' and q.get('start_date','9999')<='2021-01-01' and q.get('end_date','')>='2023-12-31':caches.setdefault(sec,{'record':str(p),'record_sha256':file_sha256(p),'raw':str(folder/r['raw_path']),'raw_sha256':r['raw_sha256'],'bounds':'EXISTING_SUPERSET_REQUEST_METADATA_NOT_COVERAGE_PROOF'})
x={'schema':'w29-frozen-development-cohort-v1','status':'HINDSIGHT_SELECTED_DEVELOPMENT_NOT_HISTORICAL_UNIVERSE_VERIFIED','source_commit':'1d9a2a7a1ee57c455ba58250f63681efbbdd9050','rule':'Exclude all W24 36;60 pre2021 current issuers by industry sorted roundrobin/hash(W29-development-v1:security),4 periodlisting hashorder,4 perioddelisted hashorder. No marketcap/returns/RankIC/label-completeness selection. No replacement after download.','cohort':cohort,'bias':'Current issuer survivors and industry labels are hindsight; 4 delisted start/type UNKNOWN, not prediction-qualified; not market-representative','period':['2021-01-01','2023-12-31'],'cache':caches,'price_queries':[r['security_id'] for r in cohort if r['security_id'] not in caches],'metadata_queries':[],'sources':{'18419':listed,'11543':delisted},'FinMind_license':'Existing W20/W21 official Free/noncommercial/attribution/no raw redistribution evidence; only TaiwanStockPrice exact endpoint, no token','no_formal_admission':True}
save(OUT/'window29-cohort.json',x);save(ROOT/'FROZEN_COHORT.json',x);save(ROOT/'COHORT_PIN.json',{'semantic_sha256':canonical_hash(x),'file_sha256':file_sha256(OUT/'window29-cohort.json')})
print(json.dumps({'cohort_n':68,'current_pre2021':60,'duringperiod':4,'delisted_start_unknown':4,'overlap_W24':0,'cache':len(caches),'new_price_queries':len(x['price_queries']),'sha256':canonical_hash(x)}))
