import json,csv,io,hashlib,sys
from pathlib import Path
sys.path.insert(0,'/home/frank/projects/MarketMamba/.worktrees/v7-window24-expanded-pilot')
from V6.experimental.v7_corrected_e5_replication import canonical_hash
W22=Path('/home/frank/projects/MarketMamba/.artifacts/window22-historical-panel');OUT=Path('/home/frank/projects/MarketMamba/.worktrees/v7-window24-expanded-pilot/research/v7');root=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot');root.mkdir(parents=True,exist_ok=True)
def official(id):
 r=json.load(open(W22/'official-resources'/f'{id}.record.json'));b=Path(r['raw']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig')))),r
rows,listed=official('18419');old,delisted=official('11543');by={r['公司代號']:r for r in rows}
valid=lambda r:len(r['公司代號'])==4 and r['公司代號'].isdigit() and not any(x in r['公司簡稱'] for x in ['KY','DR','-創','-新','特']) and r['上市日期'].isdigit() and len(r['上市日期'])==8
cohort=[]
def add(r,role):
 cohort.append({'security_id':r['公司代號'],'name':r['公司簡稱'],'industry':r['產業別'],'selection_role':role,'reported_twse_start':r['上市日期'][:4]+'-'+r['上市日期'][4:6]+'-'+r['上市日期'][6:],'reported_end':None,'current_paid_in_capital_proxy':int(r['實收資本額']),'size_evidence':'CURRENT_CAPITAL_PROXY_NOT_HISTORICAL_MARKET_CAP','source_dataset':'18419','source_raw_sha256':listed['sha256'],'security_type_evidence':'official issuer primary fourdigitcode/commonshares context; not full historical security master','daily_suspension_history':'UNKNOWN'})
existing=['2303','2317','2330','6515','9929']
for sec in existing:add(by[sec],'W23_CONTINUITY_FIXED_NOT_OUTCOME_SELECTED')
industries=sorted({r['產業別'] for r in rows if valid(r) and r['上市日期']<'20210101' and r['公司代號'] not in existing})
for i,industry in enumerate(industries[:25]):
 group=[r for r in rows if valid(r) and r['產業別']==industry and r['上市日期']<'20210101' and r['公司代號'] not in {x['security_id'] for x in cohort}]
 group.sort(key=lambda r:(int(r['實收資本額']),r['公司代號']));add(group[-1] if i%2==0 else group[0],'INDUSTRY_STRATIFIED_CURRENT_CAPITAL_'+('HIGH' if i%2==0 else 'LOW'))
new=sorted([r for r in rows if valid(r) and '20210101'<=r['上市日期']<='20231231' and r['公司代號'] not in {x['security_id'] for x in cohort}],key=lambda r:(r['上市日期'],r['公司代號']))
for r in new[:3]:add(r,'DURING_PERIOD_LISTING_OR_TRANSFER_BOUNDARY')
retired=sorted([r for r in old if len(r['上市編號'])==4 and r['上市編號'].isdigit() and not any(x in r['公司名稱'] for x in ['KY','DR']) and '1100101'<=r['終止上市日期']<='1121231'],key=lambda r:r['上市編號'])[:3]
for r in retired:
 d=r['終止上市日期'];cohort.append({'security_id':r['上市編號'],'name':r['公司名稱'],'industry':'UNKNOWN','selection_role':'DURING_PERIOD_DELISTED_DIAGNOSTIC','reported_twse_start':None,'reported_end':f'{int(d[:3])+1911:04d}-{d[3:5]}-{d[5:]}','current_paid_in_capital_proxy':None,'size_evidence':'UNKNOWN','source_dataset':'11543','source_raw_sha256':delisted['sha256'],'security_type_evidence':'official delisting primarycode; listingstart/historical commonstock master UNKNOWN','daily_suspension_history':'UNKNOWN'})
assert len(cohort)==36 and len({r['security_id'] for r in cohort})==36
obj={'schema':'w24-frozen-cohort-v1','status':'FROZEN_FOR_THIS_LIMITED_WINDOW_ONLY_NOT_FORMAL_DATA_ADMISSION','selection':'HINDSIGHT_SELECTED; deterministic industry/currentcapital strata plus known lifecycle cases; not historical universe or marketcaprepresentative','rule':'W23fivefixed +first25sortedindustrycodes one each alternatinghighest/lowestcurrentpaidcapital(pret2021listed,domestic4digit,noKYDRinnovation)+first3listingdates2021-23+lowest3domesticdelistcodes2021-23; no prices, returns or model results used','period':['2021-01-01','2023-12-31'],'source_commit':'87af76b038fcbbda9e1e4b3a4349e693a3b1ba48','source_authority':'W22immutableofficialCSV18419/11543','cohort':cohort,'existing_W22_reuse':existing,'new_price_queries':[r['security_id'] for r in cohort if r['security_id'] not in existing],'metadata_queries':[],'requests_limit':45,'price_limit':40,'planned_price_requests':31,'source_endpoint':'https://api.finmindtrade.com/api/v4/data','license':'W20/W21FinMindanonymousnoncommercialattribution/no rawredistribution; no paid use/token','no_replacement_after_coverage_or_performance':True}
for p in [OUT/'window24-cohort.json',root/'FROZEN_COHORT.json']:
 assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
(root/'COHORT_PIN.json').write_text(json.dumps({'semantic_sha256':canonical_hash(obj),'file_sha256':hashlib.sha256((root/'FROZEN_COHORT.json').read_bytes()).hexdigest()}))
print('frozen',canonical_hash(obj),'n',len(cohort),'new',len(obj['new_price_queries']));print([(r['security_id'],r['name'],r['selection_role']) for r in cohort])
