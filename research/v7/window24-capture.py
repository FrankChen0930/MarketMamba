import sys,json
from pathlib import Path
sys.path.insert(0,'/home/frank/projects/MarketMamba/.worktrees/v7-window24-expanded-pilot')
from V6.experimental.v7_finmind_proof import request,put,restore
from V6.experimental.v7_corrected_e5_replication import canonical_hash,file_sha256
root=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot');cohort=json.load(open(root/'FROZEN_COHORT.json'));plan=cohort['new_price_queries'];ids=plan[int(sys.argv[1]):int(sys.argv[2])]
for sec in ids:
 params={'dataset':'TaiwanStockPrice','data_id':sec,'start_date':'2021-01-01','end_date':'2023-12-31'}
 r=request(root/'finmind',params,kind='price',scope='W24');print(sec,r['status'],r.get('http_status'),r.get('qc',{}).get('rows'),r.get('qc',{}).get('earliest'),r.get('qc',{}).get('latest'),flush=True)
 if r['status']!='CAPTURED':break
 q=restore(root/'finmind',r['record_path'],r['record_sha256'],Path('/home/frank/projects/MarketMamba/.artifacts/window24-price-restore/finmind'));put(root/'restore-receipts'/f"{r['sequence']:02d}.json",q)
 if q['status']!='RESTORED_BYTES_HASH_SCHEMA_RECORDS_VERIFIED':break
