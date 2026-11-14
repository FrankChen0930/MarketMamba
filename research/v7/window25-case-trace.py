"""Post-audit trace of existing labels across unobserved/zero interior sessions.
No threshold, sample exclusion, metric recomputation, network or training.
"""
import json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path('/home/frank/projects/MarketMamba/.artifacts/window25-robustness-audit');W24=Path('/home/frank/projects/MarketMamba/.artifacts/window24-expanded-pilot')
run=json.load(open(W24/'PILOT_CLOSED.json'));bundle=Path(run['bundle_path']);data=json.load(open(bundle/'dataset_snapshot.json'))['value'];cov=json.load(open(bundle/'data_split_eligibility.json'))['value'];refs=json.load(open(cov['references']['sources']['path']));prices={r['security']:{x['date']:x for x in json.load(open(r['raw_path']))['data']} for r in refs};cal=json.load(open(cov['references']['calendar']['path']));days=[r['date'] for r in cal if r['engineering_calendar_ok'] and r['price_observed']];idx={d:i for i,d in enumerate(days)}
counts=Counter();cases=[]
for r in data:
 path=days[idx[r['entry']]:idx[r['exit']]+1];p=prices[r['stock_id']];missing=[d for d in path if d not in p];zero=[d for d in path if d in p and all(p[d][k]==0 for k in ['open','max','min','close'])]
 if missing:counts[r['split']+'_INTERIOR_MISSING']+=1
 if zero:counts[r['split']+'_INTERIOR_ZERO']+=1
 if r['stock_id']=='2321' and r['horizon']==10 and r['signal_session'] in ['2023-02-03','2023-02-06','2023-02-07']:cases.append({'stock_id':'2321','signal':r['signal_session'],'entry':r['entry'],'exit':r['exit'],'diagnostic_label':r['diagnostic_label'],'unobserved_intermediate_sessions':missing,'zero_price_intermediate_sessions':zero,'cause':'UNKNOWN_notconfirmedcorporateaction'})
# Verify preserved trace, never overwrite it or alter W24 outputs.
original=json.load(open(ROOT/'case-trace.json'));assert original['sample_counts_with_path_gaps']==dict(counts) and original['examples']==cases
print('Existing case trace reproduced:',len(data),'samples; no new filter or metric')
