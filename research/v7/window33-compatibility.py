"""Read-only W16/W18 TWSE raw mapping; same W29 two-feature function, no fitting."""
import runpy
from pathlib import Path
from datetime import date
from decimal import Decimal
P=runpy.run_path(str(Path(__file__).with_name('window29-panel.py')))
feature=P['feature_at']
def official_row(r):
 raw=r['日期'];day=date(int(raw[:3])+1911,int(raw[3:5]),int(raw[5:])).isoformat();out={'date':day,'stock_id':r['證券代號']}
 for k,v in [('open','開盤價'),('max','最高價'),('min','最低價'),('close','收盤價'),('Trading_Volume','成交股數')]:
  raw=r[v].strip();out[k]=None if raw in {'','--','---','----','N/A'} else float(Decimal(raw.replace(',','')))
 return out
