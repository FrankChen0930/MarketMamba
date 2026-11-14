"""No network/models; raw CSV mapping and identical inherited feature semantics."""
import runpy,unittest
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('window33-compatibility.py')))
class Compatibility(unittest.TestCase):
 def test_raw_shares_and_prices(self):
  row={'日期':'1151022','證券代號':'SYNTH','開盤價':'10.00','最高價':'11.00','最低價':'9.00','收盤價':'10.50','成交股數':'1000'};r=M['official_row'](row);self.assertEqual(r['date'],'2026-10-22');self.assertEqual(r['Trading_Volume'],1000);self.assertEqual(r['close'],10.5)
 def test_missing_not_zero_and_invalid_not_fixed(self):
  row={'日期':'1151022','證券代號':'SYNTH','開盤價':'--','最高價':'11','最低價':'9','收盤價':'0','成交股數':'1000'};r=M['official_row'](row);self.assertIsNone(r['open']);self.assertEqual(r['close'],0)
 def test_same_feature_formula_and_order_with_pinned_prior(self):
  rows={'2026-10-21':{'open':10.,'max':11.,'min':9.,'close':10.,'Trading_Volume':1000},'2026-10-22':{'open':10.,'max':11.,'min':9.,'close':10.5,'Trading_Volume':1100}}
  a=M['feature']('SYNTH','2026-10-22',['2026-10-21','2026-10-22'],rows,{'reported_twse_start':'2020-01-01','reported_end':None},[]);self.assertEqual(a['features'],[10.5/10-1,2/10.5]);self.assertTrue(a['eligible'])
 def test_observed_calendar_cannot_be_promoted(self):
  from V6.experimental.v7_research_temporal import _calendar
  cal={'source_sha256':'a'*64,'vintage_id':'observed-only','qualification_ref':'price-union-not-official','qualification':'OBSERVED_MARKET_SESSION','sessions':[{'session':'2026-10-22'}]}
  self.assertEqual(_calendar(cal)[2],'CALENDAR_QUALIFICATION_UNKNOWN')
if __name__=='__main__':unittest.main()
