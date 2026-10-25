import importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('src',Path(__file__).with_name('window29-sources.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Budget(unittest.TestCase):
 def test_allowlist_rejects_token_unknown_source_and_dates(self):
  q={'dataset':'TaiwanStockPrice','data_id':'X','start_date':'2021-01-01','end_date':'2023-12-31'};c={'price_queries':['X']};m.validate_query(q,c)
  for x in ({**q,'token':'bad'},{**q,'dataset':'Other'},{**q,'data_id':'Y'},{**q,'start_date':'2010-01-01'}):
   with self.assertRaises(ValueError):m.validate_query(x,c)
 def test_total_price_shared_limits(self):
  m.budget_check(68,68,99)
  for args in ((80,68,0),(70,70,0),(1,1,100)):
   with self.assertRaises(ValueError):m.budget_check(*args)
 def test_stop_signals_and_no_incomplete_success(self):
  for code in (402,403,429,302):self.assertTrue(m.source_stop(code,{'status':'OBSERVED_PRICE'},True))
  self.assertTrue(m.source_stop(200,{'status':'SOURCE_ERROR'},True));self.assertTrue(m.source_stop(200,{'status':'OBSERVED_PRICE'},False));self.assertFalse(m.source_stop(200,{'status':'MISSING_DATA'},True))
if __name__=='__main__':unittest.main()
