"""Offline request guard tests; never dispatch a real GET."""
import unittest,runpy
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('window33-sources.py')))
class SourceGuards(unittest.TestCase):
 def setUp(self):self.q={'dataset':'TaiwanStockPrice','data_id':'SYNTH','start_date':'2024-01-01','end_date':'2026-10-22'};self.c={'price_queries':['SYNTH']}
 def test_exact_fixed_query_allowed(self):M['validate_query'](self.q,self.c)
 def test_unapproved_dates_dataset_security_and_token_refused(self):
  for k,v in [('end_date','2026-10-23'),('start_date','2023-01-01'),('dataset','TaiwanStockPriceAdj'),('data_id','UNKNOWN'),('token','unapproved')]:
   q={**self.q,k:v}
   with self.assertRaises(ValueError):M['validate_query'](q,self.c)
 def test_total_price_and_shared_hour_caps(self):
  M['budget_check'](63,63,99)
  for v in [(70,64,0),(64,70,0),(1,1,100)]:
   with self.assertRaises(ValueError):M['budget_check'](*v)
 def test_restrictions_incomplete_and_schema_fail_stop(self):
  for code in [402,403,429,500]:self.assertTrue(M['source_stop'](code,{'status':'OBSERVED_PRICE'},True))
  self.assertTrue(M['source_stop'](200,{'status':'SOURCE_ERROR'},True));self.assertTrue(M['source_stop'](200,{'status':'OBSERVED_PRICE'},False))
if __name__=='__main__':unittest.main()
