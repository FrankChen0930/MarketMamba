import importlib.util,unittest,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('w26_source',Path(__file__).with_name('window26-sources.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SourceAndEventTest(unittest.TestCase):
 def test_no_price_or_token_or_unplanned_dataset(self):
  p={'dataset':'TaiwanStockCapitalReductionReferencePrice','data_id':'2321','start_date':'2021-01-01','end_date':'2023-12-31'};m.validate_query(p)
  for q in [{**p,'dataset':'TaiwanStockPrice'},{**p,'token':'bad'},{**p,'data_id':'other'}]:
   with self.assertRaises(ValueError):m.validate_query(q)
 def test_entry_exit_event_boundaries(self):
  self.assertTrue(m.crosses_event('2023-02-07','2023-02-20','2023-02-17'));self.assertFalse(m.crosses_event('2023-02-17','2023-02-20','2023-02-17'));self.assertFalse(m.crosses_event('2023-02-07','2023-02-16','2023-02-17'))
 def test_pending_or_existing_claim_cannot_dispatch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'finmind-01.claim.json').write_text('{}')
   with self.assertRaises(ValueError):m.reserve_request(p)

if __name__=="__main__":unittest.main()
