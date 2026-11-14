"""Synthetic request/semantics guards; no actual API calls."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from V6.experimental.v7_finmind_proof import request,diagnose,restore
P={'dataset':'TaiwanStockPrice','data_id':'2330','start_date':'2010-01-04','end_date':'2010-01-15'}
def body(**kw):
 row={'date':'2010-01-04','stock_id':'2330','open':10,'max':12,'min':9,'close':11,'Trading_Volume':1000};row.update(kw)
 return json.dumps({'status':200,'msg':'success','data':[row]}).encode()
class FinMindProofTest(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name);self.fetch=Mock(return_value=(200,{'Content-Type':'application/json'},body(),True))
 def call(self,params=None,kind='price'):return request(self.root,params or P,kind=kind,fetch=self.fetch)
 def test_valid_zero_and_missing_semantics(self):
  self.assertEqual(diagnose(body(),P)['counts']['VALID_PRICE'],1)
  self.assertEqual(diagnose(body(open=0,max=0,min=0,close=0,Trading_Volume=451),P)['counts']['NO_PUBLISHED_PRICE'],1)
  self.assertEqual(diagnose(body(close=None),P)['counts']['MISSING_DATA'],1)
 def test_emerging_only_with_explicit_context_no_universe_inference(self):
  raw=body(open=20)
  self.assertEqual(diagnose(raw,P)['counts']['UNVERIFIED'],1)
  c={'2010-01-04':{'market':'emerging','evidence':'synthetic-source'}}
  self.assertEqual(diagnose(raw,P,market_context=c)['counts']['MARKET_SEMANTIC_EXCEPTION'],1)
 def test_mismatched_stock_date_and_invalid_schema_rejected(self):
  for raw in [body(stock_id='OTHER'),body(date='2020-01-01'),b'{}',b'<html/>',body(Trading_Volume=-1)]:self.assertEqual(diagnose(raw,P)['status'],'SOURCE_ERROR')
 def test_durable_raw_and_restore_and_lost_bytes(self):
  r=self.call();self.assertEqual(r['status'],'CAPTURED')
  q=restore(self.root,r['record_path'],r['record_sha256'],self.root/'restore');self.assertEqual(q['status'],'RESTORED_BYTES_HASH_SCHEMA_RECORDS_VERIFIED')
  (self.root/r['raw_path']).unlink();self.assertEqual(restore(self.root,r['record_path'],r['record_sha256'],self.root/'restore2')['status'],'NOT_RESTORABLE')
 def test_duplicate_normal_query_refused_repeat_distinct_capture_no_overwrite(self):
  r=self.call();self.assertEqual(self.call()['status'],'REFUSED');self.assertEqual(self.fetch.call_count,1)
  rr=self.call(kind='repeat');self.assertEqual(rr['raw_sha256'],r['raw_sha256']);self.assertEqual(self.fetch.call_count,2)
  self.assertEqual(self.call(kind='repeat')['status'],'REFUSED');self.assertEqual(self.fetch.call_count,2)
 def test_access_error_stops_all_without_retry(self):
  for code in [403,429,302]:
   with tempfile.TemporaryDirectory() as d:
    f=Mock(return_value=(code,{'Content-Type':'text/html'},b'blocked',True));r=request(Path(d),P,kind='price',fetch=f);self.assertEqual(r['status'],'STOPPED_SOURCE_ERROR');request(Path(d),{**P,'data_id':'2317'},kind='price',fetch=f);self.assertEqual(f.call_count,1)
 def test_api_application_denial_and_partial_stop(self):
  for raw,complete in [(b'{"status":400,"msg":"token required","data":[]}',True),(body(),False)]:
   with tempfile.TemporaryDirectory() as d:
    f=Mock(return_value=(200,{'Content-Type':'application/json'},raw,complete));self.assertEqual(request(Path(d),P,kind='price',fetch=f)['status'],'STOPPED_SOURCE_ERROR')
 def test_token_unknown_dataset_and_large_window_no_dispatch(self):
  for p in [{**P,'token':'notallowed'},{**P,'dataset':'TaiwanStockPriceAdj'},{**P,'end_date':'2011-01-01'}]:self.assertEqual(self.call(p)['status'],'REFUSED')
  self.fetch.assert_not_called()
 def test_total_metadata_and_symbol_limits(self):
  for ds in ['TaiwanStockInfo','TaiwanStockTradingDate','TaiwanStockDelisting']:
   self.fetch.return_value=(200,{'Content-Type':'application/json'},b'{"status":200,"data":[]}',True);self.assertEqual(self.call({'dataset':ds},'metadata')['status'],'CAPTURED')
  self.assertEqual(self.call({'dataset':'TaiwanStockInfo'},'metadata')['status'],'REFUSED');self.assertEqual(self.fetch.call_count,3)
 def test_exact_22_total_and_six_symbol_three_window_budget(self):
  def fake(params):
   raw=b'{"status":200,"data":[]}' if params['dataset']!='TaiwanStockPrice' else body(stock_id=params['data_id'],date=params['start_date'])
   return 200,{'Content-Type':'application/json'},raw,True
  self.fetch.side_effect=fake;first=None
  for sec in range(6):
   for day in ['2010-01-04','2010-02-01','2010-03-01']:
    p={'dataset':'TaiwanStockPrice','data_id':str(1000+sec),'start_date':day,'end_date':day};first=first or p
    self.assertEqual(self.call(p)['status'],'CAPTURED')
  for ds in ['TaiwanStockInfo','TaiwanStockTradingDate','TaiwanStockDelisting']:self.assertEqual(self.call({'dataset':ds},'metadata')['status'],'CAPTURED')
  self.assertEqual(self.call(first,'repeat')['status'],'CAPTURED');self.assertEqual(self.fetch.call_count,22)
  self.assertEqual(self.call({**P,'data_id':'OTHER'})['status'],'REFUSED');self.assertEqual(self.fetch.call_count,22)

 def test_unknown_calendar_empty_not_claim_missing_trading_day(self):
  q=diagnose(b'{"status":200,"data":[]}',P);self.assertEqual(q['status'],'MISSING_DATA');self.assertFalse(q['missing_sessions_verified'])

class ReviewRegressionTest(unittest.TestCase):
 setUp=FinMindProofTest.setUp
 call=FinMindProofTest.call
 def test_missing_published_raw_stops_before_repeat_no_repair(self):
  r=self.call();raw=self.root/r['raw_path'];raw.unlink()
  x=self.call(kind='repeat');self.assertEqual(x['status'],'STOPPED_EVIDENCE_FAILURE');self.assertEqual(self.fetch.call_count,1);self.assertFalse(raw.exists());self.assertTrue((self.root/'STOP.json').exists())
 def test_incomplete_read_consumes_claim_and_stops_repeat(self):
  import http.client
  self.fetch.side_effect=http.client.IncompleteRead(b'partial',10)
  self.assertEqual(self.call()['status'],'FAILED');self.assertEqual(len(list((self.root/'claims').glob('*.json'))),1)
  self.assertTrue((self.root/'STOP.json').exists());self.call(kind='repeat');self.assertEqual(self.fetch.call_count,1)
 def test_dangling_claim_without_terminal_fails_closed(self):
  self.root.joinpath('claims').mkdir();(self.root/'claims/01.json').write_text(json.dumps({'sequence':1,'params':P,'kind':'price'}))
  self.assertEqual(self.call(kind='repeat')['status'],'STOPPED_EVIDENCE_FAILURE');self.fetch.assert_not_called()
 def test_closed_window_refuses_unused_budget(self):
  (self.root/'CLOSED.json').write_text('{"status":"WINDOW_COMPLETE"}')
  self.assertEqual(self.call()['status'],'WINDOW_CLOSED');self.fetch.assert_not_called()

if __name__=='__main__':unittest.main()

class Window22ScopeTest(unittest.TestCase):
 def test_continuous_window_separate_scope_and_no_reopen(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p={**P,'start_date':'2019-01-01','end_date':'2023-12-31'}
   f=Mock(return_value=(200,{},body(date='2019-01-02'),True))
   r=request(root,p,kind='price',fetch=f,scope='W22')
   self.assertEqual(r['status'],'CAPTURED');self.assertEqual(r['authority_scope'],'W22')
   self.assertEqual(request(root,P,kind='price',fetch=f)['status'],'REFUSED');self.assertEqual(f.call_count,1)
 def test_w22_rejects_unplanned_diagnostic_and_outside_period(self):
  with tempfile.TemporaryDirectory() as d:
   f=Mock()
   for p,k in [(P,'price'),({**P,'start_date':'2019-01-01','end_date':'2023-12-31'},'repeat')]:
    self.assertEqual(request(Path(d),p,kind=k,fetch=f,scope='W22')['status'],'REFUSED')
   f.assert_not_called()

class Window24ScopeTest(unittest.TestCase):
 def test_frozen_cohort_only_single_contiguous_query_and_no_old_scope(self):
  from V6.experimental.v7_corrected_e5_replication import canonical_hash
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c={'new_price_queries':['1101'],'metadata_queries':[]};(root/'FROZEN_COHORT.json').write_text(json.dumps(c));(root/'COHORT_PIN.json').write_text(json.dumps({'semantic_sha256':canonical_hash(c)}))
   p={**P,'data_id':'1101','start_date':'2021-01-01','end_date':'2023-12-31'};f=Mock(return_value=(200,{},body(date='2021-01-04',stock_id='1101'),True))
   a=request(root/'finmind',p,kind='price',fetch=f,scope='W24');self.assertEqual(a['status'],'CAPTURED')
   self.assertEqual(request(root/'finmind',{**p,'data_id':'9999'},kind='price',fetch=f,scope='W24')['status'],'REFUSED');self.assertEqual(f.call_count,1)
   c['new_price_queries'].append('9999');(root/'FROZEN_COHORT.json').write_text(json.dumps(c));self.assertEqual(request(root/'finmind',p,kind='price',fetch=f,scope='W24')['status'],'REFUSED');self.assertEqual(f.call_count,1)
