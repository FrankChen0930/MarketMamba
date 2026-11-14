import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v7_integrated_config import V6_FEATURE_COLUMNS
from v7_integrated_industry import clean_and_scale_industry,industry_mapping

def identity(frame,**kwargs):return frame.copy().fillna(0)
def fixtures():
 f=pd.DataFrame({c:np.array([1.,3.,10.,14.,20.]) for c in V6_FEATURE_COLUMNS})
 f['Date']=pd.Timestamp('2026-09-11');f['stock_id']=['A','B','C','D','U'];f['Alpha_5d']=np.arange(5.)
 info=pd.DataFrame({'stock_id':['A','B','C','D'],'industry_category':['半導體業']*2+['食品工業']*2,'date':['2026-05-06']*4})
 return f,info

class IndustryTests(unittest.TestCase):
 def test_sector_means_zero_and_labels_exclusions_unchanged(self):
  f,info=fixtures();out,r=clean_and_scale_industry(f,info,identity)
  self.assertAlmostEqual(out.Return_1d.iloc[:2].mean(),0)
  self.assertAlmostEqual(out.Return_1d.iloc[2:4].mean(),0)
  pd.testing.assert_frame_equal(out[['Open','Market_Cap_Log','VIX','Alpha_5d']],f[['Open','Market_Cap_Log','VIX','Alpha_5d']])
  self.assertEqual(r['unknown_stock_ids'],['U'])
  self.assertEqual(len(out),len(f));self.assertNotEqual(out.Return_1d.iloc[4],0)
 def test_missing_cells_are_not_artificial_peers(self):
  f,info=fixtures();f.loc[1,'Return_1d']=np.nan
  out,r=clean_and_scale_industry(f,info,identity)
  self.assertEqual(out.Return_1d.iloc[1],0);self.assertNotEqual(out.Return_1d.iloc[0],0)
  self.assertGreater(r['thin_group_cells'],0)
 def test_future_rows_cannot_change_past_values(self):
  f,info=fixtures();out,_=clean_and_scale_industry(f,info,identity)
  future=f.copy();future.Date=pd.Timestamp('2026-09-14');future['Return_1d']*=100
  extended,_=clean_and_scale_industry(pd.concat([f,future],ignore_index=True),info,identity)
  pd.testing.assert_frame_equal(out,extended.iloc[:len(f)])
 def test_no_classification_blocks_instead_of_silent_success(self):
  f,_=fixtures()
  with self.assertRaises(ValueError):clean_and_scale_industry(f,pd.DataFrame(),identity)
 def test_non_industry_labels_are_not_peer_groups(self):
  f,info=fixtures();info.industry_category='存託憑證'
  with self.assertRaises(ValueError):clean_and_scale_industry(f,info,identity)
 def test_classification_conflict_blocks(self):
  _,info=fixtures();other=info.iloc[[0]].copy();other.industry_category='食品工業'
  with self.assertRaises(ValueError):industry_mapping(pd.concat([info,other],ignore_index=True))
 def test_order_independence_and_delisted_stock_not_filtered(self):
  f,info=fixtures();info['type']='delisted'
  a,_=clean_and_scale_industry(f,info,identity)
  b,_=clean_and_scale_industry(f.iloc[::-1],info.iloc[::-1],identity)
  pd.testing.assert_frame_equal(a,b.loc[a.index])

class ChunkedIndustryTests(unittest.TestCase):
 def test_date_chunks_preserve_macro_history_and_full_cross_sections(self):
  import tempfile
  from marketmamba.data.feature_engineer import clean_and_scale
  from v7_integrated_industry import clean_and_scale_industry_chunked
  days=pd.bdate_range("2020-01-01",periods=270)
  rng=np.random.default_rng(42)
  n=len(days)*4
  f=pd.DataFrame({c:rng.normal(size=n) for c in V6_FEATURE_COLUMNS})
  f["Date"]=np.tile(days,4);f["stock_id"]=np.repeat(["A","B","C","D"],len(days))
  for c in V6_FEATURE_COLUMNS[47:]:
   f[c]=np.tile(np.arange(len(days),dtype=float)**1.2,4)
  f["Alpha_5d"]=1.;f["Alpha_10d"]=2.
  f.loc[0,list(V6_FEATURE_COLUMNS[:47])]=np.nan
  info=pd.DataFrame({"stock_id":["A","B","C","D"],"industry_category":["半導體業"]*2+["食品工業"]*2})
  expected,er=clean_and_scale_industry(f,info,clean_and_scale)
  with tempfile.TemporaryDirectory() as directory:
   actual,ar=clean_and_scale_industry_chunked(f,info,clean_and_scale,work_dir=Path(directory)/"parts",sessions_per_chunk=91)
  sort=lambda x:x.sort_values(["stock_id","Date"]).reset_index(drop=True)
  pd.testing.assert_frame_equal(sort(expected),sort(actual),rtol=1e-12,atol=1e-12)
  self.assertGreater(actual.VIX.abs().sum(),0)
  for key in ["rows","neutralized_cells","thin_group_cells","unknown_rows"]:
   self.assertEqual(er[key],ar[key])

if __name__=='__main__':unittest.main()
