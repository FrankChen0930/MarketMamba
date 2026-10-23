"""Synthetic audit evidence, never production feature admission."""
import unittest
from pathlib import Path
import pandas as pd
from V6.experimental.v7_window05_audit import inventory, quality, source_function

class FeatureAuditTest(unittest.TestCase):
 def test_exact_frozen_order_and_explicit_unknowns(self):
  rows=inventory(Path('.'))
  self.assertEqual(len(rows),48)
  self.assertEqual(rows[0]['canonical_name'],'Open')
  self.assertEqual(rows[-1]['canonical_name'],'FED_Rate')
  self.assertTrue(all(r['audit_conclusion'] in {'VERIFIED','CONDITIONAL','UNKNOWN','INVALID'} for r in rows))
  self.assertTrue(all(r['retrieval_timestamp'] is None for r in rows))
  self.assertTrue(all(r['quality_status']=='UNKNOWN' for r in rows))
 def test_date_only_macro_join_exposes_future_us_close(self):
  merge=source_function('_merge_macro')
  frame=pd.DataFrame({'Date':pd.to_datetime(['2026-07-15','2026-07-16']), 'stock_id':['A','A']})
  macro=pd.DataFrame({'Date':pd.to_datetime(['2026-07-15','2026-07-16']), 'SPX':[100.,110.]})
  got=merge(frame,macro)
  self.assertAlmostEqual(got['SPX_Return'].iloc[1],.1)
  # UTC+8 same-date TW close precedes US close; date-only joining cannot enforce it.
  from datetime import datetime
  self.assertGreater(datetime.fromisoformat('2026-07-17T04:00:00+08:00'),datetime.fromisoformat('2026-07-16T13:30:00+08:00'))
 def test_zero_fill_hides_missing_flow(self):
  got=source_function('_merge_institutional')(pd.DataFrame({'Date':['2026-07-16'],'stock_id':['A']}),None)
  self.assertEqual(got.Foreign_Net.iloc[0],0)
 def test_quality_preserves_unknown_and_computes_only_supplied_data(self):
  self.assertEqual(quality(None,['x'])['x']['status'],'UNKNOWN')
  df=pd.DataFrame({'Date':['2020-01-01','2020-01-02','2021-01-01'],'x':[0.,float('nan'),float('inf')]})
  row=quality(df,['x'])['x']
  self.assertEqual(row['nonfinite'],2);self.assertEqual(row['zero'],1)
  self.assertEqual(row['by_year']['2021']['finite'],0)
