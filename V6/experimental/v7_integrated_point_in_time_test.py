import sys,unittest
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v7_integrated_point_in_time import align_point_in_time_features

class AlignmentTests(unittest.TestCase):
 def run_alignment(self,dates,revenue,mv=None):
  frame=pd.DataFrame({'Date':pd.to_datetime(dates),'stock_id':['A']*len(dates),'Revenue_MoM':[99.]*len(dates),'Revenue_YoY':[99.]*len(dates),'Market_Cap_Log':[99.]*len(dates)})
  return align_point_in_time_features(frame,revenue,mv,pd.bdate_range('2025-01-01','2026-10-30').strftime('%Y-%m-%d').tolist())
 def test_actual_announcement_visible_next_session(self):
  rev=pd.DataFrame({'Date':['2026-08-01','2026-09-01'],'stock_id':['A','A'],'revenue':[100.,150.],'create_time':['2026-08-10','2026-09-10']})
  x,_=self.run_alignment(['2026-09-10','2026-09-11'],rev)
  self.assertTrue(np.isnan(x.Revenue_MoM.iloc[0]));self.assertAlmostEqual(x.Revenue_MoM.iloc[1],.5)
 def test_monthly_missing_month_does_not_mean_previous_row(self):
  rev=pd.DataFrame({'Date':['2026-07-01','2026-09-01'],'stock_id':['A','A'],'revenue':[100.,150.],'create_time':['2026-07-10','2026-09-10']})
  x,_=self.run_alignment(['2026-09-11'],rev);self.assertTrue(np.isnan(x.Revenue_MoM.iloc[0]))
 def test_later_revision_cannot_change_earlier_feature(self):
  rev=pd.DataFrame({'Date':['2026-08-01','2026-09-01','2026-08-01'],'stock_id':['A']*3,'revenue':[100.,150.,200.],'create_time':['2026-08-10','2026-09-10','2026-09-15']})
  x,_=self.run_alignment(['2026-09-11','2026-09-16'],rev)
  self.assertAlmostEqual(x.Revenue_MoM.iloc[0],.5);self.assertAlmostEqual(x.Revenue_MoM.iloc[1],-.25)
 def test_unknown_date_policy_is_explicit_and_not_one_month_late(self):
  rev=pd.DataFrame({'Date':['2026-08-01','2026-09-01'],'stock_id':['A']*2,'revenue':[100.,150.]})
  x,r=self.run_alignment(['2026-09-10','2026-09-11'],rev)
  self.assertTrue(np.isnan(x.Revenue_MoM.iloc[0]));self.assertAlmostEqual(x.Revenue_MoM.iloc[1],.5)
  self.assertEqual(r['revenue_assumed_release_rows'],2)
 def test_market_cap_carries_prior_history_but_expires(self):
  mv=pd.DataFrame({'Date':['2026-09-03'],'stock_id':['A'],'market_value':[100.]})
  x,_=self.run_alignment(['2026-09-04','2026-09-14'],None,mv)
  self.assertAlmostEqual(x.Market_Cap_Log.iloc[0],np.log1p(100));self.assertTrue(np.isnan(x.Market_Cap_Log.iloc[1]))
 def test_empty_source_does_not_create_raw_zero(self):
  x,_=self.run_alignment(['2026-09-11'],None)
  self.assertTrue(x[['Revenue_MoM','Revenue_YoY','Market_Cap_Log']].isna().all().all())

 def test_first_observation_cannot_be_backdated(self):
  rev=pd.DataFrame({'Date':['2026-08-01','2026-09-01'],'stock_id':['A']*2,'revenue':[100.,150.],'first_observed_at':['','2026-09-12']})
  x,_=self.run_alignment(['2026-09-11','2026-09-14'],rev)
  self.assertTrue(np.isnan(x.Revenue_MoM.iloc[0]));self.assertAlmostEqual(x.Revenue_MoM.iloc[1],.5)
 def test_pre_window_revisions_keep_order(self):
  rev=pd.DataFrame({'Date':['2024-11-01','2024-12-01','2024-11-01'],'stock_id':['A']*3,'revenue':[100.,150.,200.],'create_time':['2024-11-10','2024-12-10','2024-12-15']})
  x,_=self.run_alignment(['2025-01-02'],rev)
  self.assertAlmostEqual(x.Revenue_MoM.iloc[0],-.25)

if __name__=='__main__':unittest.main()
