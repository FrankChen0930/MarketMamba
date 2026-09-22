import unittest
import numpy as np
import pandas as pd
from V6.experimental.v7_e1_evaluate import *
def frame(sign=1.):
 rows=[]
 for day in ("2026-01-02","2026-01-05","2026-01-06"):
  for i in range(20):
   rows.append({"Date":day,"stock_id":str(i),"Alpha_5d":float(i),
    "Alpha_10d":float(i),"prediction_5d":sign*i,"prediction_10d":sign*i})
 return pd.DataFrame(rows)
class E1EvaluateTest(unittest.TestCase):
 def test_rank_ensemble_and_metrics(self):
  ensemble=rank_average_ensemble({17:frame(),29:frame()})
  result=summarize(ensemble)
  self.assertAlmostEqual(result["mean_rank_ic_5d"],1.)
  self.assertGreater(result["mean_d10_minus_d1_10d"],0)
  self.assertEqual(result["positive_days_5d"],3)
 def test_pairing_is_exact_shared_support(self):
  new=frame(); old=frame(-1).iloc[:-1]
  daily,support=paired_daily_delta(new,old)
  self.assertEqual(support["paired_rows"],59)
  self.assertTrue((daily["delta_ic_10d"]>0).all())
 def test_bootstrap_is_deterministic(self):
  x=np.linspace(.01,.03,50)
  self.assertEqual(block_bootstrap_lower(x,repeats=100),
                   block_bootstrap_lower(x,repeats=100))
 def test_frozen_decision_vocabulary(self):
  seeds={17:summarize(frame()),29:summarize(frame())}; ensemble=summarize(frame())
  paired,_=paired_daily_delta(frame(),frame(-1))
  decision=decide(seeds,ensemble,paired,bootstrap_repeats=100)
  self.assertEqual(decision["outcome"],"PASS_REFRESH_HYPOTHESIS")
  bad=dict(ensemble,mean_rank_ic_5d=-.1,mean_rank_ic_10d=-.1)
  self.assertEqual(decide(seeds,bad,paired,100)["outcome"],"FAIL_REFRESH_HYPOTHESIS")
if __name__=="__main__": unittest.main()
