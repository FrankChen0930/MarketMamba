import copy
from pathlib import Path
import unittest
from V6.experimental.v7_e1_contract import *
ROOT=Path(__file__).resolve().parents[2]
def built():
 return build_contract(ROOT/"research/v7/corrected-e5-replication-v1/incumbent-contract.json",
  ROOT/"research/v7/corrected-e5-diagnostic-v2/next-experiment-decision.json",
  ROOT/"research/v7/corrected-baseline-v1/feature-manifest.json")
class E1ContractTest(unittest.TestCase):
 def test_contract_freezes_model_training_data_and_scope(self):
  c=built()
  self.assertEqual(c["frozen"]["architecture"]["d_state"],32)
  self.assertFalse(c["frozen"]["architecture"]["graph"]["enabled"])
  self.assertEqual(c["frozen"]["features"]["group_dims"],[15,20,1,12])
  self.assertEqual(c["frozen"]["training"]["selection_metric"],"mean_daily_rank_ic_5d")
  self.assertEqual(c["seeds"],[17,29]); self.assertEqual(c["split"],SPLIT)
  self.assertEqual(c["matrix"]["namespace"],MATRIX_NAMESPACE)
  self.assertEqual(c["frozen"]["labels"]["valid_5d"],LABEL_COUNTS["5d"])
  self.assertEqual(len(c["contract_sha256"]),64)
 def test_drift_fails_closed(self):
  cases=[(("frozen","architecture","d_state"),8),
   (("frozen","training","learning_rate"),.001),
   (("frozen","preprocessing","industry_neutralization"),True),
   (("split","train_end"),"2025-11-19"),
   (("correctness_semantics","strict_phase0"),"PASS"),
   (("frozen","labels","valid_5d"),1)]
  for path,value in cases:
   with self.subTest(path=path):
    c=copy.deepcopy(built()); x=c
    for k in path[:-1]: x=x[k]
    x[path[-1]]=value
    with self.assertRaises(ValueError): validate_contract(c)
 def test_every_frozen_contract_field_and_identity_is_bound(self):
  mutations=[lambda c:c.update(output_namespace="other"),
   lambda c:c.update(hypothesis="different hypothesis"),
   lambda c:c.update(baseline="other"),
   lambda c:c["frozen"].update(environment="other"),
   lambda c:c["frozen"]["labels"].update(source_version="v1"),
   lambda c:c["matrix"].update(ordering=["stock_id","Date"]),
   lambda c:c["correctness_semantics"].update(evidence_class="VERIFIED_EXECUTABLE")]
  for mutation in mutations:
   for rehash in (False,True):
    c=built();mutation(c)
    if rehash: c["contract_sha256"]=canonical_hash({k:v for k,v in c.items() if k!="contract_sha256"})
    with self.subTest(mutation=mutation,rehash=rehash),self.assertRaises(ValueError):
     validate_contract(c)
  for identity in (None,"a"*64):
   c=built()
   if identity is None: c.pop("contract_sha256")
   else: c["contract_sha256"]=identity
   with self.assertRaises(ValueError): validate_contract(c)
 def test_checked_in_frozen_contract_remains_accepted(self):
  import json
  c=json.loads((ROOT/"research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json").read_text())
  validate_contract(c)
  self.assertEqual(c,built())
 def test_diff_and_boundaries(self):
  self.assertEqual(contract_diff(built())["substantive_changes"][0]["field"],"split.train_end")
  cases={"2013-01-01":"warmup","2013-01-02":"train","2025-11-18":"train",
   "2025-11-19":"purge","2025-12-31":"purge","2026-01-02":"research_evaluation",
   "2026-09-04":"research_evaluation","2026-09-05":"out_of_contract"}
  self.assertEqual({d:assign_split(d) for d in cases},cases)
 def test_split_validation_requires_one_assignment_per_date(self):
  for dates,splits in [(["2025-11-18", "2025-11-19"], ["train"]),
                       (["2025-11-18"], ["train", "purge"]),
                       ([], ["train"]), (["2025-11-18"], [])]:
   for as_iterator in (False, True):
    with self.subTest(dates=dates,splits=splits,iterator=as_iterator):
     with self.assertRaises(ValueError):
      validate_split_rows(iter(dates) if as_iterator else dates,
                          iter(splits) if as_iterator else splits)
 def test_split_validation_preserves_complete_iterable_boundaries(self):
  dates=["2013-01-01","2013-01-02","2025-11-18","2025-11-19",
         "2025-12-31","2026-01-02","2026-09-04","2026-09-05"]
  splits=["warmup","train","train","purge","purge",
          "research_evaluation","research_evaluation","out_of_contract"]
  validate_split_rows(iter(dates),iter(splits))
  with self.assertRaisesRegex(ValueError,"no rows"):
   validate_split_rows(iter(()),iter(()))
 def test_split_validation_rejects_leakage(self):
  validate_split_rows(["2025-11-18","2025-11-19","2026-01-02"],
                      ["train","purge","research_evaluation"])
  with self.assertRaisesRegex(ValueError,"split mismatch"):
   validate_split_rows(["2025-11-19"],["train"])

if __name__=="__main__":
 unittest.main()
