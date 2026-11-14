import tempfile,unittest
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from V6.experimental.v7_corrected_e5_matrix import MATRIX_ARRAYS
from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config,REQUIRED_SOURCE_HASHES
from V6.experimental.v7_e1_contract_test import built
from V6.experimental.v7_e1_matrix import *
ROOT=Path(__file__).resolve().parents[2]
def config(): return CorrectedE5Config.from_contract(
 ROOT/"research/v7/corrected-e5-replication-v1/incumbent-contract.json",
 ROOT/"research/v7/corrected-baseline-v1/feature-manifest.json")
class E1MatrixTest(unittest.TestCase):
 def test_arrays_use_e1_boundaries(self):
  cfg=config(); cols=list(cfg.feature_order)
  features=pd.DataFrame([{"Date":d,"stock_id":"1",**{c:1. for c in cols}}
   for d in ("2025-11-18","2025-11-19","2026-01-02")])
  universe=pd.DataFrame({"session":features.Date,"stock_id":["1"]*3,"membership":["ELIGIBLE"]*3})
  labels=pd.DataFrame({"signal_date":features.Date,"stock_id":["1"]*3,
                       "Alpha_5d":[1.,2.,3.],"Alpha_10d":[1.,2.,3.]})
  arrays=build_e1_arrays(features,universe,labels,cfg)
  self.assertEqual(arrays["split_names"].tolist(),["train","purge","research_evaluation"])
 def test_manifest_isolated_and_tamper_evident(self):
  c=built(); cfg=e1_config(config()); arrays={"dates":np.array(["2026-01-02"],dtype="datetime64[D]")}
  files={name:"a"*64 for name in MATRIX_ARRAYS}; sources={name:"b"*64 for name in REQUIRED_SOURCE_HASHES}
  doc=make_manifest(c,cfg,arrays,files,sources); self.assertEqual(doc["namespace"],MATRIX_NAMESPACE)
  bad=dict(doc); bad["split_contract"]=dict(SPLIT,train_end="2025-11-19")
  with self.assertRaises(ValueError): validate_manifest(bad,c,cfg)
 def test_rehashed_manifest_cannot_change_frozen_metadata(self):
  c=built();cfg=e1_config(config())
  original=make_manifest(c,cfg,{"dates":np.array(["2026-01-02"],dtype="datetime64[D]")},
   {name:"a"*64 for name in MATRIX_ARRAYS},{name:"b"*64 for name in REQUIRED_SOURCE_HASHES})
  mutations=[lambda d:d.update(feature_order=list(reversed(d["feature_order"]))),
   lambda d:d.update(files={}),lambda d:d["files"].update({"X.npy":"bad"}),
   lambda d:d.update(source_hashes={}),lambda d:d["source_hashes"].update(features="bad"),
   lambda d:d.update(rows=True),lambda d:d.update(rows=-1),
   lambda d:d["label_contract"].update(source_version="v1"),
   lambda d:d["preprocessing"].update(current_industry_backfill=True),
   lambda d:d["graph"].update(edge_count=1)]
  for mutate in mutations:
   d=copy.deepcopy(original);mutate(d)
   d["logical_identity"]=canonical_hash({k:v for k,v in d.items() if k!="logical_identity"})
   with self.subTest(mutation=mutate),self.assertRaises(ValueError):validate_manifest(d,c,cfg)
  validate_manifest(original,c,cfg)
 def test_v2_label_gate_is_exact(self):
  source={"classes":{"EXCHANGE_REGULAR_BOARD_VERIFIED":{"eligible":True},
                     "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR":{"eligible":True}}}
  coverage={"evidence_class":"HISTORICAL_SIMULATION_PROXY","selected_proxy_policy":"P0",
   "strict_phase0_decision":"STOP","materialization_status":"MATERIALIZED",
   "historical_simulated_valid_labels":{"5d":7023920,"10d":6923464},
   "corrected_verified_executable_labels":{"5d":0,"10d":0}}
  validate_e1_upstream(source,coverage)
  coverage["historical_simulated_valid_labels"]["5d"]=1
  with self.assertRaisesRegex(ValueError,"counts"): validate_e1_upstream(source,coverage)
 def test_write_rejects_old_namespace(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,"isolated namespace"):
    write_matrix(Path(d)/"v7_corrected_e5_replication_v1",built(),config(),{}, {})
if __name__=="__main__": unittest.main()
