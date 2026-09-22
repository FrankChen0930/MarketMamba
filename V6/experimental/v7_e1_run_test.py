import json
import tempfile,unittest
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from V6.experimental.v7_e1_checkpoint import *
from V6.experimental.v7_e1_contract_test import built
from V6.experimental.v7_e1_matrix_test import config
from V6.experimental.v7_e1_matrix import build_e1_arrays,write_matrix
from V6.experimental.v7_e1_run import gpu_sample,run_seed
class TinyModel(nn.Module):
 def __init__(self,c):
  super().__init__(); self.head=nn.Linear(len(c.feature_order),2)
 def forward(self,x,stock_ids,observation_mask=None): return self.head(x[:,-1])
class E1RunTest(unittest.TestCase):
 def test_lifecycle_adapter_round_trip(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); durable=DurableCheckpointStore(root/"drive",root/"local","c"*64,"m"*64,17)
   adapter=LifecycleCheckpointAdapter(durable)
   state={"epoch":2,"batch":3,"step":9,"phase":"validation","terminal":False,
    "model_state":{"x":1},"optimizer_state":{},"scheduler_state":{},"rng_state":{}}
   adapter.save(state,best=True)
   self.assertEqual(adapter.load("c"*64,"m"*64)["step"],9)
   self.assertEqual(adapter.load("c"*64,"m"*64,best=True)["model_state"],{"x":1})
   telemetry=adapter.snapshot_epoch()
   self.assertGreater(telemetry["checkpoint_seconds"],0)
   self.assertGreater(telemetry["local_checkpoint_seconds"],0)
   self.assertGreater(telemetry["drive_checkpoint_seconds"],0)
   self.assertIn("train_seconds",telemetry)
   self.assertIn("validation_seconds",telemetry)
   self.assertIn("updates_per_second",telemetry)
 def test_gpu_probe_is_optional(self): self.assertIsInstance(gpu_sample(),dict)
 def test_tiny_cpu_pipeline_reloads_durable_checkpoint(self):
  c=built(); cfg=config(); features=[]; universe=[]; labels=[]
  for day in ("2025-11-18","2026-01-02"):
   for offset,stock in enumerate(("1101","2330","2603")):
    row={"Date":day,"stock_id":stock}
    row.update({name:float(i+offset) for i,name in enumerate(cfg.feature_order)})
    features.append(row); universe.append({"session":day,"stock_id":stock,"membership":"ELIGIBLE"})
    labels.append({"signal_date":day,"stock_id":stock,"Alpha_5d":float(offset),
                   "Alpha_10d":float(2-offset)})
  arrays=build_e1_arrays(pd.DataFrame(features),pd.DataFrame(universe),pd.DataFrame(labels),cfg)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); matrix=root/"v7_e1_rolling_origin_refresh_v1"
   write_matrix(matrix,c,cfg,arrays,{k:k[0]*64 for k in ("features","labels","universe","provenance")})
   result=run_seed(matrix,root/"output",c,cfg,17,torch.device("cpu"),
                   smoke_steps=1,model_factory=TinyModel)
   self.assertEqual(result["status"],"smoke_complete")
   self.assertTrue(result["checkpoint_reload"])
   pointer=root/"output/seed-17/checkpoints/checkpoint.json"
   self.assertTrue(pointer.is_file())
   doc=json.loads(pointer.read_text())
   self.assertEqual(len(doc["latest"]["sha256"]),64)

if __name__=="__main__": unittest.main()
