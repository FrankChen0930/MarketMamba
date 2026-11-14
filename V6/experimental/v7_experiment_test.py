"""Regression tests for durable state, exact resume and adaptive experiment order."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from v7_experiment_storage import CheckpointStore, fingerprint, verify_matrix
from v7_experiment_train import run_experiment, lr_factor, update_selection
from v7_experiment_suite import resolve_spec, normalize_spec, STAGES

class ToyDataset:
    def __init__(self,n):self.n=n
    def __len__(self):return self.n
    def __getitem__(self,i):
        x=torch.sin(torch.arange(4*3*59).reshape(4,3,59).float()*.07+i)
        return {"x":x,"labels":torch.tensor([[3.,1.],[1.,4.],[4.,2.],[2.,3.]]),
                "padding_mask":torch.ones(4,3,dtype=torch.bool),
                "observation_mask":torch.ones(4,3,dtype=torch.bool),
                "edge_index":torch.empty(2,0,dtype=torch.long),"edge_attr":torch.empty(0),
                "stock_ids":("1","2","3","4"),"groups":[(0,torch.arange(4))],"Date":str(i)}

class ToyModel(nn.Module):
    def __init__(self,config,ssd_fn=None):
        super().__init__();self.drop=nn.Dropout(.25);self.linear=nn.Linear(59,2)
    def forward_prepared(self,sample):
        return self.linear(self.drop(sample["x"].mean(1)))

SETTINGS=dict(epochs=2,seed=17,learning_rate=.001,weight_decay=.01,warmup_fraction=.15,
              patience=5,min_epochs=5,min_delta=0.,grad_clip=1.,precision="fp32",
              prefetch=True,checkpoint_interval=2,checkpoint_seconds=180,progress_interval=1)

def invoke(path,**kwargs):
    return run_experiment(normalize_spec(resolve_spec(0,{})),SETTINGS,ToyDataset(3),ToyDataset(2),
        path,"fixture",device=torch.device("cpu"),ssd_fn=None,model_factory=ToyModel,
        emit=kwargs.pop("emit",lambda *a,**k:None),**kwargs)

class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_train_resume_exact_with_dropout(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);full=invoke(root/"full")
            paused=invoke(root/"resume",invocation_steps=2)
            self.assertEqual(paused["status"],"paused")
            resumed=invoke(root/"resume")
            self.assertEqual(full["step"],6);self.assertEqual(resumed["step"],6)
            a=CheckpointStore(root/"full/checkpoints").load(full["identity"])
            b=CheckpointStore(root/"resume/checkpoints").load(full["identity"])
            for k in a["model"]:torch.testing.assert_close(a["model"][k],b["model"][k],rtol=0,atol=0)
            self.assertEqual(a["scheduler"],b["scheduler"])
            self.assertEqual(a["best_metric"],b["best_metric"])
            for x,y in zip(a["history"],b["history"]):self.assertEqual(x["train_loss"],y["train_loss"])
    def test_validation_resume_no_retrain(self):
        with tempfile.TemporaryDirectory() as td:
            stop=[False]
            def emit(event,**kw):
                if event=="validation_progress":stop[0]=True
            paused=invoke(td,emit=emit,stop_requested=lambda:stop[0])
            state=CheckpointStore(Path(td)/"checkpoints").load(paused["identity"])
            self.assertEqual((state["phase"],state["batch"],state["val_index"],state["step"]),("validate",3,1,3))
            done=invoke(td)
            self.assertEqual(done["step"],6)
            self.assertEqual(len(done["history"]),2)
    def test_checkpoint_corruption_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            s=CheckpointStore(td)
            for i in range(4):s.save({"identity":"a","step":i},best=i==0)
            self.assertEqual(len(list(Path(td).glob("state-*.pt"))),3)
            (Path(td)/s.manifest()["latest"]["file"]).write_bytes(b"interrupted")
            self.assertEqual(s.load("a")["step"],2)
            self.assertEqual(s.load("a",best=True)["step"],0)
            with self.assertRaises(ValueError):s.load("b")
    def test_schedule_and_patience(self):
        self.assertEqual(lr_factor(0,1,.15),1.)
        self.assertAlmostEqual(lr_factor(0,100,.15),.04)
        self.assertAlmostEqual(lr_factor(14,100,.15),1.)
        self.assertAlmostEqual(lr_factor(99,100,.15),.0001)
        s={"best_metric":None,"patience_reference":None,"bad_epochs":0}
        self.assertTrue(update_selection(s,.1,.01))
        self.assertTrue(update_selection(s,.105,.01))
        self.assertEqual(s["bad_epochs"],1)
        self.assertFalse(update_selection(s,.09,.01))
        self.assertEqual(s["bad_epochs"],2)
        self.assertTrue(update_selection(s,.12,.01))
        self.assertEqual(s["bad_epochs"],0)
    def test_early_stop_after_minimum_epochs(self):
        from v7_experiment_train import fresh_state
        with tempfile.TemporaryDirectory() as td:
            cfg={**SETTINGS,"epochs":10,"patience":2,"min_epochs":3}
            fixed={"rank_ic_5d":.1,"rank_ic_10d":.08,"rank_ic_5d_dates":2,"rank_ic_10d_dates":2}
            with patch("v7_experiment_train.aggregate_validation_metrics",return_value=fixed):
                result=run_experiment(normalize_spec(resolve_spec(0,{})),cfg,ToyDataset(3),ToyDataset(2),
                    td,"fixture",device=torch.device("cpu"),ssd_fn=None,model_factory=ToyModel,emit=lambda *a,**k:None)
            self.assertEqual(result["epochs"],3)
            self.assertEqual(result["stop_reason"],"early_stopping")
            self.assertEqual(result["step"],9)
    def test_matrix_missing_or_changed(self):
        from v7_experiment_storage import digest
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(ValueError):verify_matrix(root)
            names=["features_59.parquet","feature_metadata.json","splits.json",
                   "knowledge_graph_v2_csr.npz","market_prices_raw.parquet","data_health.json"]
            for name in names:(root/name).write_bytes(b"fixture")
            (root/"feature_metadata.json").write_text(json.dumps({"artifact_kind":"full-prepared-candidate",
                "parquet_sha256":digest(root/"features_59.parquet")}))
            marker={"files":{name:digest(root/name) for name in names},"result":{"rows":3}}
            (root/".prepare-complete.json").write_text(json.dumps(marker))
            self.assertEqual(verify_matrix(root)["result"]["rows"],3)
            (root/"features_59.parquet").write_bytes(b"partial")
            with self.assertRaises(ValueError):verify_matrix(root)
    def test_adaptive_ladder(self):
        completed={}
        for i,name in enumerate(STAGES):
            spec=resolve_spec(i,completed)
            normalize_spec(spec)
            completed[name]={"status":"complete","best_rank_ic_5d":.1+i*.01,"config":spec["config"]}
        self.assertEqual(completed[STAGES[2]]["config"]["temporal_layers"],3)
        self.assertEqual(completed[STAGES[2]]["config"]["forward_layers"],2)
        self.assertEqual(completed[STAGES[5]]["config"]["d_model"],128)
        self.assertEqual(completed[STAGES[5]]["config"]["d_state"],32)
if __name__=="__main__":unittest.main()
