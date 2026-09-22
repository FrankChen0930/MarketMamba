import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import v7_depth222_suite as suite
from v7_confirmation_design import read_reference
from v7_experiment_test import ToyModel
from v7_experiment_train import run_experiment
from v7_experiment_storage import CheckpointStore

class DepthTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_fixed_architecture_and_comparison(self):
        ref=read_reference();cfg=suite.resolve_spec(ref)["config"]
        self.assertEqual((cfg["temporal_layers"],cfg["forward_layers"],cfg["reverse_layers"],cfg["d_model"],cfg["d_state"]),(2,2,2,64,32))
        self.assertEqual(suite.SEEDS,(17,));self.assertEqual(len(suite.STAGES),1)
        report=suite.compare_result(ref["results"]["E5-state32"],ref)
        self.assertEqual(report["delta_ic5"],0);self.assertEqual(report["delta_ic10"],0)
        bad=copy.deepcopy(ref["results"]["E5-state32"])
        for h in bad["history"]:h["validation"]["rank_ic_5d_dates"]=1
        with self.assertRaises(ValueError):suite.compare_result(bad,ref)
        self.assertIsNone(suite.compare_result(bad,ref,True)["baseline_metrics"])

    def test_single_run_pause_resume_skip(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            args=suite.parser().parse_args(["--data",str(root/".artifacts/v7-local-matrix/prepared-smoke"),
                "--output",td,"--device","cpu","--diagnostic","--epochs","1","--pause-after-steps","1"])
            def toy(*a,**kw):return run_experiment(*a,**kw,model_factory=ToyModel)
            with patch.object(suite,"official_ssd_callable",return_value=None),patch.object(suite,"run_experiment",side_effect=toy):
                suite.run_suite(args)
                self.assertEqual(json.loads((Path(td)/"suite.json").read_text())["status"],"paused")
                args.pause_after_steps=None;suite.run_suite(args);suite.run_suite(args)
                state=json.loads((Path(td)/"suite.json").read_text())
                self.assertEqual(state["status"],"complete");self.assertEqual(len(state["results"]),1)
                result=state["results"][suite.STAGES[0]]
                ck=CheckpointStore(Path(td)/suite.STAGES[0]/"checkpoints").load(result["identity"])
                self.assertEqual(ck["training_settings"]["seed"],17)
                events=[json.loads(x) for x in (Path(td)/"events.jsonl").read_text().splitlines()]
                self.assertEqual(sum(e["event"]=="resumed" for e in events),1)
                self.assertEqual(sum(e["event"]=="experiment_skipped_completed" for e in events),1)
                self.assertFalse(any(e["event"].startswith("benchmark") for e in events))
                args.epochs=2
                with self.assertRaises(ValueError):suite.run_suite(args)

if __name__=="__main__":unittest.main()
