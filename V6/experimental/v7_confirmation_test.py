"""Regression for confirmation scheduling, selection, provenance and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from v7_confirmation_design import read_reference, comparison, resolve_spec, validate_reference, STAGES, SEEDS
from v7_experiment_storage import fingerprint, digest
from v7_experiment_suite import STAGES as OLD_STAGES
from v7_experiment_test import ToyModel
import v7_confirmation_suite as suite
from v7_experiment_train import run_experiment

class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.ref=read_reference()

    def completed(self):
        return {name:copy.deepcopy(self.ref["results"]["E3-width64" if name.startswith("E3") else "E5-state32"]) for name in STAGES[:4]}

    def test_paired_selection_and_depth(self):
        done=self.completed()
        result=comparison(done,self.ref)
        self.assertEqual(result["selected_d_state"],32)
        self.assertEqual(result["models"]["E3"]["seeds"],[17,29,43])
        self.assertEqual(SEEDS,(29,29,43,43,17))
        cfg=resolve_spec(4,done,self.ref)["config"]
        self.assertEqual((cfg["temporal_layers"],cfg["forward_layers"],cfg["reverse_layers"],cfg["d_model"],cfg["d_state"]),(3,1,1,64,32))

    def test_one_outlier_win_not_enough(self):
        done=self.completed()
        def score(row,value):
            for h in row["history"]:h["validation"]["rank_ic_5d"]=value
            row["best_rank_ic_5d"]=value
        # Positive mean alone is insufficient when only one seed wins.
        score(self.ref["results"]["E5-state32"],.9)
        for name in ("E5-seed29","E5-seed43"):score(done[name],.1)
        decision=comparison(done,self.ref)
        self.assertGreater(sum(decision["paired_ic5_differences"]),0)
        self.assertEqual(decision["selected_d_state"],8)
        # Two wins alone are insufficient if mean improvement is negative.
        score(self.ref["results"]["E5-state32"],-1.)
        for name in ("E5-seed29","E5-seed43"):score(done[name],.12)
        decision=comparison(done,self.ref)
        self.assertEqual(sum(x>0 for x in decision["paired_ic5_differences"]),2)
        self.assertEqual(decision["selected_d_state"],8)

    def test_missing_failed_and_coverage_do_not_select(self):
        done=self.completed();del done["E5-seed43"]
        with self.assertRaises(KeyError):comparison(done,self.ref)
        done=self.completed();done["E5-seed43"]["status"]="failed"
        with self.assertRaises(ValueError):comparison(done,self.ref)
        done=self.completed()
        for h in done["E5-seed43"]["history"]:h["validation"]["rank_ic_5d_dates"]=1
        with self.assertRaises(ValueError):comparison(done,self.ref)

    def test_original_sources_unchanged(self):
        for name,sha in self.ref["original_sources"].items():
            self.assertEqual(digest(Path(__file__).parent/name),sha,name)

    def test_reference_exact_hash_and_python_patch_recovery(self):
        ref=copy.deepcopy(self.ref)
        matrix={"features":"known-hash"};splits={"train":["old"],"validation":["new"]}
        runtime={"python":"3.12.12","torch":"fixed"}
        ref["suite_identity"]=fingerprint({"matrix":matrix,"splits":splits,"settings":ref["settings"],
            "sources":ref["original_sources"],"runtime":runtime,"stages":OLD_STAGES,"diagnostic":False})
        proof=validate_reference(ref,matrix,splits,{**runtime,"python":"3.12.13"},Path(__file__).parent)
        self.assertEqual(proof["original_runtime"],runtime)
        with self.assertRaises(ValueError):
            validate_reference(ref,{"features":"changed"},splits,runtime,Path(__file__).parent)

    def test_suite_pause_resume_skip_and_settings_rejection(self):
        data=Path(__file__).resolve().parents[2]/".artifacts/v7-local-matrix/prepared-smoke"
        if not data.exists():self.skipTest("local smoke matrix unavailable")
        with tempfile.TemporaryDirectory() as td:
            args=suite.parser().parse_args(["--data",str(data),"--output",td,"--device","cpu",
                "--diagnostic","--epochs","1","--pause-after-steps","1"])
            def toy(*a,**kw):
                return run_experiment(*a,**kw,model_factory=ToyModel)
            with patch.object(suite,"official_ssd_callable",return_value=None),patch.object(suite,"run_experiment",side_effect=toy):
                suite.run_suite(args)
                self.assertEqual(json.loads((Path(td)/"suite.json").read_text())["status"],"paused")
                args.pause_after_steps=None
                suite.run_suite(args)
                state=json.loads((Path(td)/"suite.json").read_text())
                self.assertEqual(state["status"],"complete")
                self.assertEqual(list(state["experiments"]),sorted(STAGES)) # atomic_json sorts keys
                self.assertEqual(len(state["results"]),5)
                for name,seed in zip(STAGES,SEEDS):
                    from v7_experiment_storage import CheckpointStore
                    ck=CheckpointStore(Path(td)/name/"checkpoints").load(state["results"][name]["identity"])
                    self.assertEqual(ck["training_settings"]["seed"],seed)
                suite.run_suite(args)
                events=[json.loads(x) for x in (Path(td)/"events.jsonl").read_text().splitlines()]
                self.assertEqual(sum(x["event"]=="experiment_skipped_completed" for x in events),5)
                args.epochs=2
                with self.assertRaises(ValueError):suite.run_suite(args)

if __name__=="__main__":unittest.main()
