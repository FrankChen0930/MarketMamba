from pathlib import Path
import json
import tempfile
import unittest

import torch
from torch import nn

from V6.experimental.v7_corrected_e5_lifecycle import (
    CheckpointStore,
    early_stop_eligible,
    fresh_lifecycle_state,
    stage_namespace,
    update_selection,
)
from V6.experimental.v7_corrected_e5_train import lr_factor


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/v7/corrected-e5-replication-v1/training-lifecycle-audit.json"


def payload(model, optimizer, scheduler, *, step, epoch=0):
    return {
        "format": "marketmamba-v7-corrected-e5-v2",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "batch": step,
        "step": step,
        "phase": "train",
        "validation_index": 0,
        "validation_rows": [],
        "best_rank_ic_5d": None,
        "best_epoch": None,
        "patience_reference": None,
        "bad_epochs": 0,
        "history": [],
        "rng_state": {},
        "terminal": False,
        "stop_reason": None,
        "contract_identity": "contract-a",
        "matrix_identity": "matrix-a",
    }


class TrainingLifecycleContractTest(unittest.TestCase):
    def test_audit_freezes_incumbent_warmup_and_early_stop(self):
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertTrue(audit["warmup"]["warmup_enabled"])
        self.assertEqual(.15, audit["warmup"]["warmup_fraction"])
        self.assertEqual(.04, audit["warmup"]["starting_lr_factor"])
        self.assertEqual("per successful optimizer update",
                         audit["warmup"]["cadence"])
        self.assertEqual(5, audit["early_stopping"]["patience"])
        self.assertEqual(0, audit["early_stopping"]["minimum_delta"])
        self.assertEqual(5, audit["early_stopping"]["first_eligible_epoch"])

    def test_selection_tracks_best_epoch_and_independent_patience_reference(self):
        state = fresh_lifecycle_state("contract-a", "matrix-a")
        self.assertTrue(update_selection(state, .10, epoch=1, min_delta=0.))
        self.assertEqual((.10, 1, .10, 0), (
            state["best_rank_ic_5d"], state["best_epoch"],
            state["patience_reference"], state["bad_epochs"]))
        self.assertFalse(update_selection(state, .10, epoch=2, min_delta=0.))
        self.assertEqual(1, state["bad_epochs"])
        self.assertTrue(update_selection(state, .11, epoch=3, min_delta=0.))
        self.assertEqual((.11, 3, 0), (
            state["best_rank_ic_5d"], state["best_epoch"], state["bad_epochs"]))

    def test_early_stop_gate_matches_incumbent(self):
        self.assertFalse(early_stop_eligible(
            epoch=4, bad_epochs=5, epochs=20, warmup_fraction=.15,
            minimum_epochs=5, patience=5))
        self.assertTrue(early_stop_eligible(
            epoch=5, bad_epochs=5, epochs=20, warmup_fraction=.15,
            minimum_epochs=5, patience=5))

    def test_seed_namespaces_are_independent(self):
        root = Path("/drive/output")
        self.assertNotEqual(stage_namespace(root, "B-seed17", 17),
                            stage_namespace(root, "C-seed29", 29))
        self.assertIn("seed-29", str(stage_namespace(root, "C-seed29", 29)))

    def test_checkpoint_recovers_previous_generation_when_latest_is_corrupt(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(Path(folder))
            store.save(payload(model, optimizer, scheduler, step=1))
            store.save(payload(model, optimizer, scheduler, step=2))
            manifest = store.manifest()
            (Path(folder) / manifest["latest"]["file"]).write_bytes(b"corrupt")
            loaded = store.load("contract-a", "matrix-a")
            self.assertEqual(1, loaded["step"])

    def test_best_checkpoint_is_independent_from_last_and_corruption_fails(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(Path(folder))
            store.save(payload(model, optimizer, scheduler, step=2, epoch=2), best=True)
            store.save(payload(model, optimizer, scheduler, step=3, epoch=3))
            self.assertEqual(2, store.load("contract-a", "matrix-a", best=True)["epoch"])
            manifest = store.manifest()
            (Path(folder) / manifest["best"]["file"]).write_bytes(b"corrupt")
            with self.assertRaises(OSError):
                store.load("contract-a", "matrix-a", best=True)

    def test_scheduler_and_parameters_match_after_resume(self):
        inputs = [
            (torch.tensor([[float(i)]]), torch.tensor([[float(i * 2 + 1)]]))
            for i in range(8)
        ]

        def setup():
            torch.manual_seed(123)
            model = nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: lr_factor(step, len(inputs), .15))
            return model, optimizer, scheduler

        def advance(model, optimizer, scheduler, start, stop, lrs):
            for index in range(start, stop):
                x, y = inputs[index]
                lrs.append(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(x), y).backward()
                optimizer.step()
                scheduler.step()

        continuous = setup()
        continuous_lrs = []
        advance(*continuous, 0, 8, continuous_lrs)

        resumed = setup()
        resumed_lrs = []
        advance(*resumed, 0, 3, resumed_lrs)
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(Path(folder))
            store.save(payload(*resumed, step=3))
            restarted = setup()
            saved = store.load("contract-a", "matrix-a")
            restarted[0].load_state_dict(saved["model_state"])
            restarted[1].load_state_dict(saved["optimizer_state"])
            restarted[2].load_state_dict(saved["scheduler_state"])
            advance(*restarted, 3, 8, resumed_lrs)

        self.assertEqual(continuous_lrs, resumed_lrs)
        for expected, observed in zip(continuous[0].parameters(), restarted[0].parameters()):
            self.assertTrue(torch.equal(expected, observed))


    def test_early_stop_state_survives_resume(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        state = payload(model, optimizer, scheduler, step=7, epoch=4)
        state.update({
            "best_rank_ic_5d": .12,
            "best_epoch": 2,
            "patience_reference": .12,
            "bad_epochs": 2,
            "history": [{"epoch": 4, "bad_epochs": 2}],
        })
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(Path(folder))
            store.save(state, best=True)
            loaded = store.load("contract-a", "matrix-a")
        self.assertEqual((.12, 2, 2), (
            loaded["best_rank_ic_5d"], loaded["best_epoch"],
            loaded["bad_epochs"],
        ))


if __name__ == "__main__":
    unittest.main()
