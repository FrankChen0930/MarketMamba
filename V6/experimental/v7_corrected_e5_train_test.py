from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from V6.experimental.v7_corrected_e5_model import CorrectedE5Model
from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config
from V6.experimental.v7_corrected_e5_train import (
    capture_rng_state,
    evaluate_daily,
    load_checkpoint,
    lr_factor,
    replication_stages,
    save_checkpoint,
    short_loss,
)


ROOT = Path(__file__).resolve().parents[2]
CFG = CorrectedE5Config.from_contract(
    ROOT / "research/v7/corrected-e5-replication-v1/incumbent-contract.json",
    ROOT / "research/v7/corrected-baseline-v1/feature-manifest.json",
)


class IdentityMixer(nn.Module):
    def forward(self, value):
        return value


def identity_factory(_config):
    return IdentityMixer()


class CorrectedE5ModelAndTrainTest(unittest.TestCase):
    def test_graph_free_model_has_dual_heads(self):
        model = CorrectedE5Model(CFG, mixer_factory=identity_factory)
        names = tuple(name for name, _ in model.named_modules())
        self.assertFalse(any("graph" in name.lower() or "gat" in name.lower() for name in names))
        self.assertFalse(hasattr(model, "fusion"))
        x = torch.randn(4, 60, 48)
        result = model(x, ["2330", "1101", "0050", "2603"])
        self.assertEqual((4, 2), tuple(result.shape))
        self.assertTrue(hasattr(model, "head_5d"))
        self.assertTrue(hasattr(model, "head_10d"))

    def test_sequence_length_is_enforced(self):
        model = CorrectedE5Model(CFG, mixer_factory=identity_factory)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 59, 48), ["1101", "2330"])

    def test_loss_masks_heads_and_weights_ic10_half(self):
        pred = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        labels = torch.tensor([[0.0, float("nan")], [1.0, .1], [2.0, .2]])
        both = short_loss(pred, labels)
        labels[:, 1] = float("nan")
        only5 = short_loss(pred, labels)
        self.assertIsNotNone(both)
        self.assertIsNotNone(only5)
        self.assertGreater(float(both), float(only5))

    def test_schedule_matches_warmup_and_cosine_end(self):
        total = 100
        self.assertAlmostEqual(.04, lr_factor(0, total, .15), places=8)
        self.assertAlmostEqual(1.0, lr_factor(14, total, .15), places=8)
        self.assertAlmostEqual(.0001, lr_factor(99, total, .15), places=8)

    def test_replication_stages_always_include_all_seeds(self):
        self.assertEqual(
            [("A-smoke", 17), ("B-seed17", 17), ("C-seed29", 29), ("C-seed43", 43)],
            replication_stages(),
        )

    def test_checkpoint_round_trip_and_mismatch_fail_closed(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=3, batch=7, step=11, best_rank_ic_5d=.12,
                bad_epochs=2, contract_identity="contract-a",
                matrix_identity="matrix-a", rng_state=capture_rng_state(),
            )
            loaded = load_checkpoint(
                path, expected_contract_identity="contract-a",
                expected_matrix_identity="matrix-a",
            )
            self.assertEqual((3, 7, 11), (
                loaded["epoch"], loaded["batch"], loaded["step"]))
            self.assertIn("optimizer_state", loaded)
            self.assertIn("scheduler_state", loaded)
            self.assertIn("rng_state", loaded)
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path, expected_contract_identity="contract-b",
                    expected_matrix_identity="matrix-a",
                )

    def test_ic10_is_from_same_ic5_selected_checkpoint(self):
        daily = [
            {
                "Date": "2024-01-02",
                "predictions": torch.tensor([[0., 2.], [1., 1.], [2., 0.]]),
                "labels": torch.tensor([[0., 0.], [1., 1.], [2., 2.]]),
            },
            {
                "Date": "2024-01-03",
                "predictions": torch.tensor([[2., 0.], [1., 1.], [0., 2.]]),
                "labels": torch.tensor([[2., 2.], [1., 1.], [0., 0.]]),
            },
        ]
        report = evaluate_daily(daily, checkpoint_identity="best-ic5-abc")
        self.assertEqual("best-ic5-abc", report["checkpoint_identity"])
        self.assertEqual("rank_ic_5d", report["selection_metric"])
        self.assertEqual("same_checkpoint_as_rank_ic_5d",
                         report["rank_ic_10d_semantics"])
        self.assertIn("2024", report["yearly"])
        self.assertEqual(6, report["common_support"]["rows"])


if __name__ == "__main__":
    unittest.main()
