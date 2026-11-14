from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
from torch import nn

from V6.experimental.v7_corrected_e5_matrix import build_arrays, write_matrix
from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config
from V6.experimental.v7_corrected_e5_train import train_seed


ROOT = Path(__file__).resolve().parents[2]
CFG = CorrectedE5Config.from_contract(
    ROOT / "research/v7/corrected-e5-replication-v1/incumbent-contract.json",
    ROOT / "research/v7/corrected-baseline-v1/feature-manifest.json",
)


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(len(config.feature_order), 2)

    def forward(self, x, stock_ids, observation_mask=None):
        return self.head(x[:, -1])


class CorrectedE5TinyPipelineTest(unittest.TestCase):
    def test_tiny_cpu_mock_pipeline_saves_resumable_checkpoint(self):
        rows = []
        labels = []
        universe = []
        for day in ("2023-11-17", "2024-01-02"):
            for offset, stock in enumerate(("1101", "2330", "2603")):
                row = {"Date": day, "stock_id": stock}
                row.update({name: float(i + offset) for i, name in enumerate(CFG.feature_order)})
                rows.append(row)
                universe.append({"session": day, "stock_id": stock, "membership": "ELIGIBLE"})
                labels.append({
                    "signal_date": day, "stock_id": stock,
                    "Alpha_5d": float(offset), "Alpha_10d": float(2 - offset),
                    "label_status_5d": "VALID", "label_status_10d": "VALID",
                })
        arrays = build_arrays(
            pd.DataFrame(rows), pd.DataFrame(universe), pd.DataFrame(labels), CFG
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            matrix = root / "matrix"
            write_matrix(
                arrays, matrix, CFG,
                source_hashes={name: name[0] * 64 for name in
                               ("features", "labels", "universe", "provenance")},
                build_mode="full_memory",
            )
            result = train_seed(
                matrix, root / "run", CFG, seed=17, device=torch.device("cpu"),
                smoke_steps=2, model_factory=TinyModel,
            )
            self.assertEqual("smoke_complete", result["status"])
            self.assertTrue(result["checkpoint_reload"])
            self.assertIn("rank_ic_5d", result["metrics"])
            self.assertTrue((root / "run/checkpoints/checkpoint.json").is_file())


if __name__ == "__main__":
    unittest.main()
