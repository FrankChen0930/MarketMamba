import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from v7_experiment_model import ExperimentConfig
from v7_experiment_storage import CheckpointStore
from v7_stability_export import export_checkpoint_predictions


class ExportDataset:
    def __init__(self):
        self.samples = [
            {
                "Date": "2024-01-02",
                "stock_ids": ("2317", "2330"),
                "x": torch.tensor([[[1.0]], [[2.0]]]),
                "labels": torch.tensor([[0.2, float("nan")], [0.1, 0.3]]),
            },
            {
                "Date": "2024-01-03",
                "stock_ids": ("2317", "2330"),
                "x": torch.tensor([[[3.0]], [[4.0]]]),
                "labels": torch.tensor([[float("nan"), float("nan")], [0.4, 0.5]]),
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ExportModel(torch.nn.Module):
    def __init__(self, config, ssd_fn=None):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward_prepared(self, sample):
        base = sample["x"][:, -1, 0]
        return torch.stack((base, -base), dim=1) + self.bias


class StabilityExportTests(unittest.TestCase):
    def test_export_resumes_by_verified_date_chunk_and_skips_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CheckpointStore(root / "checkpoint")
            model = ExportModel(None)
            store.save({
                "identity": "checkpoint-id",
                "model_config": asdict(ExperimentConfig()),
                "model": model.state_dict(),
            }, best=True)
            kwargs = dict(
                checkpoint_dir=root / "checkpoint",
                expected_checkpoint_identity="checkpoint-id",
                dataset=ExportDataset(),
                output=root / "export",
                model_id="E5-seed17",
                data_id="matrix-sha",
                seed=17,
                device=torch.device("cpu"),
                ssd_fn=None,
                model_factory=ExportModel,
            )
            paused = export_checkpoint_predictions(**kwargs, max_dates=1)
            self.assertEqual(paused["status"], "paused")
            complete = export_checkpoint_predictions(**kwargs)
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["dates"], 2)
            skipped = export_checkpoint_predictions(**kwargs)
            self.assertEqual(skipped["status"], "skipped")
            manifest = json.loads((root / "export" / "manifest.json").read_text())
            self.assertEqual(len(manifest["chunks"]), 2)
            import pandas as pd
            rows = pd.concat([pd.read_parquet(root / "export" / item["file"])
                              for item in manifest["chunks"]]).to_dict("records")
            tail = next(row for row in rows if row["date"] == "2024-01-03" and row["stock_id"] == "2317")
            self.assertTrue(pd.isna(tail["label_5d"]))
            self.assertFalse(tail["label_5d_mature"])

    def test_export_rejects_corrupt_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CheckpointStore(root / "checkpoint")
            model = ExportModel(None)
            entry = store.save({
                "identity": "checkpoint-id",
                "model_config": asdict(ExperimentConfig()),
                "model": model.state_dict(),
            }, best=True)
            (root / "checkpoint" / entry["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(OSError, "checkpoint"):
                export_checkpoint_predictions(
                    checkpoint_dir=root / "checkpoint",
                    expected_checkpoint_identity="checkpoint-id",
                    dataset=ExportDataset(),
                    output=root / "export",
                    model_id="E5-seed17",
                    data_id="matrix-sha",
                    seed=17,
                    device=torch.device("cpu"),
                    ssd_fn=None,
                    model_factory=ExportModel,
                )


if __name__ == "__main__":
    unittest.main()
