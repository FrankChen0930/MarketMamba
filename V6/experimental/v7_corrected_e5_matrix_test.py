from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config
from V6.experimental.v7_corrected_e5_matrix import (
    build_arrays,
    validate_upstream_evidence,
    write_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
CFG = CorrectedE5Config.from_contract(
    ROOT / "research/v7/corrected-e5-replication-v1/incumbent-contract.json",
    ROOT / "research/v7/corrected-baseline-v1/feature-manifest.json",
)


class CorrectedE5MatrixTest(unittest.TestCase):
    def frames(self):
        rows = []
        for day, stocks in (("2023-11-17", ("2330", "1101")),
                            ("2024-01-02", ("2330", "1101"))):
            for offset, stock in enumerate(stocks):
                row = {"Date": day, "stock_id": stock}
                row.update({name: float(i + offset) for i, name in enumerate(CFG.feature_order)})
                rows.append(row)
        features = pd.DataFrame(rows[::-1])
        universe = pd.DataFrame([
            {"session": row["Date"], "stock_id": row["stock_id"],
             "membership": "ELIGIBLE"} for row in rows
        ])
        labels = pd.DataFrame([
            {"signal_date": row["Date"], "stock_id": row["stock_id"],
             "Alpha_5d": .1, "Alpha_10d": .2,
             "label_status_5d": "VALID", "label_status_10d": "VALID"}
            for row in rows
        ])
        return features, universe, labels

    def test_upstream_evidence_must_keep_stop_pass_separation(self):
        validate_upstream_evidence(
            {"classes": {
                "EXCHANGE_REGULAR_BOARD_VERIFIED": {"eligible": True},
                "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR": {"eligible": True},
                "UNKNOWN": {"eligible": False},
            }},
            {"evidence_class": "HISTORICAL_SIMULATION_PROXY",
             "selected_proxy_policy": "P0",
             "strict_phase0_decision": "STOP",
             "historical_simulated_valid_labels": {"5d": 6889229, "10d": 6790554}},
        )
        with self.assertRaises(ValueError):
            validate_upstream_evidence(
                {"classes": {"UNKNOWN": {"eligible": True}}},
                {"evidence_class": "HISTORICAL_SIMULATION_PROXY",
                 "selected_proxy_policy": "P0", "strict_phase0_decision": "STOP",
                 "historical_simulated_valid_labels": {"5d": 6889229, "10d": 6790554}},
            )

    def test_arrays_are_date_stock_sorted_and_dual_headed(self):
        features, universe, labels = self.frames()
        arrays = build_arrays(features, universe, labels, CFG)
        self.assertEqual((4, 48), arrays["X"].shape)
        self.assertEqual(["1101", "2330", "1101", "2330"],
                         arrays["stock_ids"].tolist())
        self.assertEqual(["train", "train", "research_evaluation",
                          "research_evaluation"], arrays["split_names"].tolist())
        self.assertTrue(np.allclose(arrays["y5"], .1))
        self.assertTrue(np.allclose(arrays["y10"], .2))
        self.assertTrue(arrays["masks"].all())
        self.assertEqual([2, 4], arrays["end_indices"].tolist())

    def test_universe_join_fails_closed(self):
        features, universe, labels = self.frames()
        universe = universe.iloc[:-1]
        arrays = build_arrays(features, universe, labels, CFG)
        self.assertEqual(3, len(arrays["dates"]))

    def test_label_status_controls_mask(self):
        features, universe, labels = self.frames()
        labels.loc[0, "label_status_5d"] = "INVALID"
        arrays = build_arrays(features, universe, labels, CFG)
        match = ((arrays["dates"].astype(str) == "2023-11-17")
                 & (arrays["stock_ids"] == "2330"))
        self.assertTrue(np.isnan(arrays["y5"][match]).all())
        self.assertTrue(np.isfinite(arrays["y10"][match]).all())

    def test_write_is_deterministic_and_manifest_complete(self):
        features, universe, labels = self.frames()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = write_matrix(
                build_arrays(features, universe, labels, CFG), Path(first), CFG,
                source_hashes={name: name[0] * 64 for name in
                               ("features", "labels", "universe", "provenance")},
                build_mode="full_memory",
            )
            two = write_matrix(
                build_arrays(features, universe, labels, CFG), Path(second), CFG,
                source_hashes={name: name[0] * 64 for name in
                               ("features", "labels", "universe", "provenance")},
                build_mode="full_memory",
            )
            self.assertEqual(one["logical_identity"], two["logical_identity"])
            self.assertEqual(set(one["files"]), {
                "X.npy", "y5.npy", "y10.npy", "stock_ids.npy", "dates.npy",
                "end_indices.npy", "splits.npy", "masks.npy",
            })


if __name__ == "__main__":
    unittest.main()
