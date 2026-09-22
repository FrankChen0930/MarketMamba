import json
from pathlib import Path
import unittest

import numpy as np

from V6.experimental.v7_corrected_e5_replication import (
    ALLOWED_SOURCE_CLASSES,
    CorrectedE5Config,
    assign_split,
    canonical_hash,
    canonical_stock_order,
    estimate_peak_ram,
    join_historical_labels,
    load_canonical_features,
    make_matrix_manifest,
    validate_matrix_manifest,
    validate_source_classes,
)


ROOT = Path(__file__).resolve().parents[2]
FEATURE_MANIFEST = ROOT / "research/v7/corrected-baseline-v1/feature-manifest.json"
CONTRACT = ROOT / "research/v7/corrected-e5-replication-v1/incumbent-contract.json"


class CorrectedE5ContractTest(unittest.TestCase):
    def test_canonical_features_are_loaded_not_duplicated(self):
        features = load_canonical_features(FEATURE_MANIFEST)
        self.assertEqual(48, len(features))
        self.assertEqual("Open", features[0])
        self.assertEqual("FED_Rate", features[-1])
        self.assertEqual(len(features), len(set(features)))

    def test_forbidden_features_absent(self):
        cfg = CorrectedE5Config.from_contract(CONTRACT, FEATURE_MANIFEST)
        forbidden = {"PER", "PBR", "Revenue_MoM", "Revenue_YoY", "EPS",
                     "EPS_Surprise", "Gross_Margin", "ROE", "Book_Value",
                     "Dividend_Yield_Fwd", "Free_Cash_Flow"}
        self.assertTrue(forbidden.isdisjoint(cfg.feature_order))

    def test_incumbent_architecture_and_corrected_gates(self):
        cfg = CorrectedE5Config.from_contract(CONTRACT, FEATURE_MANIFEST)
        self.assertEqual((15, 20, 1, 12), cfg.group_dims)
        self.assertEqual((64, 32, 60), (cfg.d_model, cfg.d_state, cfg.sequence_length))
        self.assertEqual((5, 10), cfg.horizons)
        self.assertEqual((17, 29, 43), cfg.seeds)
        self.assertEqual("fp32", cfg.precision)
        self.assertFalse(cfg.graph_enabled)
        self.assertFalse(cfg.industry_neutralization)
        self.assertFalse(cfg.current_industry_backfill)

    def test_source_gate_is_exact_and_fail_closed(self):
        self.assertEqual({
            "EXCHANGE_REGULAR_BOARD_VERIFIED",
            "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR",
        }, ALLOWED_SOURCE_CLASSES)
        validate_source_classes(list(ALLOWED_SOURCE_CLASSES))
        for value in ("UNKNOWN", "PROVIDER_MIXED_UNKNOWN_SEGMENT",
                      "SYNTHETIC/BACKFILLED", "ADJUSTED_PRICE_ONLY", ""):
            with self.assertRaises(ValueError):
                validate_source_classes([value])

    def test_ordering_is_lexical_and_duplicates_fail(self):
        self.assertEqual((0, 2, 1), canonical_stock_order(["0050", "2330", "1101"]))
        with self.assertRaises(ValueError):
            canonical_stock_order(["2330", "2330"])

    def test_split_reconstructs_research_used_window(self):
        self.assertEqual("warmup", assign_split("2013-01-01"))
        self.assertEqual("train", assign_split("2013-01-02"))
        self.assertEqual("train", assign_split("2023-11-17"))
        self.assertEqual("purge", assign_split("2023-11-20"))
        self.assertEqual("purge", assign_split("2023-12-29"))
        self.assertEqual("research_evaluation", assign_split("2024-01-02"))
        self.assertEqual("research_evaluation", assign_split("2026-09-11"))
        self.assertEqual("out_of_contract", assign_split("2026-09-12"))

    def test_label_join_is_exact_and_does_not_roll(self):
        features = [
            {"Date": "2024-01-02", "stock_id": "2330", "f": 1.0},
            {"Date": "2024-01-03", "stock_id": "2330", "f": 2.0},
        ]
        labels = [
            {"Date": "2024-01-02", "stock_id": "2330", "Alpha_5d": .1, "Alpha_10d": .2}
        ]
        joined = join_historical_labels(features, labels)
        self.assertEqual(.1, joined[0]["Alpha_5d"])
        self.assertTrue(np.isnan(joined[1]["Alpha_5d"]))

    def test_duplicate_label_key_fails(self):
        with self.assertRaises(ValueError):
            join_historical_labels(
                [{"Date": "2024-01-02", "stock_id": "2330"}],
                [{"Date": "2024-01-02", "stock_id": "2330"},
                 {"Date": "2024-01-02", "stock_id": "2330"}],
            )

    def test_hash_is_deterministic_and_list_order_sensitive(self):
        self.assertEqual(canonical_hash({"b": 1, "a": [2, 3]}),
                         canonical_hash({"a": [2, 3], "b": 1}))
        self.assertNotEqual(canonical_hash({"a": [2, 3]}),
                            canonical_hash({"a": [3, 2]}))

    def test_ram_estimate_exposes_components(self):
        estimate = estimate_peak_ram(rows=100, features=48, id_bytes=16)
        self.assertEqual({"raw_input", "dataframe_working", "final_arrays",
                          "metadata", "estimated_peak"}, set(estimate))
        self.assertGreater(estimate["estimated_peak"], estimate["final_arrays"])

    def test_manifest_completeness_and_no_leakage(self):
        cfg = CorrectedE5Config.from_contract(CONTRACT, FEATURE_MANIFEST)
        manifest = make_matrix_manifest(
            cfg,
            rows=3,
            files={"X.npy": "a" * 64, "y5.npy": "b" * 64,
                   "y10.npy": "c" * 64, "stock_ids.npy": "d" * 64,
                   "dates.npy": "e" * 64, "end_indices.npy": "f" * 64,
                   "splits.npy": "1" * 64, "masks.npy": "2" * 64},
            source_hashes={"features": "3" * 64, "labels": "4" * 64,
                           "universe": "5" * 64, "provenance": "6" * 64},
            build_mode="full_memory",
        )
        validate_matrix_manifest(manifest, cfg)
        self.assertEqual("research-used historical evaluation",
                         manifest["split_contract"]["evaluation_semantics"])
        self.assertEqual(10, manifest["split_contract"]["label_horizon_trading_days"])
        self.assertEqual(20, manifest["split_contract"]["embargo_trading_days"])
        self.assertEqual(30, manifest["split_contract"]["total_purge_trading_days"])
        altered = json.loads(json.dumps(manifest))
        altered["preprocessing"]["industry_neutralization"] = True
        with self.assertRaises(ValueError):
            validate_matrix_manifest(altered, cfg)

    def test_manifest_rejects_unrecorded_chunk_fallback(self):
        cfg = CorrectedE5Config.from_contract(CONTRACT, FEATURE_MANIFEST)
        manifest = make_matrix_manifest(
            cfg, rows=1,
            files={name: "a" * 64 for name in (
                "X.npy", "y5.npy", "y10.npy", "stock_ids.npy", "dates.npy",
                "end_indices.npy", "splits.npy", "masks.npy")},
            source_hashes={name: "b" * 64 for name in (
                "features", "labels", "universe", "provenance")},
            build_mode="chunked_after_oom",
        )
        with self.assertRaises(ValueError):
            validate_matrix_manifest(manifest, cfg)


if __name__ == "__main__":
    unittest.main()
