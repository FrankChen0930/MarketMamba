import math
import unittest

from v7_stability_contract import ensemble_percentile_ranks, validate_prediction_rows
from v7_stability_windows import build_window_splits


class StabilityContractTests(unittest.TestCase):
    def test_ensemble_uses_common_valid_sample_and_stable_ties(self):
        rows = [
            {"date": "2024-01-02", "stock_id": "2330", "seed": 17, "score_5d": 3.0, "score_10d": 1.0},
            {"date": "2024-01-02", "stock_id": "2317", "seed": 17, "score_5d": 1.0, "score_10d": 1.0},
            {"date": "2024-01-02", "stock_id": "2454", "seed": 17, "score_5d": 2.0, "score_10d": 1.0},
            {"date": "2024-01-02", "stock_id": "2330", "seed": 29, "score_5d": 1.0, "score_10d": 2.0},
            {"date": "2024-01-02", "stock_id": "2317", "seed": 29, "score_5d": 3.0, "score_10d": 2.0},
            {"date": "2024-01-02", "stock_id": "2454", "seed": 29, "score_5d": math.nan, "score_10d": 2.0},
        ]
        result = ensemble_percentile_ranks(rows, expected_seeds=(17, 29))
        by_stock = {row["stock_id"]: row for row in result}
        self.assertIsNone(by_stock["2454"]["ensemble_score_5d"])
        self.assertEqual(by_stock["2317"]["rank_5d"], 1)
        self.assertEqual(by_stock["2330"]["rank_5d"], 2)
        self.assertEqual([row["rank_10d"] for row in result], [1, 2, 3])

    def test_prediction_contract_keeps_missing_labels_distinct_from_zero(self):
        rows = [{
            "date": "2026-09-11", "stock_id": "2330", "score_5d": 0.25,
            "score_10d": -0.1, "valid_5d": True, "valid_10d": True,
            "label_5d": None, "label_10d": None,
            "label_5d_mature": False, "label_10d_mature": False,
            "model_id": "E5-seed17", "data_id": "matrix-sha",
        }]
        validated = validate_prediction_rows(rows)
        self.assertIsNone(validated[0]["label_5d"])
        self.assertFalse(validated[0]["label_5d_mature"])
        parquet_row = [dict(rows[0], label_5d=float("nan"))]
        self.assertIsNone(validate_prediction_rows(parquet_row)[0]["label_5d"])
        bad = [dict(rows[0], label_5d=0.0, label_5d_mature=False)]
        with self.assertRaisesRegex(ValueError, "immature label"):
            validate_prediction_rows(bad)


class StabilityWindowTests(unittest.TestCase):
    def test_three_windows_purge_training_tail_and_never_select_on_test(self):
        calendar = [f"{year}-{month:02d}-{day:02d}" for year in range(2013, 2027)
                    for month in range(1, 13) for day in (1, 8, 15)]
        windows = build_window_splits(calendar, purge_sessions=30)
        self.assertEqual([window["id"] for window in windows], ["W2024", "W2025", "W2026"])
        for window in windows:
            self.assertTrue(window["train"])
            self.assertTrue(window["selection"])
            self.assertTrue(window["evaluation"])
            self.assertLess(max(window["train"]), min(window["selection"]))
            self.assertLess(max(window["selection"]), min(window["evaluation"]))
            self.assertEqual(window["selection_year"], window["evaluation_year"] - 1)
            nominal_tail = [d for d in calendar if d.startswith(str(window["train_end_year"]))]
            self.assertEqual(len(set(nominal_tail) - set(window["train"])), 30)
        short_calendar = [date for date in calendar if date < "2025-01-01"]
        only_2024 = build_window_splits(short_calendar, purge_sessions=30, window_ids=("W2024",))
        self.assertEqual([window["id"] for window in only_2024], ["W2024"])


if __name__ == "__main__":
    unittest.main()
