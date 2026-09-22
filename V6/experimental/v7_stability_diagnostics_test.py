import unittest

from v7_stability_diagnostics import build_diagnostics


class StabilityDiagnosticsTests(unittest.TestCase):
    def test_same_sample_ensemble_reports_time_groups_and_stability(self):
        rows = []
        scores = {
            17: {"2317": 3.0, "2330": 2.0, "2454": 1.0},
            29: {"2317": 3.0, "2330": 2.0, "2454": 1.0},
            43: {"2317": 1.0, "2330": 2.0, "2454": 3.0},
        }
        labels = {"2317": 3.0, "2330": 2.0, "2454": 1.0}
        for day in ("2024-01-02", "2024-04-02"):
            for seed, by_stock in scores.items():
                for stock, score in by_stock.items():
                    rows.append({
                        "date": day, "stock_id": stock, "seed": seed,
                        "score_5d": score, "score_10d": score,
                        "valid_5d": True, "valid_10d": True,
                        "label_5d": labels[stock], "label_10d": labels[stock],
                        "label_5d_mature": True, "label_10d_mature": True,
                        "model_id": f"E5-seed{seed}", "data_id": "matrix-sha",
                    })
        report = build_diagnostics(
            rows,
            expected_seeds=(17, 29, 43),
            regimes={"2024-01-02": "up", "2024-04-02": "down"},
            top_ns=(2,),
        )
        daily = report["daily"]
        first = next(row for row in daily if row["date"] == "2024-01-02" and row["horizon"] == 5)
        self.assertEqual(first["common_valid"], 3)
        self.assertAlmostEqual(first["rank_ic"]["ensemble"], 1.0)
        self.assertAlmostEqual(first["rank_ic"]["seed43"], -1.0)
        self.assertEqual(first["top_stability"]["top2"]["pairwise_mean_overlap"], 2 / 3)
        groups = {(row["period_type"], row["period"]) for row in report["periods"]}
        self.assertTrue({("year", "2024"), ("quarter", "2024Q1"),
                         ("quarter", "2024Q2"), ("regime", "up"),
                         ("regime", "down")} <= groups)

    def test_unmatured_tail_keeps_prediction_stability_without_ic(self):
        rows = []
        for seed in (17, 29, 43):
            for stock, score in (("2317", 2.0), ("2330", 1.0)):
                rows.append({
                    "date": "2026-09-11", "stock_id": stock, "seed": seed,
                    "score_5d": score, "score_10d": score,
                    "valid_5d": True, "valid_10d": True,
                    "label_5d": None, "label_10d": None,
                    "label_5d_mature": False, "label_10d_mature": False,
                    "model_id": f"E5-seed{seed}", "data_id": "matrix-sha",
                })
        report = build_diagnostics(rows, expected_seeds=(17, 29, 43), top_ns=(2,))
        row = next(item for item in report["daily"] if item["horizon"] == 5)
        self.assertEqual(row["common_valid"], 2)
        self.assertEqual(row["mature_common"], 0)
        self.assertIsNone(row["rank_ic"]["ensemble"])
        self.assertEqual(row["top_stability"]["top2"]["pairwise_mean_overlap"], 1.0)

    def test_missing_seed_is_a_hard_failure(self):
        row = {
            "date": "2024-01-02", "stock_id": "2330", "seed": 17,
            "score_5d": 1.0, "score_10d": 1.0,
            "valid_5d": True, "valid_10d": True,
            "label_5d": 1.0, "label_10d": 1.0,
            "label_5d_mature": True, "label_10d_mature": True,
            "model_id": "E5-seed17", "data_id": "matrix-sha",
        }
        with self.assertRaisesRegex(ValueError, "missing seeds"):
            build_diagnostics([row], expected_seeds=(17, 29, 43))


if __name__ == "__main__":
    unittest.main()
