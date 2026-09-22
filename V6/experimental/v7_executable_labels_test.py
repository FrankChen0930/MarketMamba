from __future__ import annotations

import math
import unittest

from V6.experimental.v7_executable_labels import build_executable_labels


CALENDAR = [
    "2026-01-02",  # Fri
    "2026-01-05",  # Mon
    "2026-01-06",
    "2026-01-08",  # Jan 7 holiday in this explicit calendar
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
]


def prices(stock_id: str = "2330") -> list[dict]:
    rows = []
    for i, day in enumerate(CALENDAR):
        rows.append(
            {
                "Date": day,
                "stock_id": stock_id,
                "Open": 100.0 + i,
                "Close": 101.0 + i,
                "observation_valid": True,
                "open_executable": True,
                "suspended": False,
            }
        )
    return rows


class ExecutableLabelTest(unittest.TestCase):
    def test_horizon_counts_entry_session_as_one(self) -> None:
        output = build_executable_labels(prices(), CALENDAR, horizons=(5,))
        row = next(item for item in output if item["signal_date"] == "2026-01-02")
        self.assertEqual(row["entry_session_5d"], "2026-01-05")
        self.assertEqual(row["exit_session_5d"], "2026-01-12")
        self.assertAlmostEqual(row["Alpha_5d"], 106.0 / 101.0 - 1)

    def test_weekend_and_holiday_follow_explicit_calendar(self) -> None:
        output = build_executable_labels(prices(), CALENDAR, horizons=(2,))
        row = next(item for item in output if item["signal_date"] == "2026-01-06")
        self.assertEqual(row["entry_session_2d"], "2026-01-08")
        self.assertEqual(row["exit_session_2d"], "2026-01-09")

    def test_missing_or_locked_entry_is_unknown_not_rolled(self) -> None:
        rows = prices()
        rows[1]["open_executable"] = False
        output = build_executable_labels(rows, CALENDAR, horizons=(5,))
        row = next(item for item in output if item["signal_date"] == "2026-01-02")
        self.assertTrue(math.isnan(row["Alpha_5d"]))
        self.assertEqual(row["entry_session_5d"], "2026-01-05")
        self.assertEqual(row["label_status_5d"], "ENTRY_NOT_EXECUTABLE")

    def test_suspension_inside_holding_interval_invalidates_label(self) -> None:
        rows = prices()
        rows[3]["suspended"] = True
        output = build_executable_labels(rows, CALENDAR, horizons=(5,))
        row = next(item for item in output if item["signal_date"] == "2026-01-02")
        self.assertTrue(math.isnan(row["Alpha_5d"]))
        self.assertEqual(row["label_status_5d"], "INVALID_HOLDING_INTERVAL")

    def test_missing_intermediate_observation_invalidates_label(self) -> None:
        rows = [row for row in prices() if row["Date"] != "2026-01-08"]
        output = build_executable_labels(rows, CALENDAR, horizons=(5,))
        row = next(item for item in output if item["signal_date"] == "2026-01-02")
        self.assertTrue(math.isnan(row["Alpha_5d"]))
        self.assertEqual(row["label_status_5d"], "INVALID_HOLDING_INTERVAL")

    def test_tail_has_explicit_insufficient_future_status(self) -> None:
        output = build_executable_labels(prices(), CALENDAR, horizons=(5,))
        row = next(item for item in output if item["signal_date"] == "2026-01-06")
        self.assertTrue(math.isnan(row["Alpha_5d"]))
        self.assertEqual(row["label_status_5d"], "INSUFFICIENT_FUTURE_SESSIONS")

    def test_input_order_does_not_change_output(self) -> None:
        rows = prices("2330") + prices("1101")
        forward = build_executable_labels(rows, CALENDAR, horizons=(2, 5))
        reverse = build_executable_labels(list(reversed(rows)), CALENDAR, horizons=(2, 5))
        self.assertEqual(forward, reverse)

    def test_manifest_names_exact_formula_and_invalidation(self) -> None:
        output, manifest = build_executable_labels(prices(), CALENDAR, horizons=(5, 10), return_manifest=True)
        self.assertTrue(output)
        self.assertEqual(manifest["label_formula"], "exit_close / entry_open - 1")
        self.assertEqual(manifest["holding_session_counting"], "entry session is 1; exit is holding session h")
        self.assertIn("legacy Close[t+h]/Close[t] labels", manifest["invalidates"])


if __name__ == "__main__":
    unittest.main()
