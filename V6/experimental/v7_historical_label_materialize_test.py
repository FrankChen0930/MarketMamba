from __future__ import annotations

import unittest

import pandas as pd

from V6.experimental.v7_historical_label_materialize import build_valid_label_frame


class HistoricalLabelMaterializeTest(unittest.TestCase):
    def test_vectorized_labels_keep_frozen_next_session_contract(self) -> None:
        calendar = [f"2022-01-{day:02d}" for day in range(3, 11)]
        rows = [
            {
                "Date": day,
                "stock_id": "2330",
                "Open": 100.0 + index,
                "Close": 101.0 + index,
                "observation_valid": True,
                "open_executable": True,
                "suspended": False,
            }
            for index, day in enumerate(calendar)
        ]
        result = build_valid_label_frame(pd.DataFrame(rows), calendar, horizons=(5,))
        first = result.loc[result["signal_date"].eq(calendar[0])].iloc[0]
        self.assertEqual(first["entry_session_5d"], calendar[1])
        self.assertEqual(first["exit_session_5d"], calendar[5])
        self.assertAlmostEqual(first["Alpha_5d"], (101.0 + 5) / (100.0 + 1) - 1.0)

    def test_missing_holding_observation_cannot_silently_roll(self) -> None:
        calendar = [f"2022-01-{day:02d}" for day in range(3, 11)]
        rows = [
            {
                "Date": day,
                "stock_id": "2330",
                "Open": 100.0,
                "Close": 101.0,
                "observation_valid": True,
                "open_executable": True,
                "suspended": False,
            }
            for day in calendar
            if day != calendar[3]
        ]
        result = build_valid_label_frame(pd.DataFrame(rows), calendar, horizons=(5,))
        self.assertNotIn(calendar[0], set(result["signal_date"]))

    def test_suspension_invalidates_holding_interval(self) -> None:
        calendar = [f"2022-01-{day:02d}" for day in range(3, 11)]
        rows = [
            {
                "Date": day,
                "stock_id": "2330",
                "Open": 100.0,
                "Close": 101.0,
                "observation_valid": True,
                "open_executable": True,
                "suspended": index == 2,
            }
            for index, day in enumerate(calendar)
        ]
        result = build_valid_label_frame(pd.DataFrame(rows), calendar, horizons=(5,))
        self.assertNotIn(calendar[0], set(result["signal_date"]))


if __name__ == "__main__":
    unittest.main()
