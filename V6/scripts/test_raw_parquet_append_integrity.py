"""Regression tests for raw parquet append integrity guardrails.

These tests exercise ``marketmamba.data.fetcher._append_to_parquet`` with
temporary parquet files. They are not data-repair checks: the contract under
test is that append writes remain clean when the input batch contains duplicate
keys, a retry returns a smaller same-day universe, or a prior file contains a
stray pandas index column.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import pandas as pd
    import pyarrow  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - depends on local V6 env
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[2]
FETCHER_PATH = REPO_ROOT / "V6" / "marketmamba" / "data" / "fetcher.py"


def _load_fetcher_module():
    """Load fetcher.py with a minimal config module and no package side effects."""

    data_dir = REPO_ROOT / "V6" / "Data"
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))

    config = types.ModuleType("marketmamba.config")
    config.DATA_DIR = data_dir
    config.DATA_END_DATE = "2026-12-31"
    config.DATA_SOURCE_PRIORITY = {}
    config.DATA_START_DATE = "2012-01-01"
    config.FINMIND_TOKEN = ""
    config.MARGIN_FORWARD_FILL = True
    config.PROCESSED_DIR = data_dir / "processed_v6"
    config.TPEX_INSTITUTIONAL_URL = ""
    config.TWSE_INSTITUTIONAL_URL = ""

    package = types.ModuleType("marketmamba")
    sys.modules.setdefault("marketmamba", package)
    sys.modules["marketmamba.config"] = config

    spec = importlib.util.spec_from_file_location("fetcher_under_test", FETCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(pd is None, "pandas/pyarrow V6 data dependencies are not installed")
class RawParquetAppendIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fetcher = _load_fetcher_module()

    def test_duplicate_same_day_rows_are_removed_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices_raw.parquet"
            batch = pd.DataFrame(
                [
                    {"Date": "2026-08-28", "stock_id": "2330", "Close": 900.0},
                    {"Date": "2026-08-28", "stock_id": "2330", "Close": 901.0},
                    {"Date": "2026-08-28", "stock_id": "2317", "Close": 200.0},
                ]
            )

            self.fetcher._append_to_parquet(path, batch, "2026-08-28")

            out = pd.read_parquet(path)
            duplicated = out.duplicated(subset=["Date", "stock_id"]).sum()
            self.assertEqual(duplicated, 0)
            self.assertEqual(len(out), 2)
            close_2330 = out.loc[out["stock_id"].astype(str) == "2330", "Close"].item()
            self.assertEqual(close_2330, 901.0)

    def test_same_day_shrink_preserves_existing_missing_stock_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices_raw.parquet"
            original = pd.DataFrame(
                [
                    {"Date": "2026-08-28", "stock_id": "2330", "Close": 900.0},
                    {"Date": "2026-08-28", "stock_id": "2317", "Close": 200.0},
                    {"Date": "2026-08-27", "stock_id": "2330", "Close": 890.0},
                ]
            )
            original.to_parquet(path, index=False)
            smaller_retry = pd.DataFrame(
                [{"Date": "2026-08-28", "stock_id": "2330", "Close": 901.0}]
            )

            self.fetcher._append_to_parquet(path, smaller_retry, "2026-08-28")

            out = pd.read_parquet(path)
            today = out[pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d") == "2026-08-28"]
            self.assertEqual(set(today["stock_id"].astype(str)), {"2330", "2317"})
            close_2317 = today.loc[today["stock_id"].astype(str) == "2317", "Close"].item()
            close_2330 = today.loc[today["stock_id"].astype(str) == "2330", "Close"].item()
            self.assertEqual(close_2317, 200.0)
            self.assertEqual(close_2330, 901.0)

    def test_stray_pandas_index_columns_are_removed_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices_raw.parquet"
            old = pd.DataFrame(
                [
                    {
                        "Date": "2026-08-28",
                        "stock_id": "2330",
                        "Close": 900.0,
                        "__index_level_0__": 7,
                    }
                ]
            )
            old.to_parquet(path, index=False)
            batch = pd.DataFrame(
                [{"Date": "2026-08-28", "stock_id": "2330", "Close": 901.0}]
            )

            self.fetcher._append_to_parquet(path, batch, "2026-08-28")

            out = pd.read_parquet(path)
            self.assertNotIn("__index_level_0__", out.columns)
            self.assertEqual(out.loc[out["stock_id"].astype(str) == "2330", "Close"].item(), 901.0)


if __name__ == "__main__":
    unittest.main()
