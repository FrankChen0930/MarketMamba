"""Regression coverage for interrupted V7 feature preparation."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from v7_integrated_prepare_cache import StockCache

class PreparationCacheTests(unittest.TestCase):
    def test_interrupted_write_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            cache = StockCache(root, {"source": "one"})
            frame = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02"]), "x": [1.25]})
            cache.write("2330", frame, {"count": 1})
            data, marker = cache.paths("2317")
            data.write_bytes(b"incomplete parquet")
            resumed = StockCache(root, {"source": "one"})
            self.assertIsNotNone(resumed.valid("2330"))
            self.assertIsNone(resumed.valid("2317"))
            pd.testing.assert_frame_equal(pd.read_parquet(resumed.paths("2330")[0]), frame)

    def test_corruption_is_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            cache = StockCache(root, {})
            cache.write("2330", pd.DataFrame({"x": [1]}), {})
            cache.paths("2330")[0].write_bytes(b"damaged")
            self.assertIsNone(cache.valid("2330"))

    def test_changed_input_rejected_without_deleting_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            cache = StockCache(root, {"source": "one"})
            cache.write("2330", pd.DataFrame({"x": [1]}), {})
            with self.assertRaisesRegex(ValueError, "changed"):
                StockCache(root, {"source": "two"})
            self.assertIsNotNone(cache.valid("2330"))

if __name__ == "__main__":
    unittest.main()
