"""Golden (characterization) tests for foreign-shareholding fetching/parsing.

Runs the CURRENT fetcher's public functions against synthetic fixtures with mocked HTTP and
compares to golden JSON captured from the pre-refactor code (fixtures/foreign_shareholding/golden).
Offline: no network, no config import. Must pass before and after the parser refactor.
"""
import importlib.util
import json
import unittest
from pathlib import Path

FX = Path(__file__).with_name("fixtures") / "foreign_shareholding"
_spec = importlib.util.spec_from_file_location("fs_golden_gen", FX / "generate_golden.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ForeignShareholdingGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetcher = gen.load_fetcher()
        cls.addClassCleanup(cls.fetcher._tmp.cleanup)

    def test_fixture_and_golden_sets_match(self):
        fx = {p.name for p in gen.fixture_paths()}
        golden = {p.name for p in (FX / "golden").glob("*.json")} - {"_unit.json"}
        self.assertEqual(fx, golden)
        self.assertGreaterEqual(len(fx), 10)

    def test_cases_match_golden(self):
        for p in gen.fixture_paths():
            with self.subTest(case=p.stem):
                actual = json.loads(json.dumps(gen.run_case(self.fetcher, _load(p)), ensure_ascii=False))
                expected = _load(FX / "golden" / p.name)
                for fn in ("twse", "tpex", "direct"):
                    self.assertEqual(actual[fn]["result"], expected[fn]["result"], f"{fn} result")
                    self.assertEqual(actual[fn]["logs"], expected[fn]["logs"], f"{fn} logs")
                    self.assertEqual(actual[fn]["calls"], expected[fn]["calls"], f"{fn} calls")

    def test_unit_helpers_match_golden(self):
        actual = json.loads(json.dumps(gen.run_unit(self.fetcher), ensure_ascii=False))
        expected = _load(FX / "golden" / "_unit.json")
        self.assertEqual(actual["FS_COLS"], expected["FS_COLS"])
        self.assertEqual(actual["fs_num"], expected["fs_num"])
        self.assertEqual(actual["identity_check"], expected["identity_check"])

    def test_documented_old_behavior_quirks(self):
        g = lambda n: _load(FX / "golden" / f"{n}.json")
        # 4-digit-numeric ids pass the filter (e.g. ETF '0050'); alphanumeric warrants are skipped
        ids = [r["stock_id"] for r in g("normal")["twse"]["result"]["records"]]
        self.assertIn("0050", ids)
        self.assertNotIn("03001P", ids)
        # '--' cells become None
        r1999 = [r for r in g("normal")["twse"]["result"]["records"] if r["stock_id"] == "1999"][0]
        self.assertIsNone(r1999["ForeignInvestmentShares"])
        # None name cell becomes the literal string 'None'
        r1102 = [r for r in g("none_cells")["twse"]["result"]["records"] if r["stock_id"] == "1102"][0]
        self.assertEqual(r1102["stock_name"], "None")
        # swapped ratios: only a WARNING, data still returned
        sw = g("swapped_ratio")["twse"]
        self.assertIsNotNone(sw["result"])
        self.assertTrue(any("通過率偏低" in m for m in sw["logs"]))
        # duplicate across markets: TWSE row wins in combined output
        dup = [r for r in g("duplicate_id")["direct"]["result"]["records"] if r["stock_id"] == "2330"]
        self.assertEqual([r["stock_name"] for r in dup], ["台積電"])
        # TPEX requires payload date without separators; ISO payload date is rejected
        self.assertIsNone(g("iso_date_in_payload")["tpex"]["result"])
        # TWSE stat != OK is silent (no log)
        self.assertEqual(g("stat_not_ok")["twse"]["logs"], [])


if __name__ == "__main__":
    unittest.main()
