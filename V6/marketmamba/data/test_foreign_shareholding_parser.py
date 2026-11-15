"""Direct unit tests of parsers/foreign_shareholding.py on the recorded fixtures (no fetcher)."""
import importlib.util
import json
import logging
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FX = HERE / "fixtures" / "foreign_shareholding"
_spec = importlib.util.spec_from_file_location("fs_parser_under_test", HERE / "parsers" / "foreign_shareholding.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def fx(name):
    return json.loads((FX / f"{name}.json").read_text(encoding="utf-8"))


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.msgs = []

    def emit(self, rec):
        self.msgs.append((rec.levelname, rec.getMessage()))


def run(fn, payload, date):
    lg = logging.getLogger("fs_parser_test_" + fn.__name__)
    lg.setLevel(logging.DEBUG)
    cap = Cap()
    lg.addHandler(cap)
    try:
        return fn(payload, date, lg), cap.msgs
    finally:
        lg.removeHandler(cap)


class ParserNormalTest(unittest.TestCase):
    def test_twse_normal_schema_and_values(self):
        f = fx("normal")
        df, logs = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertEqual(list(df.columns), P.FS_COLS)
        self.assertTrue((df["date"] == pd.Timestamp(f["date"])).all())
        self.assertTrue(df["stock_id"].map(lambda s: len(s) == 4 and s.isdigit()).all())
        self.assertNotIn("03001P", set(df["stock_id"]))
        self.assertIn("0050", set(df["stock_id"]))          # ETF kept (old behavior)
        r = df.iloc[0]
        self.assertAlmostEqual(r["ForeignInvestmentSharesRatio"], 10.0)
        self.assertAlmostEqual(r["ForeignInvestmentRemainRatio"], 90.0)
        self.assertEqual(r["ForeignInvestmentShares"], 100_000_000.0)
        self.assertTrue(any(l == "INFO" and "恆等式" in m for l, m in logs))

    def test_tpex_normal(self):
        f = fx("normal")
        df, _ = run(P.parse_tpex_qfiis, f["tpex"], f["date"])
        self.assertEqual(list(df.columns), P.FS_COLS)
        self.assertTrue(df["ChineseInvestmentUpperLimitRatio"].isna().all())
        self.assertTrue(df["InternationalCode"].isna().all())
        self.assertTrue(df["RecentlyDeclareDate"].isna().all())

    def test_tables_fallback(self):
        f = fx("tables_fallback")
        df, _ = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)

    def test_default_logger_used_when_none_given(self):
        f = fx("date_mismatch")
        with self.assertLogs(P.logger, level="WARNING"):
            self.assertIsNone(P.parse_twse_qfiis(f["twse"], f["date"]))


class ParserAbnormalTest(unittest.TestCase):
    def test_silent_nones(self):
        for name, fn, key in (("stat_not_ok", P.parse_twse_qfiis, "twse"),
                              ("empty_data", P.parse_twse_qfiis, "twse")):
            f = fx(name)
            df, logs = run(fn, f[key], f["date"])
            self.assertIsNone(df, name)
            self.assertEqual(logs, [], name)

    def test_warned_nones(self):
        f = fx("date_mismatch")
        for fn, key in ((P.parse_twse_qfiis, "twse"), (P.parse_tpex_qfiis, "tpex")):
            df, logs = run(fn, f[key], f["date"])
            self.assertIsNone(df)
            self.assertEqual([l for l, _ in logs], ["WARNING"])
        f = fx("unrecognized_layout")
        df, logs = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertIsNone(df)
        self.assertIn("版面無法辨識", logs[0][1])

    def test_iso_date_payload_rejected(self):
        f = fx("iso_date_in_payload")
        for fn, key in ((P.parse_twse_qfiis, "twse"), (P.parse_tpex_qfiis, "tpex")):
            df, logs = run(fn, f[key], f["date"])
            self.assertIsNone(df)
            self.assertTrue(logs and logs[0][0] == "WARNING")

    def test_none_cells_and_short_rows(self):
        f = fx("none_cells")
        df, _ = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertIn("None", set(df["stock_name"].dropna()))       # None -> 'None' quirk
        f = fx("short_rows")
        df, _ = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertIsNotNone(df)

    def test_swapped_ratio_only_warns(self):
        f = fx("swapped_ratio")
        df, logs = run(P.parse_twse_qfiis, f["twse"], f["date"])
        self.assertIsNotNone(df)
        self.assertTrue(any(l == "WARNING" and "通過率偏低" in m for l, m in logs))

    def test_non_dict_free_minimal_payloads(self):
        self.assertIsNone(P.parse_twse_qfiis({}, "2026-07-16"))
        self.assertIsNone(P.parse_tpex_qfiis({"date": "20260716", "tables": []}, "2026-07-16"))


class HelperTest(unittest.TestCase):
    def test_fs_num(self):
        self.assertEqual(P.fs_num("25,219,056"), 25219056.0)
        self.assertEqual(P.fs_num("87.86%"), 87.86)
        for bad in (None, "", "--", "-", "abc", True):
            self.assertIsNone(P.fs_num(bad), repr(bad))
        self.assertTrue(P.fs_num("nan") != P.fs_num("nan"))
        self.assertEqual(P.fs_num("1,2,3"), 123.0)

    def test_identity_check_logs(self):
        def frame(n, ratio):
            return pd.DataFrame({"ForeignInvestmentShares": [10] * n, "NumberOfSharesIssued": [100] * n,
                                 "ForeignInvestmentSharesRatio": [ratio] * n})
        lg = logging.getLogger("fs_ident_test")
        lg.setLevel(logging.DEBUG)
        for n, ratio, expect in ((12, 10.0, ["INFO"]), (12, 90.0, ["INFO", "WARNING"]), (3, 10.0, ["WARNING"])):
            cap = Cap()
            lg.addHandler(cap)
            self.assertIsNone(P.identity_check(frame(n, ratio), "T", lg))
            lg.removeHandler(cap)
            self.assertEqual([l for l, _ in cap.msgs], expect)


if __name__ == "__main__":
    unittest.main()
