"""Parser output -> parquet -> merger-style date rename -> feature_engineer / hygiene.

OLD-parsed (legacy snapshot) vs NEW-parsed frames must behave identically downstream.
Offline; real feature_engineer/hygiene functions are loaded by path with a stubbed
`marketmamba.config` / `marketmamba.data.feature_spec` package context (like test_official_parsers).
"""
import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

HERE = Path(__file__).resolve().parent
FX = HERE / "fixtures" / "foreign_shareholding"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, spec


legacy, _ = _load("fs_integ_legacy", FX / "legacy_fetcher_fs_snapshot.py")
newp, _ = _load("fs_integ_newparser", HERE / "parsers" / "foreign_shareholding.py")
hygiene, _ = _load("fs_integ_hygiene", HERE / "hygiene.py")


def load_feature_engineer(tmpdir):
    cfg = ModuleType("marketmamba.config")
    cfg.FEATURE_COLS, cfg.FEATURE_GROUPS, cfg.SEQ_LEN = [], {}, 60
    cfg.PROCESSED_DIR = Path(tmpdir)
    pkg, dpkg = ModuleType("marketmamba"), ModuleType("marketmamba.data")
    pkg.__path__, dpkg.__path__ = [], []
    spec_fs = importlib.util.spec_from_file_location("marketmamba.data.feature_spec", HERE / "feature_spec.py")
    fspec = importlib.util.module_from_spec(spec_fs)
    mods = {"marketmamba": pkg, "marketmamba.data": dpkg, "marketmamba.config": cfg,
            "marketmamba.data.feature_spec": fspec}
    with patch.dict(sys.modules, mods):
        spec_fs.loader.exec_module(fspec)
        fe, _ = _load("fs_integ_feature_engineer", HERE / "feature_engineer.py")
    return fe


def payload(name):
    return json.loads((FX / f"{name}.json").read_text(encoding="utf-8"))


def old_new_frames(name="normal"):
    """Return (old_df, new_df) = TWSE+TPEX combined exactly as _direct does, old vs new parser."""
    f = payload(name)
    d = f["date"]
    old = [legacy_parse(f, d, "twse"), legacy_parse(f, d, "tpex")]
    new = [newp.parse_twse_qfiis(f["twse"], d), newp.parse_tpex_qfiis(f["tpex"], d)]
    comb = lambda parts: (pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
                          .drop_duplicates(subset=["date", "stock_id"], keep="first"))
    return comb(old), comb(new)


def legacy_parse(f, d, key):
    from types import SimpleNamespace
    fn = (legacy.fetch_foreign_shareholding_twse_direct if key == "twse"
          else legacy.fetch_foreign_shareholding_tpex_direct)
    resp = SimpleNamespace(status_code=200, json=lambda: f[key], raise_for_status=lambda: None)
    with patch.object(legacy.requests, "get", lambda *a, **k: resp):
        return fn(d)


def price_frame(fs, extra_days=2):
    """Daily price grid (Date, stock_id) around the fixture date incl. a day with no FS row."""
    d0 = pd.Timestamp(fs["date"].iloc[0])
    dates = [d0 - pd.Timedelta(days=1), d0, d0 + pd.Timedelta(days=extra_days)]
    ids = sorted(fs["stock_id"].unique())[:6]
    return pd.DataFrame([(d, s) for s in ids for d in dates], columns=["Date", "stock_id"])


def merger_load_equivalent(path):
    """merger.load_raw_data._load(): read parquet, date -> Date, to_datetime."""
    df = pd.read_parquet(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.fe = load_feature_engineer(cls.tmp.name)

    def _roundtrip(self, df, name):
        p = Path(self.tmp.name) / f"{name}.parquet"
        df.to_parquet(p)
        return p, merger_load_equivalent(p)

    def test_old_and_new_parsed_frames_equal_through_parquet_and_merge(self):
        for case in ("normal", "tpex_only", "none_cells", "short_rows", "swapped_ratio", "duplicate_id"):
            with self.subTest(case=case):
                old, new = old_new_frames(case)
                assert_frame_equal(old, new, check_exact=True)
                _, old_l = self._roundtrip(old, f"old_{case}")
                _, new_l = self._roundtrip(new, f"new_{case}")
                assert_frame_equal(old_l, new_l, check_exact=True)
                base = price_frame(old)
                for flags in (False, True):
                    o = self.fe._merge_foreign_shareholding(base.copy(), old_l, availability_flags=flags)
                    n = self.fe._merge_foreign_shareholding(base.copy(), new_l, availability_flags=flags)
                    assert_frame_equal(o, n, check_exact=True)

    def test_merge_without_parser_output(self):
        base = price_frame(old_new_frames()[1])
        for fs in (None, pd.DataFrame()):
            out = self.fe._merge_foreign_shareholding(base.copy(), fs, availability_flags=True)
            self.assertTrue((out["Foreign_Holding_Pct"] == 0.0).all())
            self.assertTrue((out["Avail_ForeignShare"] == 0.0).all())

    def test_picks_shares_ratio_not_remain_ratio(self):
        _, new = old_new_frames("normal")
        _, loaded = self._roundtrip(new, "pick")
        base = price_frame(new)
        out = self.fe._merge_foreign_shareholding(base.copy(), loaded)
        d0 = pd.Timestamp(payload("normal")["date"])
        got = out[out["Date"] == d0].set_index("stock_id")["Foreign_Holding_Pct"]
        exp = new.set_index("stock_id")["ForeignInvestmentSharesRatio"]
        rem = new.set_index("stock_id")["ForeignInvestmentRemainRatio"]
        for sid in got.index:
            self.assertAlmostEqual(got[sid], exp[sid])
            if exp[sid] != rem[sid]:
                self.assertNotAlmostEqual(got[sid], rem[sid])
        self.assertTrue((got.index.isin(exp.index)).all())

    def test_pit_date_alignment_no_leak_and_ffill(self):
        """Value appears on the parsed date (not before), is forward-filled after, 0.0 before."""
        _, new = old_new_frames("normal")
        _, loaded = self._roundtrip(new, "pit")
        base = price_frame(new)
        out = self.fe._merge_foreign_shareholding(base.copy(), loaded, availability_flags=True)
        d0 = pd.Timestamp(payload("normal")["date"])
        for sid, g in out.groupby("stock_id"):
            g = g.set_index("Date")
            self.assertEqual(g.loc[d0 - pd.Timedelta(days=1), "Foreign_Holding_Pct"], 0.0)
            self.assertEqual(g.loc[d0 - pd.Timedelta(days=1), "Avail_ForeignShare"], 0.0)
            v = g.loc[d0, "Foreign_Holding_Pct"]
            self.assertEqual(g.loc[d0 + pd.Timedelta(days=2), "Foreign_Holding_Pct"], v)
            self.assertEqual(g.loc[d0, "Avail_ForeignShare"], 1.0)

    def test_lowercase_date_column_renamed_like_merger(self):
        _, new = old_new_frames("normal")
        self.assertIn("date", new.columns)
        p, loaded = self._roundtrip(new, "rename")
        self.assertIn("Date", loaded.columns)
        self.assertNotIn("date", loaded.columns)
        self.assertEqual(len(loaded), len(new))

    def test_hygiene_staleness_same_for_old_and_new(self):
        old, new = old_new_frames("normal")
        d0 = pd.Timestamp(payload("normal")["date"])
        results = []
        for tag, fs in (("old", old), ("new", new)):
            root = Path(self.tmp.name) / f"hyg_{tag}"
            root.mkdir()
            ids = ["1101", "1102"]
            pr = pd.DataFrame([(d, s, 10.0) for d in pd.bdate_range(d0 - pd.Timedelta(days=30), d0 + pd.Timedelta(days=15))
                               for s in ids], columns=["Date", "stock_id", "Close"])
            pr["Date"] = pr["Date"].dt.strftime("%Y-%m-%d")
            pr.to_parquet(root / "prices_raw.parquet")
            fs.to_parquet(root / "foreign_shareholding_raw.parquet")
            with self.assertLogs(hygiene.logger, level="INFO"):
                res = hygiene.check_data_health(root)
            results.append(res)
        self.assertEqual(results[0]["stats"], results[1]["stats"])
        self.assertEqual(results[0]["warnings"], results[1]["warnings"])
        lag = results[1]["stats"]["staleness_days"]["foreign_shareholding_raw"]
        ref = pd.Timestamp(results[1]["stats"]["reference_date"])
        self.assertEqual(lag, (ref - d0).days)
        self.assertEqual(hygiene.STALENESS_BUDGET_DAYS["foreign_shareholding_raw"], 10)


if __name__ == "__main__":
    unittest.main()
