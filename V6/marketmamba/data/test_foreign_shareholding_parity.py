"""OLD vs NEW parity for foreign-shareholding parsing.

OLD = verbatim baseline (258ee09) snapshot in fixtures/foreign_shareholding/legacy_fetcher_fs_snapshot.py.
NEW = parsers/foreign_shareholding.py and the refactored fetcher (delegating). Offline; mocked HTTP.
Asserts frame equality (values + dtypes + columns + index) and identical log messages.
"""
import importlib.util
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

HERE = Path(__file__).resolve().parent
FX = HERE / "fixtures" / "foreign_shareholding"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("fs_parity_gen", FX / "generate_golden.py")
legacy = _load("fs_parity_legacy", FX / "legacy_fetcher_fs_snapshot.py")
newp = _load("fs_parity_newparser", HERE / "parsers" / "foreign_shareholding.py")


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(f"{rec.levelname}: {rec.getMessage()}")


def call(mod, fn_name, fx, *args):
    """Run mod.<fn_name>(date) with mocked requests.get; return (result, logs, http calls)."""
    pl = {"twse": fx.get("twse"), "tpex": fx.get("tpex")}
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers, timeout))
        p = pl["twse" if "twse" in url else "tpex"]
        if p is None or p == "ERROR":
            raise ConnectionError("fixture: endpoint error")
        return SimpleNamespace(status_code=200, json=lambda: p, raise_for_status=lambda: None)

    cap = _Cap()
    mod.logger.addHandler(cap)
    old = mod.logger.level
    mod.logger.setLevel(logging.DEBUG)
    try:
        with patch.object(mod.requests, "get", get):
            res = getattr(mod, fn_name)(fx["date"])
    finally:
        mod.logger.removeHandler(cap)
        mod.logger.setLevel(old)
    return res, cap.msgs, calls


def same_frame(tc, a, b, msg=""):
    if a is None or b is None:
        tc.assertIs(a, b, msg)
        return
    assert_frame_equal(a, b, check_dtype=True, check_exact=True)
    tc.assertEqual(list(a.dtypes.astype(str)), list(b.dtypes.astype(str)), msg)


class ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetcher = gen.load_fetcher()
        cls.addClassCleanup(cls.fetcher._tmp.cleanup)
        cls.fixtures = [__import__("json").loads(
            p.read_text(encoding="utf-8")) for p in gen.fixture_paths()]
        cls.names = [p.stem for p in gen.fixture_paths()]

    def test_fetcher_vs_legacy_all_fixtures(self):
        self.assertGreaterEqual(len(self.fixtures), 10)
        for name, fx in zip(self.names, self.fixtures):
            for fn in ("fetch_foreign_shareholding_twse_direct",
                       "fetch_foreign_shareholding_tpex_direct",
                       "fetch_foreign_shareholding_direct"):
                with self.subTest(case=name, fn=fn):
                    old, old_logs, old_calls = call(legacy, fn, fx)
                    new, new_logs, new_calls = call(self.fetcher, fn, fx)
                    same_frame(self, old, new)
                    self.assertEqual(old_logs, new_logs)
                    self.assertEqual(old_calls, new_calls)

    def test_new_module_vs_legacy_parse_level(self):
        """Pure module on the raw payload equals the old fetch function's result (HTTP excluded)."""
        for name, fx in zip(self.names, self.fixtures):
            for key, fn_old, fn_new in (("twse", "fetch_foreign_shareholding_twse_direct", newp.parse_twse_qfiis),
                                        ("tpex", "fetch_foreign_shareholding_tpex_direct", newp.parse_tpex_qfiis)):
                payload = fx.get(key)
                if payload in (None, "ERROR"):
                    continue
                with self.subTest(case=name, src=key):
                    old, old_logs, _ = call(legacy, fn_old, fx)
                    cap = _Cap()
                    lg = logging.getLogger(f"fs_parity_{name}_{key}")
                    lg.setLevel(logging.DEBUG)
                    lg.addHandler(cap)
                    new = fn_new(payload, fx["date"], lg)
                    same_frame(self, old, new)
                    self.assertEqual(old_logs, cap.msgs)

    def test_helpers_vs_legacy(self):
        self.assertEqual(legacy.FS_COLS, newp.FS_COLS)
        self.assertEqual(legacy.FS_COLS, self.fetcher.FS_COLS)
        for v in gen.NUM_INPUTS:
            a, b, c = legacy._fs_num(v), newp.fs_num(v), self.fetcher._fs_num(v)
            for x in (b, c):
                self.assertEqual(type(a), type(x), repr(v))
                self.assertTrue(a == x or (a != a and x != x), repr(v))

    def test_identity_check_vs_legacy(self):
        cases = [gen._identity_df(12, 10, 100, 10.0), gen._identity_df(12, 10, 100, 90.0),
                 gen._identity_df(3, 10, 100, 10.0), gen._identity_df(12, 10, 0, 10.0),
                 gen._identity_df(12, 10, 100, 10.05), gen._identity_df(12, "10", "100", "10.0")]
        for i, df in enumerate(cases):
            res = []
            for mod, fn in ((legacy, legacy._fs_identity_check), (self.fetcher, self.fetcher._fs_identity_check)):
                cap = _Cap()
                mod.logger.addHandler(cap)
                old = mod.logger.level
                mod.logger.setLevel(logging.DEBUG)
                try:
                    r = fn(df, f"TAG {i}")
                finally:
                    mod.logger.removeHandler(cap)
                    mod.logger.setLevel(old)
                res.append((r, cap.msgs))
            self.assertEqual(res[0], res[1], f"case {i}")


if __name__ == "__main__":
    unittest.main()
