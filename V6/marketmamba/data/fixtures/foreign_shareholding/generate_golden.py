"""Capture golden outputs of the foreign-shareholding fetchers (offline, mocked HTTP).

Run:  python generate_golden.py            (rewrites golden/*.json from the CURRENT fetcher)
Also imported by test_foreign_shareholding_golden.py (load_fetcher / run_case / run_unit).
Golden files MUST be generated from the pre-refactor fetcher; do not regenerate after refactor.
"""
import importlib.util
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
FETCHER = HERE.parents[1] / "fetcher.py"


def load_fetcher():
    tmp = tempfile.TemporaryDirectory()
    config = ModuleType("marketmamba.config")
    for name in ("DATA_DIR", "PROCESSED_DIR"):
        setattr(config, name, Path(tmp.name) / name)
    for name in ("DATA_END_DATE", "DATA_SOURCE_PRIORITY", "DATA_START_DATE", "FINMIND_TOKEN",
                 "MARGIN_FORWARD_FILL", "TPEX_INSTITUTIONAL_URL", "TWSE_INSTITUTIONAL_URL"):
        setattr(config, name, None)
    spec = importlib.util.spec_from_file_location("golden_fs_fetcher", FETCHER)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"marketmamba.config": config, "yfinance": SimpleNamespace(),
                                    "requests": SimpleNamespace(get=Mock(side_effect=AssertionError("network disabled")))}):
        spec.loader.exec_module(mod)
    mod._tmp = tmp
    return mod


def _clean(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


def ser(df):
    if df is None:
        return None
    return {"columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "records": [{c: _clean(v) for c, v in r.items()} for r in df.to_dict("records")]}


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(f"{rec.levelname}: {rec.getMessage()}")


def _run(fetcher, fn, payload_by_host):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        host = "twse" if "twse" in url else "tpex"
        calls.append({"url": url, "params": params})
        p = payload_by_host.get(host)
        if p is None or p == "ERROR":
            raise ConnectionError("fixture: endpoint error")
        return SimpleNamespace(status_code=200, json=lambda: p, raise_for_status=lambda: None)

    cap = _Cap()
    fetcher.logger.addHandler(cap)
    old = fetcher.logger.level
    fetcher.logger.setLevel(logging.DEBUG)
    try:
        with patch.object(fetcher.requests, "get", get):
            df = fn()
    finally:
        fetcher.logger.removeHandler(cap)
        fetcher.logger.setLevel(old)
    return {"result": ser(df), "logs": cap.msgs, "calls": calls}


def run_case(fetcher, fx):
    d, pl = fx["date"], {"twse": fx.get("twse"), "tpex": fx.get("tpex")}
    return {"twse": _run(fetcher, lambda: fetcher.fetch_foreign_shareholding_twse_direct(d), pl),
            "tpex": _run(fetcher, lambda: fetcher.fetch_foreign_shareholding_tpex_direct(d), pl),
            "direct": _run(fetcher, lambda: fetcher.fetch_foreign_shareholding_direct(d), pl)}


NUM_INPUTS = ["25,219,056", "87.86%", " 12.5 % ", "--", "", None, "abc", "1e3", "-5", "-", 0, 5, 3.5,
              float("nan"), "nan", "inf", "1,2,3", "NaN%", True, "１２３"]


def _identity_df(n, shares, issued, ratio):
    return pd.DataFrame({"ForeignInvestmentShares": [shares] * n, "NumberOfSharesIssued": [issued] * n,
                         "ForeignInvestmentSharesRatio": [ratio] * n})


def run_unit(fetcher):
    nums = []
    for v in NUM_INPUTS:
        r = fetcher._fs_num(v)
        nums.append({"input": repr(v), "output": _clean(r) if r == r else "NaN", "type": type(r).__name__})
    ident = {}
    cases = {"pass_12": _identity_df(12, 10, 100, 10.0), "low_12": _identity_df(12, 10, 100, 90.0),
             "too_few_3": _identity_df(3, 10, 100, 10.0), "zero_issued_12": _identity_df(12, 10, 0, 10.0),
             "none_cells_12": pd.DataFrame({"ForeignInvestmentShares": [None] * 12,
                                            "NumberOfSharesIssued": [100.0] * 12,
                                            "ForeignInvestmentSharesRatio": [10.0] * 12}),
             "boundary_0.05_12": _identity_df(12, 10, 100, 10.05),
             "string_numeric_12": _identity_df(12, "10", "100", "10.0")}
    for k, df in cases.items():
        cap = _Cap()
        fetcher.logger.addHandler(cap)
        old = fetcher.logger.level
        fetcher.logger.setLevel(logging.DEBUG)
        try:
            ret = fetcher._fs_identity_check(df, f"TAG {k}")
        finally:
            fetcher.logger.removeHandler(cap)
            fetcher.logger.setLevel(old)
        ident[k] = {"returns": ret, "logs": cap.msgs}
    return {"FS_COLS": list(fetcher.FS_COLS), "fs_num": nums, "identity_check": ident}


def fixture_paths():
    return sorted(p for p in HERE.glob("*.json"))


def main():
    f = load_fetcher()
    GOLDEN.mkdir(exist_ok=True)
    for p in fixture_paths():
        fx = json.loads(p.read_text(encoding="utf-8"))
        (GOLDEN / p.name).write_text(json.dumps(run_case(f, fx), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (GOLDEN / "_unit.json").write_text(json.dumps(run_unit(f), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("golden written:", len(list(GOLDEN.glob("*.json"))))


if __name__ == "__main__":
    main()
