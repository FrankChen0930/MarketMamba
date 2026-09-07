"""Regression tests for the lightweight V6.2 fetch-only scheduler path."""

from __future__ import annotations

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_v62_daily.py"
SPEC = importlib.util.spec_from_file_location("run_v62_daily_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FetchOnlyTest(unittest.TestCase):
    def _complete_sources(self):
        return {name: True for name in MODULE.DAILY_SOURCES}

    def _run_fetch_only(self, fetch_result):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"run_v62_inference", "v62_portfolio"}:
                self.fail(f"fetch-only path imported heavy pipeline module: {name}")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(sys, "argv", [str(SCRIPT), "--fetch-only"]),
            patch.object(MODULE, "fetch_data", return_value=fetch_result) as fetch,
            patch.object(MODULE, "notify") as notify,
            patch.object(builtins, "__import__", side_effect=guarded_import),
        ):
            result = MODULE.main()
        return result, fetch, notify

    def test_fetch_only_stops_before_model_pipeline_imports(self) -> None:
        result, fetch, notify = self._run_fetch_only((self._complete_sources(), []))

        self.assertEqual(result, 0)
        fetch.assert_called_once_with()
        notify.assert_not_called()

    def test_fetch_only_returns_failure_when_daily_sources_are_missing(self) -> None:
        missing = ["margin_raw（落後 1 天）"]
        result, fetch, notify = self._run_fetch_only(({"margin_raw": False}, missing))

        self.assertEqual(result, 1)
        fetch.assert_called_once_with()
        notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
