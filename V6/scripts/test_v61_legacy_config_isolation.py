#!/usr/bin/env python3
"""
Regression coverage for V6.1 legacy daily inference feature-shape isolation.

The global config may serve the current 59D V6.2/dual contract. The legacy
V6.1 daily entrypoint must still run v6_best.pt with the original 56D feature
shape without mutating repository config files or weakening current serving.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
V6_ROOT = REPO_ROOT / "V6"
SCRIPT_PATH = V6_ROOT / "run_daily_inference.py"
RS_COLS = ("RS_5d", "RS_20d", "RS_60d")
LEGACY_GROUP_DIMS = {
    "price_momentum": 12,
    "institutional_flow": 20,
    "fundamentals": 12,
    "macro_environment": 12,
}
LEGACY_CHECKPOINT_EMBED_DIMS = {
    "price_momentum": 54,
    "institutional_flow": 94,
    "fundamentals": 54,
    "macro_environment": 54,
}


def _install_import_stubs() -> dict[str, types.ModuleType | None]:
    previous: dict[str, types.ModuleType | None] = {}

    def set_module(name: str, module: types.ModuleType) -> None:
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.where = lambda condition, x, y: x
    set_module("numpy", numpy_stub)

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = type("DataFrame", (), {})
    pandas_stub.Series = lambda *args, **kwargs: None
    pandas_stub.Timestamp = lambda value=None: value
    pandas_stub.Timedelta = lambda *args, **kwargs: None
    pandas_stub.to_datetime = lambda value, *args, **kwargs: value
    pandas_stub.to_numeric = lambda value, *args, **kwargs: value
    pandas_stub.read_parquet = lambda *args, **kwargs: None
    pandas_stub.read_csv = lambda *args, **kwargs: None
    set_module("pandas", pandas_stub)

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
        get_device_name=lambda idx: "stub-cuda",
    )
    torch_stub.serialization = types.SimpleNamespace(add_safe_globals=lambda values: None)
    torch_stub.device = lambda value: value
    torch_stub.load = lambda *args, **kwargs: {"state_dict": {}, "epoch": 0, "val_loss": 0.0}
    torch_stub.manual_seed = lambda seed: None
    torch_stub.zeros = lambda *args, **kwargs: None
    torch_stub.cat = lambda values, dim=0: None

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub.no_grad = lambda: _NoGrad()
    set_module("torch", torch_stub)

    fetcher_stub = types.ModuleType("marketmamba.data.fetcher")
    fetcher_stub.run_daily_update = lambda *args, **kwargs: None
    fetcher_stub.load_ticker_universe = lambda *args, **kwargs: []
    set_module("marketmamba.data.fetcher", fetcher_stub)

    feature_engineer_stub = types.ModuleType("marketmamba.data.feature_engineer")
    feature_engineer_stub.build_features = lambda *args, **kwargs: None
    feature_engineer_stub.clean_and_scale = lambda *args, **kwargs: None
    set_module("marketmamba.data.feature_engineer", feature_engineer_stub)

    hygiene_stub = types.ModuleType("marketmamba.data.hygiene")
    hygiene_stub.check_data_health = lambda *args, **kwargs: None
    hygiene_stub.filter_tradable_universe = lambda df, *args, **kwargs: df
    set_module("marketmamba.data.hygiene", hygiene_stub)

    graph_builder_stub = types.ModuleType("marketmamba.knowledge.graph_builder")
    graph_builder_stub.update_correlation_edges = lambda *args, **kwargs: None
    set_module("marketmamba.knowledge.graph_builder", graph_builder_stub)

    report_generator_stub = types.ModuleType("marketmamba.llm.report_generator")
    report_generator_stub.generate_market_report = lambda *args, **kwargs: None
    report_generator_stub.build_market_data = lambda *args, **kwargs: {}
    set_module("marketmamba.llm.report_generator", report_generator_stub)

    return previous


def _restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _fresh_config():
    sys.path.insert(0, str(V6_ROOT))
    sys.modules.pop("marketmamba.config", None)
    return importlib.import_module("marketmamba.config")


def _load_daily_inference_module():
    previous = _install_import_stubs()
    sys.modules.pop("v61_daily_inference_under_test", None)
    spec = importlib.util.spec_from_file_location("v61_daily_inference_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        _restore_modules(previous)
    return module


def _embedding_dims_for(group_dims: dict[str, int], d_model: int = 256) -> dict[str, int]:
    total = sum(group_dims.values())
    dims = {name: max(1, int(d_model * dim / total)) for name, dim in group_dims.items()}
    remainder = d_model - sum(dims.values())
    if remainder:
        largest = max(group_dims, key=group_dims.get)
        dims[largest] += remainder
    return dims


class V61LegacyConfigIsolationTest(unittest.TestCase):
    def test_daily_entrypoint_switches_to_legacy_56d_before_binding_feature_cols(self):
        cfg = _fresh_config()
        self.assertEqual(59, cfg.INPUT_DIM)
        self.assertTrue(all(col in cfg.FEATURE_GROUPS["price_momentum"] for col in RS_COLS))

        module = _load_daily_inference_module()

        self.assertEqual(56, len(module.FEATURE_COLS))
        self.assertFalse(any(col in module.FEATURE_COLS for col in RS_COLS))
        self.assertEqual(LEGACY_GROUP_DIMS, module._V61_LEGACY_GROUP_DIMS)

        patched_cfg = sys.modules["marketmamba.config"]
        self.assertEqual(56, patched_cfg.INPUT_DIM)
        self.assertEqual(LEGACY_GROUP_DIMS, patched_cfg.GROUP_DIMS)

    def test_legacy_group_dims_match_existing_v6_best_checkpoint_embedding_shapes(self):
        module = _load_daily_inference_module()

        self.assertEqual(
            LEGACY_CHECKPOINT_EMBED_DIMS,
            _embedding_dims_for(module._V61_LEGACY_GROUP_DIMS),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
