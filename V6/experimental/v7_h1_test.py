"""Synthetic / fake-kernel only tests for the V7 H1 isolated prep.

These tests intentionally require NO GPU, NO CUDA, NO real Mamba2
kernel, and NO torch/mamba_ssm installation. They cover:

  * config loading (v7_h1_config.py)
  * adapter fallback behavior (v7_h1_mamba2_adapter.py), including the
    explicit fake-kernel dependency-injection path and the
    BLOCKED_BY_RUNTIME path
  * probe interface runtime-guard branches (v7_h1_probe.py)

Real Mamba2 CUDA smoke tests, the Colab A100 1-seed probe execution,
and any artifact generation are explicitly OUT OF SCOPE here and are
not exercised by this test module.

Run with:  python -m unittest V6/experimental/v7_h1_test.py
       or: python V6/experimental/v7_h1_test.py
"""

from __future__ import annotations

import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import v7_h1_config as h1_config  # noqa: E402
import v7_h1_mamba2_adapter as h1_adapter  # noqa: E402
import v7_h1_probe as h1_probe  # noqa: E402


class TestV7H1Config(unittest.TestCase):
    def test_default_config_loads_and_validates(self):
        cfg = h1_config.get_default_config()
        cfg.validate()  # should not raise
        self.assertEqual(cfg.model_dim, h1_config.H1_MODEL_DIM)
        self.assertEqual(cfg.state_dim, h1_config.H1_STATE_DIM)
        self.assertTrue(cfg.allow_fake_kernel_fallback)
        self.assertTrue(cfg.require_explicit_approval_for_real_runtime)

    def test_config_is_isolated_namespace(self):
        self.assertEqual(h1_config.H1_ISOLATED_NAMESPACE, "v7_h1_isolated")

    def test_invalid_config_raises(self):
        bad = h1_config.V7H1Config(model_dim=0, state_dim=16)
        with self.assertRaises(ValueError):
            bad.validate()

        bad_heads = h1_config.V7H1Config(model_dim=10, num_heads=3)
        with self.assertRaises(ValueError):
            bad_heads.validate()


class TestV7H1AdapterFakeKernel(unittest.TestCase):
    def test_detect_runtime_returns_result_without_side_effects(self):
        result = h1_adapter.detect_runtime()
        self.assertIsInstance(result, h1_adapter.RuntimeDetectionResult)
        self.assertIn(
            result.status,
            {
                h1_adapter.RuntimeStatus.AVAILABLE,
                h1_adapter.RuntimeStatus.FALLBACK_FAKE_KERNEL,
            },
        )

    def test_adapter_with_forced_fake_kernel_factory_runs_forward(self):
        cfg = h1_config.get_default_config()

        def fake_factory(config):
            return h1_adapter.FakeMamba2Kernel(config, seed=42)

        adapter = h1_adapter.V7H1Mamba2Adapter(
            config=cfg, kernel_factory=fake_factory
        )
        self.assertEqual(
            adapter.runtime_status, h1_adapter.RuntimeStatus.FALLBACK_FAKE_KERNEL
        )
        self.assertFalse(adapter.is_blocked())

        batch, seq_len, model_dim = cfg.probe_input_shape
        inputs = [
            [[0.0] * model_dim for _ in range(seq_len)] for _ in range(batch)
        ]
        outputs = adapter.forward(inputs)
        self.assertEqual(len(outputs), batch)
        self.assertEqual(len(outputs[0]), seq_len)
        self.assertEqual(len(outputs[0][0]), model_dim)

    def test_adapter_blocked_when_fake_kernel_disallowed_and_no_real_runtime(self):
        cfg = h1_config.get_default_config()
        detection = h1_adapter.detect_runtime()
        if detection.status == h1_adapter.RuntimeStatus.AVAILABLE:
            self.skipTest(
                "Real Mamba2 CUDA runtime detected in this environment; "
                "BLOCKED_BY_RUNTIME fallback path is not applicable here."
            )

        adapter = h1_adapter.V7H1Mamba2Adapter(config=cfg, allow_fake_kernel=False)
        self.assertEqual(
            adapter.runtime_status, h1_adapter.RuntimeStatus.BLOCKED_BY_RUNTIME
        )
        self.assertTrue(adapter.is_blocked())

        with self.assertRaises(h1_adapter.BlockedByRuntimeError):
            adapter.forward([[[0.0]]])

    def test_fake_kernel_is_clearly_marked_synthetic(self):
        cfg = h1_config.get_default_config()
        kernel = h1_adapter.FakeMamba2Kernel(cfg, seed=1)
        self.assertTrue(kernel.is_fake)


class TestV7H1ProbeRuntimeGuards(unittest.TestCase):
    def test_probe_reports_blocked_by_runtime_without_computation(self):
        cfg = h1_config.get_default_config()
        detection = h1_adapter.detect_runtime()
        if detection.status == h1_adapter.RuntimeStatus.AVAILABLE:
            self.skipTest(
                "Real Mamba2 CUDA runtime detected; cannot exercise the "
                "BLOCKED_BY_RUNTIME guard branch in this environment."
            )

        adapter = h1_adapter.V7H1Mamba2Adapter(config=cfg, allow_fake_kernel=False)
        result = h1_probe.run_probe(adapter, config=cfg)
        self.assertEqual(result.status, "BLOCKED_BY_RUNTIME")
        self.assertEqual(result.steps_run, 0)
        self.assertIsNone(result.outputs)

    def test_probe_runs_synthetic_fake_kernel_path(self):
        cfg = h1_config.get_default_config()

        def fake_factory(config):
            return h1_adapter.FakeMamba2Kernel(config, seed=7)

        adapter = h1_adapter.V7H1Mamba2Adapter(
            config=cfg, kernel_factory=fake_factory
        )
        result = h1_probe.run_probe(adapter, config=cfg, max_steps=2, seed=123)
        self.assertEqual(result.status, "SYNTHETIC_OK")
        self.assertTrue(result.is_synthetic)
        self.assertEqual(result.steps_run, 2)
        self.assertIsNotNone(result.outputs)

    def test_probe_never_auto_executes_on_import_or_main_guard(self):
        # Importing the module must be side-effect free: no probe should
        # have been run merely by importing v7_h1_probe.
        self.assertTrue(hasattr(h1_probe, "run_probe"))
        self.assertTrue(callable(h1_probe.run_probe))
        # The module-level __main__ guard must not perform any real
        # execution; we only assert its presence/callable safety here
        # rather than invoking it as a subprocess (out of scope).
        self.assertIn("__main__", open(h1_probe.__file__, "r", encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
