"""V7 H1 isolated experiment configuration.

This module defines constants/settings that are scoped ONLY to the
"v7-h1-isolated-preparation" experiment. It intentionally does NOT
import from, depend on, or mutate any production configuration module
(e.g. marketmamba/config.py). It must remain importable in any
environment (no GPU, no CUDA, no Mamba2 kernel, no torch) so that it
can be used by synthetic/fake-kernel tests.

Scope note: this file is part of Plan
plan-v7-h1-isolated-preparation-v1-formalization, step-1-config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# Explicit isolation marker so downstream tooling / reviewers can
# confirm at a glance that nothing here is wired into production.
H1_ISOLATED_NAMESPACE: str = "v7_h1_isolated"
H1_PLAN_ID: str = "plan-v7-h1-isolated-preparation-v1-formalization"

# --- Mamba2 adapter dimensional configuration -----------------------------
# These are deliberately small/synthetic-friendly defaults suitable for
# fake-kernel / CPU-only static testing. They are NOT tuned for any
# production training run.
H1_MODEL_DIM: int = 64
H1_STATE_DIM: int = 16
H1_NUM_HEADS: int = 4
H1_HEAD_DIM: int = H1_MODEL_DIM // H1_NUM_HEADS
H1_NUM_LAYERS: int = 2
H1_CONV_KERNEL_SIZE: int = 4
H1_EXPAND_FACTOR: int = 2
H1_DT_MIN: float = 0.001
H1_DT_MAX: float = 0.1

# --- Runtime / fallback behavior flags -------------------------------------
# When True, the adapter is permitted to fall back to a pure-Python
# "fake kernel" implementation when a real CUDA-capable Mamba2 kernel is
# not detected. This enables import/static + synthetic tests to run
# without GPU/torch/mamba_ssm installed.
H1_ALLOW_FAKE_KERNEL_FALLBACK: bool = True

# When True, adapter/probe code paths that would require a real CUDA
# Mamba2 kernel are strictly forbidden from being invoked automatically.
# This plan only prepares the isolated harness; real GPU smoke tests and
# the 1-seed Colab A100 probe run require a separate, explicit approval.
H1_REQUIRE_EXPLICIT_APPROVAL_FOR_REAL_RUNTIME: bool = True

# --- Probe (finite-step) configuration -------------------------------------
# These describe the *shape* of the eventual limited-step probe; they do
# not, by themselves, trigger any execution.
H1_PROBE_DEFAULT_MAX_STEPS: int = 1
H1_PROBE_DEFAULT_SEED: int = 0
H1_PROBE_INPUT_SEQ_LEN: int = 8
H1_PROBE_INPUT_BATCH: int = 1


@dataclass(frozen=True)
class V7H1Config:
    """Immutable bundle of the isolated H1 experiment settings.

    Kept independent from any production `Config`/`Settings` class.
    """

    model_dim: int = H1_MODEL_DIM
    state_dim: int = H1_STATE_DIM
    num_heads: int = H1_NUM_HEADS
    head_dim: int = H1_HEAD_DIM
    num_layers: int = H1_NUM_LAYERS
    conv_kernel_size: int = H1_CONV_KERNEL_SIZE
    expand_factor: int = H1_EXPAND_FACTOR
    dt_min: float = H1_DT_MIN
    dt_max: float = H1_DT_MAX
    allow_fake_kernel_fallback: bool = H1_ALLOW_FAKE_KERNEL_FALLBACK
    require_explicit_approval_for_real_runtime: bool = (
        H1_REQUIRE_EXPLICIT_APPROVAL_FOR_REAL_RUNTIME
    )
    probe_default_max_steps: int = H1_PROBE_DEFAULT_MAX_STEPS
    probe_default_seed: int = H1_PROBE_DEFAULT_SEED
    probe_input_shape: Tuple[int, int, int] = field(
        default_factory=lambda: (
            H1_PROBE_INPUT_BATCH,
            H1_PROBE_INPUT_SEQ_LEN,
            H1_MODEL_DIM,
        )
    )

    def validate(self) -> None:
        """Lightweight self-consistency check (no I/O, no side effects)."""
        if self.model_dim <= 0 or self.state_dim <= 0:
            raise ValueError("model_dim and state_dim must be positive")
        if self.num_heads <= 0 or self.model_dim % self.num_heads != 0:
            raise ValueError("num_heads must evenly divide model_dim")
        if self.probe_default_max_steps < 0:
            raise ValueError("probe_default_max_steps must be >= 0")


def get_default_config() -> V7H1Config:
    """Return a fresh default isolated H1 config instance."""
    cfg = V7H1Config()
    cfg.validate()
    return cfg
