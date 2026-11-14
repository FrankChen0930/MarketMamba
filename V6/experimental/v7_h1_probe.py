"""V7 H1 isolated finite-step probe harness.

Defines the execution *interface*/harness for a limited-step probe of
the Mamba2 adapter. This module does NOT, by itself, run any real
GPU/Colab/CUDA workload -- it only provides a structural harness with
explicit runtime guards so it can be statically loaded and exercised
via synthetic/fake-kernel tests.

Any actual Mamba2 CUDA smoke test and the 1-seed probe execution on
Colab A100 are explicitly OUT OF SCOPE for this Plan
(plan-v7-h1-isolated-preparation-v1-formalization) and require a
separate, later, explicit human approval. If a compatible runtime is
not detected, callers get a `BLOCKED_BY_RUNTIME` result rather than any
attempt to repair, install, or upgrade the environment.

Scope note: this file corresponds to step-3-probe of the approved Plan.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from v7_h1_config import V7H1Config, get_default_config  # noqa: E402
from v7_h1_mamba2_adapter import (  # noqa: E402
    RuntimeStatus,
    V7H1Mamba2Adapter,
    BlockedByRuntimeError,
)


@dataclass
class ProbeResult:
    """Outcome of a (synthetic or guarded) probe run."""

    status: str
    steps_run: int
    seed: int
    is_synthetic: bool
    notes: str
    outputs: Optional[List] = field(default=None, repr=False)


def run_probe(
    adapter: V7H1Mamba2Adapter,
    config: Optional[V7H1Config] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> ProbeResult:
    """Execute a finite-step probe against `adapter`.

    Runtime-guard behavior:
      * If `adapter.runtime_status` is BLOCKED_BY_RUNTIME, this function
        immediately returns a ProbeResult with status
        "BLOCKED_BY_RUNTIME" and performs zero computation.
      * If `adapter.runtime_status` is AVAILABLE (a real, CUDA-capable
        Mamba2 kernel was detected), this function still refuses to
        execute a real run -- that requires separate explicit approval
        for a Colab A100 smoke test -- and returns
        "REAL_RUNTIME_REQUIRES_SEPARATE_APPROVAL" instead.
      * Only when `adapter.runtime_status` is FALLBACK_FAKE_KERNEL does
        this function actually invoke `adapter.forward(...)`, and even
        then strictly through the synthetic fake-kernel path.

    This function never launches Colab, never touches the network, and
    never installs/upgrades dependencies.
    """
    cfg = config or adapter.config or get_default_config()
    resolved_max_steps = (
        cfg.probe_default_max_steps if max_steps is None else max_steps
    )
    resolved_seed = cfg.probe_default_seed if seed is None else seed

    if adapter.runtime_status == RuntimeStatus.BLOCKED_BY_RUNTIME:
        return ProbeResult(
            status="BLOCKED_BY_RUNTIME",
            steps_run=0,
            seed=resolved_seed,
            is_synthetic=False,
            notes=(
                "No compatible Mamba2 CUDA kernel/runtime detected and "
                "fake-kernel fallback was not permitted. No installation "
                "or environment modification was attempted."
            ),
        )

    if adapter.runtime_status == RuntimeStatus.AVAILABLE:
        return ProbeResult(
            status="REAL_RUNTIME_REQUIRES_SEPARATE_APPROVAL",
            steps_run=0,
            seed=resolved_seed,
            is_synthetic=False,
            notes=(
                "A real CUDA-capable Mamba2 runtime appears available, but "
                "executing a real smoke test / 1-seed Colab A100 probe is "
                "out of scope for this Plan and requires separate explicit "
                "approval. No real computation was performed."
            ),
        )

    # FALLBACK_FAKE_KERNEL: synthetic-only execution path.
    batch, seq_len, model_dim = cfg.probe_input_shape
    synthetic_inputs = [
        [[float((s + b + d) % 7) / 7.0 for d in range(model_dim)] for s in range(seq_len)]
        for b in range(batch)
    ]

    outputs = None
    steps_run = 0
    for _ in range(max(0, resolved_max_steps)):
        outputs = adapter.forward(synthetic_inputs)
        steps_run += 1

    return ProbeResult(
        status="SYNTHETIC_OK",
        steps_run=steps_run,
        seed=resolved_seed,
        is_synthetic=True,
        notes=(
            "Executed against the synthetic fake-kernel fallback only. "
            "This is NOT a real Mamba2 CUDA smoke test result."
        ),
        outputs=outputs,
    )


def build_default_adapter(config: Optional[V7H1Config] = None) -> V7H1Mamba2Adapter:
    """Convenience constructor mirroring normal (non-test) usage: relies
    on environment auto-detection, with fake-kernel fallback permitted
    per config. Does not force any particular runtime path.
    """
    return V7H1Mamba2Adapter(config=config or get_default_config())


# Intentionally NOT calling run_probe()/build_default_adapter() at
# import time or under `if __name__ == "__main__":` with any implicit
# execution. This module must remain side-effect-free on import and on
# direct script execution so it can be safely statically loaded.
if __name__ == "__main__":  # pragma: no cover - manual/documentation only
    print(
        "v7_h1_probe.py provides a harness interface only. "
        "It does not auto-execute any real GPU/Colab probe run. "
        "Use run_probe(adapter, ...) explicitly from a controlled, "
        "separately-approved context."
    )
