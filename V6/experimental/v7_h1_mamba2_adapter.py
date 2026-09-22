"""V7 H1 isolated Mamba2 adapter.

Provides a thin wrapper interface around a Mamba2 kernel with explicit
runtime detection. If no compatible CUDA Mamba2 kernel is available in
the current environment, this module never attempts to install or
upgrade any dependency; it either:

  * falls back to a pure-Python synthetic "fake kernel" (only for
    import/static/synthetic testing purposes), or
  * reports RuntimeStatus.BLOCKED_BY_RUNTIME and refuses to execute any
    real computation.

Scope note: this file is part of Plan
plan-v7-h1-isolated-preparation-v1-formalization, step-2-adapter.
It does not perform any real Mamba2 CUDA smoke test and does not launch
any GPU/Colab execution -- that remains out of scope pending separate
explicit approval.
"""

from __future__ import annotations

import enum
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

# This module deliberately avoids relying on V6/experimental being a
# proper Python package (no __init__.py is part of this Plan's allowed
# scope). Instead, it ensures its own directory is importable and pulls
# in the sibling isolated config module by plain module name.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from v7_h1_config import V7H1Config, get_default_config  # noqa: E402


class RuntimeStatus(str, enum.Enum):
    """Result of probing the local environment for a real Mamba2 kernel."""

    AVAILABLE = "AVAILABLE"
    FALLBACK_FAKE_KERNEL = "FALLBACK_FAKE_KERNEL"
    BLOCKED_BY_RUNTIME = "BLOCKED_BY_RUNTIME"


@dataclass(frozen=True)
class RuntimeDetectionResult:
    status: RuntimeStatus
    torch_available: bool
    cuda_available: bool
    mamba_ssm_available: bool
    detail: str


def detect_runtime() -> RuntimeDetectionResult:
    """Detect whether a real, CUDA-capable Mamba2 kernel is usable here.

    This function performs read-only detection: it only attempts
    `import` statements guarded by try/except and simple attribute
    checks. It never installs, upgrades, or modifies any dependency.
    """
    torch_available = False
    cuda_available = False
    mamba_ssm_available = False

    try:
        import torch  # type: ignore

        torch_available = True
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    except Exception:
        torch_available = False

    try:
        import mamba_ssm  # type: ignore  # noqa: F401

        mamba_ssm_available = True
    except Exception:
        mamba_ssm_available = False

    if torch_available and cuda_available and mamba_ssm_available:
        return RuntimeDetectionResult(
            status=RuntimeStatus.AVAILABLE,
            torch_available=torch_available,
            cuda_available=cuda_available,
            mamba_ssm_available=mamba_ssm_available,
            detail="Real Mamba2 CUDA kernel dependencies detected.",
        )

    # No real kernel usable. Decide between fake-kernel fallback (for
    # synthetic/static testing) and a hard BLOCKED_BY_RUNTIME report.
    return RuntimeDetectionResult(
        status=RuntimeStatus.FALLBACK_FAKE_KERNEL,
        torch_available=torch_available,
        cuda_available=cuda_available,
        mamba_ssm_available=mamba_ssm_available,
        detail=(
            "Real Mamba2 CUDA kernel not usable "
            f"(torch_available={torch_available}, cuda_available={cuda_available}, "
            f"mamba_ssm_available={mamba_ssm_available}). "
            "Fake-kernel fallback is available for synthetic testing only."
        ),
    )


class FakeMamba2Kernel:
    """Pure-Python synthetic stand-in for a real Mamba2 kernel.

    This is NOT a numerical approximation of Mamba2 -- it merely
    preserves input shape semantics (batch, seq_len, model_dim) via a
    deterministic, seed-controlled transform so that adapter/probe
    plumbing can be exercised without torch, CUDA, or mamba_ssm
    installed. It must never be mistaken for, or reported as, a real
    Mamba2 CUDA smoke test result.
    """

    is_fake: bool = True

    def __init__(self, config: V7H1Config, seed: int = 0) -> None:
        self.config = config
        self.seed = seed

    def forward(self, inputs: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
        """Deterministic synthetic transform: identity plus a small,
        seeded perturbation. Shape is preserved.
        """
        rng = random.Random(self.seed)
        output: List[List[List[float]]] = []
        for batch_item in inputs:
            out_batch: List[List[float]] = []
            for step in batch_item:
                out_step = [float(v) + (rng.random() - 0.5) * 1e-6 for v in step]
                out_batch.append(out_step)
            output.append(out_batch)
        return output


class BlockedByRuntimeError(RuntimeError):
    """Raised when adapter execution is attempted without a usable runtime
    and without an explicit fake-kernel opt-in for testing."""


class V7H1Mamba2Adapter:
    """Wrapper interface around a Mamba2 kernel for the isolated H1 prep.

    Parameters
    ----------
    config:
        Isolated H1 config (see v7_h1_config.py). Defaults to
        `get_default_config()`.
    kernel_factory:
        Optional injection point used by tests to force a specific fake
        kernel implementation, bypassing environment auto-detection.
        Production/real usage should leave this as None.
    allow_fake_kernel:
        Explicit, caller-controlled opt-in to use the synthetic fake
        kernel fallback when no real runtime is available. Defaults to
        the value in `config.allow_fake_kernel_fallback`.
    """

    def __init__(
        self,
        config: Optional[V7H1Config] = None,
        kernel_factory: Optional[Callable[[V7H1Config], Any]] = None,
        allow_fake_kernel: Optional[bool] = None,
    ) -> None:
        self.config = config or get_default_config()
        self.config.validate()
        self._allow_fake_kernel = (
            self.config.allow_fake_kernel_fallback
            if allow_fake_kernel is None
            else allow_fake_kernel
        )

        self.detection: RuntimeDetectionResult = detect_runtime()
        self._kernel: Optional[Any] = None
        self.runtime_status: RuntimeStatus

        if kernel_factory is not None:
            # Test/dependency-injection path: caller fully controls the
            # kernel implementation regardless of environment detection.
            self._kernel = kernel_factory(self.config)
            self.runtime_status = RuntimeStatus.FALLBACK_FAKE_KERNEL
            return

        if self.detection.status == RuntimeStatus.AVAILABLE:
            # Real kernel construction is intentionally NOT performed in
            # this Plan's scope. Even when detection reports AVAILABLE,
            # actually instantiating/running the real Mamba2 kernel
            # (a CUDA smoke test) requires separate explicit approval.
            self.runtime_status = RuntimeStatus.AVAILABLE
            self._kernel = None
        elif self._allow_fake_kernel:
            self._kernel = FakeMamba2Kernel(self.config)
            self.runtime_status = RuntimeStatus.FALLBACK_FAKE_KERNEL
        else:
            self.runtime_status = RuntimeStatus.BLOCKED_BY_RUNTIME
            self._kernel = None

    def is_blocked(self) -> bool:
        return self.runtime_status == RuntimeStatus.BLOCKED_BY_RUNTIME

    def forward(self, inputs: Sequence[Sequence[Sequence[float]]]):
        """Run a forward pass, strictly limited to the fake-kernel path.

        Real CUDA Mamba2 execution is out of scope for this Plan and is
        never performed here, even when the environment nominally
        supports it -- that requires a separate, explicitly approved
        smoke test.
        """
        if self.runtime_status == RuntimeStatus.BLOCKED_BY_RUNTIME:
            raise BlockedByRuntimeError(
                "BLOCKED_BY_RUNTIME: no compatible Mamba2 CUDA kernel detected "
                "and fake-kernel fallback was not permitted."
            )
        if self.runtime_status == RuntimeStatus.AVAILABLE and self._kernel is None:
            raise BlockedByRuntimeError(
                "Real Mamba2 CUDA execution is out of scope for this Plan "
                "and requires separate explicit approval (Colab A100 smoke test)."
            )
        if self._kernel is None:
            raise BlockedByRuntimeError("No kernel available to run forward().")
        return self._kernel.forward(inputs)
