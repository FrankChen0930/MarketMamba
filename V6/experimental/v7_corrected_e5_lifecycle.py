"""Incumbent-faithful lifecycle state and durable checkpoint primitives."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from typing import Any, Mapping

import torch

from V6.experimental.v7_corrected_e5_replication import file_sha256


CHECKPOINT_FORMAT = "marketmamba-v7-corrected-e5-v2"
REQUIRED_CHECKPOINT_FIELDS = {
    "format", "model_state", "optimizer_state", "scheduler_state",
    "epoch", "batch", "step", "phase", "validation_index",
    "validation_rows", "best_rank_ic_5d", "best_epoch",
    "patience_reference", "bad_epochs", "history", "rng_state",
    "contract_identity", "matrix_identity", "terminal", "stop_reason",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value, handle, ensure_ascii=False, sort_keys=True,
                indent=2, allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    """Immutable, digest-verified latest/previous/best checkpoint generations."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "checkpoint.json"

    def manifest(self) -> dict[str, Any]:
        if not self.pointer.exists():
            return {}
        try:
            value = json.loads(self.pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OSError("checkpoint pointer is unreadable") from exc
        if not isinstance(value, dict):
            raise OSError("checkpoint pointer must be an object")
        return value

    def save(self, state: Mapping[str, Any], *, best: bool = False) -> dict[str, Any]:
        missing = REQUIRED_CHECKPOINT_FIELDS - set(state)
        if missing:
            raise ValueError(
                "incomplete lifecycle checkpoint: " + ", ".join(sorted(missing))
            )
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported lifecycle checkpoint format")
        old = self.manifest()
        name = "state-" + uuid.uuid4().hex + ".pt"
        destination = self.root / name
        with tempfile.TemporaryDirectory(prefix="v7-corrected-e5-checkpoint-") as folder:
            local = Path(folder) / name
            torch.save(dict(state), local)
            expected = file_sha256(local)
            with local.open("rb") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024**2)
                target.flush()
                os.fsync(target.fileno())
            if file_sha256(destination) != expected:
                destination.unlink(missing_ok=True)
                raise OSError("checkpoint copy verification failed")
        entry = {
            "file": name,
            "bytes": destination.stat().st_size,
            "sha256": expected,
            "epoch": int(state["epoch"]),
            "batch": int(state["batch"]),
            "step": int(state["step"]),
            "phase": str(state["phase"]),
        }
        updated = {
            "schema_version": "v7-corrected-e5-checkpoint-pointer-v1",
            "latest": entry,
            "previous": old.get("latest"),
            "best": entry if best else old.get("best"),
        }
        _atomic_json(self.pointer, updated)
        keep = {
            row["file"] for key in ("latest", "previous", "best")
            if (row := updated.get(key)) is not None
        }
        for path in self.root.glob("state-*.pt"):
            if path.name not in keep:
                path.unlink()
        return entry

    def load(
        self,
        expected_contract_identity: str,
        expected_matrix_identity: str,
        *,
        best: bool = False,
    ) -> dict[str, Any] | None:
        manifest = self.manifest()
        candidates = (
            [manifest.get("best")]
            if best else [manifest.get("latest"), manifest.get("previous")]
        )
        for entry in candidates:
            if not entry:
                continue
            path = self.root / str(entry.get("file", ""))
            if (
                not path.is_file()
                or path.stat().st_size != entry.get("bytes")
                or file_sha256(path) != entry.get("sha256")
            ):
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(state, dict):
                continue
            missing = REQUIRED_CHECKPOINT_FIELDS - set(state)
            if missing or state.get("format") != CHECKPOINT_FORMAT:
                continue
            if state.get("contract_identity") != expected_contract_identity:
                raise ValueError("checkpoint contract identity mismatch")
            if state.get("matrix_identity") != expected_matrix_identity:
                raise ValueError("checkpoint matrix identity mismatch")
            return state
        if manifest:
            raise OSError("no verified checkpoint generation is available")
        return None


def fresh_lifecycle_state(
    contract_identity: str,
    matrix_identity: str,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "epoch": 0,
        "batch": 0,
        "step": 0,
        "phase": "train",
        "validation_index": 0,
        "validation_rows": [],
        "best_rank_ic_5d": None,
        "best_epoch": None,
        "patience_reference": None,
        "bad_epochs": 0,
        "history": [],
        "terminal": False,
        "stop_reason": None,
        "contract_identity": contract_identity,
        "matrix_identity": matrix_identity,
    }


def update_selection(
    state: dict[str, Any],
    score: float,
    *,
    epoch: int,
    min_delta: float,
) -> bool:
    best = (
        state.get("best_rank_ic_5d") is None
        or score > float(state["best_rank_ic_5d"])
    )
    if best:
        state["best_rank_ic_5d"] = float(score)
        state["best_epoch"] = int(epoch)
    reference = state.get("patience_reference")
    if reference is None or score > float(reference) + min_delta:
        state["patience_reference"] = float(score)
        state["bad_epochs"] = 0
    else:
        state["bad_epochs"] = int(state.get("bad_epochs", 0)) + 1
    return best


def early_stop_eligible(
    *,
    epoch: int,
    bad_epochs: int,
    epochs: int,
    warmup_fraction: float,
    minimum_epochs: int,
    patience: int,
) -> bool:
    first = max(minimum_epochs, math.ceil(epochs * warmup_fraction) + 1)
    return epoch >= first and bad_epochs >= patience


def stage_namespace(root: Path, stage: str, seed: int) -> Path:
    safe_stage = stage.replace("/", "-").replace("\\", "-")
    return Path(root) / "checkpoints" / f"{safe_stage}-seed-{int(seed)}"
