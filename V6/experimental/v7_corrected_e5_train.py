"""Faithful, resumable trainer and evaluation for Corrected E5 replication."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from V6.experimental.v7_corrected_e5_model import CorrectedE5Model
from V6.experimental.v7_corrected_e5_replication import (
    CorrectedE5Config,
    canonical_hash,
    file_sha256,
    validate_matrix_manifest,
)
from V6.experimental.v7_corrected_e5_lifecycle import (
    CheckpointStore,
    early_stop_eligible,
    fresh_lifecycle_state,
    update_selection,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _centered_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = torch.cumsum(counts, 0)
    ranks = torch.empty_like(values)
    ranks[order] = torch.repeat_interleave(
        (ends - counts + ends - 1).to(values.dtype) / 2, counts
    )
    return ranks - ranks.mean()


def rank_center_targets(labels: Tensor) -> Tensor:
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("labels must have shape (N, 2)")
    targets = torch.zeros_like(labels)
    for column in range(2):
        valid = torch.isfinite(labels[:, column])
        if valid.any():
            targets[valid, column] = _centered_ranks(labels[valid, column])
    return targets


def _listnet(prediction: Tensor, target: Tensor) -> Tensor:
    return -(torch.softmax(target, 0) * torch.log_softmax(prediction, 0)).sum()


def short_loss(predictions: Tensor, labels: Tensor) -> Tensor | None:
    """Exact incumbent dual-head objective: IC10 head receives weight 0.5."""
    if predictions.shape != labels.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and labels must have shape (N, 2)")
    if not torch.isfinite(predictions).all():
        raise ValueError("nonfinite predictions")
    targets = rank_center_targets(labels)
    terms = []
    for head, weight in ((0, 1.), (1, .5)):
        valid = torch.isfinite(labels[:, head])
        if valid.any():
            terms.append(weight * (
                F.mse_loss(predictions[valid, head], targets[valid, head])
                + .5 * _listnet(predictions[valid, head], targets[valid, head])
            ))
    return sum(terms) if terms else None


def _rank_ic(prediction: Tensor, label: Tensor) -> float:
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite predictions")
    valid = torch.isfinite(label)
    if int(valid.sum()) < 2:
        return float("nan")
    x = _centered_ranks(prediction[valid])
    y = _centered_ranks(label[valid])
    denominator = x.norm() * y.norm()
    return float((x @ y / denominator).item()) if float(denominator) > 0 else float("nan")


def rank_ic_by_horizon(predictions: Tensor, labels: Tensor) -> dict[str, float]:
    return {
        "rank_ic_5d": _rank_ic(predictions[:, 0], labels[:, 0]),
        "rank_ic_10d": _rank_ic(predictions[:, 1], labels[:, 1]),
    }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("rank_ic_5d", "rank_ic_10d"):
        values = np.array([row[name] for row in rows], dtype=np.float64)
        result[name] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else None
        )
        result[name + "_dates"] = int(np.isfinite(values).sum())
    return result


def evaluate_daily(
    daily: Sequence[Mapping[str, Any]],
    *,
    checkpoint_identity: str,
) -> dict[str, Any]:
    rows = []
    yearly: dict[str, list[dict[str, float]]] = defaultdict(list)
    common_rows = 0
    common_daily = []
    for sample in daily:
        predictions = sample["predictions"].detach().float().cpu()
        labels = sample["labels"].detach().float().cpu()
        metrics = rank_ic_by_horizon(predictions, labels)
        day = str(sample["Date"])[:10]
        row = {"Date": day, **metrics}
        rows.append(row)
        yearly[day[:4]].append(metrics)
        common = torch.isfinite(labels).all(dim=1)
        common_rows += int(common.sum())
        common_daily.append(rank_ic_by_horizon(
            predictions[common], labels[common]
        ) if int(common.sum()) >= 2 else {
            "rank_ic_5d": float("nan"), "rank_ic_10d": float("nan")
        })
    if not rows:
        raise ValueError("evaluation requires at least one daily cross-section")
    return {
        "checkpoint_identity": checkpoint_identity,
        "selection_metric": "rank_ic_5d",
        "rank_ic_10d_semantics": "same_checkpoint_as_rank_ic_5d",
        "overall": _aggregate(rows),
        "yearly": {year: _aggregate(values) for year, values in sorted(yearly.items())},
        "common_support": {"rows": common_rows, **_aggregate(common_daily)},
        "daily": rows,
        "evaluation_semantics": "research-used historical evaluation",
    }


def lr_factor(step: int, total: int, warmup_fraction: float) -> float:
    if total <= 1:
        return 1.
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return .04 + .96 * step / max(1, warmup - 1)
    progress = min(1., (step - warmup) / max(1, total - 1 - warmup))
    return .0001 + .9999 * .5 * (1 + math.cos(math.pi * progress))


def adamw_parameter_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.},
    ]


def replication_stages() -> list[tuple[str, int]]:
    return [
        ("A-smoke", 17),
        ("B-seed17", 17),
        ("C-seed29", 29),
        ("C-seed43", 43),
    ]


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    batch: int,
    step: int,
    best_rank_ic_5d: float | None,
    bad_epochs: int,
    contract_identity: str,
    matrix_identity: str,
    rng_state: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] = (),
) -> str:
    payload = {
        "format": "marketmamba-v7-corrected-e5-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "batch": int(batch),
        "step": int(step),
        "best_rank_ic_5d": best_rank_ic_5d,
        "bad_epochs": int(bad_epochs),
        "contract_identity": contract_identity,
        "matrix_identity": matrix_identity,
        "rng_state": dict(rng_state or capture_rng_state()),
        "history": list(history),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_sha256(path)


def load_checkpoint(
    path: Path,
    *,
    expected_contract_identity: str,
    expected_matrix_identity: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "marketmamba-v7-corrected-e5-v1":
        raise ValueError("unsupported checkpoint format")
    if payload.get("contract_identity") != expected_contract_identity:
        raise ValueError("checkpoint contract identity mismatch")
    if payload.get("matrix_identity") != expected_matrix_identity:
        raise ValueError("checkpoint matrix identity mismatch")
    required = {
        "model_state", "optimizer_state", "scheduler_state", "epoch", "batch",
        "step", "best_rank_ic_5d", "bad_epochs", "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError("incomplete checkpoint: " + ", ".join(sorted(missing)))
    return payload


class CorrectedMatrixDataset:
    """Daily cross-sections with per-stock trailing windows and no future rows."""
    def __init__(self, root: Path, config: CorrectedE5Config, split_code: int,
                 manifest_validator=validate_matrix_manifest):
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest_validator(manifest, config)
        self.manifest = manifest
        self.config = config
        self.split_code = split_code
        self.X = np.load(root / "X.npy", mmap_mode="r")
        self.y5 = np.load(root / "y5.npy", mmap_mode="r")
        self.y10 = np.load(root / "y10.npy", mmap_mode="r")
        self.stock_ids = np.load(root / "stock_ids.npy", mmap_mode="r")
        self.dates = np.load(root / "dates.npy", mmap_mode="r")
        self.splits = np.load(root / "splits.npy", mmap_mode="r")
        self.feature_masks = np.load(root / "masks.npy", mmap_mode="r")
        by_stock: dict[str, list[int]] = defaultdict(list)
        for index, stock in enumerate(self.stock_ids):
            by_stock[str(stock)].append(index)
        self.by_stock = {
            key: np.asarray(value, dtype=np.int64) for key, value in by_stock.items()
        }
        self.days = [
            value for value in np.unique(self.dates[self.splits == split_code])
        ]

    def __len__(self):
        return len(self.days)

    def __getitem__(self, item):
        day = self.days[item]
        targets = np.flatnonzero((self.dates == day) & (self.splits == self.split_code))
        n = len(targets)
        sequence = np.zeros(
            (n, self.config.sequence_length, len(self.config.feature_order)),
            dtype=np.float32,
        )
        observations = np.zeros((n, self.config.sequence_length), dtype=bool)
        for row, target in enumerate(targets):
            history = self.by_stock[str(self.stock_ids[target])]
            history = history[history <= target][-self.config.sequence_length:]
            # Explicit invariant: no row after the target can enter a sequence.
            if len(history) and history[-1] > target:
                raise RuntimeError("future-row leakage")
            start = self.config.sequence_length - len(history)
            sequence[row, start:] = self.X[history]
            observations[row, start:] = self.feature_masks[history].all(axis=1)
        labels = np.stack((self.y5[targets], self.y10[targets]), axis=1)
        return {
            "Date": str(day),
            "x": torch.from_numpy(sequence),
            "observation_mask": torch.from_numpy(observations),
            "labels": torch.from_numpy(labels),
            "stock_ids": [str(value) for value in self.stock_ids[targets]],
        }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_model(
    model, dataset, device, checkpoint_identity: str, predictions_path=None,
):
    model.eval()
    daily = []
    prediction_rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            predictions = model(
                sample["x"].to(device), sample["stock_ids"],
                sample["observation_mask"].to(device),
            )
            values = predictions.detach().float().cpu()
            daily.append({
                "Date": sample["Date"],
                "predictions": values,
                "labels": sample["labels"],
            })
            if predictions_path is not None:
                prediction_rows.extend({
                    "Date": str(sample["Date"])[:10],
                    "stock_id": stock_id,
                    "prediction_5d": float(values[row, 0]),
                    "prediction_10d": float(values[row, 1]),
                } for row, stock_id in enumerate(sample["stock_ids"]))
    result = evaluate_daily(daily, checkpoint_identity=checkpoint_identity)
    if predictions_path is not None:
        import pandas as pd
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
        pd.DataFrame(prediction_rows).to_parquet(temporary, index=False)
        os.replace(temporary, predictions_path)
    return result


def train_seed(
    matrix: Path,
    output: Path,
    config: CorrectedE5Config,
    *,
    seed: int,
    device: torch.device,
    smoke_steps: int | None = None,
    model_factory=CorrectedE5Model,
    result_path: Path | None = None,
    predictions_path: Path | None = None,
    checkpoint_store=None,
    matrix_manifest_validator=validate_matrix_manifest,
    telemetry_hook=None,
    telemetry_provider=None,
) -> dict[str, Any]:
    if device.type != "cuda" and smoke_steps is None:
        raise ValueError("formal training requires CUDA; CPU is smoke-only")
    set_seed(seed)
    train = CorrectedMatrixDataset(matrix, config, split_code=1,
                                   manifest_validator=matrix_manifest_validator)
    evaluation = CorrectedMatrixDataset(matrix, config, split_code=2,
                                        manifest_validator=matrix_manifest_validator)
    model = model_factory(config).to(device)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
    )
    total = len(train) * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total, config.warmup_fraction)
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_store or CheckpointStore(output / "checkpoints")
    matrix_doc = train.manifest
    matrix_identity = matrix_doc["logical_identity"]
    state = checkpoints.load(config.sha256, matrix_identity)
    if state is None:
        state = fresh_lifecycle_state(config.sha256, matrix_identity)
    else:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        restore_rng_state(state["rng_state"])
    def persist(*, best: bool = False) -> None:
        state.update({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
        })
        checkpoints.save(state, best=best)

    updates_this_call = 0
    smoke_reload_performed = False
    stop_reason = state.get("stop_reason") or "epoch_ceiling"
    while not state["terminal"] and int(state["epoch"]) < config.epochs:
        current_epoch = int(state["epoch"])
        if state["phase"] == "train":
            model.train()
            start_batch = int(state["batch"])
        else:
            start_batch = len(train)
        for current_batch in range(start_batch, len(train)):
            sample = train[current_batch]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                sample["x"].to(device), sample["stock_ids"],
                sample["observation_mask"].to(device),
            )
            loss = short_loss(predictions.float(), sample["labels"].to(device))
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                state["step"] = int(state["step"]) + 1
                updates_this_call += 1
            state["batch"] = current_batch + 1
            persist()
            if smoke_steps is not None and updates_this_call == 1:
                restored = checkpoints.load(config.sha256, matrix_identity)
                model.load_state_dict(restored["model_state"])
                optimizer.load_state_dict(restored["optimizer_state"])
                scheduler.load_state_dict(restored["scheduler_state"])
                restore_rng_state(restored["rng_state"])
                state = restored
                smoke_reload_performed = True
            if smoke_steps is not None and updates_this_call >= smoke_steps:
                batch_metrics = rank_ic_by_horizon(
                    predictions.detach().float().cpu(), sample["labels"]
                )
                return {
                    "status": "smoke_complete", "seed": seed,
                    "step": int(state["step"]),
                    "checkpoint": str(checkpoints.pointer),
                    "checkpoint_reload": smoke_reload_performed,
                    "metrics": batch_metrics,
                }
        if state["phase"] == "train":
            state.update({
                "phase": "validation", "batch": 0,
                "validation_index": 0, "validation_rows": [],
            })
            persist()

        model.eval()
        with torch.no_grad():
            for validation_index in range(
                int(state["validation_index"]), len(evaluation)
            ):
                sample = evaluation[validation_index]
                predictions = model(
                    sample["x"].to(device), sample["stock_ids"],
                    sample["observation_mask"].to(device),
                ).detach().float().cpu()
                state["validation_rows"].append({
                    "Date": str(sample["Date"])[:10],
                    **rank_ic_by_horizon(predictions, sample["labels"]),
                })
                state["validation_index"] = validation_index + 1
                persist()
        epoch_metrics = _aggregate(state["validation_rows"])
        score = epoch_metrics["rank_ic_5d"]
        if score is None:
            raise RuntimeError("Rank IC 5d unavailable; checkpoint selection cannot change")
        completed_epoch = current_epoch + 1
        improved = update_selection(
            state, float(score), epoch=completed_epoch, min_delta=0.
        )
        state["history"].append({
            "epoch": completed_epoch, "step": int(state["step"]),
            "rank_ic_5d": score,
            "rank_ic_10d_same_checkpoint": epoch_metrics["rank_ic_10d"],
            "best_rank_ic_5d": state["best_rank_ic_5d"],
            "best_epoch": state["best_epoch"],
            "patience_reference": state["patience_reference"],
            "bad_epochs": state["bad_epochs"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        state.update({
            "epoch": completed_epoch, "batch": 0, "phase": "train",
            "validation_index": 0, "validation_rows": [],
        })
        if early_stop_eligible(
            epoch=completed_epoch, bad_epochs=int(state["bad_epochs"]),
            epochs=config.epochs, warmup_fraction=config.warmup_fraction,
            minimum_epochs=config.minimum_epochs, patience=config.patience,
        ):
            state["terminal"] = True
            state["stop_reason"] = "early_stopping"
            stop_reason = state["stop_reason"]
        if telemetry_hook is not None and telemetry_provider is not None:
            telemetry_hook(completed_epoch, telemetry_provider())
        persist(best=improved)
        if stop_reason == "early_stopping":
            break
    best_state = checkpoints.load(config.sha256, matrix_identity, best=True)
    if not state["terminal"]:
        state["terminal"] = True
        state["stop_reason"] = "epoch_ceiling"
        stop_reason = state["stop_reason"]
        persist()
    model.load_state_dict(best_state["model_state"])
    best_entry = checkpoints.manifest()["best"]
    checkpoint_identity = best_entry["sha256"]
    final_metrics = evaluate_model(
        model, evaluation, device, checkpoint_identity, predictions_path
    )
    result = {
        "schema_version": "v7-corrected-e5-result-v1",
        "status": "complete", "seed": seed,
        "config_sha256": config.sha256,
        "matrix_identity": matrix_identity,
        "checkpoint_identity": checkpoint_identity,
        "selection_metric": "rank_ic_5d",
        "rank_ic_10d_semantics": "same_checkpoint_as_rank_ic_5d",
        "metrics": final_metrics,
        "best_epoch": best_state["best_epoch"],
        "last_epoch": state["epoch"],
        "stop_reason": stop_reason,
        "history": state["history"],
        "evidence_class": "HISTORICAL_SIMULATION_PROXY",
        "strict_phase0": "STOP",
        "historical_simulation_readiness": "PASS",
    }
    result_path = result_path or output / "result.json"
    _atomic_json(result_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("A-smoke", "B-seed17", "C-seed29", "C-seed43"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    config = CorrectedE5Config.from_contract(args.contract, args.feature_manifest)
    mapping = dict(replication_stages())
    smoke = 2 if args.stage == "A-smoke" else None
    result = train_seed(
        args.matrix, args.output / "checkpoints" / args.stage, config,
        seed=mapping[args.stage], device=torch.device(args.device),
        smoke_steps=smoke,
        result_path=args.output / "metrics" / f"{args.stage}.json",
        predictions_path=(
            None if smoke else
            args.output / "predictions" / f"{args.stage}.parquet"
        ),
    )
    print(json.dumps({
        "status": result["status"], "stage": args.stage,
        "seed": mapping[args.stage],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
