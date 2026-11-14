"""Resumable per-session prediction export from immutable best checkpoints."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from v7_experiment_model import ExperimentConfig, ExperimentModel
from v7_experiment_storage import CheckpointStore, atomic_json, digest, fingerprint
from v7_stability_contract import validate_prediction_rows

REVISION = "v7-stability-export-v1"


def _move_sample(sample, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()}


def _chunk_rows(sample, predictions, *, model_id, data_id, seed):
    labels = sample["labels"].detach().cpu()
    scores = predictions.detach().float().cpu()
    stocks = tuple(str(value) for value in sample["stock_ids"])
    if scores.shape != labels.shape or scores.shape != (len(stocks), 2):
        raise ValueError("prediction, label and stock shapes do not match the two-head contract")
    rows = []
    for index, stock in enumerate(stocks):
        row = {
            "date": str(sample["Date"])[:10],
            "stock_id": stock,
            "score_5d": float(scores[index, 0]) if torch.isfinite(scores[index, 0]) else None,
            "score_10d": float(scores[index, 1]) if torch.isfinite(scores[index, 1]) else None,
            "valid_5d": bool(torch.isfinite(scores[index, 0])),
            "valid_10d": bool(torch.isfinite(scores[index, 1])),
            "label_5d": float(labels[index, 0]) if torch.isfinite(labels[index, 0]) else None,
            "label_10d": float(labels[index, 1]) if torch.isfinite(labels[index, 1]) else None,
            "label_5d_mature": bool(torch.isfinite(labels[index, 0])),
            "label_10d_mature": bool(torch.isfinite(labels[index, 1])),
            "model_id": str(model_id),
            "data_id": str(data_id),
            "seed": int(seed),
        }
        rows.append(row)
    return validate_prediction_rows(rows)


def export_checkpoint_predictions(
    *,
    checkpoint_dir,
    expected_checkpoint_identity,
    dataset,
    output,
    model_id,
    data_id,
    seed,
    device,
    ssd_fn,
    model_factory=ExperimentModel,
    max_dates=None,
):
    if max_dates is not None and max_dates < 1:
        raise ValueError("max_dates must be positive")
    store = CheckpointStore(checkpoint_dir)
    state = store.load(expected_checkpoint_identity, best=True)
    if state is None:
        raise OSError("best checkpoint is missing")
    source = store.manifest().get("best")
    identity = fingerprint({
        "revision": REVISION,
        "checkpoint_identity": expected_checkpoint_identity,
        "checkpoint": source,
        "model_id": str(model_id),
        "data_id": str(data_id),
        "seed": int(seed),
    })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = (
        __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else
        {"revision": REVISION, "identity": identity, "status": "running", "chunks": []}
    )
    if manifest.get("identity") != identity:
        raise ValueError("export source or settings changed; use a new output directory")
    completed = {}
    for item in manifest.get("chunks", []):
        path = output / item["file"]
        if not path.is_file() or digest(path) != item["sha256"]:
            raise OSError("export chunk verification failed: " + item["file"])
        completed[item["date"]] = item
    if manifest.get("status") == "complete":
        if len(completed) != len(dataset):
            raise OSError("complete export has the wrong number of date chunks")
        return {"status": "skipped", "dates": len(completed), "identity": identity}

    config = ExperimentConfig(**state["model_config"])
    config.validate()
    model = model_factory(config, ssd_fn=ssd_fn).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    added = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            day = str(sample["Date"])[:10]
            if day in completed:
                continue
            predictions = model.forward_prepared(_move_sample(sample, device))
            rows = _chunk_rows(sample, predictions, model_id=model_id, data_id=data_id, seed=seed)
            filename = "predictions-" + day + ".parquet"
            destination = output / filename
            temporary = output / (filename + ".tmp")
            pd.DataFrame(rows).to_parquet(temporary, index=False)
            temporary.replace(destination)
            entry = {"date": day, "file": filename, "sha256": digest(destination), "rows": len(rows)}
            manifest["chunks"].append(entry)
            manifest["chunks"].sort(key=lambda item: item["date"])
            atomic_json(manifest_path, manifest)
            completed[day] = entry
            added += 1
            if max_dates is not None and added >= max_dates and len(completed) < len(dataset):
                manifest["status"] = "paused"
                atomic_json(manifest_path, manifest)
                return {"status": "paused", "dates": len(completed), "identity": identity}
    if len(completed) != len(dataset):
        raise ValueError("dataset contains duplicate dates or export omitted a date")
    manifest["status"] = "complete"
    manifest["dates"] = len(completed)
    manifest["rows"] = sum(item["rows"] for item in manifest["chunks"])
    atomic_json(manifest_path, manifest)
    return {"status": "complete", "dates": len(completed), "identity": identity}
