"""Contracts and fail-closed helpers for Corrected E5 replication.

This module is deliberately side-effect free.  Full-history I/O lives in the
matrix CLI so importing contracts never allocates the dataset or probes CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ALLOWED_SOURCE_CLASSES = {
    "EXCHANGE_REGULAR_BOARD_VERIFIED",
    "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR",
}
REQUIRED_MATRIX_FILES = {
    "X.npy", "y5.npy", "y10.npy", "stock_ids.npy", "dates.npy",
    "end_indices.npy", "splits.npy", "masks.npy",
}
REQUIRED_SOURCE_HASHES = {"features", "labels", "universe", "provenance"}
FORBIDDEN_FEATURES = {
    "PER", "PBR", "Revenue_MoM", "Revenue_YoY", "EPS", "EPS_Surprise",
    "Gross_Margin", "ROE", "Book_Value", "Dividend_Yield_Fwd",
    "Free_Cash_Flow",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_features(path: str | Path) -> tuple[str, ...]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = tuple(document["feature_order"])
    if document.get("feature_count") != 48 or len(columns) != 48:
        raise ValueError("canonical corrected feature contract must contain 48 columns")
    if len(set(columns)) != len(columns):
        raise ValueError("canonical feature order contains duplicates")
    if FORBIDDEN_FEATURES.intersection(columns):
        raise ValueError("uncertified feature present in canonical corrected contract")
    expected = hashlib.sha256(
        json.dumps(list(columns), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if document.get("feature_order_sha256") != expected:
        raise ValueError("canonical feature-order hash mismatch")
    return columns


@dataclass(frozen=True)
class CorrectedE5Config:
    feature_order: tuple[str, ...]
    group_dims: tuple[int, int, int, int]
    d_model: int
    d_state: int
    expand: int
    head_dim: int
    n_groups: int
    sequence_length: int
    temporal_layers: int
    forward_layers: int
    reverse_layers: int
    dropout: float
    horizons: tuple[int, int]
    seeds: tuple[int, int, int]
    epochs: int
    patience: int
    minimum_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    minimum_lr_factor: float
    gradient_clip: float
    precision: str
    graph_enabled: bool
    industry_neutralization: bool
    current_industry_backfill: bool
    train_start: str
    train_end: str
    evaluation_start: str
    evaluation_end: str
    embargo_trading_days: int
    evidence_class: str
    proxy_policy: str

    @classmethod
    def from_contract(
        cls, contract_path: str | Path, feature_manifest_path: str | Path,
    ) -> "CorrectedE5Config":
        doc = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        fields = doc["fields"]
        architecture = fields["architecture"]["value"]
        training = fields["training"]["value"]
        split = fields["split"]["value"]
        labels = fields["labels"]["value"]
        matrix = doc["matrix"]["preprocessing"]
        config = cls(
            feature_order=load_canonical_features(feature_manifest_path),
            group_dims=tuple(fields["features"]["value"]["group_dims"]),
            d_model=int(architecture["d_model"]),
            d_state=int(architecture["d_state"]),
            expand=int(architecture["expand"]),
            head_dim=int(architecture["head_dim"]),
            n_groups=int(architecture["n_groups"]),
            sequence_length=int(architecture["sequence_length"]),
            temporal_layers=int(architecture["temporal_layers"]),
            forward_layers=int(architecture["cross_stock_forward_layers"]),
            reverse_layers=int(architecture["cross_stock_reverse_layers"]),
            dropout=float(architecture["dropout"]),
            horizons=tuple(architecture["horizons"]),
            seeds=tuple(fields["seeds"]["value"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            minimum_epochs=int(training["minimum_epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_fraction=float(training["warmup_fraction"]),
            minimum_lr_factor=float(training["minimum_lr_factor"]),
            gradient_clip=float(training["gradient_clip"]),
            precision=str(training["precision"]),
            graph_enabled=bool(architecture["graph"]["enabled"]),
            industry_neutralization=bool(matrix["industry_neutralization"]),
            current_industry_backfill=bool(matrix["current_industry_backfill"]),
            train_start=split["train_start"],
            train_end=split["train_end"],
            evaluation_start=split["research_evaluation_start"],
            evaluation_end=split["research_evaluation_end"],
            embargo_trading_days=int(split["embargo_trading_days"]),
            evidence_class=labels["evidence_class"],
            proxy_policy=labels["proxy_policy"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.feature_order) != 48 or sum(self.group_dims) != 48:
            raise ValueError("corrected E5 must use 48 features in groups 15/20/1/12")
        if self.group_dims != (15, 20, 1, 12):
            raise ValueError("factor-group contract drift")
        if (self.d_model, self.d_state, self.sequence_length) != (64, 32, 60):
            raise ValueError("incumbent E5 dimensions drift")
        if (self.temporal_layers, self.forward_layers, self.reverse_layers) != (1, 1, 1):
            raise ValueError("incumbent E5 depth drift")
        if self.horizons != (5, 10) or self.seeds != (17, 29, 43):
            raise ValueError("head or seed contract drift")
        if self.precision != "fp32":
            raise ValueError("formal corrected E5 training is FP32 only")
        if self.graph_enabled or self.industry_neutralization or self.current_industry_backfill:
            raise ValueError("corrected E5 prohibits graph and industry-derived preprocessing")
        if self.evidence_class != "HISTORICAL_SIMULATION_PROXY" or self.proxy_policy != "P0":
            raise ValueError("historical label evidence contract drift")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "features": self.feature_order,
            "groups": self.group_dims,
            "architecture": {
                "d_model": self.d_model, "d_state": self.d_state,
                "sequence_length": self.sequence_length,
                "temporal_layers": self.temporal_layers,
                "forward_layers": self.forward_layers,
                "reverse_layers": self.reverse_layers,
                "horizons": self.horizons,
                "graph_enabled": self.graph_enabled,
            },
            "training": {
                "epochs": self.epochs, "patience": self.patience,
                "minimum_epochs": self.minimum_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_fraction": self.warmup_fraction,
                "minimum_lr_factor": self.minimum_lr_factor,
                "gradient_clip": self.gradient_clip,
                "precision": self.precision, "seeds": self.seeds,
            },
            "split": {
                "train_start": self.train_start, "train_end": self.train_end,
                "evaluation_start": self.evaluation_start,
                "evaluation_end": self.evaluation_end,
                "embargo_trading_days": self.embargo_trading_days,
            },
            "labels": {
                "evidence_class": self.evidence_class,
                "proxy_policy": self.proxy_policy,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_hash(self.identity_payload())


def validate_source_classes(values: Iterable[str]) -> None:
    observed = set(values)
    invalid = observed - ALLOWED_SOURCE_CLASSES
    if invalid or not observed:
        raise ValueError(
            "ineligible or missing OHLCV source class; fail closed: "
            + ", ".join(sorted(invalid or {"<empty>"}))
        )


def canonical_stock_order(stock_ids: Sequence[object]) -> tuple[int, ...]:
    keys = tuple(str(item) for item in stock_ids)
    if len(keys) != len(set(keys)):
        raise ValueError("stock IDs must be unique within a daily cross-section")
    return tuple(sorted(range(len(keys)), key=lambda index: (keys[index], index)))


def assign_split(value: str) -> str:
    day = date.fromisoformat(str(value)[:10])
    if day < date(2013, 1, 2):
        return "warmup"
    if day <= date(2023, 11, 17):
        return "train"
    if day <= date(2023, 12, 29):
        return "purge"
    if date(2024, 1, 2) <= day <= date(2026, 9, 11):
        return "research_evaluation"
    return "out_of_contract"


def join_historical_labels(
    feature_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in label_rows:
        key = (str(row["Date"])[:10], str(row["stock_id"]))
        if key in lookup:
            raise ValueError(f"duplicate historical label key: {key}")
        lookup[key] = row
    result = []
    for source in feature_rows:
        row = dict(source)
        key = (str(row["Date"])[:10], str(row["stock_id"]))
        label = lookup.get(key)
        row["Alpha_5d"] = (
            float(label["Alpha_5d"])
            if label is not None and label.get("Alpha_5d") is not None
            else float("nan")
        )
        row["Alpha_10d"] = (
            float(label["Alpha_10d"])
            if label is not None and label.get("Alpha_10d") is not None
            else float("nan")
        )
        result.append(row)
    return result


def estimate_peak_ram(rows: int, features: int = 48, id_bytes: int = 16) -> dict[str, int]:
    if rows < 0 or features < 1 or id_bytes < 1:
        raise ValueError("RAM estimate dimensions must be positive")
    raw = rows * (features * 8 + id_bytes + 32)
    dataframe = rows * (features * 8 + id_bytes + 96)
    final = rows * (features * 4 + 2 * 4 + 2 + id_bytes + 16)
    metadata = max(1 << 20, rows // 4)
    peak = raw + dataframe + final + metadata
    return {
        "raw_input": raw,
        "dataframe_working": dataframe,
        "final_arrays": final,
        "metadata": metadata,
        "estimated_peak": peak,
    }


def make_matrix_manifest(
    config: CorrectedE5Config,
    *,
    rows: int,
    files: Mapping[str, str],
    source_hashes: Mapping[str, str],
    build_mode: str,
    oom_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "v7-corrected-e5-matrix-v1",
        "baseline": "E5-PIT-Clean-v1",
        "rows": int(rows),
        "feature_order": list(config.feature_order),
        "feature_order_sha256": canonical_hash(list(config.feature_order)),
        "config_sha256": config.sha256,
        "files": dict(files),
        "source_hashes": dict(source_hashes),
        "build_mode": build_mode,
        "oom_evidence": dict(oom_evidence) if oom_evidence else None,
        "preprocessing": {
            "industry_neutralization": False,
            "current_industry_backfill": False,
            "cross_sectional_scaling": "causal per-session",
            "macro_scaling": "causal expanding",
        },
        "source_gate": sorted(ALLOWED_SOURCE_CLASSES),
        "graph": {"enabled": False, "edge_count": 0, "fusion_branch": "ABSENT"},
        "label_contract": {
            "evidence_class": "HISTORICAL_SIMULATION_PROXY",
            "proxy_policy": "P0",
            "horizons": [5, 10],
        },
        "split_contract": {
            "train": [config.train_start, config.train_end],
            "purge": ["2023-11-20", "2023-12-29"],
            "research_evaluation": [config.evaluation_start, config.evaluation_end],
            "evaluation_semantics": "research-used historical evaluation",
            "label_horizon_trading_days": max(config.horizons),
            "embargo_trading_days": config.embargo_trading_days,
            "total_purge_trading_days": max(config.horizons) + config.embargo_trading_days,
        },
        "ordering": ["Date", "stock_id"],
    }
    manifest["logical_identity"] = canonical_hash({
        key: value for key, value in manifest.items() if key != "logical_identity"
    })
    return manifest


def validate_matrix_manifest(
    manifest: Mapping[str, Any], config: CorrectedE5Config,
) -> None:
    errors = []
    if manifest.get("schema_version") != "v7-corrected-e5-matrix-v1":
        errors.append("schema")
    if tuple(manifest.get("feature_order", ())) != config.feature_order:
        errors.append("feature_order")
    if manifest.get("config_sha256") != config.sha256:
        errors.append("config_sha256")
    if set(manifest.get("files", {})) != REQUIRED_MATRIX_FILES:
        errors.append("files")
    if set(manifest.get("source_hashes", {})) != REQUIRED_SOURCE_HASHES:
        errors.append("source_hashes")
    if manifest.get("preprocessing") != {
        "industry_neutralization": False,
        "current_industry_backfill": False,
        "cross_sectional_scaling": "causal per-session",
        "macro_scaling": "causal expanding",
    }:
        errors.append("preprocessing")
    if manifest.get("graph") != {
        "enabled": False, "edge_count": 0, "fusion_branch": "ABSENT",
    }:
        errors.append("graph")
    if manifest.get("source_gate") != sorted(ALLOWED_SOURCE_CLASSES):
        errors.append("source_gate")
    if manifest.get("label_contract", {}).get("evidence_class") != "HISTORICAL_SIMULATION_PROXY":
        errors.append("labels")
    split_contract = manifest.get("split_contract", {})
    if (
        split_contract.get("evaluation_semantics") != "research-used historical evaluation"
        or split_contract.get("label_horizon_trading_days") != 10
        or split_contract.get("embargo_trading_days") != 20
        or split_contract.get("total_purge_trading_days") != 30
    ):
        errors.append("split")
    if manifest.get("build_mode") == "chunked_after_oom" and not manifest.get("oom_evidence"):
        errors.append("unrecorded_chunk_fallback")
    if manifest.get("build_mode") not in {"full_memory", "chunked_after_oom"}:
        errors.append("build_mode")
    observed_identity = canonical_hash({
        key: value for key, value in manifest.items() if key != "logical_identity"
    })
    if manifest.get("logical_identity") != observed_identity:
        errors.append("logical_identity")
    if errors:
        raise ValueError("invalid corrected E5 matrix manifest: " + ", ".join(errors))
