"""Full-memory-first corrected E5 matrix materialization.

The formal path consumes the immutable prepared cache, verified lifecycle
universe, P0 historical-simulation labels, and frozen provenance policy.  It
does not recompute or neutralize features: the prepared cache already contains
the incumbent causal cross-sectional and expanding-macro transforms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from V6.experimental.v7_corrected_e5_replication import (
    ALLOWED_SOURCE_CLASSES,
    CorrectedE5Config,
    assign_split,
    canonical_hash,
    estimate_peak_ram,
    file_sha256,
    make_matrix_manifest,
    validate_matrix_manifest,
)

MATRIX_ARRAYS = {
    "X.npy": "X", "y5.npy": "y5", "y10.npy": "y10",
    "stock_ids.npy": "stock_ids", "dates.npy": "dates",
    "end_indices.npy": "end_indices", "splits.npy": "splits",
    "masks.npy": "masks",
}



def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required by the matrix materializer") from exc
    return pd


def validate_upstream_evidence(
    source_reliability: Mapping[str, Any],
    label_coverage: Mapping[str, Any],
) -> None:
    classes = source_reliability.get("classes", {})
    admitted = {name for name, row in classes.items() if row.get("eligible") is True}
    if admitted != ALLOWED_SOURCE_CLASSES:
        raise ValueError(
            "source eligibility must admit exactly the two frozen exchange classes"
        )
    if any(row.get("eligible") is True for name, row in classes.items()
           if name not in ALLOWED_SOURCE_CLASSES):
        raise ValueError("unapproved provenance class marked eligible")
    expected = {
        "evidence_class": "HISTORICAL_SIMULATION_PROXY",
        "selected_proxy_policy": "P0",
        "strict_phase0_decision": "STOP",
    }
    for key, value in expected.items():
        if label_coverage.get(key) != value:
            raise ValueError(f"historical label contract drift: {key}")
    counts = label_coverage.get("historical_simulated_valid_labels", {})
    if counts != {"5d": 6889229, "10d": 6790554}:
        raise ValueError("historical simulated label counts drift")


def _normalized_frames(features, universe, labels, config: CorrectedE5Config):
    pd = _require_pandas()
    missing = {"Date", "stock_id", *config.feature_order} - set(features.columns)
    if missing:
        raise ValueError("feature columns missing: " + ", ".join(sorted(missing)))
    features = features.loc[:, ["Date", "stock_id", *config.feature_order]].copy()
    features["Date"] = pd.to_datetime(features["Date"]).dt.strftime("%Y-%m-%d")
    features["stock_id"] = features["stock_id"].astype(str)

    universe = universe.copy()
    if "session" not in universe or "stock_id" not in universe:
        raise ValueError("universe requires session and stock_id")
    universe["Date"] = pd.to_datetime(universe["session"]).dt.strftime("%Y-%m-%d")
    universe["stock_id"] = universe["stock_id"].astype(str)
    if "membership" in universe:
        universe = universe[universe["membership"].isin(
            ("ELIGIBLE", "COMMON_STOCK", "VERIFIED_COMMON_STOCK")
        )]
    universe = universe.loc[:, ["Date", "stock_id"]].drop_duplicates()
    if universe.duplicated(["Date", "stock_id"]).any():
        raise ValueError("duplicate universe key")

    labels = labels.copy()
    source_date = "signal_date" if "signal_date" in labels else "Date"
    required = {source_date, "stock_id", "Alpha_5d", "Alpha_10d"}
    if not required.issubset(labels.columns):
        raise ValueError("historical labels lack required dual-head columns")
    labels["Date"] = pd.to_datetime(labels[source_date]).dt.strftime("%Y-%m-%d")
    labels["stock_id"] = labels["stock_id"].astype(str)
    if labels.duplicated(["Date", "stock_id"]).any():
        raise ValueError("duplicate historical label key")
    for horizon in (5, 10):
        status = f"label_status_{horizon}d"
        if status in labels:
            labels.loc[labels[status] != "VALID", f"Alpha_{horizon}d"] = np.nan
    labels = labels.loc[:, ["Date", "stock_id", "Alpha_5d", "Alpha_10d"]]

    frame = features.merge(
        universe, on=["Date", "stock_id"], how="inner", validate="many_to_one"
    )
    frame = frame.merge(
        labels, on=["Date", "stock_id"], how="left", validate="one_to_one"
    )
    frame = frame.sort_values(["Date", "stock_id"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["Date", "stock_id"]).any():
        raise ValueError("duplicate matrix key after joins")
    return frame


def build_arrays(features, universe, labels, config: CorrectedE5Config) -> dict[str, np.ndarray]:
    frame = _normalized_frames(features, universe, labels, config)
    values = frame.loc[:, config.feature_order].to_numpy(dtype=np.float32, copy=True)
    observation_mask = np.isfinite(values)
    values[~observation_mask] = 0.0
    dates = frame["Date"].to_numpy(dtype="datetime64[D]")
    stock_ids = frame["stock_id"].astype(str).to_numpy(dtype="U32")
    y5 = frame["Alpha_5d"].to_numpy(dtype=np.float32, copy=True)
    y10 = frame["Alpha_10d"].to_numpy(dtype=np.float32, copy=True)
    split_names = np.array([assign_split(str(value)) for value in dates], dtype="U24")
    encoded = np.array([
        {"warmup": 0, "train": 1, "research_evaluation": 2, "out_of_contract": 3, "purge": 4}[name]
        for name in split_names
    ], dtype=np.uint8)
    _, counts = np.unique(dates, return_counts=True)
    end_indices = np.cumsum(counts, dtype=np.int64)
    return {
        "X": values,
        "y5": y5,
        "y10": y10,
        "stock_ids": stock_ids,
        "dates": dates,
        "end_indices": end_indices,
        "splits": encoded,
        "split_names": split_names,
        "masks": observation_mask,
    }


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_matrix(
    arrays: Mapping[str, np.ndarray],
    output: Path,
    config: CorrectedE5Config,
    *,
    source_hashes: Mapping[str, str],
    build_mode: str,
    oom_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    names = MATRIX_ARRAYS
    for filename, key in names.items():
        _atomic_numpy(output / filename, np.asarray(arrays[key]))
    hashes = {filename: file_sha256(output / filename) for filename in names}
    manifest = make_matrix_manifest(
        config, rows=len(arrays["dates"]), files=hashes,
        source_hashes=source_hashes, build_mode=build_mode,
        oom_evidence=oom_evidence,
    )
    manifest["arrays"] = {
        name: {"shape": list(np.asarray(arrays[key]).shape),
               "dtype": str(np.asarray(arrays[key]).dtype)}
        for name, key in names.items()
    }
    manifest["ram_estimate"] = estimate_peak_ram(len(arrays["dates"]), 48)
    # arrays/ram are operational metadata and therefore part of the final identity.
    manifest["logical_identity"] = canonical_hash({
        key: value for key, value in manifest.items() if key != "logical_identity"
    })
    validate_matrix_manifest(manifest, config)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def hash_tree(paths: Sequence[Path], root: Path | None = None) -> str:
    rows = []
    root = root or Path.cwd()
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return canonical_hash(rows)


def _read_many(paths: Sequence[Path], columns: Sequence[str] | None = None):
    pd = _require_pandas()
    if not paths:
        raise FileNotFoundError("no Parquet inputs found")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def estimate_inputs(args, config: CorrectedE5Config) -> dict[str, Any]:
    """Estimate all large allocations from Parquet metadata before loading."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for pre-build RAM estimation") from exc
    feature_paths = sorted(args.feature_cache.glob("*.parquet"))
    if not feature_paths:
        feature_paths = sorted(args.feature_cache.glob(".prepare-cache/*.parquet"))
    universe_paths = sorted(args.universe.glob("eligible-universe-*.parquet"))
    label_paths = sorted(args.labels.glob("historical-simulated-labels-*.parquet"))
    if not feature_paths or not universe_paths or not label_paths:
        raise FileNotFoundError("feature, universe, and label Parquet inputs are required")
    row_counts = {
        "features": sum(pq.ParquetFile(path).metadata.num_rows for path in feature_paths),
        "universe": sum(pq.ParquetFile(path).metadata.num_rows for path in universe_paths),
        "labels": sum(pq.ParquetFile(path).metadata.num_rows for path in label_paths),
    }
    admitted_upper_bound = min(row_counts.values())
    estimate = estimate_peak_ram(admitted_upper_bound, len(config.feature_order))
    input_bytes = {
        "features": sum(path.stat().st_size for path in feature_paths),
        "universe": sum(path.stat().st_size for path in universe_paths),
        "labels": sum(path.stat().st_size for path in label_paths),
    }
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except ImportError:
        available = None
    return {
        "schema_version": "v7-corrected-e5-ram-estimate-v1",
        "row_counts": row_counts,
        "admitted_rows_upper_bound": admitted_upper_bound,
        "input_file_bytes": input_bytes,
        "allocations": estimate,
        "available_ram_bytes": available,
        "full_memory_advisory": (
            None if available is None
            else ("LIKELY_FITS" if estimate["estimated_peak"] < available * .8
                  else "OOM_POSSIBLE_FULL_MEMORY_STILL_REQUIRED_FIRST")
        ),
        "fallback_policy": "only after an actual MemoryError",
    }

def load_full_inputs(args, config: CorrectedE5Config):
    feature_paths = sorted(args.feature_cache.glob("*.parquet"))
    if not feature_paths:
        feature_paths = sorted(args.feature_cache.glob(".prepare-cache/*.parquet"))
    universe_paths = sorted(args.universe.glob("eligible-universe-*.parquet"))
    label_paths = sorted(args.labels.glob("historical-simulated-labels-*.parquet"))
    features = _read_many(feature_paths, ["Date", "stock_id", *config.feature_order])
    universe = _read_many(universe_paths)
    labels = _read_many(label_paths)
    source_hashes = {
        "features": hash_tree(feature_paths, args.feature_cache),
        "labels": hash_tree(label_paths, args.labels),
        "universe": hash_tree(universe_paths, args.universe),
        "provenance": canonical_hash({
            "source_reliability": file_sha256(args.source_reliability),
            "label_coverage": file_sha256(args.label_coverage),
        }),
    }
    return features, universe, labels, source_hashes


def build_full_memory(args, config: CorrectedE5Config) -> dict[str, Any]:
    features, universe, labels, hashes = load_full_inputs(args, config)
    arrays = build_arrays(features, universe, labels, config)
    return write_matrix(
        arrays, args.output, config, source_hashes=hashes,
        build_mode="full_memory",
    )


def build_chunked_after_oom(
    args, config: CorrectedE5Config, evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic year chunks, then bounded copy into final memory maps."""
    pd = _require_pandas()
    feature_paths = sorted(args.feature_cache.glob("*.parquet"))
    if not feature_paths:
        feature_paths = sorted(args.feature_cache.glob(".prepare-cache/*.parquet"))
    universe_paths = sorted(args.universe.glob("eligible-universe-*.parquet"))
    label_paths = sorted(args.labels.glob("historical-simulated-labels-*.parquet"))
    chunks = args.output / ".chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    chunk_rows = []
    for label_path in label_paths:
        year = label_path.stem.rsplit("-", 1)[-1]
        universe_path = args.universe / f"eligible-universe-{year}.parquet"
        if not universe_path.exists():
            raise FileNotFoundError(f"missing universe year {year}")
        pieces = []
        for path in feature_paths:
            piece = pd.read_parquet(path, columns=["Date", "stock_id", *config.feature_order])
            piece = piece[pd.to_datetime(piece["Date"]).dt.year == int(year)]
            if not piece.empty:
                pieces.append(piece)
        if not pieces:
            continue
        arrays = build_arrays(
            pd.concat(pieces, ignore_index=True),
            pd.read_parquet(universe_path),
            pd.read_parquet(label_path),
            config,
        )
        target = chunks / f"{year}.npz"
        np.savez(target, **{key: arrays[key] for key in (
            "X", "y5", "y10", "stock_ids", "dates", "splits", "masks")})
        chunk_rows.append((target, len(arrays["dates"])))
        del arrays, pieces
        gc.collect()
    if not chunk_rows:
        raise RuntimeError("OOM fallback produced no chunks")
    total_rows = sum(rows for _, rows in chunk_rows)
    arrays = {}
    for key in ("X", "y5", "y10", "stock_ids", "dates", "splits", "masks"):
        with np.load(chunk_rows[0][0], allow_pickle=False) as first:
            sample = first[key]
            shape = (total_rows, *sample.shape[1:])
            dtype = sample.dtype
        merged_path = chunks / f"merged-{key}.npy"
        target = np.lib.format.open_memmap(
            merged_path, mode="w+", dtype=dtype, shape=shape
        )
        offset = 0
        for chunk_path, rows in chunk_rows:
            with np.load(chunk_path, allow_pickle=False) as chunk:
                value = chunk[key]
                if len(value) != rows:
                    raise ValueError(f"chunk row count drift: {chunk_path}")
                target[offset:offset + rows] = value
            offset += rows
        target.flush()
        del target
        arrays[key] = np.load(merged_path, allow_pickle=False, mmap_mode="r")
    dates = arrays["dates"]
    ends = np.flatnonzero(dates[1:] != dates[:-1]).astype(np.int64) + 1
    arrays["end_indices"] = np.concatenate((
        ends, np.array([len(dates)], dtype=np.int64)
    ))
    hashes = {
        "features": hash_tree(feature_paths, args.feature_cache),
        "labels": hash_tree(label_paths, args.labels),
        "universe": hash_tree(universe_paths, args.universe),
        "provenance": canonical_hash({
            "source_reliability": file_sha256(args.source_reliability),
            "label_coverage": file_sha256(args.label_coverage),
        }),
    }
    return write_matrix(
        arrays, args.output, config, source_hashes=hashes,
        build_mode="chunked_after_oom", oom_evidence=evidence,
    )


def run(args) -> int:
    config = CorrectedE5Config.from_contract(args.contract, args.feature_manifest)
    source = json.loads(args.source_reliability.read_text(encoding="utf-8"))
    labels = json.loads(args.label_coverage.read_text(encoding="utf-8"))
    validate_upstream_evidence(source, labels)
    args.output.mkdir(parents=True, exist_ok=True)
    ram_estimate = estimate_inputs(args, config)
    _atomic_json(args.output / "ram-estimate.json", ram_estimate)
    print(json.dumps({"event": "ram_estimate", **ram_estimate}, ensure_ascii=False), flush=True)
    try:
        manifest = build_full_memory(args, config)
    except MemoryError as exc:
        evidence = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
            "fallback": "deterministic_year_chunks",
        }
        _atomic_json(args.output / "oom-evidence.json", evidence)
        manifest = build_chunked_after_oom(args, config, evidence)
    print(json.dumps({
        "status": "complete", "rows": manifest["rows"],
        "build_mode": manifest["build_mode"],
        "logical_identity": manifest["logical_identity"],
    }, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--feature-cache", type=Path, required=True)
    command.add_argument("--universe", type=Path, required=True)
    command.add_argument("--labels", type=Path, required=True)
    command.add_argument("--source-reliability", type=Path, required=True)
    command.add_argument("--label-coverage", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--feature-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv=None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
