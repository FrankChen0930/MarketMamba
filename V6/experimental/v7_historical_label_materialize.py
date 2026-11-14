"""Materialize simulation-only labels from provenance-eligible exchange observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from V6.experimental.v7_temporal_contract import normalize_calendar


SCHEMA_VERSION = "v7-historical-simulated-labels-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_valid_label_frame(frame, calendar: Sequence[str], *, horizons=(5, 10)):
    """Return only labels that satisfy the frozen no-roll executable contract."""
    import pandas as pd

    days = normalize_calendar(calendar)
    positions = {day: index for index, day in enumerate(days)}
    requested = tuple(int(value) for value in horizons)
    if not requested or any(value < 1 for value in requested):
        raise ValueError("horizons must be positive")

    work = frame.copy()
    work["Date"] = work["Date"].astype(str).str[:10]
    work["stock_id"] = work["stock_id"].astype(str).str.strip()
    work["_position"] = work["Date"].map(positions)
    work = work[
        work["_position"].notna()
        & work["observation_valid"].eq(True)
        & ~work["suspended"].eq(True)
    ].copy()
    group_columns = ["stock_id"]
    if "market" in work.columns:
        group_columns = ["market", "stock_id"]
    work = work.sort_values(group_columns + ["_position"], kind="mergesort")
    grouped = work.groupby(group_columns, sort=False, observed=True)

    identity = ["stock_id"]
    if "market" in work.columns:
        identity.insert(0, "market")
    result = work[identity + ["Date"]].rename(columns={"Date": "signal_date"}).copy()
    any_valid = pd.Series(False, index=work.index)
    for horizon in requested:
        suffix = f"{horizon}d"
        entry_position = grouped["_position"].shift(-1)
        exit_position = grouped["_position"].shift(-horizon)
        entry_open = grouped["Open"].shift(-1)
        exit_close = grouped["Close"].shift(-horizon)
        entry_executable = grouped["open_executable"].shift(-1).eq(True)
        valid = (
            entry_executable
            & entry_position.eq(work["_position"] + 1)
            & exit_position.eq(work["_position"] + horizon)
            & entry_open.map(lambda value: isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
            & exit_close.map(lambda value: isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
        )
        any_valid |= valid
        entry_session = pd.Series(None, index=work.index, dtype=object)
        exit_session = pd.Series(None, index=work.index, dtype=object)
        entry_session.loc[valid] = (
            work.loc[valid, "_position"].add(1).astype(int).map(days.__getitem__)
        )
        exit_session.loc[valid] = (
            work.loc[valid, "_position"].add(horizon).astype(int).map(days.__getitem__)
        )
        result[f"entry_session_{suffix}"] = entry_session
        result[f"exit_session_{suffix}"] = exit_session
        result[f"entry_price_{suffix}"] = entry_open.where(valid)
        result[f"exit_price_{suffix}"] = exit_close.where(valid)
        result[f"Alpha_{suffix}"] = (exit_close / entry_open - 1.0).where(valid)
        result[f"label_status_{suffix}"] = valid.map({True: "VALID", False: "NOT_MATERIALIZED_INVALID"})
    return result.loc[any_valid].reset_index(drop=True)


def _load_eligible_observations(prices_path: Path, universe_dir: Path):
    import numpy as np
    import pandas as pd

    prices = pd.read_parquet(
        prices_path,
        columns=["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"],
    )
    prices["Date"] = prices["Date"].astype(str).str[:10]
    prices["stock_id"] = prices["stock_id"].astype(str).str.strip()
    finite = np.isfinite(prices[["Open", "High", "Low", "Close", "Volume"]]).all(axis=1)
    prices = prices[
        finite
        & prices["Open"].gt(0)
        & prices["High"].gt(0)
        & prices["Low"].gt(0)
        & prices["Close"].gt(0)
        & prices["Volume"].gt(0)
        & prices["Low"].le(prices["Open"])
        & prices["Open"].le(prices["High"])
    ].copy()

    frames = []
    for path in sorted(universe_dir.glob("*.parquet")):
        universe = pd.read_parquet(
            path, columns=["session", "stock_id", "market", "membership", "tradability"]
        )
        universe["session"] = universe["session"].astype(str).str[:10]
        universe["stock_id"] = universe["stock_id"].astype(str).str.strip()
        universe = universe[
            universe["membership"].eq("ELIGIBLE")
            & ~universe["tradability"].eq("SUSPENDED")
        ]
        year = path.stem.rsplit("-", 1)[-1]
        price_year = prices[prices["Date"].str.startswith(year)]
        joined = price_year.merge(
            universe[["session", "stock_id", "market"]],
            left_on=["Date", "stock_id"],
            right_on=["session", "stock_id"],
            how="inner",
            validate="one_to_one",
        ).drop(columns="session")
        frames.append(joined)
    observations = pd.concat(frames, ignore_index=True)
    observations["observation_valid"] = True
    observations["open_executable"] = True
    observations["suspended"] = False
    return observations


def materialize(args) -> dict[str, Any]:
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    selected = validation["selection"]["selected_policy"]
    if selected != "P0" or validation["policies"]["P0"]["critical_false_executable"] != 0:
        raise RuntimeError("historical labels require accepted frozen P0 with zero critical false executable")

    calendar_document = json.loads(args.calendar.read_text(encoding="utf-8"))
    calendar = calendar_document["trading_calendar"]
    observations = _load_eligible_observations(args.verified_prices, args.universe_dir)
    labels = build_valid_label_frame(observations, calendar, horizons=(5, 10))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for year, frame in labels.groupby(labels["signal_date"].str[:4], sort=True):
        target = args.output_dir / f"historical-simulated-labels-{year}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        frame.sort_values(["signal_date", "market", "stock_id"], kind="mergesort").to_parquet(
            temporary, index=False
        )
        temporary.replace(target)
        outputs.append(
            {
                "path": str(target),
                "year": year,
                "rows": len(frame),
                "sha256": _sha(target),
                "bytes": target.stat().st_size,
            }
        )

    counts = {
        f"{horizon}d": int(labels[f"Alpha_{horizon}d"].notna().sum())
        for horizon in (5, 10)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline": "E5-PIT-Clean-v1",
        "evidence_class": "HISTORICAL_SIMULATION_PROXY",
        "selected_proxy_policy": "P0",
        "strict_phase0_decision": "STOP",
        "corrected_verified_executable_labels": {"5d": 0, "10d": 0},
        "historical_simulated_valid_labels": counts,
        "materialization_status": "MATERIALIZED",
        "contract": {
            "signal_timing": "session t after close",
            "entry": "next frozen calendar session open; never roll",
            "exit": "h-th holding session close with entry session counted as 1",
            "formula": "exit_close / entry_open - 1",
            "holding_interval": "every session must have provenance-eligible observation and not be suspended",
        },
        "inputs": {
            "verified_prices": {
                "path": str(args.verified_prices),
                "sha256": _sha(args.verified_prices),
            },
            "calendar": {"path": str(args.calendar), "sha256": _sha(args.calendar)},
            "validation": {"path": str(args.validation), "sha256": _sha(args.validation)},
        },
        "observations_admitted": len(observations),
        "output_files": outputs,
        "gpu_used": False,
        "legacy_abc_modified": False,
    }
    _write_json(args.artifact_root / "label-coverage.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--verified-prices", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    manifest = materialize(parser.parse_args())
    print(json.dumps({
        "historical_simulated_valid_labels": manifest["historical_simulated_valid_labels"],
        "output_files": len(manifest["output_files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
