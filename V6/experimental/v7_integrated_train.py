"""Colab entry point for lazy V7 training and held-out evaluation.

Imports are CPU safe. Data, CUDA, Mamba, and portfolio code are resolved only
inside an explicit command. Prepared Parquet is the primary data interface;
an all-samples tensor bundle is deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from v7_integrated_config import (DataProtocol, V6_FEATURE_COLUMNS, V7IntegratedConfig,
                                  default_manifest_path)  # noqa: E402
from v7_integrated_model import MarketMambaV7Integrated, canonical_stock_order  # noqa: E402
from v7_integrated_data_quality import (QualityBlocked, QualityPolicy, calendar_contract, protocol_metadata, validate_protocol, assess_prices, calendar_labels, assess_training, validate_stock_ids, coverage_entries, select_declared_prices)
from v7_integrated_probe import load_runtime_metadata, preflight  # noqa: E402


@dataclass(frozen=True)
class PreparedStockBatch:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    padding_mask: Tensor
    stock_ids: tuple[str, ...]
    observation_mask: Tensor
    original_size: int
    canonical_original_indices: Tensor

    def scatter_to_original(self, values: Tensor) -> Tensor:
        if values.shape[0] != len(self.stock_ids):
            raise ValueError("values do not match prepared stock count")
        restored = values.new_zeros((self.original_size, *values.shape[1:]))
        restored[self.canonical_original_indices.to(values.device)] = values
        return restored


def prepare_stock_batch(x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                        padding_mask: Tensor, stock_ids: Sequence[object], observation_mask: Tensor | None = None) -> PreparedStockBatch:
    if x.ndim != 3 or padding_mask.shape != x.shape[:2] or len(stock_ids) != x.shape[0]:
        raise ValueError("x, padding_mask, and stock_ids are inconsistent")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError("edge tensors are inconsistent")
    if observation_mask is not None and observation_mask.shape != padding_mask.shape:
        raise ValueError("observation_mask shape mismatch")
    observation_mask = padding_mask.bool() if observation_mask is None else observation_mask.bool() & padding_mask.bool()
    eligible = observation_mask[:, -1]
    eligible_original = torch.nonzero(eligible, as_tuple=False).flatten()
    eligible_ids = [str(stock_ids[index]) for index in eligible_original.tolist()]
    order, _ = canonical_stock_order(eligible_ids)
    canonical_original = eligible_original[order]
    mapping = torch.full((x.shape[0],), -1, dtype=torch.long, device=edge_index.device)
    mapping[canonical_original.to(edge_index.device)] = torch.arange(canonical_original.numel(), device=edge_index.device)
    valid = ((edge_index >= 0) & (edge_index < x.shape[0])).all(0)
    safe_edges, safe_attrs = edge_index[:, valid], edge_attr[valid]
    keep = eligible.to(edge_index.device)[safe_edges[0]] & eligible.to(edge_index.device)[safe_edges[1]]
    return PreparedStockBatch(x=x[canonical_original.to(x.device)],
        edge_index=mapping[safe_edges[:, keep]], edge_attr=safe_attrs[keep],
        padding_mask=padding_mask[canonical_original].bool(), observation_mask=observation_mask[canonical_original],
        stock_ids=tuple(eligible_ids[index] for index in order.tolist()), original_size=x.shape[0],
        canonical_original_indices=canonical_original.cpu())


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def zero_macro_group(x: Tensor, config: V7IntegratedConfig) -> Tensor:
    if x.shape[-1] != config.input_dim:
        raise ValueError("feature width must be 59")
    result = x.clone(); start = sum(config.group_dims[:3])
    result[..., start:start + config.group_dims[3]] = 0
    return result


def validate_feature_metadata(metadata: Mapping[str, Any], config: V7IntegratedConfig) -> tuple[str, ...]:
    columns = metadata.get("feature_columns", metadata.get("feature_order"))
    if not isinstance(columns, list) or len(columns) != config.input_dim or len(set(columns)) != len(columns):
        raise ValueError("metadata must contain 59 unique ordered feature columns")
    if not isinstance(metadata.get("feature_fingerprint"), str) or not metadata["feature_fingerprint"]:
        raise ValueError("feature_fingerprint is required")
    order_hash = canonical_fingerprint(columns)
    if metadata.get("feature_order_sha256") not in (None, order_hash):
        raise ValueError("feature order does not match feature_order_sha256")
    if tuple(columns) != V6_FEATURE_COLUMNS:
        raise ValueError("feature columns must match the frozen V6 FEATURE_COLS order")
    return tuple(str(column) for column in columns)


@dataclass(frozen=True)
class StockHistory:
    dates: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    valid: np.ndarray


class PreparedDataIndex:
    """Compact shared index: per-stock arrays plus date -> current stock IDs."""
    def __init__(self, histories: Mapping[str, StockHistory], eligible_by_date: Mapping[str, tuple[str, ...]], calendar: Sequence[str]):
        calendar_set = set(calendar)
        if any(not set(map(str,h.dates)) <= calendar_set for h in histories.values()):
            raise ValueError('prepared rows outside calendar')
        self.histories = dict(histories)
        self.eligible_by_date = dict(eligible_by_date)
        self.calendar = np.asarray(calendar, dtype="datetime64[D]")
        self.search_operations = 0

    @classmethod
    def from_parquet(cls, path: str | Path, metadata: Mapping[str, Any], *,
                     parquet_reader: Callable[..., Any] | None = None) -> "PreparedDataIndex":
        import pandas as pd
        calendar, _ = validate_protocol(metadata)
        features = validate_feature_metadata(metadata, V7IntegratedConfig())
        p = DataProtocol(); reader = parquet_reader or pd.read_parquet
        frame = reader(Path(path), columns=[p.date_column, p.stock_id_column, *features, *p.label_columns, "observation_valid"]).copy()
        required = {p.date_column, p.stock_id_column, *features, *p.label_columns, "observation_valid"}
        if required - set(frame.columns):
            raise ValueError(f"prepared Parquet is missing columns: {sorted(required - set(frame.columns))}")
        if frame.empty: raise ValueError("BLOCK insufficient usable prepared data")
        if frame.observation_valid.isna().any() or not frame.observation_valid.isin([True, False]).all():
            raise ValueError("BLOCK ambiguous observation validity schema")
        parsed = pd.to_datetime(frame[p.date_column], errors="coerce")
        if parsed.isna().any(): raise ValueError("prepared Parquet contains invalid dates")
        frame[p.date_column] = parsed.dt.strftime("%Y-%m-%d")
        frame[p.stock_id_column] = validate_stock_ids(frame[p.stock_id_column])
        if frame.duplicated([p.date_column, p.stock_id_column]).any():
            raise ValueError("prepared Parquet contains duplicate Date/stock_id keys")
        frame = frame.sort_values([p.stock_id_column, p.date_column], kind="stable")
        observation_values = frame.observation_valid.to_numpy(bool)
        feature_values = frame[list(features)].to_numpy(dtype=np.float32)
        if not np.isfinite(feature_values[observation_values, :47]).all(): raise ValueError("active features must be finite")
        label_values = frame[list(p.label_columns)].to_numpy(dtype=np.float32)
        dates = frame[p.date_column].to_numpy(dtype="datetime64[D]")
        stocks = frame[p.stock_id_column].to_numpy(dtype=str)
        histories: dict[str, StockHistory] = {}
        boundaries = np.r_[0, np.flatnonzero(stocks[1:] != stocks[:-1]) + 1, len(stocks)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            stock_id = str(stocks[start])
            histories[stock_id] = StockHistory(dates[start:stop], feature_values[start:stop], label_values[start:stop], observation_values[start:stop])
        eligible: dict[str, list[str]] = {}
        for day, stock_id, valid in zip(frame[p.date_column], stocks, frame.observation_valid):
            if valid: eligible.setdefault(day, []).append(stock_id)
        return cls(histories, {day: tuple(sorted(ids)) for day, ids in eligible.items()}, calendar)

    def window(self, stock_id: str, target: str, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        history = self.histories[stock_id]
        self.search_operations += 1
        end = int(np.searchsorted(self.calendar, np.datetime64(target, 'D')))
        if end >= len(self.calendar) or self.calendar[end] != np.datetime64(target, 'D'):
            raise ValueError('target outside independent calendar')
        days = self.calendar[max(0,end-length+1):end+1]
        positions = np.searchsorted(history.dates, days)
        bounded = np.minimum(positions,len(history.dates)-1)
        present = (positions < len(history.dates)) & (history.dates[bounded] == days)
        valid = present & history.valid[bounded]
        values = np.zeros((len(days),59),np.float32)
        values[valid] = history.features[bounded[valid]]
        labels = history.labels[bounded[-1]] if valid[-1] else np.full(2,np.nan,np.float32)
        return values, labels, days >= history.dates[0], valid


class DailyCrossSectionDataset:
    """Materialize one 60-step cross-section from a shared compact index."""
    def __init__(self, feature_parquet: str | Path, dates: Sequence[str], metadata: Mapping[str, Any], *,
                 sequence_length: int = 60, parquet_reader: Callable[..., Any] | None = None,
                 graph_provider: Callable[[Sequence[str]], tuple[Tensor, Tensor]] | None = None,
                 prepared_index: PreparedDataIndex | None = None):
        validate_protocol(metadata)
        self.metadata = metadata
        self.path, self.dates = Path(feature_parquet), tuple(str(value) for value in dates)
        if not self.dates or list(self.dates) != sorted(self.dates) or len(set(self.dates)) != len(self.dates):
            raise ValueError("dataset dates must be non-empty, unique, and ordered")
        self.config = V7IntegratedConfig(); self.features = validate_feature_metadata(metadata, self.config)
        self.sequence_length, self._reader, self._graph_provider = int(sequence_length), parquet_reader, graph_provider
        self.prepared_index = prepared_index
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target, p = self.dates[index], DataProtocol()
        if self.prepared_index is None:
            self.prepared_index = PreparedDataIndex.from_parquet(self.path,
                self.metadata,
                parquet_reader=self._reader)
        stock_ids = self.prepared_index.eligible_by_date.get(target, ())
        if not stock_ids: raise ValueError(f"date {target} has no eligible current rows")
        x = torch.zeros((len(stock_ids), self.sequence_length, len(self.features)), dtype=torch.float32)
        mask = torch.zeros((len(stock_ids), self.sequence_length), dtype=torch.bool)
        observation_mask = torch.zeros_like(mask)
        labels = torch.empty((len(stock_ids), 2), dtype=torch.float32)
        for row, stock_id in enumerate(stock_ids):
            window, current_labels, history_mask, valid_mask = self.prepared_index.window(stock_id, target, self.sequence_length)
            values = torch.from_numpy(window)
            x[row, -len(values):] = values
            mask[row, -len(values):] = torch.from_numpy(history_mask)
            observation_mask[row, -len(values):] = torch.from_numpy(valid_mask)
            labels[row] = torch.from_numpy(current_labels)
        x = zero_macro_group(x, self.config)
        edges = self._graph_provider(stock_ids) if self._graph_provider else (
            torch.empty((2, 0), dtype=torch.long), torch.empty(0))
        return {"x": x, "padding_mask": mask, "observation_mask": observation_mask, "labels": labels, "stock_ids": stock_ids,
                "edge_index": edges[0], "edge_attr": edges[1], "Date": target}


class KnowledgeGraphCSR:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as arrays:
            required = {"stock_ids", "indptr", "indices", "weights"}
            if required - set(arrays.files): raise ValueError("CSR graph fields are incomplete")
            self.stock_ids = tuple(str(item) for item in arrays["stock_ids"])
            self.indptr, self.indices, self.weights = (np.asarray(arrays[name]).copy()
                                                        for name in ("indptr", "indices", "weights"))
        if len(set(self.stock_ids)) != len(self.stock_ids): raise ValueError("CSR stock IDs must be unique")
        if (self.indptr.ndim != 1 or len(self.indptr) != len(self.stock_ids) + 1
                or self.indptr[0] != 0 or self.indptr[-1] != len(self.indices)
                or (np.diff(self.indptr) < 0).any()):
            raise ValueError("CSR indptr is invalid")
        if (self.indices.ndim != 1 or self.weights.ndim != 1 or len(self.indices) != len(self.weights)
                or (self.indices < 0).any() or (self.indices >= len(self.stock_ids)).any()):
            raise ValueError("CSR indices/weights shapes or bounds are invalid")
        if not np.isfinite(self.weights).all(): raise ValueError("CSR weights must be finite")
        self.lookup = {stock_id: index for index, stock_id in enumerate(self.stock_ids)}

    def edges_for(self, stock_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
        local, sources, targets, weights = {s: i for i, s in enumerate(stock_ids)}, [], [], []
        for source_id, source_local in local.items():
            source = self.lookup.get(source_id)
            if source is None: continue
            for pos in range(int(self.indptr[source]), int(self.indptr[source + 1])):
                target_id = self.stock_ids[int(self.indices[pos])]
                if target_id in local:
                    sources.append(source_local); targets.append(local[target_id]); weights.append(self.weights[pos])
        return torch.tensor([sources, targets], dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


def convert_knowledge_graph(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Convert the genuine V2 edge-list NPZ to validated CSR without pickle."""
    with np.load(source, allow_pickle=False) as arrays:
        required = {"stock_ids", "edge_index", "edge_attr"}
        if required - set(arrays.files): raise ValueError("graph requires stock_ids, edge_index, edge_attr")
        stock_ids = np.asarray(arrays["stock_ids"])
        edges = np.asarray(arrays["edge_index"])
        weights = np.asarray(arrays["edge_attr"]).reshape(-1)
    if stock_ids.ndim != 1 or stock_ids.dtype.kind not in "US" or len(set(map(str, stock_ids))) != len(stock_ids):
        raise ValueError("graph stock_ids must be unique one-dimensional strings")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != weights.size:
        raise ValueError("graph edge_index/edge_attr shapes are invalid")
    if edges.dtype.kind not in "iu" or (edges < 0).any() or (edges >= len(stock_ids)).any():
        raise ValueError("graph edge bounds are invalid")
    if not np.isfinite(weights).all(): raise ValueError("graph weights must be finite")
    order = np.lexsort((edges[1], edges[0]))
    sources = edges[0, order].astype(np.int64, copy=False)
    indices = edges[1, order].astype(np.int64, copy=False)
    weights = weights[order].astype(np.float32, copy=False)
    indptr = np.empty(len(stock_ids) + 1, dtype=np.int64); indptr[0] = 0
    np.cumsum(np.bincount(sources, minlength=len(stock_ids)), out=indptr[1:])
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, stock_ids=stock_ids, indptr=indptr,
                        indices=indices, weights=weights)
    return {"nodes": len(stock_ids), "edges": int(weights.size),
            "source_sha256": file_sha256(source), "output_sha256": file_sha256(destination)}


def finite_feature_rows(frame, columns):
    valid = np.ones(len(frame), dtype=bool)
    for column in columns:
        valid &= np.isfinite(frame[column].to_numpy(dtype=np.float64))
    return valid


def check_prepared_frames(features_frame: Any, market_frame: Any, metadata: Mapping[str, Any],
                          splits: Mapping[str, Sequence[str]], *,
                          graph_stock_ids: Sequence[str]) -> dict[str, Any]:
    """Strict in-memory health checks used by prepare and data-check."""
    import pandas as pd
    validate_protocol(metadata)
    config = V7IntegratedConfig(); feature_columns = validate_feature_metadata(metadata, config)
    required = {"Date", "stock_id", "observation_valid", *feature_columns, "Alpha_5d", "Alpha_10d"}
    if required - set(features_frame.columns): raise ValueError("prepared feature columns are incomplete")
    if {"Date", "stock_id", "Close"} - set(market_frame.columns):
        raise ValueError("market prices require Date, stock_id, raw Close")
    if not isinstance(metadata.get("raw_source_sha256"), dict) or not metadata["raw_source_sha256"]:
        raise ValueError("raw-source SHA provenance is required")
    frame = features_frame.copy(deep=False)
    prices = market_frame[['Date','stock_id','Close']].copy()
    for value, name in ((frame, "features"), (prices, "market")):
        value["Date"] = pd.to_datetime(value["Date"], errors="coerce")
        if value["Date"].isna().any(): raise ValueError(f"{name} contains invalid dates")
        value["stock_id"] = validate_stock_ids(value["stock_id"])
        if value.duplicated(["Date", "stock_id"]).any(): raise ValueError(f"{name} has duplicate Date/stock_id")
        if any(not group["Date"].is_monotonic_increasing for _, group in value[["stock_id","Date"]].groupby("stock_id", sort=False)):
            raise ValueError(f"{name} is not chronological within stock")
    close = prices["Close"].to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("raw market Close must be finite and positive")
    separation = frame[["Date", "stock_id", "Close"]].merge(
        prices[["Date", "stock_id", "Close"]], on=["Date", "stock_id"], suffixes=("_feature", "_raw"))
    if separation.empty or np.allclose(separation["Close_feature"], separation["Close_raw"], equal_nan=False):
        raise ValueError("normalized feature Close must be distinct from raw economic Close")
    graph_ids = set(map(str, graph_stock_ids)); feature_ids = set(frame.loc[frame.observation_valid,"stock_id"])
    if not feature_ids <= graph_ids: raise ValueError("eligible stocks lack graph coverage")
    days, policy = validate_protocol(metadata)
    if not set(metadata['selected_universe']) <= graph_ids:
        raise QualityBlocked('BLOCK declared expected stocks lack graph coverage', 'GRAPH_UNIVERSE_INCOMPATIBLE')
    if not set(frame.stock_id) <= set(metadata['selected_universe']):
        raise QualityBlocked('BLOCK prepared stocks outside declared universe')
    if not set(frame.Date.dt.strftime('%Y-%m-%d')) <= set(days):
        raise QualityBlocked('BLOCK prepared dates outside calendar')
    effective = frame[['Date','stock_id','observation_valid']].copy()
    effective['observation_valid'] = (frame.observation_valid.eq(True)
        & finite_feature_rows(frame, feature_columns[:47])
        & frame.stock_id.isin(graph_ids))
    entries = coverage_entries(effective, days, metadata['expected_universe'], policy)
    if any(entry['severity'] == 'BLOCK' for entry in entries):
        error = QualityBlocked('BLOCK effective feature coverage outage', 'MAJOR_COVERAGE_OUTAGE')
        error.report['entries'] = entries
        error.report['summary_zh_TW'] = '\n'.join(f"{e['severity']}：有效特徵覆蓋；日期 {e['dates']}；分母 {e['denominator']}；門檻 {e['threshold']}" for e in entries)
        raise error
    if not finite_feature_rows(frame, feature_columns[:47])[frame.observation_valid].all():
        raise ValueError("active features must be finite")
    if metadata.get("quality_report", {}).get("blocking"): raise ValueError("BLOCK prepared quality report")
    validate_frozen_dates(splits)
    train_dates = set(pd.to_datetime(splits["train"]))
    train_rows = frame.loc[frame["Date"].isin(train_dates), ["Date","observation_valid","Alpha_5d","Alpha_10d"]]
    train_rows = train_rows[train_rows.observation_valid]
    counts = train_rows.assign(Alpha_5d=np.isfinite(train_rows.Alpha_5d), Alpha_10d=np.isfinite(train_rows.Alpha_10d)).groupby('Date')[['Alpha_5d','Alpha_10d']].sum()
    _, policy = validate_protocol(metadata)
    if int((counts >= 2).any(axis=1).sum()) < policy.minimum_usable_days:
        raise ValueError('BLOCK insufficient usable training data')
    return {"rows": len(frame), "stocks": frame["stock_id"].nunique(),
            "date_min": str(frame["Date"].min().date()), "date_max": str(frame["Date"].max().date()),
            "label_missing": {name: int(frame[name].isna().sum()) for name in ("Alpha_5d", "Alpha_10d")},
            "artifact_kind": metadata.get("artifact_kind", "unspecified")}


RAW_SOURCES = {
    "prices": "prices_raw.parquet", "inst": "institutional_raw.parquet",
    "margin": "margin_raw.parquet", "per": "per_raw.parquet",
    "securities": "securities_raw.parquet", "market_value": "market_value_raw.parquet",
    "daytrade": "daytrade_raw.parquet", "holdings": "holdings_raw.parquet",
    "revenue": "revenue_raw.parquet", "financials": "financials_raw.parquet",
    "balance_sheet": "balance_sheet_raw.parquet", "cashflow": "cashflow_raw.parquet",
    "macro": "macro_raw.parquet", "futures_inst": "futures_institutional_raw.parquet",
    "options_inst": "options_institutional_raw.parquet", "dividend": "dividend_raw.parquet",
    "foreign_shareholding": "foreign_shareholding_raw.parquet",
    "fear_greed": "fear_greed.parquet", "business_indicator": "business_indicator.parquet",
    "fed_rate": "fed_rate.parquet",
}

TRADING_CALENDAR_SOURCES = ("prices", "inst", "margin", "per", "securities",
                            "market_value", "daytrade", "foreign_shareholding")

# These point-in-time sources must retain observations preceding a restricted
# price sample. V6's as-of joins need that history to reproduce the full build.
CONTEXT_HISTORY_SOURCES = frozenset({
    "per", "market_value", "revenue", "financials", "balance_sheet", "cashflow", "dividend",
    "macro", "fear_greed", "business_indicator", "fed_rate",
})


def source_read_bounds(date_from: str | None, date_to: str | None) -> dict[str, tuple[str | None, str | None]]:
    return {name: (None if name in CONTEXT_HISTORY_SOURCES else date_from, date_to)
            for name in RAW_SOURCES}


def preparation_artifact_kind(diagnostic: bool, stock_ids: Sequence[str] | None,
                              date_from: str | None, date_to: str | None) -> str:
    restricted = bool(diagnostic or stock_ids or date_from or date_to)
    return ("diagnostic-restricted-not-performance-evidence" if restricted
            else "full-prepared-candidate")


def require_twii_coverage_for_active_rs(prices: Any, macro: Any) -> None:
    """Reject a selected price tail for which active TWII-relative features cannot exist."""
    import pandas as pd
    if macro is None or "Date" not in macro or len(macro) == 0:
        raise ValueError("active RS features require TWII macro coverage for selected prices")
    twii_columns = [column for column in macro.columns if "twii" in str(column).lower()]
    if not twii_columns:
        raise ValueError("active RS features require a TWII field in macro data")
    twii = macro[twii_columns].apply(pd.to_numeric, errors="coerce")
    valid_twii = (np.isfinite(twii) & (twii > 0)).any(axis=1)
    valid_dates = pd.to_datetime(macro.loc[valid_twii, "Date"], errors="coerce").dropna()
    price_dates = pd.to_datetime(prices["Date"], errors="coerce").dropna()
    if valid_dates.empty or price_dates.empty or not set(price_dates) <= set(valid_dates):
        coverage = None if valid_dates.empty else str(valid_dates.max().date())
        selected = None if price_dates.empty else str(price_dates.max().date())
        raise ValueError("active RS features cannot be formed: valid TWII macro coverage "
                         f"ends at {coverage}, before selected prices end at {selected}")


def _read_bounded_parquet(path: Path, stock_ids: Sequence[str] | None,
                          date_from: str | None, date_to: str | None) -> Any:
    """Arrow predicate pushdown that adapts string/timestamp and Date/date schemas."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, format="parquet")
    names = set(dataset.schema.names)
    date_column = next((name for name in ("Date", "date", "Week") if name in names), None)
    expression = None
    def add(term: Any) -> None:
        nonlocal expression
        expression = term if expression is None else expression & term
    if stock_ids and "stock_id" in names:
        add(ds.field("stock_id").isin([str(value) for value in stock_ids]))
    if date_column and (date_from or date_to):
        field_type = dataset.schema.field(date_column).type
        def scalar(value: str) -> Any:
            stamp = pd.Timestamp(value)
            if pa.types.is_string(field_type) or pa.types.is_large_string(field_type): return value
            if pa.types.is_date(field_type): return stamp.date()
            return stamp.to_pydatetime()
        if date_from: add(ds.field(date_column) >= scalar(date_from))
        if date_to: add(ds.field(date_column) <= scalar(date_to))
    frame = dataset.to_table(filter=expression).to_pandas()
    if date_column in ("date", "Week"): frame = frame.rename(columns={date_column: "Date"})
    if "Date" in frame: frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if "stock_id" in frame: frame["stock_id"] = validate_stock_ids(frame["stock_id"])
    return frame


def cross_source_trading_calendar(data: Mapping[str, Any]) -> list[str]:
    """Union dates from daily Taiwan-market sources; never infer the calendar from prices alone."""
    import pandas as pd
    dates: set[str] = set()
    for name in TRADING_CALENDAR_SOURCES:
        frame = data.get(name)
        if frame is None or "Date" not in frame or len(frame) == 0:
            continue
        parsed = pd.to_datetime(frame["Date"], errors="coerce")
        if parsed.isna().any(): raise ValueError(f"{name} has invalid trading-calendar dates")
        dates.update(parsed.dt.strftime("%Y-%m-%d"))
    if not dates: raise ValueError("cross-source trading calendar is empty")
    return sorted(dates)


def _source_health(data: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}; all_dates: set[str] = set()
    for name, frame in data.items():
        if frame is None: continue
        entry: dict[str, Any] = {"rows": len(frame)}
        if "Date" in frame:
            if frame["Date"].isna().any(): raise ValueError(f"{name} has invalid dates")
            dates = frame["Date"].dt.strftime("%Y-%m-%d")
            all_dates.update(dates.unique())
            entry.update(date_min=dates.min() if len(dates) else None,
                         date_max=dates.max() if len(dates) else None,
                         dates=int(dates.nunique()))
        keys = [column for column in ("Date", "stock_id") if column in frame]
        type_column = next((column for column in ("Type", "type", "institution_type") if column in frame), None)
        if type_column: keys.append(type_column)
        entry["duplicate_keys"] = int(frame.duplicated(keys).sum()) if keys else 0
        entry["missing_cells"] = int(frame.isna().sum().sum())
        report[name] = entry
    report["cross_source_calendar"] = {"dates": len(all_dates),
        "date_min": min(all_dates) if all_dates else None, "date_max": max(all_dates) if all_dates else None}
    return report


def audit_raw_source_calendars(paths: Mapping[str, Path], *, date_from: str | None = None,
                               date_to: str | None = None) -> dict[str, Any]:
    """Stream only date columns so large raw sources are never materialized for coverage audit."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    report: dict[str, Any] = {}; union: set[str] = set()
    for name, source in paths.items():
        path = Path(source)
        if not path.is_file():
            report[name] = {"present": False, "expected_path": path.name}
            continue
        dataset = ds.dataset(path, format="parquet")
        date_column = next((column for column in ("Date", "date", "Week")
                            if column in dataset.schema.names), None)
        if date_column is None:
            report[name] = {"present": True, "rows": int(dataset.count_rows()),
                            "dates": None, "date_column": None}
            continue
        field_type = dataset.schema.field(date_column).type
        def scalar(value: str) -> Any:
            stamp = pd.Timestamp(value)
            if pa.types.is_string(field_type) or pa.types.is_large_string(field_type): return value
            if pa.types.is_date(field_type): return stamp.date()
            return stamp.to_pydatetime()
        expression = None
        if date_from: expression = ds.field(date_column) >= scalar(date_from)
        if date_to:
            upper = ds.field(date_column) <= scalar(date_to)
            expression = upper if expression is None else expression & upper
        dates: set[str] = set(); rows = invalid = 0
        for batch in dataset.to_batches(columns=[date_column], filter=expression, batch_size=262_144):
            values = pd.to_datetime(batch.column(0).to_pandas(), errors="coerce")
            rows += len(values); invalid += int(values.isna().sum())
            dates.update(values.dropna().dt.strftime("%Y-%m-%d"))
        union.update(dates)
        report[name] = {"present": True, "rows": rows, "dates": len(dates),
                        "date_column": date_column, "invalid_dates": invalid,
                        "date_min": min(dates) if dates else None,
                        "date_max": max(dates) if dates else None}
    report["cross_source_calendar"] = {"dates": len(union),
        "date_min": min(union) if union else None, "date_max": max(union) if union else None}
    return report


def audit_selected_prices(prices: Any) -> dict[str, Any]:
    """Return a serializable, blocking quality report without repairing source rows."""
    import pandas as pd
    required = {"Date", "stock_id", "Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"prices are missing audit columns: {missing}")
    numeric = prices[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    invalid_dates = pd.to_datetime(prices["Date"], errors="coerce").isna()
    invalid_numeric = (~np.isfinite(numeric)).any(axis=1)
    invalid_values = (numeric["Close"] <= 0) | (numeric["Volume"] < 0)
    ohlc = ((numeric["Low"] > numeric[["Open", "Close"]].min(axis=1))
            | (numeric["High"] < numeric[["Open", "Close"]].max(axis=1))
            | (numeric["Low"] > numeric["High"]))
    blocking_rows = invalid_dates | invalid_numeric | invalid_values | ohlc
    affected = sorted(prices.loc[blocking_rows, "stock_id"].astype(str).unique())
    return {"rows": int(len(prices)), "invalid_dates": int(invalid_dates.sum()),
            "invalid_numeric_or_price_volume": int((invalid_numeric | invalid_values).sum()),
            "ohlc_contradictions": int(ohlc.sum()),
            "zero_volume_nonblocking": int((numeric["Volume"] == 0).sum()),
            "affected_stocks": affected, "blocking": bool(blocking_rows.any())}


def build_frozen_splits(frame: Any, market_calendar: Sequence[str],
                        explicit: Mapping[str, Sequence[str]] | None = None) -> dict[str, list[str]]:
    if explicit is not None:
        result = {name: list(values) for name, values in explicit.items()}; validate_frozen_dates(result, market_calendar)
        return result
    import pandas as pd
    p = DataProtocol(); calendar = sorted(set(map(str, market_calendar)))
    valid = frame.loc[frame.observation_valid, ["Date", "Alpha_5d", "Alpha_10d"]].copy()
    valid["Date"] = pd.to_datetime(valid["Date"]).dt.strftime("%Y-%m-%d")
    by_date = valid.groupby("Date")[["Alpha_5d", "Alpha_10d"]].apply(lambda value: (np.isfinite(value).sum() >= 2).any())
    usable = set(by_date[by_date].index)
    train_calendar = [day for day in calendar if p.train_start <= day <= p.train_cutoff]
    if len(train_calendar) <= p.total_purge_rows: raise ValueError("insufficient train calendar for 30-row purge")
    train = [day for day in train_calendar[:-p.total_purge_rows] if day in usable]
    validation = [day for day in calendar if p.train_cutoff < day <= p.validation_end and day in usable]
    result = {"train": train, "validation": validation}
    validate_frozen_dates(result, calendar)
    return result


def gap_safe_features(prices, data, calendar, build_features):
    import pandas as pd
    positions = {pd.Timestamp(day): i for i,day in enumerate(calendar)}
    pieces = []
    stock_tables = {name: {str(stock): group for stock,group in table.groupby('stock_id',sort=False)}
                    for name,table in data.items() if table is not None and 'stock_id' in table}
    for stock, history in prices.groupby('stock_id',sort=False):
        valid = history[history.observation_valid].sort_values('Date')
        offsets = valid.Date.map(positions).to_numpy()
        boundaries = np.r_[0,np.flatnonzero(np.diff(offsets)!=1)+1,len(valid)]
        for start,stop in zip(boundaries[:-1],boundaries[1:]):
            segment = valid.iloc[start:stop]
            if segment.empty: continue
            context = {}
            for name,table in data.items():
                if table is None: context[name] = None; continue
                table = stock_tables[name].get(str(stock),table.iloc[:0]) if name in stock_tables else table
                if 'Date' in table: table = table[table.Date <= segment.Date.max()]
                context[name] = table
            pieces.append(build_features(df_price=segment.copy(), df_inst=context["inst"], df_margin=context["margin"],
        df_per=context["per"], df_securities=context["securities"], df_market_value=context["market_value"],
        df_daytrade=context["daytrade"], df_holdings=context["holdings"], df_rev=context["revenue"],
        df_fin=context["financials"], df_balance_sheet=context["balance_sheet"], df_cashflow=context["cashflow"],
        df_macro=context["macro"], df_futures_inst=context["futures_inst"], df_options_inst=context["options_inst"],
        df_dividend=context["dividend"], df_foreign_shareholding=context["foreign_shareholding"],
        df_fear_greed=context["fear_greed"], df_business_indicator=context["business_indicator"],
        df_fed_rate=context["fed_rate"], fundamentals_v2=True, availability_flags=False))
    if not pieces: raise ValueError('BLOCK no valid price segments')
    return pd.concat(pieces,ignore_index=True)



def checkpointed_features(prices, paths, bounds, selected_ids, calendar, output_dir,
                          identity, build_features):
    """Load bounded stock batches; persist narrow unscaled frames before global scaling."""
    import gc
    import pandas as pd
    import pyarrow.dataset as ds
    from v7_integrated_prepare_cache import StockCache, progress
    from v7_integrated_point_in_time import align_point_in_time_features
    cache = StockCache(output_dir, identity)
    columns = ["Date", "stock_id", *V6_FEATURE_COLUMNS, "Alpha_5d", "Alpha_10d"]
    global_data, stock_paths = {}, {}
    for name, path in paths.items():
        if name == "prices":
            continue
        if not path.is_file():
            global_data[name] = None
        elif "stock_id" in ds.dataset(path, format="parquet").schema.names:
            stock_paths[name] = path
        else:
            global_data[name] = _read_bounded_parquet(path, None, *bounds[name])
    require_twii_coverage_for_active_rs(prices, global_data.get("macro"))
    health = _source_health(global_data)
    health["stock_source_audit_scope"] = "Disjoint stock batches; global sources audited once."
    stock_order = sorted(map(str, selected_ids))
    completed = 0
    reports = []
    files = []
    for offset in range(0, len(stock_order), 32):
        batch_ids = stock_order[offset:offset + 32]
        progress(output_dir, "load_stock_batch", completed=completed, total=len(stock_order),
                 first_stock=batch_ids[0], batch_size=len(batch_ids))
        batch = {name: _read_bounded_parquet(path, batch_ids, *bounds[name])
                 for name, path in stock_paths.items()}
        batch_health = _source_health(batch)
        health.setdefault("stock_batches", []).append({"stock_ids": batch_ids, "sources": batch_health})
        batch_prices = prices[prices.stock_id.astype(str).isin(batch_ids)]
        for stock in batch_ids:
            record = cache.valid(stock)
            if record is None:
                progress(output_dir, "stock_started", stock_id=stock,
                         completed=completed, total=len(stock_order))
                history = batch_prices[batch_prices.stock_id.astype(str) == stock]
                context = dict(global_data)
                context.update({name: table[table.stock_id.astype(str) == stock]
                                for name, table in batch.items()})
                context["prices"] = history
                if history.observation_valid.any():
                    # Drop wide helper intermediates after EACH contiguous segment.
                    def narrow_build(**kwargs):
                        return build_features(**kwargs)[columns].copy()
                    frame = gap_safe_features(history, context, calendar, narrow_build)
                else:
                    frame = pd.DataFrame(columns=columns)
                frame, alignment = align_point_in_time_features(
                    frame, context.get("revenue"), context.get("market_value"), calendar)
                record = cache.write(stock, frame, alignment)
                del frame, context, history
                event = "stock_saved"
            else:
                event = "stock_resumed"
            completed += 1
            reports.append(record["alignment"])
            if record["rows"]:
                files.append(cache.paths(stock)[0])
            progress(output_dir, event, stock_id=stock, completed=completed,
                     total=len(stock_order), rows=record["rows"])
        del batch, batch_prices
        gc.collect()
    if not files:
        raise ValueError("No valid feature segments")
    # All stock context tables are released before full-market cross-sectional work.
    del global_data
    gc.collect()
    alignment = dict(reports[0])
    constants = {"version", "market_cap_max_age_sessions"}
    for key in alignment:
        if key not in constants:
            alignment[key] = sum(report[key] for report in reports)
        elif any(report[key] != alignment[key] for report in reports):
            raise ValueError("Inconsistent checkpoint alignment policy")
    progress(output_dir, "assemble_narrow_features", stocks=len(files))
    # Arrow avoids retaining a list of full pandas frames plus their concatenation.
    table = ds.dataset([str(path) for path in files], format="parquet").to_table()
    engineered = table.to_pandas(split_blocks=True, self_destruct=True)
    progress(output_dir, "features_assembled", rows=len(engineered), columns=len(engineered.columns))
    return engineered, health, alignment


def prepare_artifacts(raw_dir: Path, output_dir: Path, *, v6_root: Path,
                      stock_ids: Sequence[str] | None = None, date_from: str | None = None,
                      date_to: str | None = None, diagnostic: bool = False,
                      frozen_splits: Mapping[str, Sequence[str]] | None = None, calendar_document: Mapping[str, Any] | None = None, policy: QualityPolicy = QualityPolicy()) -> dict[str, Any]:
    """Build V7 calendar-aware 59D artifacts through isolated protected helpers."""
    import os
    import pandas as pd
    raw_dir, output_dir, v6_root = raw_dir.resolve(), output_dir.resolve(), v6_root.resolve()
    if output_dir == raw_dir or raw_dir in output_dir.parents:
        raise ValueError("output directory must be new and outside the read-only raw directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir/".prepare-cache"/"identity.json").is_file():
        raise ValueError("Nonempty legacy output cannot resume; use a new output directory. Existing files preserved.")
    if diagnostic and (not stock_ids or not date_from or not date_to):
        raise ValueError("diagnostic preparation requires stock IDs and date bounds for predicate pushdown")
    artifact_kind = preparation_artifact_kind(diagnostic, stock_ids, date_from, date_to)
    paths = {name: raw_dir / filename for name, filename in RAW_SOURCES.items()}
    if not paths["prices"].is_file(): raise FileNotFoundError(paths["prices"])
    if calendar_document is None: raise QualityBlocked('BLOCK independent calendar input required')
    contract = protocol_metadata(calendar_document, policy)
    if stock_ids and set(map(str, stock_ids)) != set(contract['selected_universe']):
        raise QualityBlocked('BLOCK stock filters must equal declared calendar universe')
    selected_ids = contract['selected_universe']
    calendar_start, calendar_end = contract['trading_calendar'][0], contract['trading_calendar'][-1]
    if (date_from and date_from > calendar_start) or (date_to and date_to < calendar_end):
        raise QualityBlocked('BLOCK date filters omit declared calendar sessions')
    date_from, date_to = calendar_start, calendar_end
    output_dir.mkdir(parents=True, exist_ok=True)
    from v7_integrated_prepare_cache import progress, StockCache, atomic_json
    progress(output_dir, "verify_input_identity")
    identity = {
        "sources": {path.name: file_sha256(path) for path in paths.values() if path.is_file()},
        "calendar": canonical_fingerprint(calendar_document),
        "policy": canonical_fingerprint(contract),
        "code": {path.name: file_sha256(path) for path in sorted(_HERE.glob("v7_integrated*.py")) if not path.name.endswith("test.py")},
        "helpers": {path.name: file_sha256(path) for path in sorted((v6_root/"marketmamba/data").glob("*.py"))},
        "config": file_sha256(v6_root/"marketmamba/config.py"),
        "industry": file_sha256(raw_dir/"stock_info.parquet"),
        "graph": file_sha256(raw_dir/"knowledge_graph_v2.npz"),
        "frozen_splits": frozen_splits, "artifact_kind": artifact_kind,
    }
    StockCache(output_dir, identity)
    completion = output_dir / ".prepare-complete.json"
    if completion.is_file():
        saved = json.loads(completion.read_text())
        if all((output_dir/name).is_file() and file_sha256(output_dir/name) == sha
               for name, sha in saved["files"].items()):
            progress(output_dir, "prepared_reused", rows=saved["result"]["rows"])
            return {**saved["result"], "reused": True}
        progress(output_dir, "completed_output_changed_rebuilding_from_stock_cache")
    raw_prices = _read_bounded_parquet(paths["prices"], None, date_from, date_to)
    raw_prices, selection_audit = select_declared_prices(raw_prices, calendar_document)
    os.environ["MARKETMAMBA_DATA_ROOT"] = str(raw_dir.parent)
    if str(v6_root) not in sys.path: sys.path.insert(0, str(v6_root))
    prices = raw_prices.sort_values(
        ["stock_id", "Date"], kind="stable").reset_index(drop=True)
    if prices.duplicated(["Date", "stock_id"]).any(): raise ValueError("filtered prices have duplicate keys")
    try:
        if calendar_document is None: raise QualityBlocked('BLOCK independent calendar input required')
        market_calendar = calendar_contract(calendar_document)
        prices, price_quality = assess_prices(prices, calendar_document, policy, calendar_document.get('expected_universe'))
    except QualityBlocked as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/'data_health.json').write_text(json.dumps(error.report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'data_health.json').write_text(json.dumps(price_quality,ensure_ascii=False,indent=2),encoding='utf-8')
    if price_quality['blocking']: raise ValueError('BLOCK quality policy; see data_health.json')
    bounds = source_read_bounds(date_from, date_to)
    price_source_health = _source_health({"prices_raw_unfiltered": raw_prices, "prices": prices})
    del raw_prices
    from marketmamba.data.feature_engineer import build_features, clean_and_scale
    engineered, health, alignment_report = checkpointed_features(
        prices, paths, bounds, selected_ids, market_calendar, output_dir, identity, build_features)
    health.update(price_source_health)
    health["selected_price_quality"] = price_quality
    health["point_in_time_alignment"] = alignment_report
    progress(output_dir, "global_industry_scaling", rows=len(engineered))
    missing_before = {name: int(engineered[name].isna().sum()) for name in V6_FEATURE_COLUMNS}
    rows_before_clean = len(engineered)
    from v7_integrated_industry import clean_and_scale_industry_chunked, REVISION
    industry_source = raw_dir / "stock_info.parquet"
    if not industry_source.is_file():
        raise FileNotFoundError("BLOCK industry classification required: " + str(industry_source))
    industry_info = pd.read_parquet(industry_source)
    frame, industry_report = clean_and_scale_industry_chunked(engineered, industry_info, clean_and_scale, work_dir=output_dir/'.scaled-parts')
    del engineered
    industry_report["source_sha256"] = file_sha256(industry_source)
    health["industry_neutralization"] = industry_report
    print(json.dumps({"event": "industry_neutralization", **industry_report}, ensure_ascii=False), flush=True)
    labels = calendar_labels(prices, market_calendar, policy.interior_gap_invalidates_target)
    frame = frame.drop(columns=['Alpha_5d','Alpha_10d']).merge(labels,on=['Date','stock_id'],validate='one_to_one')
    frame = prices[['Date','stock_id','observation_valid']].merge(frame,on=['Date','stock_id'],how='left',validate='one_to_one')
    frame['observation_valid'] &= finite_feature_rows(frame, V6_FEATURE_COLUMNS[:47])
    frame = frame[["observation_valid", "Date", "stock_id", *V6_FEATURE_COLUMNS, "Alpha_5d", "Alpha_10d"]].sort_values(
        ["stock_id", "Date"], kind="stable").reset_index(drop=True)
    market = prices.loc[prices.observation_valid, ["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]].copy()
    market = market.sort_values(["stock_id", "Date"], kind="stable").reset_index(drop=True)
    del prices, labels
    p = DataProtocol()
    training_calendar = [day for day in market_calendar if p.train_start <= day <= p.train_cutoff]
    allowed_training = training_calendar[:-p.total_purge_rows]
    assess_training(frame[['Date','observation_valid','Alpha_5d','Alpha_10d']], list(frozen_splits['train']) if frozen_splits else allowed_training, policy, price_quality)
    (output_dir/'data_health.json').write_text(json.dumps(price_quality,ensure_ascii=False,indent=2),encoding='utf-8')
    if price_quality['blocking']: raise ValueError('BLOCK insufficient usable training data; see data_health.json')
    splits = build_frozen_splits(frame, market_calendar, frozen_splits)
    graph_source = raw_dir / "knowledge_graph_v2.npz"
    if not graph_source.is_file(): raise FileNotFoundError(graph_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_report = convert_knowledge_graph(graph_source, output_dir / "knowledge_graph_v2_csr.npz")
    source_hashes = {path.name: file_sha256(path) for path in paths.values() if path.is_file()}
    source_hashes[graph_source.name] = file_sha256(graph_source)
    source_hashes[industry_source.name] = file_sha256(industry_source)
    helper_sources = [v6_root / "marketmamba/data/feature_engineer.py",
                      v6_root / "marketmamba/config.py", v6_root / "marketmamba/data/hygiene.py"]
    metadata = {**protocol_metadata(calendar_document, policy), "quality_report":price_quality, "universe_selection_audit":selection_audit, "schema_version": 2, "feature_columns": list(V6_FEATURE_COLUMNS),
        "feature_order_sha256": canonical_fingerprint(list(V6_FEATURE_COLUMNS)),
        "feature_fingerprint": canonical_fingerprint({"order": V6_FEATURE_COLUMNS,
            "preprocessing": "build_features(fundamentals_v2=True,availability_flags=False)+v7-pit-alignment-v1+" + REVISION,
            "industry_source_sha256": file_sha256(industry_source), "min_industry_peers": 2}),
        "raw_source_sha256": source_hashes, "raw_source_health": health,
        "v6_helper_source_sha256": {str(path.relative_to(v6_root)): file_sha256(path) for path in helper_sources},
        "feature_health": {"rows_before_clean": rows_before_clean, "rows_after_clean": len(frame),
                           "missing_before_clean": missing_before,
                           "missing_after_clean": {name: int(frame[name].isna().sum()) for name in V6_FEATURE_COLUMNS},
                           "rs_source_macro_max": health.get("macro", {}).get("date_max")},
        "artifact_kind": artifact_kind,
        "expected_universe_fingerprint":canonical_fingerprint(calendar_document.get("expected_universe")),
        "point_in_time_alignment": alignment_report,
        "industry_neutralization": industry_report,
        "v7_industry_source_sha256": file_sha256(_HERE / "v7_integrated_industry.py"),
        "v7_alignment_source_sha256": file_sha256(_HERE / "v7_integrated_point_in_time.py"),
        "no_future_feature_information_claim": "Revenue actual publication uses next session; unknown dates retain explicit assumptions; industry uses a frozen classification snapshot (not PIT); other V6 PIT helpers reused, not fully vintage-certified",
        "market_prices": "separate unnormalized prices_raw Close; never feature Close"}
    try:
        with np.load(output_dir / "knowledge_graph_v2_csr.npz", allow_pickle=False) as graph:
            effective_entries = coverage_entries(frame[['Date','stock_id','observation_valid']], market_calendar, metadata['expected_universe'], policy)
            price_quality['entries'].extend(effective_entries)
            price_quality['blocking'] |= any(e['severity'] == 'BLOCK' for e in effective_entries)
            price_quality['summary_zh_TW'] += '\n' + '\n'.join(
                f"{e['severity']}：有效特徵覆蓋；日期 {e['dates']}；分母 {e['denominator']}；門檻 {e['threshold']}"
                for e in effective_entries)
            (output_dir/'data_health.json').write_text(json.dumps(price_quality,ensure_ascii=False,indent=2),encoding='utf-8')
            check_prepared_frames(frame, market, metadata, splits, graph_stock_ids=graph["stock_ids"])
    except QualityBlocked as error:
        (output_dir/'data_health.json').write_text(json.dumps(error.report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    feature_path, market_path = output_dir / "features_59.parquet", output_dir / "market_prices_raw.parquet"
    frame.to_parquet(feature_path, index=False); market.to_parquet(market_path, index=False)
    metadata["parquet_sha256"] = file_sha256(feature_path); metadata["market_sha256"] = file_sha256(market_path)
    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    split_doc = {"protocol": "V6 frozen two-set; validation is evaluation, not untouched test",
                 "splits": splits, "trading_calendar": market_calendar}
    (output_dir / "splits.json").write_text(json.dumps(split_doc, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "data_health.json").write_text(json.dumps({"quality_report": price_quality, "sources": health, "graph": graph_report}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result = {"status": "prepared", "artifact_kind": metadata["artifact_kind"], "rows": len(frame),
              "output_dir": str(output_dir), "splits": {k: len(v) for k, v in splits.items()}}
    final_files = ["features_59.parquet", "market_prices_raw.parquet", "feature_metadata.json",
                   "splits.json", "data_health.json", "knowledge_graph_v2_csr.npz"]
    atomic_json(completion, {"files": {name: file_sha256(output_dir/name) for name in final_files},
                             "result": result})
    progress(output_dir, "prepared_complete", rows=len(frame))
    return result



def _parse_dates(values: Sequence[str], name: str) -> list[date]:
    parsed = [date.fromisoformat(str(value)) for value in values]
    if not parsed or parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} dates must be non-empty, unique, and ordered")
    return parsed


def validate_frozen_dates(splits: Mapping[str, Sequence[str]], trading_calendar: Sequence[str] | None = None) -> None:
    p = DataProtocol()
    if set(splits) not in (set(p.required_splits), {*p.required_splits, "test"}):
        raise ValueError("train and validation are required; only an optional separate test is allowed")
    train, validation = (_parse_dates(splits[name], name) for name in p.required_splits)
    test = _parse_dates(splits["test"], "test") if "test" in splits else None
    start, cutoff, val_end = map(date.fromisoformat, (p.train_start, p.train_cutoff, p.validation_end))
    if train[0] < start or train[-1] > cutoff or validation[0] <= cutoff or validation[-1] > val_end:
        raise ValueError("dates violate frozen V6 split boundaries")
    if test is not None and test[0] <= validation[-1]:
        raise ValueError("optional test must follow validation")
    sets = [set(train), set(validation)] + ([set(test)] if test else [])
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("split date contamination detected")
    if trading_calendar is not None:
        calendar = _parse_dates(trading_calendar, "trading calendar"); positions = {d: i for i, d in enumerate(calendar)}
        boundaries = [(train[-1], validation[0])] + ([(validation[-1], test[0])] if test else [])
        for left, right in boundaries:
            if left not in positions or right not in positions or positions[right] - positions[left] <= p.total_purge_rows:
                raise ValueError("split boundary lacks horizon+embargo 30-trading-row purge")


def _centered_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True); sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = torch.cumsum(counts, 0); ranks = torch.empty_like(values)
    ranks[order] = torch.repeat_interleave((ends - counts + ends - 1).to(values.dtype) / 2, counts)
    return ranks - ranks.mean()


def rank_center_targets(labels: Tensor) -> Tensor:
    if labels.ndim != 2 or labels.shape[1] != 2: raise ValueError("labels must have shape (N, 2)")
    targets = torch.zeros_like(labels)
    for column in range(2):
        valid = torch.isfinite(labels[:, column])
        if valid.any(): targets[valid, column] = _centered_ranks(labels[valid, column])
    return targets


def _listnet(prediction: Tensor, target: Tensor) -> Tensor:
    return -(torch.softmax(target, 0) * torch.log_softmax(prediction, 0)).sum()


def short_loss(predictions: Tensor, labels: Tensor) -> Tensor | None:
    if not torch.isfinite(predictions).all(): raise ValueError('nonfinite predictions')
    targets = rank_center_targets(labels)
    terms = []
    for head, weight in ((0,1.),(1,.5)):
        valid = torch.isfinite(labels[:,head])
        if valid.any():
            terms.append(weight * (F.mse_loss(predictions[valid,head], targets[valid,head])
                         + .5 * _listnet(predictions[valid,head],targets[valid,head])))
    if not terms: return None
    return sum(terms)


def training_step(model, optimizer, scheduler, predictions, labels):
    loss = short_loss(predictions, labels)
    if loss is None: return None
    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('nonfinite gradients')
    optimizer.step(); scheduler.step()
    return float(loss.detach())


def _rank_ic(prediction: Tensor, label: Tensor) -> float:
    if not torch.isfinite(prediction).all(): raise ValueError("nonfinite predictions")
    valid = torch.isfinite(label)
    if int(valid.sum()) < 2: return float("nan")
    x, y = _centered_ranks(prediction[valid]), _centered_ranks(label[valid]); denominator = x.norm() * y.norm()
    return float((x @ y / denominator).item()) if float(denominator) > 0 else float("nan")


def rank_ic_by_horizon(predictions: Tensor, labels: Tensor) -> dict[str, float]:
    if predictions.shape != labels.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and labels must have shape (N, 2)")
    return {f"rank_ic_{h}d": _rank_ic(predictions[:, i], labels[:, i]) for i, h in enumerate((5, 10))}


def aggregate_validation_metrics(daily_metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average cross-sectional metrics across dates with explicit NaN semantics."""
    if not daily_metrics:
        raise ValueError("validation requires at least one date")
    names = ("rank_ic_5d", "rank_ic_10d")
    result: dict[str, float] = {}
    for name in names:
        values = np.asarray([row[name] for row in daily_metrics], dtype=np.float64)
        result[name] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        result[name + "_dates"] = int(np.isfinite(values).sum())
    return result


def compare_predictive_frames(candidate: Any, baseline: Any) -> dict[str, dict[str, float]]:
    """Compare candidate and frozen baseline on exactly matching held-out keys."""
    keys, score_columns, labels = ["Date", "stock_id"], ["Score_5d", "Score_10d"], ["Alpha_5d", "Alpha_10d"]
    required_candidate, required_baseline = set(keys + score_columns + labels), set(keys + score_columns)
    if required_candidate - set(candidate.columns) or required_baseline - set(baseline.columns):
        raise ValueError("predictive score files are missing required columns")
    if candidate.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise ValueError("predictive score keys must be unique")
    candidate_keys = set(map(tuple, candidate[keys].astype(str).to_numpy()))
    baseline_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    if candidate_keys != baseline_keys:
        raise ValueError("candidate and baseline must cover identical held-out dates/stocks")
    merged = candidate.merge(baseline[keys + score_columns], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one")
    result = {"v7_integrated_candidate": {}, "v2_kg_nomacro": {}}
    for day, frame in merged.groupby("Date", sort=True):
        del day
        target = torch.as_tensor(frame[labels].to_numpy(dtype=np.float32))
        for name, suffix in (("v7_integrated_candidate", "candidate"), ("v2_kg_nomacro", "baseline")):
            prediction = torch.as_tensor(frame[[f"{column}_{suffix}" for column in score_columns]].to_numpy(dtype=np.float32))
            metrics = rank_ic_by_horizon(prediction, target)
            for metric, value in metrics.items():
                result[name].setdefault(metric, []).append(value)
    return {name: {metric: float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                   for metric, values in metrics.items()} for name, metrics in result.items()}


def economic_f20_config() -> dict[str, Any]:
    return {"N": 50, "buffer": 1.5, "frequency": 20, "weighting": "equal"}


def run_economic_replay(scores_path: Path, market_data_path: Path, output_dir: Path) -> Any:
    forbidden_parts = {"production", "live", "results"}
    if forbidden_parts & {part.lower() for part in Path(output_dir).parts}:
        raise ValueError("economic replay requires an isolated non-live output directory")
    module = importlib.import_module("portfolio_lab"); runner = getattr(module, "run_config", None)
    if runner is None: raise RuntimeError("existing portfolio_lab.run_config is unavailable")
    import pandas as pd
    scores, market = pd.read_parquet(scores_path), pd.read_parquet(market_data_path)
    keys = ["Date", "stock_id"]
    if set(keys + ["Score_5d"]) - set(scores.columns): raise ValueError("scores require Date, stock_id, Score_5d")
    if set(keys + ["Close"]) - set(market.columns): raise ValueError("market requires Date, stock_id, Close")
    for frame in (scores, market):
        frame["Date"] = pd.to_datetime(frame["Date"]); frame["stock_id"] = validate_stock_ids(frame["stock_id"])
        if frame.duplicated(keys).any(): raise ValueError("duplicate Date/stock_id keys")
    if not np.isfinite(scores["Score_5d"].to_numpy(dtype=np.float64)).all():
        raise ValueError("scores must be finite where supplied")
    close = market["Close"].to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("market Close must be finite and positive")
    score_keys = set(map(tuple, scores[keys].to_numpy()))
    market_keys = set(map(tuple, market[keys].to_numpy()))
    if not score_keys <= market_keys: raise ValueError("market data is missing scored-stock prices")
    first, last = scores["Date"].min(), scores["Date"].max()
    market = market[(market["Date"] >= first) & (market["Date"] <= last)].copy()
    dates = pd.DatetimeIndex(sorted(market["Date"].unique()))
    stocks = sorted(scores["stock_id"].unique())
    aligned = market[market["stock_id"].isin(stocks)]
    px = aligned.pivot(index="Date", columns="stock_id", values="Close").reindex(index=dates, columns=stocks)
    if px.isna().all(axis=0).any(): raise ValueError("market Close coverage cannot be entirely missing for a stock")
    rank = scores.pivot(index="Date", columns="stock_id", values="Score_5d").reindex(index=dates, columns=stocks)
    # V6 portfolio_lab semantics: lexical stock columns plus stable first ranking.
    rank = rank.rank(axis=1, method="first", ascending=False)
    class SuppliedMarket:
        """Minimum Market-compatible view backed only by caller-supplied data."""
    mkt = SuppliedMarket(); mkt.dates = dates; mkt.stocks = stocks
    mkt.px_full = px; mkt.px = px; mkt.ret = px.ffill().pct_change().reindex(dates)
    mkt.liq_pct = pd.DataFrame(np.nan, index=dates, columns=stocks)
    mkt.dvol = mkt.liq_pct.copy(); mkt.vol = mkt.liq_pct.copy(); mkt.sector = {}
    mkt.at_limit_up = np.zeros(px.shape, dtype=bool); mkt.at_limit_down = np.zeros(px.shape, dtype=bool)
    mkt.under_disposal = np.zeros(px.shape, dtype=bool)
    date_positions = {day: i for i, day in enumerate(dates)}
    score_dates = list(pd.DatetimeIndex(sorted(scores["Date"].unique())))
    reb_idx = [date_positions[day] for day in score_dates if date_positions[day] % 20 == 0]
    raw = runner(mkt, rank, 50, 1.5, 20, None, cost_mult=1.0, weight="equal",
                 block_limit=False, block_disposal=False, reb_idx=reb_idx)
    daily = np.asarray(raw.pop("_daily"), dtype=np.float64)
    no_cost = runner(mkt, rank, 50, 1.5, 20, None, cost_mult=0.0, weight="equal",
                     block_limit=False, block_disposal=False, reb_idx=reb_idx)
    costs = np.asarray(no_cost["_daily"], dtype=np.float64) - daily
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": dates, "daily_return": daily, "transaction_cost": costs}).to_parquet(
        output_dir / "daily_returns.parquet", index=False)
    def json_value(value: Any) -> Any:
        value = value.item() if isinstance(value, np.generic) else value
        return None if isinstance(value, float) and not np.isfinite(value) else value
    summary = {key: json_value(value) for key, value in raw.items()}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return summary


def adamw_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def capture_rng_state(include_cuda: bool = False) -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if include_cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(model_state: Mapping[str, Any], optimizer_state: Mapping[str, Any], *, epoch: int,
                     step: int, provenance: Mapping[str, Any], batch_index: int = 0,
                     scheduler_state: Mapping[str, Any] | None = None,
                     rng_state: Mapping[str, Any] | None = None,
                     best_validation_metric: float | None = None,
                     validation_metrics: Mapping[str, float] | None = None, training_counters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not provenance: raise ValueError("checkpoint provenance must not be empty")
    return {"format": "marketmamba-v7-calendar-v1", "model_state": dict(model_state),
            "optimizer_state": dict(optimizer_state), "scheduler_state": dict(scheduler_state or {}),
            "rng_state": dict(rng_state or capture_rng_state()), "epoch": int(epoch),
            "batch_index": int(batch_index), "step": int(step), "provenance": dict(provenance),
            "best_validation_metric": best_validation_metric,
            "validation_metrics": dict(validation_metrics or {}), "training_counters":dict(training_counters or {})}


def save_checkpoint(checkpoint: Mapping[str, Any], destination: str | Path | BinaryIO) -> None:
    torch.save(dict(checkpoint), destination)


def load_checkpoint(source: str | Path | BinaryIO, *, expected_provenance: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "marketmamba-v7-calendar-v1": raise ValueError("unsupported checkpoint format")
    if not expected_provenance.get("protocol"): raise ValueError("checkpoint protocol fingerprint required")
    observed = checkpoint.get("provenance", {})
    differences = {key: (value, observed.get(key)) for key, value in expected_provenance.items() if observed.get(key) != value}
    if differences: raise ValueError(f"checkpoint provenance mismatch: {differences}")
    return checkpoint


def should_update(step: int, max_steps: int) -> bool:
    return int(step) < int(max_steps)


def normalize_resume_position(epoch: int, batch_index: int, batches_per_epoch: int) -> tuple[int, int]:
    if epoch < 0 or batch_index < 0 or batches_per_epoch < 1:
        raise ValueError("resume position and dataset size are invalid")
    extra_epochs, normalized_batch = divmod(batch_index, batches_per_epoch)
    return epoch + extra_epochs, normalized_batch


def resolve_training_budget(command: str, *, epochs: int, batches_per_epoch: int,
                            max_steps: int | None, smoke_steps: int) -> dict[str, int | bool]:
    if epochs < 1 or batches_per_epoch < 1 or smoke_steps < 1:
        raise ValueError("training budget values must be positive")
    computed = epochs * batches_per_epoch
    if command == "smoke":
        return {"epochs": 1, "max_steps": min(smoke_steps, batches_per_epoch),
                "computed_steps": batches_per_epoch, "explicit_cap": True}
    cap = computed if max_steps is None else min(max_steps, computed)
    if cap < 1: raise ValueError("max_steps must be positive when supplied")
    return {"epochs": epochs, "max_steps": cap, "computed_steps": computed,
            "explicit_cap": max_steps is not None}


def training_event_due(step: int, *, checkpoint_interval: int, progress_interval: int,
                       epoch_end: bool = False, stopping: bool = False) -> dict[str, bool]:
    if checkpoint_interval < 1 or progress_interval < 1:
        raise ValueError("checkpoint and progress intervals must be positive")
    boundary = bool(epoch_end or stopping)
    return {
        "checkpoint": boundary or step % checkpoint_interval == 0,
        "progress": boundary or step % progress_interval == 0,
    }


def _emit_training_progress(*, step: int, max_steps: int, epoch: int, batch_index: int,
                            batches_per_epoch: int, loss: float) -> None:
    print(json.dumps({"event": "training_progress", "step": step, "max_steps": max_steps,
                      "epoch": epoch, "batch_index": batch_index,
                      "batches_per_epoch": batches_per_epoch, "loss": loss}, sort_keys=True),
          file=sys.stderr, flush=True)


def official_ssd_callable():
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as exc:
        raise RuntimeError("official mamba-ssm SSD runtime is required; no fallback is enabled") from exc
    return mamba_chunk_scan_combined


def _runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torch-geometric", "mamba-ssm", "triton"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = "missing"
    return versions


def build_provenance(args: argparse.Namespace, config: V7IntegratedConfig,
                     metadata: Mapping[str, Any], splits: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(metadata)
    sources = [Path(__file__), _HERE / "v7_integrated_model.py", Path(sys.modules["v7_integrated_config"].__file__), _HERE / "v7_integrated_data_quality.py"]
    return {"protocol": metadata["protocol_fingerprint"], "config": config.sha256(), "model_source": canonical_fingerprint({p.name: file_sha256(p) for p in sources}),
            "runtime": canonical_fingerprint(_runtime_versions()), "features": str(metadata["feature_fingerprint"]),
            "kg": file_sha256(args.kg), "data": file_sha256(args.feature_parquet),
            "split": canonical_fingerprint(splits), "seed": args.seed, "candidate": "v7_integrated",
            "predictive_baseline": config.predictive_baseline, "economic_baseline": config.economic_baseline}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_validation(model: nn.Module, validation: DailyCrossSectionDataset,
                         device: torch.device) -> dict[str, float]:
    daily_metrics: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for sample in validation:
            prepared = prepare_stock_batch(sample["x"], sample["edge_index"], sample["edge_attr"],
                                           sample["padding_mask"], sample["stock_ids"], sample["observation_mask"])
            prediction = model(prepared.x.to(device), prepared.edge_index.to(device),
                               prepared.edge_attr.to(device), prepared.padding_mask.to(device), prepared.observation_mask.to(device)).cpu()
            labels = sample["labels"][prepared.canonical_original_indices]
            daily_metrics.append(rank_ic_by_horizon(prediction, labels))
    model.train()
    return aggregate_validation_metrics(daily_metrics)


def _run_training(args: argparse.Namespace, config: V7IntegratedConfig, metadata: Mapping[str, Any],
                  splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    graph = KnowledgeGraphCSR(args.kg)
    shared_index = PreparedDataIndex.from_parquet(args.feature_parquet, metadata)
    dataset = DailyCrossSectionDataset(args.feature_parquet, splits["train"], metadata,
        sequence_length=config.sequence_length, graph_provider=graph.edges_for, prepared_index=shared_index)
    validation = DailyCrossSectionDataset(args.feature_parquet, splits["validation"], metadata,
        sequence_length=config.sequence_length, graph_provider=graph.edges_for, prepared_index=shared_index)
    set_reproducible_seed(args.seed); device = torch.device(args.device)
    model = MarketMambaV7Integrated(config, ssd_fn=official_ssd_callable()).to(device)
    optimizer = torch.optim.AdamW(adamw_parameter_groups(model, args.weight_decay), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    provenance = build_provenance(args, config, metadata, splits); epoch = batch_index = step = 0
    best_metric, last_metrics = -float("inf"), {}
    if args.resume:
        checkpoint = load_checkpoint(args.resume, expected_provenance=provenance)
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        for optimizer_values in optimizer.state.values():
            for key, value in optimizer_values.items():
                if isinstance(value, Tensor): optimizer_values[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler_state"]); restore_rng_state(checkpoint["rng_state"])
        epoch, batch_index, step = checkpoint["epoch"], checkpoint["batch_index"], checkpoint["step"]
        epoch, batch_index = normalize_resume_position(epoch, batch_index, len(dataset))
        resumed_counters = checkpoint.get("training_counters", {})
        best_metric = checkpoint.get("best_validation_metric")
        best_metric = -float("inf") if best_metric is None else float(best_metric)
        last_metrics = dict(checkpoint.get("validation_metrics", {}))
    budget = resolve_training_budget(args.command, epochs=args.epochs, batches_per_epoch=len(dataset),
                                     max_steps=args.max_steps, smoke_steps=args.smoke_steps)
    if args.checkpoint_interval < 1 or args.progress_interval < 1:
        raise ValueError("checkpoint and progress intervals must be positive")
    max_steps = int(budget["max_steps"])
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_record = {"observed_packages": _runtime_versions(), "supplied_runtime":
                      (load_runtime_metadata(args.runtime_metadata).__dict__ if args.runtime_metadata else None),
                      "provenance": provenance}
    args.checkpoint.with_suffix(args.checkpoint.suffix + ".runtime.json").write_text(
        json.dumps(runtime_record, indent=2, sort_keys=True), encoding="utf-8")
    best_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".best")
    if not should_update(step, max_steps):
        last_metrics = _evaluate_validation(model, validation, device)
        return {"status": "already_complete", "step": step, "actual_epochs": epoch,
                "budget": budget, "train_dates": len(dataset), "validation_dates": len(validation),
                "validation_metrics": last_metrics, "provenance": provenance}
    stop_reason = "epoch_limit"
    last_loss = float("nan")
    skipped_batches = resumed_counters.get("skipped_batches",0) if args.resume else 0
    head_target_counts = resumed_counters.get("head_target_counts",[0,0]) if args.resume else [0,0]
    completed_epochs = epoch
    for current_epoch in range(epoch, int(budget["epochs"])):
        for current_batch in range(batch_index if current_epoch == epoch else 0, len(dataset)):
            sample = dataset[current_batch]
            prepared = prepare_stock_batch(sample["x"], sample["edge_index"], sample["edge_attr"],
                                           sample["padding_mask"], sample["stock_ids"], sample["observation_mask"])
            labels = sample["labels"][prepared.canonical_original_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(prepared.x.to(device), prepared.edge_index.to(device),
                                prepared.edge_attr.to(device), prepared.padding_mask.to(device), prepared.observation_mask.to(device))
            head_target_counts = [head_target_counts[i]+int(torch.isfinite(labels[:,i]).sum()) for i in range(2)]
            outcome = training_step(model, optimizer, scheduler, predictions, labels)
            if outcome is None:
                skipped_batches += 1
                continue
            step += 1
            last_loss = outcome
            epoch_end = current_batch + 1 == len(dataset)
            stopping = not should_update(step, max_steps)
            events = training_event_due(step, checkpoint_interval=args.checkpoint_interval,
                                        progress_interval=args.progress_interval,
                                        epoch_end=epoch_end, stopping=stopping)
            if events["progress"]:
                _emit_training_progress(step=step, max_steps=max_steps, epoch=current_epoch,
                    batch_index=current_batch + 1, batches_per_epoch=len(dataset), loss=last_loss)
            # Epoch/end/cap checkpoints are written after validation below. Periodic
            # mid-epoch checkpoints carry the exact next batch and current RNG state.
            if events["checkpoint"] and not epoch_end:
                save_checkpoint(build_checkpoint(model.state_dict(), optimizer.state_dict(), epoch=current_epoch,
                    batch_index=current_batch + 1, step=step, provenance=provenance,
                    scheduler_state=scheduler.state_dict(),
                    rng_state=capture_rng_state(str(device).startswith("cuda")),
                    best_validation_metric=best_metric, validation_metrics=last_metrics, training_counters={"skipped_batches": skipped_batches, "head_target_counts":head_target_counts}), args.checkpoint)
            if stopping:
                final_epoch = epoch_end and current_epoch + 1 == int(budget["epochs"])
                stop_reason = "epoch_limit" if final_epoch else "step_limit"
                break
        epoch_complete = current_batch + 1 == len(dataset)
        resume_epoch = current_epoch + 1 if epoch_complete else current_epoch
        resume_batch = 0 if epoch_complete else current_batch + 1
        completed_epochs = current_epoch + int(epoch_complete)
        last_metrics = _evaluate_validation(model, validation, device)
        if step == 0: raise RuntimeError(f"no-target whole run: no optimizer updates; skipped_batches={skipped_batches}")
        selection = last_metrics["rank_ic_5d"]
        if not np.isfinite(selection): selection = last_metrics["rank_ic_10d"]
        if not np.isfinite(selection):
            raise RuntimeError("validation produced no finite per-date rank_ic_5d; checkpoint selection is impossible")
        if selection > best_metric:
            best_metric = selection
            save_checkpoint(build_checkpoint(model.state_dict(), optimizer.state_dict(), epoch=resume_epoch,
                batch_index=resume_batch, step=step, provenance=provenance, scheduler_state=scheduler.state_dict(),
                rng_state=capture_rng_state(str(device).startswith("cuda")), best_validation_metric=best_metric,
                validation_metrics=last_metrics, training_counters={"skipped_batches":skipped_batches,"head_target_counts":head_target_counts}), best_path)
        save_checkpoint(build_checkpoint(model.state_dict(), optimizer.state_dict(), epoch=resume_epoch,
            batch_index=resume_batch, step=step, provenance=provenance, scheduler_state=scheduler.state_dict(),
            rng_state=capture_rng_state(str(device).startswith("cuda")), best_validation_metric=best_metric,
            validation_metrics=last_metrics, training_counters={"skipped_batches":skipped_batches,"head_target_counts":head_target_counts}), args.checkpoint)
        if stop_reason == "step_limit":
            break
    if step == 0: raise RuntimeError("no-target whole run: no optimizer updates")
    return {"status": stop_reason, "step": step, "actual_epochs": completed_epochs,
            "loss": last_loss, "budget": budget, "skipped_batches": skipped_batches, "head_target_counts": head_target_counts,
            "train_dates": len(dataset), "validation_dates": len(validation),
            "validation_metrics": last_metrics, "best_checkpoint": str(best_path), "provenance": provenance}


def _run_forecast(args: argparse.Namespace, config: V7IntegratedConfig, metadata: Mapping[str, Any],
                  splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    import pandas as pd
    graph = KnowledgeGraphCSR(args.kg)
    if args.split not in splits: raise ValueError(f"requested split {args.split!r} is absent")
    dataset = DailyCrossSectionDataset(args.feature_parquet, splits[args.split], metadata,
        sequence_length=config.sequence_length, graph_provider=graph.edges_for)
    provenance = build_provenance(args, config, metadata, splits)
    checkpoint = load_checkpoint(args.checkpoint, expected_provenance=provenance)
    model = MarketMambaV7Integrated(config, ssd_fn=official_ssd_callable()).to(args.device)
    model.load_state_dict(checkpoint["model_state"]); model.eval(); rows = []
    with torch.no_grad():
        for sample in dataset:
            prepared = prepare_stock_batch(sample["x"], sample["edge_index"], sample["edge_attr"],
                                           sample["padding_mask"], sample["stock_ids"], sample["observation_mask"])
            scores = model(prepared.x.to(args.device), prepared.edge_index.to(args.device),
                           prepared.edge_attr.to(args.device), prepared.padding_mask.to(args.device), prepared.observation_mask.to(args.device)).cpu()
            labels = sample["labels"][prepared.canonical_original_indices]
            rows.extend({"Date": sample["Date"], "stock_id": stock_id, "Score_5d": float(scores[i, 0]),
                         "Score_10d": float(scores[i, 1]), "Alpha_5d": float(labels[i, 0]),
                         "Alpha_10d": float(labels[i, 1])} for i, stock_id in enumerate(prepared.stock_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True); destination = args.output_dir / "v7_integrated_scores.parquet"
    pd.DataFrame(rows).to_parquet(destination, index=False)
    return {"status": "forecasted", "scores": str(destination), "rows": len(rows), "provenance": provenance}


def run_data_check(feature_parquet: Path, feature_metadata: Path, splits_path: Path,
                   kg_path: Path, market_path: Path) -> dict[str, Any]:
    import pandas as pd
    metadata, split_document = _load_json(feature_metadata), _load_json(splits_path)
    validate_protocol(metadata)
    if split_document.get("trading_calendar") != metadata["trading_calendar"]:
        raise ValueError("split calendar fingerprint mismatch")
    if metadata.get("parquet_sha256") != file_sha256(feature_parquet):
        raise ValueError("prepared Parquet fingerprint mismatch")
    if metadata.get("market_sha256") != file_sha256(market_path):
        raise ValueError("raw market-price fingerprint mismatch")
    with np.load(kg_path, allow_pickle=False) as graph:
        report = check_prepared_frames(pd.read_parquet(feature_parquet), pd.read_parquet(market_path),
            metadata, split_document["splits"], graph_stock_ids=graph["stock_ids"])
    validate_frozen_dates(split_document["splits"], split_document["trading_calendar"])
    return {"status": "data-check-passed", "quality_report": metadata["quality_report"], **report}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare":
        explicit = _load_json(args.frozen_splits)["splits"] if args.frozen_splits else None
        return prepare_artifacts(args.raw_dir, args.output_dir, v6_root=args.v6_root,
            stock_ids=args.stock_ids, date_from=args.date_from, date_to=args.date_to,
            diagnostic=args.diagnostic, frozen_splits=explicit, calendar_document=_load_json(args.calendar) if args.calendar else None,
            policy=QualityPolicy(args.major_outage_fraction,args.minimum_usable_days))
    if args.command == "data-check":
        return run_data_check(args.feature_parquet, args.feature_metadata, args.splits,
                              args.kg, args.market_data)
    runtime = load_runtime_metadata(args.runtime_metadata)
    if args.command == "preflight": return preflight(args.manifest, runtime)
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs and max_steps must be positive")
    config = V7IntegratedConfig(); config.validate()
    metadata, split_document = _load_json(args.feature_metadata), _load_json(args.splits)
    validate_protocol(metadata)
    validate_feature_metadata(metadata, config); splits = split_document["splits"]
    if "trading_calendar" not in split_document:
        raise ValueError("split document must include the trading calendar for 30-row purge verification")
    if split_document["trading_calendar"] != metadata["trading_calendar"]:
        raise ValueError("split calendar fingerprint mismatch")
    validate_frozen_dates(splits, split_document["trading_calendar"])
    if metadata.get("parquet_sha256") != file_sha256(args.feature_parquet):
        raise ValueError("prepared Parquet fingerprint mismatch")
    if args.command in ("smoke", "train"):
        flight = preflight(args.manifest, runtime)
        if flight["missing_packages"]:
            raise RuntimeError(f"runtime dependencies are not installed: {flight['missing_packages']}")
        return _run_training(args, config, metadata, splits)
    if args.command == "forecast":
        flight = preflight(args.manifest, runtime)
        if flight["missing_packages"]:
            raise RuntimeError(f"runtime dependencies are not installed: {flight['missing_packages']}")
        if not args.checkpoint or not args.output_dir: raise ValueError("forecast requires checkpoint and output directory")
        return _run_forecast(args, config, metadata, splits)
    if not args.candidate_scores or not args.baseline_scores or not args.market_data or not args.output_dir:
        raise ValueError("evaluate requires candidate/baseline scores, market data, and explicit output directory")
    import pandas as pd
    predictive = compare_predictive_frames(pd.read_parquet(args.candidate_scores), pd.read_parquet(args.baseline_scores))
    candidate_economic = run_economic_replay(args.candidate_scores, args.market_data, args.output_dir / "candidate_f20")
    baseline_economic = run_economic_replay(args.baseline_scores, args.market_data, args.output_dir / "v2_kg_nomacro_f20")
    return {"status": "evaluated", "predictive": predictive, "economic_f20": economic_f20_config(),
            "economic": {"v7_integrated_candidate": candidate_economic,
                         "v2_kg_nomacro_f20": baseline_economic}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "data-check", "preflight", "smoke", "train", "forecast", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--runtime-metadata", type=Path)
    for name in ("feature-parquet", "feature-metadata", "splits", "kg", "candidate-scores", "baseline-scores", "market-data",
                 "output-dir", "checkpoint", "resume"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="optional explicit cap; omitted train runs all epochs")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="save an exact resumable mid-epoch checkpoint every N updates")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="emit a JSON progress record to stderr every N updates")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--v6-root", type=Path, default=_HERE.parent)
    parser.add_argument("--stock-ids", nargs="+")
    parser.add_argument("--date-from"); parser.add_argument("--date-to")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--frozen-splits", type=Path)
    parser.add_argument("--calendar", type=Path, help="independent calendar/provenance/expected_universe JSON")
    parser.add_argument("--major-outage-fraction", type=float, default=.5)
    parser.add_argument("--minimum-usable-days", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.raw_dir is None or args.output_dir is None: raise SystemExit("prepare requires raw-dir and new output-dir")
        print(json.dumps(run(args), indent=2, sort_keys=True, default=str)); return 0
    if args.command == "data-check":
        required_check = (args.feature_parquet, args.feature_metadata, args.splits, args.kg, args.market_data)
        if any(value is None for value in required_check):
            raise SystemExit("data-check requires feature-parquet, feature-metadata, splits, kg, and market-data")
        print(json.dumps(run(args), indent=2, sort_keys=True, default=str)); return 0
    required = ("feature_parquet", "feature_metadata", "splits", "kg")
    if args.command != "preflight" and any(getattr(args, name) is None for name in required):
        raise SystemExit("feature-parquet, feature-metadata, splits, and kg are required")
    if args.command in ("smoke", "train") and args.checkpoint is None: raise SystemExit("checkpoint is required")
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
