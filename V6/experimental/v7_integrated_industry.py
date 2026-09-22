"""V7 industry-neutral features using explicit, versioned classification input."""
from __future__ import annotations
import numpy as np
import pandas as pd
from marketmamba.data.feature_spec import resolve_sector, NON_INDUSTRY_LABELS, NEUTRALIZE_EXCLUDE
from v7_integrated_config import V6_FEATURE_COLUMNS

REVISION = "v7-industry-neutral-v1"

def industry_mapping(info):
    if info is None or info.empty:
        raise ValueError("BLOCK industry classification source is empty")
    # resolve_sector is the existing canonical alias / fine-category resolver.
    mapping = resolve_sector(info)
    mapping = mapping[~mapping.sector.isin(NON_INDUSTRY_LABELS)].copy()
    if mapping.empty:
        raise ValueError("BLOCK no usable industry classifications")
    # Ambiguous equal-priority labels must not depend on input row ordering.
    other = resolve_sector(info.iloc[::-1])
    joined = mapping.merge(other, on="stock_id", suffixes=("", "_reverse"))
    conflicts = joined.loc[joined.sector != joined.sector_reverse, "stock_id"].tolist()
    if conflicts:
        raise ValueError("BLOCK ambiguous industry classification: " + ", ".join(conflicts[:20]))
    return mapping

def clean_and_scale_industry(frame, info, scale_fn, min_peers=2):
    """Demean eligible factors per day/industry, retain unknown/thin groups.

    The V6 scaler first winsorizes and applies a daily affine z transform.
    Subtracting industry means cancels that affine offset. A final common daily
    scale gives the residuals a comparable scale without restoring industry means.
    Original missing cells are excluded from peer means and filled only afterward.
    """
    if min_peers < 2:
        raise ValueError("min_peers must be at least two")
    mapping = industry_mapping(info)
    out = scale_fn(frame, macro_norm="ts", neutralize="none")
    if not out.index.is_unique or not frame.index.is_unique:
        raise ValueError("non-unique feature index")
    sector = out.stock_id.astype(str).map(dict(zip(mapping.stock_id, mapping.sector)))
    cols = [c for c in V6_FEATURE_COLUMNS[:47] if c not in NEUTRALIZE_EXCLUDE]
    values = out[cols].where(frame.loc[out.index, cols].notna())
    groups = values.groupby([out.Date, sector], sort=False, dropna=True)
    counts = groups.transform("count")
    supported = counts.ge(min_peers) & values.notna()
    residual = values.where(~supported, values - groups.transform("mean"))
    daily_scale = residual.groupby(out.Date, sort=False).transform("std")
    final = residual / (daily_scale + 1e-9)
    # A one-observation day has no scale; use neutral zero as the V6 scaler does.
    out[cols] = final.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    means = final.where(supported).groupby([out.Date, sector], sort=False).mean()
    check = means.abs().to_numpy()
    maximum = float(np.nanmax(check)) if np.isfinite(check).any() else 0.0
    if maximum > 1e-7:
        raise ValueError("BLOCK industry residual means are not near zero")
    report = {
        "version": REVISION, "mode": "industry", "min_valid_peers_per_feature": min_peers,
        "neutralized_columns": cols, "excluded_columns": [c for c in V6_FEATURE_COLUMNS if c not in cols],
        "classification_policy": "frozen accumulated stock_info snapshot; not historical point-in-time classification",
        "rows": len(out), "classified_rows": int(sector.notna().sum()),
        "unknown_rows": int(sector.isna().sum()),
        "unknown_stock_ids": sorted(out.loc[sector.isna(), "stock_id"].astype(str).unique().tolist()),
        "classified_fraction": float(sector.notna().mean()) if len(out) else 0.0,
        "neutralized_cells": int(supported.to_numpy().sum()),
        "thin_group_cells": int((counts.lt(min_peers) & values.notna()).to_numpy().sum()),
        "post_scale_max_abs_supported_industry_mean": maximum,
        "unknown_or_thin_policy": "retain daily standardized factor, then common residual scaling; never discard stock",
    }
    return out, report

def clean_and_scale_industry_chunked(frame, info, scale_fn, *, work_dir, sessions_per_chunk=32):
    """Keep full daily cross sections; compute macro history once before date chunks."""
    import gc
    from pathlib import Path
    import pyarrow.dataset as ds
    from marketmamba.data.feature_engineer import macro_ts_zscore
    from v7_integrated_prepare_cache import progress
    if sessions_per_chunk < 1:
        raise ValueError("sessions_per_chunk must be positive")
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    count = np.zeros(len(frame), dtype=np.int16)
    for col in V6_FEATURE_COLUMNS:
        count += frame[col].notna().to_numpy()
    eligible = count >= int(.7 * len(V6_FEATURE_COLUMNS))
    macro = {}
    for col in V6_FEATURE_COLUMNS[47:]:
        daily = frame.loc[eligible, ["Date", col]].groupby("Date")[col].first().sort_index()
        macro[col] = macro_ts_zscore(daily).fillna(0.0)
    dates = sorted(frame.Date.unique())
    files, reports = [], []
    for offset in range(0, len(dates), sessions_per_chunk):
        chosen = dates[offset:offset+sessions_per_chunk]
        mask = frame.Date.isin(chosen).to_numpy()
        if not (mask & eligible).any():
            continue
        sub = frame.loc[mask].copy()
        out, report = clean_and_scale_industry(sub, info, scale_fn)
        for col, series in macro.items():
            out[col] = out.Date.map(series).fillna(0.0)
        path = directory / ("%06d.parquet" % offset)
        tmp = path.with_suffix(".tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(path)
        files.append(str(path)); reports.append(report)
        progress(directory.parent, "date_chunk_saved",
                 through=str(pd.Timestamp(chosen[-1]).date()),
                 dates_completed=min(offset+sessions_per_chunk,len(dates)), total_dates=len(dates),
                 rows=len(out))
        del sub, out
        gc.collect()
    if not files:
        raise ValueError("No usable rows after cleaning")
    report = dict(reports[0])
    for key in ["rows", "classified_rows", "unknown_rows", "neutralized_cells", "thin_group_cells"]:
        report[key] = sum(item[key] for item in reports)
    report["unknown_stock_ids"] = sorted(set().union(*(set(item["unknown_stock_ids"]) for item in reports)))
    report["classified_fraction"] = report["classified_rows"] / report["rows"]
    report["post_scale_max_abs_supported_industry_mean"] = max(
        item["post_scale_max_abs_supported_industry_mean"] for item in reports)
    table = ds.dataset(files, format="parquet").to_table()
    return table.to_pandas(split_blocks=True, self_destruct=True), report
