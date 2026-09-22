"""V7 point-in-time alignment; protected V6 helpers remain unchanged."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _revenue_events(revenue, calendar):
    counters = {"revenue_assumed_release_rows": 0, "revenue_invalid_rows": 0}
    if revenue is None or revenue.empty:
        return {}, counters
    rev = revenue.copy()
    rev["Date"] = pd.to_datetime(rev["Date"])
    periods = (pd.PeriodIndex(year=rev["revenue_year"].astype(int), month=rev["revenue_month"].astype(int), freq="M")
               if {"revenue_year", "revenue_month"} <= set(rev)
               else pd.PeriodIndex(rev["Date"], freq="M") - 1)
    rev["_period"] = periods
    raw_release = rev.get("create_time", pd.Series("", index=rev.index)).fillna("").astype(str).str.strip()
    actual = pd.to_datetime(raw_release, errors="coerce").dt.normalize()
    fallback = pd.Series([p.end_time.normalize() + pd.Timedelta(days=11) for p in periods], index=rev.index)
    # Exact announcement day is conservatively usable on the next exchange session.
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar))
    release = fallback.copy()
    for idx, day in actual.dropna().items():
        pos = sessions.searchsorted(day, side="right")
        release.loc[idx] = (sessions[pos] if len(sessions) and sessions[0] <= day < sessions[-1]
                            else day + pd.Timedelta(days=1))
    observed = pd.to_datetime(rev.get("first_observed_at", pd.Series(pd.NaT, index=rev.index)), errors="coerce").dt.normalize()
    unknown = actual.isna()
    release.loc[unknown & observed.notna()] = pd.concat([release, observed], axis=1).max(axis=1)
    counters["revenue_assumed_release_rows"] = int((unknown & observed.isna()).sum())
    values = pd.to_numeric(rev["revenue"], errors="coerce")
    bad = (~np.isfinite(values)) | (raw_release.ne("") & actual.isna())
    bad |= actual.notna() & (actual < pd.Series([p.end_time.normalize() for p in periods], index=rev.index))
    counters["revenue_invalid_rows"] = int(bad.sum())
    rev = rev.assign(_available=release, _value=values).loc[~bad]
    events = {}
    for sid, group in rev.groupby("stock_id", sort=False):
        history = {}; rows = []
        for available, batch in group.sort_values("_available", kind="stable").groupby("_available", sort=True):
            for period, same in batch.groupby("_period"):
                vals = same["_value"].unique()
                if len(vals) != 1:
                    raise ValueError("ambiguous revenue revisions at the same availability time")
                history[period] = float(vals[0])
            latest = max(history)
            def change(offset):
                previous = history.get(latest - offset, np.nan)
                return history[latest] / previous - 1 if np.isfinite(previous) and previous != 0 else np.nan
            rows.append((available, change(1), change(12)))
        events[str(sid)] = pd.DataFrame(rows, columns=["available", "Revenue_MoM", "Revenue_YoY"])
    return events, counters

def align_point_in_time_features(frame, revenue, market_value, calendar, max_market_cap_age=5):
    """Replace only revenue growth and size features before cross-sectional scaling."""
    result = frame.copy()
    result["Date"] = pd.to_datetime(result.Date)
    for col in ("Revenue_MoM", "Revenue_YoY", "Market_Cap_Log"):
        result[col] = np.nan
    events, report = _revenue_events(revenue, calendar)
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar))
    mv_groups = {}
    if market_value is not None and not market_value.empty:
        mv = market_value.copy()
        mv["Date"] = pd.to_datetime(mv.Date)
        mv["market_value"] = pd.to_numeric(mv.market_value, errors="coerce")
        mv = mv[np.isfinite(mv.market_value) & (mv.market_value > 0)]
        if mv.duplicated(["Date", "stock_id"]).any():
            raise ValueError("ambiguous market-value keys")
        mv_groups = {str(sid): g.sort_values("Date") for sid, g in mv.groupby("stock_id", sort=False)}
    report.update(version="v7-pit-alignment-v1", market_cap_max_age_sessions=max_market_cap_age,
                  market_cap_carried_rows=0, market_cap_unavailable_rows=0)
    for sid, sub in result.groupby("stock_id", sort=False):
        dates = pd.DatetimeIndex(sub.Date)
        event = events.get(str(sid))
        if event is not None and not event.empty:
            positions = pd.DatetimeIndex(event.available).searchsorted(dates, side="right") - 1
            valid = positions >= 0
            for col in ("Revenue_MoM", "Revenue_YoY"):
                result.loc[sub.index[valid], col] = event[col].to_numpy()[positions[valid]]
        values = mv_groups.get(str(sid))
        valid = np.zeros(len(sub), dtype=bool)
        if values is not None:
            positions = pd.DatetimeIndex(values.Date).searchsorted(dates, side="right") - 1
            found = positions >= 0
            selected_dates = pd.DatetimeIndex(values.Date.iloc[np.maximum(positions, 0)])
            ages = sessions.searchsorted(dates, side="right") - sessions.searchsorted(selected_dates, side="right")
            valid = found & (ages <= max_market_cap_age)
            result.loc[sub.index[valid], "Market_Cap_Log"] = np.log1p(values.market_value.to_numpy()[positions[valid]])
            report["market_cap_carried_rows"] += int((valid & (selected_dates != dates)).sum())
        report["market_cap_unavailable_rows"] += int((~valid).sum())
    report["revenue_missing_feature_rows"] = int(result[["Revenue_MoM", "Revenue_YoY"]].isna().any(axis=1).sum())
    return result, report
