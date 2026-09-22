"""A/B diagnostics over identical per-date samples for single seeds and ensembles."""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

from v7_stability_contract import ensemble_percentile_ranks, validate_prediction_rows


def _stable_ranks(values):
    return {stock: index + 1 for index, (stock, _) in enumerate(
        sorted(values, key=lambda item: (-item[1], item[0]))
    )}


def _pearson(left, right):
    if len(left) < 2:
        return None
    lx = sum(left) / len(left)
    rx = sum(right) / len(right)
    numerator = sum((a - lx) * (b - rx) for a, b in zip(left, right))
    left_norm = sum((a - lx) ** 2 for a in left)
    right_norm = sum((b - rx) ** 2 for b in right)
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator else None


def _rank_ic(scores, labels):
    stocks = sorted(scores)
    predicted = _stable_ranks([(stock, scores[stock]) for stock in stocks])
    observed = _stable_ranks([(stock, labels[stock]) for stock in stocks])
    # Smaller rank is better, so rank and label ordering have the same sign.
    return _pearson([predicted[stock] for stock in stocks], [observed[stock] for stock in stocks])


def _top_stability(seed_scores, top_ns):
    output = {}
    for requested in top_ns:
        sets = []
        for scores in seed_scores.values():
            count = min(int(requested), len(scores))
            ordered = sorted(scores, key=lambda stock: (-scores[stock], stock))
            sets.append(set(ordered[:count]))
        overlaps = []
        for left, right in combinations(sets, 2):
            denominator = min(len(left), len(right))
            overlaps.append(len(left & right) / denominator if denominator else None)
        valid = [value for value in overlaps if value is not None]
        output[f"top{requested}"] = {
            "requested_n": int(requested),
            "effective_n": min(int(requested), min((len(values) for values in seed_scores.values()), default=0)),
            "pairwise_mean_overlap": sum(valid) / len(valid) if valid else None,
            "pair_count": len(valid),
        }
    return output


def build_diagnostics(rows, *, expected_seeds=(17, 29, 43), regimes=None, top_ns=(20, 50)):
    rows = validate_prediction_rows(rows)
    seeds = tuple(int(value) for value in expected_seeds)
    by_date_seed = defaultdict(dict)
    for row in rows:
        seed = int(row["seed"])
        if seed not in seeds:
            continue
        by_date_seed[(row["date"], seed)][row["stock_id"]] = row
    dates = sorted({date for date, _ in by_date_seed})
    for date in dates:
        missing = [seed for seed in seeds if (date, seed) not in by_date_seed]
        if missing:
            raise ValueError(f"date {date} missing seeds: {missing}")
    ensemble = {(row["date"], row["stock_id"]): row for row in
                ensemble_percentile_ranks(rows, expected_seeds=seeds)}
    daily = []
    for date in dates:
        for horizon in (5, 10):
            prediction_common = sorted(set.intersection(*(
                set(by_date_seed[(date, seed)]) for seed in seeds
            )))
            prediction_common = [
                stock for stock in prediction_common
                if all(by_date_seed[(date, seed)][stock][f"valid_{horizon}d"] for seed in seeds)
            ]
            mature_common = [
                stock for stock in prediction_common
                if all(by_date_seed[(date, seed)][stock][f"label_{horizon}d_mature"] for seed in seeds)
            ]
            labels = {}
            seed_scores = {seed: {} for seed in seeds}
            for stock in prediction_common:
                for seed in seeds:
                    seed_scores[seed][stock] = float(
                        by_date_seed[(date, seed)][stock][f"score_{horizon}d"]
                    )
            for stock in mature_common:
                values = [float(by_date_seed[(date, seed)][stock][f"label_{horizon}d"]) for seed in seeds]
                if max(values) != min(values):
                    raise ValueError(f"label mismatch across seeds: {(date, stock, horizon)}")
                labels[stock] = values[0]
            ensemble_scores = {
                stock: float(ensemble[(date, stock)][f"ensemble_score_{horizon}d"])
                for stock in mature_common
            }
            metrics = {
                f"seed{seed}": _rank_ic(
                    {stock: seed_scores[seed][stock] for stock in mature_common}, labels
                )
                for seed in seeds
            }
            metrics["ensemble"] = _rank_ic(ensemble_scores, labels)
            daily.append({
                "date": date,
                "horizon": horizon,
                "common_valid": len(prediction_common),
                "mature_common": len(mature_common),
                "rank_ic": metrics,
                "top_stability": _top_stability(seed_scores, top_ns),
                "regime": regimes.get(date) if regimes else None,
            })
    period_buckets = defaultdict(list)
    for row in daily:
        year = row["date"][:4]
        month = int(row["date"][5:7])
        quarter = f"{year}Q{(month - 1) // 3 + 1}"
        period_buckets[("year", year, row["horizon"])].append(row)
        period_buckets[("quarter", quarter, row["horizon"])].append(row)
        if row["regime"] is not None:
            period_buckets[("regime", str(row["regime"]), row["horizon"])].append(row)
    periods = []
    model_keys = [f"seed{seed}" for seed in seeds] + ["ensemble"]
    for (period_type, period, horizon), values in sorted(period_buckets.items()):
        rank_ic = {}
        for model in model_keys:
            finite = [row["rank_ic"][model] for row in values if row["rank_ic"][model] is not None]
            rank_ic[model] = sum(finite) / len(finite) if finite else None
        periods.append({
            "period_type": period_type,
            "period": period,
            "horizon": horizon,
            "dates": len(values),
            "common_valid_total": sum(row["common_valid"] for row in values),
            "mature_common_total": sum(row["mature_common"] for row in values),
            "mean_daily_rank_ic": rank_ic,
        })
    return {
        "schema_version": 1,
        "comparison_sample": "intersection of all expected seeds with mature labels per date and horizon",
        "expected_seeds": list(seeds),
        "data_ids": sorted({row["data_id"] for row in rows}),
        "daily": daily,
        "periods": periods,
    }
