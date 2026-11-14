"""Stable, JSON-safe research outputs shared by ranking, reports and portfolios."""
from __future__ import annotations

import math
from collections import defaultdict

PREDICTION_FIELDS = (
    "date", "stock_id", "score_5d", "score_10d", "valid_5d", "valid_10d",
    "label_5d", "label_10d", "label_5d_mature", "label_10d_mature",
    "model_id", "data_id",
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_prediction_rows(rows):
    validated = []
    seen = set()
    for position, source in enumerate(rows):
        missing = [field for field in PREDICTION_FIELDS if field not in source]
        if missing:
            raise ValueError(f"prediction row {position} missing fields: {', '.join(missing)}")
        row = dict(source)
        key = (str(row["date"]), str(row["stock_id"]), str(row["model_id"]))
        if key in seen:
            raise ValueError(f"duplicate prediction identity: {key}")
        seen.add(key)
        row["date"], row["stock_id"] = key[0], key[1]
        if not row["date"] or not row["stock_id"] or not row["model_id"] or not row["data_id"]:
            raise ValueError("prediction identity fields must be non-empty")
        for horizon in (5, 10):
            score = row[f"score_{horizon}d"]
            valid = row[f"valid_{horizon}d"]
            label = row[f"label_{horizon}d"]
            mature = row[f"label_{horizon}d_mature"]
            if not isinstance(valid, bool) or not isinstance(mature, bool):
                raise ValueError("valid and maturity flags must be boolean")
            if valid != _finite(score):
                raise ValueError(f"valid_{horizon}d must match finite score")
            if mature and not _finite(label):
                raise ValueError(f"mature label_{horizon}d must be finite")
            if not mature and label is not None:
                if isinstance(label, (int, float)) and not isinstance(label, bool) and math.isnan(label):
                    row[f"label_{horizon}d"] = None
                else:
                    raise ValueError(f"immature label_{horizon}d must remain null")
        validated.append(row)
    return validated


def _stable_percentiles(values):
    ordered = sorted(values, key=lambda item: (-item[1], item[0]))
    denominator = max(len(ordered) - 1, 1)
    return {stock: 1.0 - index / denominator for index, (stock, _) in enumerate(ordered)}


def ensemble_percentile_ranks(rows, *, expected_seeds=(17, 29, 43)):
    by_date_seed = defaultdict(dict)
    for source in rows:
        date, stock, seed = str(source["date"]), str(source["stock_id"]), int(source["seed"])
        key = (date, seed)
        if stock in by_date_seed[key]:
            raise ValueError(f"duplicate seed prediction: {(date, stock, seed)}")
        by_date_seed[key][stock] = source
    dates = sorted({key[0] for key in by_date_seed})
    seeds = tuple(int(seed) for seed in expected_seeds)
    output = []
    for date in dates:
        missing = [seed for seed in seeds if (date, seed) not in by_date_seed]
        if missing:
            raise ValueError(f"date {date} missing seeds: {missing}")
        stocks = sorted(set().union(*(by_date_seed[(date, seed)] for seed in seeds)))
        result = {stock: {"date": date, "stock_id": stock, "seed_count": len(seeds)} for stock in stocks}
        for horizon in (5, 10):
            common = [
                stock for stock in stocks
                if all(_finite(by_date_seed[(date, seed)].get(stock, {}).get(f"score_{horizon}d"))
                       for seed in seeds)
            ]
            seed_percentiles = {}
            for seed in seeds:
                values = [(stock, float(by_date_seed[(date, seed)][stock][f"score_{horizon}d"]))
                          for stock in common]
                seed_percentiles[seed] = _stable_percentiles(values)
            for stock in stocks:
                result[stock][f"ensemble_score_{horizon}d"] = (
                    sum(seed_percentiles[seed][stock] for seed in seeds) / len(seeds)
                    if stock in common else None
                )
            ranked = sorted(
                (stock for stock in common),
                key=lambda stock: (-result[stock][f"ensemble_score_{horizon}d"], stock),
            )
            for stock in stocks:
                result[stock][f"rank_{horizon}d"] = None
                result[stock][f"valid_{horizon}d"] = stock in common
            for rank, stock in enumerate(ranked, 1):
                result[stock][f"rank_{horizon}d"] = rank
        output.extend(result[stock] for stock in stocks)
    return output
