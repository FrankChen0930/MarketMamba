"""
MarketMamba V6.2 — Entry Condition Contribution Analyzer
=========================================================
回答「四個進場條件（＋型態加分）各自到底有沒有預測力」——目前 30/25/25/20 權重
與 70/90 門檻全是手設值，本分析器用歷史歸檔提供校準證據。

方法比照 dual_ic_analyzer.py，改讀每日歸檔的 scanner 輸出：

    V6/results/{YYYY-MM-DD}/action_signals.json   （buy_signals + watch_list，含各條件 met 旗標）
    V6/results/{YYYY-MM-DD}/df_kelly.csv           （當日 Top50 池，作為對照基準）

對每個 horizon（5d/20d）計算：
  - 各條件（rank_stability / high_confidence / institutional_buy / relative_low
    / pattern）觸發股票池的前瞻超額報酬（去當日全市場均值）
  - 對照基準：當日 Top50 全體的前瞻超額報酬 → edge = 條件池 − Top50
    （條件在候選池「之上」有沒有額外貢獻，比絕對報酬更能歸因）
  - 評分分桶（<40 / 40–69 / ≥70）——直接檢驗統一後的 70 分門檻歷史上是否有效
  - 條件數分桶（1/2/3/4 條件）

輸出：V6/results/condition_analysis.json

⚠️ 誠實註記：
  - 歸檔為 90 天滾動，樣本天數有限；n_days < 20 的結論僅供方向參考
  - 條件旗標是「當日已知資訊」、報酬取自未來交易日 → 無 look-ahead
  - 每日條件池取等權平均後才跨日彙總（每天一票，避免訊號多的日子主導）
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = {"5d": 5, "20d": 20}
CONDITIONS = ["rank_stability", "high_confidence", "institutional_buy", "relative_low"]
SCORE_BUCKETS = [("<40", 0, 40), ("40-69", 40, 70), (">=70", 70, 10**9)]
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MIN_MARKET_STOCKS = 200   # 當日有效價格股數低於此 → 跳過（市場均值不可靠）


def _list_archive_days(results_dir: Path) -> list[Path]:
    """回傳含 action_signals.json 的 dated 歸檔目錄（日期遞增）。"""
    days = [
        d for d in results_dir.iterdir()
        if d.is_dir() and _DATE_DIR_RE.match(d.name) and (d / "action_signals.json").exists()
    ]
    return sorted(days, key=lambda d: d.name)


def _load_signals(day_dir: Path) -> Optional[list[dict]]:
    """讀取當日 buy_signals + watch_list（兩者都有完整條件旗標）。"""
    try:
        with open(day_dir / "action_signals.json", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"  [{day_dir.name}] action_signals.json unreadable: {e}")
        return None
    return list(data.get("buy_signals", [])) + list(data.get("watch_list", []))


def _load_top50(day_dir: Path) -> list[str]:
    """當日 df_kelly 前 50 名 ticker（scanner 的掃描池，作為對照基準）。"""
    path = day_dir / "df_kelly.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, nrows=50, dtype={"Ticker": str})
        return df["Ticker"].astype(str).tolist() if "Ticker" in df.columns else []
    except Exception:
        return []


def _summarize(day_values: list[dict], top50_by_day: dict[str, float]) -> dict:
    """跨日彙總一個池子的每日等權超額報酬。

    day_values: [{"date": d, "excess": pool 當日平均超額 %, "n": 檔數}]
    """
    if not day_values:
        return {"n_days": 0, "n_signals": 0, "mean_excess_pct": None,
                "hit_rate_pct": None, "edge_vs_top50_pct": None}
    ex = np.array([v["excess"] for v in day_values])
    edges = [v["excess"] - top50_by_day[v["date"]]
             for v in day_values if v["date"] in top50_by_day]
    return {
        "n_days":             len(ex),
        "n_signals":          int(sum(v["n"] for v in day_values)),
        "mean_excess_pct":    round(float(ex.mean()), 3),
        "hit_rate_pct":       round(float((ex > 0).mean() * 100), 1),
        "edge_vs_top50_pct":  round(float(np.mean(edges)), 3) if edges else None,
    }


def compute_condition_analysis(results_dir: Path, prices_df: pd.DataFrame) -> dict[str, Any]:
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")
    prices_df["stock_id"] = prices_df["stock_id"].astype(str)
    prices_df = prices_df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index()
    price_dates = price_wide.index.tolist()

    day_dirs = _list_archive_days(results_dir)
    if not day_dirs:
        logger.warning("Condition Analyzer: no archived action_signals found")
        return _empty_result()

    logger.info(
        f"Condition Analyzer: {len(day_dirs)} archived days "
        f"({day_dirs[0].name} → {day_dirs[-1].name})"
    )

    # per horizon → pool_name → list of {"date", "excess", "n"}
    pools: dict[str, dict[str, list[dict]]] = {h: {} for h in HORIZONS}
    top50_by_day: dict[str, dict[str, float]] = {h: {} for h in HORIZONS}

    for day_dir in day_dirs:
        date = day_dir.name
        if date not in price_wide.index:
            continue
        try:
            idx = price_dates.index(date)
        except ValueError:
            continue

        signals = _load_signals(day_dir)
        if not signals:
            continue
        top50 = _load_top50(day_dir)

        p0 = price_wide.loc[date]

        for h_label, n_fwd in HORIZONS.items():
            if idx + n_fwd >= len(price_dates):
                continue   # 未來還沒到，正常情況
            p1 = price_wide.loc[price_dates[idx + n_fwd]]
            valid = p0.notna() & p1.notna() & (p0 > 0)
            if valid.sum() < MIN_MARKET_STOCKS:
                continue
            ret = (p1[valid] / p0[valid] - 1) * 100
            market_mean = float(ret.mean())
            excess = ret - market_mean   # 每股超額報酬 %

            def pool_excess(tickers: list[str]) -> Optional[tuple[float, int]]:
                vals = excess.reindex([str(t) for t in tickers]).dropna()
                if vals.empty:
                    return None
                return float(vals.mean()), int(len(vals))

            # 對照基準：Top50
            t50 = pool_excess(top50)
            if t50 is not None:
                top50_by_day[h_label][date] = t50[0]

            # 各池子
            buckets: dict[str, list[str]] = {}
            for sig in signals:
                t = str(sig.get("ticker", ""))
                if not t:
                    continue
                for cond in CONDITIONS:
                    if sig.get(cond, {}).get("met"):
                        buckets.setdefault(f"cond:{cond}", []).append(t)
                if sig.get("pattern"):
                    buckets.setdefault("cond:pattern", []).append(t)
                n_met = int(sig.get("conditions_met", 0))
                if n_met >= 1:
                    buckets.setdefault(f"nmet:{min(n_met, 4)}", []).append(t)
                score = int(sig.get("score", 0))
                for b_name, lo, hi in SCORE_BUCKETS:
                    if lo <= score < hi:
                        buckets.setdefault(f"score:{b_name}", []).append(t)
                        break

            for pool_name, tickers in buckets.items():
                pe = pool_excess(tickers)
                if pe is not None:
                    pools[h_label].setdefault(pool_name, []).append(
                        {"date": date, "excess": pe[0], "n": pe[1]}
                    )

    # ── 彙總 ─────────────────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "generated_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "n_archive_days": len(day_dirs),
        "period_start":   day_dirs[0].name,
        "period_end":     day_dirs[-1].name,
        "horizons":       {},
    }

    for h_label in HORIZONS:
        t50_days = top50_by_day[h_label]
        top50_summary = (
            {"n_days": len(t50_days),
             "mean_excess_pct": round(float(np.mean(list(t50_days.values()))), 3),
             "hit_rate_pct": round(float(np.mean([v > 0 for v in t50_days.values()]) * 100), 1)}
            if t50_days else {"n_days": 0, "mean_excess_pct": None, "hit_rate_pct": None}
        )

        conditions_out = {
            cond: _summarize(pools[h_label].get(f"cond:{cond}", []), t50_days)
            for cond in CONDITIONS + ["pattern"]
        }
        by_nmet = {
            str(n): _summarize(pools[h_label].get(f"nmet:{n}", []), t50_days)
            for n in (1, 2, 3, 4)
        }
        by_score = {
            b_name: _summarize(pools[h_label].get(f"score:{b_name}", []), t50_days)
            for b_name, _, _ in SCORE_BUCKETS
        }

        result["horizons"][h_label] = {
            "top50_baseline":    top50_summary,
            "conditions":        conditions_out,
            "by_conditions_met": by_nmet,
            "by_score_bucket":   by_score,
        }

        # 人類可讀輸出（規則 7：數值必須明確顯示）
        print(f"\n[condition_analysis/{h_label}] baseline Top50: "
              f"n={top50_summary['n_days']} mean_excess={top50_summary['mean_excess_pct']}% "
              f"hit={top50_summary['hit_rate_pct']}%", flush=True)
        for cond, s in conditions_out.items():
            print(f"  {cond:<18} n_days={s['n_days']:>3} signals={s['n_signals']:>4} "
                  f"excess={s['mean_excess_pct']}% hit={s['hit_rate_pct']}% "
                  f"edge_vs_top50={s['edge_vs_top50_pct']}%", flush=True)
        for b_name, s in by_score.items():
            print(f"  score {b_name:<12} n_days={s['n_days']:>3} signals={s['n_signals']:>4} "
                  f"excess={s['mean_excess_pct']}% hit={s['hit_rate_pct']}%", flush=True)

    n5 = result["horizons"]["5d"]["top50_baseline"]["n_days"]
    if n5 < 20:
        result["note"] = f"樣本僅 {n5} 天（5d），統計力有限，建議累積 20+ 天再據以調整權重"
        print(f"\n⚠️ {result['note']}", flush=True)

    return result


def run_condition_analysis(
    results_dir: Path,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Compute entry-condition contribution analysis → condition_analysis.json."""
    result = compute_condition_analysis(results_dir, prices_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ condition_analysis saved → {output_path.name}")
    return result


def _empty_result() -> dict[str, Any]:
    empty = {"n_days": 0, "n_signals": 0, "mean_excess_pct": None,
             "hit_rate_pct": None, "edge_vs_top50_pct": None}
    return {
        "generated_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "n_archive_days": 0,
        "period_start":   None,
        "period_end":     None,
        "horizons": {
            h: {
                "top50_baseline": {"n_days": 0, "mean_excess_pct": None, "hit_rate_pct": None},
                "conditions": {c: dict(empty) for c in CONDITIONS + ["pattern"]},
                "by_conditions_met": {str(n): dict(empty) for n in (1, 2, 3, 4)},
                "by_score_bucket": {b: dict(empty) for b, _, _ in SCORE_BUCKETS},
            }
            for h in HORIZONS
        },
    }
