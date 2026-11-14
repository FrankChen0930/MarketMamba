"""
portfolio_checker.py — 每日持倉退場條件預計算
=================================================
在每日推論 Step 7（訊號掃描後）執行。

讀取：
  - V6/results/df_kelly.csv            今日推論結果
  - V6/results/YYYY-MM-DD/df_kelly.csv 最近 5 天歷史（計算 streak）
  - V6/results/pattern_signals.json    型態訊號（bearish_pattern + failure_stop）
  - Data/processed_v6/prices_raw.parquet 機構籌碼（inst_sell_streak）

輸出：
  - V6/results/portfolio_exit_check.json
    → 後端 /api/portfolio/exit-check 提供給 PersonalOS Portfolio 頁面使用

每支股票輸出 market_data 欄位，供 PersonalOS 前端呼叫
signal_conditions.check_exit_conditions() 的等效邏輯。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 常數 ─────────────────────────────────────────────────────────────────────
TOP_N        = 300   # 輸出前 N 名股票（涵蓋一般持倉範圍）
HISTORY_DAYS = 5     # 回溯 N 個交易日計算 streak


# ── 主要入口 ──────────────────────────────────────────────────────────────────

def run_portfolio_check(
    results_dir: Path,
    data_dir: Path,
    date_str: Optional[str] = None,
) -> dict:
    """
    計算全市場 Top-N 股票的退場指標，輸出 portfolio_exit_check.json。

    Parameters
    ----------
    results_dir : V6/results/
    data_dir    : Data/processed_v6/
    date_str    : 推論日期（YYYY-MM-DD），預設使用今天

    Returns
    -------
    output dict（同時寫入 results_dir/portfolio_exit_check.json）
    """
    if date_str is None:
        date_str = datetime.today().strftime("%Y-%m-%d")

    print(f"[portfolio_checker] 開始計算退場指標：{date_str}", flush=True)

    # 1. 今日 df_kelly ──────────────────────────────────────────────────────────
    kelly_path = results_dir / "df_kelly.csv"
    if not kelly_path.exists():
        print(f"[portfolio_checker] ERROR: df_kelly.csv 不存在，跳過", flush=True)
        return {}

    df_today = pd.read_csv(kelly_path)
    df_today["Ticker"] = df_today["Ticker"].astype(str)
    # O3：優先用未截斷 raw 值排序（Top 區截斷並列 10.0 無法區分）
    _sq_col = "Signal_Quality_Raw" if "Signal_Quality_Raw" in df_today.columns else "Signal_Quality"
    df_today = df_today.sort_values(_sq_col, ascending=False).reset_index(drop=True)
    df_today["alpha_rank"] = df_today.index + 1
    total_stocks = len(df_today)

    df_top = df_today[df_today["alpha_rank"] <= TOP_N].copy()
    tickers_top = df_top["Ticker"].tolist()
    print(f"[portfolio_checker] 今日可投資：{total_stocks} 支，輸出前 {len(df_top)} 支", flush=True)

    # 2. 歷史資料（streak 計算）────────────────────────────────────────────────
    history_frames = _load_history_frames(results_dir, date_str, HISTORY_DAYS)

    # 3. Streak 指標 ────────────────────────────────────────────────────────────
    streak_map = _compute_streaks(tickers_top, history_frames, df_today)

    # 4. 機構淨賣出 streak ───────────────────────────────────────────────────────
    inst_map = _compute_inst_sell_streak(tickers_top, data_dir, date_str, HISTORY_DAYS)

    # 5. 型態訊號 ────────────────────────────────────────────────────────────────
    pattern_map = _load_pattern_signals(results_dir)

    # 6. 組裝輸出 ────────────────────────────────────────────────────────────────
    stocks_out: dict[str, dict] = {}
    for _, row in df_top.iterrows():
        ticker = str(row["Ticker"])
        streaks  = streak_map.get(ticker, {})
        patterns = pattern_map.get(ticker, {})
        rank     = int(row["alpha_rank"])

        stocks_out[ticker] = {
            "ticker": ticker,
            # 價格 / 排名
            "current_price":       _safe_float(row.get("Close")),
            "alpha_rank":          rank,
            "signal_quality":      _safe_float(row.get("Signal_Quality")),
            "signal_quality_pct":  round(rank / max(total_stocks, 1), 4),
            "uncertainty":         _safe_float(row.get("Uncertainty")),
            # Alpha
            "alpha_5d":            _safe_float(row.get("Exp_Alpha_5d")),
            "alpha_20d":           _safe_float(row.get("Exp_Alpha_20d")),
            "alpha_60d":           _safe_float(row.get("Exp_Alpha_60d")),
            # Streak 指標
            "rank_out50_streak":        streaks.get("rank_out50_streak", 0),
            "alpha_20d_declining_days": streaks.get("alpha_20d_declining_days", 0),
            # 機構籌碼
            "inst_sell_streak": inst_map.get(ticker, 0),
            # RS（df_kelly 目前不含 RS_20d，前端由其他來源補充）
            "rs_20d":              None,
            "rs_20d_negative_days": 0,
            "rs_20d_declining":    False,
            # RSI（暫無，由前端 quant 資料補充）
            "rsi": None,
            # 型態訊號
            "bearish_pattern":      patterns.get("bearish_pattern"),
            "pattern_failure_stop": patterns.get("failure_stop"),
        }

    output = {
        "date":          date_str,
        "generated_at":  datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_stocks":  total_stocks,
        "top_n":         len(stocks_out),
        "stocks":        stocks_out,
    }

    out_path = results_dir / "portfolio_exit_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    _print_summary(output)
    return output


# ── 內部函式 ──────────────────────────────────────────────────────────────────

def _load_history_frames(
    results_dir: Path,
    today_str: str,
    n_days: int,
) -> list[dict]:
    """
    讀取最近 n_days 個交易日的 df_kelly.csv（不含今日）。
    回傳 list[dict]，index 0 = 昨日（newest first）。
    每個 dict 含 {date, rank_map, alpha_map}。
    """
    frames: list[dict] = []
    today = datetime.strptime(today_str, "%Y-%m-%d").date()

    # 往前掃描最多 n_days*3 個日曆天（跳過無資料的週末 / 假日）
    for delta in range(1, n_days * 3 + 1):
        d = today - timedelta(days=delta)
        path = results_dir / d.strftime("%Y-%m-%d") / "df_kelly.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["Ticker"] = df["Ticker"].astype(str)
                _c = "Signal_Quality_Raw" if "Signal_Quality_Raw" in df.columns else "Signal_Quality"
                df = df.sort_values(_c, ascending=False).reset_index(drop=True)
                df["alpha_rank"] = df.index + 1
                frames.append({
                    "date":      d,
                    "rank_map":  dict(zip(df["Ticker"], df["alpha_rank"])),
                    "alpha_map": dict(zip(df["Ticker"], df["Exp_Alpha_20d"].fillna(0))),
                })
            except Exception as e:
                logger.debug(f"無法讀取 {path}: {e}")
        if len(frames) >= n_days:
            break

    print(f"[portfolio_checker] 歷史資料：找到 {len(frames)} 天", flush=True)
    return frames  # newest first


def _compute_streaks(
    tickers: list[str],
    history_frames: list[dict],
    df_today: pd.DataFrame,
) -> dict:
    """
    rank_out50_streak：從今日往回數，連續 alpha_rank > 50 的天數。
    alpha_20d_declining_days：從今日往回數，連續 alpha_20d 下滑（今 < 昨）的天數。
    """
    today_rank  = dict(zip(df_today["Ticker"], df_today["alpha_rank"]))
    today_alpha = dict(zip(df_today["Ticker"], df_today["Exp_Alpha_20d"].fillna(0)))

    result: dict[str, dict] = {}
    for ticker in tickers:
        # 建立時間序列（今日 → 昨日 → ...），遇到缺失日截斷
        days_rank  = [today_rank.get(ticker, 9999)]
        days_alpha = [today_alpha.get(ticker, 0.0)]

        for frame in history_frames:
            r = frame["rank_map"].get(ticker)
            a = frame["alpha_map"].get(ticker)
            if r is None:
                break  # 該日無資料 → 截斷
            days_rank.append(r)
            days_alpha.append(a)

        # rank_out50_streak：從今日往回，連續 > 50
        rank_streak = 0
        for r in days_rank:
            if r > 50:
                rank_streak += 1
            else:
                break  # 進入 Top50 → 停止

        # alpha_20d_declining_days：連續「今比昨小」的天數
        alpha_streak = 0
        for i in range(len(days_alpha) - 1):
            if days_alpha[i] < days_alpha[i + 1]:  # 今日 < 昨日 → 下滑
                alpha_streak += 1
            else:
                break

        result[ticker] = {
            "rank_out50_streak":        rank_streak,
            "alpha_20d_declining_days": alpha_streak,
        }

    return result


def _compute_inst_sell_streak(
    tickers: list[str],
    data_dir: Path,
    date_str: str,
    n_days: int,
) -> dict:
    """
    從 institutional_raw.parquet 讀取最近 n_days 天的外資淨買賣，
    計算連續機構淨賣出（Foreign_Net < 0）天數（newest first）。

    註：舊版讀 prices_raw.parquet 的 Foreign_Buy/Foreign_Sell——但 prices_raw
    只有 OHLCV 欄位，該讀取每天都 KeyError fallback、streak 恆為 0（條件空轉）。
    """
    result: dict[str, int] = {t: 0 for t in tickers}
    inst_path = data_dir / "institutional_raw.parquet"

    if not inst_path.exists():
        print(f"[portfolio_checker] institutional_raw.parquet 不存在，inst_sell_streak 全設為 0", flush=True)
        return result

    try:
        df = pd.read_parquet(inst_path, columns=["stock_id", "Date", "Foreign_Buy", "Foreign_Sell"])
        df["stock_id"] = df["stock_id"].astype(str)

        cutoff = datetime.strptime(date_str, "%Y-%m-%d").date() - timedelta(days=n_days * 3)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[(df["Date"] >= cutoff) & (df["stock_id"].isin(tickers))].copy()
        df["net_foreign"] = df["Foreign_Buy"].fillna(0) - df["Foreign_Sell"].fillna(0)

        for ticker in tickers:
            sub = (
                df[df["stock_id"] == ticker]
                .sort_values("Date", ascending=False)
                .head(n_days)
            )
            streak = 0
            for _, row in sub.iterrows():
                if row["net_foreign"] < 0:
                    streak += 1
                else:
                    break  # 淨買入 → 中斷
            result[ticker] = streak

        sell_cnt = sum(1 for v in result.values() if v > 0)
        print(f"[portfolio_checker] inst_sell_streak: {sell_cnt} 支有機構賣出紀錄", flush=True)

    except KeyError:
        print(f"[portfolio_checker] institutional_raw 缺少 Foreign_Buy/Foreign_Sell 欄位，inst_sell_streak 全設為 0", flush=True)
    except Exception as e:
        print(f"[portfolio_checker] inst_sell_streak 計算失敗（非致命）：{e}", flush=True)
        logger.warning(f"inst_sell_streak 失敗：{e}")

    return result


def _load_pattern_signals(results_dir: Path) -> dict:
    """
    從 pattern_signals.json 讀取：
    - bullish signals → failure_stop（型態失敗退場價）
    - bearish signals → bearish_pattern（空方型態 ID）
    """
    pattern_path = results_dir / "pattern_signals.json"
    result: dict[str, dict] = {}

    if not pattern_path.exists():
        print(f"[portfolio_checker] pattern_signals.json 不存在，跳過型態訊號", flush=True)
        return result

    try:
        with open(pattern_path, encoding="utf-8") as f:
            data = json.load(f)

        for sig in data.get("signals", []):          # bullish
            ticker = str(sig.get("stock_id", ""))
            if not ticker:
                continue
            if ticker not in result:
                result[ticker] = {}
            fs = sig.get("failure_stop")
            if fs is not None:
                result[ticker]["failure_stop"] = float(fs)

        for sig in data.get("bearish_signals", []):  # bearish
            ticker = str(sig.get("stock_id", ""))
            if not ticker:
                continue
            if ticker not in result:
                result[ticker] = {}
            result[ticker]["bearish_pattern"] = sig.get("pattern_id")

    except Exception as e:
        print(f"[portfolio_checker] pattern_signals.json 讀取失敗：{e}", flush=True)
        logger.warning(f"pattern_signals 讀取失敗：{e}")

    bullish_cnt = sum(1 for v in result.values() if v.get("failure_stop") is not None)
    bearish_cnt = sum(1 for v in result.values() if v.get("bearish_pattern"))
    print(f"[portfolio_checker] 型態訊號：{bullish_cnt} 支有 failure_stop，{bearish_cnt} 支有空方型態", flush=True)
    return result


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if f != f else f  # NaN 轉 default
    except Exception:
        return default


def _print_summary(output: dict) -> None:
    """輸出人類可讀的診斷摘要。"""
    stocks = output.get("stocks", {})
    if not stocks:
        return

    print(f"\n[portfolio_checker] ═══ 退場指標摘要 ═══", flush=True)
    print(f"  日期：{output['date']} | 總股票數：{output['total_stocks']} | 輸出：Top {output['top_n']}", flush=True)

    out50 = sorted(
        [(t, d["rank_out50_streak"]) for t, d in stocks.items() if d["rank_out50_streak"] > 0],
        key=lambda x: -x[1],
    )
    if out50:
        top5 = ", ".join(f"{t}({s}天)" for t, s in out50[:5])
        print(f"  rank_out50_streak>0：{len(out50)} 支，最長前5：{top5}", flush=True)
    else:
        print(f"  rank_out50_streak>0：0 支（所有前{output['top_n']}均在 Top50 以內）", flush=True)

    declining = sorted(
        [(t, d["alpha_20d_declining_days"]) for t, d in stocks.items() if d["alpha_20d_declining_days"] > 0],
        key=lambda x: -x[1],
    )
    if declining:
        top5 = ", ".join(f"{t}({s}天)" for t, s in declining[:5])
        print(f"  alpha_20d_declining>0：{len(declining)} 支，最長前5：{top5}", flush=True)

    inst = sorted(
        [(t, d["inst_sell_streak"]) for t, d in stocks.items() if d["inst_sell_streak"] > 0],
        key=lambda x: -x[1],
    )
    if inst:
        top5 = ", ".join(f"{t}({s}天)" for t, s in inst[:5])
        print(f"  inst_sell_streak>0：{len(inst)} 支，最長前5：{top5}", flush=True)

    bearish = [t for t, d in stocks.items() if d.get("bearish_pattern")]
    if bearish:
        print(f"  空方型態：{len(bearish)} 支：{', '.join(bearish[:10])}", flush=True)

    print(f"[portfolio_checker] ═══════════════════════\n", flush=True)
