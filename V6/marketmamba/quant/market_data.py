"""
MarketMamba V6 — Quant Market Data Module
==========================================
Computes daily market-wide quantitative indicators from already-cached parquet files.
Only ONE new network call: yfinance ^TWII (~3 s) for TAIEX technicals.

Output: V6/results/quant_market.json
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from marketmamba.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

_CACHE_DIR = DATA_DIR / "cache_v6"
QUANT_MARKET_PATH = RESULTS_DIR / "quant_market.json"


# ════════════════════════════════════════════════════════════════
# Technical Indicator Helpers (pure pandas/numpy, no ta-lib)
# ════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return round(float((100 - 100 / (1 + rs)).iloc[-1]), 2)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = close.ewm(span=fast,  adjust=False).mean()
    ema_s = close.ewm(span=slow,  adjust=False).mean()
    macd  = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist   = macd - signal
    return (round(float(macd.iloc[-1]), 2),
            round(float(signal.iloc[-1]), 2),
            round(float(hist.iloc[-1]), 2))


def _kd(high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 9, smooth: int = 3):
    low_min  = low.rolling(k_period,  min_periods=1).min()
    high_max = high.rolling(k_period, min_periods=1).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
    k   = rsv.ewm(com=smooth - 1, adjust=False).mean()
    d   = k.ewm(com=smooth - 1,   adjust=False).mean()
    return round(float(k.iloc[-1]), 2), round(float(d.iloc[-1]), 2)


def _bollinger_pct_b(close: pd.Series, period: int = 20) -> float:
    ma, std = close.rolling(period).mean(), close.rolling(period).std()
    pct_b = (close.iloc[-1] - (ma - 2 * std).iloc[-1]) / (4 * std.iloc[-1] + 1e-10)
    return round(float(pct_b), 4)


def _bias(close: pd.Series, period: int = 20) -> float:
    ma = close.rolling(period).mean()
    return round(float((close.iloc[-1] - ma.iloc[-1]) / (ma.iloc[-1] + 1e-10) * 100), 2)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return round(float(tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1]), 2)


def _realized_vol(close: pd.Series, period: int = 10) -> float:
    """Annualised realised vol over last `period` days."""
    ret = close.pct_change().dropna()
    rv  = ret.iloc[-period:].std() * np.sqrt(252) * 100
    return round(float(rv), 2)


def _beta(twii_close: pd.Series, spx_close: pd.Series, period: int = 60) -> float:
    r_tw  = twii_close.pct_change().dropna().iloc[-period:]
    r_spx = spx_close.pct_change().dropna().iloc[-period:]
    aligned = pd.concat([r_tw, r_spx], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return round(float(cov[0, 1] / (cov[1, 1] + 1e-10)), 2)


def _ma_score(close: pd.Series) -> int:
    """Bullish MA alignment score 0-100."""
    if len(close) < 120:
        return 50
    c = close.iloc[-1]
    ma20  = close.rolling(20).mean().iloc[-1]
    ma60  = close.rolling(60).mean().iloc[-1]
    ma120 = close.rolling(120).mean().iloc[-1]
    if c > ma20 > ma60 > ma120: return 85
    if c > ma20 > ma60:         return 70
    if c > ma20:                return 55
    if c > ma60:                return 40
    if c > ma120:               return 30
    return 15


def _build_tech_radar(rsi: float, macd_hist: float, kd_k: float,
                      bb_pct_b: float, ma_sc: int,
                      atr: float, atr_series: pd.Series) -> list[dict]:
    rsi_sc = min(100, max(0, round(rsi)))
    macd_sc = min(100, max(0, round(50 + macd_hist / (atr + 1e-6) * 100)))
    kd_sc   = min(100, max(0, round(kd_k)))
    bb_sc   = min(100, max(0, round(bb_pct_b * 100)))
    # ATR: lower = calmer = higher score
    vals = atr_series.dropna().values
    if len(vals) > 30:
        p20, p80 = np.percentile(vals, [20, 80])
        atr_sc = round(25 + 55 * (1 - (atr - p20) / (p80 - p20 + 1e-10)))
        atr_sc = min(80, max(15, atr_sc))
    else:
        atr_sc = 50
    return [
        {"subject": "RSI 動能",  "A": rsi_sc},
        {"subject": "MACD 趨勢", "A": macd_sc},
        {"subject": "KD 隨機",   "A": kd_sc},
        {"subject": "Bollinger", "A": bb_sc},
        {"subject": "MA 排列",   "A": ma_sc},
        {"subject": "ATR 波動",  "A": atr_sc},
    ]


# ════════════════════════════════════════════════════════════════
# Sub-computations
# ════════════════════════════════════════════════════════════════

def _taiex_technicals(macro_path: Path) -> dict:
    """Fetch ^TWII and compute full technical indicator suite."""
    logger.info("  [quant] TAIEX 技術指標：下載 ^TWII…")
    try:
        df = yf.download("^TWII", period="6mo", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 30:
            logger.warning("  [quant] ^TWII 資料不足")
            return {}

        close = df["Close"]
        high  = df["High"]  if "High"   in df.columns else close
        low   = df["Low"]   if "Low"    in df.columns else close
        vol   = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=close.index)

        # Build ATR series for radar normalisation
        tr_series = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr_series.ewm(com=13, min_periods=14).mean()

        rsi_v              = _rsi(close)
        macd_v, sig_v, hist_v = _macd(close)
        kd_k, kd_d         = _kd(high, low, close)
        bb_pb              = _bollinger_pct_b(close)
        bias_v             = _bias(close)
        atr_v              = _atr(high, low, close)
        rv10               = _realized_vol(close, 10)
        ma_sc              = _ma_score(close)
        obv_up             = (np.sign(close.diff()).fillna(0) * vol).cumsum()
        obv_trend          = "up" if obv_up.iloc[-1] > obv_up.rolling(20).mean().iloc[-1] else "down"

        cur   = round(float(close.iloc[-1]), 0)
        ma20  = round(float(close.rolling(20).mean().iloc[-1]), 0)
        ma60  = round(float(close.rolling(60).mean().iloc[-1]), 0)
        ma120 = round(float(close.rolling(120).mean().iloc[-1]), 0) if len(close) >= 120 else ma60

        # Beta vs SPX (use macro parquet if available)
        beta_v = 0.0
        if macro_path and macro_path.exists():
            try:
                macro = pd.read_parquet(macro_path)
                if "VIX" in macro.columns:   # macro has SPX column
                    spx_col = next((c for c in ["SPX", "^GSPC"] if c in macro.columns), None)
                    twii_col = next((c for c in ["TWII", "^TWII"] if c in macro.columns), None)
                    if spx_col:
                        spx_close = macro[spx_col].dropna()
                        if twii_col:
                            tw_close = macro[twii_col].dropna()
                        else:
                            # Use the same ^TWII we just downloaded
                            tw_close = close.copy()
                            tw_close.index = pd.to_datetime(tw_close.index)
                        beta_v = _beta(tw_close, spx_close)
            except Exception:
                pass

        # Helper: status tags
        def _status(v, pos_thr, neg_thr, pos_label, neg_label, mid_label="中性"):
            if v > pos_thr: return pos_label, "var(--positive)"
            if v < neg_thr: return neg_label, "var(--negative)"
            return mid_label, "var(--accent-amber)"

        rsi_s, rsi_c   = _status(rsi_v, 60, 40, "偏多", "偏空")
        macd_s, macd_c = ("偏多", "var(--positive)") if hist_v > 0 else ("偏空", "var(--negative)")
        kd_s, kd_c     = _status(kd_k, 60, 40, "偏多", "偏空")
        bb_s, bb_c     = _status(bb_pb, 0.8, 0.2, "上軌趨近", "下軌趨近", "帶內運行")
        obv_s, obv_c   = ("量能配合", "var(--positive)") if obv_trend == "up" else ("量能背離", "var(--negative)")

        tech_radar = _build_tech_radar(rsi_v, hist_v, kd_k, bb_pb, ma_sc, atr_v, atr_series)

        return {
            "close": cur, "ma_20": ma20, "ma_60": ma60, "ma_120": ma120,
            "rsi_14": rsi_v,
            "macd": macd_v, "macd_signal": sig_v, "macd_hist": hist_v,
            "kd_k": kd_k, "kd_d": kd_d,
            "bb_pct_b": bb_pb,
            "bias_20ma": bias_v,
            "atr_14": atr_v,
            "obv_trend": obv_trend,
            "realized_vol_10d": rv10,
            "beta_vs_spx": beta_v,
            "tech_radar": tech_radar,
            "table": [
                {"label": "RSI(14)",        "value": str(rsi_v),            "status": rsi_s,  "color": rsi_c},
                {"label": "MACD",           "value": f"{macd_v:+.1f}",      "status": macd_s, "color": macd_c},
                {"label": "KD(9,3,3) K",    "value": str(kd_k),             "status": kd_s,   "color": kd_c},
                {"label": "KD D值",          "value": str(kd_d),             "status": "黃金交叉" if kd_k > kd_d else "死亡交叉", "color": "var(--positive)" if kd_k > kd_d else "var(--negative)"},
                {"label": "Bollinger %B",   "value": f"{bb_pb:.2f}",        "status": bb_s,   "color": bb_c},
                {"label": "乖離率(20MA)",    "value": f"{bias_v:+.2f}%",     "status": "偏高" if bias_v > 5 else ("偏低" if bias_v < -5 else "正常"), "color": "var(--accent-amber)"},
                {"label": "ATR(14)",        "value": f"{atr_v:.1f}",        "status": "波動正常", "color": "var(--text-muted)"},
                {"label": "OBV 趨勢",        "value": "上升" if obv_trend == "up" else "下降", "status": obv_s, "color": obv_c},
            ],
            "risk_metrics": [
                {"label": "10日實現波動率",  "value": f"{rv10:.1f}%",        "desc": "年化"},
                {"label": "台股 Beta (vs SPX)", "value": str(beta_v),       "desc": "近60日"},
                {"label": "MA 20",          "value": f"{ma20:,.0f}",        "desc": "20日均線"},
                {"label": "MA 60",          "value": f"{ma60:,.0f}",        "desc": "60日均線"},
            ],
        }
    except Exception as e:
        logger.warning(f"  [quant] TAIEX 技術指標失敗：{e}")
        return {}


def _institutional_summary(date_str: str) -> dict:
    """Aggregate three-institution data from parquet (last 5 trading days)."""
    inst_path = PROCESSED_DIR / "institutional_raw.parquet"
    if not inst_path.exists():
        logger.warning("  [quant] institutional_raw.parquet 不存在")
        return {}
    try:
        df = pd.read_parquet(inst_path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        # Last 5 trading days up to date_str
        all_dates = sorted(df["Date"].unique())
        dates5    = [d for d in all_dates if d <= date_str][-5:]
        if not dates5:
            return {}
        df = df[df["Date"].isin(dates5)].copy()

        # Need to convert 千股 → 億元 using close prices
        prices_path = PROCESSED_DIR / "prices_raw.parquet"
        if prices_path.exists():
            pr = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])
            pr["Date"] = pd.to_datetime(pr["Date"]).dt.strftime("%Y-%m-%d")
            pr = pr[pr["Date"].isin(dates5)]
            # Foreign_Net is in 千股; × Close × 1000 ÷ 1e8 = 億元
            for col in ["Foreign_Net", "Investment_Trust_Net", "Dealer_Net"]:
                if col in df.columns:
                    merged = df[["Date", "stock_id", col]].merge(
                        pr[["Date", "stock_id", "Close"]], on=["Date", "stock_id"], how="left"
                    )
                    merged[col + "_bn"] = merged[col] * merged["Close"].fillna(0) / 1e5  # 億
                    df = df.merge(merged[["Date", "stock_id", col + "_bn"]],
                                  on=["Date", "stock_id"], how="left")

            agg = df.groupby("Date").agg({
                k: "sum" for k in [c for c in df.columns if c.endswith("_bn")]
            }).reset_index().sort_values("Date")
            f_col = "Foreign_Net_bn"
            t_col = "Investment_Trust_Net_bn"
            d_col = "Dealer_Net_bn"
        else:
            # Fallback: raw 千股 units
            agg = df.groupby("Date").agg({
                c: "sum" for c in ["Foreign_Net", "Investment_Trust_Net", "Dealer_Net"]
                if c in df.columns
            }).reset_index().sort_values("Date")
            for c in ["Foreign_Net", "Investment_Trust_Net", "Dealer_Net"]:
                if c in agg.columns:
                    agg[c + "_bn"] = agg[c] / 1e5   # rough approximation
            f_col = "Foreign_Net_bn"
            t_col = "Investment_Trust_Net_bn"
            d_col = "Dealer_Net_bn"

        history = []
        for _, row in agg.iterrows():
            d = str(row["Date"])
            history.append({
                "date": d,
                "date_label": d[5:].replace("-", "/"),
                "foreign_net": round(float(row.get(f_col, 0)), 1),
                "investment_trust_net": round(float(row.get(t_col, 0)), 1),
                "dealer_net": round(float(row.get(d_col, 0)), 1),
            })

        f5 = round(float(agg[f_col].sum() if f_col in agg.columns else 0), 1)
        t5 = round(float(agg[t_col].sum() if t_col in agg.columns else 0), 1)
        d5 = round(float(agg[d_col].sum() if d_col in agg.columns else 0), 1)

        return {
            "foreign_net_5d_bn": f5,
            "investment_trust_net_5d_bn": t5,
            "dealer_net_5d_bn": d5,
            "total_net_5d_bn": round(f5 + t5 + d5, 1),
            "history": history,
        }
    except Exception as e:
        logger.warning(f"  [quant] 三大法人彙總失敗：{e}")
        return {}


def _breadth_history(date_str: str) -> list[dict]:
    """Compute daily advancing/declining counts from prices_raw for last 5 days."""
    prices_path = PROCESSED_DIR / "prices_raw.parquet"
    if not prices_path.exists():
        return []
    try:
        pr = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])
        pr["Date"] = pd.to_datetime(pr["Date"]).dt.strftime("%Y-%m-%d")

        # Get last 6 days (need prev day for comparison)
        all_dates = sorted(pr["Date"].unique())
        dates6    = [d for d in all_dates if d <= date_str][-6:]
        pr = pr[pr["Date"].isin(dates6)].copy()

        # Pivot and compute day-over-day change
        pv = pr.pivot_table(index="stock_id", columns="Date", values="Close")
        pv = pv.sort_index(axis=1)

        results = []
        date_cols = list(pv.columns)
        for i in range(1, len(date_cols)):
            prev_d, curr_d = date_cols[i - 1], date_cols[i]
            diff = pv[curr_d] - pv[prev_d]
            adv  = int((diff > 0).sum())
            dec  = int((diff < 0).sum())
            ratio = round(adv / max(dec, 1), 2)
            results.append({
                "date":       curr_d,
                "date_label": curr_d[5:].replace("-", "/"),
                "adv": adv, "dec": dec, "ratio": ratio,
            })

        return results[-5:]   # last 5 trading days
    except Exception as e:
        logger.warning(f"  [quant] 市場廣度計算失敗：{e}")
        return []


def _margin_summary(date_str: str) -> dict:
    """Summarise margin/short from parquet."""
    margin_path = PROCESSED_DIR / "margin_raw.parquet"
    if not margin_path.exists():
        return {}
    try:
        df = pd.read_parquet(margin_path)
        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        latest_date = max(d for d in df["Date"].unique() if d <= date_str)
        df = df[df["Date"] == latest_date]

        # Try various column name conventions from FinMind
        def _sum_col(*candidates) -> float:
            for c in candidates:
                if c in df.columns:
                    return float(df[c].sum())
            return 0.0

        margin_bal = _sum_col("MarginPurchase_TodayBalance",
                              "margin_purchase_today_balance", "融資餘額")
        short_bal  = _sum_col("ShortSale_TodayBalance",
                              "short_sale_today_balance", "融券餘額")

        ratio = round(margin_bal / max(short_bal, 1), 1)

        return {
            "date": latest_date,
            "margin_balance_bn": round(margin_bal / 1e8, 1),    # 億元
            "short_balance_bn":  round(short_bal  / 1e8, 1),
            "margin_ratio":      ratio,
        }
    except Exception as e:
        logger.warning(f"  [quant] 融資融券彙總失敗：{e}")
        return {}


def _sector_flow(date_str: str) -> list[dict]:
    """Net institutional buying per sector (last 5 days), top 8."""
    inst_path    = PROCESSED_DIR / "institutional_raw.parquet"
    universe_path = _CACHE_DIR / "ticker_universe.parquet"
    prices_path  = PROCESSED_DIR / "prices_raw.parquet"

    if not inst_path.exists() or not universe_path.exists():
        return []
    try:
        uni = pd.read_parquet(universe_path)[["stock_id", "industry_category"]]
        uni = uni.rename(columns={"industry_category": "sector"})

        df = pd.read_parquet(inst_path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        all_dates = sorted(df["Date"].unique())
        dates5 = [d for d in all_dates if d <= date_str][-5:]
        df = df[df["Date"].isin(dates5)].copy()
        df = df.merge(uni, on="stock_id", how="left")
        df["sector"] = df["sector"].fillna("其他")

        # Convert to 億 using prices
        if prices_path.exists():
            pr = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])
            pr["Date"] = pd.to_datetime(pr["Date"]).dt.strftime("%Y-%m-%d")
            pr = pr[pr["Date"].isin(dates5)]
            df = df.merge(pr, on=["Date", "stock_id"], how="left")
            if "Foreign_Net" in df.columns:
                df["flow_bn"] = df["Foreign_Net"] * df["Close"].fillna(0) / 1e5
            else:
                df["flow_bn"] = 0.0
        else:
            df["flow_bn"] = df.get("Foreign_Net", 0) / 1e5

        flow = (df.groupby("sector")["flow_bn"].sum()
                  .sort_values(key=abs, ascending=False)
                  .head(8))

        return [
            {
                "sector": sec,
                "net_5d_bn": round(float(val), 1),
                "direction": "in" if val >= 0 else "out",
            }
            for sec, val in flow.items()
            if sec and sec != "nan"
        ]
    except Exception as e:
        logger.warning(f"  [quant] 產業資金輪動失敗：{e}")
        return []


# ════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════

def run_market_data(date_str: str,
                    output_path: Optional[Path] = None) -> dict:
    """
    Compute all market-wide quant data and save to quant_market.json.

    Args:
        date_str   : Target date 'YYYY-MM-DD'
        output_path: Override default output path (optional)

    Returns:
        The assembled dict (also saved to JSON).
    """
    output_path = output_path or QUANT_MARKET_PATH
    macro_path  = PROCESSED_DIR / "macro_raw.parquet"

    logger.info(f"[quant/market] 量化市場資料計算中… [{date_str}]")

    t0 = time.monotonic()
    tech    = _taiex_technicals(macro_path)
    inst    = _institutional_summary(date_str)
    breadth = _breadth_history(date_str)
    margin  = _margin_summary(date_str)
    sector  = _sector_flow(date_str)
    elapsed = time.monotonic() - t0

    result = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "taiex_technicals": tech,
        "institutional": inst,
        "breadth_history": breadth,
        "margin": margin,
        "sector_flow": sector,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_fields = sum(1 for v in [tech, inst, margin] if v) + len(breadth) + len(sector)
    logger.info(f"[quant/market] ✓ quant_market.json 完成 "
                f"（{n_fields} 資料組，耗時 {elapsed:.1f}s）")
    return result
