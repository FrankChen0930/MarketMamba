"""
MarketMamba V6 — LLM Report Generator
========================================
Generates a daily market commentary by calling Claude API
AFTER MarketMamba's quantitative inference has completed.

Role separation:
  MarketMamba : pure quant (46D features, no sentiment in model)
  LLM         : market context, narrative, per-stock analysis, risk commentary
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from marketmamba.config import (
    ANTHROPIC_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REPORT_PATH,
)

logger = logging.getLogger(__name__)

# ── Static ticker name lookup (offline-safe) ──────────────────────────────────

_TICKER_MAP: dict[str, str] = {}

def _load_ticker_map() -> dict[str, str]:
    global _TICKER_MAP
    if _TICKER_MAP:
        return _TICKER_MAP
    candidates = [
        Path(__file__).parent.parent.parent.parent / "app" / "backend" / "ticker_mapping.json",
        Path(__file__).parent.parent.parent / "ticker_mapping.json",
        Path("/mnt/d/Desktop/work/ProjectForMe/MarketMamba/app/backend/ticker_mapping.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                _TICKER_MAP = {k.split(".")[0]: v for k, v in raw.items() if isinstance(v, str)}
                logger.info(f"Loaded {len(_TICKER_MAP)} ticker names for LLM context")
                return _TICKER_MAP
            except Exception:
                pass
    return {}

def _get_name(ticker: str) -> str:
    m = _load_ticker_map()
    return m.get(str(ticker), str(ticker))


# ============================================================
# Market Data Builder
# ============================================================

def build_market_data(df_prices: pd.DataFrame | None = None) -> dict[str, Any]:
    """Assemble today's market snapshot. Falls back gracefully."""
    import yfinance as yf

    today = datetime.today().strftime("%Y-%m-%d")

    def _safe_pct(ticker: str) -> float:
        try:
            df = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
            if len(df) >= 2:
                return float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)
        except Exception:
            pass
        return 0.0

    def _safe_price(ticker: str) -> float:
        try:
            df = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
            if len(df) >= 1:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    return {
        "date":        today,
        "twii_change": _safe_pct("^TWII"),
        "spx_change":  _safe_pct("^GSPC"),
        "vix":         _safe_price("^VIX"),
        "gold_change": _safe_pct("GC=F"),
        "usd_twd":     _safe_price("TWD=X"),
    }


# ============================================================
# Claude API
# ============================================================

def _call_claude(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return ""
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        return ""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"Claude API error: {e}")
        return ""


def _call_openai(prompt: str) -> str:
    try:
        import openai
    except ImportError:
        return ""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI API error: {e}")
        return ""


# ============================================================
# Prompt Builder — Enriched for Sonnet 4.6
# ============================================================

def _build_prompt(top10_rows: list[dict], market_data: dict) -> str:
    date_str   = market_data.get("date", "今日")
    twii_chg   = market_data.get("twii_change", 0.0)
    spx_chg    = market_data.get("spx_change", 0.0)
    vix        = market_data.get("vix", 0.0)
    gold_chg   = market_data.get("gold_change", 0.0)
    usd_twd    = market_data.get("usd_twd", 0.0)

    # Build per-stock lines with company names
    top10_lines = []
    for i, row in enumerate(top10_rows):
        ticker = str(row.get("Ticker", "?"))
        name   = _get_name(ticker)
        alpha  = row.get("Exp_Alpha_20d", 0)
        sharpe = row.get("Signal_Quality", 0)
        weight = row.get("Suggested_Weight", 0)
        conf   = row.get("Confidence", "中信心")
        top10_lines.append(
            f"  {i+1:2d}. {ticker} {name} | "
            f"20d Alpha={alpha:+.2%} | Sharpe={sharpe:.2f} | "
            f"建議比重={weight:.1%} | 信心={conf}"
        )
    top10_text = "\n".join(top10_lines)

    prompt = f"""你是一位資深台灣股市量化策略分析師，擅長結合宏觀數據、基本面分析與技術面判讀。
今天是 {date_str}，請根據以下量化模型輸出與市場數據，撰寫一份完整的每日市場分析報告。

═══════════════════════════════════════
【今日全球市場快照】
═══════════════════════════════════════
- 台股加權指數（TAIEX）：{twii_chg:+.2f}%
- 美股 S&P 500：{spx_chg:+.2f}%
- VIX 恐慌指數：{vix:.2f}
- 黃金（期貨）：{gold_chg:+.2f}%
- USD/TWD 匯率：{usd_twd:.3f}

═══════════════════════════════════════
【MarketMamba V6 量化模型 Top 10 選股】
（模型：Mamba+GAT 多尺度時序 + 知識圖譜，46維特徵）
═══════════════════════════════════════
{top10_text}

═══════════════════════════════════════

請用**繁體中文**撰寫以下四個章節的報告，每個章節都要有實質內容：

## 🌐 今日市場氛圍與大盤解讀
根據 TAIEX、SPX、VIX、黃金的數據，分析今日整體市場情緒。
- 說明當前風險偏好方向（Risk-on / Risk-off）
- VIX 的位置代表什麼含義
- 台美股市聯動性分析（今日是跟隨美股還是獨立走勢？）
- 匯率對台股外資行為的潛在影響

## ⚠️ 宏觀風險與注意事項
- 列出 2-3 個當前最需要關注的宏觀或市場結構風險
- 若 VIX 偏高，說明波動背景
- 對量化信號的執行影響評估（高波動期信號可靠性下降說明）

## 📊 量化選股分析
針對 Top 10 選股給出有意義的分析：
- 產業分布觀察（這 10 檔集中在哪些產業？代表什麼輪動信號？）
- 挑出 3-4 檔最值得關注的個股，說明它們入選的可能原因（結合台灣市場知識，非臆測）
- 模型 Alpha 與 Sharpe 分布的整體解讀

## 💡 操作策略建議
- 配合今日市場環境，給出分批建倉 / 觀望 / 減碼的方向建議
- 建議的資金配置比例（保守/積極）
- 特別提示：哪類股票在當前環境應避開

語氣要求：**專業、客觀、有邏輯支撐**，避免過度樂觀或恐慌，不做無根據的漲跌預測。
字數控制在 600-800 字之間。報告末尾加一行：「⚠️ 本報告由 AI 自動生成，僅供研究參考，不構成投資建議。」
"""
    return prompt.strip()


# ============================================================
# Main Entry Point
# ============================================================

def generate_market_report(
    df_kelly:    pd.DataFrame,
    market_data: dict[str, Any] | None = None,
    save:        bool = True,
) -> dict[str, Any]:
    """
    Generate a daily LLM market commentary with enriched per-stock analysis.

    Args:
        df_kelly    : full alpha output from inference
        market_data : dict from build_market_data() — fetched if None
        save        : whether to write market_summary.json (and dated archive)

    Returns:
        dict with keys: date, summary, top10, market_data, model
    """
    if market_data is None:
        market_data = build_market_data()

    # Extract top 10
    if df_kelly.empty:
        top10_rows = []
    else:
        sort_col   = "Signal_Quality" if "Signal_Quality" in df_kelly.columns else df_kelly.columns[0]
        top10      = df_kelly.nlargest(10, sort_col)
        top10_rows = top10.to_dict("records")
        # Inject company names into top10_rows for frontend display
        for row in top10_rows:
            row["Name"] = _get_name(str(row.get("Ticker", "")))

    prompt = _build_prompt(top10_rows, market_data)
    logger.info(f"Calling {LLM_MODEL} for market report ({len(prompt)} chars prompt)...")

    summary_text = _call_claude(prompt)
    if not summary_text:
        logger.info("Claude unavailable, trying OpenAI fallback...")
        summary_text = _call_openai(prompt)
    if not summary_text:
        summary_text = "⚠️ LLM 報告生成失敗，請確認 API Key 設定。"
        logger.warning("LLM report generation failed — check API keys")

    result = {
        "date":        market_data["date"],
        "summary":     summary_text,
        "top10":       top10_rows,
        "market_data": market_data,
        "model":       LLM_MODEL,
    }

    if save:
        # Save latest (for frontend)
        LLM_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM report saved → {LLM_REPORT_PATH}")

        # Archive by date (for historical analysis)
        archive_dir = LLM_REPORT_PATH.parent / "archive"
        archive_dir.mkdir(exist_ok=True)
        dated_path = archive_dir / f"market_summary_{market_data['date']}.json"
        with open(dated_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM report archived → {dated_path}")

    return result
