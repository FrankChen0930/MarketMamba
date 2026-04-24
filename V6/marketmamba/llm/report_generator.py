"""
MarketMamba V6 — LLM Report Generator
========================================
Generates a daily market commentary by calling Claude (or GPT) API
AFTER MarketMamba's quantitative inference has completed.

MarketMamba role : pure quant (46D features, no sentiment in model)
LLM role        : market context, narrative, risk commentary

This module is completely decoupled from the model.
Swapping LLM provider only requires changing the generate_market_report() function.

Usage:
    from marketmamba.llm.report_generator import generate_market_report
    report = generate_market_report(df_kelly_top10, market_data)
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


# ============================================================
# Data Structures
# ============================================================

def build_market_data(
    df_prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Assemble today's market summary dict.
    Falls back gracefully if any source is unavailable.
    """
    import yfinance as yf

    today = datetime.today().strftime("%Y-%m-%d")

    def _safe_return(ticker: str) -> float:
        try:
            df = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
            if len(df) >= 2:
                return float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)
        except Exception:
            pass
        return 0.0

    market_data = {
        "date":         today,
        "twii_change":  _safe_return("^TWII"),
        "spx_change":   _safe_return("^GSPC"),
        "vix":          _safe_return("^VIX"),
        "gold_change":  _safe_return("GC=F"),
        "usd_twd":      _safe_return("TWD=X"),
    }
    return market_data


# ============================================================
# Claude API (primary)
# ============================================================

def _call_claude(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return ""

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set in environment")
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


# ============================================================
# GPT Fallback
# ============================================================

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
# Prompt Builder
# ============================================================

def _build_prompt(top10_rows: list[dict], market_data: dict) -> str:
    date_str    = market_data.get("date", "今日")
    twii_chg    = market_data.get("twii_change", 0.0)
    spx_chg     = market_data.get("spx_change", 0.0)
    vix         = market_data.get("vix", 0.0)
    gold_chg    = market_data.get("gold_change", 0.0)

    top10_text = "\n".join([
        f"  {i+1}. {row.get('Ticker','?')} | "
        f"預期Alpha(20d)={row.get('Exp_Alpha_20d', 0):+.2%} | "
        f"Sharpe={row.get('Sharpe_Score', 0):.2f} | "
        f"建議比重={row.get('Suggested_Weight', 0):.1%}"
        for i, row in enumerate(top10_rows)
    ])

    prompt = f"""
你是一位資深台灣股市量化策略分析師。
今天是 {date_str}，以下是今日市場與量化模型的數據摘要，請根據這些資訊撰寫分析報告。

【今日市場數據】
- 台股加權指數：{twii_chg:+.2f}%
- 美股 S&P 500 ：{spx_chg:+.2f}%
- VIX 恐慌指數：{vix:+.2f}%（日變化）
- 黃金：{gold_chg:+.2f}%

【量化模型 Top 10 選股】（基於 Mamba+GAT 多尺度時序模型）
{top10_text}

請用繁體中文撰寫一份簡潔的市場背景摘要，格式如下（不超過250字）：

🌐 **今日市場氛圍**
（一句話描述整體情緒：風險偏好/避險氛圍）

⚠️ **需關注的宏觀風險**
（條列 1-2 點最重要的外部風險，若無則標示「當前無特別風險」）

📊 **量化信號解讀**
（根據模型選股的產業分布或集中度，給出簡短觀察）

💡 **操作建議方向**
（配合量化信號的執行提示，不超過2句）

語氣：專業、客觀、簡潔，避免過度樂觀或恐慌性措辭。
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
    Generate a daily LLM market commentary.

    Args:
        df_kelly    : full kelly/alpha output from inference (must include 'Ticker', 'Sharpe_Score', etc.)
        market_data : dict from build_market_data() — if None, fetched automatically
        save        : whether to write market_summary.json

    Returns:
        dict with keys: date, summary, top10, market_data
    """
    if market_data is None:
        market_data = build_market_data()

    # Extract top 10
    if df_kelly.empty:
        top10_rows = []
    else:
        sort_col = "Sharpe_Score" if "Sharpe_Score" in df_kelly.columns else df_kelly.columns[0]
        top10 = df_kelly.nlargest(10, sort_col)
        top10_rows = top10.to_dict("records")

    prompt = _build_prompt(top10_rows, market_data)

    # Try Claude first, then OpenAI
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
        LLM_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM report saved → {LLM_REPORT_PATH}")

    return result
