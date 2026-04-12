"""
MarketMamba V5.5 — 自動調倉 + 帳本管理模組
負責：讀取雲端帳本、買賣邏輯、帳本日期統一修復、存檔
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

from marketmamba.config import get_repo_output_dir, get_today_str

logger = logging.getLogger('MarketMamba.robot')

LEDGER_URL = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/robot_ledger.json"


def _fix_date_format(date_str: str) -> str:
    """修復帳本日期格式不一致：統一轉為 YYYY-MM-DD"""
    return date_str[:10]  # 截斷 "2026-03-11 21:30" → "2026-03-11"


def load_ledger() -> dict:
    """從 GitHub 讀取雲端帳本，失敗時建立新帳本"""
    try:
        ledger = requests.get(LEDGER_URL, timeout=10).json()
        logger.info("📒 帳本已從 GitHub 載入")
    except Exception as e:
        logger.warning(f"⚠️ 無法讀取雲端帳本 ({e})，建立新帳本")
        ledger = {
            "start_date": get_today_str(),
            "cash": 1000000.0,
            "holdings": {},
            "history": [],
        }

    # 修復歷史日期格式
    for entry in ledger.get("history", []):
        entry["date"] = _fix_date_format(entry["date"])

    return ledger


def rebalance(df_kelly: pd.DataFrame = None,
              current_prices: dict = None,
              ledger: dict = None) -> dict:
    """
    自動調倉邏輯

    策略：
    1. 賣出不在 Top 10 的持股
    2. 依凱利建議權重買入 Top 10

    Args:
        df_kelly: 凱利評分表 (如未傳入則從 CSV 讀取)
        current_prices: {stock_id: close_price} 字典
        ledger: 帳本 dict (如未傳入則從 GitHub 讀取)

    Returns:
        更新後的帳本 dict
    """
    print("🤖 啟動量化實盤機器人...")

    # 讀取帳本
    if ledger is None:
        ledger = load_ledger()

    # 讀取凱利評分表
    if df_kelly is None:
        kelly_path = os.path.join(get_repo_output_dir(), 'df_kelly.csv')
        df_kelly = pd.read_csv(kelly_path)

    # 取得最新收盤價
    if current_prices is None:
        # 嘗試從 Parquet 取得
        try:
            from marketmamba.config import PROCESSED_DIR
            df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'V5_Mamba_Matrix.parquet'))
            latest_date = df['Date'].max()
            current_prices = (
                df[df['Date'] == latest_date]
                .set_index('stock_id')['Close']
                .to_dict()
            )
        except Exception:
            logger.warning("⚠️ 無法從 Parquet 取得收盤價，使用帳本成本價")
            current_prices = {}

    # 計算總權益
    total_equity = ledger["cash"]
    for t, position in ledger["holdings"].items():
        cost_val = position.get("avg_cost", position.get("cost", 0))
        total_equity += position["shares"] * current_prices.get(t, cost_val)

    buy_list = df_kelly.head(10)['Ticker'].astype(str).tolist()

    # === 賣出邏輯 ===
    for t in list(ledger["holdings"].keys()):
        if t not in buy_list:
            sell_price = current_prices.get(t, 0)
            if sell_price > 0:
                proceeds = ledger["holdings"][t]["shares"] * sell_price
                ledger["cash"] += proceeds
                print(f"  🔴 賣出 {t} × {ledger['holdings'][t]['shares']} 股 @ {sell_price:.2f}")
                del ledger["holdings"][t]

    # === 買入邏輯 ===
    for _, row in df_kelly.head(10).iterrows():
        t = str(row['Ticker'])
        weight = row['Suggested_Weight']

        if t not in current_prices:
            continue

        price = current_prices[t]
        target_value = total_equity * weight
        current_shares = ledger["holdings"].get(t, {}).get("shares", 0)
        money_to_invest = target_value - (current_shares * price)

        if money_to_invest > 0 and ledger["cash"] > money_to_invest:
            shares_to_buy = int(float(money_to_invest) // price)
            if shares_to_buy > 0:
                ledger["cash"] -= shares_to_buy * price
                old_s = ledger["holdings"].get(t, {}).get("shares", 0)
                old_c = ledger["holdings"].get(t, {}).get(
                    "avg_cost",
                    ledger["holdings"].get(t, {}).get("cost", price)
                )
                new_s = old_s + shares_to_buy
                new_avg = ((old_s * old_c) + (shares_to_buy * price)) / new_s

                ledger["holdings"][t] = {
                    "shares": new_s,
                    "avg_cost": new_avg,
                }
                print(f"  🟢 買入 {t} × {shares_to_buy} 股 @ {price:.2f}")

    # 記錄今日淨值
    today_str = get_today_str()
    if not ledger["history"] or ledger["history"][-1]["date"] != today_str:
        ledger["history"].append({
            "date": today_str,
            "equity": total_equity,
        })

    # 存檔
    output_dir = get_repo_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    ledger_path = os.path.join(output_dir, 'robot_ledger.json')

    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=4, ensure_ascii=False)

    print(f"✅ 機器人調倉完畢！淨值: ${total_equity:,.0f}")
    return ledger
