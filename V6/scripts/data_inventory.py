"""
MarketMamba — 資料完整性總盤點（2026-07-29）
=============================================
逐一列出每個資料源的涵蓋期間、缺口、以及「缺口落在訓練期還是驗證期」，
用來回答「哪些資料補不齊、對訓練有什麼影響」。

用法：python V6/scripts/data_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

P = Path(PROCESSED_DIR)

# 方向二協定的切分（docs/baseline-experiment-protocol-draft-2026-07-11.md）
TRAIN_END = pd.Timestamp("2023-12-31")

SOURCES = [
    ("prices_raw", "Group A 價量（全部技術指標的基礎）"),
    ("institutional_raw", "Group B 三大法人"),
    ("margin_raw", "Group B 融資融券"),
    ("daytrade_raw", "Group B 當沖"),
    ("securities_raw", "Group B 借券"),
    ("holdings_raw", "Group B 集保股權分散"),
    ("foreign_shareholding_raw", "Group B 外資持股"),
    ("per_raw", "Group C 本益比/淨值比"),
    ("market_value_raw", "Group C 市值"),
    ("revenue_raw", "Group C 月營收"),
    ("financials_raw", "Group C 季財報"),
    ("balance_sheet_raw", "Group C 資產負債表"),
    ("dividend_raw", "Group C 股利"),
    ("macro_raw", "Group D 總體（V6.1 下全歸零）"),
    ("futures_institutional_raw", "Group D 期貨法人（V6.1 下歸零）"),
    ("options_institutional_raw", "Group D 選擇權法人（V6.1 下歸零）"),
    ("fear_greed", "Group D 恐懼貪婪（V6.1 下歸零）"),
    ("business_indicator", "Group D 景氣燈號（V6.1 下歸零）"),
    ("ex_rights_raw", "還原因子（不進特徵，供價格還原）"),
]


def _date_col(df: pd.DataFrame) -> str | None:
    for c in ("Date", "date", "Week"):
        if c in df.columns:
            return c
    return None


def main() -> None:
    today = pd.Timestamp.today().normalize()
    print("=" * 100)
    print(f"■ 資料源完整性盤點（基準日 {today.date()}，訓練切分 train ≤ {TRAIN_END.date()}）")
    print("=" * 100)
    print(f"{'資料源':<28}{'起':<12}{'迄':<12}{'落後':>6}  {'訓練期是否完整':<16}說明")
    print("-" * 100)

    for name, desc in SOURCES:
        p = P / f"{name}.parquet"
        if not p.exists():
            print(f"{name:<28}{'—':<12}{'—':<12}{'—':>6}  {'檔案不存在':<16}{desc}")
            continue
        df = pd.read_parquet(p)
        c = _date_col(df)
        if c is None:
            print(f"{name:<28}{'—':<12}{'—':<12}{'—':>6}  {'無日期欄':<16}{desc}")
            continue
        d = pd.to_datetime(df[c], errors="coerce").dropna()
        lo, hi = d.min(), d.max()
        lag = (today - hi).days
        train_ok = "✓ 完整" if hi >= TRAIN_END else f"✗ 只到 {hi.date()}"
        print(f"{name:<28}{str(lo.date()):<12}{str(hi.date()):<12}{lag:>4}天  "
              f"{train_ok:<16}{desc}")

    # ── 近期缺口（訓練切分之後）─────────────────────────────────────
    print()
    print("=" * 100)
    print("■ 2026 年的缺口明細（這些都在訓練切分之後，只影響驗證期尾端與每日推論）")
    print("=" * 100)
    for name, desc in SOURCES:
        p = P / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        c = _date_col(df)
        if c is None:
            continue
        d = pd.to_datetime(df[c], errors="coerce").dropna()
        hi = d.max()
        lag = (today - hi).days
        if lag > 10:
            print(f"  {name:<28} 停在 {hi.date()}（落後 {lag} 天）— {desc}")


if __name__ == "__main__":
    main()
