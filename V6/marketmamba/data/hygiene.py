"""
MarketMamba V6 — Data Hygiene
==============================
資料衛生層：宇宙過濾 + 每日資料健檢。

存在理由（2026-07-27 資料品質稽核）：三個「不會報錯」的問題各自安靜地跑了數週到數月，
共同原因是**寫入端與推論端都沒有健檢**：

  P1  prices_raw 每日約 1,550 列重複（2026-07-06 起）——多來源合併時 Date 型別不一致
      導致 drop_duplicates 失效。818 支股票的時序特徵被污染，2432 的 Return_5d
      含重複為 -0.53%、去重後 +1.07%（正負號相反）。
  P2  股票池 2026-05-22 的 2,321 支 → 05-25 的 1,968 支，一個交易日少 353 支
      （不可能是下市），來源切換造成，兩個月無人察覺。
  P3  14 個資料源停更 77–173 天（business_indicator 停在 2026-02-01），
      59 維中約 27 維實際上是凍結值或常數。

本模組提供的檢查刻意全部 **non-fatal**：只印數值與警告，不中斷推論。
線上 V6.1 每天要出訊號給人看，健檢不該成為新的失敗點。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ============================================================
# 宇宙過濾
# ============================================================

_COMMON_STOCK_RE = re.compile(r"^\d{4}$")

# 4 位數字、00 開頭者為 ETF/ETN（0050 0051 … 0081），不是普通股。
# 台股普通股代號區間為 1101–9958，不存在 00xx，故此規則精確無誤殺。
# 5–6 碼的 ETF（00631L、00679B 等）本來就被 ^\d{4}$ 擋掉。
_ETF_RE = re.compile(r"^00\d{2}$")

# 興櫃股票代號快取。stock_info 是「現況快照」，不是逐日的市場別歷史，
# 因此只讀一次即可（見 _emerging_ids 的限制說明）。
_EMERGING_CACHE: set[str] | None = None


def load_stock_info(latest_only: bool = True) -> pd.DataFrame:
    """
    讀取 `stock_info.parquet` 的**唯一權威入口**。所有消費端都應該走這裡（B-5）。

    【為什麼是「提供存取函式」而不是「在源頭刪重複列」】
      `stock_info` 是多次快照的累積（4,097 列 / 3,086 支），乍看是髒資料，
      但那些重複列其實**帶有資訊**：它們記錄了市場別的變遷
      （1563 巧新 2024-05 emerging → 2026-05 twse）。那是 PIT 相關的事實，
      直接去重會把它永久刪掉。

      所以源頭的修正不是刪資料，而是**把「正確的讀法」定義在一個地方**——
      消費端不必記得去重，也不會有人漏掉，而歷史仍然保留。
      （使用者 2026-07-28 的原則：「在源頭修 vs 靠每個消費端各自防禦，
        源頭修一次比較省事、也比較不會有人漏掉」。）

    【為什麼非做不可】未去重直接篩 `type == "emerging"` 會命中舊快照，
      把已轉上市櫃的公司整支排除——實測 31 支、其中 30 支仍在交易。

    Args:
        latest_only: True（預設）時每支只留最新快照。
                     False 回傳完整累積表，供需要市場別變遷史的分析使用。
    """
    from marketmamba.config import PROCESSED_DIR
    p = Path(PROCESSED_DIR) / "stock_info.parquet"
    if not p.exists():
        return pd.DataFrame()
    si = pd.read_parquet(p)
    si["stock_id"] = si["stock_id"].astype(str)
    if not latest_only:
        return si
    if "date" in si.columns:
        # na_position="first" 讓沒有日期的舊列墊底，被有日期的新列取代
        si["date"] = pd.to_datetime(si["date"], errors="coerce")
        si = si.sort_values("date", na_position="first")
    return si.drop_duplicates(subset=["stock_id"], keep="last").reset_index(drop=True)


def _emerging_ids() -> set[str]:
    """
    取得興櫃股票代號集合（stock_info.type == "emerging"）。

    ⚠️ **必須取每支股票的最新快照**（2026-07-29 修正的實質 bug）。
       `stock_info` 不是單一快照，而是**多次快照的累積**（`date` 欄有 2024-12-30、
       2026-05-03 等多批）。初版直接對整張表篩 `type == "emerging"` 而未去重，
       於是「舊快照是興櫃、後來已轉上市櫃」的股票被舊那一列命中，
       **整支被排除在宇宙之外**——實測 31 支，其中 30 支今天仍在交易
       （1563 巧新、2072 世紀風電、2248 華勝-KY 等 emerging → twse）。
       這不是「興櫃期的資料留著」那種小瑕疵，是把已上市公司整支弄丟。

    ⚠️ 殘留限制：即使取最新快照，本名單仍是**現況**而非 PIT——
       「目前仍是興櫃」者被全歷史排除（正確，它們從未上市櫃），
       但已轉上市櫃者的**興櫃期間資料**仍會留在宇宙裡。
       不過這一項在 2026-07-29 之後實質已解：新的 `prices_raw` 來自交易所直連，
       該股在交易所資料中的**首次出現日就是上市櫃日**，興櫃期本來就不在資料裡。

    背景：2026-05-25 資料源由 yfinance 切到交易所直連時，宇宙一日之間
    從 2,321 支掉到 1,968 支（少 353 支），其中 341 支正是興櫃——
    yfinance 涵蓋興櫃、交易所端點不涵蓋。那是一次**無聲的定義變更**，
    本函式把它變成全歷史一致的顯式規則。
    """
    global _EMERGING_CACHE
    if _EMERGING_CACHE is not None:
        return _EMERGING_CACHE
    ids: set[str] = set()
    try:
        from marketmamba.config import PROCESSED_DIR
        p = Path(PROCESSED_DIR) / "stock_info.parquet"
        if p.exists():
            si = load_stock_info(latest_only=True)          # ← 唯一權威入口（B-5）
            ids = set(si.loc[si["type"].astype(str).str.lower() == "emerging",
                             "stock_id"])
            logger.info(f"[hygiene] 興櫃名單載入 {len(ids):,} 支"
                        f"（stock_info 取每支最新快照 {len(si):,} 支）")
        else:
            logger.warning("[hygiene] 找不到 stock_info.parquet，興櫃無法排除")
    except Exception as e:                                        # noqa: BLE001
        logger.warning(f"[hygiene] 興櫃名單載入失敗，不排除興櫃：{e}")
    _EMERGING_CACHE = ids
    return ids


def filter_tradable_universe(
    df: pd.DataFrame,
    exclude_etf: bool = True,
    exclude_emerging: bool = True,
    col: str = "stock_id",
) -> pd.DataFrame:
    """
    把 raw 表過濾成「普通股宇宙」。

    Args:
        exclude_etf: True 時另外剔除 0050/0056 等 14 支 ETF。
                     它們在 stock_info 明確標為 type="ETF"，卻因為是 4 位數字而
                     通過了既有的 ^\\d{4}$ 過濾，會進入橫斷面統計與 Top50 排名。
        exclude_emerging: True 時剔除興櫃股（2026-07-28 決策）。興櫃流動性極低、
                     且新資料源本來就不涵蓋，留著只會造成 2026-05-25 那種
                     「宇宙定義無聲改變」的斷層。全歷史一致排除。
    """
    if df is None or df.empty or col not in df.columns:
        return df
    sid = df[col].astype(str)
    keep = sid.str.match(_COMMON_STOCK_RE)
    if exclude_etf:
        keep &= ~sid.str.match(_ETF_RE)
    if exclude_emerging:
        em = _emerging_ids()
        if em:
            keep &= ~sid.isin(em)
    return df[keep]


# ============================================================
# 每日健檢
# ============================================================

# 各 raw 相對於 prices_raw 的可容忍落後天數。超過就警示。
# 非日頻的資料源（月營收、季財報、月景氣燈號）門檻自然要放寬。
STALENESS_BUDGET_DAYS: dict[str, int] = {
    "institutional_raw":         3,
    "macro_raw":                 5,
    "margin_raw":                5,
    "per_raw":                   5,
    "market_value_raw":          5,
    "daytrade_raw":              5,
    "securities_raw":            5,
    "foreign_shareholding_raw": 10,
    "holdings_raw":             14,   # 集保週資料
    "fear_greed":                7,
    "futures_institutional_raw": 5,
    "options_institutional_raw": 5,
    "dividend_raw":             40,
    "revenue_raw":              45,   # 月營收，次月 10 日公布
    "financials_raw":          120,   # 季財報
    "business_indicator":       90,   # 月景氣燈號，落後約 2 個月公布
}

# 每日檔數相對前一交易日的容許變動率。P2 那種一天 -15% 會直接被抓到。
UNIVERSE_DELTA_TOLERANCE = 0.05

# 損壞列統計只看近端窗口。健檢每天跑，全檔掃描（prices 8.7M 列、institutional 32.8M 列）
# 對 24GB 筆電太重，而且推論當下真正該關心的是「今天有沒有出問題」。
HEALTH_WINDOW_DAYS = 90


# ------------------------------------------------------------
# parquet 輔助：只讀 metadata / 只讀需要的列，避免整檔進記憶體
# ------------------------------------------------------------

def _date_column(path: Path) -> tuple[str | None, object]:
    """回傳 (日期欄名, pyarrow 型別)。各 raw 不一致：prices/institutional 是
    large_string，margin/revenue 等是 timestamp[ns]，holdings 用 Week。"""
    schema = pq.read_schema(path)
    for c in ("Date", "date", "Week"):
        if c in schema.names:
            return c, schema.field(c).type
    return None, None


def _max_date(path: Path) -> pd.Timestamp | None:
    """用 parquet row-group 統計取最大日期——不讀任何實際資料，成本 O(row groups)。"""
    col, _ = _date_column(path)
    if col is None:
        return None
    pf = pq.ParquetFile(path)
    md = pf.metadata
    idx = pf.schema_arrow.names.index(col)
    best = None
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(idx).statistics
        if st is None or not st.has_min_max:
            continue
        v = pd.Timestamp(st.max)
        best = v if best is None or v > best else best
    if best is None:                                   # 沒有統計 → 退回只讀該欄
        best = pd.to_datetime(pd.read_parquet(path, columns=[col])[col]).max()
    return best


def _date_literal(dt: pd.Timestamp, arrow_type) -> object:
    """把界線值轉成與該欄型別相符的 filter 字面值（字串欄用 YYYY-MM-DD，
    時間戳欄用 Timestamp），否則 pyarrow 的 predicate pushdown 會失效或報錯。"""
    return dt.strftime("%Y-%m-%d") if "string" in str(arrow_type) else dt


def _read_window(path: Path, columns: list[str], since: pd.Timestamp) -> pd.DataFrame:
    """只讀 since 之後的列（pyarrow predicate pushdown，記憶體只裝得下窗口內資料）。"""
    col, typ = _date_column(path)
    df = pd.read_parquet(path, columns=columns,
                         filters=[(col, ">=", _date_literal(since, typ))])
    df[col] = pd.to_datetime(df[col])
    return df


def check_data_health(
    processed_dir: Path,
    universe_delta_tolerance: float = UNIVERSE_DELTA_TOLERANCE,
) -> dict:
    """
    對 Data/processed_v6/ 做一次資料健檢，把所有數字明確印出來（規則 7）。

    Returns:
        {"warnings": [...], "stats": {...}}  —— 呼叫端可自行決定要不要顯示在推論視窗。
    """
    warnings: list[str] = []
    stats: dict = {}

    price_path = processed_dir / "prices_raw.parquet"
    if not price_path.exists():
        logger.warning("[健檢] prices_raw.parquet 不存在，跳過健檢")
        return {"warnings": ["prices_raw.parquet 不存在"], "stats": {}}

    ref_date = _max_date(price_path)                   # 只讀 metadata
    since = ref_date - pd.Timedelta(days=HEALTH_WINDOW_DAYS)
    pr = _read_window(price_path, ["Date", "stock_id", "Close"], since)
    stats["reference_date"] = str(ref_date.date())
    stats["window_days"] = HEALTH_WINDOW_DAYS

    logger.info("=" * 62)
    logger.info(f"[健檢] 基準日 = {ref_date.date()}（損壞列統計窗口：近 "
                f"{HEALTH_WINDOW_DAYS} 天 = {len(pr):,} 列）")
    logger.info("=" * 62)

    # ── 1. 各資料源時效 ────────────────────────────────────────────────
    logger.info("[健檢 1/4] 資料源時效")
    stale = {}
    for name, budget in STALENESS_BUDGET_DAYS.items():
        p = processed_dir / f"{name}.parquet"
        if not p.exists():
            warnings.append(f"{name}.parquet 不存在")
            logger.warning(f"  {name:<28} ❌ 檔案不存在")
            continue
        try:
            mx = _max_date(p)
        except Exception as e:                              # noqa: BLE001
            logger.warning(f"  {name:<28} ⚠ 讀取失敗：{e}")
            continue
        if mx is None:
            continue
        lag = (ref_date - mx).days
        stale[name] = lag
        if lag > budget:
            warnings.append(f"{name} 停更 {lag} 天（容許 {budget}）")
            logger.warning(f"  {name:<28} {str(mx.date()):<12} 落後 {lag:>4} 天  "
                           f"⚠️ 超過容許值 {budget} 天")
        else:
            logger.info(f"  {name:<28} {str(mx.date()):<12} 落後 {lag:>4} 天  ✓")
    stats["staleness_days"] = stale

    # ── 2. 股票池規模變動 ──────────────────────────────────────────────
    logger.info("[健檢 2/4] 股票池規模")
    per_day = pr.groupby("Date")["stock_id"].nunique().sort_index()
    stats["universe_today"] = int(per_day.iloc[-1])
    if len(per_day) >= 2:
        today_n, prev_n = int(per_day.iloc[-1]), int(per_day.iloc[-2])
        delta = (today_n - prev_n) / max(prev_n, 1)
        stats["universe_prev"] = prev_n
        stats["universe_delta_pct"] = round(delta * 100, 2)
        line = (f"  今日 {today_n} 支 | 前一交易日 {prev_n} 支 | "
                f"變動 {delta:+.2%}")
        if abs(delta) > universe_delta_tolerance:
            warnings.append(f"股票池單日變動 {delta:+.1%}（{prev_n} → {today_n}）")
            logger.warning(line + f"  ⚠️ 超過容許值 ±{universe_delta_tolerance:.0%}")
        else:
            logger.info(line + "  ✓")
        recent = per_day.tail(6)
        logger.info("  近 6 個交易日：" +
                    ", ".join(f"{d.date()}={int(n)}" for d, n in recent.items()))

        # ── 交易日曆缺口 ────────────────────────────────────────────
        # 既有的「停更天數」只看最新日期，中間漏掉的交易日完全看不出來——
        # 而 prices_raw 缺過 2026-04-27/04-28 兩個真實交易日，是靠回補 margin 時
        # 意外發現的。本檢查用「連續兩筆資料之間的工作日間隔」找洞：
        # 台股最長連假（春節）約 9 天，故 >5 個工作日的空檔值得看一眼。
        # 只掃近一年，避免把久遠的歷史缺口每天重報一次。
        try:
            days = per_day.index[per_day.index >= per_day.index.max()
                                 - pd.Timedelta(days=365)]
            if len(days) >= 2:
                s = pd.Series(days)
                # 兩個相鄰交易日之間的工作日數（扣掉週末）
                bd = np.busday_count(
                    s.iloc[:-1].values.astype("datetime64[D]"),
                    s.iloc[1:].values.astype("datetime64[D]"))
                holes = [(s.iloc[i], s.iloc[i + 1], int(bd[i]))
                         for i in range(len(bd)) if bd[i] > 5]
                if holes:
                    warnings.append(f"交易日曆有 {len(holes)} 處 >5 個工作日的空檔")
                    logger.warning(
                        f"  ⚠️ 近一年有 {len(holes)} 處交易日空檔（可能漏抓，"
                        f"也可能是連假）：" +
                        "、".join(f"{a.date()}→{b.date()}({n} 工作日)"
                                  for a, b, n in holes[:5]))
                else:
                    logger.info("  近一年交易日曆無 >5 工作日的空檔  ✓")
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"  交易日曆缺口檢查失敗（不影響推論）：{e}")

    # ── 3. prices_raw 資料損壞 ─────────────────────────────────────────
    logger.info("[健檢 3/4] prices_raw 資料損壞")
    n_dup = int(pr.duplicated(subset=["Date", "stock_id"]).sum())
    dup_today = int(pr[pr["Date"] == ref_date].duplicated(subset=["stock_id"]).sum())
    close = pd.to_numeric(pr["Close"], errors="coerce")
    n_bad_close = int((close <= 0).sum())
    bad_today = int(((pr["Date"] == ref_date) & (close <= 0)).sum())
    stats.update(duplicate_rows_window=n_dup, duplicate_rows_today=dup_today,
                 nonpositive_close_window=n_bad_close, nonpositive_close_today=bad_today)
    _w = f"近 {HEALTH_WINDOW_DAYS} 天"

    if dup_today or n_dup:
        warnings.append(f"重複列：今日 {dup_today} 列、{_w} {n_dup:,} 列")
        logger.warning(f"  (Date, stock_id) 重複：今日 {dup_today} 列 / "
                       f"{_w} {n_dup:,} 列  ⚠️")
    else:
        logger.info(f"  (Date, stock_id) 重複：今日 0 列 / {_w} 0 列  ✓")

    if bad_today or n_bad_close:
        warnings.append(f"Close<=0：今日 {bad_today} 列、{_w} {n_bad_close:,} 列")
        logger.warning(f"  Close <= 0：今日 {bad_today} 列 / {_w} {n_bad_close:,} 列  ⚠️")
    else:
        logger.info(f"  Close <= 0：今日 0 列 / {_w} 0 列  ✓")

    # ── 4. 法人資料覆蓋率 ──────────────────────────────────────────────
    logger.info("[健檢 4/4] 法人資料覆蓋率")
    inst_path = processed_dir / "institutional_raw.parquet"
    if inst_path.exists():
        # 只讀基準日當天（institutional_raw 有 32.8M 列，整檔載入會吃掉數 GB）
        inst = _read_window(inst_path, ["Date", "stock_id"], ref_date)
        # ⚠️ 分母要用「推論實際使用的宇宙」，不是原始 prices 宇宙。
        # 2026-07-29 切換到官方還原價後，prices 從 1,948 支變 2,100 支，
        # 多出來的絕大多數是 ETF（133 支）——交易所本來就不出個股法人明細給它們，
        # 用原始宇宙當分母會讓覆蓋率從 96.1% 假摔到 89.5%，掛上一個永遠不會消的警報。
        # 套用與 run_daily_inference 相同的過濾後，實際覆蓋率是 96.2%。
        day = pr.loc[pr["Date"] == ref_date, ["stock_id"]]
        price_ids = set(filter_tradable_universe(
            day, exclude_etf=True, exclude_emerging=True)["stock_id"])
        inst_ids = set(inst.loc[inst["Date"] == ref_date, "stock_id"])
        hit = len(price_ids & inst_ids)
        cover = hit / max(len(price_ids), 1)
        stats["institutional_coverage"] = round(cover, 4)
        line = (f"  可交易宇宙 {len(price_ids)} 支 | 法人命中 {hit} 支 | "
                f"覆蓋率 {cover:.1%} | 缺 {len(price_ids) - hit} 支")
        if cover < 0.90:
            warnings.append(f"法人覆蓋率僅 {cover:.1%}")
            logger.warning(line + "  ⚠️（缺的股票 Foreign_Net 等 5 欄會被補 0，"
                                  "模型分不出『沒買賣』與『沒資料』）")
        else:
            logger.info(line + "  ✓")

    logger.info("=" * 62)
    if warnings:
        logger.warning(f"[健檢] 共 {len(warnings)} 項警告：")
        for w in warnings:
            logger.warning(f"  ⚠️ {w}")
    else:
        logger.info("[健檢] 全部通過 ✓")
    logger.info("=" * 62)

    return {"warnings": warnings, "stats": stats}
