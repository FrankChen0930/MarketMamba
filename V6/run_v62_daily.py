"""
run_v62_daily.py — V6.2 每日一鍵：抓資料 → 檢查 → 三個模型推論 → 組合層 → 告警
================================================================================
**自給自足**：自己呼叫 `run_daily_update()` 抓資料，不依賴 V6.1。
（2026-08-05 使用者決定退役 V6.1 → V6.2 不能再吃 V6.1 更新好的 raw parquet。）

⚠️ **過渡期（V6.1 還在跑的那幾天）**
   兩支都會呼叫 `run_daily_update()` 寫同一批 parquet。`_append_to_parquet`
   有去重保險，重複抓是安全的，但**兩者不可同時執行**（同時寫 parquet ＋
   記憶體雙倍）。排程要錯開：V6.1 21:30 起跑約 33 分 → V6.2 排 22:15。
   若想在過渡期省一次抓取，V6.2 可加 `--no-fetch`。
   **V6.1 退役後把 `--no-fetch` 拿掉即可，不需要改任何程式。**

為什麼要有這一支
----------------
① **特徵矩陣建一次、三個模型共用**。矩陣建構才是成本（約 3–5 分），
   GPU 前向每個 arm 只要 1 秒 → 並行跑多模型的邊際成本幾乎是零。
   這正是「多模型並行累積實戰紀錄」可行的原因。
② **當日資料檢查**。既有健檢的容許值是「停更 5 天內算 ✓」，
   對每日源來說太寬——margin 落後 1 天會顯示 ✓，但那正是 2026-08-05
   查出來的真問題（19:35 抓太早、被迫 ffill 昨天的值）。
   這裡改成**每日源必須有當日資料**，否則告警。
③ **完整性隨當天的持股一起落檔**。幾個月後看到某段表現差，要能區分
   「模型不好」與「那幾天資料缺了」。事後補不回來，所以當天就要記。

⚠️ 執行時間：**21:30 之後**。實測 2026-08-05：19:35 時 TWSE 的 margin 與
   daytrade 都還「尚未公布」，21:11 才有當日資料。跑太早會靜默用到昨天的值
   （`_merge_margin` 會 ffill，不會報錯）。

用法（WSL）
-----------
    python V6/run_v62_daily.py              # 全部 arm
    python V6/run_v62_daily.py --arms v2_kg_nomacro
    python V6/run_v62_daily.py --first-day  # 上線第一天：強制建倉
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

_V6 = Path(__file__).resolve().parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v62_daily")

# 每日源：**必須有當日資料**（容許 0 天）。
# 週/月/季源不列入——它們本來就不是每天更新，列進來只會製造永遠不消的假警報，
# 而長期假警報會訓練人忽略警報，比沒有警報更危險（2026-07-29 已記過這個教訓）。
DAILY_SOURCES = [
    "prices_raw", "institutional_raw", "margin_raw", "daytrade_raw",
    "per_raw", "market_value_raw", "securities_raw", "foreign_shareholding_raw",
    "futures_institutional_raw", "options_institutional_raw",
]
# 已知停更且**已決定不修**的源（Group D 已證實為負貢獻）→ 不檢查、不告警
KNOWN_STALE = {"macro_raw", "fear_greed", "business_indicator"}


# ============================================================
# 1. 當日資料檢查
# ============================================================
def check_freshness() -> tuple[dict, list[str]]:
    """回 ({source: 是否有當日資料}, 缺漏清單)。只讀 parquet metadata，很便宜。"""
    from marketmamba.config import PROCESSED_DIR
    from marketmamba.data.hygiene import _max_date

    import pandas as pd
    ref = _max_date(Path(PROCESSED_DIR) / "prices_raw.parquet")
    complete, missing = {}, []
    logger.info("=" * 62)
    logger.info(f"[檢查] 當日資料完整性（基準日 = {ref.date()}）")
    logger.info("=" * 62)
    for name in DAILY_SOURCES:
        p = Path(PROCESSED_DIR) / f"{name}.parquet"
        if not p.exists():
            complete[name] = False
            missing.append(f"{name}（檔案不存在）")
            logger.warning(f"  {name:<30} ❌ 檔案不存在")
            continue
        try:
            mx = _max_date(p)
        except Exception as e:                                  # noqa: BLE001
            complete[name] = False
            missing.append(f"{name}（讀取失敗）")
            logger.warning(f"  {name:<30} ⚠ 讀取失敗：{e}")
            continue
        lag = (ref - mx).days if mx is not None else 999
        ok = lag <= 0
        complete[name] = bool(ok)
        if ok:
            logger.info(f"  {name:<30} {str(mx.date()):<12} ✓ 當日")
        else:
            missing.append(f"{name}（落後 {lag} 天，停在 {mx.date()}）")
            logger.warning(f"  {name:<30} {str(mx.date()):<12} ❌ 落後 {lag} 天"
                           f"{'  ← 是不是跑太早了？margin/daytrade 約 21:00 才公布' if lag == 1 else ''}")
    logger.info(f"[檢查] {sum(complete.values())}/{len(DAILY_SOURCES)} 個每日源有當日資料"
                f"{'  ✓ 全數完整' if not missing else ''}")
    return complete, missing


# ============================================================
# 2. Telegram 告警
# ============================================================
def notify(title: str, body: str) -> bool:
    """
    有設 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 就發，沒設就印在 log 上。

    刻意不 raise：告警機制自己壞掉，不該讓整個每日流程失敗
    （健檢一律 non-fatal，2026-07-27 已定的原則）。
    """
    msg = f"*{title}*\n{body}"
    logger.warning(f"[告警] {title}\n{body}")
    try:
        from dotenv import load_dotenv
        load_dotenv(_V6 / ".env")
    except Exception:                                           # noqa: BLE001
        pass
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        logger.warning("[告警] 未設 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID → "
                       "只印在 log。要收到手機通知請加進 V6/.env")
        return False
    try:
        import httpx
        r = httpx.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                       json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
                       timeout=15)
        ok = r.status_code == 200
        logger.info(f"[告警] Telegram {'已送出' if ok else f'失敗 {r.status_code}'}")
        return ok
    except Exception as e:                                      # noqa: BLE001
        logger.error(f"[告警] Telegram 例外（不影響流程）：{e}")
        return False


# ============================================================
# 3. 資料更新（V6.2 自己抓，不再依賴 V6.1）
# ============================================================
def fetch_data(retry_wait_min: int = 10) -> tuple[dict, list[str]]:
    """
    呼叫與 V6.1 **同一個** `run_daily_update()`——不另寫一份抓取邏輯。
    那支累積了 13 條交易所 API 雷區的處理（TPEX 斜線日期、Big5 編碼、
    非交易日回 HTML、TWSE/TPEX 欄序相反……），重寫一份等於把那些坑再踩一次。

    抓完用**本檔的嚴格判準**（每日源必須有當日資料）複查；缺的話等一次再抓。
    只重試一次：22:15 起跑，一次 10 分鐘的等待還在合理範圍，
    而真正的保險是 Telegram 告警——重試無限次只會把問題藏起來。
    """
    import time

    from marketmamba.data.fetcher import run_daily_update

    for attempt in (1, 2):
        logger.info(f"[抓取] run_daily_update（第 {attempt} 次）…")
        try:
            fr = run_daily_update()
            fwd = fr.get("forward_filled", [])
            if fwd:
                logger.warning(f"[抓取] forward-fill：{'、'.join(fwd)}")
        except Exception as e:                                  # noqa: BLE001
            logger.error(f"[抓取] 失敗：{e}")
            if attempt == 2:
                raise
            time.sleep(retry_wait_min * 60)
            continue

        complete, missing = check_freshness()
        if not missing or attempt == 2:
            return complete, missing
        logger.warning(f"[抓取] 仍缺 {len(missing)} 個源，{retry_wait_min} 分後重試一次…")
        time.sleep(retry_wait_min * 60)
    return check_freshness()


# ============================================================
# 4. 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="預設全部")
    ap.add_argument("--first-day", action="store_true",
                    help="上線第一天：強制建倉（否則要等距上次 20 個交易日）")
    ap.add_argument("--skip-check", action="store_true", help="跳過資料檢查（不建議）")
    ap.add_argument("--no-fetch", action="store_true",
                    help="不自己抓資料（V6.1 還在跑、且已經抓過時用）")
    a = ap.parse_args()

    t0 = datetime.now()
    import run_v62_inference as R
    import v62_portfolio as P

    arms = a.arms or list(R.ARMS)
    if a.skip_check:
        complete, missing = {}, []
    elif a.no_fetch:
        complete, missing = check_freshness()
    else:
        complete, missing = fetch_data()

    if missing:
        notify("⚠️ V6.2 當日資料缺漏",
               "以下每日源沒有當天的資料：\n• " + "\n• ".join(missing) +
               "\n\n分數仍會產生，但那幾欄是 ffill 昨天的值。"
               "\n若 margin/daytrade 落後 1 天 → 多半是跑太早（約 21:00 才公布）。")

    # 特徵矩陣建一次，三個 arm 共用（矩陣是成本大宗，前向只要 1 秒）
    logger.info("[2/4] 建特徵矩陣（三個模型共用）…")
    import pandas as pd
    df = R.build_feature_df()
    df["Date"] = pd.to_datetime(df["Date"])
    date = df["Date"].max().strftime("%Y-%m-%d")
    logger.info(f"[2/4] ✓ 完成｜交易日 {date}")

    failed = []
    for arm in arms:
        try:
            logger.info(f"[3/4] 推論 arm={arm} …")
            out = R.infer(df, date, arm=arm)
            spec = R.ARMS[arm]
            R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out.to_csv(R.RESULTS_DIR / f"{spec.out_name}.csv", index=False)
            arch = R.RESULTS_DIR / "archive"
            arch.mkdir(parents=True, exist_ok=True)
            out.to_csv(arch / f"{spec.out_name}_{date}.csv", index=False)

            logger.info(f"[4/4] 組合層 arm={arm} …")
            P.step(arm, R.RESULTS_DIR / f"{spec.out_name}.csv",
                   data_complete=complete, force_rebalance=a.first_day)
        except Exception as e:                                  # noqa: BLE001
            failed.append(arm)
            logger.error(f"arm={arm} 失敗：{e}\n{traceback.format_exc()[:1200]}")

    el = (datetime.now() - t0).total_seconds() / 60
    if failed:
        notify("❌ V6.2 推論失敗",
               f"日期 {date}\n失敗的模型：{', '.join(failed)}\n"
               f"成功：{', '.join(x for x in arms if x not in failed) or '無'}\n"
               f"耗時 {el:.1f} 分。詳見 V6/logs/。")
        return 1

    logger.info(f"✅ V6.2 完成｜{date}｜{len(arms)} 個模型｜{el:.1f} 分")
    if missing:
        logger.warning(f"⚠️ 但當日資料有 {len(missing)} 個源缺漏，已記進 state 的 data_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
