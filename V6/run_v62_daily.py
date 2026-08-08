"""
run_v62_daily.py — V6.2 每日一鍵：抓資料 → 檢查 → 推論 → 多頻率組合層 → 告警
================================================================================
兩階段（2026-08-08 起）：
  [3/4] **分數 arm**（模型 × 預測頭）→ 每個跑一次 GPU 前向，寫一份 csv
  [4/4] **組合 arm**（分數 × n/k/freq）→ 純 CPU，一份分數可餵給多個再平衡頻率

再平衡率是組合層參數、不是模型參數，所以 12 個組合只需要 4 份分數。
先去重再前向——否則 5 個頻率會讓同一顆 checkpoint 白白前向 5 次。

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
① **特徵矩陣建一次、所有 arm 共用**。矩陣建構才是成本（約 3–5 分），
   GPU 前向每個 arm **實測 1.5 秒**（2026-08-08 量的）→ 並行跑多模型的
   邊際成本幾乎是零。這正是「多模型並行累積實戰紀錄」可行的原因。
   組合層更便宜：完全不碰 GPU，一份分數可同時餵給 5 個再平衡頻率。
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
    python V6/run_v62_daily.py                          # 全部組合（12 個）
    python V6/run_v62_daily.py --portfolios v2_kg_nomacro_f20   # 只跑主線
    python V6/run_v62_daily.py --first-day              # 上線第一天：強制建倉
    python V6/v62_portfolio.py --list                   # 看有哪些組合 arm
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
# 3a. B 類經典模型（Ridge / GBDT / GRU）—— 獨立 process
# ============================================================
def run_baselines(date: str, arms: list[str] | None = None) -> list[str]:
    """呼叫 `run_v62_baselines.py`，回傳失敗的 arm 清單。

    ⚠️ **必須是獨立 process，不能 import 進來跑。**
       Mamba 線把 config 切成 59 維、baseline 線切成 66 維（`MM_PROTOCOL=v2`），
       而 config patch 是 module 級全域、在 import 當下就綁進各模組的預設參數。
       同一個 process 混跑會**靜默**吃到錯的維度（CLAUDE.md 記過 47 維那次）。

    ⚠️ 失敗不影響 Mamba 線：baseline 是對照組，掛掉不該讓主線的持股紀錄中斷。
    """
    import subprocess

    script = _V6 / "run_v62_baselines.py"
    if not script.exists():
        logger.warning(f"[baseline] 找不到 {script.name} → 跳過")
        return ["ridge", "gbdt", "gru"]

    cmd = [sys.executable, str(script)]
    if arms:
        cmd += ["--arms", *arms]
    env = {**os.environ, "MM_PROTOCOL": "v2"}
    logger.info(f"[baseline] 另起 process（MM_PROTOCOL=v2）：{' '.join(cmd[1:])}")
    t0 = datetime.now()
    r = subprocess.run(cmd, cwd=str(_V6.parent), env=env,
                       capture_output=True, text=True, errors="replace")
    el = (datetime.now() - t0).total_seconds() / 60
    # 規則 7：子行程的關鍵輸出要看得見，不能只看 returncode
    for line in (r.stdout or "").splitlines():
        if any(k in line for k in ("[ridge]", "[gbdt]", "[gru]", "[flat]",
                                   "[macro]", "[panel]", "❌", "✅")):
            logger.info(f"  │ {line.strip()[:160]}")
    if r.returncode != 0:
        logger.error(f"[baseline] 失敗（exit {r.returncode}，{el:.1f} 分）\n"
                     f"{(r.stderr or '')[-1500:]}")
        return ["ridge", "gbdt", "gru"]
    logger.info(f"[baseline] ✓ 完成（{el:.1f} 分）")
    return []


# ============================================================
# 3b. 推送結果到 GitHub
# ============================================================
def push_to_github(date: str, max_attempts: int = 3, retry_sec: int = 10) -> bool:
    """
    `git add V6/results/` → commit → push origin main → 清 V6.2 後端快取。

    **為什麼不 import V6.1 的 `_push_to_github`**：那支住在 `run_daily_inference.py`，
    而 V6.1 就要退役了——依賴一個即將被刪掉的模組，等於幫自己埋一顆定時炸彈。
    這裡是四十行沒有領域陷阱的 git 呼叫（不像 fetcher 那種累積了 13 條交易所
    雷區的東西），自帶一份的代價遠低於耦合的代價。本檔的設計原則就是自給自足。

    ⚠️ **非致命**：推不上去不該讓當天的持股紀錄作廢——state 與 jsonl 已經
       落在本機，補推即可。但**一定要告警**，否則 dashboard 會安靜地停在舊資料。

    ⚠️ 過渡期：V6.1（21:30）也會 `git add V6/results/`。兩者相隔 45 分鐘、
       V6.1 全鏈路實測約 33 分，正常不會撞到 git index.lock。若真的撞到，
       這裡的重試（10 秒 × 3）足以等過去。
    """
    import subprocess

    repo = _V6.parent                      # V6/ → MarketMamba/（.git 在這裡）
    # WSL2：跨 /mnt 檔案系統邊界時 git 找不到 repo，要這個環境變數
    env = {**os.environ, "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1"}

    def _git(*args, check=True):
        return subprocess.run(["git", *args], cwd=repo, check=check,
                              capture_output=True, env=env)

    try:
        # 整個 results/ 一起加——新 arm 會帶進新檔名（v62_state_*.json /
        # v62_portfolio_*.jsonl），逐檔列舉會在檔案還不存在時炸掉（V6.1 踩過）
        _git("add", "V6/results/")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            logger.info("[push] results 沒有變動 → 跳過（已是最新）")
            return True
        _git("commit", "-m", f"v62: daily portfolio {date}")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else str(e)
        logger.error(f"[push] git add/commit 失敗：{err.strip()[:300]}")
        return False

    import time
    for attempt in range(1, max_attempts + 1):
        try:
            _git("push", "origin", "main")
            logger.info(f"✅ [push] 已推送到 GitHub"
                        f"{f'（第 {attempt} 次嘗試）' if attempt > 1 else ''}")
            _refresh_backend_cache()
            return True
        except subprocess.CalledProcessError as e:
            err = (e.stderr.decode(errors="replace") if e.stderr else str(e)).strip()[:200]
            if attempt < max_attempts:
                logger.warning(f"[push] 第 {attempt}/{max_attempts} 次失敗，"
                               f"{retry_sec} 秒後重試：{err}")
                time.sleep(retry_sec)
            else:
                logger.error(f"[push] 重試 {max_attempts} 次仍失敗：{err}\n"
                             f"   結果在本機，手動補推："
                             f"git add V6/results/ && git commit -m 'v62 {date}' && git push")
    return False


def _refresh_backend_cache() -> None:
    """清 Render 上的 V6.2 快取（否則要等 1h TTL 才看得到今天的持股）。

    ⚠️ 打的是 **`/api/v62/cache/refresh`**，不是 V6.1 的 `/api/signals/cache/refresh`
       ——兩個 router 各有各的 cache dict，清錯那個對 V6.2 沒有任何作用。
    """
    import urllib.request

    base = os.getenv("RENDER_BACKEND_URL", "").rstrip("/")
    if not base:
        logger.info("[push] 未設 RENDER_BACKEND_URL → 略過快取刷新（最多等 1h TTL）")
        return
    url = f"{base}/api/v62/cache/refresh"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="POST"), timeout=15) as r:
            logger.info(f"📡 [push] V6.2 後端快取已刷新：{r.status}")
    except Exception as e:                                      # noqa: BLE001
        logger.warning(f"[push] 快取刷新失敗（非致命，1h 後自然過期）：{e}")


# ============================================================
# 4. 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="強制指定分數 arm（預設由 --portfolios 反推）")
    ap.add_argument("--portfolios", nargs="*", default=None,
                    help="要跑的組合 arm，預設全部（見 v62_portfolio.py --list）")
    ap.add_argument("--first-day", action="store_true",
                    help="上線第一天：強制建倉（否則要等距上次 20 個交易日）")
    ap.add_argument("--skip-check", action="store_true", help="跳過資料檢查（不建議）")
    ap.add_argument("--no-fetch", action="store_true",
                    help="不自己抓資料（V6.1 還在跑、且已經抓過時用）")
    ap.add_argument("--no-ui", action="store_true", help="不開進度視窗")
    ap.add_argument("--skip-push", action="store_true",
                    help="不推送到 GitHub（測試用；dashboard 不會更新）")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="不跑 B 類經典模型（Ridge/GBDT/GRU，另一個 process 約 4 分）")
    a = ap.parse_args()

    if a.no_ui:
        return _pipeline(a, None)
    from progress_window import ProgressWindow
    ui = ProgressWindow(
        ["抓取資料", "建特徵矩陣", "模型推論", "組合層", "推送 GitHub"],
        ["Fetch data", "Build features", "Inference", "Portfolio", "Push"])
    return ui.run(lambda u: _pipeline(a, u))


def _pipeline(a, ui) -> int:
    """實際流程。`ui` 可為 None（--no-ui 或無 DISPLAY）→ 所有 UI 呼叫變 no-op。"""
    def _ui(idx: int, status: str, note: str = "") -> None:
        if ui is not None:
            ui.update(idx, status, note)

    t0 = datetime.now()
    import run_v62_inference as R
    import v62_portfolio as P

    # 要跑哪些組合 → 反推需要哪些分數 arm（去重，保持宣告順序）
    pfs = a.portfolios or list(P.PORTFOLIOS)
    unknown = [x for x in pfs if x not in P.PORTFOLIOS]
    if unknown:
        logger.error(f"未知的組合 arm：{unknown}（可用：{list(P.PORTFOLIOS)}）")
        return 2
    need = {P.PORTFOLIOS[x].score_arm for x in pfs}
    arms = a.arms or [x for x in R.ARMS if x in need]
    logger.info(f"[計畫] {len(pfs)} 個組合 ← {len(arms)} 份分數：{', '.join(arms)}")

    if a.skip_check:
        complete, missing = {}, []
    elif a.no_fetch:
        complete, missing = check_freshness()
    else:
        _ui(0, "running")
        complete, missing = fetch_data()
    _ui(0, "done" if not missing else "skipped",
        f"{len(missing)} 個源缺漏" if missing else "全數當日")

    if missing:
        notify("⚠️ V6.2 當日資料缺漏",
               "以下每日源沒有當天的資料：\n• " + "\n• ".join(missing) +
               "\n\n分數仍會產生，但那幾欄是 ffill 昨天的值。"
               "\n若 margin/daytrade 落後 1 天 → 多半是跑太早（約 21:00 才公布）。")

    # 特徵矩陣建一次、所有 arm 共用（矩陣是成本大宗，前向實測只要 1.5 秒）
    logger.info(f"[2/4] 建特徵矩陣（{len(arms)} 份分數共用）…")
    _ui(1, "running")
    import pandas as pd
    df = R.build_feature_df()
    df["Date"] = pd.to_datetime(df["Date"])
    date = df["Date"].max().strftime("%Y-%m-%d")
    logger.info(f"[2/4] ✓ 完成｜交易日 {date}")
    _ui(1, "done", f"{len(df):,} 列")
    if ui is not None:
        ui.set_info(f"{date}｜{len(arms)} 份分數 / {len(pfs)} 個組合")

    # ── [3/4] 推論：每個**分數 arm** 只跑一次前向 ─────────────────────
    #    多個組合 arm 可能共用同一份分數（同分數 × 不同再平衡率），
    #    所以先去重再前向——否則 5 個頻率會讓同一顆 checkpoint 前向 5 次。
    failed_arms: list[str] = []
    score_csv: dict[str, Path] = {}
    _ui(2, "running")
    for arm in arms:
        try:
            logger.info(f"[3/4] 推論 arm={arm} …")
            out = R.infer(df, date, arm=arm)
            spec = R.ARMS[arm]
            R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            csv = R.RESULTS_DIR / f"{spec.out_name}.csv"
            out.to_csv(csv, index=False)
            arch = R.RESULTS_DIR / "archive"
            arch.mkdir(parents=True, exist_ok=True)
            out.to_csv(arch / f"{spec.out_name}_{date}.csv", index=False)
            score_csv[arm] = csv
        except Exception as e:                                  # noqa: BLE001
            failed_arms.append(arm)
            logger.error(f"arm={arm} 失敗：{e}\n{traceback.format_exc()[:1200]}")

    # ── B 類經典模型（獨立 process，自己建 66 維面板，約 4 分）──
    _BASELINE_ARMS = {"ridge", "gbdt", "gru"}
    need_bl = sorted(_BASELINE_ARMS & {P.PORTFOLIOS[x].score_arm for x in pfs})
    if need_bl and not a.skip_baselines:
        for nm in run_baselines(date, need_bl):
            failed_arms.append(nm)
        for nm in need_bl:
            if nm not in failed_arms:
                score_csv[nm] = R.RESULTS_DIR / f"df_v62_{nm}.csv"
    elif need_bl:
        logger.info("[baseline] --skip-baselines：略過 B 類")
        failed_arms.extend(need_bl)

    _ui(2, "failed" if failed_arms else "done",
        f"{len(arms) + len(need_bl) - len(failed_arms)}/{len(arms) + len(need_bl)} 成功")

    # ── [4/4] 組合層：一份分數餵給多個頻率的狀態機 ─────────────────────
    #    再平衡率是組合層參數、不是模型參數 → 這一段完全不碰 GPU。
    _ui(3, "running")
    failed_pf: list[str] = []
    for name in pfs:
        p = P.PORTFOLIOS[name]
        if p.score_arm not in score_csv:
            failed_pf.append(name)
            logger.error(f"組合 {name} 跳過：分數 arm={p.score_arm} 沒有產出")
            continue
        try:
            logger.info(f"[4/4] 組合層 {name}（{p.score_arm} × {p.freq} 日, {p.tier}）…")
            P.step(name, score_csv[p.score_arm], data_complete=complete,
                   force_rebalance=a.first_day, n=p.n, k=p.k, freq=p.freq,
                   tier=p.tier)
        except Exception as e:                                  # noqa: BLE001
            failed_pf.append(name)
            logger.error(f"組合 {name} 失敗：{e}\n{traceback.format_exc()[:1200]}")

    _ui(3, "failed" if failed_pf else "done",
        f"{len(pfs) - len(failed_pf)}/{len(pfs)} 成功")

    # 發布 arm 清單給後端（PORTFOLIOS 是唯一真相，router 不再自帶一份）
    P.write_manifest()

    # ── [5/5] 推送：沒有這一步，結果只留在本機，dashboard 永遠停在舊資料 ──
    if a.skip_push:
        logger.info("[5/5] --skip-push：略過推送（dashboard 不會更新）")
        _ui(4, "skipped", "--skip-push")
        pushed = True
    else:
        logger.info("[5/5] 推送到 GitHub …")
        _ui(4, "running")
        pushed = push_to_github(date)
        _ui(4, "done" if pushed else "failed",
            "已推送" if pushed else "推送失敗，結果在本機")

    el = (datetime.now() - t0).total_seconds() / 60
    if failed_arms or failed_pf or not pushed:
        notify("❌ V6.2 部分失敗",
               f"日期 {date}\n"
               f"推論失敗：{', '.join(failed_arms) or '無'}\n"
               f"組合層失敗：{', '.join(failed_pf) or '無'}\n"
               f"推送：{'✅' if pushed else '❌ 失敗 — 持股已存在本機，需手動補推'}\n"
               f"耗時 {el:.1f} 分。詳見 V6/logs/。")
        return 1

    # ⚠️ 收尾這行不可寫死「已推送」——`--skip-push` 時那是假話，
    #    而幾個月後回頭查 log 時，沒有人分得出來當天到底推了沒。
    logger.info(f"✅ V6.2 完成｜{date}｜{len(score_csv)} 份分數 × {len(pfs)} 個組合"
                f"｜{'略過推送' if a.skip_push else '已推送'}｜{el:.1f} 分")
    if missing:
        logger.warning(f"⚠️ 但當日資料有 {len(missing)} 個源缺漏，已記進 state 的 data_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
