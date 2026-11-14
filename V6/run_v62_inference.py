"""
run_v62_inference.py — V6.2 每日推論（`v2_kg_nomacro`，規格 `5d/20`）
=====================================================================
把 F6 2×2 最佳格（no_macro + v2 圖、訊號層 IC +0.1145、組合層 +38.0%/Sharpe 1.713）
接到本機每日推論路徑上，產出**與回測同口徑**的每日分數。

⚠️ 完全獨立、純附加
-------------------
不改 `run_daily_inference.py`（V6.1）與 `run_dual_inference.py`（雙模型）——
那兩條每天在跑。本檔在自己的 process 內把 config 切成 59 維，輸出獨立檔名。

與 `run_dual_inference.py` 的五個差異（每一個都會靜默算錯，故逐條標註來源）
--------------------------------------------------------------------------
 ① `build_features(fundamentals_v2=True)`  —— dual 沒傳 → 預設 False。
    對照 `experimental/baseline_common.py:396`（PROTOCOL["FUNDAMENTALS_V2"]=True）。
 ② KG 換 `knowledge_graph_v2.npz` —— dual 吃 `trainer.KG_CACHE_PATH`（舊圖）。
    對照 `experimental/score_mamba_local.py:273-279`。
 ③ **Group D 那 12 欄輸入端歸零** —— dual 完全沒做。checkpoint 是在 Group D
    全 0 的輸入上訓練的，餵沒歸零的 macro 進去**維度相同、不會報錯**，但推論
    輸入與訓練輸入不一致。這是 `macro_norm`／`fundamentals_v2` 已踩過兩次的同一種坑。
 ④ checkpoint 換 `v6_short_GD_no_macro_gatv2.pt`。
 ⑤ **宇宙過濾** —— dual 完全沒有（`run_daily_inference` 有，在 `_sanitize`）。
    base matrix 走 `^\\d{4}$` + `filter_tradable_universe`
    （`baseline_common.py:256-262`）。ETF 與興櫃留在橫斷面裡會改變
    `clean_and_scale` 的 winsorize / z-score **分母** → 59 維每一維的值都會偏。

兩個照抄回測、不可自由發揮的口徑
--------------------------------
 • **分數＝`model.eval()` 的單次前向，不是 MC-Dropout 平均。**
   +38.0% 那組是 eval 模式算的（`score_mamba_local.py:269, 313`），兩者不相等。
   MC-Dropout 只在 `--mc` 時額外算，**當顯示用的不確定性、不參與排序**。
 • **排序用 raw score，不用 `SQ = score/unc`。** 回測就是用 raw score 排的；
   SQ 是雙模型那條線的設計，混用會讓實盤與回測不同義。

驗證模式（`--verify-date`）
--------------------------
本檔的特徵管線（`merge_all_data` → trim 730 天）與產出 +38.0% 的
base matrix（自 2011 全量）**是兩條路徑**，不可假設一致 → 對窗內某一天實測：
與 `experimental/result/scores/v2_kg_nomacro.parquet` 同日逐股比對。

    判準（2026-08-05 跑之前定死，看到數字不改）：
        Spearman ρ ≥ 0.95  且  Top50 重疊 ≥ 40/50

用法（WSL）
-----------
    # 驗證（先跑這個，過了才有意義往下）
    wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && \\
      conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && \\
      python V6/run_v62_inference.py --verify-date 2026-06-02"

    # 最新交易日
    ... python V6/run_v62_inference.py
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ── 1) 先把 config 切 59 維（必須在 import 任何讀 FEATURE_COLS 的模組之前）──
#    `architecture.py` 在 import 當下就把 GROUP_DIMS/INPUT_DIM 綁進
#    `FactorGroupedEmbedding.__init__` 的預設參數（def 執行時求值）。
import marketmamba.config as cfg                                    # noqa: E402

_RS = ["RS_5d", "RS_20d", "RS_60d"]
if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in _RS):
    cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + _RS
cfg.INPUT_DIM = 59
cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"] + cfg.FEATURE_GROUPS["institutional_flow"]
                    + cfg.FEATURE_GROUPS["fundamentals"] + cfg.FEATURE_GROUPS["macro_environment"])
cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
assert len(cfg.FEATURE_COLS) == 59, f"expected 59 features, got {len(cfg.FEATURE_COLS)}"
# RS 必須在 group A 的**末端**（位置 12–14）——放錯 proj_A 會吃到錯欄位且不會報錯
assert cfg.FEATURE_COLS[12:15] == _RS, f"RS 位置錯誤：{cfg.FEATURE_COLS[9:16]}"

# ── 2) 切完才 import 重模組（它們會綁定 patched 後的 59 維設定）──
from marketmamba.config import AMP_ENABLED, PROCESSED_DIR, ROOT_DIR   # noqa: E402
from marketmamba.data.feature_engineer import build_features, clean_and_scale  # noqa: E402
from marketmamba.data.hygiene import filter_tradable_universe        # noqa: E402
from marketmamba.data.merger import merge_all_data                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_v62")

FEATURE_COLS = list(cfg.FEATURE_COLS)
MACRO_COLS   = list(cfg.FEATURE_GROUPS["macro_environment"])
# Group D 在 59 維張量裡的位置（用欄名反查，不寫死 47:59）
MACRO_IDX    = [FEATURE_COLS.index(c) for c in MACRO_COLS]

RESULTS_DIR   = ROOT_DIR / "results"
CKPT_DIR      = ROOT_DIR / "checkpoints"
REF_SCORE_DIR = ROOT_DIR / "experimental" / "result" / "scores"
LOOKBACK_DAYS = 730          # 個股大表 trim（同 run_dual_inference，避免 OOM）
DROPOUT       = 0.2          # 與 kg_ablation / groupd_ablation 相同（eval 下不生效，但架構要一致）
N_MC          = 30

# 驗證判準：跑之前定死（2026-08-05）
VERIFY_MIN_RHO     = 0.95
VERIFY_MIN_OVERLAP = 40      # Top50 重疊檔數


@dataclass(frozen=True)
class Arm:
    """一組上線設定。目前只有一個，但留成表格是為了之後並行跑多個 Mamba 變體時
    只需新增一列——邊際成本幾乎是零（特徵矩陣建構才是成本，且共用）。"""
    ckpt:       str
    kg_file:    str
    zero_macro: bool
    ref_score:  str | None     # 驗證用的既有分數檔（experimental/result/scores/）
    out_name:   str
    head:       str = "5d"     # 讀 forward 輸出的哪一欄（"5d"=第0欄、"10d"=第1欄）
    use_gat:    bool = True    # ⚠️ `no_gat` 的 state_dict **少了 graph_layer /
                               #    gate / norm_fuse`（short_model.py:74 的警告）。
                               #    用 use_gat=True 建模型再 load 會 strict 失敗。
    note:       str = ""


# ⚠️ 並行跑的模型集合**必須在開始累積實戰紀錄的那天定案**——晚加入的模型少了那段
#    紀錄，就再也無法與其他模型公平並列。加一個 arm 的邊際成本幾乎是零
#    （特徵矩陣是成本大宗、而且共用），所以寧可一開始就多放幾個。
ARMS: dict[str, Arm] = {
    # ── 規格模型（`5d/20`）：F6 2×2 最佳格 ───────────────────────────
    "v2_kg_nomacro": Arm(
        ckpt="v6_short_GD_no_macro_gatv2.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=True, ref_score="v2_kg_nomacro.parquet", out_name="df_v62",
        head="5d", note="上線規格：5 日頭 + 每 20 交易日再平衡"),
    # ── 同一顆 checkpoint 的第二欄（2026-08-08 新增）──────────────────
    #    **不是新模型**——`ShortModelV6` 一次前向就吐出 (N,2)，這裡只是改讀第 1 欄。
    #    回測 20 日 +39.2%／Sharpe 1.838，對 Alpha_20d 的 IC +0.1311（本專案最高）。
    #    價值在高頻端：分數變動比 5d 頭慢 → 換手低 → 1 日再平衡的成本
    #    31.1% vs 5d 頭的 42.6%，淨年化因此高 9.6pp（超出 ±6pp 雜訊底線）。
    "v2_kg_nomacro_h10": Arm(
        ckpt="v6_short_GD_no_macro_gatv2.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=True, ref_score="v2_kg_nomacro__head10d.parquet",
        out_name="df_v62_nomacro_h10", head="10d",
        note="與 v2_kg_nomacro 同 checkpoint，讀第 1 欄（10d 頭）"),
    # ── 長 horizon 變體：同設定，只有第二顆頭學的標籤不同（purge 20）──
    #    ⚠️ 兩者都讀**第 1 欄**（forward 輸出的第二顆頭），不論它學的是 10d 還是 20d
    "head10d": Arm(
        ckpt="v6_short_H_h10.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=True, ref_score="h20abl_h10__head10d.parquet", out_name="df_v62_h10",
        head="10d", note="第二顆頭學 Alpha_10d"),
    "head20d": Arm(
        ckpt="v6_short_H_h20.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=True, ref_score="h20abl_h20__head10d.parquet", out_name="df_v62_h20",
        head="10d", note="第二顆頭學 Alpha_20d（對 Alpha_20d 的 IC 0.1388，三顆頭最高）"),
    # ── F6 GAT / Group D 消融的四個 arm（2026-08-08 加入並行）────────
    #    ⚠️ 這四個 **zero_macro=False**（吃 Group D）——所以 `build_feature_df()`
    #    的 macro 全歷史貼回對它們是**必要的**，不是可有可無的優化。
    #    定位是對照組：八模型表裡它們都輸給 `v2_kg_nomacro`
    #    （decile 1.665~1.924 vs 4.999），留著是為了同一段真實 OOS 的並列紀錄。
    "v2_kg": Arm(
        ckpt="v6_short_KG_v2_kg.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=False, ref_score="v2_kg.parquet", out_name="df_v62_v2_kg",
        head="5d", note="v2 圖 + Group D 照常（decile 1.905、+26.0%）"),
    "v3_kg": Arm(
        ckpt="v6_short_KG_v3_kg.pt", kg_file="knowledge_graph_v3.npz",
        zero_macro=False, ref_score="v3_kg.parquet", out_name="df_v62_v3_kg",
        head="5d", note="v3 圖（+4,504 條相關性邊，實測無效應）"),
    "old_kg": Arm(
        ckpt="v6_short_KG_old_kg.pt", kg_file="knowledge_graph_cache.npz",
        zero_macro=False, ref_score="old_kg.parquet", out_name="df_v62_old_kg",
        head="5d", note="舊（壞）圖——2330 的鄰居是電器電纜（decile 1.229）"),
    "no_gat": Arm(
        ckpt="v6_short_KG_no_gat.pt", kg_file="knowledge_graph_v2.npz",
        zero_macro=False, ref_score="no_gat.parquet", out_name="df_v62_no_gat",
        head="5d", use_gat=False,
        note="⚠️ 無 GAT：state_dict 少 graph_layer/gate/norm_fuse（KG 僅佔位不使用）"),
}
DEFAULT_ARM = "v2_kg_nomacro"


# ============================================================
# 1. 特徵管線
# ============================================================
def _apply_universe(pr: pd.DataFrame) -> pd.DataFrame:
    """協定 v2 的宇宙規則。**逐行對照 `baseline_common.py:256-262`**——
    產出 +38.0% 的那份矩陣就是這樣濾的，兩邊不一致會讓橫斷面 z-score 的分母不同。"""
    n0 = len(pr)
    pr = pr[pr["stock_id"].astype(str).str.match(r"^\d{4}$")]
    keep = set(filter_tradable_universe(
        pd.DataFrame({"stock_id": sorted(pr["stock_id"].astype(str).unique())})
    )["stock_id"])
    pr = pr[pr["stock_id"].astype(str).isin(keep)]
    logger.info(f"宇宙過濾：{n0:,} → {len(pr):,} 列（{pr['stock_id'].nunique()} 支）")
    return pr


def build_feature_df(target_date: str | None = None,
                     history_start: str | None = None) -> pd.DataFrame:
    """raw parquet → trim → 宇宙過濾 → 59 維特徵 → clean_and_scale(macro_norm='ts')。

    `history_start`：明確指定窗首（`--score-window` 用，要涵蓋 582 個評分日
    再往前 252 個交易日）。未指定時沿用 `LOOKBACK_DAYS` 天。

    ⚠️ **macro ts 的窗長依賴（2026-08-08 修）**
       `clean_and_scale(macro_norm="ts")` 的 Group D 是 expanding 統計量，
       算在傳進去的日期範圍上 → trim 後的值與訓練時（全歷史）不同。
       實測兩個窗長互比：`Oil_Return` max|Δ|=**2.27**、`TNX` 1.07、`VIX` 0.87。

       原本這裡只寫「本模型 Group D 一律歸零，所以 trim 不構成落差」——
       那對 `v2_kg_nomacro` 系成立，但 **`v2_kg`/`v3_kg`/`old_kg`/`no_gat`
       是吃 Group D 的**，加進 ARMS 之後就不成立了。
       → 一律用 `macro_ts_full.splice()` 把那 12 欄換成全歷史版本。
       對歸零的 arm 完全沒差（反正會被歸零），對吃 Group D 的 arm 是必要的。
    """
    t0 = time.time()
    data = merge_all_data()

    prices = data["prices"].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    end = pd.Timestamp(target_date) if target_date else prices["Date"].max()
    cutoff = (pd.Timestamp(history_start) if history_start
              else end - pd.Timedelta(days=LOOKBACK_DAYS))
    prices = prices[(prices["Date"] >= cutoff) & (prices["Date"] <= end)].copy()

    # Close<=0 是 2026-04-30~05-22 來源切換期的損壞列。**刻意不剔除**——
    # base matrix（baseline_common）也沒有剔除，這裡多做一步就會讓兩邊宇宙不同。
    # 但要看得見（規則 7）：真的有影響時數字會在這裡出現。
    n_bad = int((pd.to_numeric(prices["Close"], errors="coerce") <= 0).sum())
    logger.info(f"個股 trim（窗首 {cutoff.date()}）：{len(prices):,} 筆 "
                f"（{prices['Date'].min().date()} → {prices['Date'].max().date()}）"
                f"｜Close<=0 {n_bad} 列（不剔除，與 base matrix 一致）")

    prices = _apply_universe(prices)

    def _trim(d):
        if d is None or "Date" not in getattr(d, "columns", []):
            return d
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date"])
        return d[(d["Date"] >= cutoff) & (d["Date"] <= end)].copy()

    df = build_features(
        df_price=prices, df_inst=_trim(data["inst"]), df_margin=_trim(data["margin"]),
        df_per=_trim(data["per"]), df_securities=_trim(data["securities"]),
        df_market_value=_trim(data["market_value"]), df_daytrade=_trim(data["daytrade"]),
        df_holdings=_trim(data["holdings"]), df_rev=_trim(data["revenue"]),
        df_fin=_trim(data["financials"]), df_balance_sheet=_trim(data["balance_sheet"]),
        df_cashflow=_trim(data["cashflow"]), df_macro=data["macro"],
        df_futures_inst=_trim(data["futures_inst"]), df_options_inst=_trim(data["options_inst"]),
        df_dividend=_trim(data["dividend"]),
        df_foreign_shareholding=_trim(data["foreign_shareholding"]),
        df_fear_greed=data["fear_greed"], df_business_indicator=data["business_indicator"],
        df_fed_rate=data["fed_rate"],
        fundamentals_v2=True,        # ← 落差 ①（對照 baseline_common.py:396）
    )
    # 窗內的**未標準化** macro（每日一列）——chunk 停在它建立那天，
    # 今天的日期只能由這裡補，否則貼回會是 NaN（比偏掉更糟）
    _m = [c for c in MACRO_COLS if c in df.columns]
    recent_macro = df.groupby("Date")[_m].first().sort_index() if _m else None

    df = clean_and_scale(df, macro_norm="ts")
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last")

    import macro_ts_full
    df = macro_ts_full.splice(df, _m, logger=logger, recent_raw=recent_macro)

    logger.info(f"特徵矩陣：{len(df):,} 列 × {df['stock_id'].nunique()} 支"
                f"｜{df['Date'].min().date()} → {df['Date'].max().date()}"
                f"｜{(time.time()-t0)/60:.1f} 分")
    return df


# ============================================================
# 2. 推論
# ============================================================
def infer(df: pd.DataFrame, date: str, arm: str = DEFAULT_ARM,
          mc: bool = False) -> pd.DataFrame:
    """單一交易日的 cross-section 推論。回傳 stock_id / Date / score / rank(+unc)。"""
    import torch
    import marketmamba.models.trainer as T
    from marketmamba.models.trainer import (
        TemporalCrossSectionDataset, build_kg_csr, get_batch_edges_csr,
    )
    # Group D 歸零沿用 groupd_ablation 的同一份實作（F5 方法紀律 ③：不另寫一份）
    from experimental.groupd_ablation import describe_macro, zeroed_macro
    from experimental.short_model import ShortModelV6

    spec = ARMS[arm]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = CKPT_DIR / spec.ckpt
    if not ck.exists():
        raise SystemExit(f"❌ 找不到 checkpoint：{ck}\n"
                         f"   （從 D:\\Downloads\\{spec.ckpt} 複製過來）")

    print(f"\n{'='*70}\n[v62] arm={arm}｜date={date}｜kg={spec.kg_file}"
          f"｜macro={'歸零' if spec.zero_macro else '照常'}\n"
          f"[v62] ckpt={ck.name}｜dev={dev}\n{'='*70}", flush=True)

    # ShortModelV6 是 2 頭 [5d, 10d]；Dataset 需要對應的標籤欄
    _orig = (T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS)
    T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = ["Alpha_5d", "Alpha_10d"], [5, 10], [5, 10]
    if "Alpha_10d" not in df.columns:      # 只當 Dataset 的佔位，分數只取 preds[:,0]
        df = df.copy()
        df["Alpha_10d"] = df["Alpha_5d"] if "Alpha_5d" in df.columns else np.nan
    try:
        model = ShortModelV6(use_gat=spec.use_gat, dropout=DROPOUT).to(dev)
        state = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(state.get("state_dict", state))    # strict=True：載錯當場失敗
        model.eval()
        # 參數量：有 GAT 1,659,005；無 GAT 少了 graph_layer/gate/norm_fuse
        print(f"[v62] checkpoint ep{state.get('epoch')} val_ic_5d={state.get('val_ic_5d')}"
              f"｜use_gat={spec.use_gat}｜參數 {model.n_parameters:,}"
              f"（use_gat=True 應為 1,659,005）", flush=True)

        # ── 落差 ②：換 v2 圖（`build_kg_csr` 讀模組層的 KG_CACHE_PATH）──
        _o = T.KG_CACHE_PATH
        T.KG_CACHE_PATH = Path(PROCESSED_DIR) / spec.kg_file
        if not T.KG_CACHE_PATH.exists():
            T.KG_CACHE_PATH = _o
            raise SystemExit(f"❌ 找不到 KG：{PROCESSED_DIR / spec.kg_file}")
        try:
            kg, s2i = build_kg_csr()
        finally:
            T.KG_CACHE_PATH = _o

        if spec.zero_macro:
            describe_macro(df, MACRO_COLS)   # 規則 7：歸零前先讓 Group D 的實測狀態看得見

        # ── 落差 ③：歸零的 context **必須包住整個推論**——Dataset 是 lazy loading，
        #    tensor 在 __getitem__ 才建，提早還原就會吃到未歸零的 macro（且不報錯）──
        ctx = zeroed_macro(df, MACRO_COLS) if spec.zero_macro else contextlib.nullcontext()
        with ctx:
            ds = TemporalCrossSectionDataset(df, [date], mode="test", n_sample=None)
            if len(ds) == 0:
                raise SystemExit(f"❌ {date} 沒有有效 cross-section"
                                 f"（該日不存在，或其前方不足 {cfg.SEQ_LEN} 個交易日）")
            X, _Y, stocks, _pm = ds[0]

            # 保險：確認歸零真的傳到模型輸入張量（不是只改了 df）
            probe = float(X[:, :, MACRO_IDX].abs().max())
            exp = "應為 0.000000" if spec.zero_macro else "應 > 0"
            bad = (probe > 0) if spec.zero_macro else (probe == 0)
            print(f"[v62] cross-section {X.shape[0]} 支｜Group D 12 個位置 absmax = "
                  f"{probe:.6f}（{exp}）{' ❌ 與預期不符！' if bad else ' ✓'}", flush=True)
            if bad:
                raise SystemExit("❌ Group D 輸入狀態與設定不符，停止"
                                 "（推論輸入與訓練輸入不一致，分數不可用）")

            ei, ea = get_batch_edges_csr(stocks, kg, s2i, dev)
            print(f"[v62] 本日子圖邊數 {ei.shape[1]:,}", flush=True)

            # ⚠️ 分數＝eval 單次前向（與產出 +38.0% 的 score_mamba_local 逐行相同）
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=AMP_ENABLED and dev.type == "cuda"):
                    p = model(X.to(dev), ei, ea)
            # ShortModelV6 是雙頭 [head_5d, head_10d]；換頭只是換讀哪一欄，不動權重
            score = p[:, {"5d": 0, "10d": 1}[spec.head]].float().cpu().numpy()
            print(f"[v62] 前向完成 {time.time()-t0:.1f}s", flush=True)

            unc = None
            if mc:
                unc = _mc_uncertainty(model, X, ei, ea, dev, date)
    finally:
        T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = _orig

    out = pd.DataFrame({"stock_id": [str(s) for s in stocks], "Date": date,
                        "score": score.astype(np.float32)})
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    if unc is not None:
        out = out.merge(pd.DataFrame({"stock_id": [str(s) for s in stocks],
                                      "uncertainty": unc.astype(np.float32)}),
                        on="stock_id", how="left")
    print(f"[v62] 分數：{len(out)} 支｜min {out['score'].min():+.4f}"
          f"｜median {out['score'].median():+.4f}｜max {out['score'].max():+.4f}", flush=True)
    return out


def _mc_uncertainty(model, X, ei, ea, dev, date: str) -> np.ndarray:
    """MC-Dropout 標準差（**顯示用，不參與排序**）。以日期為 seed，同日重跑可重現。"""
    import torch
    seed = int(pd.Timestamp(date).strftime("%Y%m%d"))
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.train()
    preds = torch.zeros(N_MC, X.shape[0])
    with torch.no_grad():
        for i in range(N_MC):
            with torch.amp.autocast('cuda', enabled=AMP_ENABLED and dev.type == "cuda"):
                preds[i] = model(X.to(dev), ei, ea)[:, 0].float().cpu()
    model.eval()
    sd = preds.std(dim=0).numpy()
    print(f"[v62] MC-Dropout N={N_MC} seed={seed}｜unc median {np.median(sd):.4f}", flush=True)
    return np.clip(sd, 0, None)


# ============================================================
# 2b. 整窗重評：現有 checkpoint × **現在的資料** × 582 天
# ============================================================
def score_window(arm: str = DEFAULT_ARM, out_name: str | None = None) -> dict:
    """
    把 F6 的 582 個 val 日**用現在的資料**全部重評一次，回答一件事：

        `+38.0% / Sharpe 1.713` 是在 2026-07-30 那份矩陣上量的，
        而 Group C 已於 08-04（FCF 兩層修正）與 08-04/05（MOPS 補齊）改過。
        **那個數字在修正後的資料上還成不成立？**

    ⚠️ 這不是「新管線 vs 舊管線」的驗證（那個已經做過：Group A/B 逐位元相同）。
       這是「同一個 checkpoint，餵它比較正確的 Group C，表現怎麼變」。
       checkpoint 是在**修正前**的 Group C 上訓練的，所以這一輪本質上在量
       **train/serve skew 的代價**，退步是有可能的、而且退步本身就是資訊。

    輸出寫到 `result/scores/{arm}__live.parquet`，**不覆蓋**對照用的
    `{arm}.parquet`——那份是舊資料基礎的紀錄，要留著才能對照。
    """
    import torch
    import marketmamba.models.trainer as T
    from marketmamba.models.trainer import (
        TemporalCrossSectionDataset, build_kg_csr, compute_ic, get_batch_edges_csr,
        make_dataloader,
    )
    from experimental.groupd_ablation import describe_macro, zeroed_macro
    from experimental.short_model import ShortModelV6
    # val 日期沿用 score_mamba_local 的同一個權威來源（Colab 記錄的 582 天），
    # 不另寫一份切分——級間差異混進實作差異正是 F5 方法紀律第 ③ 條要擋的事
    from experimental.score_mamba_local import _val_dates

    spec = ARMS[arm]
    val_dates = _val_dates()
    start = (pd.Timestamp(val_dates[0]) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    print(f"[window] val 窗 {len(val_dates)} 天（{val_dates[0]} → {val_dates[-1]}）"
          f"｜特徵窗首 {start}（需含 252 個交易日暖機）", flush=True)

    df = build_feature_df(target_date=val_dates[-1], history_start=start)
    df["Date"] = pd.to_datetime(df["Date"])
    n_hist = df[df["Date"] < pd.Timestamp(val_dates[0])]["Date"].nunique()
    print(f"[window] val 起點前有 {n_hist} 個交易日（需 ≥252）", flush=True)
    assert n_hist >= 252, f"歷史不足（{n_hist} < 252），請把 start 往前調"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = CKPT_DIR / spec.ckpt
    _orig = (T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS)
    T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = ["Alpha_5d", "Alpha_10d"], [5, 10], [5, 10]
    try:
        # ⚠️ `use_gat` 必須跟著 spec 走。2026-08-08 我只修了 `infer()`、**漏了這裡**，
        #    結果 `no_gat` 的窗評分在 load_state_dict 當場炸（少 graph_layer/gate/norm_fuse）。
        #    同一個設定分散在兩處建模型，就會有「修了一處以為修完」的漏網。
        model = ShortModelV6(use_gat=spec.use_gat, dropout=DROPOUT).to(dev)
        state = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(state.get("state_dict", state))
        model.eval()
        print(f"[window] ckpt={ck.name} ep{state.get('epoch')}｜use_gat={spec.use_gat}"
              f"｜參數 {model.n_parameters:,}", flush=True)

        _o = T.KG_CACHE_PATH
        T.KG_CACHE_PATH = Path(PROCESSED_DIR) / spec.kg_file
        try:
            kg, s2i = build_kg_csr()
        finally:
            T.KG_CACHE_PATH = _o

        if spec.zero_macro:
            describe_macro(df, MACRO_COLS)
        ctx = zeroed_macro(df, MACRO_COLS) if spec.zero_macro else contextlib.nullcontext()
        with ctx:
            ds = TemporalCrossSectionDataset(df, val_dates, mode="val", n_sample=None)
            loader = make_dataloader(ds, shuffle=False)
            rows, ic_by_day, probe = [], {}, None
            t0 = time.time()
            with torch.no_grad():
                for i, (X, Y, stks, _m) in enumerate(loader):
                    if X.shape[0] <= 1:
                        continue
                    if probe is None:
                        probe = float(X[:, :, MACRO_IDX].abs().max())
                        bad = (probe > 0) if spec.zero_macro else (probe == 0)
                        print(f"[window] 第一個 batch {X.shape[0]} 支｜Group D absmax "
                              f"{probe:.6f}{' ❌' if bad else ' ✓'}", flush=True)
                        if bad:
                            raise SystemExit("❌ Group D 輸入狀態不符，停止")
                    d = str(ds.valid_dates[i])[:10]
                    ei, ea = get_batch_edges_csr(stks, kg, s2i, dev)
                    with torch.amp.autocast('cuda', enabled=AMP_ENABLED and dev.type == "cuda"):
                        p = model(X.to(dev), ei, ea)
                    s = p[:, {"5d": 0, "10d": 1}[spec.head]].float().cpu().numpy()
                    ic_by_day[d] = float(compute_ic(s, Y[:, 0].float().cpu().numpy()))
                    rows.append(pd.DataFrame({"Date": d, "stock_id": [str(x) for x in stks],
                                              "score": s.astype(np.float32)}))
                    if (i + 1) % 100 == 0:
                        el = time.time() - t0
                        print(f"  {i+1}/{len(ds.valid_dates)}｜{el:.0f}s｜"
                              f"ETA {el/(i+1)*(len(ds.valid_dates)-i-1):.0f}s", flush=True)
    finally:
        T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = _orig

    out = pd.concat(rows, ignore_index=True)
    name = out_name or f"{arm}__live"
    REF_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    dst = REF_SCORE_DIR / f"{name}.parquet"
    out.to_parquet(dst, index=False)

    ics = np.array([v for v in ic_by_day.values() if np.isfinite(v)])
    res = {"arm": arm, "n_days": len(ic_by_day), "n_rows": len(out),
           "mean_ic": round(float(ics.mean()), 4),
           "icir": round(float(ics.mean() / ics.std()), 3),
           "pct_pos": round(float((ics > 0).mean()), 3), "output": dst.name}
    print(f"\n[window] {len(out):,} 列 / {len(ic_by_day)} 天 → {dst.name}"
          f"｜{(time.time()-t0)/60:.1f} 分", flush=True)
    print(f"[window] **現在資料** mean IC {res['mean_ic']:+.4f}｜ICIR {res['icir']}"
          f"｜IC>0 {res['pct_pos']:.1%}", flush=True)
    # ⚠️ 2026-08-09 移除一個會誤導的輸出。
    #    這裡原本硬編 `對照（舊資料基礎）：mean IC +0.1145｜ICIR 1.340`——
    #    那是 **`v2_kg_nomacro` 一個 arm 的舊值**，卻對每個 arm 都印同一行。
    #    重評分八個 arm 時 log 會長成「head10d 從 0.1145 掉到 0.1094」，
    #    但 head10d 的舊值根本不是 0.1145。**看起來像對照，其實是拿別人的基準。**
    #
    #    不在這裡重算舊 IC 的理由：本函式的 IC 是用 Dataset 的 Y（當日 z-score 後的
    #    Alpha）逐日算的，外面拿不到同一份標籤；換一個標籤路徑算出來的「舊 IC」
    #    與這裡的「新 IC」又不可比——**那只是把誤導換一種形式**。
    #    要做新舊 IC 對照，請用同一支腳本、同一份標籤對兩份分數各算一次。
    #    本函式提供的可比對照是下面那段**逐日 Spearman**（它用該 arm 自己的 ref_score）。
    print(f"[window] （不列 IC 對照：舊值須用同一份標籤另外算，"
          f"見下方逐日 Spearman——那個才是本 arm 自己的對照）", flush=True)

    # 逐日比對新舊分數：一天一個 Spearman，看整窗漂移多大、有沒有集中在某段
    ref_p = REF_SCORE_DIR / (spec.ref_score or "")
    if ref_p.exists():
        ref = pd.read_parquet(ref_p)
        ref["Date"] = ref["Date"].astype(str).str.slice(0, 10)
        ref["stock_id"] = ref["stock_id"].astype(str)
        mg = out.merge(ref, on=["Date", "stock_id"], how="inner", suffixes=("_new", "_ref"))
        per_day = mg.groupby("Date").apply(
            lambda g: g["score_new"].corr(g["score_ref"], method="spearman"),
            include_groups=False)
        pd_arr = per_day.dropna().to_numpy()
        res["score_rho_median"] = round(float(np.median(pd_arr)), 4)
        res["score_rho_min"] = round(float(pd_arr.min()), 4)
        h1, h2 = pd_arr[:len(pd_arr)//2], pd_arr[len(pd_arr)//2:]
        print(f"[window] 新舊分數逐日 Spearman：median {res['score_rho_median']:.4f}"
              f"｜最低 {res['score_rho_min']:.4f}"
              f"｜前半 {h1.mean():.4f} / 後半 {h2.mean():.4f}", flush=True)
        print(f"[window] （後半若明顯較低＝MOPS 補的 2026 財報影響集中在窗尾，符合預期）",
              flush=True)
    print(f"\n下一步：MM_PROTOCOL=v2 python V6/experimental/portfolio_lab.py "
          f"--sweep --models {name}   ← **在 Windows 端跑**（WSL 的 pandas 3.0 會炸）",
          flush=True)
    return res


# ============================================================
# 3. 驗證：新管線 vs 產出 +38.0% 的那份分數
# ============================================================
def verify(out: pd.DataFrame, date: str, arm: str = DEFAULT_ARM) -> bool:
    """對同一天比逐股 Spearman ρ 與 Top50 重疊。判準在檔頭，跑之前定死。"""
    spec = ARMS[arm]
    if not spec.ref_score:
        print("[驗證] ⚠ 此 arm 無對照分數檔", flush=True)
        return False
    p = REF_SCORE_DIR / spec.ref_score
    if not p.exists():
        print(f"[驗證] ⚠ 找不到對照 {p}", flush=True)
        return False

    ref = pd.read_parquet(p)
    ref["Date"] = ref["Date"].astype(str).str.slice(0, 10)
    ref = ref[ref["Date"] == date]
    if ref.empty:
        print(f"[驗證] ⚠ 對照檔沒有 {date}（窗：{pd.read_parquet(p)['Date'].min()} → "
              f"{pd.read_parquet(p)['Date'].max()}）", flush=True)
        return False
    ref["stock_id"] = ref["stock_id"].astype(str)

    mg = out[["stock_id", "score"]].merge(
        ref[["stock_id", "score"]], on="stock_id", how="inner", suffixes=("_new", "_ref"))
    only_new = len(out) - len(mg)
    only_ref = len(ref) - len(mg)
    rho = float(mg["score_new"].corr(mg["score_ref"], method="spearman"))

    top_new = set(out.nsmallest(50, "rank")["stock_id"])
    top_ref = set(ref.nlargest(50, "score")["stock_id"])
    ov = len(top_new & top_ref)

    ok = (rho >= VERIFY_MIN_RHO) and (ov >= VERIFY_MIN_OVERLAP)
    print(f"\n{'='*70}\n[驗證] {date}｜新管線 {len(out)} 支 vs 對照 {len(ref)} 支"
          f"｜交集 {len(mg)}（只在新 {only_new} / 只在舊 {only_ref}）\n"
          f"[驗證] Spearman ρ = {rho:.4f}（判準 ≥{VERIFY_MIN_RHO}）"
          f"｜Top50 重疊 = {ov}/50（判準 ≥{VERIFY_MIN_OVERLAP}）\n"
          f"[驗證] {'✅ 通過——推論端接線正確' if ok else '❌ 未過'}\n{'='*70}", flush=True)
    if not ok:
        print("[驗證] 未過時先查（按可能性排序）：\n"
              "  ① 宇宙過濾兩邊是否真的一致（z-score 分母不同 → 59 維全體偏移）\n"
              "  ② trim 730 天是否讓某些股票的**財報 as-of** 找不到來源（_trim 會把\n"
              "     cutoff 之前的財報整批砍掉，長期未更新的個股會退化成 NaN/ffill）\n"
              "  ③ fundamentals_v2 是否真的傳進去了\n"
              "  ④ rolling 特徵（MA_60/ATR_14）在窗首附近不足\n"
              "**不要放寬判準**——判準放寬等於拿結果改測試。", flush=True)
    return ok


# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=DEFAULT_ARM, choices=list(ARMS))
    ap.add_argument("--date", default=None, help="指定交易日（預設：資料最新日）")
    ap.add_argument("--verify-date", default=None,
                    help="驗證模式：對該日與 experimental/result/scores 的既有分數比對")
    ap.add_argument("--mc", action="store_true",
                    help="額外算 MC-Dropout 不確定性（顯示用，不參與排序）")
    ap.add_argument("--score-window", action="store_true",
                    help="用**現在的資料**重評 F6 的 582 個 val 日 → {arm}__live.parquet")
    ap.add_argument("--no-save", action="store_true", help="不寫輸出檔（驗證時預設就不寫）")
    a = ap.parse_args()

    if a.score_window:
        score_window(arm=a.arm)
        return

    target = a.verify_date or a.date
    df = build_feature_df(target)
    df["Date"] = pd.to_datetime(df["Date"])
    date = target or df["Date"].max().strftime("%Y-%m-%d")

    out = infer(df, date, arm=a.arm, mc=a.mc)

    if a.verify_date:
        ok = verify(out, date, arm=a.arm)
        sys.exit(0 if ok else 1)

    if not a.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        name = ARMS[a.arm].out_name
        out.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
        arch = RESULTS_DIR / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        out.to_csv(arch / f"{name}_{date}.csv", index=False)
        print(f"✅ {name}.csv（{len(out)} 支，依 score 排序）→ {RESULTS_DIR / f'{name}.csv'}\n"
              f"📦 已歸檔 {date} → {arch}", flush=True)
        print("\n⚠️ 這一步只到「分數」。組合層（N=50 / k=1.5 / 每 20 個交易日再平衡）"
              "還沒接——分數本身不是持股名單。", flush=True)


if __name__ == "__main__":
    main()
