"""
MarketMamba — Feature Spec v2（67 維）
=======================================
版本狀態：**v2.0，規格已凍結**（2026-07-30）
凍結範圍：可得性旗標的定義與語意、產業正規化規則、中性化排除清單、67 維的分組配置。
對應文件：`docs/feature-protocol-v2.md`

尚未定案、**允許在 F5 全量結果出來後修改**的只有兩項：
  1. `AVAILABILITY_FLAGS` 的成員數（死旗標若在全量下仍為常數 → 砍到 5 個、INPUT_DIM 64）
  2. `NEUTRALIZE` 的預設值（none / industry / industry_mktcap）
除此之外的改動都會使已跑的實驗失去可比性，請先確認再動。

V6.3 特徵規格的**唯一事實來源**：可得性旗標、產業分類正規化、中性化排除清單、
以及 59 → 67 維的 runtime config patch。

【為什麼放在 production 套件而不是 experimental】
`feature_engineer.py`（production）必須用到旗標定義才能產生旗標欄，
而 production **不可以** import experimental（依賴方向會反向、experimental 隨時會被改）。
本模組是純資料 + 純函式，不改變任何既有行為，對 V6.1 零影響。

【內容從哪來】全部來自 2026-07-29 對真實資料的量測
（`V6/scripts/report_feature_availability.py`），不是從文件推論的。

---------------------------------------------------------------------------
一、為什麼需要可得性旗標（決策1）

各資料源對 `prices_raw` 宇宙的 (日期, 股票) 命中率實測，缺失有**兩種性質**：

  A. 整欄全缺（daytrade 2014 前、holdings/foreign_shareholding 2018 前）
     → 當日全市場同值 → 橫斷面 z-score 自動歸零 → 本身無害，
       但模型無從知道「這一維今天是關著的」。

  B. 部分缺（institutional 2005–2011 僅 10–18%、securities 全期 0.3–18%）
     → 被 `fillna(0)` 補的列，與「淨買超恰為 0」「無借券餘額」這種**合法的 0**
       完全無法區分 → 模型改為學習「這支有沒有被資料涵蓋」這個選擇性代理變數。

B 才是真正要處理的。旗標把「捏造的 0」與「真實的 0」分開，讓模型自己決定怎麼用。

---------------------------------------------------------------------------
二、旗標的語意（重要，實作時不要弄錯）

  1 = 該來源對這支股票在這一天有**真實觀測**（可能是 ffill 帶下來的，
      但源頭確實存在過一筆）
  0 = 純粹捏造的填補值

刻意**不用**「今天當場有沒有新觀測」當語意：holdings 是週頻、margin 有公布延遲，
那樣定義會讓旗標大部分日子都是 0，反映的是公布頻率而不是資料有無。

沿用 `_merge_per_pbr` 既有的 `PER__obs` 慣例（fundamentals_v2 已在用同一招）。

⚠️ 已知限制：旗標描述「值是不是真的」，**不描述新鮮度**。
   一支股票 2013 年有融資資料、2020 年被取消信用交易資格，
   旗標仍會是 1（因為 ffill 帶著 2020 年的舊值）。
   新鮮度是另一個維度，若之後證實需要，應該另加欄位而不是改這個的語意。

---------------------------------------------------------------------------
實測驗證（2026-07-29 首次上線就抓到一個隱藏的資料缺口）

`Avail_ForeignShare` 在測試樣本中精確卡在 50%，追查後發現不是程式錯，
而是 `foreign_shareholding_raw` 真的缺資料：**199 支 2018 年前就上市的公司
沒有歷史外資持股資料**（只有 2026-05-06 起直連回補的 58 筆），
而且缺失高度集中在 **9xxx 代號區間（缺失率 82.8%，其他首碼只有 5–15%）**
——9904 寶成、9910 豐泰、9914 美利達、9917 中保科 都在其中。

沒有旗標的話，這些列的 `Foreign_Holding_Pct` 會是捏造的 0，
而「外資持股 0%」本身是個有意義的訊號（乏人問津），
模型會在 2.5% 的訓練列上學到完全錯誤的東西，而且永遠不會有人發現。

---------------------------------------------------------------------------
三、旗標**不可以**被橫斷面 z-score（`clean_and_scale` 必須跳過）

0/1 旗標若逐日 z-score：
  - 全市場都有資料的那天 → std=0 → 整欄變 0
  - 全市場都沒資料的那天 → std=0 → 也變 0
兩種相反的狀態變成同一個值，旗標就失去存在意義（而「整欄全缺」正是它要表達的事）。
"""
from __future__ import annotations

import sys

import pandas as pd

# ============================================================
# 一、可得性旗標
# ============================================================

AVAIL_PREFIX = "Avail_"

# flag → (放進哪個 FEATURE_GROUPS 分組, 來源檔, 它保護哪些特徵欄, 說明)
AVAILABILITY_FLAGS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "Avail_Institutional": (
        "institutional_flow", "institutional_raw",
        ("Foreign_Buy", "Foreign_Sell", "Foreign_Net",
         "Investment_Trust_Net", "Dealer_Net"),
        "2005–2011 覆蓋僅 10–18%，2012 跳 56%、2013 起 74%+。0=當日無法人買賣（合法值）",
    ),
    "Avail_Margin": (
        "institutional_flow", "margin_raw",
        ("Margin_Purchase", "Margin_Repay", "Short_Sale",
         "Short_Cover", "Margin_Balance", "Short_Balance"),
        "長期穩定 ~92%，缺的 8% 多為無信用交易資格。0=無融資餘額（合法值）",
    ),
    "Avail_Daytrade": (
        "institutional_flow", "daytrade_raw",
        ("Day_Trade_Volume",),
        "2014 前 0%（當沖制度 2014 才開放），2016 起 ~80%。0=當日無當沖（合法值）",
    ),
    "Avail_Holdings": (
        "institutional_flow", "holdings_raw",
        ("Holdings_Large_Pct", "Holdings_Large_Change"),
        "2018-01 起才有。週頻，靠 ffill 帶到每日",
    ),
    "Avail_ForeignShare": (
        "institutional_flow", "foreign_shareholding_raw",
        ("Foreign_Holding_Pct",),
        "2018-01 起才有，之後穩定 ~92%",
    ),
    "Avail_Securities": (
        "institutional_flow", "securities_raw",
        ("Securities_Balance",),
        "全期覆蓋極低（2005 年 0.3% → 2026 年 48.8%）——借券本來就只有部分股票有。"
        "0=無借券餘額（合法值），旗標同時帶有「這支可被借券/放空」的資訊",
    ),
    "Avail_Valuation": (
        "fundamentals", "per_raw（+ fundamentals_v2 自算）",
        ("PER", "PBR"),
        "per 2007 後穩定 ~96%，2005 僅 21%，2026 掉到 72.9%（新來源只涵蓋上市）。"
        "**刻意不含 Market_Cap_Log**——它覆蓋率 94–99%，OR 進來會讓旗標恆為 1、失去資訊",
    ),
    "Avail_Financials": (
        "fundamentals", "financials_raw + balance_sheet_raw + cashflow_raw",
        ("EPS", "EPS_Surprise", "Gross_Margin", "ROE",
         "Book_Value", "Free_Cash_Flow"),
        "balance_sheet 2011-12 起才有 → Book_Value/ROE 在那之前無來源",
    ),
}

AVAIL_COLS: list[str] = list(AVAILABILITY_FLAGS.keys())

# 各分組要新增的旗標（順序固定 = 特徵張量的欄位順序，不可更動）
AVAIL_BY_GROUP: dict[str, list[str]] = {}
for _flag, (_grp, *_rest) in AVAILABILITY_FLAGS.items():
    AVAIL_BY_GROUP.setdefault(_grp, []).append(_flag)


# ============================================================
# 二、產業分類正規化
# ============================================================
#
# 實測（2026-07-29）：TPEX（上櫃）與 TWSE（上市）對**同一個產業**用不同名稱。
# 若直接拿 `industry_category` 建產業邊或中性化 dummies，
# 同一個產業會沿「上市/上櫃」被切成兩個互不相連的群——而那條界線在經濟上毫無意義。
#
#   上櫃「運動休閒類」(10 支) vs 上市「運動休閒」(18 支)
#   上櫃「其他電子類」(53 支) vs 上市「其他電子業」(27 支)
#   上櫃「金融業」    (10 支) vs 上市「金融保險」  (46 支)
#
# 另有交易所歷次分類改版留下的舊名（含一個錯字：創新「版」→創新「板」）。
# 統一以**上市（TWSE）的名稱**為正規名，因為上市檔數較多、名稱較穩定。

SECTOR_ALIASES: dict[str, str] = {
    # ── 上櫃「XX類」→ 上市「XX」 ──
    "居家生活類": "居家生活",
    "數位雲端類": "數位雲端",
    "綠能環保類": "綠能環保",
    "運動休閒類": "運動休閒",
    # ── 跨市場同義但名稱不同 ──
    "其他電子類": "其他電子業",
    "金融業":     "金融保險",
    # ── 交易所歷次分類改版留下的舊名 ──
    "觀光事業":   "觀光餐旅",
    "創新版股票": "創新板股票",   # 錯字修正：版 → 板
    "電子商務業": "數位雲端",
    "農業科技":   "農業科技業",
}

# 刻意**不**合併的（看起來像同義，實際兩邊都仍在使用、成員不同）：
#   資訊服務業（上櫃 33 / 上市 5）≠ 數位雲端      — 兩者現行並存
#   化學生技醫療（上市 36）≠ 化學工業 / 生技醫療業 — 上市的合併類別，無法拆
#   電子工業（上市 250）                          — 上市的大類，無上櫃對應

_SECTOR_UNKNOWN = "Unknown"

# ── 同一支股票、同一個快照日期會有多個 industry_category 的兩類原因 ──
#
# 實測（2026-07-29）：605 組 (股票, 日期) 同時帶兩種以上標籤。
# `load_stock_info(latest_only=True)` 的 drop_duplicates(keep="last") 在日期相同時
# 是**依 parquet 列序任意挑一個**——2330 因此被標成「電子工業」而不是「半導體業」，
# 於是在知識圖譜裡連到一個 250 支的大雜燴，而不是它真正的半導體同業。
#
# 原因一：交易所同時提供「早年大類」與「現行細類」
#   電子工業 ↔ 電子零組件業(106) / 半導體業(102) / 光電業(79) / 電腦及週邊設備業(71)
#             / 其他電子業(50) / 通信網路業(48) / 電子通路業(25) / 資訊服務業(13)
#   化學生技醫療 ↔ 生技醫療業(56) / 化學工業(30)
LEGACY_AGGREGATE_SECTORS: frozenset[str] = frozenset({"電子工業", "化學生技醫療"})

# 原因二：有些標籤根本不是產業，是板別／商品類型
NON_INDUSTRY_LABELS: frozenset[str] = frozenset(
    {"創新板股票", "創新版股票", "存託憑證", _SECTOR_UNKNOWN, "None", "nan", ""})


def resolve_sector(info: pd.DataFrame) -> pd.DataFrame:
    """
    從 `stock_info`（完整累積表）解析出每支股票**唯一且盡量細**的產業別。

    優先序：現行細類 > 早年大類 > 板別標籤 > Unknown；同層再取最新快照。
    回傳 [stock_id, sector]。

    這一步不能省略：不做的話，250 支股票（含台積電）會落在「電子工業」這個
    早年大類裡，產業邊等於連到一個大雜燴，中性化的 dummy 也會把半導體、面板、
    網通當成同一個產業去回歸。
    """
    if info.empty or "industry_category" not in info.columns:
        return pd.DataFrame({"stock_id": [], "sector": []})

    d = info[["stock_id", "industry_category"]].copy()
    d["stock_id"] = d["stock_id"].astype(str)
    d["sector"] = canonical_sector(d["industry_category"])
    d["_date"] = (pd.to_datetime(info["date"], errors="coerce")
                  if "date" in info.columns else pd.NaT)

    # 分數越小越優先
    d["_rank"] = 0
    d.loc[d["sector"].isin(LEGACY_AGGREGATE_SECTORS), "_rank"] = 1
    d.loc[d["sector"].isin(NON_INDUSTRY_LABELS), "_rank"] = 2

    d = d.sort_values(["_rank", "_date"], ascending=[True, False],
                      na_position="last", kind="stable")
    out = d.drop_duplicates(subset=["stock_id"], keep="first")
    return out[["stock_id", "sector"]].reset_index(drop=True)


def canonical_sector(s: pd.Series) -> pd.Series:
    """
    把 `industry_category` 正規化成跨市場一致的產業名。

    先套 `SECTOR_ALIASES`，再對「結尾是『類』且去掉後仍是已知產業」的殘餘情況兜底
    （交易所之後若新增類別，不必回來改這張表也不會被切開）。
    """
    out = s.astype(str).replace(SECTOR_ALIASES)
    known = set(out.unique())
    extra = {c: c[:-1] for c in known if c.endswith("類") and c[:-1] in known}
    if extra:
        out = out.replace(extra)
    return out.replace({"": _SECTOR_UNKNOWN, "None": _SECTOR_UNKNOWN,
                        "nan": _SECTOR_UNKNOWN})


# ============================================================
# 三、中性化排除清單
# ============================================================
#
# 中性化的意義是「把因子裡不想賭的系統性成分（產業、市值）移除」。
# 以下欄位做中性化沒有意義、甚至有害：

NEUTRALIZE_EXCLUDE: frozenset[str] = frozenset(
    AVAIL_COLS                                # 旗標：0/1 指示變數，取殘差會失去語意
    + ["Open", "High", "Low", "Close", "Volume"]   # 原始價量水位：對絕對價格做市值中性化沒有意義
    + ["Market_Cap_Log"]                      # 就是中性化的自變數本身，殘差恆為 0
    + ["TWII_Return", "SPX_Return", "VIX", "TNX", "Gold_Return", "Oil_Return",
       "USD_TWD", "Futures_OI_Foreign", "Options_PC_Ratio", "Fear_Greed",
       "Business_Signal", "FED_Rate"]         # Group D macro：同日全市場同值，橫斷面迴歸無定義
)


# ============================================================
# 四、59 → 67 維的 runtime config patch
# ============================================================

_RS_COLS = ["RS_5d", "RS_20d", "RS_60d"]


def patch_config_67d(strict: bool = True) -> int:
    """
    把 `marketmamba.config` 就地改成 V6.3 的 67 維規格並回傳 INPUT_DIM。

    做法比照 `baseline_common.py` / `run_dual_inference.py` 既有的 runtime 自切模式，
    **不改 `config.py` 檔案本身**——本機那份是刻意保持 56 維、不進 git 的 dirty 檔。

    組成：56（V6.1）+ 3（RS，V6.2）+ 8（可得性旗標，V6.3）= 67
      Group A price_momentum      12 → 15
      Group B institutional_flow  20 → 26
      Group C fundamentals        12 → 14
      Group D macro_environment   12 → 12

    ⚠️ **必須在 import 任何 `marketmamba.models.*` 之前呼叫。**
       `architecture.py` 是 `from config import GROUP_DIMS, INPUT_DIM`——import 當下就綁值，
       而且 `FactorGroupedEmbedding.__init__` 的預設參數 `group_dims=GROUP_DIMS`
       是在 **def 執行時**求值的，事後改 module 全域也救不回來。
       若模型模組已載入，embedding 會**靜靜地**沿用舊維度、不報錯，
       直到 IC 莫名其妙很差為止。故 strict=True 直接擋下來。

    ⚠️ **`feature_engineer` 是另一種情況，必須主動同步（2026-07-30 實測踩到）。**
       `marketmamba/data/__init__.py` 會 `from ... feature_engineer import ...`，
       所以**光是 import 本模組（feature_spec）就會連帶載入 feature_engineer**，
       它的 module 級 `FEATURE_COLS` 因此在 patch 之前就綁成舊清單。
       後果：旗標會被算出來，然後在 `build_features` 尾端重排欄位時靜靜丟掉，
       連 `_cfg_has_flags` 的防呆也因為讀到舊清單而不會觸發。
       所幸 `feature_engineer` 是在**呼叫時**讀 module 全域（不像 architecture 綁在
       預設參數上），所以事後覆寫是有效的——下方顯式同步。
    """
    if strict:
        loaded = [m for m in sys.modules if m.startswith("marketmamba.models")]
        if loaded:
            raise RuntimeError(
                f"patch_config_67d() 必須在 import marketmamba.models.* 之前呼叫，"
                f"但這些模組已載入：{loaded}。"
                f"architecture.py 在 import 時就綁定了 GROUP_DIMS/INPUT_DIM，"
                f"現在才 patch 會讓 embedding 靜默維持舊維度。"
            )

    import marketmamba.config as cfg

    pm = list(cfg.FEATURE_GROUPS["price_momentum"])
    if not all(r in pm for r in _RS_COLS):
        pm = pm + _RS_COLS
    cfg.FEATURE_GROUPS["price_momentum"] = pm

    for grp, flags in AVAIL_BY_GROUP.items():
        cur = list(cfg.FEATURE_GROUPS[grp])
        cfg.FEATURE_GROUPS[grp] = cur + [f for f in flags if f not in cur]

    cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"]
                        + cfg.FEATURE_GROUPS["institutional_flow"]
                        + cfg.FEATURE_GROUPS["fundamentals"]
                        + cfg.FEATURE_GROUPS["macro_environment"])
    cfg.INPUT_DIM = len(cfg.FEATURE_COLS)
    cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}

    assert cfg.INPUT_DIM == 67, f"expected 67 features, got {cfg.INPUT_DIM}"

    # 顯式同步已載入模組的 module 級綁定（見上方 docstring 第二個 ⚠️）。
    # 這些模組都是「呼叫時才讀 module 全域」，事後覆寫有效。
    for _name, _attrs in (
        ("marketmamba.data.feature_engineer", ("FEATURE_COLS", "FEATURE_GROUPS")),
    ):
        _m = sys.modules.get(_name)
        if _m is not None:
            for _a in _attrs:
                if hasattr(_m, _a):
                    setattr(_m, _a, getattr(cfg, _a))

    # 驗證同步確實生效——這個 bug 的可怕之處在於它完全沒有徵兆，
    # 所以不能只是「寫了同步的程式碼」，要當場確認結果。
    _fe = sys.modules.get("marketmamba.data.feature_engineer")
    if _fe is not None:
        _n = len(getattr(_fe, "FEATURE_COLS", []))
        assert _n == 67, (
            f"feature_engineer.FEATURE_COLS 同步失敗（{_n} 維）。"
            f"沒有這一步，旗標會被算出來卻在重排欄位時靜默丟掉。")

    return cfg.INPUT_DIM


# ============================================================
# 五、特徵可得時間對照表（檢查表 G1；協定 v2.0 的必要附件）
# ============================================================
#
# 欄位：特徵群 → (來源, 資料起始, 公告延遲, 缺失語意)
# 「公告延遲」= `feature_engineer.py` 實際套用的 available_from 位移。

AVAILABILITY_TABLE: list[dict[str, str]] = [
    {"特徵群": "OHLCV / 技術指標（12 維）", "來源": "prices_raw（交易所直連，官方還原價）",
     "起始": "2005-01-03", "延遲": "當日收盤即可得", "缺失語意": "無缺（宇宙定義來源）"},
    {"特徵群": "RS_5d / RS_20d / RS_60d（3 維）",
     "來源": "prices_raw − macro_raw 的 TWII_Return_*",
     "起始": "2005-01-03", "延遲": "當日",
     "缺失語意": "⚠️ **唯一跨 Group 相依的特徵**：`_add_rs_features` 必須在 "
                 "`_merge_macro` 之後執行（需要 TWII_Return_5d/20d/60d）。"
                 "調整 build_features 的呼叫順序時要特別注意；"
                 "TWII 缺值時退回 0 → RS 退化為個股報酬本身"},
    {"特徵群": "法人買賣（5 維）", "來源": "institutional_raw（TWSE T86 / TPEX）",
     "起始": "2005-01-03", "延遲": "當日盤後", "缺失語意": "0 可能是無買賣或無資料 → Avail_Institutional"},
    {"特徵群": "融資融券（6 維）", "來源": "margin_raw（TWSE MI_MARGN / TPEX）",
     "起始": "2005-01-03", "延遲": "當日盤後（ffill）", "缺失語意": "0 可能是無餘額或無資格 → Avail_Margin"},
    {"特徵群": "當沖（1 維）", "來源": "daytrade_raw（TWSE TWTB4U / TPEX）",
     "起始": "2014-01-06", "延遲": "當日盤後", "缺失語意": "2014 前制度不存在 → Avail_Daytrade"},
    {"特徵群": "大戶持股（2 維）", "來源": "holdings_raw（TDCC）",
     "起始": "2018-01-05", "延遲": "週頻，ffill 至每日", "缺失語意": "2018 前無來源 → Avail_Holdings"},
    {"特徵群": "外資持股比（1 維）", "來源": "foreign_shareholding_raw（TWSE MI_QFIIS / TPEX）",
     "起始": "2018-01-02", "延遲": "當日盤後", "缺失語意": "2018 前無來源 → Avail_ForeignShare"},
    {"特徵群": "借券餘額（1 維）", "來源": "securities_raw（TWSE SBL/TWT96U）",
     "起始": "2005-01-03（極稀疏）", "延遲": "當日盤後", "缺失語意": "0 多為真的無借券 → Avail_Securities"},
    {"特徵群": "本益比 / 淨值比 / 市值", "來源": "per_raw + market_value_raw",
     "起始": "2005-09-02", "延遲": "當日（ffill）", "缺失語意": "2026 起僅涵蓋上市 → Avail_Valuation + 自算補值"},
    {"特徵群": "月營收（2 維）", "來源": "revenue_raw（FinMind）",
     "起始": "2002-02-01", "延遲": "+11 天（法定 10 日前公告）", "缺失語意": "as-of join，無值則沿用前期"},
    {"特徵群": "季報 EPS / 毛利率 / ROE", "來源": "financials_raw + balance_sheet_raw",
     "起始": "2004-03-31（資產負債表 2011-12）", "延遲": "+45 天（Q4/年報 fundamentals_v2 改 +90 天）",
     "缺失語意": "2011 前無 Book_Value/ROE 來源 → Avail_Financials"},
    {"特徵群": "自由現金流（1 維）", "來源": "cashflow_raw",
     "起始": "2008-06-30", "延遲": "+45 天", "缺失語意": "併入 Avail_Financials"},
    {"特徵群": "股利殖利率（1 維）", "來源": "dividend_raw（MOPS t187ap45）",
     "起始": "2005-05-23", "延遲": "董事會分派日", "缺失語意": "無配息即 0（合法值）"},
    {"特徵群": "Group D 總經（12 維）", "來源": "macro_raw + TAIFEX",
     "起始": "2005-01-03（期貨/選擇權 2018-06）", "延遲": "當日",
     "缺失語意": "同日全市場同值；macro_norm='ts' 需 252 天暖機"},
]
