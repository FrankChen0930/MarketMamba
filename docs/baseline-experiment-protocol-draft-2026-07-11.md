# Baseline 對照實驗協定規格書（v1.0 定案）

> 對應計畫：`planing/研究計畫_方向二_Baseline對照.md` Step 1
> 狀態：**定案凍結（2026-07-12）**——§8 四個決定已由使用者拍板，全部採建議方案。自此不再更動 universe / 特徵集 / label / 切分 / 成本模型（否則所有 baseline 重跑）；§4 的確切欄位清單於首次實作時補入附錄後一併凍結。
> 鐵律（不可妥協）：同一 universe、同一切分、同一 label、同一成本模型；GBDT 必須配 lag/rolling 特徵工程，不可用「當下時點特徵 vs 完整序列」的作弊對照。

---

## 1. 對照階梯

| 階 | 模型 | 可解釋性 | 訓練環境 |
|---|---|---|---|
| 1 | Ridge / Lasso | 係數直接可讀（地板） | 本機 CPU |
| 2 | LightGBM（GBDT，必備） | SHAP | 本機 CPU |
| 3 | LSTM 或 GRU 擇一 | 低 | 本機 RTX 3060，先試；過慢才上 Colab |
| 4 | Mamba+GATv2（現有） | 低（scale gate / KG 邊可部分解讀） | 沿用既有結果，不重訓 |

第 3 階的作用是隔離「贏是因為 Mamba 架構本身，還是任何序列模型皆可」；第 1、2 階吃扁平特徵向量（含 lag/rolling 工程化時序資訊），第 3、4 階吃完整序列。

## 2. Universe（與現行管線完全一致）

- 台股 4 位數字代碼（`^\d{4}$`），上市 + 上櫃
- 市值 ≥ 5 億台幣、5 日均成交值 ≥ 1,000 萬台幣（`config.MIN_MARKET_CAP_TWD` / `MIN_AVG_VOLUME_5D`）
- 每個交易日 cross-section 只納入「至少 202 天歷史」的股票（SEQ_LEN 252 × 0.8，與 `TemporalCrossSectionDataset` 一致）——**此規則對四階模型一體適用**，即使 Ridge 只吃當日特徵也套同樣的納入條件，確保每天的評分母體四階完全相同
- 已知資料污染處理：prices_raw 過濾非 4 位數代碼、(stock_id, Date) 去重 keep=last

**已知限制（誠實聲明，四階共同承受）**：universe 來自現行 FinMind 抓取清單，含存活者偏差（D3，未修復）。因四階模型吃同一份資料，偏差影響「絕對數字」但不影響「模型間相對比較」的公平性。

## 3. 特徵集

- **59 維 `FEATURE_COLS`**（V6.2 訓練 config：Group A 15 維含 RS_5d/20d/60d + B 20 + C 12 + D 12），與雙模型（對照目標）訓練時完全一致
- 清洗與標準化沿用 `clean_and_scale()`：橫斷面 z-score ± 3σ 截斷；Group D macro 用 `macro_norm="ts"`（time-series expanding z-score，shift(1) 無 look-ahead）
- 序列模型（第 3、4 階）輸入：(N, 252, 59)；扁平模型（第 1、2 階）輸入見 §4

## 4. GBDT / 線性模型的時序特徵工程（公平性核心）

扁平模型除「當日 59 維」外，補上工程化時序資訊：

| 類型 | 定義 | 適用欄位 | 新增維度 |
|---|---|---|---|
| Lag | t-1, t-5, t-20 快照 | 全部 59 維 | 177 |
| Rolling mean | 5/20/60 日 | 價量與籌碼核心欄（Close, Volume, Return_1d, Foreign_Net, Investment_Trust_Net, RSI_14, Volatility_20d 等 ~12 欄） | ~36 |
| Rolling std | 20/60 日 | 同上 | ~24 |
| 動能 | 過去 5/10/20/60 日累積報酬 | Close | 4 |

合計約 300 維（實作時定案確切清單並寫入本協定附錄）。線性模型另做特徵標準化後可加 L1/L2 正則路徑圖。

## 5. Label

- **主 label：`rank(Alpha_5d)`**——前瞻 5 日累積報酬 − TWII 同期報酬（`_add_alpha_targets()` 現行定義），再做每日橫斷面百分位 rank 轉換（與 v6_short 訓練目標一致）
- 副 label：`rank(Alpha_20d)`（對照 v6_trend）
- 理由：現行 production 線（雙模型）就是 rank 目標，且 Phase 2 已實證 rank 目標近乎翻倍 IC；baseline 用同一目標才是對「現在的系統」公平的對照

## 6. 資料切分（見 §8 決定 1）

**建議方案 A（主）：與 Phase 3 同 harness 的單一切分**
- train：2012-01-01 ~ 2023-12-31；val（early-stop 用）：train 尾端 15%；test：2024-01-01 ~ 2026-06-02（≈580 個交易日，與 dropout sweep 對照實驗同窗）
- 優點：Mamba 端數字（v6_short 5d IC 0.0951 / 0.0870 同 harness 重跑值）可直接引用、不需重訓、零 Colab 成本
- 引用紀律：與 baseline 對照時**用同 harness 重跑值 0.0870 當 Mamba 的數字**（Phase 3-A 的教訓：歷史峰值 0.0951 因切分/seed 差異不可直接比）

**方案 B（補強，只跑便宜的階 1–2）：Expanding-Window Walk-Forward**
- 沿用 `walk_forward.py` 參數：2012 起、測試窗 6 個月、步進 3 個月、最少 3 年訓練 → ~36 folds
- Ridge/GBDT 每 fold 重訓成本低，本機可行；**Mamba 不跑**（需 36 次 Colab 重訓，屬 U7 例行化的範圍）
- 產出「baseline 的 IC 時間穩定性」當附註，明確標示 Mamba 無對應 WF 數字

## 7. 評估指標與成本模型

**訊號層（主要對照，無成本）**：
- 每日 Spearman IC（預測 vs 實際 Alpha_5d）、mean IC、ICIR（mean/std）、IC>0 天數比例、t 統計量
- 注意 5d 窗口重疊 → 日 IC 序列自相關，t 統計量以 Newey-West（lag=horizon）修正後另列

**組合層（Sharpe/MDD 用，扣成本）**：
- Top50 等權、每 5 個交易日再平衡（與 horizon 對齊）、以收盤價成交
- 成本：買 0.15%、賣 0.45%（手續費 0.15% + 證交稅 0.30%，與 `sim_engine_v3` 一致）；不另計滑價（四階相同故不影響排序，註明即可）
- 指標：年化報酬、年化 Sharpe、最大回撤、換手率、對 TAIEX 超額

**可解釋性（定性欄）**：係數表（階 1）/ SHAP summary（階 2）/ 無或有限（階 3、4），對照表每階附一句「這個模型能告訴你什麼」

## 8. 四個關鍵決定（2026-07-12 使用者拍板，全採建議方案）

| # | 決定 | 定案 | 理由 |
|---|---|---|---|
| 1 | 切分：A 單一切分（同 Phase 3 harness）為主 + B 便宜階 WF 為輔，或全面 WF？ | **A 為主、B 為輔** | 全面 WF 需 36 次 Mamba 重訓（Colab 成本高、且模型實驗暫停中）；A 可直接引用既有數字 |
| 2 | Label：rank 或 raw alpha？ | **rank** | 與現行 production 線一致；IC 本身就是 rank 相關 |
| 3 | 主 horizon：5d 或 20d？ | **5d 主、20d 副** | 使用者操作短線；v6_short 是主要對照目標 |
| 4 | 組合層規格：Top50 等權 / 5 日再平衡 OK？ | 如 §7 | 簡單、可重現、與 horizon 對齊；不引入 Kelly 權重等額外自由度 |

---

**定案後的凍結範圍**：§2 universe、§3 特徵集、§5 label、§6 切分、§7 成本模型。§4 的確切欄位清單已於首次實作（2026-07-12）定案補入附錄 A，自此凍結。

---

## 附錄 A：扁平模型特徵清單（2026-07-12 實作定案，凍結；階 1 線性與階 2 GBDT 共用）

實作：`V6/experimental/baseline_common.py`（單一事實來源，欄位名與順序以 `all_feature_names()` 為準）。
快取：`Data/processed_v6/baseline_cache/`（不進 git）。

**合計 300 維 = 59 base + 177 lag + 36 rolling mean + 24 rolling std + 4 momentum**

| # | 區塊 | 定義 | 維度 |
|---|---|---|---|
| 1 | Base | 59 維 `FEATURE_COLS`（V6.2 config：A 15 + B 20 + C 12 + D 12），經 `clean_and_scale(macro_norm="ts")` | 59 |
| 2 | Lag | 全部 59 維的 t-1、t-5、t-20 快照（欄名 `{col}_lag{n}`） | 177 |
| 3 | Rolling mean | 核心 12 欄 × 窗 {5, 20, 60}（欄名 `{col}_rmean{w}`） | 36 |
| 4 | Rolling std | 核心 12 欄 × 窗 {20, 60}（欄名 `{col}_rstd{w}`） | 24 |
| 5 | 動能 | `Mom_5d/10d/20d/60d`：原始收盤價過去累積報酬，再每日橫斷面 winsorize [1%,99%] + z-score | 4 |

**Rolling 核心 12 欄（凍結）**：`Close, Volume, Return_1d, Foreign_Net, Investment_Trust_Net, Dealer_Net, Margin_Balance, Short_Balance, OBV, RSI_14, Volatility_20d, Foreign_Holding_Pct`

**計算慣例（凍結）**：
1. Lag 與 rolling 皆計算在「橫斷面標準化後」的特徵上（lag = 該股 N 天前的橫斷面 z 分數；rolling mean = 相對強弱的持續性），全特徵空間尺度一致，免除二次標準化。
2. Lag/rolling 以股票內「列」為單位 shift；`clean_and_scale` 偶發剔除列時為交易日 shift 的近似（剔除集中在歷史起點、协定窗內影響可忽略）。
3. Rolling 用 `min_periods=window`（嚴格），衍生特徵 NaN 一律補 0（= 橫斷面均值，與 `clean_and_scale` 慣例一致）。
4. Matrix 自 2010-01-01 起建（raw 自 2009 起做 lookback 緩衝）：train 2012 起前留 ≥202 天納入門檻 + macro ts 暖機。macro expanding 統計以 2010 起算（非 2005 全歷史）——macro 為橫斷面常數，對 rank 類 label/每日 IC 無影響（同 2026-06-19 dual inference「macro 近似只造成水位平移、不動排名」實證）。
5. Label = per-date pct-rank 置中 [-0.5, +0.5]（`method="average"`，同 `short_model.rank_transform` 語意），只在 eligible（≥202 天歷史）且 Alpha 非空的股票間排名。
6. **已知事實（誠實聲明）**：`_add_alpha_targets` 的 TWII 減基準分支從未觸發（macro_raw 欄名為 `TWII_Close` 非 `TWII`，Colab 訓練 Mamba 同路徑）→ 所有模型的 Alpha label 實為 raw forward return。per-date rank 對當日常數平移免疫，故四階 label 完全一致、對照公平性不受影響；不回頭修改（改了會破壞與既有 Mamba 數字的同 harness 可比性）。
