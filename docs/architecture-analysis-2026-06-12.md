# MarketMamba V6 整體架構分析

> 分析日期：2026-06-12｜範圍：資料管線 → 特徵工程 → 模型架構 → 推論輸出 → 訊號系統
> 依據：`config.py`、`feature_engineer.py`、`trainer.py`、`architecture.py`、`inference.py`、`run_daily_inference.py`、`V6/results/` 實際輸出

---

## 1. 系統總覽

```
FinMind / yfinance / TWSE direct（2005 至今，~2,515 支）
  → merger.py（只 load raw parquet）
  → feature_engineer.build_features()（56/59 維，4 因子組，left join 到 prices_raw）
  → clean_and_scale()（per-date winsorize 1%/99% + cross-sectional z-score）
  → TemporalCrossSectionDataset（252 天視窗，≥202 天門檻，lazy loading）
  → MarketMambaV6（~4M 參數：GroupedEmbedding → 3-branch Mamba → GATv2 → Gating → 3 heads）
  → MC-Dropout ×30 → df_kelly.csv / df_traj.csv
  → scanner / pattern_scanner / signal_conditions → action_signals.json 等
  → git push → Render API → Vercel 前端
```

整體設計水準高於一般個人專案：point-in-time 處理（財報 +2 月發布延遲）、NaN-aware target 標準化、CSR 子圖快取、MC-Dropout 不確定性、走期驗證設定都有到位。以下聚焦「實際發現的問題」與升級空間。

---

## 2. 資料層分析

### 設計現況

| 面向 | 現況 | 評價 |
|------|------|------|
| 來源 | yfinance → TWSE direct → FinMind 三層 fallback | ✅ 韌性佳 |
| 覆蓋 | 2005 至今、4 位數代碼過濾、市值/流動性門檻 | ✅ |
| 合併方向 | 全部 left join 到 prices_raw | ✅ universe 單一決定者 |
| Point-in-time | 財報以發布日 +2 月對齊（`_published_ref`） | ✅ 避免 look-ahead |
| 標準化 | per-date winsorize + cross-sectional z-score，NaN→0 | ⚠️ 見 D1 |

### 發現的問題

**D1（嚴重）：Group D macro 特徵被 cross-sectional z-score 歸零。**
`_merge_macro()` 以 Date join，同一天所有股票的 `VIX`、`TWII_Return`、`FED_Rate`、`Fear_Greed` 等 12 維完全相同。`clean_and_scale()` 對全部 `FEATURE_COLS` 做 per-date z-score：std=0 → `(x-mean)/(0+1e-9) = 0`。**整個 Group D 在訓練與推論時恆為全 0**，模型的 macro_environment 分支（佔 d_model 256 中的 54 維投影）完全沒有資訊輸入，V6.1 新增的 5 個 macro 特徵實際無效。驗證方式：

```python
# 在 build_features + clean_and_scale 之後執行
print(df.groupby("Date")["VIX"].std().describe())   # 全 0 → 確認
print(df["VIX"].abs().max())                          # 0.0 → 確認
```

修法：macro 特徵改用 time-series 標準化（如 expanding 或 rolling 252d z-score，僅用過去資料避免 look-ahead），其餘股票級特徵維持 cross-sectional。這需要重訓才能吃到效益。

**D2（中）：每日有效 cross-section 只有 ~1,000–1,300 支，遠低於 universe 2,515。**
`ic_analysis.json` 顯示每日 n_stocks 約 1,017–1,291。原因疊加：≥202 天歷史門檻（新股一年內進不來，合理）+ `clean_and_scale` 的 dropna（>30% 特徵 NaN 即剔除）。建議印出每日「剔除原因統計」（規則 7），確認沒有可修復的 NaN 來源（例如某 raw 檔案斷供導致整組特徵 NaN）。

**D3（中）：存活者偏差。** `ticker_universe` 來自當前 FinMind 上市清單，已下市股票不在訓練資料中。對 2005–2026 的訓練區間而言，模型從未看過「走向下市的股票長什麼樣」，Alpha 預測對地雷股的辨識力存疑。完整修復成本高（需歷史成分股名單）；務實做法是至少在回測報告中註明此偏差，並依賴流動性過濾降低實害。

**D4（低，工程債）：兩套推論實作並存。** `models/inference.py:run_inference()` 與 `run_daily_inference.py:run_inference()` 是兩份獨立邏輯，欄位都已分歧（前者輸出 `Uncertainty_5d/20d/60d` + `Slippage`，後者輸出 `Uncertainty` + `Slippage_Est`，實際線上跑的是後者）。建議刪除或明確標註前者為 deprecated，避免日後改 A 忘了改 B。

---

## 3. 特徵工程分析

四組 56 維（V6.2 → 59 維加 RS）配置合理，機構籌碼 20 維是台股的正確重點。幾個觀察：

- **Group A 缺「截面相對」概念**：`Return_5d` 等是絕對報酬，cross-sectional z-score 後雖隱含相對性，但 V6.2 的 RS_5d/20d/60d 補上明確的相對強度是正確方向。可再考慮加 cross-sectional rank 特徵（rank 比 z-score 對肥尾更穩健）。
- **MA_20/MA_60 與 Close 高度共線**，z-score 後資訊量有限；可改成 `Close/MA_20 - 1`（乖離率）形式，資訊密度更高。
- **產業相對特徵缺席**：KG 已有產業結構，但特徵層沒有「個股 vs 所屬產業均值」的相對量（報酬、PER、外資買超）。這類特徵通常對台股 alpha 貢獻明顯。
- Target 設計（forward alpha vs TWII，per-date z-score，NaN-aware）正確。

---

## 4. 模型架構分析

### 組件評價

| 組件 | 設計 | 評價 |
|------|------|------|
| FactorGroupedEmbedding | 按組維度比例分配 sub_dim | ✅ 合理；但 D1 使 macro 的 54 維浪費 |
| MultiScaleMambaEncoder | 20/60/252 三分支 + softmax gate | ✅ 概念好；已知 gate 偏 Long，待 V6.2 觀察 |
| GATv2 | 單層、4 heads、KG 邊 + 邊權重 | ⚠️ 見 M1；單層只能聚合一階鄰居 |
| Gating Fusion | sigmoid gate 混合 temporal/graph | ✅ |
| MultiHorizonHead | 3 個獨立 Linear | 簡單但夠用；無共享 MLP 趨於保守 |
| Loss | 3-horizon MSE + ListNet top-1（20d） | ⚠️ ListNet top-1 只關心第一名的機率分布，對 Top-50 排序的監督偏弱 |

### 發現的問題

**M1（嚴重）：推論時 GAT 的圖被 mini-batch 切碎。**
`run_daily_inference.py` 為了 6GB VRAM 以 `INFER_BATCH=128` 分批，`get_batch_edges_csr(batch_stocks, ...)` 只取**批內**邊。跨批的 KG 邊全部丟失——以 ~1,200 支股票、640K 條邊計，每批 128 支只保留約 1% 量級的邊，且批次按 stock_id 順序切，等於 GAT 在推論時大部分退化成 identity（無邊時直接 return h）。但訓練時是整個 cross-section 一次 forward、用完整圖。**訓練/推論的圖結構嚴重不一致**，GAT 學到的 cross-stock 修正在線上幾乎沒有發揮。

修法（擇一）：
1. **兩段式推論**：先整批跑 embedding+encoder 得 `h_temporal`（逐批、無圖依賴），再把全部 `h_temporal`（1,200×256，<2MB）一次餵 GAT+fusion+head——GAT 層本身吃不了多少 VRAM，瓶頸只在 Mamba 的 252 步序列。改動小、不需重訓，**預期是目前 CP 值最高的單一修復**。
2. 或按 KG community/產業分群切批，至少保留群內邊。

**M2（高，部署陷阱）：推論未傳 padding_mask。**
`X, _, valid_stocks, _ = test_ds[0]` 把第 4 個回傳值（padding_mask）直接丟棄，`model(x_b, edge_index, edge_attr)` 沒帶 mask。目前 V6.1 checkpoint 訓練時也沒用 mask，所以一致；但 **V6.2（USE_PADDING_MASK=True）訓練完成部署時，推論端必須同步傳入 mask**，否則上歷史不足 252 天的股票會出現 train/inference 行為不一致。建議現在就改好（mask 為 None 時行為不變，向下相容）。

**M3（中）：MC-Dropout 的不確定性未校準，且無固定 seed。**
30 次 dropout 採樣的 std 是「模型內部分歧度」，不是校準後的預測區間；且每日推論的隨機性會讓同一支股票的 Uncertainty / Signal_Quality 在邊界附近日間抖動，間接造成排名穩定性條件（scanner 30 分權重）的 noise。低成本改善：推論時固定 `torch.manual_seed(date 為 seed)`，使日間差異只反映資料變化；中期改善見升級建議 U3。

**M4（已知，追蹤中）：scale_gate 偏 Long。** 已列在 backlog（branch dropout / gate entropy 正則化），不重複。補充一點：D1 修復後 macro 資訊進入 Long branch 的內容會改變，建議 D1 與 gate 正則化分開實驗，避免歸因混淆。

---

## 5. 輸出層分析

### 實測表現（`ic_analysis.json`，2026-04-29 ~ 06-11）

| Horizon | 天數 | mean IC | ICIR | IC>0 比例 |
|---------|------|---------|------|-----------|
| 5d | 24 | 0.071 | 0.46 | 75% |
| 20d | 9 | 0.136 | 2.15 | 100% |

20d 數字亮眼但僅 9 個樣本（且觀察窗重疊，自相關高估 ICIR），尚不能下結論；5d 的 IC 0.07 已超過 config 的接受門檻 0.05。**建議持續累積到 60+ 天再評估**，並把 IC 時序圖納入前端監控頁。

### 發現的問題

**O1（中）：`Suggested_Weight` 名為 Kelly 實為「正 SQ 比例分配」**，權重攤到數百支正 SQ 股票上，單檔 ~0.007，無法直接執行。下游 scanner/sim_engine 已另有進場邏輯，此欄位實質上只是排序的另一種表達。建議：要嘛改名（避免誤導），要嘛真做組合建構（見 U4）。

**O2（低）：`Confidence` 標籤用固定 bins（0.02/0.05）切 Uncertainty**，模型重訓後 uncertainty 分布會漂移，標籤即失真。scanner 用當日 Q30 分位數的做法才正確，建議 df_kelly 的 Confidence 也改成分位數制。

**O3（低）：`Signal_Quality` 上限 10 截斷造成 Top 區壓縮**——今日輸出第一名就是 10.0 滿格，前幾名之間失去解析度。可改 clip 前先存 raw 值欄位，前端顯示用 clip 值、排序用 raw 值。

訊號系統（scanner 1.2/1.3 + signal_conditions 四層退場 + pattern_scanner + sim_engine_v3）設計完整、規則可解釋，屬於成熟的 rule-based 層，這裡不展開；主要依賴上游 alpha/uncertainty 品質，故上述 M1/M3 的修復會直接改善訊號穩定性。

---

## 6. 升級建議（按優先級）

### P0 — 不需重訓、直接改善線上品質

| # | 項目 | 對應問題 | 工作量 |
|---|------|----------|--------|
| U1 | **推論改兩段式，GAT 吃完整圖** | M1 | 小（改 run_daily_inference 推論迴圈） |
| U2 | 推論端補傳 padding_mask（None 相容） | M2 | 極小 |
| U2b | MC-Dropout 固定每日 seed | M3 | 極小 |
| U2c | 印出每日 cross-section 剔除原因統計 | D2 | 小 |

U1 完成後建議比對改前/改後同一天的 Top 50 排名差異，量化 GAT 實際貢獻——若差異極小，也是有價值的資訊（代表 GAT 可考慮簡化）。

### P1 — 需重訓，預期 alpha 提升最大

| # | 項目 | 對應問題 |
|---|------|----------|
| U3a | **修 macro z-score 歸零**：Group D 改 time-series 標準化（expanding/rolling，只用過去資料） | D1 |
| U3b | V6.2 RS 特徵上線（已規劃）+ 乖離率取代原始 MA + 產業相對特徵 | §3 |
| U3c | ListNet top-1 → ListMLE 或 pairwise RankNet（Top-K 排序監督更強）；或加 soft-IC loss（直接最大化 batch 內 Pearson/Spearman 近似） | M 表 |

U3a 是「免費資訊」：12 維特徵 + 54 維表徵空間目前完全閒置，修復後 macro regime（VIX、利率、景氣燈號）才真正進得了模型——這也與 scanner 的保守模式判斷（TWII vs MA60）形成模型內/外雙層 regime 感知。

### P2 — 中期架構演進

- **U4 組合建構層**：以 `Net_Alpha_20d` 為期望報酬、Uncertainty 為風險，做簡單 mean-variance 或風險平價的 Top-N（10–20 檔）權重，取代比例分配的 Suggested_Weight；可直接餵 sim_engine_v3 驗證。
- **U5 不確定性升級**：MC-Dropout → 小型 deep ensemble（3–5 個不同 seed 的 checkpoint 平均）。校準更好、推論還更快（5 次 forward vs 30 次），RTX 3060 可行；或用 conformal prediction 在不重訓下校準現有 uncertainty。
- **U6 GAT 強化**（在 U1 驗證 GAT 有貢獻的前提下）：單層 → 2 層（二階鄰居）、edge type embedding（conglomerate/supply_chain/corr 分型）取代純權重純量。
- **U7 評估基礎建設**：把 walk-forward 驗證（config 已有參數但似乎未常態執行）跑成例行報告，含分年 IC、分產業 IC、換手率，作為每次重訓的 go/no-go 依據。

### P3 — 觀察與保留

- scale_gate 偏 Long 的處置等 V6.2 訓練結果再定（已在 backlog）。
- 存活者偏差（D3）：成本高，先記錄在回測免責中。
- 刪除/封存 `models/inference.py` 的重複實作（D4）。

---

## 7. 結論

系統的工程面（資料韌性、point-in-time、自動化、訊號可解釋性）已相當紮實。最大的隱藏問題在兩處「設計正確但執行斷裂」：**macro 特徵被標準化流程歸零（D1）**與**GAT 在推論時被 mini-batch 切碎（M1）**——兩者都讓已付出訓練成本的模型能力在線上無法兌現。建議順序：先做 P0（不重訓、立即生效），把 D1 修復併入 V6.2/V6.3 重訓，再依 walk-forward 證據推進 P2。
