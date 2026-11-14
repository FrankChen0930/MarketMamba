# MarketMamba V6.1 — 模型架構與訓練資料完整說明

> 最後更新：2026-05-29  
> 對應版本：V6.1（`v6_best.pt`，epoch 14，val_IC = 0.0825）

---

## 目錄

1. [模型總覽](#1-模型總覽)
2. [架構細節](#2-架構細節)
   - [FactorGroupedEmbedding（因子分組嵌入）](#21-factorgroupedembedding)
   - [MultiScaleMambaEncoder（多尺度時序編碼）](#22-multiscalemambaencoder)
   - [GraphAttentionLayer（GATv2 截面互動）](#23-graphattentionlayer)
   - [Gating Fusion（門控融合）](#24-gating-fusion)
   - [MultiHorizonHead（多期預測頭）](#25-multihorizonhead)
3. [輸入輸出規格](#3-輸入輸出規格)
4. [特徵工程（56 維）](#4-特徵工程56-維)
5. [訓練資料](#5-訓練資料)
6. [訓練配置](#6-訓練配置)
7. [損失函數](#7-損失函數)
8. [知識圖譜（Knowledge Graph）](#8-知識圖譜)
9. [推論流程](#9-推論流程)
10. [Walk-Forward 驗證](#10-walk-forward-驗證)
11. [推論輸出欄位](#11-推論輸出欄位)
12. [超參數速查表](#12-超參數速查表)

---

## 1. 模型總覽

| 項目 | 值 |
|------|-----|
| **模型名稱** | MarketMamba V6.1 |
| **總參數量** | 11,456,643（≈ 11.5M） |
| **輸入維度** | (N_stocks, 252 天, 56 特徵) |
| **輸出維度** | (N_stocks, 3) → [Alpha_5d, Alpha_20d, Alpha_60d] |
| **主要預測目標** | Alpha_20d（20 日超額報酬 vs TWII 基準） |
| **訓練環境** | Google Colab A100 |
| **推論環境** | 本機 WSL2 + RTX 3060（CUDA） |
| **Checkpoint** | `V6/models/v6_best.pt`（epoch 14, val_IC=0.0825） |

**整體資料流**：

```
(N, 252, 56)
  → FactorGroupedEmbedding       → (N, 252, 256)   ← 因子分組投影
  → MultiScaleMambaEncoder       → (N, 256)         ← 多尺度時序建模
  → GraphAttentionLayer (GATv2)  → (N, 256)         ← 截面股票互動
  → Gating Fusion                → (N, 256)         ← 時序 + 圖注意力融合
  → MultiHorizonHead             → (N, 3)           ← 5d / 20d / 60d
```

---

## 2. 架構細節

### 2.1 FactorGroupedEmbedding

**功能**：將 56 維特徵按因子類型分組，各自投影到獨立子空間後合併，避免不同性質的因子訊號相互污染。

**版本特性（V6.1 方案 B — 比例分配）**：每個因子組獲得的子空間維度與其輸入維度成正比，而非均分。機構流量（20 維）比價格動能（12 維）獲得更多表示空間。

```python
# 輸入: (N, 252, 56)
# 分組投影 (比例 = group_dim / total_dim * d_model)
proj_A: Linear(12 → 55)   # price_momentum      (12/56 × 256 ≈ 55)
proj_B: Linear(20 → 91)   # institutional_flow  (20/56 × 256 ≈ 91)  ← 最大
proj_C: Linear(12 → 55)   # fundamentals        (12/56 × 256 ≈ 55)
proj_D: Linear(12 → 55)   # macro_environment   (12/56 × 256 ≈ 55)

# Concat → LayerNorm → Dropout
# 輸出: (N, 252, 256)
```

| 參數 | 值 |
|------|-----|
| `d_model` | 256 |
| 各組子維度分配 | 比例於 input_dim，餘數補至最大組（institutional_flow） |
| Dropout | 0.1 |

---

### 2.2 MultiScaleMambaEncoder

**功能**：三條並行的 Mamba SSM 分支，分別捕捉短、中、長期時序模式，最後透過可學習的注意力權重融合。

```
Short branch (20 天, 2 層 Mamba)  → h_short: (N, 256)  ← 動量、短線籌碼
Mid   branch (60 天, 3 層 Mamba)  → h_mid  : (N, 256)  ← 季度趨勢、法人中期
Long  branch (252天, 3 層 Mamba)  → h_long : (N, 256)  ← 基本面循環、年度規律

scale_gate: Linear(256×3 → 3) + Softmax
fused = softmax_weights · [h_short, h_mid, h_long]  → (N, 256)
```

| 參數 | 值 |
|------|-----|
| `d_model` | 256 |
| `d_state` | 32 |
| `d_conv` | 4 |
| `expand` | 2 |
| 各分支層數 | Short=2, Mid=3, Long=3（共 8 層） |
| 各分支輸入序列長度 | 20 / 60 / 252 天 |

**MambaStack 結構**（每個分支內部）：

```python
for mamba, norm in zip(layers, norms):
    x = x + mamba(norm(x))   # pre-norm residual
```

---

### 2.3 GraphAttentionLayer

**功能**：跨股票截面的圖注意力，利用知識圖譜（供應鏈、集團、同業、滾動相關性）建立股票間的資訊傳遞。

```python
GATv2Conv(
    in_channels  = 256,
    out_channels = 64,    # head_dim = d_model / n_heads = 256/4
    heads        = 4,
    edge_dim     = 1,     # 邊權重（0.5~0.8）
    dropout      = 0.1,
    concat       = True,  # 輸出 = 64×4 = 256
)
# Residual + Pre-norm
h = h_in + dropout(GAT(norm(h)))  → (N, 256)
```

| 參數 | 值 |
|------|-----|
| `n_heads_gat` | 4 |
| `MAX_NEIGHBORS_GAT` | 每股最多 15 個鄰居（防止注意力崩塌） |
| 邊數量（知識圖譜） | ~43,000 條 |

---

### 2.4 Gating Fusion

**功能**：用可學習的門控機制動態決定每支股票應更偏重「時序表示」還是「圖注意力表示」。

```python
gate_input  = cat([h_temporal, h_graph], dim=-1)   # (N, 512)
gate_weight = Linear(512 → 256) + Sigmoid()         # (N, 256)
h_fused     = LayerNorm(gate_weight × h_temporal + (1 - gate_weight) × h_graph)
```

---

### 2.5 MultiHorizonHead

**功能**：三個獨立的線性預測頭，分別輸出不同期 Alpha。

```python
head_5d  : Linear(256 → 1)   # 主要：輔助信號
head_20d : Linear(256 → 1)   # 主要：核心預測目標
head_60d : Linear(256 → 1)   # 主要：長期方向
# 輸出: (N, 3) → [pred_5d, pred_20d, pred_60d]
```

---

## 3. 輸入輸出規格

### 輸入

| 維度 | 說明 |
|------|------|
| N | 當日可投資股票數（約 1,800~2,500 檔） |
| T = 252 | 序列長度：1 個完整交易年 |
| F = 56 | 特徵維度（V6.1 擴充，V6.0 為 46） |

不足 252 天的股票以零填充（左側補 0），不足 252 × 0.8 = 201 天者排除。

### 輸出

| 欄位 | 說明 |
|------|------|
| `pred_5d` | 5 日 Alpha 預測（相對 TWII） |
| `pred_20d` | 20 日 Alpha 預測（主要評估指標） |
| `pred_60d` | 60 日 Alpha 預測（長期參考） |

---

## 4. 特徵工程（56 維）

所有特徵在訓練前經過**截面標準化**：  
1. Winsorize（截斷 1%–99% 分位數）  
2. 截面 Z-score（以當日全市場為基準）

### Group A — price_momentum（12 維）

| 特徵 | 計算方式 |
|------|---------|
| `Open, High, Low, Close, Volume` | TWSE 原始 OHLCV |
| `Return_1d` | Close.pct_change(1) |
| `Return_5d` | Close.pct_change(5) |
| `Return_20d` | Close.pct_change(20) |
| `MA_20` | 20 日簡單移動平均（min_periods=10） |
| `MA_60` | 60 日簡單移動平均（min_periods=30） |
| `RSI_14` | 14 日 RSI（EWM 計算，避免 Wilder 平滑誤差） |
| `ATR_14` | 14 日 ATR（EWM span=14） |

### Group B — institutional_flow（20 維）

**核心法人** (5)：

| 特徵 | 來源 | 說明 |
|------|------|------|
| `Foreign_Buy` | TWSE 三大法人 | 外資買進 |
| `Foreign_Sell` | TWSE 三大法人 | 外資賣出 |
| `Foreign_Net` | TWSE 三大法人 | 外資淨買超 |
| `Investment_Trust_Net` | TWSE 三大法人 | 投信淨買超 |
| `Dealer_Net` | TWSE 三大法人 | 自營商淨買超 |

**融資融券** (6)：

| 特徵 | 說明 |
|------|------|
| `Margin_Purchase` | 融資買進 |
| `Margin_Repay` | 融資賣出（還款） |
| `Short_Sale` | 融券賣出 |
| `Short_Cover` | 融券回補 |
| `Margin_Balance` | 融資餘額（當日） |
| `Short_Balance` | 融券餘額（當日） |

**技術指標** (5)：

| 特徵 | 計算方式 |
|------|---------|
| `Day_Trade_Volume` | 當日沖銷比率（0~1） |
| `KD_K` | KD 隨機指標 K 值（9,3,3） |
| `KD_D` | KD 隨機指標 D 值 |
| `OBV` | On Balance Volume（滾動 60 日 Z-score 標準化） |
| `Volatility_20d` | 20 日對數報酬標準差 |

**V6.1 新增** (4)：

| 特徵 | 來源 | 說明 |
|------|------|------|
| `Holdings_Large_Pct` | 集保戶股權分散表（週） | 千張以上大戶持股占比 = 1 - 散戶持股比 |
| `Holdings_Large_Change` | 計算自上項 | 大戶持股比例週變化（diff 5天） |
| `Securities_Balance` | FinMind / TWSE | 借券餘額 |
| `Foreign_Holding_Pct` | FinMind | 外資累計持股比例（`ForeignInvestmentSharesRatio`） |

### Group C — fundamentals（12 維）

**基本估值** (V6.0 原有 10 維)：

| 特徵 | 更新頻率 | 說明 |
|------|---------|------|
| `PER` | 每日 | 本益比（前向填充） |
| `PBR` | 每日 | 股價淨值比 |
| `Revenue_MoM` | 月（+11 天 lag） | 月營收環比 |
| `Revenue_YoY` | 月 | 月營收年增率 |
| `EPS` | 季（+45 天 lag） | 每股盈餘 |
| `EPS_Surprise` | 季 | EPS 季環比增長（pct_change 4 季） |
| `Gross_Margin` | 季 | 毛利率 |
| `ROE` | 季 | 股東權益報酬率 |
| `Market_Cap_Log` | 每日 | log(市值)（log1p 處理） |
| `Book_Value` | 季 | 每股帳面價值 |

**V6.1 新增** (2)：

| 特徵 | 更新頻率 | 說明 |
|------|---------|------|
| `Dividend_Yield_Fwd` | 公告後 | 預期股利殖利率 = 最新現金股利 / Close（上限 20%） |
| `Free_Cash_Flow` | 季（+45 天 lag） | FCF = 營業現金流 + 投資現金流 |

> **Look-ahead 保護**：Revenue 用發布日（月底 +11 天）做 as-of join；財務報表用 +45 天 lag；借券餘額使用前向填充。

### Group D — macro_environment（12 維）

**基礎總體** (7)：

| 特徵 | 來源 | 說明 |
|------|------|------|
| `TWII_Return` | macro_raw | 加權指數日報酬 |
| `SPX_Return` | macro_raw（QQQ proxy） | 美股指數日報酬 |
| `VIX` | macro_raw | CBOE 恐慌指數 |
| `TNX` | macro_raw | 美 10 年期公債殖利率 |
| `Gold_Return` | macro_raw | 黃金期貨日報酬 |
| `Oil_Return` | macro_raw | 原油期貨日報酬 |
| `USD_TWD` | macro_raw | 美元對台幣匯率 |

**V6.1 新增** (5)：

| 特徵 | 來源 | 說明 |
|------|------|------|
| `Futures_OI_Foreign` | FinMind 期貨法人 | 外資期貨未平倉淨口數（多 - 空） |
| `Options_PC_Ratio` | FinMind 選擇權法人 | Put/Call 成交量比（預設值 1.0） |
| `Fear_Greed` | macro_raw（CNN F&G） | CNN 恐懼貪婪指數（0~100） |
| `Business_Signal` | macro_raw（台灣景氣燈號） | NDEV 景氣綜合判斷分數（+60 天 lag） |
| `FED_Rate` | macro_raw | 聯準會基準利率（macro_raw 優先，broken fed_rate.parquet 為 fallback） |

---

## 5. 訓練資料

### 股票宇宙

| 項目 | 值 |
|------|-----|
| **涵蓋範圍** | 台灣上市（TWSE）+ 上櫃（TPEX）全市場 |
| **股票數量** | 約 2,500～2,888 支 |
| **篩選條件 1** | 市值 ≥ 5 億台幣（排除微型股） |
| **篩選條件 2** | 5 日平均日均量 ≥ 1,000 萬台幣（實盤流動性） |

### 時間範圍

| 項目 | 值 |
|------|-----|
| **開始日期** | 2005-01-01（法人資料最早可用日） |
| **結束日期** | 動態（每日推論更新至最新） |
| **序列長度** | 252 個交易日（1 個完整年度） |
| **訓練集比例** | 約 85%（`VAL_RATIO = 0.15`） |

### 訓練標籤（Alpha 目標）

```
Alpha_Nd = 股票 N 日累積報酬 - 加權指數（TWII）N 日累積報酬
```

| 標籤 | 說明 | 損失權重 |
|------|------|---------|
| `Alpha_5d` | 5 日超額報酬（輔助） | 0.3 |
| `Alpha_20d` | 20 日超額報酬（**主要**） | 1.0 + ListNet 0.5 |
| `Alpha_60d` | 60 日超額報酬（輔助） | 0.3 |

> 標籤為未來值（look-ahead），僅用於訓練，推論時不可用。  
> 訓練時對每個 batch（一日截面）進行**截面 Z-score 標準化**（NaN-aware）。

### 資料來源優先級

```
1. yfinance（價格、總體指標）
2. TWSE Direct API（三大法人）
3. FinMind API（法人、融資、財務報表）
```

---

## 6. 訓練配置

### Dataset 設計（TemporalCrossSectionDataset）

- **Lazy Loading**：每個 `__getitem__` 回傳一整日的截面（N_stocks, SEQ_LEN, INPUT_DIM），非預先計算（預計算需 ~280 GB RAM）
- **每日 batch**：1 個 batch = 1 個完整交易日截面（`batch_size=1`）
- **子採樣**：訓練時可設定 `N_SAMPLE_TRAIN` 限制每 batch 最大股票數（現為 None = 使用全部）
- **KG 邊提取**：使用 scipy CSR 稀疏矩陣子圖提取（~1ms/batch vs. Python loop ~500ms/batch）

### 優化器與排程

| 項目 | 值 |
|------|-----|
| **優化器** | AdamW |
| **學習率** | 7e-5（V6.1 從 1e-4 調降，防止暖身期過早 peak） |
| **Weight Decay** | 1e-4 |
| **排程器** | OneCycleLR（餘弦退火） |
| **Warmup 比例** | 15%（V6.1 從 10% 延長，平滑暖身到衰減過渡） |
| **Gradient Clip** | 1.0（防止梯度爆炸） |
| **混合精度** | FP16（AMP enabled） |

### 訓練超參數

| 項目 | 值 |
|------|-----|
| **最大 Epoch** | 60 |
| **Early Stop** | 10 個 epoch IC 無提升 |
| **評估指標** | Spearman IC（截面 Alpha_20d vs. pred_20d） |
| **Checkpoint 策略** | IC 提升時儲存，ICR 模式優先 |

---

## 7. 損失函數

### Multi-Horizon Loss（多期組合損失）

```
Total Loss =   1.0 × MSE(pred_20d, alpha_20d)    ← 核心
             + 0.3 × MSE(pred_5d,  alpha_5d)     ← 輔助短期
             + 0.3 × MSE(pred_60d, alpha_60d)    ← 輔助長期
             + 0.5 × ListNet(pred_20d, alpha_20d) ← 排名損失
```

### ListNet Ranking Loss

排名損失使用 Top-1 Probability 形式（標準 ListNet）：

```python
pred_prob   = softmax(pred_20d,   dim=0)  # 全截面軟化排名
target_prob = softmax(alpha_20d,  dim=0)
loss = -mean(target_prob × log(pred_prob))
```

> 使用 **mean** 而非 sum，確保損失尺度與截面股票數 N 無關（避免 train/val N 不同導致損失不可比）。

NaN 目標用 valid mask 跳過，防止梯度傳播中斷。

---

## 8. 知識圖譜

### 四種邊類型

| 邊類型 | 邊權重 | 說明 |
|--------|--------|------|
| **Conglomerate（集團）** | 0.8 | 鴻海族、台積電生態系、台塑集團等人工定義 |
| **Supply Chain（供應鏈）** | 0.6 | TPEX 產業鏈平台爬蟲，同產業鏈內的上下游 |
| **TWSE Sector（同業）** | 0.5 | TWSE 產業分類相同的公司（每個產業上限 15 個鄰居） |
| **Rolling Correlation（滾動相關）** | 動態 | 60 天滾動 Pearson 相關 > 0.7 的股票對 |

### 圖快取規格

| 項目 | 值 |
|------|-----|
| **快取路徑** | `V6/data/cache_v6/knowledge_graph_cache.npz` |
| **總邊數** | ~43,000 條 |
| **每股鄰居上限** | 15（`MAX_NEIGHBORS_GAT`） |
| **格式** | `.npz`：`stock_ids`, `edge_index (2, E)`, `edge_attr (E,)` |
| **重建方式** | 一般不需重建；需要時執行 `graph_builder.py` |

### 訓練時邊提取

使用 **scipy CSR 矩陣**做子圖提取（batch_stocks 的局部索引），時間複雜度 O(nnz in subgraph)，比舊版 Python loop 快約 500 倍。

---

## 9. 推論流程

### MC-Dropout 不確定性估算

```python
model.train()   # 開啟 Dropout
for _ in range(30):
    pred = model(X, edge_index, edge_attr)
    preds.append(pred)

pred_mean = mean(stack(preds), dim=0)   # (N, 3) 期望 Alpha
pred_std  = std(stack(preds),  dim=0)   # (N, 3) 不確定性
```

- **MC 採樣次數**：30 次（`n_mc=30`）
- **不確定性**：`pred_std` 作為信心評估依據

### 推論後處理

1. **Alpha 截斷**（`Net_Alpha_20d`）：
   ```
   Net_Alpha_20d = (Exp_Alpha_20d - Slippage - 0.003).clip(min=-1.0)
   ```

2. **滑點分級**（依流動性）：
   | 5日日均量 | Slippage |
   |-----------|----------|
   | < 1億 | 0.8% |
   | 1億～5億 | 0.4% |
   | > 5億 | 0.2% |

3. **流動性過濾**：5 日日均量 < 1,000 萬 → `Net_Alpha_20d = -999`（排除）

4. **Signal_Quality（信號品質分）**：
   ```
   Signal_Quality = (Net_Alpha_20d / Uncertainty_20d).clip(-10, 10)
   ```

5. **信心標籤**：
   | Uncertainty_20d | 標籤 |
   |-----------------|------|
   | < 0.02 | 高信心 |
   | 0.02～0.05 | 中信心 |
   | ≥ 0.05 | 低信心 |

6. **Suggested_Weight**：`Signal_Quality` 正值比例正規化（Kelly-like）

---

## 10. Walk-Forward 驗證

| 參數 | 值 |
|------|-----|
| **測試視窗** | 6 個月（`WF_TEST_WINDOW_MONTHS = 6`） |
| **滾動步長** | 3 個月（`WF_STEP_MONTHS = 3`） |
| **最小訓練年數** | 3 年（`WF_MIN_TRAIN_YEARS = 3`） |
| **IC 門檻** | ≥ 0.05（`IC_THRESHOLD`） |
| **ICIR 門檻** | ≥ 0.5（`ICIR_THRESHOLD`） |

**評估指標 IC（Information Coefficient）**：  
Spearman 秩相關係數（截面 pred_20d vs. alpha_20d），NaN-aware，至少需 5 個有效樣本。

---

## 11. 推論輸出欄位

### df_kelly.csv（全市場 Alpha 排名）

| 欄位 | 說明 |
|------|------|
| `Ticker` | 股票代碼 |
| `Date` | 推論日期 |
| `Exp_Alpha_5d` | MC 平均 5 日 Alpha 預測 |
| `Exp_Alpha_20d` | MC 平均 20 日 Alpha 預測 |
| `Exp_Alpha_60d` | MC 平均 60 日 Alpha 預測 |
| `Uncertainty_20d` | MC 標準差（不確定性） |
| `Net_Alpha_20d` | 扣除滑點後 Alpha（流動性不足 = -999） |
| `Signal_Quality` | Net_Alpha / Uncertainty，截斷至 [-10, 10] |
| `Confidence` | 高信心 / 中信心 / 低信心 |
| `Suggested_Weight` | 建議建倉比重（Signal_Quality 正值正規化） |

### df_traj.csv（多期預測軌跡）

| 欄位 | 說明 |
|------|------|
| `Pred_5d / 20d / 60d` | 各期 Alpha 預測（原始） |
| `Uncertainty_5d / 20d / 60d` | 各期 MC 標準差 |

---

## 12. 超參數速查表

| 超參數 | 變數名稱 | 值 |
|--------|----------|-----|
| 模型維度 | `D_MODEL` | 256 |
| Mamba 狀態維度 | `D_STATE` | 32 |
| GAT 注意力頭數 | `N_HEADS_GAT` | 4 |
| GAT 鄰居上限 | `MAX_NEIGHBORS_GAT` | 15 |
| Dropout | `DROPOUT` | 0.1 |
| 序列長度 | `SEQ_LEN` | 252 |
| 特徵維度 | `INPUT_DIM` | 56（V6.0 為 46） |
| 預測期數 | `PRED_HORIZONS` | [5, 20, 60] 天 |
| 學習率 | `LR` | 7e-5 |
| 最大 Epoch | `EPOCHS` | 60 |
| Early Stop | `EARLY_STOP` | 10 |
| Warmup 比例 | `WARMUP_PCT` | 0.15 |
| 梯度裁剪 | `GRAD_CLIP_NORM` | 1.0 |
| 混合精度 | `AMP_ENABLED` | True（FP16） |
| 批次大小 | `BATCH_SIZE` | 1 截面 / step |
| 損失 20d 權重 | `mse_20d` | 1.0 |
| 損失 5d 權重 | `mse_5d` | 0.3 |
| 損失 60d 權重 | `mse_60d` | 0.3 |
| 損失排名權重 | `listnet_20d` | 0.5 |
| MC-Dropout 採樣數 | `n_mc` | 30 |
| KG 相關性窗口 | `KG_CORR_WINDOW` | 60 天 |
| KG 相關性門檻 | `KG_CORR_THRESHOLD` | 0.7 |
| 驗證集比例 | `VAL_RATIO` | 0.15 |

---

*此文件由程式碼直接分析整理，反映 V6.1 實際實作。如有版本更新請同步修改本文件。*
