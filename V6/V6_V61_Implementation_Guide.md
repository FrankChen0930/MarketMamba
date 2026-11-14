# MarketMamba V6 / V6.1 — 技術實作文件

> **最後更新**：2026-05-12
> **涵蓋範圍**：V6.0 設計 → V6.1 實作 → Full Retrain 生產流程

---

## 一、版本演進總覽

MarketMamba 是一個基於 **Mamba SSM（State Space Model）+ GATv2（Graph Attention）** 的台股量化選股模型。V6 為重大架構升級，V6.1 在此基礎上擴展特徵與調參。

### 版本對照表

| 項目 | V5.5 | V6.0（設計） | V6.1（實作） |
|------|------|-------------|-------------|
| **輸入維度** | 84D（含 FinBERT 情緒） | 46D（純量化） | **56D（擴展純量化）** |
| **序列長度** | 180 天 | 252 天 | 252 天 |
| **Mamba 架構** | 6 層單一流 | 多尺度三路並行（8 層） | 多尺度三路並行（8 層） |
| **Embedding** | `nn.Linear(84, d_model)` | FactorGroupedEmbedding（等分） | FactorGroupedEmbedding（**比例分配**） |
| **知識圖譜邊** | 同產業靜態邊 | 供應鏈 + 滾動相關性（混合） | 供應鏈 + 滾動相關性（混合） |
| **預測目標** | 30 天 Alpha（單輸出） | 5d/20d/60d（多輸出） | 5d/20d/60d（多輸出） |
| **Loss 函數** | 純 MSE | MSE + ListNet | MSE + ListNet |
| **驗證指標** | Val MSE | IC / ICIR | IC / ICIR |
| **學習率** | 1e-4 | 1e-4 | **7e-5（降低）** |
| **Warmup** | 10% | 10% | **15%（延長）** |
| **訓練資料** | 2019-2024 | 2005 起 | 2005 起 |

---

## 二、模型架構

### 2-1. 整體流程

```
輸入: (N_stocks, 252, 56)          ← 一整年日線，56 維純量化特徵
         ↓
[FactorGroupedEmbedding]           → (N, 252, d_model=256)
         ↓
[MultiScaleMambaEncoder]
  ├─ Short (x[:, -20:])  → 2 層 Mamba → h_short  ← 短線動能/籌碼
  ├─ Mid   (x[:, -60:])  → 3 層 Mamba → h_mid    ← 中線趨勢
  └─ Long  (x[:, :])     → 3 層 Mamba → h_long   ← 年線週期
  → Adaptive Scale Fusion (learned softmax gate)
         ↓                          → (N, d_model)
[GATv2 GraphAttentionLayer]        → (N, d_model)   ← 供應鏈+相關性邊
         ↓
[Gating Fusion]                    → (N, d_model)
  gate = σ(W·[h_temporal ‖ h_graph])
  h = gate·h_temporal + (1-gate)·h_graph
         ↓
[MultiHorizonHead]
  ├─ head_5d  → pred_5d   (短線輔助)
  ├─ head_20d → pred_20d  (主輸出)
  └─ head_60d → pred_60d  (季線輔助)
         ↓
輸出: (N, 3)  →  [pred_5d, pred_20d, pred_60d]
```

### 2-2. 核心組件

#### FactorGroupedEmbedding

將 56 維特徵按因子類型分成 4 組，各組獨立投影後拼接。

**V6.0 設計**：等分 `d_model // 4 = 64` 給每組。
**V6.1 實作**：**比例分配** — 各組子空間大小與其特徵數成正比。

```
Group A (price_momentum):      12 dims → d_model × 12/56 ≈ 54
Group B (institutional_flow):  20 dims → d_model × 20/56 ≈ 92  ← 最大組
Group C (fundamentals):        12 dims → d_model × 12/56 ≈ 54
Group D (macro_environment):   12 dims → d_model × 12/56 ≈ 56 (含餘數)
Concat → LayerNorm → Dropout → (N, 252, 256)
```

> **設計理由**：資訊密度不同的因子組應獲得對應比例的表達空間。

#### MultiScaleMambaEncoder

三路並行 Mamba，捕捉不同時間尺度：

| 分支 | 輸入窗口 | Mamba 層數 | 捕捉目標 |
|------|---------|-----------|---------|
| Short | 最後 20 天 | 2 層 | 短線動能、籌碼異動 |
| Mid | 最後 60 天 | 3 層 | 季度趨勢、法人中期佈局 |
| Long | 全部 252 天 | 3 層 | 年度循環、基本面週期 |

融合方式：Adaptive Scale Fusion（學習型 softmax 門控加權）

#### MambaStack

每層使用 **Pre-Norm Residual**：`x = x + Mamba(LayerNorm(x))`
參數：`d_model=256, d_state=32, d_conv=4, expand=2`

#### GraphAttentionLayer (GATv2)

4 head attention，每股最多 15 個鄰居。安全機制：空邊時 identity。

#### MultiHorizonHead

三個獨立 `nn.Linear(d_model, 1)` 預測頭。

---

## 三、特徵工程（V6.0: 46D → V6.1: 56D）

### V6.1 新增的 10 個特徵

#### Group B — 籌碼流向（16 → **20 dims，+4**）

| 新增特徵 | 說明 |
|---------|------|
| Holdings_Large_Pct | 千張以上大戶持股占比 |
| Holdings_Large_Change | 大戶持股比例週變化 |
| Securities_Balance | 借券餘額 |
| Foreign_Holding_Pct | 外資累計持股比例 |

#### Group C — 基本面（10 → **12 dims，+2**）

| 新增特徵 | 說明 |
|---------|------|
| Dividend_Yield_Fwd | 預期股利殖利率 |
| Free_Cash_Flow | 自由現金流 |

#### Group D — 總經環境（8 → **12 dims，+5，-1**）

| 新增特徵 | 說明 |
|---------|------|
| Futures_OI_Foreign | 外資期貨未平倉淨口數 |
| Options_PC_Ratio | 選擇權 Put/Call 比 |
| Fear_Greed | CNN 恐懼貪婪指數 |
| Business_Signal | 台灣景氣燈號（取代 Market_Closed） |
| FED_Rate | 聯準會基準利率 |

---

## 四、知識圖譜

### 四種邊類型

| 邊類型 | 權重 | 性質 | 說明 |
|--------|------|------|------|
| TWSE Sector | 0.5 | 靜態 | 同產業連邊 |
| Conglomerate | 0.8 | 靜態 | 大型集團關係 |
| TPEX Supply Chain | 0.6 | 靜態 | TPEX 產業鏈 |
| Rolling Correlation | 動態 | 動態 | 60 天 ρ > 0.7 |

**V6 改進**：V5.5 只有 TWSE 靜態邊 + O(N²) Python loop → V6 使用 **scipy CSR 稀疏矩陣** O(1) 子圖提取（~1ms/batch）

---

## 五、訓練管線

### Loss 函數

```
Loss = 1.0 × MSE_20d + 0.3 × MSE_5d + 0.3 × MSE_60d + 0.5 × ListNet_20d
```

ListNet 使用 **mean 正規化**，避免 val/train 股票數不同導致的 loss 尺度偏差。

### 兩階段訓練

| 階段 | 訓練資料 | Epochs | Early Stop | 產出 |
|------|---------|--------|------------|------|
| Phase ① | Train: 2005-2023 / Val: 2024+ | 100 | patience=15 | 找到 best epoch=14 |
| Phase ② | **ALL data** | **固定 14** | **關閉** | **生產模型** |

Phase ② 使用 `N_SAMPLE_TRAIN=2000`（每 batch 抽 2000 股）防止 OOM。

---

## 六、選股信號系統

### 進場（2/4 觸發 BUY，市場謹慎時 3/4）

1. **Rank Stability** — Top 10 ≥2 天 OR Top 50 ≥3 天
2. **High Confidence** — MC-Dropout 不確定性 < 0.02
3. **Relative Low** — RSI(14) < 40 OR Price < MA(20)
4. **Institutional Buy** — 外資連續淨買超 ≥ 2 天

### 市場體制

- **Normal**（TWII > MA60）：2/4 門檻
- **Cautious**（TWII < MA60）：3/4 門檻

### 出場

- 排名連續 2 天跌出 Top 50
- 外資連續賣超 ≥ 3 天

---

## 七、部署架構

```
本機 (WSL2)                GitHub          Render            Vercel
┌──────────────────┐ push  ┌──────┐ hook  ┌──────────┐     ┌──────────┐
│ daily_inference   │ ───→  │ main │ ───→  │ Backend  │ ←── │ Frontend │
│ ├─ 資料更新       │       └──────┘       │ (FastAPI)│     │ (React)  │
│ ├─ 推論+Scanner   │                      └──────────┘     └──────────┘
│ └─ git push       │
└──────────────────┘
```

---

## 八、關鍵檔案索引

| 檔案 | 說明 |
|------|------|
| `V6/marketmamba/config.py` | 全域超參數、特徵定義 |
| `V6/marketmamba/models/architecture.py` | 模型架構 |
| `V6/marketmamba/models/trainer.py` | 訓練 + Loss |
| `V6/marketmamba/data/feature_engineer.py` | 56D 特徵工程 |
| `V6/marketmamba/knowledge/graph_builder.py` | 知識圖譜 |
| `V6/marketmamba/signals/scanner.py` | 選股信號 |
| `V6/run_daily_inference.py` | 每日推論腳本 |
| `V6/notebooks/V6_FullRetrain_Cell.py` | Full Retrain 腳本 |
