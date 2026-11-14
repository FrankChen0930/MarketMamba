# 🐍 MarketMamba V6 — 完整升級主計畫

> **狀態**：規劃中 (Planning)
> **最後更新**：2026-04-24
> **版本目標**：MarketMamba V6.0
> **文件性質**：本文整合 V6_Upgrade_Plan.md 與 V6_redesign_proposal.md，為唯一權威版本

---

## 一、V5.5 現況健檢

> 此段為完整代碼審查後的總結，供後續快速掌握現況。

### 1-1. 已完成模組盤點

| 模組 | 檔案 | 品質評估 |
|------|------|---------|
| 資料擷取 | `fetcher.py` (394行) | 優：帶重試、增量補齊、五大資料源 |
| 跨頻融合 | `merger.py` | 良：季報/月營收時間對齊有保護 |
| 特徵工程 | `feature_engineer.py` (165行) | 良：基礎技術指標、Rev_YoY 防未來函數 |
| 模型架構 | `architecture.py` (195行) | 優：Mamba×6層 + GATv2，Gating Fusion |
| 推論引擎 | `inference.py` (282行) | 優：自動偵測版本、checkpoint 反向工程 |
| 訓練模組 | `trainer.py` (431行) | 優：FastAlphaDataset、AMP FP16、滑窗步幅 |
| 知識圖譜 | `graph_builder.py` (99行) | ⚠️ 初版：只有同產業=0.5靜態規則 |
| 情緒引擎 | `sentiment/` (6個模組) | ⚠️ 將移除：改由 LLM 外包 |
| 型態偵測 | `detectors.py` (284行) | 優：6大型態、動態頸線、等幅測距 |
| 投組管理 | `portfolio_manager.py` (239行) | ⚠️ 待強化：缺流動性濾網、無滑價折抵 |
| 前端展示 | `app.py` (449行) | 良：5頁面 Streamlit，Plotly互動圖 |

### 1-2. 現行模型規格

```
輸入維度：84D（46 量價籌碼 + 38 情緒特徵）
時序框架：Mamba SSM × 6層，seq_len=180，d_model=256，d_state=32
圖推理：  GATv2Conv (4 head)，KG-Enhanced Edge（cosine + 產業分類）
預測目標：未來30天 Alpha 軌跡（相對 TWII 超額累積報酬）
訓練策略：MSELoss + AdamW + CosineAnnealingLR + EarlyStop(7) + AMP FP16
訓練資料：2019-2024（1,250 個截面）
```

### 1-3. 從代碼直接識別的現存問題

| 嚴重度 | 問題 | 說明 |
|--------|------|------|
| 🔴 邏輯錯誤 | **Cross-Section 污染** | `FastAlphaDataset` 滑窗取樣讓 batch 內混入不同時間步的股票，GAT 跨時間步計算 cosine similarity 在邏輯上是錯誤的 |
| 🔴 實盤脫節 | **流動性濾網不存在** | `compute_kelly_sharpe()` 完全沒有日均量或滑價，df_kelly 裡甚至沒有 Avg_Volume 欄位 |
| 🟡 Alpha 缺口 | **KG 過於粗糙** | `_build_knowledge_similarity()` 只有二元規則，且是 O(N²) Python loop |
| 🟡 特徵缺口 | **RSI、KD、ATR 完全沒有** | 台股最重要的三個即時動能指標在 46 維特徵中缺席 |
| 🟡 資料量不足 | **訓練起始點 2019** | 實際可從 2012 起（FinMind 籌碼早在 2012 就有），資料量少了 2.4 倍 |
| 🟡 評估指標錯誤 | **只看 MSE** | 量化業界標準是 IC / ICIR，MSE 低不代表選股排名好 |
| 🟡 時效瓶頸 | **依賴 FinMind，推論被迫晚上才能跑** | FinMind 融資融券更新最晚 19:00，且每天要重開 Colab |

---

## 二、V6 設計原則

1. **MarketMamba 回歸純量化角色**：移除 FinBERT，input_dim 從 84D → 46D，消息面交由 LLM 外包
2. **Mamba 骨幹不動**：繼續用 Mamba SSM + GATv2，但在架構上深化（多尺度、因子分組）
3. **財金理論先行**：訓練目標從「最小化 MSE」轉向「最大化 IC / ICIR」
4. **Walk-Forward 是唯一誠實的考試**：沒有 Walk-Forward，模型績效無法相信
5. **推論本機化**：每日推論遷移至本機 RTX 3060 + WSL2，省成本、省時間、可自動化

---

## 三、V6 核心架構重設計

### 3-1. 整體模型結構

```
輸入: (N_stocks, 252, 46)   ← 一整年日線，純量價籌碼，無情緒特徵
         ↓
[FactorGroupedEmbedding]     → (N, 252, d_model)
         ↓
[MultiScaleMambaEncoder]
  ├─ Short Branch: x[:, -20:, :]   → 2層Mamba → h_short (N, d_model)  ← 短線動能
  ├─ Mid   Branch: x[:, -60:, :]   → 3層Mamba → h_mid   (N, d_model)  ← 中線趨勢
  └─ Long  Branch: x[:, :,    :]   → 3層Mamba → h_long  (N, d_model)  ← 年線週期
  → Adaptive Scale Fusion → (N, d_model)
         ↓
[GATv2 + Hybrid Edges]       → (N, d_model)   ← 供應鏈邊 + 滾動相關性邊
         ↓
[Gating Fusion]              → (N, d_model)
         ↓
[MultiHorizonHead]
  ├─ pred_5d  (N, 1)   ← 1週短線
  ├─ pred_20d (N, 1)   ← 月線（主輸出）
  └─ pred_60d (N, 1)   ← 季線輔助
         ↓
Loss = MSE_20d + 0.3*(MSE_5d + MSE_60d) + 0.5*ListNet_20d
```

**與 V5.5 的差異對照**：

| 元件 | V5.5 | V6 |
|------|------|-----|
| 輸入維度 | 84D（含情緒） | **46D（純量化）** |
| 序列長度 | seq_len=180 | **seq_len=252（一整年）** |
| Mamba 架構 | 6層單一流 | **多尺度三路並行（8層總量）** |
| Embedding | nn.Linear(84, d_model) | **FactorGroupedEmbedding（按因子類型分組）** |
| 圖邊 | TWSE產業分類靜態邊 | **供應鏈邊 + 滾動相關性邊（混合動靜態）** |
| 預測目標 | 30天 Alpha MSE | **5d/20d/60d 多輸出頭（主目標20d）** |
| Loss 函數 | 純 MSELoss | **MSE + ListNet Ranking Loss** |
| 驗證指標 | Val MSE | **IC（截面相關係數）+ ICIR** |

---

### 3-2. 因子分組 Embedding（FactorGroupedEmbedding）

**現在**：46個特徵直接進 `nn.Linear(46, d_model)`，不同性質的因子互相污染。

**V6**：按量化因子的自然分組，各類先在子空間學習後融合。

```python
class FactorGroupedEmbedding(nn.Module):
    """
    將 46 維特徵按因子類別分組，各組先投影再融合

    Group A - 價量動能 (12維)：OHLCV + 各種報酬率 + MA
    Group B - 籌碼流向 (16維)：三大法人、融資融券、借券、當沖
    Group C - 基本面   (10維)：月營收、EPS、PER、PBR、毛利率
    Group D - 總經環境  (8維)：TWII、美股、VIX、TNX、黃金、油
    """
    def __init__(self, group_dims: dict, d_model: int):
        super().__init__()
        sub_dim = d_model // 4
        self.proj_A = nn.Linear(group_dims['price_momentum'], sub_dim)
        self.proj_B = nn.Linear(group_dims['institutional_flow'], sub_dim)
        self.proj_C = nn.Linear(group_dims['fundamentals'], sub_dim)
        self.proj_D = nn.Linear(group_dims['macro'], sub_dim)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (N, T, 46)，按 group 切分後各自投影再 concat
        hA = self.proj_A(x[..., :12])
        hB = self.proj_B(x[..., 12:28])
        hC = self.proj_C(x[..., 28:38])
        hD = self.proj_D(x[..., 38:46])
        return self.fusion_norm(torch.cat([hA, hB, hC, hD], dim=-1))
```

---

### 3-3. 多尺度 Mamba（MultiScaleMambaEncoder）

台股的 Alpha 信號存在於不同時間尺度，單一 seq_len=180 無法同時捕捉：

| 時間尺度 | 對應信號 | Branch |
|---------|---------|--------|
| 20天（1個月） | 短線動能、籌碼面 | Short（2層 Mamba） |
| 60天（1季） | 季線趨勢、外資中期佈局 | Mid（3層 Mamba） |
| 252天（1年） | 基本面週期、獲利動能 | Long（3層 Mamba） |

```python
class MultiScaleMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state):
        self.mamba_short = MambaStack(layers=2)   # 接受 (N, 20,  d_model)
        self.mamba_mid   = MambaStack(layers=3)   # 接受 (N, 60,  d_model)
        self.mamba_long  = MambaStack(layers=3)   # 接受 (N, 252, d_model)
        # 可學習的尺度融合注意力
        self.scale_attn  = nn.Linear(d_model * 3, 3)

    def forward(self, x):
        # x: (N, 252, d_model)
        h_short = self.mamba_short(x[:, -20:, :])[:, -1, :]   # 最後一步 hidden
        h_mid   = self.mamba_mid(x[:, -60:, :])[:, -1, :]
        h_long  = self.mamba_long(x)[:, -1, :]

        # Adaptive Fusion
        cat_h  = torch.cat([h_short, h_mid, h_long], dim=-1)
        scales = torch.softmax(self.scale_attn(cat_h), dim=-1)  # (N, 3)
        output = (scales[:, 0:1] * h_short +
                  scales[:, 1:2] * h_mid   +
                  scales[:, 2:3] * h_long)
        return output  # (N, d_model)
```

---

### 3-4. Hybrid 圖邊（動靜態混合）

| 邊的類型 | 更新頻率 | 資料來源 | 意義 |
|---------|---------|---------|------|
| **TPEX 供應鏈邊** | 每季 | TPEX 爬蟲（ic.tpex.org.tw） | 結構性關係，慢變，權重 0.6 |
| **集團從屬邊** | 手動更新 | 自建表（鴻海族、台積電生態等） | 高關聯，權重 0.8 |
| **滾動相關性邊** | 每週 | 60天滾動 Pearson 相關 > 0.7 | 市場動態關係，快變 |
| **TWSE 產業分類** | 靜態 | 現有資料 | 基底，權重 0.5 |

```python
# 兩種主要邊的加權融合（alpha 是可學習參數）
alpha       = torch.sigmoid(self.edge_gate)   # 0~1
edge_weight = alpha * kg_edge_weight + (1 - alpha) * corr_edge_weight
```

**實作位置**：改寫 `marketmamba/knowledge/graph_builder.py`
**快取位置**：`config.PROCESSED_DIR/knowledge_graph_cache.npz`（O(1) 讀取，不影響訓練速度）

---

### 3-5. 多輸出頭與 Loss 函數

```python
class MultiHorizonHead(nn.Module):
    def __init__(self, d_model):
        self.head_5d  = nn.Linear(d_model, 1)   # 短線動能（1週）
        self.head_20d = nn.Linear(d_model, 1)   # 月線（主輸出）
        self.head_60d = nn.Linear(d_model, 1)   # 季線（輔助約束）

# Loss 組合
Loss = MSE_20d + 0.3 * MSE_5d + 0.3 * MSE_60d + 0.5 * ListNet_20d
```

**為什麼改 20d？**
```
30天 → 20天（一個交易月）的理由：
- 20天正好是一個日曆月的交易日數，有自然的週期性
- 財金文獻短期動能效應最強處在 1-3 個月
- 30天比20天多 50% 噪音，不值得
- Walk-Forward 每期 6 個月，可以累積更多評估樣本
```

**為什麼加 ListNet？**
```
現在 MSE 要求模型預測「精確的數值」，但選股真正關心的是：
  「哪支股票會贏，哪支股票會輸（排名）」，而不是「贏了幾個百分點」
ListNet Ranking Loss 直接優化選股排名，比 Pairwise Loss 更穩健，
完全連續可微，不使用任何 Heaviside/階躍函數。
```

---

### 3-6. 訓練穩定性機制

| 風險 | 原因 | V6 對應 |
|------|------|---------|
| **梯度爆炸** | 多尺度 Mamba 堆疊 | Gradient Clipping（max_norm=1.0，已有） |
| **GAT 注意力崩潰** | 邊數量過多 | 限制每股最多 K=15 個鄰居 |
| **早期訓練不穩** | Adam lr 初期步幅過大 | Warmup + CosineAnnealing（OneCycleLR） |
| **多輸出頭衝突** | 不同 horizon 梯度方向不一致 | GradNorm 動態平衡各輸出頭梯度 |

```python
# Warmup + CosineAnnealing 組合（比純 CosineAnnealing 更穩定）
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_training_steps,
    pct_start=0.1,        # 前 10% 做 warmup
    anneal_strategy='cos'
)
```

---

## 四、預測目標與評估指標重設計

### 4-1. 為什麼 MSE 不夠

```
V5.5 現在優化：min MSE(pred_30d_alpha, actual_30d_alpha)
問題：
  ① 30天絕對 Alpha 噪音極大，模型學到的是噪音，不是信號
  ② MSE 把「預測精確度」和「選股排名能力」混在一起
  ③ TWII 相對報酬未做 Fama-French 因子剝離（有 Size/Value 偏差）
```

### 4-2. V6 的正確評估框架

| 指標 | V5.5 | V6 目標 |
|------|------|---------|
| 訓練 Loss | MSE | **MSE_20d + ListNet_20d** |
| 驗證指標 | Val MSE 降就好 | **IC（每日截面 Spearman 相關係數）** |
| 穩定性指標 | 無 | **ICIR = IC_mean / IC_std（越高越穩定）** |
| 選股基準 | 相對 TWII Alpha | **Fama-French 4因子調整後超額報酬** |

**業界通過標準（量化基金參考）**：
- `IC_mean > 0.05`：信號有效
- `ICIR > 0.5`：信號穩定，可實盤
- `Top10 等權報酬 > TWII × 1.2`：實際跑贏大盤

### 4-3. 台股 Fama-French 因子的資料來源

美股 FF 因子可直接從 [Ken French 網站](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) 免費下載，但**台股需要自己建構**：

```
台股 FF4 因子計算方式（每月重新計算）：

  Market Factor (Mkt-RF)：TWII 月報酬 - 台灣無風險利率（1年期定存利率）
  Size Factor (SMB)      ：市值最小30% 股票月報酬 - 市值最大30% 股票月報酬
  Value Factor (HML)     ：B/M最高30% 月報酬 - B/M最低30% 月報酬
  Momentum Factor (MOM)  ：前12個月報酬最高30% - 最低30%（排除最近1個月）

資料來源：
  - 個股市值、PBR → FinMind TaiwanStockInfo（每月更新）
  - 個股月報酬    → 自行從日線資料計算
  - 無風險利率    → 中央銀行統計資料（cbportal.cbc.gov.tw）
```

**實作位置**：新增 `marketmamba/evaluation/factor_model.py`

> **實務簡化**：V6 初期可先用「相對 TWII 的超額報酬」作為 Alpha 目標（現有邏輯），
> 等 Walk-Forward 框架穩定後再補上 FF4 因子剝離。兩個指標都輸出，逐步過渡。

---

## 五、Walk-Forward Validation 框架

### 5-1. 為什麼這是最重要的改變

> 沒有 Walk-Forward，你只是在測試模型「有沒有記住歷史」。
> Walk-Forward 是量化策略的**唯一誠實考試**。

任何 train/test 單次切割都有「選時偏差」——你不知道模型的好結果是真本事還是恰好選到好測試期。Walk-Forward 把這個問題消除：用多個不同歷史時期驗證模型，綜合表現才算數。

### 5-2. 窗口類型選擇：Expanding Window（已確認採用）

有兩種 Walk-Forward 模式，V6 採用 **Expanding Window**：

| 模式 | 說明 | 適合情境 |
|------|------|--------|
| **Expanding Window** ✅ | 起點固定在 2012，訓練集隨時間增長 | **資料量有限（台股 ~3,000 截面）、希望用盡所有歷史** |
| Rolling Window | 固定訓練窗口長度（如5年），視窗滾動 | 擔心遠古資料的市場結構與現在差太遠（美股 30 年資料適用）|

**選 Expanding Window 的理由**：台股 2012-2024 只有 3,000 個截面，Rolling Window 每次只用 1,250 個截面訓練，浪費了好不容易擴充的歷史資料。Expanding Window 讓每個 Fold 的訓練集越來越豐富，更符合我們的資料量現況。

### 5-3. V6 Walk-Forward 設計

```
參數設定（Expanding Window）：
  起始點（固定）           ：2012-01-01
  測試窗口（Test Window） ：6個月
  滾動步幅（Step）        ：3個月
  最小訓練期              ：3年（確保模型有足夠歷史才開始測試）

時間軸：
  Fold 1:  Train [2012-2015]      → Test [2015 H1]  ← 3年訓練才開始測試
  Fold 2:  Train [2012-2015 H2]   → Test [2015 H2]
  Fold 3:  Train [2012-2016]      → Test [2016 H1]
  ...
  Fold N:  Train [2012-2023 H2]   → Test [2024 H1]

共約 36 個 Fold（9年測試期 × 4個 Fold/年）
```

### 5-4. 每個 Fold 的輸出指標

```python
@dataclass
class WalkForwardResult:
    fold_id:       int
    test_period:   str
    train_end:     str     # 每個 Fold 訓練集的結束日期（Expanding Window 會增長）
    ic_mean:       float   # 平均 IC（每日截面 Spearman 相關係數）
    ic_std:        float   # IC 標準差
    ic_ir:         float   # ICIR = IC_mean / IC_std
    top10_return:  float   # Top 10 等權重報酬 vs 大盤
    max_drawdown:  float   # 最大回撤
    win_rate:      float   # 正 IC 日的比例（> 50% 才有信號穩定性）
```

### 5-5. 實作位置

新增模組：`marketmamba/evaluation/walk_forward.py`

**Phase 0 策略**：先用 V5.5 模型跑一輪 Walk-Forward，建立 baseline；之後 V6 模型只需和 baseline 比較 ICIR 就能量化提升幅度。

---

## 六、資料擴充計畫（2012 → 2024）

### 6-1. 為什麼要從 2019 擴充到 2012

| | V5.5 | V6 |
|--|------|-----|
| 訓練起始 | 2019 | **2012** |
| 截面總數 | ~1,250 | **~3,000（+2.4倍）** |
| Walk-Forward Fold 數 | ~4 | **~24** |
| 涵蓋市場環境 | 2019熊市尾+牛市 | **+2012~2018（多個完整牛熊週期）** |

### 6-2. 為什麼是 2012 而不是更早

```
FinMind 各資料集可追溯時間：
  個股日線價量 (TaiwanStockPrice)               : ~1994
  三大法人買賣超 (TaiwanStockInstitutionalInvestors) : ~2012  ← 決定性因素
  融資融券 (TaiwanStockMarginPurchaseShortSale)   : ~2005
  月營收 (TaiwanStockMonthRevenue)               : ~2010

結論：2012-01-01 是籌碼特徵（Group B，16維）完整存在的最早起點。
```

### 6-3. 擴充步驟

1. 修改 `config.py`：`DATA_START_DATE = "2012-01-01"`
2. 重跑 `fetcher.py` 抓取 2012-2018 歷史資料（FinMind 批量拉取）
3. 重跑 `feature_engineer.py` 補齊特徵
4. 驗證 2012-2018 資料的缺值比例（早期資料某些欄位可能較稀疏）

---

## 七、特徵工程深化（46D 重新整理）

### 7-1. 移除情緒特徵，新增技術指標

**V6 的 input_dim = 46D（純量化）**：

移除 FinBERT 的 38D 情緒特徵，同時補入以下缺席的技術指標：

| 指標 | 說明 | 為什麼重要 |
|------|------|----------|
| **RSI (14日)** | 相對強弱指數 | 台股教科書指標，散戶在 RSI<30 有強烈買入慣性 |
| **KD 隨機指標 (9/3/3)** | K值、D值 | 台股量化因子中的高頻 Alpha 生產來源 |
| **ATR (14日)** | 真實波動幅度 | 比 Volatility_20d 更精確，可取代 Kelly 裡的 raw_vol |
| **OBV** | 累積量價關係 | 對中長線資金動向有更好描述 |
| **EPS 驚喜度** | 與上季 EPS 比較成長率 | 重要的基本面衝量因子 |

**實作位置**：`marketmamba/data/feature_engineer.py`

### 7-2. V6 特徵分組（完整 46D 定義）

```python
V6_FEATURE_GROUPS = {
    "price_momentum": [        # Group A，12維
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_1d', 'Return_5d', 'Return_20d',
        'MA_20', 'MA_60', 'RSI_14', 'ATR_14',
    ],
    "institutional_flow": [    # Group B，16維
        'Foreign_Buy', 'Foreign_Sell', 'Foreign_Net',
        'Investment_Trust_Net', 'Dealer_Net',
        'Margin_Purchase', 'Margin_Repay', 'Short_Sale', 'Short_Cover',
        'Margin_Balance', 'Short_Balance',
        'Day_Trade_Volume', 'KD_K', 'KD_D', 'OBV', 'Volatility_20d',
    ],
    "fundamentals": [          # Group C，10維
        'PER', 'PBR', 'Revenue_MoM', 'Revenue_YoY',
        'EPS', 'EPS_Surprise', 'Gross_Margin', 'ROE',
        'Market_Cap_Log', 'Book_Value',
    ],
    "macro_environment": [     # Group D，8維
        'TWII_Return', 'SPX_Return', 'VIX', 'TNX',
        'Gold_Return', 'Oil_Return', 'USD_TWD', 'Market_Closed',
    ],
}
```

---

## 八、Cross-Section DataLoader 修正

### 8-1. 問題根源

`FastAlphaDataset.__getitem__` 每次取一個隨機窗口，batch 內不同股票可能屬於不同時間步。GAT 在這種情況下計算的 cosine similarity 是**跨時間污染的**，圖訊息傳遞在邏輯上無效。

### 8-2. 正確修法：TemporalCrossSectionDataset

```python
class TemporalCrossSectionDataset(Dataset):
    """
    每個 sample = 某交易日 t 的全市場截面 (N_stocks, seq_len, input_dim)
    保證 GAT 的 cross-sectional 計算邏輯正確
    """
    def __init__(self, dates: list, date_to_tensors: dict):
        self.dates = dates
        self.date_to_tensors = date_to_tensors

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        X, Y = self.date_to_tensors[date]   # (N, seq_len, D), (N, 3)  ← 3個horizon
        return X, Y
```

**注意事項**：
- DataLoader `batch_size=1`（每個 sample 已是全市場截面）
- GPU 顯存需求提高，若 2000 × 252 × 256 超過 A100 上限，可每截面隨機抽樣 N_sample 支股票訓練（推論時仍用完整截面）
- 這是 V6 必須完成的重訓前提條件，不可跳過

**DataLoader 記憶體優化設定**：
```python
# 每個截面 ~500MB（2000 × 252 × 256 × float32），A100 40GB 放得下
# 但多個 prefetch 同時存在時可能 OOM，需控制預取數量
train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,        # 最多預取 2 個截面，不要設太高
    persistent_workers=True,  # 避免每 epoch 重建 worker
)
```
若 2000 支股票仍 OOM，可在 Dataset 內加入截面抽樣機制：
```python
# 訓練時隨機抽 N_sample 支，保留排名多樣性
N_SAMPLE_TRAIN = 800   # 抽 800 支股票訓練，推論仍用全部
if self.mode == 'train' and N_SAMPLE_TRAIN < len(stock_idx):
    stock_idx = np.random.choice(stock_idx, N_SAMPLE_TRAIN, replace=False)
```

**實作位置**：`marketmamba/models/trainer.py`，取代 `FastAlphaDataset`

---

## 九、實盤流動性防護網

### 9-1. 絕對流動性濾網（Hard Filter）

```python
# inference.py / compute_kelly_sharpe() 新增
# 近 5 日平均成交金額低於 1000 萬台幣 → 直接剔除
zombie_vol_mask = df['Avg_Volume_5D'] < 1e7
sharpe[zombie_vol_mask] = -999.0
```

### 9-2. 滑價與摩擦力折抵（Slippage Penalty）

```
Net Alpha = Model Expected Return - (Tax + Fee + Liquidity Slippage)
```
- 滑價依市值反比計算：大型股 ~0.2%，小型股 ~0.8%
- 只有 Net Alpha 仍具備爆發力的股票才進入資金分配

**注意**：`Avg_Volume_5D` 欄位需要在特徵工程或推論時補齊傳入。

---

## 十、部署策略：本機推論 + 雲端訓練分離

### 10-1. 架構決策

> **每日推論 → 本機 RTX 3060 + WSL2**
> **重新訓練 → Colab A100**（僅大版本升級時使用）

### 10-2. 為什麼本機比 Colab 更適合做每日推論

| 面向 | Colab | 本機 RTX 3060 |
|------|-------|--------------|
| 啟動成本 | 每次重裝 Mamba kernel ~5min | **一次安裝，永久有效** |
| 推論時間（2000股） | ~3-8 分鐘（含啟動） | **~5-12 分鐘（直接執行）** |
| 每日成本 | GPU quota 消耗 | **電費 < NT$0.3** |
| 自動化 | 需手動開啟 Colab | **Windows Task Scheduler 全自動** |
| 穩定性 | Session 可能中斷 | **本機穩定** |

### 10-3. WSL2 安裝 Mamba 環境（一次性設定）

```bash
# 1. 安裝 CUDA Toolkit（WSL2 Ubuntu）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# 2. 建立 MarketMamba 環境
conda create -n mamba_env python=3.12
conda activate mamba_env
pip install -r requirements.txt
pip install causal_conv1d-*.whl mamba_ssm-*.whl   # 用已有的 wheel

# 3. 一次編譯，永久使用，不需每天重裝
```

### 10-4. Windows Task Scheduler 自動化

```batch
:: daily_inference.bat
@echo off
wsl -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python run_daily_inference.py"
```

設定每天 **17:00** 觸發（見模組十一的資料時效優化）。

### 10-5. 訓練環境：Colab A100（維持現況）

```bash
# 本機：打包資料上傳
cd Data && tar czf data_snapshot.tar.gz processed/
rclone copy data_snapshot.tar.gz gdrive:MarketMamba/

# Colab：掛 Drive 解壓後直接訓練
# 訓練完：checkpoint 下載回本機 models/ 目錄
```

### 10-6. 已確認不可行的方案

| 方案 | 結論 | 原因 |
|------|------|------|
| GitHub Actions | ❌ | 只有 CPU，無法跑 Mamba CUDA kernel |
| Kaggle 排程 | ❌ | CUDA kernel 無法跨 session 保留（已確認） |
| Render/Railway CPU | ❌ | Mamba 必須有 CUDA |

---

## 十一、LLM 整合層（取代 FinBERT）

### 11-1. 架構決策

> MarketMamba 回歸**純量化角色**，消息面分析交由 LLM（Claude/GPT）負責。

| 面向 | FinBERT 整合進模型 | LLM 外包分析 |
|------|-----------------|------------|
| 新聞理解品質 | 有限（POS/NEG/NEU） | 遠優（可理解語境） |
| 模型複雜度 | input_dim 84D | **降回 46D（更乾淨）** |
| 升級難度 | 需重訓模型 | 換 API 即可 |
| 訓練穩定性 | 情緒特徵雜訊多 | 完全隔離 |

### 11-2. 每日推論 Pipeline

```
[步驟 1] MarketMamba 量化推論（本機 RTX 3060，~10分鐘）
  → 輸出：df_kelly.csv（Alpha、Kelly、Sharpe 等）
  → 輸出：df_traj.csv（多horizon Alpha軌跡）

[步驟 2] LLM 報告生成（本機 CPU，API call，~10秒）
  → 輸入：① df_kelly Top 10 結果
           ② 今日市場數據（TWII漲跌、VIX、美股）
           ③ 恐懼貪婪指數（API）
  → 輸出：market_summary.json（市場背景 + 情緒分析）

[步驟 3] 合併推送 GitHub
  → df_kelly.csv + df_traj.csv + market_summary.json → 前端更新
```

### 11-3. 實作

新增模組：`marketmamba/llm/report_generator.py`

```python
import anthropic

def generate_market_report(df_kelly_top10: dict, market_data: dict) -> dict:
    """推論完後呼叫，生成當日市場背景報告。不需要 Agent，單次 API call。"""
    client = anthropic.Anthropic()

    prompt = f"""
    今天是 {market_data['date']}，台股大盤收 {market_data['twii_change']:+.2f}%。
    以下是今日量化模型的 Top 選股結果：
    {df_kelly_top10}

    請生成一份200字以內的繁體中文市場背景摘要，包含：
    1. 今日市場整體氛圍（一句話）
    2. 需要特別注意的宏觀風險（如有）
    3. 配合量化信號的操作建議方向

    格式：條列式，emoji 標記重要程度。
    """

    response = client.messages.create(
        model="claude-haiku-4-5",   # 成本最低，每次 < $0.001
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"date": market_data['date'], "summary": response.content[0].text}
```

### 11-4. 費用估算

| LLM 選擇 | 每次呼叫成本 | 每月（30天） |
|---------|------------|------------|
| **Claude Haiku（推薦）** | ~$0.001 | **~$0.03/月** |
| Claude Sonnet | ~$0.005 | ~$0.15/月 |
| GPT-4o mini | ~$0.002 | ~$0.06/月 |

---

## 十二、資料時效優化（解決 FinMind 延遲瓶頸）

### 12-1. 問題：FinMind 讓推論被迫等到晚上

| 資料集 | FinMind 更新時間 |
|--------|----------------|
| 個股價量 (TaiwanStockPrice) | ~17:00-18:00 |
| 三大法人 (TaiwanStockInstitutionalInvestors) | ~17:30-18:30 |
| 融資融券 (TaiwanStockMarginPurchaseShortSale) | ~18:00-19:00 |

> 現況：最快也要等到 **18:30+** 才能安全執行推論。

### 12-2. 解決方案：Hybrid 分層資料源策略

```
[層 1] 立即可用（收盤後 15 分鐘內）
  來源：Yahoo Finance (yfinance)
  資料：個股 OHLCV 價量（取代 FinMind TaiwanStockPrice）

[層 2] 官方直連（約 16:30-17:00）
  來源：證交所公開資料平台 (data.twse.com.tw)
  API:  https://www.twse.com.tw/rwd/zh/fund/T86?date={YYYYMMDD}&response=json
  資料：三大法人買賣超（比 FinMind 快 30-60 分鐘）

[層 3] FinMind（保留為備援，約 17:30-19:00）
  資料：融資融券、借券、八大行庫
  策略：若 FinMind 尚未更新，使用前一天的值 Forward Fill
```

### 12-3. 關鍵判斷：融資融券 Forward Fill 合理嗎？

> ✅ **合理**。融資融券餘額每天變動極小（隔天不會突然暴增暴減）。
> Forward Fill 引入的預測誤差，遠小於「等到 19:00 才推論」帶來的時效延遲問題。
> 在極端行情（千股跌停等）時，融資餘額會快速變化，此時 FinMind 若能及時更新則優先使用。

### 12-4. 代碼示意

```python
# fetcher.py 新增（個股價量優先 yfinance）
def fetch_daily_prices_fast(target_date: str) -> None:
    all_tickers = [f"{stk}.TW" for stk in tw_universe] + \
                  [f"{stk}.TWO" for stk in otc_universe]
    df_batch = yf.download(all_tickers, start=target_date, end=target_date,
                           auto_adjust=True, progress=False, group_by='ticker')
    # 抓不到的股票才 fallback 到 FinMind
    missing = [t for t in all_tickers if df_batch[t].dropna().empty]
    if missing:
        _fetch_missing_from_finmind(missing, target_date)

# fetcher.py 新增（三大法人直連 TWSE）
def fetch_institutional_from_twse(date_str: str) -> pd.DataFrame:
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str.replace('-', ''), "response": "json"}
    res = requests.get(url, params=params,
                       headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = res.json()
    if data.get('stat') != 'OK':
        return None   # fallback 到 FinMind
    # 解析 data['data'] ...

# fetcher.py（融資融券 Forward Fill fallback）
def fetch_margin_with_fallback(date_str: str) -> pd.DataFrame:
    df = _fetch_finmind(session, "TaiwanStockMarginPurchaseShortSale", date_str)
    if df is None:
        df = load_yesterday_margin_cache()
        logger.warning(f"融資融券使用 Forward Fill: {date_str}")
    return df
```

### 12-5. 優化後的時間軸

```
舊流程（全部 FinMind）：
  13:30 收盤 → 等 FinMind → 18:30+ 推論 → 19:00 完成

新流程（Hybrid）：  
  13:30 收盤
  13:45 yfinance 抓價量完成
  16:30 TWSE 三大法人直連
  17:00 ✅ 開始推論（比舊流程早 1.5 小時）
  17:15 推論完成，LLM 報告生成
  17:30 推送 GitHub，前端更新

  融資融券：Forward Fill，接受此近似
```

---

## 十三、MC-Dropout 不確定性估計

### 13-1. 實作方式

```python
def run_inference_with_uncertainty(model, test_x, n_samples=30):
    model.train()   # 保持 Dropout 活化
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(test_x).cpu().numpy())
    pred_mean = np.mean(preds, axis=0)   # 最終預測
    pred_std  = np.std(preds, axis=0)    # 模型置信度
    return pred_mean, pred_std
```

### 13-2. 效益

- 在前端標示「高信心 / 低信心」選股
- 低置信度標的提醒用戶謹慎
- **不需要修改模型架構**，只修改 `inference.py`

---

## 十四、回測引擎與歷史存檔

### 14-1. 必要性

沒有回測引擎：
- 無法知道模型在不同市場環境（牛市/熊市/盤整）的勝率
- 無法量化 V6 vs V5.5 的真實效益差距
- 無法知道「情緒特徵」到底帶來了多少 Alpha

### 14-2. 規劃

新增目錄：`marketmamba/backtest/`

```python
class BacktestEngine:
    def run(self, df_kelly_history, df_price_history) -> BacktestResult:
        # 輸出：累計報酬、最大回撤、夏普比率、勝率、年化Alpha
        # 比較：Top10等權重策略 vs 凱利加權策略 vs TWII指數
```

### 14-3. GitHub 歷史存檔策略

```
問題：每天覆蓋 df_kelly.csv，歷史資料消失。

解法：Rolling Window 保留
  - 每天推送時，同時存到 results/YYYY-MM-DD/df_kelly.csv
  - 保留最近 90 天，自動刪除舊的
  - 容量估算：1MB/天 × 90天 = ~90MB，遠低於 GitHub 1GB 建議上限
```

---

## 十五、V6 目錄結構

V6 開發的標準目錄樹（`V6/` 資料夾下）：

```
MarketMamba/
├── V6/                              ← V6 主開發區
│   ├── V6_Master_Plan.md            ← 本文件（唯一規劃權威）
│   └── marketmamba/                 ← V6 核心套件（從 V5.5 改寫）
│       ├── __init__.py
│       ├── config.py                ← V6 全域設定（見第十六節）
│       ├── data/
│       │   ├── fetcher.py           ← Hybrid 資料源（yfinance + TWSE 直連）
│       │   ├── merger.py
│       │   ├── feature_engineer.py  ← 新增 RSI/KD/ATR/OBV
│       │   └── cleaner.py
│       ├── models/
│       │   ├── architecture.py      ← MultiScaleMamba + FactorGroupedEmbedding
│       │   ├── trainer.py           ← TemporalCrossSectionDataset + GradNorm
│       │   └── inference.py         ← MC-Dropout + 多horizon輸出
│       ├── knowledge/
│       │   ├── graph_builder.py     ← Hybrid 圖邊（供應鏈 + 滾動相關性）
│       │   └── tpex_crawler.py      ← [NEW] TPEX 供應鏈爬蟲
│       ├── evaluation/              ← [NEW] 評估模組
│       │   ├── walk_forward.py      ← Walk-Forward Validation 框架
│       │   ├── factor_model.py      ← 台股 FF4 因子計算
│       │   └── metrics.py           ← IC / ICIR 計算工具
│       ├── llm/                     ← [NEW] LLM 整合層
│       │   └── report_generator.py  ← Claude/GPT API 報告生成
│       ├── backtest/                ← [NEW] 回測引擎
│       │   └── engine.py
│       ├── pattern/
│       │   └── detectors.py         ← 沿用 V5.5
│       ├── robot/
│       │   └── portfolio_manager.py ← 新增流動性濾網 + 滑價折抵
│       └── deploy/
│           └── publisher.py
├── notebooks/
│   ├── V6_Training.py               ← [NEW] V6 Colab 訓練腳本
│   └── V6_Pipeline.py               ← [NEW] V6 本機推論腳本
├── scripts/
│   ├── daily_inference.bat          ← [NEW] Windows Task Scheduler 觸發腳本
│   └── run_daily_inference.py       ← [NEW] 推論主程式（WSL2 執行）
└── archive/
    └── V5_5/                        ← V5.5 封存（不刪，保留可運行狀態）
```

**新增目錄說明**：
- `evaluation/`：Walk-Forward、IC/ICIR 計算，和模型架構完全解耦
- `llm/`：LLM 呼叫層，未來換 API 只需改這個目錄
- `backtest/`：歷史回測，和推論引擎完全分離

---

## 十六、V6 config.py 修改清單

> V6 的設定改動比較多，統一列在這裡，避免之後實作時遺漏。

```python
# ============================================================
# V6 config.py 主要修改項目
# ============================================================

# --- 資料設定 ---
DATA_START_DATE = "2012-01-01"   # V5.5 是 2019，V6 擴充到 2012
SEQ_LEN = 252                    # V5.5 是 180，V6 改為一整年
PRED_HORIZONS = [5, 20, 60]      # V5.5 是 [30]，V6 多輸出頭
PRED_MAIN_HORIZON = 20           # 主要評估 horizon

# --- 特徵設定 ---
INPUT_DIM = 46                   # V5.5 是 84（含情緒），V6 降回純量化
FEATURE_GROUPS = {               # 因子分組（對應 FactorGroupedEmbedding）
    'price_momentum':   12,
    'institutional_flow': 16,
    'fundamentals':     10,
    'macro_environment': 8,
}

# --- 模型超參數 ---
D_MODEL = 256                    # 維持不變
D_STATE = 32                     # 維持不變
N_HEADS_GAT = 4                  # 維持不變
MAX_NEIGHBORS_GAT = 15           # [NEW] GAT 鄰居數上限，防注意力崩潰
MULTI_SCALE_LAYERS = [2, 3, 3]  # [NEW] Short/Mid/Long 各自的 Mamba 層數

# --- 訓練設定 ---
LR = 1e-4
LOSS_WEIGHTS = {                 # [NEW] 多輸出頭 Loss 權重
    'mse_20d': 1.0,
    'mse_5d':  0.3,
    'mse_60d': 0.3,
    'listnet_20d': 0.5,
}
WARMUP_PCT = 0.1                 # OneCycleLR warmup 比例

# --- Walk-Forward 設定 ---
WF_TEST_WINDOW_MONTHS = 6        # 測試窗口：6個月
WF_STEP_MONTHS = 3               # 滾動步幅：3個月
WF_MIN_TRAIN_YEARS = 3           # 最少訓練年數後才開始測試

# --- 資料時效設定 ---
DATA_SOURCE_PRIORITY = ['yfinance', 'twse_direct', 'finmind']  # [NEW]
MARGIN_FORWARD_FILL = True       # [NEW] 融資融券允許 Forward Fill

# --- 路徑設定 ---
KG_CACHE_PATH = PROCESSED_DIR / 'knowledge_graph_cache.npz'   # [NEW]
LLM_REPORT_PATH = RESULTS_DIR / 'market_summary.json'          # [NEW]
```

---

## 十七、美股擴展計畫（Phase 4，長期）

### 15-1. 資料來源

| 資料 | 來源 | 備註 |
|------|------|------|
| 個股日線 OHLCV | `yfinance` | 免費，1990+ |
| 基本面（EPS/PER） | `yfinance` + `financialmodelingprep` | Free tier 足夠 |
| 機構持股 | SEC 13F 申報（EDGAR） | 季頻，免費 |
| 空頭資料 | FINRA Short Volume | 免費，日頻 |
| 因子標準值 | Fama-French Data Library | 免費，學術級別 |

**Benchmark**：SPY 取代 TWII。無三大法人資料，Group B 改為 Short Volume / 機構持股變化。

### 15-2. 三種整合策略

| 策略 | 說明 | 建議 |
|------|------|------|
| **A. 分開訓練** | 台股模型 + 美股模型各自獨立 | **短期優先** |
| **B. 聯合訓練** | 加入 market_id embedding，一個模型同時學 | 中期目標 |
| **C. 遷移學習** | 美股大資料集預訓練，fine-tune 到台股 | 長期最優 |

---

## 十六、升級優先排序與 Roadmap

### 16-1. 優先排序表

| 序號 | 任務 | 需要重訓 | 預估工作量 | 理由 |
|------|------|---------|---------|------|
| 🥇 **1** | **Cross-Section DataLoader 修正（第八節）** | 是 | 2天 | 邏輯錯誤，不修則圖推理結果不可信 |
| 🥈 **2** | **流動性濾網 + 滑價折抵（第九節）** | 否 | 1天 | Quick Win，立竿見影 |
| 🥉 **3** | **Hybrid 資料源 + 時效優化（第十二節）** | 否 | 2天 | 推論提前 1.5hr，日常體驗大升級 |
| **4** | **本機 WSL2 推論環境建置（第十節）** | 否 | 半天 | 省 Colab quota，全自動化 |
| **5** | **LLM 報告生成層（第十一節）** | 否 | 1天 | 移除 FinBERT，input_dim 降回 46D |
| **6** | **Walk-Forward 框架，先用 V5.5 跑 baseline（第五節）** | 否 | 2天 | 建立基準，才能量化 V6 提升 |
| **7** | **資料擴充到 2012（第六節）** | 是 | 3天 | 資料量 ×2.4，Walk-Forward 才有意義 |
| **8** | **RSI/KD/ATR/OBV 特徵（第七節）** | 是 | 1天 | 台股最重要技術指標缺口 |
| **9** | **因子分組 Embedding + 多尺度 Mamba（第三節）** | 是 | 3天 | 核心架構升級 |
| **10** | **多輸出頭（5d/20d/60d）+ ListNet Loss（第三節）** | 是 | 2天 | 訓練目標重設計 |
| **11** | **TPEX 爬蟲 + Hybrid 圖邊（第三節）** | 是 | 3-5天 | KG 真正升級 |
| **12** | **V6 整合重訓（Colab A100）** | 是 | 2-3天 | 最終產出 |
| **13** | **回測引擎 MVP（第十四節）** | 否 | 3天 | 量化 V6 效益 |
| **14** | **MC-Dropout 置信度（第十三節）** | 否 | 1天 | 前端體驗升級 |
| **15** | **美股接入（第十五節）** | 是 | 5-7天 | Phase 4 長期目標 |

### 16-2. Roadmap（分 Phase）

```
Phase 0：邏輯修復（立刻，不能跳過）
  ☐ 實作 TemporalCrossSectionDataset，取代 FastAlphaDataset
  ☐ 用 V5.5 model 先跑一輪 Walk-Forward，建立 baseline ICIR

Phase 1：基礎設施升級（Quick Win，不需重訓）
  ☐ Hybrid 資料源：yfinance 取代 FinMind 價量，直連 TWSE 三大法人
  ☐ WSL2 推論環境建置 + Windows Task Scheduler 自動化
  ☐ LLM 報告生成層（新增 report_generator.py）
  ☐ 流動性濾網與滑價折抵
  ☐ 回測引擎 + Rolling History 存檔機制
  ✅ 完成後：推論從 19:00 提前至 17:30，本機全自動，無需每天開 Colab

Phase 2：資料與特徵升級（需重訓）
  ☐ 資料擴充：拉取 2012-2018 FinMind 歷史資料
  ☐ 移除 FinBERT，新增 RSI/KD/ATR/OBV，input_dim = 46D
  ☐ 架構實作：FactorGroupedEmbedding + MultiScaleMambaEncoder
  ☐ 多輸出頭（5d/20d/60d）+ ListNet Loss
  ☐ TPEX 爬蟲 + Hybrid 圖邊（供應鏈 + 滾動相關性）
  ☐ Walk-Forward 完整評估（24 Fold，ICIR 必須 > 0.5）
  ☐ Colab A100 重訓 V6，checkpoint 下載本機

Phase 3：體驗與生態完善
  ☐ MC-Dropout 置信度標示
  ☐ 前端新增「每日 LLM 報告」頁面
  ☐ 前端新增「策略回測報告」頁面

Phase 4：美股擴展（長期，條件成熟後啟動）
  ☐ 接入美股資料（yfinance + SEC 13F）
  ☐ 策略 A（分開訓練）：美股獨立模型
  ☐ 評估策略 B（聯合訓練，market_id embedding）
```

---

## 十七、關鍵決策備忘錄

### 已做決策（不再反覆討論）

| # | 決策 | 說明 | 日期 |
|---|------|------|------|
| 1 | **情緒分析架構** | FinBERT 移除，改由 LLM API（Claude/GPT）在推論後生成報告；input_dim 84D → 46D | 2026-04-23 |
| 2 | **每日推論部署** | 本機 RTX 3060 + WSL2；訓練仍用 Colab A100 | 2026-04-23 |
| 3 | **資料源策略** | 個股價量改用 yfinance（快）；三大法人改直連 TWSE；FinMind 降為 fallback + Forward Fill | 2026-04-23 |
| 4 | **預測目標** | 30天 Alpha → 20天 Alpha（主），同時輸出 5d / 60d 輔助 | 2026-04-23 |
| 5 | **評估指標** | Val MSE → IC / ICIR（業界標準）；通過標準 ICIR > 0.5 | 2026-04-23 |
| 6 | **訓練起始點** | 2019 → 2012（籌碼資料完整起點） | 2026-04-23 |
| 7 | **Cross-Section 修法** | TemporalCrossSectionDataset（同一交易日全截面），不妥協 | 2026-04-20 |
| 8 | **Loss 策略** | MSELoss 為基礎 + 可選 ListNet；永遠不用 Heaviside 懲罰 | 2026-04-20 |
| 9 | **KG 資料來源** | TPEX 產業鏈平台（ic.tpex.org.tw，已確認可爬，不需 API key） | 2026-04-20 |
| 10 | **GitHub 回測歷史** | 90天滾動保留，不需 Git LFS | 2026-04-20 |

### 技術陷阱備忘

| 陷阱 | 說明 | 應對方式 |
|------|------|---------|
| Mamba 環境依賴 | Mamba SSM 必須在 Linux + CUDA 即時編譯，wheel 不跨環境 | **WSL2 + CUDA** 一次安裝，永久有效 |
| V4 Target Collapse | Heaviside 方向懲罰導致模型輸出全 0 | 永遠只用平滑連續的 Loss |
| Cross-Section 污染 | 滑窗 batch 讓 GAT 跨時間步計算 | 用 TemporalCrossSectionDataset |
| KG Python O(N²) loop | `_build_knowledge_similarity` 是雙迴圈 | 改用 torch 向量化，或預算存 npz |
| GitHub 容量 | 單檔 100MB 限制 | CSV 用滾動保留，模型權重禁 commit |
| yfinance 缺股問題 | 部分興櫃/ETF 可能抓不到 | Fallback 到 FinMind；建立股票黑名單 |
| TWSE API 格式變更 | 政府網站可能無預警改版 | 爬蟲加版本偵測，失敗時 fallback 到 FinMind |
| 多輸出頭梯度衝突 | 3個 horizon 梯度方向可能不一致 | GradNorm 動態平衡 |

---

*本文件為 MarketMamba V6.0 唯一權威升級計畫。*
*所有關於 V6 的架構決策、模組設計與 Roadmap 以本文為準。*
*如有更新，請在本文直接修訂並更新頂部的「最後更新」日期。*
