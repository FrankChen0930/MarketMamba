# 🐍 MarketMamba — 專案導覽

> **台股截面因子模型**：Mamba SSM + KG-Enhanced GATv2 + 雙軌 FinBERT 情緒引擎  
> 自動化每日選股 × 量化決策系統 × 全棧 Web Dashboard

---

## 📌 一句話定位

MarketMamba 是一套**個人量化投資自動化系統**，每日收盤後對台股全市場（~2888 支）進行深度學習推論，輸出 Alpha 訊號排名，並透過 Web Dashboard 呈現選股結果、持倉追蹤與消息面分析。

---

## 🗂️ 目錄結構總覽

```
MarketMamba/
├── V6/                          ← 🔥 當前主力版本（V6 量化引擎）
│   ├── marketmamba/             ← 核心 Python 套件
│   │   ├── config.py            ← 全域超參數 & 路徑設定
│   │   ├── data/                ← 資料擷取 & 特徵工程
│   │   ├── models/              ← Mamba + GATv2 模型架構
│   │   ├── knowledge/           ← 知識圖譜建構
│   │   ├── signals/             ← 交易訊號掃描器（V6.1 新功能）
│   │   ├── evaluation/          ← IC / Walk-Forward 評估
│   │   ├── backtest/            ← 回測框架
│   │   ├── llm/                 ← LLM 消息面分析
│   │   ├── robot/               ← 投資組合管理機器人
│   │   └── deploy/              ← 部署相關工具
│   ├── notebooks/
│   │   └── V6_Training.py       ← Google Colab 訓練腳本（Cell 1~8）
│   ├── scripts/                 ← 工具腳本
│   ├── results/                 ← 每日推論輸出（df_kelly.csv 等）
│   ├── models/                  ← 模型 checkpoint（.gitignore）
│   ├── run_daily_inference.py   ← 每日推論主腳本
│   └── requirements.txt
│
├── app/                         ← 🌐 全棧 Web 應用
│   ├── frontend/                ← Vite + React 前端（已建好）
│   │   └── src/pages/           ← Dashboard / QuantAnalysis / MarketView / Portfolio
│   ├── backend/                 ← FastAPI 後端（規劃中）
│   └── API_REFERENCE.md         ← API 端點文件
│
├── AI_Trader_main/              ← 舊版 AI Trader（歷史版本）
├── Data/                        ← 資料存放區（.gitignore）
├── archive/                     ← 舊版本封存
├── docs/                        ← 文件
│
├── OVERVIEW.md                  ← 📄 本文件
├── PROJECT.md                   ← 技術架構深度文件
├── HANDOFF.md                   ← 對話交接文件（給新 AI 對話用）
├── README.md                    ← V5.5 技術說明（歷史版本）
├── signal_scanner_plan.md       ← V6.1 訊號掃描器設計規格
└── render.yaml                  ← Render 部署設定
```

---

## 🏛️ 系統架構

### 整體資料流

```
FinMind API（原始股價 / 財報 / 法人資料）
    │
    ▼ data/fetcher.py
Raw Parquet（~2888 支台股，2005年起）
    │
    ▼ data/feature_engine.py
特徵矩陣（46 個因子，技術面 + 籌碼面 + 基本面）
    │
    ├──▶ knowledge/graph_builder.py → 知識圖譜（KG）
    │
    ▼ models/trainer.py（Google Colab A100）
v6_final.pt（訓練好的模型權重）
    │
    ▼ run_daily_inference.py（本機 RTX 3060，每日收盤後）
Alpha 訊號 CSV（每日，2888 支股票 × Alpha 得分）
    │
    ├──▶ signals/scanner.py → action_signals.json
    ├──▶ git push → GitHub
    │
    ▼ FastAPI Backend（app/backend/）
前端 Dashboard（app/frontend/）
```

---

## 🧠 核心模型：MarketMambaV6

### 模型架構

```python
MarketMambaV6(
    INPUT: (N_stocks, 252 days, 46 features)

    # Stage 1：時序建模（每支股票獨立處理）
    Embedding: Linear(46 → 256)
    MambaBlocks × 8  (d_model=256, d_state=32)  # Mamba SSM 捕捉時序模式

    # Stage 2：截面互動（股票之間的關聯）
    GATv2: (N, 256) → (N, 256)  # 知識圖譜引導的跨股票注意力

    # Stage 3：預測頭
    Output: Linear(256 → 3)  # [Alpha_5d, Alpha_20d, Alpha_60d]
)

損失函數：MSE（各期報酬）+ ListNet（截面排名損失）
```

### 為何選用 Mamba SSM？

| 特性 | Transformer | Mamba SSM |
|------|-------------|-----------|
| 時序長度 | 受 attention 計算限制 | **線性複雜度**，可處理 252 天長序列 |
| 記憶選擇性 | 全局 attention | **選擇性狀態空間**，動態過濾重要資訊 |
| 效率 | O(N²) | O(N)，訓練更快 |

---

## 🕸️ 知識圖譜（Knowledge Graph）

共 **~43,000 條邊**，涵蓋 **2890 個節點（上市公司）**：

| 邊類型 | 權重 | 說明 |
|--------|------|------|
| TWSE 產業分類 | 0.5 | 同行業公司關聯 |
| 集團從屬 | 0.8 | 鴻海族群、台積電生態圈等 |
| TPEX 供應鏈 | 0.6 | 上下游廠商關係 |
| 60 日滾動相關 | 動態 | Pearson 相關 > threshold |

**混合邊建構公式（V5.5 參考）：**
```
edge_score = 0.7 × cosine_similarity + 0.3 × sector_similarity
```

---

## 📊 輸入特徵（46 維）

| 類別 | 特徵數量 | 內容範例 |
|------|---------|---------|
| 技術指標 | ~20 | MA5/10/20/60、RSI、MACD、KD、BB |
| 量價籌碼 | ~15 | 成交量、法人淨買超、融資融券 |
| 基本面 | ~8 | 營收 YoY、EPS、本益比 |
| 市場結構 | ~3 | 大盤指數位置、波動率 |

---

## 📈 交易訊號系統（V6.1）

### 入場條件（滿足 2/4 觸發推薦）

| # | 條件 | 判斷邏輯 |
|---|------|---------|
| 1 | **排名穩定性** | Top 10 連續 ≥2 天 **或** Top 50 連續 ≥3 天 |
| 2 | **高信心** | Uncertainty < 0.02（MC-Dropout） |
| 3 | **相對低點** | RSI < 40 **或** 當前價 < 20MA |
| 4 | **機構淨買入** | 外資/投信連續 2 天淨買入 |

> **大盤環境過濾**：TWII > 60MA（正常市場）需 2/4；TWII < 60MA（保守模式）需 3/4

### Trailing Stop 機制

| 持倉報酬 | 止損線位置 |
|----------|-----------|
| < +5% | 固定 -5%（成本價） |
| ≥ +5% | 成本 +2%（鎖利） |
| ≥ +10% | 成本 +6% |
| ≥ +15% | 成本 +10% |

---

## 🌐 Web Dashboard

### 前端（Vite + React）

已建好 4 個頁面（目前使用 mock 資料）：

| 頁面 | 路徑 | 功能 |
|------|------|------|
| **Dashboard** | `/` | 今日選股排名、Alpha 訊號 |
| **量化分析** | `/quant` | IC 趨勢、Walk-Forward 結果 |
| **AI 消息面** | `/market` | FinBERT 情緒分析、新聞摘要 |
| **持倉追蹤** | `/portfolio` | 永豐持倉、排名趨勢圖 |

> 🔒 個人版（4 頁籤含持倉）；公開版（3 頁籤，環境變數 `VITE_USER_MODE=public` 切換）

### 後端（FastAPI，規劃中）

```
GET  /api/signals              → 今日選股排名
GET  /api/signals/{date}       → 指定日期訊號
GET  /api/performance          → IC 趨勢、Walk-Forward 結果
GET  /api/portfolio            → 永豐持倉（shioaji）
POST /api/run-inference        → 觸發今日推論
POST /api/signals/cache/refresh → 刷新後端快取
```

---

## 🖥️ 部署架構

```
本機（RTX 3060）
    → 每日推論 run_daily_inference.py
    → git push 結果 CSV → GitHub

GitHub（Raw URL）
    → Render Backend 每小時拉取快取
    → 前端 API 請求

前端（Vercel）: https://marketmamba.vercel.app
後端（Render）: https://marketmamba-api.onrender.com
訓練（Google Colab A100）: 需要時手動觸發
```

---

## ⚙️ 技術棧

| 層次 | 技術 |
|------|------|
| **時序模型** | Mamba SSM (`mamba-ssm 2.3.0`) |
| **圖神經網路** | GATv2Conv (`PyTorch Geometric`) |
| **情緒分析** | HuggingFace Transformers（ProsusAI/finbert + chinese-finbert） |
| **資料源** | FinMind API + yfinance |
| **持倉 API** | 永豐證券 shioaji |
| **前端** | Vite + React + Recharts |
| **後端** | FastAPI + Uvicorn |
| **前端部署** | Vercel |
| **後端部署** | Render（免費方案） |
| **訓練平台** | Google Colab A100 |
| **本機推論** | RTX 3060 |

---

## 🔑 環境設定

### 所需 API Keys（存入 `.env`，絕不 commit）

```env
FINMIND_TOKEN=...          # FinMind 資料 API
SINOPAC_API_KEY=...        # 永豐證券（行情/帳務）
SINOPAC_SECRET_KEY=...     # 永豐證券
ANTHROPIC_API_KEY=...      # Claude API（LLM 消息面，待申請）
```

### 環境建置

```bash
# Python 環境（V6 引擎）
cd V6
pip install -r requirements.txt

# 前端（React Dashboard）
cd app/frontend
npm install
npm run dev   # → localhost:5173
```

---

## 📅 開發進度（截至 2026-05）

### 已完成 ✅

- [x] V6 模型架構（Mamba × 8 + GATv2）
- [x] 資料管線（FinMind 爬蟲 → 特徵工程 → KG 建構）
- [x] 知識圖譜（~43,000 條邊，4 種邊類型）
- [x] Quick Fold 驗證（N=500，IC=+0.0744 @epoch 15）
- [x] 前端骨架（4 個頁面，mock 資料）
- [x] 每日推論腳本（`run_daily_inference.py`）
- [x] V6.1 訊號掃描器設計規格

### 進行中 🔄

- [ ] Final Training（全部 2888 支股票，Colab A100）

### 待建 ⏳

- [ ] Walk-Forward（36 folds）歷史驗證
- [ ] 推論腳本整合訊號掃描器（V6.1 Signal Scanner）
- [ ] FastAPI 後端（接推論輸出 + 永豐持倉）
- [ ] 前端接上真實資料（移除 mock）
- [ ] Claude 新聞整合（LLM 消息面分析）
- [ ] PersonalOS 整合（Electron / Tauri 桌面應用）

---

## 📚 文件索引

| 文件 | 用途 |
|------|------|
| [OVERVIEW.md](./OVERVIEW.md) | 📄 **本文件**：專案全覽 |
| [PROJECT.md](./PROJECT.md) | 🔧 技術架構深度說明（模型參數、API 介面） |
| [HANDOFF.md](./HANDOFF.md) | 🤝 AI 對話交接文件（新對話可直接接續進度） |
| [signal_scanner_plan.md](./signal_scanner_plan.md) | 📡 V6.1 訊號掃描器設計規格 |
| [V6/V6_Master_Plan.md](./V6/V6_Master_Plan.md) | 🗺️ V6 主計畫文件（詳細技術規格） |
| [V6/V6_V61_Implementation_Guide.md](./V6/V6_V61_Implementation_Guide.md) | 🛠️ V6 → V6.1 實作指南 |
| [app/API_REFERENCE.md](./app/API_REFERENCE.md) | 🌐 FastAPI 端點參考文件 |
| [README.md](./README.md) | 📖 V5.5 歷史版本說明 |

---

## 🔮 未來規劃

1. **Final Training 完成** → 下載 `v6_final.pt`，建立推論腳本
2. **Walk-Forward** → 評估模型跨時期穩健性（36 folds）
3. **V6.1 Signal Scanner** → 系統化進出場決策
4. **FastAPI + 前端整合** → 從 mock 資料切換到真實訊號
5. **Claude 新聞整合** → LLM 驅動的消息面報告
6. **PersonalOS** → 整合進個人桌面應用 monorepo

---

## 🏆 實測驗證

> **3416 融程電** 實測：Model 推薦 #1 → 買入 → 兩天後 **+7%** 賣出  
> 驗證了 Alpha 訊號有效性，推動 V6.1 訊號掃描器的開發。

---

*Last updated: 2026-05 | MIT License © 2024-2026 FrankChen*
