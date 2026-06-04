# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> 最後更新：2026-06-03

---

## 互動規則（請嚴格遵守）

1. **永遠用繁體中文回應**，包含程式碼以外的所有說明、分析與建議。
2. **動手改程式前先列出計畫**，說明要改哪些檔案、改什麼、為什麼，等我明確確認後再執行。
3. **改完後列出受影響的檔案清單**（檔案路徑 + 一行說明改了什麼）。
4. **Line Notify 已於 2025 年 3 月底停止服務**，不可在任何腳本或文件中加入或建議使用 Line Notify 相關功能。
5. **禁止修改 `V6/models/` 目錄下的任何檔案**（包含 `.pt` checkpoint），那是訓練好的模型權重，誤改無法復原。
6. **推論腳本在 WSL2（Ubuntu）環境執行**，路徑以 `/mnt/d/...` 掛載，呼叫方式為 `wsl -d Ubuntu -- bash -lc "..."`。

---

## 專案定位

MarketMamba 是一套**個人台股量化投資自動化系統**。每日收盤後（17:00）對全市場約 2,515 支股票做深度學習推論，輸出 Alpha 訊號排名，再透過雲端 Web Dashboard 呈現選股結果、持倉追蹤與 LLM 市場報告。

---

## 目錄結構速覽

```
MarketMamba/
├── V6/                          ← 當前主力量化引擎
│   ├── marketmamba/             ← 核心 Python 套件
│   │   ├── config.py            ← 全域超參數 & 路徑（改這裡就能改大部分設定）
│   │   ├── data/
│   │   │   ├── fetcher.py       ← FinMind + yfinance 資料爬蟲（含指數退避重試）
│   │   │   ├── merger.py        ← 載入各 raw parquet（不做 join，只 load）
│   │   │   └── feature_engineer.py ← 56/59 維特徵工程（見下方說明）
│   │   ├── models/              ← Mamba + GATv2 架構（⚠️ 不可修改）
│   │   ├── signals/
│   │   │   └── scanner.py       ← 交易訊號掃描器（加權評分系統 v1.2）
│   │   ├── llm/
│   │   │   └── report_generator.py ← Claude API 每日市場報告
│   │   ├── backtest/engine.py   ← 回測引擎
│   │   └── robot/portfolio_manager.py ← 持倉管理
│   ├── run_daily_inference.py   ← 每日推論主入口（WSL2 執行）
│   ├── notebooks/
│   │   └── v6_colab_training.py ← Colab 訓練/續訓主腳本（見 Colab 工作流程）
│   ├── scripts/
│   │   └── daily_inference.bat  ← Windows Task Scheduler 觸發點（17:00）
│   ├── results/                 ← 每日推論輸出（git push 到 GitHub）
│   └── models/                  ← ⚠️ 模型 checkpoint，禁止修改
│
├── Data/                        ← 本機資料目錄（不進 git）
│   ├── processed_v6/            ← feature matrix + raw parquet 快取
│   └── cache_v6/                ← ticker universe 等小型快取
│
├── app/
│   ├── backend/                 ← FastAPI（部署到 Render）
│   └── frontend/                ← Vite + React（部署到 Vercel）
│
└── archive/                     ← 舊版本（V3–V5.5），只讀參考
```

---

## 核心模型架構

**MarketMambaV6**（~4M 參數，Google Colab A100 訓練，本機 RTX 3060 推論）：

```
輸入：(N_stocks, SEQ_LEN=252, INPUT_DIM=56 或 59)
  ↓ FactorGroupedEmbedding（按 4 組比例分配投影 → d_model=256）
      Group A price_momentum    (12 dims) → sub_dim 54
      Group B institutional_flow(20 dims) → sub_dim 94
      Group C fundamentals      (12 dims) → sub_dim 54
      Group D macro_environment (12 dims) → sub_dim 54
  ↓ MultiScaleMambaEncoder（3 分支並行：short 2層/mid 3層/long 3層，自適應融合）
  ↓ GATv2（知識圖譜引導，CSR 稀疏矩陣，~640K 條邊）
  ↓ Gating Fusion（gate = sigmoid(Linear(2×d_model → d_model))）
  ↓ MultiHorizonHead（3 個獨立 Linear → pred_5d/20d/60d）
輸出：[Alpha_5d, Alpha_20d, Alpha_60d]
```

**特徵維度說明**：
- `config.py` 的 `INPUT_DIM` 控制推論時使用的維度
- V6.1（推論）：`INPUT_DIM=56`，RS 相對強度特徵未啟用
- V6.2（訓練）：`INPUT_DIM=59`，`RS_5d/RS_20d/RS_60d` 啟用（在 `FEATURE_GROUPS["price_momentum"]` 取消注釋後生效）
- 切換時須同步修改 `config.py` 的 `INPUT_DIM` 與 `FEATURE_GROUPS`，並確保 `assert len(FEATURE_COLS) == INPUT_DIM` 通過

**其他重點**：
- **MC-Dropout**（N=30 次採樣）估算每股不確定性（`Uncertainty`）
- **Alpha 截斷**：±2.0（防止離群值）
- **Signal_Quality**：`Net_Alpha_20d / (Uncertainty + 1e-6)`，截斷至 [-10, +10]（舊版叫 `Sharpe_Score`，已全面改名）

---

## 每日推論流程（17:00 自動觸發）

```
daily_inference.bat（Windows Task Scheduler）
  └─ WSL2 → run_daily_inference.py
        [1/7] 資料更新（yfinance + FinMind）
        [2/7] 特徵矩陣建構（56 因子）+ 資料新鮮度檢查
        [3/7] Mamba+GAT 推論 → df_kelly.csv, df_traj.csv
        [4/7] LLM 市場報告 → market_summary.json
        [5/7] 歸檔（90 天滾動）
        [6/7] 訊號掃描 → action_signals.json
        [7/7] git push → GitHub → Render 快取更新
```

推論進度透過 tkinter 視窗即時顯示（WSLg）。成功 3 秒自動關閉；失敗保持開啟並置頂。

---

## 訊號掃描器（V6.1，`signals/scanner.py`，scan_version 1.2）

### 入場加權評分（I4，取代舊版二元 2/4）

| 條件 | 權重 |
|------|------|
| 排名穩定（Top10 ≥2天 or Top50 ≥3天） | 30 分 |
| 不確定度低（< 當日 Q30 分位數） | 25 分 |
| 機構連續淨買（≥2天） | 25 分 |
| 相對低點（RSI<40 or 價格<MA20） | 20 分 |
| **BUY 門檻**：正常市場 ≥55 分，謹慎市場 ≥70 分 | — |

### 持倉風控（I3）

- 最多 15 檔，單股建議比重上限 10%（自動正規化）
- 不足 5 檔時輸出警告

---

## Colab 訓練工作流程

訓練腳本是 `V6/notebooks/v6_colab_training.py`（對應 Colab notebook 的各 Cell）。

### 首次訓練 / 全新環境
```
Cell 0 → 1 → 2 → 3 → 3b → 4
```

### Colab 斷線後 Resume
```
Cell 0 → 1 → 2 → 3 → 3b → 4b
```
Cell 4b（Resume）的重要行為：
- **Optimizer 狀態**：嘗試從 checkpoint 還原（若 shape 不符則 fresh start）
- **Scheduler 狀態**：**刻意不還原**，建立新的 OneCycleLR（`RESUME_LR=5e-5`，`pct_start=0.05`）。這是設計上的 fine-tuning 行為，LR log 顯示「5e-05」是正確的，不是 bug。
- **History**：從 checkpoint 的 history 接續，圖表會顯示 resume 前後的完整曲線

### 資料上傳流程（本機 → Colab）
```
本機 Data\processed_v6\ → 壓縮為 processed_v6.zip
  → 上傳到 Google Drive: MyDrive/MarketMamba_V6/processed_v6.zip
  → Colab Cell 2 解壓縮 → PROCESSED_DIR
```
Cell 3 用 `merge_all_data()` 讀取 raw parquet → `build_features()` 重建 feature matrix。

---

## 資料管線注意事項

### `ticker_universe.parquet` 快取（重要）

`Data/cache_v6/ticker_universe.parquet` 是 `load_ticker_universe()` 的持久化快取，**永遠優先於 FinMind API**，且 `run_full_data_sync(force_rebuild=True)` 不會重建它。

若此快取損壞（例如包含 `00400A`、`00679B` 等非 4 位數字代碼），將導致 `prices_raw.parquet` 包含數萬支非股票工具，feature matrix 的 stock 數量會異常膨脹。

**症狀**：`[Dataset init] 46488 valid days | 46488 stocks pre-indexed`

**修復**：
```python
# 刪除快取，下次 load_ticker_universe() 會重新從 FinMind 抓取並套用 ^\d{4}$ 過濾
Path("Data/cache_v6/ticker_universe.parquet").unlink()

# 若 prices_raw.parquet 也已污染，過濾修復（不需要重新抓資料）：
df = pd.read_parquet("Data/processed_v6/prices_raw.parquet")
df = df[df["stock_id"].str.match(r"^\d{4}$")]
df.to_parquet("Data/processed_v6/prices_raw.parquet")
```

### feature_engineer.py 的 join 方向

所有資料合併（institutional、margin、shareholding 等）都是 `how="left"` join 到 `prices_raw` 上。因此 prices_raw 是 universe 的唯一決定者，其他 raw 檔案即使包含額外 stock_id 也不影響結果。

### 訓練資料驗證

重建 feature matrix 後，確認以下數字正常：
- `Unique stocks` ≈ 2,515（台股有效歷史資料）
- `Unique dates` ≈ 5,000–5,500（2005 至今）
- `[Dataset init]` 顯示的 valid days 應為 train_dates 的子集（不可能大於傳入的 dates 數量）

---

## 部署資訊

| 服務 | 網址 | 觸發方式 |
|------|------|---------|
| 後端（Render） | `https://marketmamba-api.onrender.com` | push to `main` 自動部署 |
| 前端（Vercel） | `https://marketmamba.vercel.app` | push to `main` 自動部署 |
| Render rootDir | `app/backend` | — |
| Vercel rootDir | `app/frontend` | — |

**Render 免費方案**：15 分鐘無流量會 spin down，首次請求慢 30–60 秒。

**後端資料來源**：啟動時從 GitHub raw URL 拉 `V6/results/df_kelly.csv` 等檔案快取至記憶體（1 小時 TTL，`asyncio.Lock` 防競態）。

強制刷新快取：
```bash
curl -X POST https://marketmamba-api.onrender.com/api/signals/cache/refresh
```

---

## 環境變數

**WSL2 / `V6/.env`**：
```
FINMIND_TOKEN=...        # FinMind 資料 API
ANTHROPIC_API_KEY=...    # Claude LLM 報告
RENDER_BACKEND_URL=https://marketmamba-api.onrender.com
```

**Render 環境變數**：
```
GITHUB_RESULTS_URL=https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/V6/results/df_kelly.csv
ALLOWED_ORIGINS=https://marketmamba.vercel.app
```

---

## 常見開發任務

### 手動執行推論
```bash
wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_daily_inference.py"
# 跳過 git push（測試用）：加上 --skip-push 旗標
```

### 強制刷新 Render 快取
```bash
curl -X POST https://marketmamba-api.onrender.com/api/signals/cache/refresh
```

### 推送結果到 GitHub
```bash
git add V6/results/ && git commit -m "update results" && git push
```

### 本地啟動前端
```bash
cd app/frontend && npm run dev   # → localhost:5173
```

---

## 注意事項

- **`archive/`** 下的舊程式碼只做歷史參考，不在活躍維護範圍。
- **知識圖譜**（`knowledge/graph_builder.py`）構建耗時，快取在 `Data/cache_v6/knowledge_graph_cache.npz`。KG 的 node 數量可能多於當前訓練 universe（CSR 子圖提取會自動處理），不需重建。
- **Google Colab 訓練**只在需要重訓時手動觸發，不要在本機嘗試訓練（VRAM 不足）。
- **`history_index.json`** 由每日推論自動維護（保留最近 60 個交易日），訊號掃描器的排名穩定性判斷依賴它。
- **`TemporalCrossSectionDataset`** 是 LAZY LOADING 設計——tensor 在 `__getitem__` 建立，不在 `__init__` 預建。每支股票至少需要 `SEQ_LEN × 0.8 = 202` 天資料才會被納入該交易日的 cross-section。
