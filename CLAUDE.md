# MarketMamba — AI 助手指引

> 最後更新：2026-05-25

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

MarketMamba 是一套**個人台股量化投資自動化系統**。每日收盤後（17:00）對全市場約 2,888 支股票做深度學習推論，輸出 Alpha 訊號排名，再透過雲端 Web Dashboard 呈現選股結果、持倉追蹤與 LLM 市場報告。

---

## 目錄結構速覽

```
MarketMamba/
├── V6/                          ← 當前主力量化引擎
│   ├── marketmamba/             ← 核心 Python 套件
│   │   ├── config.py            ← 全域超參數 & 路徑（改這裡就能改大部分設定）
│   │   ├── data/
│   │   │   ├── fetcher.py       ← FinMind + yfinance 資料爬蟲（含指數退避重試）
│   │   │   └── feature_engineer.py ← 46 維特徵工程
│   │   ├── models/              ← Mamba + GATv2 架構（⚠️ 不可修改）
│   │   ├── signals/
│   │   │   └── scanner.py       ← 交易訊號掃描器（加權評分系統 v1.2）
│   │   ├── llm/
│   │   │   └── report_generator.py ← Claude API 每日市場報告
│   │   ├── backtest/engine.py   ← 回測引擎
│   │   └── robot/portfolio_manager.py ← 持倉管理
│   ├── run_daily_inference.py   ← 每日推論主入口（WSL2 執行）
│   ├── scripts/
│   │   └── daily_inference.bat  ← Windows Task Scheduler 觸發點（17:00）
│   ├── results/                 ← 每日推論輸出（git push 到 GitHub）
│   │   ├── df_kelly.csv         ← 全市場 Alpha 排名（~2,500 支）
│   │   ├── df_traj.csv          ← 多期預測軌跡
│   │   ├── action_signals.json  ← 入場 / 退場訊號
│   │   ├── history_index.json   ← 每日 Top 50 歷史追蹤
│   │   └── market_summary.json  ← LLM 市場報告
│   └── models/                  ← ⚠️ 模型 checkpoint，禁止修改
│
├── app/
│   ├── backend/                 ← FastAPI（部署到 Render）
│   │   ├── main.py
│   │   ├── schemas.py           ← Pydantic 資料結構定義
│   │   └── routers/
│   │       ├── signals.py       ← 訊號 API + 快取（asyncio.Lock）
│   │       └── market.py        ← 大盤指標 API
│   └── frontend/                ← Vite + React（部署到 Vercel）
│       └── src/pages/           ← Dashboard / MarketView / Portfolio / ...
│
└── archive/                     ← 舊版本（V3–V5.5），只讀參考，不需修改
```

---

## 核心模型架構

**MarketMambaV6**（11.5M 參數，Google Colab A100 訓練，本機 RTX 3060 推論）：

```
輸入：(N_stocks, 252 天, 46 特徵)
  ↓ Linear Embedding (46 → 256)
  ↓ Mamba SSM × 8 blocks  (d_model=256, d_state=32)  — 時序建模
  ↓ GATv2  (知識圖譜引導，~43,000 條邊)              — 截面互動
  ↓ Linear (256 → 3)
輸出：[Exp_Alpha_5d, Exp_Alpha_20d, Exp_Alpha_60d]
```

- **MC-Dropout**（N=30 次採樣）估算每股不確定性（`Uncertainty`）
- **Alpha 截斷**：±2.0（防止離群值）
- **Signal_Quality**：`Net_Alpha_20d / (Uncertainty + 1e-6)`，截斷至 [-10, +10]（注意：舊版叫 `Sharpe_Score`，已全面改名）

---

## 每日推論流程（17:00 自動觸發）

```
daily_inference.bat（Windows Task Scheduler）
  └─ WSL2 → run_daily_inference.py
        [1/7] 資料更新（yfinance + FinMind）
        [2/7] 特徵矩陣建構（46 因子）+ 資料新鮮度檢查
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
# WSL2 直接執行
wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_daily_inference.py"

# 跳過 git push（測試用）
# 加上 --skip-push 旗標
```

### 強制刷新 Render 快取
```bash
curl -X POST https://marketmamba-api.onrender.com/api/signals/cache/refresh
```

### 推送結果到 GitHub（觸發 Render 重新抓資料）
```bash
git add V6/results/ && git commit -m "update results" && git push
```

### 本地啟動前端
```bash
cd app/frontend && npm run dev   # → localhost:5173
```

---

## 注意事項

- **`archive/`** 下的舊程式碼只做歷史參考，不在活躍維護範圍，修改時請確認是否真的需要。
- **知識圖譜**（`knowledge/graph_builder.py`）構建耗時，一般不需要重建，快取存在 `V6/data/cache_v6/` 下。
- **Google Colab 訓練**只在需要重訓時手動觸發，不要在本機嘗試訓練（VRAM 不足）。
- **`history_index.json`** 由每日推論自動維護（保留最近 60 個交易日），訊號掃描器的排名穩定性判斷依賴它。
