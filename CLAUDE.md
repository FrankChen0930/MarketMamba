# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> 最後更新：2026-06-12（前端體驗修補完成）

---

## 互動規則（請嚴格遵守）

1. **永遠用繁體中文回應**，包含程式碼以外的所有說明、分析與建議。
2. **動手改程式前先列出計畫**，說明要改哪些檔案、改什麼、為什麼，等我明確確認後再執行。
3. **改完後列出受影響的檔案清單**（檔案路徑 + 一行說明改了什麼）。
4. **Line Notify 已於 2025 年 3 月底停止服務**，不可在任何腳本或文件中加入或建議使用 Line Notify 相關功能。
5. **禁止修改 `V6/models/` 目錄下的任何檔案**（包含 `.pt` checkpoint），那是訓練好的模型權重，誤改無法復原。
6. **推論腳本在 WSL2（Ubuntu）環境執行**，路徑以 `/mnt/d/...` 掛載，呼叫方式為 `wsl -d Ubuntu -- bash -lc "..."`。
7. **輸出結果必須對人類可讀**：實作任何訓練 log、推論進度、診斷資訊時，數值必須明確顯示（例如 `scale_gate: [0.312, 0.487, 0.201]`），不可只實作邏輯而省略實際數字的輸出語句。如果一個功能「有做但看不到結果」，視同未完成。
8. **每次任務完成並獲得我確認後，主動更新 CLAUDE.md 的 Current Status 區塊**：把剛完成的事移到「最近完成」；更新「進行中」與「下一步」；若有重要設計決策，記錄到「決策紀錄」；更新頂部的「最後更新」日期。

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
│   │   │   ├── scanner.py           ← 交易訊號掃描器（加權評分系統 v1.2）
│   │   │   └── signal_conditions.py ← 共用進退場條件模組（V6.2 新增）
│   │   ├── quant/
│   │   │   └── pattern_scanner.py   ← 型態辨識（V6.2 重寫：5多方+2空方型態）
│   │   ├── llm/
│   │   │   └── report_generator.py  ← Claude API 每日市場報告
│   │   ├── backtest/
│   │   │   ├── engine.py            ← 回測引擎
│   │   │   ├── sim_engine_v2.py     ← 舊版模擬機器人
│   │   │   └── sim_engine_v3.py     ← 有狀態日更模擬機器人（V6.2 新增）
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
  ↓ MultiScaleMambaEncoder（3 分支並行：short 2層/mid 3層/long 3層，自適應融合；Long branch 套用 padding_mask）
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
- **Zero-Padding Mask（V6.2）**：`USE_PADDING_MASK = True`（`trainer.py`）。Short/Mid branch 取最後 20/60 步，均為真實資料不需 mask；Long branch 使用完整 252 步，padding 位置乘 0 截斷梯度（`architecture.py:MultiScaleMambaEncoder.forward`）
- **Scale Gate 監控**：每個 epoch validation 後 print `[scale_gate] Short/Mid/Long`，並儲存在 `TrainingHistory.scale_gates`；訓練圖表第 4 欄顯示三條線的 epoch 曲線

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

## 訊號系統（V6.2）

### `signals/scanner.py`（scan_version 1.2）— 產出 `action_signals.json`

| 條件 | 權重 |
|------|------|
| 排名穩定（Top10 ≥2天 or Top50 ≥3天） | 30 分 |
| 不確定度低（< 當日 Q30 分位數） | 25 分 |
| 機構連續淨買（≥2天） | 25 分 |
| 相對低點（RSI<40 or 價格<MA20） | 20 分 |

### `signals/signal_conditions.py`（V6.2 新增）— 共用進退場條件

**進場評分（最高 150 分）**：Scanner 100 + 型態加分最高 40 + 雙確認 +10
- 型態分數 60–74 → +20；75–89 → +30；≥90 → +40
- 雙確認（型態 ≥60 且 Alpha rank ≤200）額外 +10
- **門檻**：正常市場（TWII > MA60）≥70 分；保守模式 ≥90 分

**四層退場（`check_exit_conditions()`）**：
- 第一層（立即全出）：Trailing Stop / 型態失敗線跌破 / 外資連賣 3 天 / M頭或假突破確認 / 持有 >30 天
- 第二層（立即全出）：排名連 2 天出 Top50 / Uncertainty 超進場 2 倍 / RS_20d 負值 3 天 / 排名穩定性消失
- 第三層（減半倉）：RSI>75 且動能下滑 / 報酬 ≥+20% / Alpha_20d 連降 3 天
- 第四層（換倉）：SQ 排名落後市場後 50% 且有新訊號且滿倉

**Trailing Stop 四檔**：峰值 <+5%→止損 -5%；≥+5%→+2%；≥+10%→+6%；≥+15%→+10%

**進場理由記憶**：`EntryRecord.main_conditions` 記錄進場時觸發條件，退場時優先檢查該條件是否消失

### `quant/pattern_scanner.py`（V6.2 完整重寫）— 產出 `pattern_signals.json`

**多方型態（5種）**：W底、彈簧型W底、頭肩底、收斂三角底部、上飄旗形
**空方型態（2種）**：M頭（退場用）、假突破向下（退場用）

每個多方訊號含 `failure_stop`（型態失敗退場價，供四層退場第一層使用）
空方訊號輸出在 `bearish_signals` 列表（與 `signals` 分開）

**評分**：型態強度 40 + 成交量 30 + 位置（波段跌幅）20 + RSI 10 + 漂亮加分 + Alpha 加成（Top200→+10, Top300→+5）

### `backtest/sim_engine_v3.py`（V6.2 新增）— 有狀態日更模擬機器人

- 每日結束後寫 `V6/results/sim_state.json`（持倉完整狀態），隔天讀取繼續
- 交易紀錄 append 到 `V6/results/sim_trades.jsonl`
- 入口：`run_daily_update(date)` 日更；`run_backtest(reset=True)` 全量回放
- 進場評分使用 `signal_conditions.compute_entry_score()`（scanner + pattern 合計）
- 退場使用 `signal_conditions.check_exit_conditions()` 四層邏輯

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
- **`TemporalCrossSectionDataset`** 是 LAZY LOADING 設計——tensor 在 `__getitem__` 建立，不在 `__init__` 預建。每支股票至少需要 `SEQ_LEN × 0.8 = 202` 天資料才會被納入該交易日的 cross-section。`__getitem__` 回傳 4 個值：`(X, Y, stock_ids, padding_mask)`，其中 `padding_mask` 在 `USE_PADDING_MASK=True` 時為 bool tensor，`False` 代表 zero-padding 位置。
- **Scale Gate 觀察**：訓練中途停止後 `model` 不在 notebook 全域變數（函式未 return）。若需查看 scale gate，從 checkpoint 重新載入：`ckpt = torch.load("V6/models/v6_best.pt", ...); model = MarketMambaV6().cuda(); model.load_state_dict(ckpt["state_dict"])`，再跑一個 val batch 填入 `_last_scales`。

---

## 🔄 Current Status

> 最後更新：2026-06-12（玉山 API 金鑰外洩修復完成）

### 最近完成
- **安全修復：玉山 API 金鑰從 git 歷史移除（2026-06-12）**：
  - 發現 commit `dcca0fb`（2026-05-29）曾將整個 `玉山/` 資料夾（含 `E125721827_20270525.p12` 憑證、`config.simulation.ini` 內含完整 API Key/Secret/帳號）推上公開 GitHub repo，雖然當時 `main` 最新檔案列表已不含該資料夾，但歷史紀錄仍可被任何人挖出
  - 使用者已至玉山證券後台撤銷並重新申請該組 API Key/Secret/憑證
  - 用 `git filter-repo --path 玉山/ --invert-paths --force` 將 `玉山/` 從整個 git 歷史移除，並 `git push origin main --force` 覆蓋遠端（commit 9aee4a9）
  - `.gitignore` 新增 `玉山/`、`*.p12`、`*.ini`，新憑證已放回原路徑但不再被追蹤
- **前端體驗修補（2026-06-12）**：
  - **SYS-04 / UX-06**：`app/backend/routers/market.py` 的 `get_ticker()` 不再寫死 `price="—"`；改成 `asyncio.gather()` 並行呼叫 `_yf_v8(f"{ticker}.TW")` 取得 top7 訊號股報價，查不到（非上市）再 fallback `.TWO`（上櫃），填入真實 price/change/pct/up
  - **UX-05**：`PersonalOS/src/renderer/src/pages/MarketMamba/TradingSignals.jsx` 買入推薦空狀態改依 `data?.date` 區分：有日期但 `buySignals.length === 0` → 顯示「✅ 今日無股票達到入場條件」；無日期 → 顯示「⚠️ 今日訊號資料尚未生成，請稍後重新整理」
  - **UX-02 確認已解決**：檢查 `Portfolio.jsx` 第 566-570 行的 `chartData`，發現 PR 3 改寫時已改為真實「持有成本 vs 目前市值」長條圖，原本的 `Math.sin` 假資料已不存在，無需再修改
- **SYS-08：git push 失敗 retry 邏輯（2026-06-12）**：
  - 修改 `V6/run_daily_inference.py` 的 `_push_to_github()`：`git push origin main` 失敗時自動重試最多 3 次，每次間隔 10 秒；每次嘗試的成功/失敗都用 `logger.warning`/`logger.info` 印出明確次數與錯誤訊息（截取前 200 字元）；3 次都失敗才回傳 `False` 並提示手動修復指令
  - `git add` / `git commit` 失敗仍視為非暫時性錯誤，不重試（維持原行為）
- **推論穩定性強化 + Telegram 後台告警 + 自動化頁面美化（2026-06-12）**：
  - `PersonalOS/scripts/run_daily.py` 的 `run_wsl_inference()` 重構：
    - 新增 `_wsl_warmup()`：主推論前先跑 `wsl -d Ubuntu -e echo ok`（30s timeout）確認 VM 有反應，沒反應就直接記錄並跳過本次推論，不浪費 60 分鐘
    - 新增 `_wsl_shutdown()` + 重試邏輯：第一次嘗試逾時（60 分鐘）→ 執行 `wsl --shutdown` 重啟 VM → 等 10 秒 → 重試第二次（60 分鐘）
    - 所有步驟（暖機結果、各次嘗試耗時、是否觸發 shutdown）都用 `log()` 印出明確時間戳與數值
  - **SYS-07 完成**：新增 `send_telegram_notification()`，讀取 `.env` 的 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`，於 `main()` 失敗通知區塊與 Windows Toast 並行發送，內容含各失敗任務狀態/耗時/錯誤摘要，若有觸發 wsl 重試也會註明
  - 通知分工定案：**PWA 推播**（後續開發）負責 PersonalOS 前端一般通知；**Telegram**負責背景排程（WSL2 推論等）告警，兩者並存
  - `PersonalOS/src/renderer/src/pages/Automation/AdminOps.jsx` 整頁改寫：「純模型推論」與「完整每日自動化」改為**階段卡片清單**（⚪待執行/⚙️執行中動畫/✅完成/❌失敗），取代單條 progress bar；原始輸出改為預設收合的「詳細輸出」，並自動過濾 `UserWarning`/`FutureWarning`/cuDNN 等套件雜訊行（可勾選顯示）；「完整每日自動化」idle 時顯示 4 步驟預覽清單
- **6/10 推論失敗診斷 + 自動化逾時診斷強化（2026-06-11）**：
  - 診斷出 6/10、6/2 的推論失敗都是同一模式：`PersonalOS/scripts/run_daily.py` 的 `run_wsl_inference()` 透過 `wsl -d Ubuntu -e bash -c "...python V6/run_daily_inference.py..."` 執行，外層 60 分鐘 `subprocess.run(timeout=3600)` 超時被砍掉，且 60 分鐘內**完全沒有任何輸出**（連 conda activate 訊息都沒有），`tee` 緩衝區未 flush 導致 `inference.log` 也沒留下任何 6/10 紀錄
  - 正常推論耗時 9~14 分鐘（6/3=610s、6/8=768s、6/9=815s），但 6/1=2237s（37分，壓線過關）、6/2 與 6/10 都是 60 分超時失敗——約每週一次的偶發性嚴重變慢，疑似 WSL2 啟動/CUDA 初始化卡死，可能與本機 GPU 是否同時被遊戲/Blender 等占用有關（待使用者觀察確認）
  - 修改 `PersonalOS/scripts/run_daily.py` 的 `run_wsl_inference()`：呼叫 wsl 前後各印一行明確時間戳；`TimeoutExpired` 分支補抓 `e.stdout`/`e.stderr`（最多印最後 30 行），用來判斷下次卡死時究竟卡在哪一步
- 建立 CLAUDE.md 兩層架構（靜態規則 + 動態狀態）
- 新增規則 7（輸出可讀性）與規則 8（任務完成後自動更新本區塊）
- **V6.2 Zero-Padding Mask**：`USE_PADDING_MASK = True`，Long branch 套用 mask 截斷 padding 梯度，Short/Mid branch 不需 mask（`trainer.py`、`architecture.py`）
- **Scale Gate 監控強化**：每個 epoch print 數值、儲存至 `TrainingHistory.scale_gates`、訓練圖表新增第 4 欄折線圖（`trainer.py`、`v6_colab_training.py`）
- **訊號系統 V6.2 整修**：新增 `signal_conditions.py`（140 分進場評分 + 四層退場 + Trailing Stop + 進場理由記憶）；重寫 `pattern_scanner.py`（5 多方 + 2 空方型態 + `failure_stop`）；新增 `sim_engine_v3.py`（有狀態日更機器人，`sim_state.json` 持久化）
- **PR 3 — 持倉四層退場 checker（2026-06-09）**：
  - 新建 `V6/marketmamba/backtest/portfolio_checker.py`：讀最近 5 天 df_kelly archives 計算 streak（rank_out50_streak、alpha_20d_declining_days）+ prices_raw 計算 inst_sell_streak + pattern_signals 取 bearish/failure_stop，輸出 `portfolio_exit_check.json`（Top 300）
  - 修改 `V6/run_daily_inference.py`：Step 7 新增 `run_portfolio_check()` 呼叫
  - 修改 `app/backend/routers/signals.py`：新增 `GET /api/signals/portfolio/exit-check`，同 1h TTL cache 模式；`/cache/refresh` 一併清空
  - 修改 `PersonalOS/src/renderer/src/api/mm.js`：新增 `fetchPortfolioExitCheck()`
  - 完整改寫 `PersonalOS/src/renderer/src/pages/MarketMamba/Portfolio.jsx`：`ExitConditionModal` 升級為四層退場 UI（L1 停損、L2 信號惡化、L3 減倉、L4 換倉）；Trailing Stop 由前端從 avg_price 計算；風險分數改用四層觸發加權

### 進行中
- V6.2 訓練中（觀察 scale_gate 是否因 padding mask 改善均衡性）

### 下一步
- [ ] 驗證 Telegram 通知實際送達（使用者已自行測試 `send_telegram_notification()`，待下次真實失敗時確認格式可讀）
- [ ] 觀察下次推論異常變慢/超時時，warm-up 檢查 + 自動 `wsl --shutdown` 重試是否成功避開 60 分鐘卡死（`PersonalOS/scripts/logs/daily_*.log`）
- [ ] **6/10 當天使用者人不在家、電腦無人操作**（已排除遊戲/Blender 佔用 GPU 的可能）。新懷疑方向改為系統層級因素：(a) 電腦睡眠後被排程喚醒，WSL2 VM 冷啟動卡住 (b) Windows Update 背景安裝撞期 (c) Avira 排程全機掃描佔用磁碟 I/O (d) WSL2 VM 閒置過久喚醒卡死（已知通病）。下次卡住時比對 Windows 事件檢視器（Power-Troubleshooter / Kernel-Power）的睡眠喚醒時間點 + Windows Update / Avira 排程時間
- [ ] 觀察新一輪訓練的 scale_gate 數值，確認 Short/Mid/Long 是否趨於均衡
- [ ] 若 scale_gate 仍極度偏 Long，考慮在 MultiScaleMambaEncoder 加入 branch-level dropout 或 loss 正則化
- [ ] `sim_engine_v3.py` 實際跑一次 backtest，驗證四層退場邏輯正確觸發
- [ ] 觀察下次 git push 是否曾觸發過 SYS-08 重試（從 `inference.log` 確認訊息格式可讀）
- [ ] 部署後驗證 Ticker Bar 個股報價是否正確顯示（含至少一支上櫃股票，確認 `.TWO` fallback 有效）

> **PWA 推播通知**：使用者表示暫時不需要，已擱置，待之後有需求再提出
> **PR 3（持倉四層退場 / Portfolio 頁面）**：使用者已確認頁面內容完成，視為驗收通過

### 決策紀錄
- **padding mask 只加在 Long branch**：Short 取最後 20 步、Mid 取最後 60 步，在 ≥202 天資料的前提下這兩個 branch 輸入全為真實資料，不需 mask；只有 Long 使用完整 252 步才有 padding 問題
- **scale_gate 改為 `print()` 而非 `logger.info()`**：Colab 預設 logging level = WARNING，`logger.info` 會靜默丟棄；`print(flush=True)` 永遠可見，與其他訓練 log 風格一致
- **claude.ai Project 知識庫改放 OVERVIEW.md 等靜態文件，CLAUDE.md 動態狀態區塊僅供 Claude Code 使用，不需同步到 Project**
- **PR 3 的 rs_20d / rsi 欄位留 null**：df_kelly.csv 目前不含 RS_20d（那是 feature matrix 的中間產物），前端顯示四層時這兩個欄位以 0 fallback，不影響 L1~L4 主要條件判斷；待 V6.2 模型若輸出 RS 相關信號時再補
- **portfolio_exit_check 的 inst_sell_streak 從 prices_raw 計算而非 action_signals**：prices_raw 含原始 Foreign_Buy/Foreign_Sell，精確度更高；action_signals 的 institutional_buy 只是 scanner 的 boolean flag
