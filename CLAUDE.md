# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> 最後更新：2026-08-02（**F6 這一輪收尾完成**：2×2（Group D × GAT）四格全齊，最佳格 `no_macro + v2 圖` **訊號層 IC +0.1145、ICIR 1.340** 為全專案新高；兩個效應**可加**（交互作用 t=−0.59）。GRU 的 decile Sharpe 2.846 也查清楚了：**不是 bug、不是 panel 差異**。**下一步：把最佳 checkpoint 用 `score_mamba_local.py` 產分數過 portfolio_lab**（口徑不同，在那之前不可與 GRU 的組合層數字並列）。完整紀錄見 `docs/f6-training-log-and-readout.md`、`docs/portfolio-lab-results-2026-08-01.md`）

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

## 協作偏好 / 開發習慣（2026-06-19 整理，供 Claude Code 接手）

> 從長期協作歸納的工作風格，補充上面「互動規則」。

**工作節奏**
- 計畫先行、確認後執行：動程式前先列「改哪些檔、改什麼、為什麼」，等我 OK 才動手（規則 2，我很在意）。
- 診斷先做、production 一次到位：先用便宜的隔離實驗收集資訊（**一次只改一個變因**，結果才可歸因），問清楚了才動正式程式、且只動一次，避免「改來改去」。
- 收尾要記錄：段落完成且我確認後，更新 Current Status + 決策紀錄；重要實驗也記 obsidian `03 架構筆記/訓練紀錄.md`。

**隔離原則（最重要）**
- 線上 V6.1 是紅線、**絕不能弄壞**（家人每天看 dashboard）。新東西一律**附加、並行、不動既有**：新 router／新頁／新檔，不改既有 endpoint／頁面／資料流。
- 診斷實驗一律在 `V6/experimental/` 副本；受保護的 `marketmamba/models/` 不碰，要動需我逐次明確授權。
- 改動若可能碰到線上，先用隔離方式（獨立 process、獨立輸出檔、`try/except` 包住）並說明為何安全。

**驗證與誠實**
- 凡事先驗證再相信：輸出做健檢（筆數／NaN／分布／前幾名合理性）、數字程式化核對、語法／邏輯抽驗。出乎意料的好結果要主動點出可疑處 + 給確認方式，不報喜不報憂。
- 誠實勝過順從：tradeoff、限制、真正的資料依賴（如 sim 需先累積歷史）直說，別硬做沒意義的事。當思考夥伴、不是 yes-man；歡迎對我 push back。

**執行分工**
- 我自己跑 Colab 訓練／WSL 推論／git push；你負責準備好程式 + 給**可直接複製貼上的指令**（含 wsl/conda 殼）。沙箱跑不了本系統推論（無 torch/GPU/資料），runtime 除錯靠我貼 log 給你判讀。
- git：指定檔案 `git add <檔>`、**不要 `git add -A`**（本機有 56 維 config 等 dirty 檔不能上）。

**領域脈絡**
- 我以**短線操作**為主（驅動了 5d／雙模型方向）。訓練 Colab A100、推論本機 RTX 3060 + WSL2。
- **56 維（本機 V6.1）vs 59 維（Colab／雙模型）config 分裂**是反覆出現的坑：本機 `config.py` 是 56 維、遠端／Colab 是 59 維，動到要小心。
- 兩個 repo：**MarketMamba**（量化系統）、**PersonalOS**（個人自動化 + dashboard host；排程 `scripts/run_daily.py`、交易日 gate `scripts/trading_day.py` 查 TWSE）。
- 預算（Colab 費）不是主要限制，但討厭浪費的重訓。

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

### `signals/scanner.py`（scan_version 1.4）— 產出 `action_signals.json`

| 條件 | 權重 |
|------|------|
| 排名穩定（Top10 ≥2天 or Top50 ≥3天） | 30 分 |
| 不確定度低（< 當日 Q30 分位數） | 25 分 |
| 機構連續淨買（Foreign_Net ≥2天） | 25 分 |
| 相對低點（RSI<40 or 價格<MA20） | 20 分 |

- **BUY 判斷（2026-07-07 統一）**：複合分數（4 條件 + 型態加分）**≥70 分**（保守模式 ≥90），與 `signal_conditions.compute_entry_score()` / sim_engine_v3 同一套標準；權重常數 import 自 signal_conditions，不再自帶副本。條件數（x/4）保留為顯示資訊
- 退場訊號**不在 scanner 產出**（`exit_signals` 恆為空列表、僅向下相容）；真正退場 = signal_conditions 四層 → `portfolio_exit_check.json`
- regime 判斷（TWII vs MA60）：prices_raw 找不到指數時 fallback `macro_raw.parquet` 的 TWII_Close，**含 10 天新鮮度檢查**（macro 太舊會明講並維持 NORMAL）。⚠️ macro_raw 目前停在 2026-04-24（每日更新不含 macro），保守模式閘門實質尚未啟用

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
| 前端（Vercel） | `https://market-mamba-pi.vercel.app` | push to `main` 自動部署 |
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
ALLOWED_ORIGINS=https://market-mamba-pi.vercel.app
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
- **`marketmamba/models/inference.py` 已棄用（D4）**：實際線上推論是 `V6/run_daily_inference.py:run_inference()`，兩者欄位已分歧（前者輸出 `Uncertainty_5d/20d/60d`+`Slippage`，後者 `Uncertainty`+`Slippage_Est`）。因 models/ 目錄受保護不直接修改檔案，修推論一律改 `run_daily_inference.py`，不要動 `models/inference.py`
- **`TemporalCrossSectionDataset`** 是 LAZY LOADING 設計——tensor 在 `__getitem__` 建立，不在 `__init__` 預建。每支股票至少需要 `SEQ_LEN × 0.8 = 202` 天資料才會被納入該交易日的 cross-section。`__getitem__` 回傳 4 個值：`(X, Y, stock_ids, padding_mask)`，其中 `padding_mask` 在 `USE_PADDING_MASK=True` 時為 bool tensor，`False` 代表 zero-padding 位置。
- **Scale Gate 觀察**：訓練中途停止後 `model` 不在 notebook 全域變數（函式未 return）。若需查看 scale gate，從 checkpoint 重新載入：`ckpt = torch.load("V6/models/v6_best.pt", ...); model = MarketMambaV6().cuda(); model.load_state_dict(ckpt["state_dict"])`，再跑一個 val batch 填入 `_last_scales`。

---

## 🔄 Current Status

> 最後更新：2026-08-02（F6 2×2 完成、GRU decile 異常查清。**V6.1 已非紅線，使用者表示可停**。Phase 3 模型實驗仍暫停中。使用者 08-02 出門、電腦關機，進度暫停）

### 最近完成
- **★★ F6 2×2（Group D × GAT）四格全齊：最佳格訊號層 IC +0.1145，全專案新高（2026-08-02）**。第四格 JSON 在 `D:\Downloads\groupd_ablation_result_gatv2.json`（另三格：`groupd_ablation_result.json`、`kg_ablation_result.json`）：

  | 重評 mean IC（582 天） | with macro | no macro | **Δ 移除 macro** |
  |---|---|---|---|
  | **no GAT** | +0.0884 | +0.1070 | **+0.0186**（NW t=2.93） |
  | **v2 圖** | +0.0991 | **+0.1145** | **+0.0154**（NW t=3.23） |
  | **Δ 加 GAT** | +0.0107（t=1.73） | +0.0075（t=**3.80**） | |

  - **兩個效應可加**：可加預測 +0.1177、實際 **+0.1145**、**交互作用 −0.0032（t=−0.59，與 0 無法區分）** → 不是搶同一份訊號，是各自獨立。跑之前預估的「+0.117 附近」幾乎命中
  - **+0.1145 是同窗訊號層新高**。用同一把尺重算的對照（見下一條 GRU 診斷）：GBDT +0.1027、GRU +0.1016、v2_kg +0.0989（與 Colab 記的 +0.0991 對得上 → 兩邊確為同尺）。**ICIR 1.340、IC>0 比例 90.9%** 拉開更多（其他模型 ICIR 都在 0.73~0.85）
  - **★ 移除 Group D 讓 GAT 的比較乾淨了 2.9 倍**：GAT 效應在 with macro 下 Δ=+0.0107 但 t 只有 1.73（**std(Δ)=0.0792**）；在 no macro 下 Δ 較小（+0.0075）**t 反而衝到 3.80**（**std(Δ)=0.0273**）。與 GAT 消融「C−B 的 t=5.17 來自離散度小 5.6 倍」同一個現象 → **Group D 除了製造過擬合，還在對逐日 IC 注入雜訊**（第二條獨立證據）
  - **★ 一個掛著的異常被解掉了**：舊紀錄「`v2_kg` 訊號層贏 `no_gat` 但**組合層輸**（+8.3% vs +10.7%）」是第四次「IC 與組合層不同調」。移除 Group D 後**兩層同調**——組合層（舊口徑 Top50/5日）：no_gat 從 +10.7%(Sh 0.572) → **+27.7%(1.177)**、v2 圖從 +8.3%(0.578) → **+32.5%(1.417)**。**GAT 在 with macro 下傷組合層（−2.5pp）、在 no macro 下幫組合層（+4.8pp）**
  - **健康檢查通過**：`D` 格 val IC 曲線 `+0.1145(ep5) … +0.1145(ep10)` **走平後 early stop 觸發 → 沒有被排程截斷**，數字不是低估。四格 harness 核對過（同列參數量 1,394,301 / 1,659,005、同 seed 20260730、都是 582 天、KG 2,245 節點/32,083 邊、第一個 batch 1,668 支）
  - **穩健性**：兩個 macro 效應前後半同向（後半更強）、去極端 5 天後仍在（+0.0157 / +0.0134）
  - ⚠️ **限制**：仍是**單一 seed**（run-to-run σ 從沒量過）；三個 with-macro 格出自 `kg_ablation`、與第四格**跨腳本**（但欄位逐項相同、可比）；GAT 那兩個是**髒配對**（參數量差 264,704、RNG 分岔），macro 那兩個才乾淨；單一 582 天多頭窗、無 WF；`FED_Rate` 是死維 → 一律寫「**11 維有效特徵**」
  - **⚠️ 口徑警告**：上面的 +32.5% 是 `portfolio_backtest` 舊口徑（Top50/5日/無緩衝），**不可與 portfolio_lab 的數字（k=1.5/20日）並列**。要可比必須先產分數過 portfolio_lab（見「下一步」）
- **★ GRU 的 decile spread Sharpe 2.846 查清楚了：不是 bug、不是 panel 差異，是真的（2026-08-02）**。診斷腳本在 scratchpad（純讀，未動 `V6/experimental/` 任何既有檔案）：
  - **四個「這是假象」的假設全部排除**：① panel —— gru 與 ridge 的 **(Date, stock_id) 逐對完全相同**（0 差異、無重複、無 NaN，皆 582 天 × 1,915 支）② 切分洩漏 —— train ≤2023-12-31、test 2024-01 起、full-train 只用 train 列 ③ 小型/低流動股 —— **限制在高流動前 1/3 後 Sharpe 2.66**（v2_kg 1.41、gbdt 1.25），**優勢反而更大** ④ 少數幾天/下市邊緣列 —— 去最好 20 天仍有 1.56（v2_kg 0.56）；兩端 NaN 標籤比例 ≤0.02% 低於基準 1.21%
  - **必須先修的比較基礎**：GRU 的 +0.1018 出自 `baseline_rnn`、v2_kg 的 +0.0991 出自 Colab `kg_ablation`，**是兩條計算路徑**。用同一把尺重算後：GBDT **+0.1027** ≈ GRU **+0.1016** > v2_kg +0.0989 > Ridge +0.0893
  - **機制＝十分位輪廓的形狀**：GRU 的十格平均前瞻 5d alpha **接近嚴格單調遞增**且兩端推得最開（D0 −0.0015 最低、D9 +0.0061 最高、D9−D0 = **0.0076** vs gbdt 0.0061、v2_kg 0.0049）；GBDT 的 D1~D6 平坦且 D4 下凹。**Spearman IC 由整個分布主導、對輪廓斜率不敏感** → 同樣的 IC 可以對應到差 25% 的兩端差
  - **★ 更有意思的型態：GRU 的兩端訊號更耐放**。spread Sharpe 在 freq=1 時 **GBDT 4.52 > GRU 4.32**，freq=5 時 GRU 2.85 > GBDT 1.69，freq=20 時 GRU **1.46** > GBDT 0.34 → **頻率一降下來 GRU 就拉開**，正好是本專案已確立「最會賺錢」的那個軸
  - **配對 NW t**（同日相減）：vs ridge **+3.48**、vs gbdt **+3.15**、vs v2_kg **+2.86**、vs v3_kg **+2.80**，前後半同向（後半更強）
  - **組合層也是最好的**（portfolio_lab、N=50/k=1.5/20日）：gru **+31.3%**/Sh 1.401/超額 **+13.9%** > v3_kg +26.8%/1.522/+9.4% > v2_kg +26.0% > ridge +21.2% > gbdt +17.9%
  - **這對既有結論是個修正**：Step 4/5 的「模型形式只在 ±0.01 內移動」**在 IC 層仍成立**（0.089~0.103），但**組合層拉開了 13pp** → 模型形式在訊號層看不出差別，在組合層看得出來
  - ⚠️ **限制**：單一 seed、單一 582 天多頭窗、GRU 無 WF；優勢有相當部分在 **short 腳**（−13.5%），台股放空有借券與平盤下限制，**decile spread 高估了可落袋的部分**（不過 long 腳 +36.3% 也是全場最高）；PROTOCOL **沒有 purge**（TRAIN_END 緊接 TEST_START、5 天標籤重疊），但 ridge/gbdt/gru 共用，解釋不了彼此差異
  - ⚠️ **`wf_ridge` 3.007 / `wf_gbdt` 4.136 的 decile Sharpe 比 GRU 還高，但那是 11 年窗，不可與 582 天的數字並列**——這個指標對窗長很敏感
  - **更正一處我自己的口誤**：逐季是 **9/10** 贏過其他三個模型（2024Q4 輸給 GBDT，0.578 vs 0.761），不是 10/10
- **★ Group D 消融：移除總經 12 維讓 IC 顯著上升（2026-08-02）—— F6 系列量到最大的單一效應**。完整逐 epoch 數字在 Drive 的 `groupd_ablation_result.json`：
  - **`with_macro` 峰值 +0.0884 @ep5（ICIR 0.733、年化 +10.7%）vs `no_macro` 峰值 +0.1070 @ep9（ICIR 1.191、年化 +27.7%）→ Δ = −0.0186、配對 NW t = −3.12**。兩組參數量相同（1,394,301）＝ mask 設計成立、架構不是變因
  - **機制清楚**：Group D 是**每日橫斷面常數**（實測抽 5 天，12 維的當日相異值數都是 1）→ 在橫斷面排序上零資訊，但給了模型「記住是哪一天」的通道 → **它的作用是製造過擬合**。`with_macro` 的 val_loss ep2 谷底後單調惡化（0.12799→0.1303），`no_macro` 一路降到 ep6 後平穩（0.12743→0.12677）
  - **`no_macro` 在十個 epoch 每一格都贏**（Δ 從 −0.009 到 −0.0325）→ **結論方向不需要重跑就成立**
  - ⚠️ ~~**但幅度被低估**：`no_macro` 峰值在 ep9、代表被排程截斷~~ → **2026-08-02 更正，這句寫得太重**：曲線尾端是 `+0.1055, +0.1070, +0.1068`，**其實已經走平**（接近平台期，不是被截斷）。第四格 `no_macro + v2 圖` 更明確——峰值 ep5、ep10 仍是同一個 +0.1145 後 early stop 觸發。依 §4.1 規則本該用更長 epochs 重跑（14.4h），但**使用者與 Claude 判斷改跑 2×2 第四格更划算**，事後看是對的
  - **實測確認 `FED_Rate` 是死維**（相異值 1、std 0）→ 結論一律寫成「**11 維有效特徵**」
  - **這推翻了既有決策**：CLAUDE.md 原記「先證實 Group D 有貢獻，再補 `fear_greed`/`business_indicator`/`fed_rate` 資料源」——現在證實**不但沒貢獻還是負的**，那三個資料源**不用補了**，且 `INPUT_DIM` 應考慮 59 → 47
- **GRU 在 v2 基礎重訓完成 + GBDT walk-forward 完成（2026-08-02）**：
  - **GRU**（WSL RTX 3060、h64 only、5d only、**排除 7 個 Avail 旗標 → 59 維**才與其他模型同基礎）：test IC **+0.1018**、ICIR 0.785、NW t 11.73；分層 IC 小量 +0.1235 / 大量 +0.0893。分數已在 `result/scores/gru.parquet`
  - ⚠️ **GRU 的 decile spread Sharpe = 2.846 高得異常**（同窗第二名 v3_kg 只有 1.928）。**尚未查證**，可能是「序列模型在兩端區辨力更強」也可能是 panel 建構方式不同造成的問題（GRU 自己從 base matrix 切窗、不走 `load_xy`）→ **列為待查**
  - **GBDT WF**（12 個年度檔、4,552,371 列 / 2,724 天、2015-04~2026-06，固定 100 輪不逐 fold early-stop）：訊號 IC **+0.1265** > Ridge +0.1090
  - **★ 修正一個結構性結論**：11 年 WF 下 **GBDT 的最佳頻率是 5 日（+26.2%）而不是 20 日（+21.9%）**，Ridge 則是單調到 20 日（+20.7%）。→ **穩健的結論是「不要每日再平衡」（兩者一致：Ridge −12.1%、GBDT −5.0%），而不是「20 日最好」**。緩衝在 20 日下對兩者都只有微幅效果（Ridge 1.439→1.518、GBDT 1.265→1.296，k=1.0→2.0）
  - ⚠️ **我踩到自己的參數 bug**：`--years 2015 2026` 被 `nargs="*"` 解讀成「這兩年」不是範圍，第一輪只跑了 2015/2026。程序是**正常結束**不是被砍，補跑後才完整
- **組合建構基準版 v1.0 凍結 + 第一批結果（2026-08-01）**。規格 `docs/portfolio-construction-baseline-v1.md`（跑之前凍結）、結果 `docs/portfolio-lab-results-2026-08-01.md`、實作 `V6/experimental/portfolio_lab.py`（**純附加，不動 `portfolio_backtest`**）。使用者提供的《組合建構基本版檢核表》為骨架，依本專案實測**修正五處、補三處**（逐項對照見規格 §7）：
  - **動機的量化依據**：現行口徑換手 70–77%/次 × 每年 50.4 次 → **年化成本拖累 20.6%**，而 `v2_kg` 淨年化只有 +8.3% → **成本吃掉的比留下的還多**。與 D4「成本 ×2 讓 Ridge 掉 20.3pp」獨立吻合。→ **修這把尺不是「改做組合層」，是讓年化這個量尺本身可用**
  - **實作驗證通過**：新口徑 k=1.0/N=50/5日 算出 Ridge +10.8%、換手 76%，對照 F5 舊口徑 +11.3%/76%；且「Top50 隔 5 日留存 24.8%」→ 預期換手 75%，與觀察值吻合
  - **發現①緩衝有效且單調**：k 1.0→2.0，Ridge 換手 76%→64%、年化 **+10.8%→+16.4%**、Sharpe 0.678→0.971。**單一條規則值 +5.6pp**，而 IC 上量過最大的一次（GAT +0.0107）約值 1–2pp
  - **發現②每日再平衡是災難**（Ridge −29.7%~−10.0%、GBDT −38.2%~−20.4%），**20 日最好**（+21.9%）。**推翻使用者原本偏好每日的直覺**。根因不是成本假設而是訊號性質：**Top50 名單隔天只留存 47%（Ridge）/ 34.6%（GBDT）**，排名 lag1 自相關 0.79/0.70
  - **發現③Ridge 與 GBDT 的訊號住在完全相反的排名段**：N=10→224 時 Ridge Sharpe 1.126→0.524、GBDT 0.264→**0.551**。而 decile spread 是 GBDT **1.689** > Ridge 1.064（排序能力更強、但分散在整段）。**這就是四次「IC 與組合層不同調」的機制**，與 Step 3 的診斷一致。**連帶推論：固定用 Top50 當量尺會系統性偏袒「訊號集中在頭部」的模型——過去所有以 Top50 年化排序做的模型比較都帶這個偏誤**
  - **發現④（最重要也最不好看）**：**等權 eligible 宇宙年化 +15.2%**（新參照點，舊口徑從沒算過）。k=1.5/5日 下**幾乎所有 N 的超額都是負的**（Ridge −7.3%~+6.5%、GBDT −10.3%~−5.4%）→ **在現行交易頻率下扣掉成本後打不贏「等權買全市場」**；改成 20 日後 Ridge 才穩定勝出（+21.2% vs +15.2%）。decile spread 證明排序有效，問題在成本吃光 alpha
  - **發現⑤（探索性，不在凍結規格內）分數平滑**：排名前先做 w 日移動平均，換手砍到約一半、Sharpe 一致上升（Ridge k1.5/5日 換手 70%→34%、Sharpe 0.858→0.971；GBDT 72%→36%、0.432→0.646）。治的是根因（分數日間噪音）而非症狀
  - **⚠️ headline 設計失誤，已如實記錄不偷改**：我把「貼近實際操作」定成「每日再平衡」，但使用者說的是「**每天看 dashboard**」——那不等於每天換股。v1.1 修訂提案（改 20 日 + 新增平滑維度）寫在規格 §8，**待確認才生效**，且採用後所有模型要重跑、新舊不得混用
  - **五個模型已全部完成（2026-08-01 補）**：Ridge / GBDT 本機；**Mamba 三組用 WSL 的 RTX 3060 前向、各 7 分鐘，不需要 Colab、不需要重訓**（新檔 `V6/experimental/score_mamba_local.py`）。GRU 尚未做（`result/gru_5d.pt`，同樣 WSL 可跑）
  - **本機矩陣可用性有做決定性驗證**：F6 用的是 Colab 自 2005 建的矩陣，本機只有自 2011 建的 `baseline_cache_v2`（`macro_norm="ts"` 是 expanding z-score → Group D 數值不同，即 §4.4 的二階差異）。門檻**跑之前先定**（|平均差|<0.005 且逐日相關>0.95），實測三組 **差 −0.0012 / −0.0001 / +0.0001、逐日相關 0.9917 / 0.9977 / 0.9976 → 全過**。⚠️ 40 天小樣本先跑時 `no_gat` 差 −0.0069 **沒過門檻**，全量 582 天才收斂 → 小樣本的失敗是抽樣雜訊。**附帶好處：五個模型共用同一份特徵矩陣，跨模型比較比「Colab Mamba vs 本機 Ridge」更乾淨**
  - **⑥ 純排序能力 `v2_kg` 最強**：decile spread Sharpe **1.905** > GBDT 1.689 > no_gat 1.664 > old_kg 1.231 > Ridge 1.064
  - **⑦ Ridge 是唯一的異類**：訊號集中頭部（N=10 Sharpe 1.126 → N=224 0.524），**GBDT 與 Mamba 三組全都是分散型**（v2_kg 0.002 → 0.829）。→ **具體修正：過去 Ridge 在組合層一路贏，有相當一部分是 Top50 這把尺造成的**
  - **⑧ GAT 的價值在組合層被放大一個數量級**：N=50/k=1.5/**20 日**下 no_gat +12.2%/Sh 0.664、old_kg +16.7%/1.038、**v2_kg +26.1%/1.490** → C−A = **+13.9pp 年化、Sharpe 翻倍**，而訊號層只有 +0.0107（t=1.80 未達顯著）。**且 A<B<C 在四個 N（25/50/100/224）上全部成立**，不是單點。→ 支持使用者「GAT 有用」的判斷，也讓 Phase 4-A 的期望值可能要上修。⚠️ 但這是 240 格網格裡的觀察、非預先指定的檢定
  - **⑨「20 日只是因為台股大漲」的檢定（使用者 2026-08-01 提出）**：已補上 **10 日**（v1.1 提案 C，直接生效）並做兩個檢定。**前後半切分測不到這件事**——切點 2025-03-24，等權宇宙前半 +14.2% / 後半 +15.9%，**兩半都是多頭**。改依「等權宇宙過去 20 日累積報酬正負」切上升段（378 天）/ 下跌段（204 天）才問得出來：**低頻在下跌段也比較好**（Ridge 3日 −26.1% → 20日 **−15.8%**；`v2_kg` 3日 −28.9% → 20日 **−17.3%**）→ 低頻的好處主要來自省成本，而成本兩種行情都要付 → **部分推翻原假設**（例外：GBDT 與 `no_gat` 在下跌段不單調）
  - **⑩ 本輪最有意思的發現：`v2_kg` 的優勢來自下跌段保護，不是多頭助攻**。20 日下 `no_gat` 上升段 +57.9% / 下跌段 **−40.4%**，`v2_kg` 上升段 +58.3% / 下跌段 **−17.3%**——**上升段幾乎相同，下跌段差 23 個百分點**。→ GAT 帶進來的關係資訊，作用是**在下跌段避開一起崩的股票**，不是在多頭挑到更會漲的。與 MDD（−26.9% vs −33.4%）一致
  - **⑪ 前後半切分意外很有鑑別度**：20 日 N=50 的 Sharpe 前半→後半，Ridge **2.066→0.736**、GBDT **2.158→0.271**（漂亮數字幾乎全來自前半）、`old_kg` 1.387→0.908、**`v2_kg` 1.789→1.402（唯一又高又穩）**
  - **⚠️ 選擇偏誤要記住**：240 格 × 5 模型 = 1,200 個組合，最好的那格必然帶偏誤。**可信度排序：系統性型態（頻率單調、線性 vs 非線性的 N 依賴相反、A<B<C 一致）＞ 任何單一格子**。最大的外部效度風險是「20 日最好」可能只是 2024–2026 這段行情的特性 → 下一步應做子期間切分
- **F6 GAT 三組消融全部完成（2026-08-01）**。完整紀錄與逐 epoch 數字見 `docs/f6-training-log-and-readout.md` §2 / §2.4：
  - **結果**：`no_gat` +0.0884（ICIR 0.733、年化 +10.7%、MDD −33.4%、換手 77%）／`old_kg` +0.0939（0.754、+6.3%、−26.2%、70%）／**`v2_kg` +0.0991（0.844、+8.3%、−26.9%、71%）**。三組參數 1,394,301 / 1,659,005 / 1,659,005
  - **最關鍵的數字腳本沒印出來**（它只印對 `no_gat` 的比較）：**`C − B = +0.0052、配對 NW t = +5.17`、67.9% 的日子為正**。B vs C 是**唯一乾淨的配對**（同架構、同參數量、同 seed、同資料順序——實測第一個 batch 都是 1,668 支，A 是 1,558）
  - **t 值差異的原因已量化**：`std(C−B)=0.0140` vs `std(C−A)=0.0791`，**差 5.6 倍**。B−A 的 t 只有 0.91、C−A 只有 1.80，**不是效應小，是 A 的 RNG 不同讓配對差的雜訊淹掉訊號**——正是判讀清單 §4.3 第 2 點事前預測的情況
  - **穩健性全過**：前半 Δ=+0.0035（t=3.18）／後半 +0.0068（t=4.20）、去極端 5 天後 +0.0048、10d horizon 同向（C +0.1054 > B +0.0968 > A +0.0844）。且 **`v2_kg` 全面支配 `old_kg`**（IC/ICIR/正比例/年化/Sharpe 全贏）→ `kg_builder_v2` 驗收通過
  - **`random_kg`（第四組）不需要跑**：參數量干擾在 C−B 上被完全排除（兩組都是 1,659,005）
  - **實驗健康檢查全過**：峰值 epoch A ep5 / B ep3 / C ep3（都遠早於 ep9–10 → **不需三組重跑**）、三組 train/val 皆 2,661/582、v2 圖摘要印出 2,245 節點/32,083 邊（沒載錯圖）
  - **§1 對照表的一句話因此作廢**：原本寫「1.4M 參數的 Mamba 打不贏 300 維的 Ridge」——**補上一張修好的圖之後 `v2_kg` +0.0991 已超越 Ridge +0.0899、逼近 GBDT +0.1036**。但**組合層仍輸**（年化 +8.3% vs Ridge +11.3%）＝ **IC 與組合層不同調的第四次**
  - **未處理的限制**：C−B 的 t=5.17 是跨「日」不是跨「seed」，單一 seed = 一次抽樣
- **F5 的 −0.0079 拆開了 + Group D 消融腳本備妥（2026-08-01，Colab 訓練期間的並行工作）**。完整紀錄在 `docs/feature-protocol-v2.md` **§9.6**（新增）與 `docs/f6-training-log-and-readout.md` §6.1/§6.2：
  - **主因不是原本猜的那兩項**。原記載「剩約 −0.0065 來自宇宙過濾 ETF＋興櫃與外資持股回補」漏掉了最大的一項——查檔期才發現 **R0 的矩陣建於 2026-07-12，而 `prices_raw` 切換成全歷史除權息還原是 07-29**，中間 raw 幾乎被翻修了一遍。拆解結果：**資料修復整包 −0.0053（t=−2.22，佔 71%）** > fund_v2 −0.0014（19%） > 宇宙過濾＋起點 ≈ −0.0008（10%）
  - **我的第二個假設也被自己的實驗推翻**：原以為 −0.0053 來自除權息還原（未還原時除息日的假跌幅可預測 → 虛假 IC）。**兩個檢查都不支持**：① 580 個測試日按「前瞻 5 日窗內除權息事件數」分組，`corr(事件數, Δ) = +0.059`，方向相反且不單調 ② 逐欄比對舊/新矩陣（2024-01、39,754 共同列、跨列 Spearman）顯示**價格類 ρ 全部 ≥0.985**（Close/Open/High/Low/MA_20/MA_60/ATR_14）——**還原是逐股的單調重新縮放，橫斷面排名幾乎不動**，事後想想是應該預期到的
  - **真正的主因是 Group B 籌碼欄的正確性修正**：`Day_Trade_Volume` 相異值 **1 → 30,906**（整維從死值變活值）、`Short_Sale` ρ=**0.78** / `Short_Cover` ρ=**0.86**（券賣/券買標反修正，33.2% 的列）、`Holdings_Large_Change` 0.91、`Foreign_Holding_Pct` 0.95。Group C 的 `Gross_Margin`(0.05)/`ROE`(0.19) ρ 雖低但相異值只有 10–14 個、近乎死值。**Group D 那 8 個變動欄對 Ridge 零影響**（每日橫斷面常數），但順帶獨立確認 `FED_Rate` 相異值 22 → **1** ＝ F6 矩陣的死維。→ **與 purge、Q4 延遲同類：誠實化的代價，不是「這個改動不好」**
  - **20d 完全沒有這個代價**（`R0c − R0 = +0.0010`，t=+0.27）→ 資料修復的 IC 代價只出現在 5d
  - **程式改動**：`baseline_common.py` 把宇宙規則從 `PROTOCOL_VERSION` 抽成獨立的 `PROTOCOL["UNIVERSE"]` 鍵（未設變體時逐位元 no-op，有 assert 保證），新增變體 `v1like`（v1 規格跑在新資料上）與 `v1univ`（v2 規格 + v1 宇宙）；`f5_r_series.py` 加 R0c/R0d 三筆配對比較 + `decomposition_check()`（三段相加 vs 整包差，對不上直接標 ❌）。**兩個變體都刻意不覆寫 `AVAILABILITY_FLAGS`**——`_DIM` 在 import 期就被 `patch_config_67d()` 綁成 66，關掉會讓 `build_features` 不產生 `Avail_*` 欄 → KeyError，而且是在矩陣建了十幾分鐘之後才炸；旗標照建、跑階時 `--flags off` 遮掉後扁平維度 307−7 = **300，與 v1 完全相同**
  - **R0 重跑完全重現**（5d +0.0977 / 20d +0.0989，2.1 分）並補上原本缺的逐日 IC → 之後對 R0 的比較才有 t 值
  - **`groupd_ablation.py` 已寫好並測過**（診斷／mask 歸零+還原／控制組匯入的接受與拒絕路徑／對照表四種判定分支）。**設計改動：用 mask 不砍維度**——砍維度會改 `GROUP_DIMS` → sub_dim 重新分配、參數量變、RNG 分岔，完全重演 GAT 消融 A vs B 的三個干擾項。**省一半 GPU**：`use_gat=False` 讓 `with_macro` 與已完成的 `no_gat` 設定逐項相同 → 預設從 `kg_ablation_result.json` 匯入控制組（十項核對含 epochs，不符就拒絕），只需訓練 `no_macro` 一組約 3.6 小時
  - **⚠️ 操作教訓：背景任務跑不完這種長 build**。R0d 三次都在 `clean_and_scale` 的逐日 winsorize+z-score 階段被外部終止（存活 22 / 14 / 10.5 分遞減、**無 Python traceback、無殘留程序、無系統錯誤事件**），但同一機制下 R0c 卻順利跑完 38 分鐘 → 真因診斷不出來，**長 build 改在使用者自己的終端機跑**。每次被回報 killed 都先確認子孫程序真的死了才啟接力（F5 教訓），三次都確認過、chunk 完整不需重建
  - **R0d 決定不跑（使用者拍板）**：它要確認的是那個 ≈ −0.0008，比實務門檻 0.009 低一個數量級，不論結果如何都不改變任何規格決策，跑它只是完整性檢查 → 殘差在文件中誠實標為「相減得出、未獨立驗證」。Group D 消融也排到 GAT 三組跑完之後再說
- **F6 開跑：GAT 三組消融（2026-07-31~08-01，進行中）**。**完整紀錄、每一輪的設定/數字/判讀清單、以及三個 Colab 坑，全部寫在 `docs/f6-training-log-and-readout.md`**——接手前先讀那份，這裡只記重點：
  - **規格三層驗證全過**：59 特徵 + 6 meta（`Alpha_10d` 是 V6.2 加的，meta 是 6 不是 5）、`Avail_*` 0 個、宇宙 2,479→**2,245** 支。**第三層「內容指紋」是唯一擋得住『參數設對但吃到舊快取』的一層**：`Book_Value` 相異值 1（舊財報）→ **5,033,154**、`Gross_Margin` 524 → 7,002,546
  - **`no_gat` 完成**：峰值 5d IC **+0.0884 @ep5**、ICIR 0.733、年化 +10.7%、參數 1,394,301。**重評 mean IC 與訓練迴圈 ep5 的 val IC 完全相同 → checkpoint 重載+逐日 IC 那段程式正確**
  - **同一批 582 天的橫向對照**（F6 val 窗刻意對齊 F5 test 窗）：GBDT +0.1036 > Ridge +0.0899 > **Mamba 單尺度 +0.0884**。1.4M 參數的 Mamba 打不贏 300 維 Ridge，與方向二 Step 4 的收斂結論一致；但 GAT 兩組未出，先不定論
  - **多尺度主模型（Cell 4）已中止、數字不可用**：`epochs=100` 讓 OneCycleLR 暖身長達 15 個 epoch → **模型在暖身期就過擬合，從頭到尾沒進入退火階段**（ep7 峰值時 LR 只有 max 的 47% 且還在爬）。根因是 **`epochs` 一參數身兼兩職**（最多跑幾輪 vs OneCycle 排程長度），有 early stopping 時不該綁一起。要重跑需 `epochs=15~20`
  - **白賺：scale gate 第三次塌到 Mid**（Short 0.047 / **Mid 0.890** / Long 0.063）。Phase 0 舊資料是 0.004/0.80/0.20，**同形狀在全新資料基礎上重現**。加上 Phase 1 deep supervision，已有三條獨立證據：5d 目標下多尺度實質退化成單尺度
  - **`FED_Rate` 是死維**：`fed_rate.parquet` 只有 **8 列 / 1 個相異日期（2004-01-01）**。59 維中實質 58 維有訊息，Group D 是 **11 活 + 1 死** → 消融結論必須寫成「11 維有效特徵的貢獻」。依既有決策「先證實有貢獻再補來源」，**暫不修**
  - **三個 Colab 坑（都踩過）**：① **模組快取**——`git pull` 後 Python 仍用 `sys.modules` 舊版、**完全不報錯**，實際導致 `train_start` 沒生效、訓練集變成 2005 起的 4,651 天 ② 特徵矩陣快取不帶規格資訊，只印一行 "Loading cached..." ③ `train_short_model` 改 `T.TARGET_COLS` 沒還原 → 接著跑 Cell 4 得到 `IndexError: index 2 is out of bounds`，而錯誤指向 `trainer.py`、與真因差很遠（已於 `d71b8ac` 用 try/finally 修好）
  - **隔離的誠實補充**：固定 seed 只保證 **B vs C（old_kg vs v2_kg）逐參數同初始化、同資料順序**。**A（no_gat）因少建 GAT 層而少消耗 RNG → head 初始化與 DataLoader 打亂順序都不同**（實測：A 第一個 batch 1,558 支、B 是 1,668 支）。A vs B 另有參數量 +264,704 的干擾 → **A vs B 要保守讀，B vs C 才乾淨**
  - **⚠️ 更正記載**：`macro_raw` 實際到 **2026-07-23**（5,620 列、無斷層），不是 CLAUDE.md 原記的 2026-04-24。但 `run_daily_update` **不含 macro**（已逐項列出它呼叫的 17 個 fetcher）→ 會再度停更，而 scanner regime 閘門有 10 天新鮮度檢查
- **F5 R-series 完成、特徵工程層規格正式凍結（2026-07-30）**：跑階器 `V6/experimental/f5_r_series.py`（新）+ `baseline_common.py` 加 `MM_VARIANT`（`nofund`/`neuind`/`neuindmc` 各自獨立快取，未設時為 no-op 並 assert）。完整數字 `V6/experimental/result/f5_r_series_result.json`、判讀寫進 `docs/feature-protocol-v2.md` §9。共 **14 級**（8 級主梯 + 4 級子修正拆解 + 完整性檢查 + 3 級 GBDT 裁決）
  - **原計畫的 R0→R5 直線疊加行不通**：v1→v2 是一次跳四個變因（`fundamentals_v2`／宇宙過濾 ETF＋興櫃／矩陣起點／外資持股 9xxx 回補），只靠兩份快取量不出單一變因 → 改成**全部在 v2 內部爬梯子**，只有真正寫進矩陣值的變因才另建快取
  - **誠實總結論：沒有任何一項達到 +0.009 的實務門檻，唯一顯著的效應是負的。** 起點 2012→2013 **+0.0001**（t=0.46，免費，照資料品質理由採用）、旗標 **+0.0000**、中性化 +0.0025（t=1.06 不顯著）。F5 真正的價值不在 IC 上，而在**把不該算的 0.003 拿掉**
  - **`fundamentals_v2` 的負 Δ 完全來自 look-ahead 移除**（使用者要求拆解後量出）：四個子修正裡 **(a) Q4/年報 `available_from` 45→90 天 = −0.0016（t=−4.06，全 F5 最顯著）**，(b) 三維取真值 +0.0004、(c) EPS_Surprise 季頻 −0.0000、(d) PER/PBR 自算 +0.0000。**乾淨的隔離點是 `EPS`**——它在舊路徑本來就正常（`Book_Value` 在舊路徑只有 **1 個相異值**），故 `EPS` 12.2% 的差異純粹來自 Q4 延遲；把它換回舊時序＝把 look-ahead 放回去，IC 從 +0.0899 升到 **+0.0916**。→ **與 purge 同列「正確性修正、一律採用、不套效益判讀表」**，兩者合計誠實化代價 5d **−0.0028**
  - **「換欄」技巧**（省掉 4 次 27 分鐘的重建）：`fundamentals_v2` 只有一個布林旗標卻 gate 四個修正。但 v2 與 `nofund` 兩份矩陣**可按 (Date, stock_id) 對齊**（v2 是子集、只少 7 列 NaN）→ 拿 v2、只把某組欄位**含其 lag1/5/20** 換成 nofund 的值即等價（只換 base 會讓同一特徵有兩種來源混在一起；`ROLL_CORE` 不含 Group C 故 rolling 不必動）。**完整性檢查通過**：四項一起換回 = +0.0913/+0.0972，與整份 nofund 矩陣的 R3 四位小數完全相同
  - **`INPUT_DIM` 定 59、旗標不採用**（使用者拍板）：**我原本「Ridge 定不了、GBDT 會替旗標平反」的假設被自己的實驗推翻**——GBDT 5d 無效應、**20d 顯著扣 0.0060（t=−2.93）**、7 個旗標 **gain 佔比合計 0.32%** 而 307 維均分是 0.326%（七維只拿到約一維的份量，`Avail_Financials` 在 5d 是 0.000%）＝「有能力用交互作用但不用」。全量下 7 個旗標**都不是死常數**（推翻 40 支小樣本判定，`Avail_Valuation` 先前已因全量 mean 0.9966 移除 → 67→66）。**非顯而易見的原因：起點移到 2013 與旗標功能高度重疊**——旗標要標的「該來源還不存在」期間（Daytrade 2014、Holdings/ForeignShare 2018）大多已被砍掉，這也對得上 `Avail_Daytrade` 是七個裡 gain 最高的。原本最主要動機（9xxx 語意翻轉）已用回補**在源頭修好**。**未採用「只留 gain 高的那幾個」**——那是依 test 集挑特徵＝test-set selection
  - **`NEUTRALIZE` 定 "none"、留到 F6**（使用者拍板）：四個測量（2 模型 × 2 horizon）**方向全正**，最大 +0.0068；GBDT 5d 的 `Δ_trim5=+0.0038` 比原始 Δ 更大（去極端日反而增強 → 不是被少數幾天帶起來）；G-R4a 是全場最好的 ICIR 0.950 / NW t 14.50。**但 t 值 0.76–1.42 過不了跑前定好的 |t|≥2**，且成本不是零——中性化在推論路徑多一個必須同步的步驟，訓練/推論不一致正是踩過兩次的坑（`macro_norm`、`fundamentals_v2`）。`industry_mktcap` 未量
  - **方法紀律四條**（都寫進 `f5_r_series.py` 檔頭）：① 存 582 天**逐日 IC**、級間用**配對** NW t（Δ=0.002 時肉眼判不了）② **判讀規則跑之前先定死**（看到數字才選規則就沒意義）③ 數值全部 import 自 `baseline_ridge_lasso`，不另寫一套 Ridge（否則級間差異混進實作差異）④ **GBDT 刻意不重掃網格**，沿用 Step 3 的 leaves=127/min_leaf=2000（各自掃網格會讓超參數吸收掉特徵差異）
  - **兩個原設計錯誤（使用者質疑後才修，兩者都不會報錯）**：① R3「fund_v2 關但旗標開」不自洽——`Avail_Financials` 在 `_merge_fundamentals(fundamentals_v2=...)` **之後**才算、判準是 `notna()`，而舊路徑那三欄是**死常數不是 NaN** → 實測該旗標在 nofund 矩陣裡是**死常數 1.0**（其餘 6 個逐位元不變）；改成兩側都關旗標、對照 R1 ② 表格漏標 horizon（實際每級都同時算 5d/20d，閘門 5d、20d 佐證，方向相反時印警告不自動採納）
  - **⚠️ 操作教訓**：背景任務被回報 `killed` 後**它的子孫其實還活著**，我另啟接力腳本 → 兩個 `--build` 同時寫同一個輸出檔。已終止並刪半成品重建；`baseline_cache_v2` 與共用 chunk 因唯讀而未受影響（設計擋住了）、結果 JSON 完整。**啟接力前要先確認沒有殘留 process，不能相信通知**
  - **v1→v2 最大的一項 R0b−R0 = −0.0079 不可歸因到單一變因**（fund_v2 佔 −0.0014、起點 +0.0001，剩約 −0.0065 來自宇宙過濾與外資持股回補），與 D1 分層診斷一致（小型/低流動股墊高 IC），是誠實方向的下降。要拆開需再建一份矩陣（v2 規格＋v1 宇宙規則），未做
- **Group D 全數為死值的實測確認 + 集保改 TDCC 直連（2026-07-29）**：
  - **⚠️ 重要事實：V6.1 下 Group D 全部 12 維都是 0**（不只 `fear_greed`）。實測 `clean_and_scale` 後 `TWII_Return`/`SPX_Return`/`VIX`/`TNX`/`Gold_Return`/`Oil_Return`/`USD_TWD`/`Futures_OI_Foreign`/`Options_PC_Ratio`/`Fear_Greed`/`Business_Signal`/`FED_Rate` **std 全為 0.000000、absmax 全為 0.000000**。原始值其實有訊息（`Fear_Greed` 5~78 共 67 個相異值、`Business_Signal` 23~40），是被 `macro_norm="cross"` 消滅的（macro 同日對所有股票同值 → 橫斷面 z-score → std=0）。**這代表 V6.1 是在整組 Group D 都死掉的狀態下達到 IC ~0.08 的——目前沒有任何證據顯示這 12 維有用**
  - **決策：暫不追 `fear_greed`/`business_indicator` 來源**。正確順序是等 V6.2 改 `macro_norm="ts"` 後先做 Group D 消融，證實有貢獻再補來源；否則是為未經證實的特徵接資料源。（同理，先前做的 TAIFEX 期貨/選擇權對 V6.1 也是零影響，但那是為 V6.2 鋪路且順帶擺脫 FinMind 付費牆）
  - **`holdings` 改 TDCC 直連 ✅**：FinMind `TaiwanStockHoldingSharesPer` 已需付費層。新增 `fetch_holdings_tdcc_direct` + `_catch_up_holdings`，接進 `run_daily_update`。**健檢警告 3 → 2**
  - **順帶修掉一個死常數**：舊聚合把**分級 17 =「合計」**（恆為 100.00%）也加進總和 → `總計 ≈ 200`、`Whale = 200 − 散戶 ≈ 199.67` 被 `clip(0,100)` 壓成 100 → **`Whale_Hold_Ratio` 在全部 848,269 列都是 100.0（只有 2 個相異值）**。改用分級 17 當總計後 median 99.580%、僅 147/2,956 支飽和。**對下游無影響也無接縫問題**：`_merge_holdings` 早就繞過該欄改用 `Holdings_Large_Pct = 1 − Retail/100`，而 `Retail` 一直是對的，修好的 Whale 恰等於該式 ×100
  - **已知限制**：TDCC 開放資料**只有最新一週**（`date`/`DATE`/`d` 三種寫法實測都回同一週）；查詢頁雖有 51 週歷史但**逐股查詢**（2,000 支 × 11 週 = 22,000 次請求，不成比例）→ 2026-05-08 ~ 07-24 的缺口補不回來，只能從現在起逐週累積。⚠️ `Holdings_Large_Change` 在接續的第一週會出現一次「11 週變化壓縮成 1 週」的假跳動
  - **另記錄**：分級 15（>1,000,000 股 ≈ >1000 張）才是真正的「大戶持股比例」（2330 為 84.70%），比「100 − 散戶」有意義得多；但歷史 parquet 只存 Whale/Retail 兩欄、無分級明細，無法回算，故未新增
- **股利分派改 MOPS 直連（2026-07-29）**：FinMind 的 `TaiwanStockDividend` 是**逐股查詢**（~2,000 次）會撞爆免費層 600 次/日額度；MOPS `t187ap45_L`/`_O` 是**兩個 CSV、兩次請求**且含上市與上櫃。新增 `fetch_dividends_mops_direct` + `_catch_up_dividends`，接進 `run_daily_update`，淨增 2,023 列 → 最新 2026-07-27。**健檢警告 4 → 3**（整輪工作起點是 14）
  - **踩到「拆開 vs 合併」的口徑陷阱**：MOPS 把現金股利**依來源拆三欄**（盈餘分配／法定盈餘公積／**資本公積**），而 FinMind 是**總額全放 `CashEarningsDistribution`**、`CashStatutorySurplus` 恆為 0。照欄名直譯的話，**全部由資本公積配發**的公司會變成 0 元股利——1101 台泥就是（實際 0.8 元），而且不會有任何錯誤訊息。改為三欄加總後，**量級交叉驗證 MOPS/FinMind 比值 median = 1.000**（82.8% 落在 0.5~2.0；不同年度本就不會恰為 1）
  - **`date` 欄語意變化（刻意，已記錄）**：既有 `date` 實測是**除息交易日 + 6 天**（median 6、p10/p90 = 6/8）＝股利發放後才出現，而真正公告日早約 22 天；`_merge_dividend_feature` 的 docstring 寫的卻是「available once announced (before ex-date)」——**意圖與實作本來就不一致**。MOPS 只有董事會分派日（＝真正公告日），故新資料會比歷史早約 28 天被特徵看到。**兩者都不含未來資訊**，差別只在時效，新做法反而符合原意
  - **新列的除權息日期欄為 NA**（MOPS 沒有）：無下游依賴——特徵只用 `date` 與現金股利欄，B-3 的因子已改用交易所官方 `ex_rights_raw`，不再從 `dividend_raw` 反推
- **期貨/選擇權三大法人改 TAIFEX 直連（2026-07-29）**：FinMind 免費層對這兩個 dataset 回 400 "Your level is register"，不訂閱就**永遠拿不到**；TAIFEX 端點公開且是**不同主機**（不與 TWSE/TPEX 搶速率、不受 FinMind 額度影響）。新增 `fetch_futures_institutional_direct`／`fetch_options_institutional_direct`／`_catch_up_taifex`，接進 `run_daily_update`。回補 2026-05-05 → 07-29（期貨 +177 列、選擇權 +354 列）。**健檢警告 6 → 4**
  - **踩到四個對映陷阱**：① 成交金額欄實際叫「多方**交易**契約金額(千元)」，我少寫「交易」二字且子字串比對方向相反 → 靜默留 NaN ② **期貨與選擇權用詞不同**（期貨「多方/空方」、選擇權「買方/賣方」），初版只寫期貨那組，選擇權 8 個數值欄全落空 ③ 編碼是 **MS950（Big5）**，`apparent_encoding` 猜測在中文上不可靠，改以 header 宣告為準 ④ 非交易日回 **HTML 錯誤頁**而非空 CSV，需明確偵測
  - **兩個必須正規化的口徑差異**（不處理會讓特徵在接續日無聲跳階）：TAIFEX 回**所有商品**（單日 69 列）而既有 parquet 只有 TX/TXO（每日 3 列）→ `Futures_OI_Foreign` 是跨商品加總，不過濾會跳一個量級；法人別 TAIFEX 是「外資及陸資」、既有是「外資」
  - **接縫連續性驗證通過**：`Futures_OI_Foreign` 接縫當日變動 2,280 vs 全期 `|日變動|` median 1,877（同量級）；`Options_PC_Ratio` 接縫前 median 1.087 vs 接縫後 1.073。每日列數 futures 3 / options 6 與既有一致
  - **歷史深度限制**：TAIFEX 這兩個端點只有約 2.5 年（2023 下半年起），更早仍靠既有 parquet（FinMind 抓的 2018-06 起），兩者在此銜接。區間查詢一次一個月（1,242 列 / 18 個交易日），比逐日省 20 倍請求
- **極端報酬歸因 + 健檢日曆缺口 + B-5 stock_info（2026-07-29）**：
  - **極端報酬逐筆歸因 ✅**（`V6/scripts/classify_extreme_returns.py`）：台股有 ±10% 漲跌幅限制，故 `|單日報酬|>40%` 制度上不可能是真實單日波動。471 筆中 **96.2%（453 筆）是交易中斷後復牌**（停牌期間跨數週）、僅 **17 筆無法解釋**（佔全體 8.28M 列的 0.0002%）。**還原自我檢查抓到 4 筆現金增資的假報酬**：2429 於 2024-07-02 官方 `ref_price 29.15` 但原始價 `38.9 → 42.75` 是 +9.9% 漲停——市場根本沒重設到參考價，我們套因子反而製造 +46.66% 假報酬。系統性檢查：25,998 個事件中僅 **4 筆（0.015%）**「原始正常但還原後異常」且全是 `權`；整體原始 `|報酬|` median 0.0435 → 還原後 **0.0138**（改善 3 倍）。**現金增資的參考價是理論值、認購是選擇性的**，市場不一定跟隨 → 列為已知限制不追（要正確處理需區分「強制重設」與「理論參考價」）
  - **健檢加交易日曆缺口偵測 ✅**：原本只看「最新日期落後幾天」，中間漏掉的交易日完全看不出來（prices_raw 曾缺 2026-04-27/04-28 兩天，是回補 margin 時意外發現的）。改用 `np.busday_count` 掃近一年、>5 個工作日的空檔告警。現況「無 >5 工作日空檔 ✓」
  - **B-5 `stock_info` ✅，而且抓到一個比預期嚴重的 bug**：該檔是**多次快照的累積**（4,097 列 / 3,086 支，`date` 有 2024-12-30、2026-05-03 等多批）。`_emerging_ids()` 未去重就篩 `type=="emerging"`，命中舊快照 → **把已轉上市櫃的公司整支排除**，實測 31 支、其中 **30 支仍在交易**（1563 巧新、2072 世紀風電、2248 華勝-KY 等 emerging → twse）。修正後宇宙 1,899 → **1,929 支**。**修法不是在源頭刪重複列**——那些列帶有市場別變遷史（PIT 相關事實），改為提供單一權威入口 `hygiene.load_stock_info(latest_only=True)`，`_build_kg.py` 與 `v6_colab_training.py` 同步改用（列膨脹 3,386 → 2,479，−36.6%）
  - **FinMind 額度耗盡與 IP 封鎖 ⚠️**：回補期間發現 `_finmind_fetch` 只對 429/503 重試，**402（額度用盡）落進 `logger.debug` 回 `None`**＝與「這支沒新資料」無法區分 → 回補連跑六輪、900 支、每輪淨增 0 而日誌正常。更糟的是接著觸發 **`403 ip banned`**（短時間打 ~1,000 次請求）——**請求速率本身就會觸發封鎖**，不是把每日 600 次用完才擋。已修：402/403 拋 `FinMindQuotaExceeded` 立即中止；每日滾動批量降到 80+40（約額度 20%）、間隔 1 秒。**對每日推論無影響**（主要源皆已直連）。實際進度：`revenue` 2026-07=551 支 / 2026-04=1,373 支；`financials` 2026-03=16 支 / 2025-12=1,925 支。封鎖解除後建議**不再跑 bulk 回補**，交給每日滾動慢慢追
- **`prices_raw` 切換 + 每日更新結構性修復（2026-07-29）**：
  - **切換完成 ✅**：`V6/scripts/switch_to_adjusted_prices.py`。**原則「只改值、不改型別」**——新檔 `Date` 是 `timestamp[ns]`、production 是 `large_string`，照抄過去會讓每日更新的字串日期與之混型、`drop_duplicates` 靜默失效（＝重製問題 1 的 10,591 列重複）。故寫入前轉回字串、不帶 `src` 欄。切換後 schema 逐欄型別完全相同。備份 `prices_raw_backup_before_adj_20260729.parquet`
  - **切換後驗證**：健檢重複 0／`Close<=0` 0；`build_features` + `clean_and_scale` 實跑通過（48,719 列 × 62 欄、無 inf、報酬 std 0.0197）；唯 4 個含 NaN 欄是 `Alpha_5d/10d/20d/60d` **標籤欄**（尾端本來就 NaN、新舊一致，非回歸）
  - **修健檢的量測分母**：法人覆蓋率 96.1% → 89.5% 是**假摔**（分母多了 133 支 ETF，交易所本來就不出 ETF 的個股法人明細）。改用推論端同一套 `filter_tradable_universe` 後為 1,899 支 / **96.2%**。不修會留下永遠不消的假警報——**長期假警報會訓練人忽略警報，比沒有警報更危險**
  - **commit `0cb262f`**：`fetcher.py`/`hygiene.py`/`feature_engineer.py`/`run_daily_inference.py` + 16 支腳本 + 4 份文件，7,460 行。已排除本機 56 維 `config.py`
  - **`revenue`/`financials` 接進每日更新 ✅**：這兩個源原本只在 `force_rebuild=True` 時抓、平時走快取分支＝**永遠不會更新**（停更 118/119 天）。**FinMind 免費層的限制是「形狀」不是速率**——實測不帶 `data_id`（全市場）回 HTTP 400 "Your level is register"、帶 `data_id`（單股）回 200，所以只能逐股查 ~1,900 次。改為**滾動逐股補齊**（每天最舊的 120/60 支，約 16 天輪一輪，每天多花 1–2 分鐘）+ `V6/scripts/backfill_monthly_finmind.py` 一次追平。**踩到「補最舊 N 筆」的陷阱**：初版淨增 0 列，因為排序把已下市股票排最前（最舊停在 2002-02-01、永遠不會更新、每輪耗光額度）→ 加 `_live_universe()` 過濾（2,311 → 1,924 支）後單輪淨增 90 列
  - **確認 FinMind 免費層已擋掉的**：`TaiwanFuturesInstitutionalInvestors`、`TaiwanStockHoldingSharesPer` 皆回 "Your level is register" → `futures_inst`/`options_inst` 需改 TAIFEX 直連、`holdings` 需 TDCC（**但 TDCC 開放資料忽略日期參數、只回最新一週**，歷史無來源）
- **B-1~B-5 五項決策落地（2026-07-28，使用者逐項拍板後執行）**：
  - **B-2 興櫃全歷史一致排除 ✅**：`hygiene.py` 新增 `_emerging_ids()` + `filter_tradable_universe(exclude_emerging=True)`，`run_daily_inference.py` 呼叫端同步。**2026-05-22→05-25 的宇宙斷層由 −344 支（−14.9%）降到 −7 支（−0.4%）**，月底檔數從「5月2314→6月1976」跳水變成穩定 1933→1930；全歷史剔除 401 支 / 450,701 列。已知限制：`stock_info` 是現況快照非 PIT，「曾是興櫃後轉上市櫃」者的興櫃期資料仍在（明確揭露，不假裝解決）
  - **B-1 PER/PBR 自行推算 ✅**（`fundamentals_v2` 旗標下）：`PER = Close / EPS_TTM`、`PBR = market_value / Book_Value`（用市值÷權益而非股價÷每股淨值，避開股數換算 → 對減資/增資免疫）。**`EPS_TTM` 在季頻算好再走既有 as-of join**，每季受各自 `available_from` 保護（使用者特別指定；若在日頻 rolling 會重演 `EPS_Surprise` 那個 bug）。**踩到兩個坑**：① 初版只 `combine_first` 填 NaN 幾乎沒效果（+308 列），因為那 865 支不是缺值而是被 ffill **凍結**在 04-24 舊值 → 改三層優先序「官方觀測 > 自算 > 凍結 ffill」（為此在 `_merge_per_pbr` 加 `PER__obs`/`PBR__obs` 暫時欄位），實測 PBR 凍結列 96.6%、PER 56.0% 被取代 ② 我推測「交叉驗證分母被 ffill 污染」，**改成只用官方觀測列後仍是 0.9494 → 自己的量測推翻自己的推論**，那是真實系統差異。因自算值與官方值共存於同一橫斷面，5% 水位差會製造假排名 → 加**當日橫斷面校準**（只用當日資訊、無 look-ahead；係數 median PER 0.9512 / PBR 1.0013）。PBR 交叉驗證 median 1.0014、±10% 內 92.2%
  - **B-4 `Day_Trade_Volume` ✅**：使用者「V6.1 停了沒關係」→ 維持真值，**不需程式改動**
  - **B-5 `stock_info` 去重**：決策為「在原始表層級修」但不急，已記待辦。使用者給的通則值得記：「**在源頭修 vs 靠每個消費端各自防禦**，源頭修一次比較省事、也比較不會有人漏掉」
  - **回歸驗收**：`fundamentals_v2=False` 與 git HEAD **62 欄逐位元最大絕對差 `0.000e+00`** → 線上 V6.1 零影響
- **B-3 全歷史還原重建（2026-07-28 啟動，進行中）**：使用者決策「回頭重建，順便解掉減資無標記（`adj_factor>1` 直接是減資清單，一魚兩吃）」。**我在執行前發現使用者決策時還不知道的事實**：`ex_rights_raw` 只涵蓋上市——宇宙 1,942 支中上櫃 823 支只有 **2 支**有官方紀錄，只用它重建會變成「上市已還原、上櫃未還原」，比現況（yfinance 兩邊一致還原）**更糟**。使用者追加決策：先驗證上櫃來源、過了才重抓；2007-07 前上櫃維持現況並標記。
  - **步驟 1 驗證 ✅ 通過**（`V6/scripts/validate_dividend_formula.py`）：關鍵設計＝TWT49U 同時給「前收盤價」與「參考價」，故可**純公式比對、完全不碰 prices_raw**，不受「歷史已還原/近期未還原」混源影響（公式裡現金股利是絕對金額、不具尺度不變性，用被縮放的價格反推會失真）。配對 14,429 筆（95.0%）。**按 kind 拆解才看得出真相**：息 99.69%、權 45.95%、權息 72.04% → 抓出兩個欄位語意錯誤：**`CashIncreaseSubscriptionRate` 不是配股率**（純現金增資事件反解後只有 12.3% 一致，正確是 `ci_shares/total_shares` 的 72.0%）、股票股利要 **÷10**（對照組不除 10 時 權 1.09%/權息 0.04%，決定性排除）。**誤差幅度才是決策依據**：息 median 0.000%、權 0.025%、權息 0.024%，>1% 者僅 2.04%；典型還原修正量本身 4.5% → 平均抓到 **99.5%** 的修正量，對照「不還原＝除息日 4.5% 假跌幅」判定可用
  - **步驟 2 重抓執行中**：`V6/scripts/refetch_raw_prices_full.py`（5,301 交易日、逐年落檔可續跑、19:15–20:15 自動避開每日推論）。**端點深度實測**：TWSE 自 2005-01-04；**TPEX 2007 上半年才開始**（2007-01-04 無、07-02 有 528 檔）→ 2007-07 前上櫃無來源。ETA 約 1.9 小時
  - **追加：上櫃找到官方來源 ✅（推翻先前結論）**：TPEX `bulletin/exDailyQ` 不是「忽略 date」，而是要用 **`startDate`/`endDate` + 斜線格式**（`20250801` 靜默失敗、`2025/08/01` 成功）——又是雷區第 2 條的重演。新增 `fetch_ex_rights_tpex_direct`（純附加），重建後 **25,718 筆 / 2,123 支**（twse 15,775 + tpex 9,943）。**宇宙覆蓋率 54.3% → 97.4%，且兩市場對稱**（上市 97.5% / 上櫃 97.4%，原本上櫃只有 0.2%）——原先最擔心的「上市已還原、上櫃未還原」不對稱問題消失。連帶：上櫃不需 dividend_raw 公式自算（該公式降為交叉檢查）、**`dividend_raw` 停更不再阻塞 B-3**、減資清單擴大到 29 筆。過程踩到「暫時性失敗靜默變成沒資料」（TPEX 2022–2024 整段落空但總筆數看似合理，靠年度分布凹陷對齊抓取區塊邊界才發現）→ 已加重試 + 逐年逐市場缺口檢查
  - **✅ B-3 步驟 2/3 全部完成（2026-07-29）**：重抓 3.84 小時 / 22 個年度檔（2012 與 2025 因暫時性失敗各缺 4/2 天，已重跑至空回應 0）。**又抓到一類漏掉的公司行為：減資**——第一版還原後 `|報酬|>100%` 從 184 **上升**到 273（唯一變差的指標），追查發現「前收 1–3 元 → 停牌 12–23 天 → 復牌 10–40 元、倍率 ~10×」是彌補虧損減資的指紋，273 筆中 203 筆屬此類，而純減資走股票換發、**完全不在 TWT49U/exDailyQ 裡**。補上 TWSE `reducation/TWTAUU` + TPEX `bulletin/revivt`（`adj_factor = 恢復買賣參考價 / 停止買賣前收盤價格`，同一形式），新增 **667 筆**（彌補虧損 374/退還股款 208/現金減資 85）→ 因子表 26,385 筆 / 2,143 支。**最終 `prices_adj_raw.parquet` 8,282,049 列 / 2,479 支，每一項指標都優於舊資料**：`|報酬|>40%` 850→**471**（2026 年 93→**4**）、`|報酬|>100%` 184→**144**（2026 年 5→**0**）、報酬 std 0.0817→**0.0349**、p99.9 **+12.52%→+10.00%**（正好是台股漲跌幅上限，舊值在制度上不可能存在）、最近交易日檔數 1,948→**2,100**（順帶修好 P2 的「2026-05-25 少 353 支」）。除權息日報酬 median −4.01%→+0.48%、`<−2%` 比例 73.08%→13.69%（基準 12.87%）。2412 於 07-09 由 −4.30% → **−0.60%**，與官方隱含 −3.73% 完全吻合。**接縫**：2007-07 前上櫃段用等比縮放（243,560 列 / 667 支、係數 median 0.9857）並標 `src="legacy_scaled"`。**✅ 已於 2026-07-29 切換 `prices_raw`**（使用者確認後）。`V6/scripts/switch_to_adjusted_prices.py`——**關鍵原則「只改值、不改型別」**：新檔 `Date` 是 `timestamp[ns]`、production 是 `large_string`，若直接換，每日更新寫進來的字串日期會與 timestamp 混型 → `drop_duplicates` 靜默失效，正是問題 1 的根因；故寫入前轉回字串，且不帶 `src` 欄（每日新增列不會有，會產生 NaN）。切換後 schema 逐欄型別與切換前完全相同。備份 `prices_raw_backup_before_adj_20260729.parquet`（複製回去即可回復）。**切換後驗證**：健檢重複 0／`Close<=0` 0；`build_features` + `clean_and_scale` 實跑通過（48,719 列 × 62 欄、無 inf、報酬 std 0.0197），唯 4 個含 NaN 的欄位是 `Alpha_5d/10d/20d/60d` 標籤欄（序列尾端本來就是 NaN，**新舊一致**、非回歸）
  - **順帶修健檢的量測分母**：法人覆蓋率原本用「原始 prices 宇宙」當分母，切換後 prices 從 1,948 支變 2,100 支（多出的 133 支是 ETF，交易所本來就不出個股法人明細），覆蓋率會從 96.1% **假摔到 89.5%** 並掛上永遠不會消的警報。改用與 `run_daily_inference` 相同的 `filter_tradable_universe` 過濾後為 **可交易宇宙 1,899 支、覆蓋率 96.2% ✓**（健檢警告 8 → 7）
  - **接縫分析順帶修好 B-2 的 PIT 限制**：舊檔比交易所多的 374,228 列中，**93.0% 落在該股「首次出現於交易所資料」之前**＝上市櫃前的興櫃期，另 7.0% 是成交量 median **90 股**的日子。「首次出現於交易所資料的日期」＝上市櫃日，是 **PIT 正確**的判準，自動排除「曾是興櫃後轉上市櫃」股票的興櫃期——那正是 B-2 用 `stock_info` 現況快照時無法處理、原記為已知限制的部分，該限制已消失
  - **步驟 3 核心邏輯 ✅ 已寫好並驗證**（`V6/scripts/apply_ex_rights_adjustment.py`，用已完成的 2005–2010 年度檔試跑）：`adjusted(t) = raw(t) × Π{adj_factor(e) : e > t}`，除權息日當天用 `searchsorted(side="right")` 排除自己的因子。**驗證：除權息日報酬 median −5.83% → +0.77%、`<−2%` 比例 80.93% → 15.43%（一般交易日基準 18.00%）**；乘數上升 4,383 次（= 除權息事件）、下降 12 次且**全部可由減資解釋**；OHLC 保序、無 NaN／`<=0`。9105 於 2005-04-12 原始 −89.31%、官方隱含 −90.00%、還原後 +6.88%。**踩到一個自己寫反的斷言**：初版斷言「乘數應隨時間遞減」，但 `mult(後) = mult(前)/f(e)`，除息 `f<1` → 乘數**遞增**，於是把 4,383 筆正確結果報成違反。**還原後除權息日仍 +0.77% 經查為真實填息**：還原幅度 <1% 那組精確落在基準 0.0000%，且中間三組（1-3%/3-6%/6-12%）約 +0.65% 持平**不隨幅度等比放大** → 排除系統性因子誤差。**Volume 刻意不還原**（`adj_factor` 混合了現金股利，拿去調股數會錯，需另存無償配股率）。剩：接縫處理（2007-07 前上櫃無原始價），等資料到齊再寫
  - **`foreign_shareholding` 改直連 ✅（2026-07-28）**：新增 `fetch_foreign_shareholding_twse_direct`（MI_QFIIS，需 `selectType=ALLBUT0999`）／`_tpex_direct`（`insti/qfii`——端點名不直觀，forgnHold/foreignHold 都是 404）／`_direct`，接進 `run_daily_update`（`_catch_up_generic` 加 `date_col` 參數，因該檔日期欄是小寫 `date`）。**恆等式「持股比率 ≈ 持股數/發行股數×100」兩市場皆 100% 通過**——這是必要驗證，因為該檔有 `SharesRatio`（實際持股，下游用這個）與 `RemainRatio`（尚可投資空間）兩個語意相反的欄位。回補 58 個交易日 / 111,579 列 → 2026-07-28、3,672,808 列。過程中 3 天（06-16/17/23）TWSE 回非 JSON 只拿到上櫃 890 列，靠「逐日列數 median 1,980」的檢查抓出並重抓。**健檢警告 8 → 7**
  - **⚠️ 操作教訓**：外資持股回補與價格重抓**同時打 TWSE**，導致重抓端出現 4 次 `HTTP 307`（限流）→ 2012 年度檔會有 4 天缺 TWSE，該年跑完刪 `raw_2012.parquet` 重跑即可。**全量重抓期間只做不同主機的作業**（TAIFEX／TDCC／MOPS 各自獨立）
  - **`holdings` 無法回補**：TDCC 開放資料 `getOD.ashx?id=1-5` **忽略日期參數**（三種寫法都回最新一週 20260724），集保歷史無開放來源 → 只能從現在起逐週累積，81 天缺口需另想辦法（或維持 FinMind）
- **資料源直連化第二批 + 還原因子表（2026-07-28，使用者外出期間執行）**：
  - **`per_raw` / `securities_raw` / `market_value_raw` ✅**：使用者 07-24 跑好的 staging 檔經驗證後 merge（`V6/scripts/merge_staging_202607.py`）。驗證方式＝**接縫連續性檢定**（production 04-24 FinMind vs staging 04-27 交易所）：PER median 0.9806、PBR 1.0000、market_value 0.9897 皆連續；三檔皆無非交易日假資料、無重複鍵（對照組：daytrade staging 在非交易日 07-10 有 1,710 列假資料）。schema 需對齊（`dividend_yield`→`DY`、`Securities_Balance`→`Securities_Lending`）。並新增 `_catch_up_generic()` / `_catch_up_market_value()` 接進每日更新
  - **`prices_raw` 缺漏與損壞 ✅**：回補 2026-04-27/04-28（交易所直連；判定依據＝實測 04-24 資料與交易所**原始價逐檔相同**，median 1.0000 零偏差 → 該段是原始價，用交易所回補才同基準）+ 清除 329 列 `Close<=0`。現 8,761,018 列、`Close<=0` = 0
  - **除權息還原因子表 ✅（新資產）**：`Data/processed_v6/ex_rights_raw.parquet`，由 TWSE TWT49U 建（支援區間查詢，全歷史 8 次請求）。**15,766 筆 / 1,188 支 / 2005-01-11 → 2026-07-28**。`adj_factor = 除權息參考價 / 除權息前收盤價`＝交易所官方口徑，一次涵蓋現金股利+股票股利+現金增資。**`adj_factor > 1` 的 20 筆代表減資**，順帶解掉檢查表 C3。⚠️ 只有上市；TPEX 的 `exDailyQ` 忽略 date 參數（只回「即將除權息」前瞻窗口）
  - **健檢警告 14 → 8 項**，資料損壞類（重複列、`Close<=0`）全部歸零。六個資料源（prices/margin/daytrade/per/securities/market_value）皆為 2026-07-27
  - **五項待拍板事項已於 2026-07-28 全數決策並執行**，決策內容與執行結果寫在 `MarketMamba_資料問題處理記錄_2026-07-27.md` 的 B 章節（每項附原始選項脈絡 + 決策 + 量測結果）
- **資料源直連化第一批 + 三個資料源修復完成（2026-07-27~28）**：使用者目標＝「先把資料完全修好，再談模型」。**關鍵決策：不訂閱 FinMind VIP**——查證後發現 VIP 解決的是錯的問題（FinMind 免費層強迫逐股查詢 ~2,000 次/3.6 小時，交易所端點是逐日整批 60 次/1.5 分鐘，差別在 API 形狀不在速率上限）。策略改為**按更新頻率決定來源**：每日/每週源改交易所直連、月/季源（revenue/financials）留 FinMind 免費層過夜跑。
  - **`margin_raw` ✅**：新增 `fetch_margin_twse_direct`/`_tpex_direct`/`fetch_margin_direct`（含回傳日期硬性核對 + 恆等式驗證）。交叉驗證時**抓出 FinMind 對上櫃股把券賣/券買標反**（全歷史 2010–2026，`corr(ΔShort_Balance, Sale−Cover)` 在 OTC 為 −0.85~−0.93、符號一致率僅 1%）→ `fix_margin_short_swap.py` 交換 2,638,958 列（33.2%），符號一致率 **65.69% → 97.18%**。回補 63 個交易日/116,292 列**僅 100 秒**。接進 `_catch_up_margin()`（補缺口而非只抓今天，對公布延遲自我修復）。**發現第二個根因：舊的 `fetch_margin_finmind` 只寫 CACHE_DIR 單日暫存檔、從未 append 到 margin_raw**——即使 VIP 沒到期也不會更新
  - **`daytrade_raw` ✅**：發現 `Day_Trade_Volume` **2014–2026 全部 4,263,330 列都是 0**（整欄自始死值，非停更）。新增兩個直連 fetcher + `daytrade_shares_to_ratio()`，全歷史重建 70.7 分鐘 → **3,868,497 列**、3,051 交易日 × 1,965 支。年度 median 0.026(2014) → 0.220(2026) 單調上升，對得上當沖制度放寬時序＝強驗證訊號
  - **`prices_raw` ✅**：修好 `fetch_prices_tpex_direct`——舊版打的 OpenAPI **完全忽略 date 參數**（五種格式含不傳都回當天資料），改用 `/www/zh-tw/afterTrading/otc`（需 `type=EW`）。用正確端點重抓覆蓋 2026-05-25 起的 OTC（38,350 列覆蓋 + 154 列新增），修掉 236 列嚴重 Volume 壞值
  - **財報三維 ✅**（`fundamentals_v2` 旗標下）：`Gross_Margin`/`ROE`/`Book_Value` 自 2005 年起是死常數。根因是 **FinMind 在損益表把「淨利歸屬母公司業主」的英文 type 標成 `EquityAttributableToOwnersOfParent`**（要看 `origin_name` 才看得出來），且 **`df_balance_sheet` 一直是從未被讀取的參數**。新增 `_balance_sheet_equity()` 從資產負債表取真正的權益。驗證：2330 毛利率 62.3%/ROE 年化 37.3%、2317 5.9%/12.1%、2454 46.1%/23.0%，全部對得上實際
  - **閘門 ✅**：`is_trading_day()`（非交易日一律不寫任何檔）+ `_check_universe_coverage()`（單日變動 >±5% 告警，實測正確抓到 2026-05-25 的 −15.2%）
  - **進度**：59 維中原本 14 維凍結 + 4 維永久死值 → 現剩約 8 維凍結（`per`/`market_value`/`securities`/`holdings`/`foreign_shareholding`），等對應資料源改直連
  - **文件**：`MarketMamba_資料問題處理記錄_2026-07-27.md`（15 個問題，每個含「怎麼發現/問題是什麼/如何解決/產生原因」+ 方法論教訓）、`docs/data-source-implementation-traps.md`（13 項雷區，寫新 fetcher 前必看）
- **資料品質全面稽核（36 項）+ 第一輪程式修復（2026-07-27，使用者已驗收）**：依使用者更新後的 `MarketMamba_資料品質檢查表.md` 逐項對真實 parquet 實測（8 支審計腳本，非讀碼推論），結果寫入 `MarketMamba_資料品質檢查表_驗證結果_2026-07-27.md`。**36 項：通過 11、有疑慮 24、不適用 1**。
  - **G4 特徵計算順序（使用者特別點名）→ 生產路徑證實正確**：用「子集不變性」檢定——宇宙從 60 支縮到 6 支，56 維逐位元最大絕對差 `0.000e+00`；`MA_20`/`Return_5d`/`Volatility_20d` 與原始 Close 手算誤差皆 0；`clean_and_scale` 後 42/56 欄改變（證明橫斷面確實是最後一步）。對照組：若顛倒，`Return_5d` 與正確值 Spearman 僅 **-0.035**（訊號全毀且只表現為 IC 偏低）。**研究路徑 `baseline_common` 有中**——rolling 建在已 z-score 的值上（base matrix 最後一日 Close mean=-0.0000/std=1.0000），而同檔的 `Mom_*` 卻是正確順序
  - **三個「不會報錯」的線上問題，根因收斂到 `fetcher.py:1044` 一個型別 bug**：yfinance 的 `Date` 是 `Timestamp`、TWSE/TPEX direct 是 `str`，concat 後成 object 混型 → `drop_duplicates` 完全失效。這一個 bug 同時造成 ① 每日 ~1,550 列重複（2026-07-06 起，818 支受影響，2432 的 Return_5d 含重複 -0.53%／去重後 +1.07%，**正負號相反**）② 同日還原/未還原兩種價格並存（1459 於 07-22 兩列比值 1.188）③ Volume 單位不一致（1213 為 3000 vs 1）
  - **修復（純程式，六項）**：`fetcher.py` concat 前統一 Date 型別再去重（`keep="first"` = yfinance 還原價優先）+ `_append_to_parquet` 寫檔前保險絲；`run_daily_inference.py` 新增 `_sanitize()` 把去重／剔除 `Close<=0`／排除 ETF 前移到 `build_features` **之前**（原本在 `clean_and_scale` 之後，時序特徵與當日 z-score 早已被污染）；新檔 `marketmamba/data/hygiene.py`（宇宙過濾 + 資料健檢，用 parquet statistics 取最大日期，健檢自身只吃 0.14 GB RAM）；`feature_engineer.py` 新增 `fundamentals_v2` 旗標修 `EPS_Surprise` 與 Q4 年報 look-ahead；`baseline_common.build_derived_roll()` 改建在 chunk 原始值上
  - **回歸測試（V6.1 一致性的關鍵）**：`fundamentals_v2=False` 與 git HEAD 版本 60 欄逐位元比對最大絕對差 `0.000e+00`、`clean_and_scale` 後亦 `0.000e+00` → 線上零影響
  - **生產驗證（2026-07-27 19:30 推論實跑）**：`_append_to_parquet` 保險絲觸發、`偵測到 11,384 列重複，已去除`（存量 10,591 + 當日新增 793）→ **parquet 重複列存量被自動清空**，健檢確認「今日 0 列 / 近 90 天 0 列 ✓」；`_sanitize` 剔除 `Close<=0` 329 列 + ETF 49,906 列；健檢報 14 項警告（13 個停更源 + `Close<=0` 存量）。原本 07-24 的「去重後 1,076,052 → 1,065,461」那 10,591 列正是同一批，證實舊流程是在特徵算完後才去重
  - **其他確認發現**：股票池 2026-05-22 的 2,321 支 → 05-25 的 1,968 支，一日少 353 支（來源切換，非下市；同日 `Close<=0` 損壞列也恰好停止）；14 個資料源停更 77–176 天，59 維中約 27 維是凍結值或常數；`EPS_Surprise` 是 bug（`pct_change(4)` 套在日頻列，2330/2317/1301 皆「換值 N 次 → 非零 4N 列」）；Q4 年報 `available_from` 早 45 天（法定 3/31、程式給 2/14）；OHLC 違規 322,300 列（3.7%）全為 `Open` 越界且集中在冷門股（成交量中位數僅全體 1/16、`High==Low` 比例高 7 倍）→ `Open` 與 HLC 來源定義不一致；2026 年極端報酬 515 筆中 422 筆（82%）源自 `Close<=0`；14 支 ETF 混在個股宇宙
  - **`baseline_common` roll 快取已重建（2026-07-27 20:21，17 分鐘、峰值 RSS 3.13 GB）**：`7,273,199 × 64`，新檔 2.31 GB。驗證：行序與 base 逐列完全對齊（`load_xy` 的 assert 會通過）、66 欄無缺漏、`Close_rmean20` mean=+0.0000 std=0.9925。**但誠實校準：這個修正的實際影響比預期小**——新舊版 `Close_rmean20` 的 Spearman 高達 **+0.9999**（價格水位高度自我相關，z-score 的移動平均 ≈ 移動平均的 z-score，36 個 rmean 維度排序幾乎沒動）；真正有差的是 24 個 rstd 維度：Spearman +0.7862，而且**舊版 `Close_rstd20` 的 std 只有 0.0556**（橫斷面排名日間幾乎不動 → 舊版是個近乎退化的小尺度特徵），新版已正確逐日標準化為 std≈1.0。→ 預期 Ridge/GBDT/GRU 重跑後結論不會翻盤，但數字要重出才能與新特徵並列
  - **推翻既有認知一項**：CLAUDE.md 原記載的「還原/未還原混源接縫」在歷史段**不成立**——24,357 筆除息事件的決定性檢定（`corr(當日報酬, -股利/前收)`，未還原應趨近 +1）在 2015–2026-05 全部 ≈ 0。但盲區明確：`dividend_raw` 停更於 05-05、2026-06 之後 0 筆事件，而台股除息旺季正是 6–9 月；2412 於 07-09 跌 6.0 元（-4.3%、爆量 2.4 倍、歷年配息 4.5–5.0 元）仍高度可疑。**根本原因是 prices_raw 逐日增量寫入，未來的除息永遠不會回頭調整已寫入的歷史** → 需定期全量重抓價格（資料操作，非程式）
- **方向二 Step 5 對照表 + 方向一 Step 8/9 前端頁完成（2026-07-15，使用者已驗收；僅剩 push 部署）**：
  - **Step 5**：`docs/baseline-comparison-table-2026-07-15.md`（方向二最終交付物）——四階（Ridge/GBDT/GRU/Mamba）5d 主表：訊號層 IC（全市場+高流動分層）、組合層（年化/Sharpe/MDD/換手/成本×2）、可解釋性並排；20d 副表；引用紀律四條；判讀=「用可解釋性換到多少效益？答案是負的」+ 三條收斂證據（模型形式 ±0.01 內移動／序列 vs 扁平無差／架構紅利不成立）+ GBDT 教訓（IC 排名 ≠ 落袋排名，表格必須訊號層與組合層並排）
  - **Step 8**：對照表精簡版嵌入 `docs/breadth-pipeline-page-draft-2026-07-12.md` §5.5（§5 與 §6 之間預留位置）
  - **Step 9（選項 A，使用者拍板）**：`/breadth` 頁尚不存在（屬頁面樹重整計畫），先做**獨立頁** `app/frontend/src/pages/Pipeline.jsx` + 路由 `/pipeline` + nav 入口「🔬 Pipeline」，之後 `/breadth` 重整時再移入。內容=底稿 §1~§7 全部（三層 fallback、PIT、59 維特徵表、輸入規格、架構論述含 rank 目標消融、§5.5 對照表、驗證結果含樣本警語、§7 誠實限制八條摺疊卡片三態標記）。**純附加**：只動 `App.jsx`/`AppLayout.jsx` 各 +2 行，既有頁面零修改；`npx vite build` 通過。數字為靜態研究成果（核實日 2026-07-15）寫死在頁內，非每日更新資料
  - 待辦：使用者本地 `npm run dev` 看過 `/pipeline` 頁 → 指定檔案 git push → Vercel 自動部署
- **方向二 Step 4：階 3 序列模型（GRU）完成（2026-07-14~15，使用者已驗收）**：`V6/experimental/baseline_rnn.py`（直接切 `baseline_base_59d.parquet` 的 (60,59) 視窗、4 組網格 hidden{64,128}×lr val 日 IC 選組 + early stopping、full-train 重訓、分層 IC／成本×2 沿用標準輸出；RTX 3060 實跑 12.2 小時）。**5d test IC +0.1113（高流動組 +0.0867）四階最高**；20d +0.1081 追平 Ridge、勝 GBDT。**階 3 問題有了答案：Mamba 贏不是架構紅利**——~49K 參數的 2 層 GRU 吃同一份 59 維特徵、同 window 60、同 rank label，比 1.66M 參數 v6_short（+0.0870）高 +0.024；且 GRU 對 GBDT 僅 +0.0015 → 三條證據收斂「模型形式（線性/樹/RNN/SSM）只在 ±0.01 內移動，訊號來自特徵/label/廣度」。**組合層無 GBDT 反差：Top50 年化 +22.8%（Sharpe 1.13、換手 78%）> Ridge +18.7%，最強落袋 baseline 由 Ridge 換 GRU**；成本×2 仍轉負（−2.6%，四階同款脆弱）。兩個 horizon 都是最小 h64 勝出，h128 更快過擬合且 epoch 時間異常放大（37 秒→25~50 分、佔 12.2 小時中 ~11 小時，之後同類實驗砍 h128 可省一個數量級）。WF 本階未跑（協定定位便宜階為輔；h64-only 補跑方案記在報告）。報告 `docs/baseline-step4-rnn-2026-07-15.md`、完整數字 `V6/experimental/result/baseline_rnn_result.json`、模型 `result/gru_5d.pt`、log `result/rnn_run.log`
- **方向二 Step 3：階 2 GBDT（LightGBM）完成（2026-07-13~14，使用者已驗收）**：`V6/experimental/baseline_gbdt.py`（共用 300 維快取、4 組網格 val 日 IC 選參控制多重測試、SHAP 用原生 pred_contrib、分層 IC／成本×2／等權基準為排查後標準輸出）。**5d test IC +0.1098（高流動組 +0.0802）小勝 Ridge +0.1015；20d +0.1004 反輸 Ridge +0.1081**——非線性紅利小且僅在短 horizon。**WF 46/46 fold 全正、mean +0.1285、每 fold 系統性高 Ridge 約 +0.014**（增量非 regime 僥倖；兩模型最弱段同落 2022-10~2023-07 熊市後段）。**組合層反差：GBDT IC 較高但 Top50 年化 +10.8% < Ridge +18.7%**（換手 84%、IC 增量集中小型股與中段排名、成本×2 轉負 −13.6%）→「IC 排名 ≠ 落袋排名」，以照 Top50 買為準最強 baseline 仍是 Ridge。報告 `docs/baseline-step3-gbdt-2026-07-13.md`、完整數字 `V6/experimental/result/baseline_gbdt_result.json`、模型 `result/gbdt_5d.txt`
- **Baseline IC 0.1015 排查完成（2026-07-13，診斷 D0–D5 全數執行，使用者已驗收）**：新增隔離腳本 `V6/experimental/baseline_ic_diagnosis.py`（重建 Ridge 後 test IC +0.1015 分毫不差重現），結果報告 `docs/baseline-ic-diagnosis-results-2026-07-13.md`、完整數字 `V6/experimental/result/baseline_ic_diagnosis_result.json`。**結論：不是程式 bug、不需重抓資料**：
  - **D0（資料錯誤→機械性反轉）排除**：排除 test 中 ±5 天內含超限報酬的列（2.08%）後 IC 反升至 +0.1046
  - **D1 分層 IC**：流動性小/中/大 = +0.140/+0.100/+0.0705；市值小/中/大 = +0.133/+0.096/+0.0626——headline 被小型股墊高，但大市值高流動組仍 ≈ Mamba 水準（0.07）
  - **D2 消融推翻「反轉主導」**：去短窗反轉 IC 僅 −0.006（+0.0954）；去全部價格技術特徵（只剩籌碼+基本面+宏觀 190 維）仍 +0.0717；單因子 −Return_5d 只有 +0.044 → 0.10 是幾十個弱訊號分散疊加（廣度效應也解釋 ICIR 1.2 / 46/46 fold「太乾淨」）
  - **D3 證實假說 3b 基準錯配**：等權全 eligible 不排序就對 TWII −87.2%（TWII +118% 為權值股拉抬）；策略 +49.7% vs 等權宇宙 +44.1% vs 隨機 Top50 −33.0% → 對正確基準是正超額
  - **D4 成本是真正脆弱點**：成本 ×2 → 年化 +18.7% 變 −5.4%；但 Top50 限流動性前 1/3 仍 +16%/年（Sharpe 0.82），存在可交易子集
  - **D5 + 靜態稽核過**：籌碼特徵同日 ρ=1.000/前一日 0.12 無位移；營收/財報 available_from 延遲正確（+11天/+45天/+2月）
  - **引用紀律（重要）**：方向二對外陳述 0.1015 必須分層（全市場 +0.1015 / 高流動 +0.0705 / 純籌碼基本面 +0.0717）+ 註明存活者偏差未量化；組合層基準改等權宇宙、「對 TWII −90%」不再單獨引用
  - **附帶發現三個資料衛生問題（使用者已納入計畫、不急修）**：① 2026-04-30 起 327 筆 Close=0 損壞列（零成交日寫入 0 價，與「寫入端缺 gate」待辦同源）② 超限報酬 2025–26 偏多（多為 ① 的 inf 與減資/IPO 合法事件）③ prices_raw 還原/未還原混源接縫（yfinance auto_adjust 歷史 vs 日增量未還原，2412 的 2026-07-09 除息跳空實證；季度全量重抓價格可解）
- **研究計畫三方向啟動：不需 Colab 的前四步完成（2026-07-12，使用者已驗收）**：依 `planing/研究計畫_主檔.md` 排序執行，全程未動 production、未動 `V6/models/`：
  - **方向三-C 首輪分析（C1~C3 完成、C4 首輪結論已出）**：新增隔離腳本 `V6/experimental/conviction_c_analysis.py`（讀 50 天 df_kelly 歸檔 + prices_raw 真實股價，可重跑、樣本自動累積），報告 `docs/conviction-c-analysis-2026-07-11.md`。**核心發現：U 校準在 20d 大致成立（U 最低分位箱內 IC 0.120、最高兩分位 0.083–0.089；SQ Top50 +2.79%/20d t=3.94、比純 Alpha 排序多 +1.3%），但在 5d post-P0 樣本上反向（最有信心分位反而最差、SQ Top50 −1.09%/5d）**——惟 post-P0 5d 僅 14 天且窗口重疊（實質獨立觀察 ~3 個），不能定論。**結構性發現：dashboard 的 SQ 是 20d 訊號、使用者操作是 5d 週期，horizon 錯配**；5d 該看 v6_short 的 SQ_5d（archive 累積中，同款分析可套用）。另證實「低 U 硬門檻交集」不如「SQ 連續值」（交集組 +1.51 輸給高 U 對照組 +1.96）。數字已抽驗（手工重算 06-25 與腳本一致）
  - **方向二 Step1 協定凍結（v1.0，四決定使用者拍板全採建議）**：`docs/baseline-experiment-protocol-draft-2026-07-11.md`——①切分：Phase 3 同 harness 單一切分為主（train ≤2023-12-31、test 2024-01~2026-06；Mamba 引用同 harness 重跑值 0.0870 不用歷史 0.0951）+ Ridge/GBDT 加跑 WF 為輔 ②label：rank(Alpha_5d) ③5d 主、20d 副 ④組合層 Top50 等權、5 日再平衡、買 0.15%/賣 0.45%。自此凍結，中途不改
  - **方向一 1~7 底稿完成**：`docs/breadth-pipeline-page-draft-2026-07-12.md`（含 §5「為什麼 rank 目標」論述：Phase 0→1→2 消融證據 0.0434→0.0487→0.0951；§7 誠實限制八條）。**勘誤**：資料 fallback 實際是 yfinance → TWSE/TPEX direct → FinMind，計畫文件寫的「玉山證券」只用於持倉同步、不在研究管線
  - **資料清理（使用者授權）**：prices_raw 刪除 2026-06-07（週日）整日 5,194 筆假資料（4,317 筆非 4 位數代碼 + 877 筆 4 位數，模式同 06-19 端午節事件；備份 `prices_raw_backup_20260712.parquet`，穩定後可刪）。877 筆 4 位數假資料曾讓週日混進交易日曆 → 刪除後重跑 C 分析，post-P0 數字不變、pre-P0 小幅修正、結論全部不變。**另發現重複寫入 07-07 起又出現（每日 ~1,550 筆）**，與非交易日假資料疑同源（每日更新寫入端缺 gate）、來源仍待查
- **進場標準統一 + 機構資料管線修復 + 前端 K 線圖（2026-07-07，commit `b72afc2`）**：使用者暫停模型實驗期間，轉做「模型以外」的調整。K 線圖已驗收；scanner 新邏輯待 07-07 推論確認：
  - **進場標準統一**：`scanner.py`（scan_version 1.4）BUY 判斷由「條件數 ≥2/4」改為**複合分數 ≥70（保守 ≥90）**，權重/型態加分改 import `signal_conditions.compute_entry_score()`——dashboard 訊號與 sim_engine_v3 從此同一套標準，sim 績效可直接回答「照 dashboard 買會不會賺」。前端 `TradingSignals.jsx` 規則文案同步改分數制。**性格變化**：高分型態股（1/4 條件+W底）能進榜、「2 條件無型態 50 分」降觀察清單
  - **修四個實際 bug**：① sim_engine_v3 把複合分數當 base score 再加型態分（重複計算最多 +50，改用 `base_score`）② scanner「外資連買」誤用 Foreign_Buy 總買進（大型股天天>0）→ 改 Foreign_Net ③ portfolio_checker 讀 prices_raw 不存在的外資欄位 → 退場「外資連賣3天」恆為 0，改讀 institutional_raw ④ scanner 退場死碼移除（run_scan 從未收到 portfolio_positions、exit_signals 恆空；輸出保留空列表向下相容）
  - **機構資料管線修復（重大）**：診斷發現 2026-04-25 起 institutional_raw 每日更新只寫進 7 支水泥股——**TWSE T86 缺 `selectType=ALLBUT0999` 參數**（預設只回水泥類）+ **TPEX 舊端點改版只回 HTML**。修 `fetcher.py`（TWSE 加參數+19 欄索引重對映、TPEX 換新端點 `/www/zh-tw/insti/dailyTrade` 24 欄、欄位語意用「買-賣=淨、分項加總=合計」數值驗證）；`V6/scripts/backfill_institutional.py` 回補 4/25→7/6 共 50 交易日（每日 ~1,900 筆，TWSE+TPEX）。**驗證**：修復前 28 天/888 筆訊號的機構條件 100% 「無此股機構資料」；修復後正常觸發（外資連買 3 天×10 支等）。附帶清理：06-19 端午節被 API 異常寫入的假資料已刪、prices_raw 去重 51,801 筆（5/26 起每天 ~800 筆重複寫入，值相同、來源待查）
  - **條件貢獻分析器** `V6/marketmamba/backtest/condition_analyzer.py`：仿 dual_ic_analyzer，讀 `V6/results/{date}/action_signals.json` 歸檔，對四條件+型態各算前瞻 5d/20d 超額報酬、hit rate、對 Top50 基準的 edge、評分分桶（<40/40-69/≥70 直接檢驗 70 分門檻）。掛進 run_daily_inference（non-fatal）輸出 `condition_analysis.json`。**首測 28 天**：「排名穩定」正貢獻（edge +0.28%）、「相對低點」5d 負貢獻（-0.47%、hit 18%）——樣本少僅供方向，自動累積。機構條件歷史旗標全 False，其統計從 07-07 起才有意義
  - **前端 K 線圖（已驗收 ✅）**：後端 `GET /api/market/kline/{ticker}`（Yahoo v8 OHLCV proxy、.TW→.TWO fallback、15 分快取）+ `KLineChart.jsx`（klinecharts v9：日K+成交量+MA5/20/60+十字游標+紅漲綠跌+3月/6月/1年切換），嵌入 `StockModal`（420→720 寬），有型態訊號時畫 failure_stop/目標價水平線。TradingSignals/Dashboard 點股即看
- **雙模型真實市場效益追蹤上線（2026-07-06）**：使用者決定暫停 Phase 3 實驗，等真倉（V6.1）先驗證有沒有賺錢再繼續做實驗；釐清目前**真倉用的是 V6.1（20d 目標）**，雙模型（v6_short/v6_trend）現階段定位是**拿真實市場走勢驗證效益**，但雙模型自 6/19 上線以來從未真正算過「排名結果對實際報酬有沒有效」，只有訓練時的驗證集 IC（短線 0.095、趨勢 0.096）。新增自動化追蹤、之後不需要手動介入：
  - 新增 `V6/marketmamba/backtest/dual_ic_analyzer.py`：仿 `ic_analyzer.py`，讀 `archive/df_short_*.csv`/`df_trend_*.csv` + `prices_raw.parquet` 真實股價，對 5d/10d/20d/60d 各自算 IC/ICIR/t值/IC>0比例 + **Top50 by SQ 的實現超額報酬**（比 IC 更直覺回答「照排名選會不會賺錢」）。輸出 `V6/results/dual_ic_analysis.json`
  - `V6/run_dual_inference.py`：歸檔後呼叫分析器（non-fatal），輸出加入 `_git_push()` 清單一起推送。⚠️ 注意分析器必須讀 `prices_raw.parquet` 拿真實股價，不可誤用推論用的 `df`（Close 已被 `clean_and_scale` z-score 過）
  - `app/backend/routers/dual.py` 新增 `GET /dual/ic`（1h TTL，比照現有 `/dual/signals`），`/dual/cache/refresh` 一併清空
  - `PersonalOS` + `app/frontend` 的 `InvestmentSim.jsx`（投資模擬機器人頁面）新增第三分頁「🔬 雙模型驗證」：短線(5d/10d)+趨勢(20d/60d) 四個 horizon 各自的 IC/ICIR/Top50超額報酬/樣本天數，n_days<20 顯示「樣本仍少，建議 20+ 天再參考」、n_days=0 顯示「尚未累積足夠時間」——文案寫死、之後不需要再調整
  - **首次實測**（4 天樣本，僅 5d 可算）：mean IC +0.129、ICIR 1.32、Top50 實現超額 +1.35%/期、4/4 天 IC>0——方向正面但 n=4 統計上毫無意義；10d/20d/60d 目前 0 天可算（20d 最早約 7 月中下旬才有第一筆）。頁面文案已講清楚這點，不需要之後再解釋
  - **定位釐清**：V6.1（20d）= 真倉標的、雙模型 = 效益觀察對象，兩者獨立、互不影響
- **Phase 3 實驗 A（dropout sweep）實跑完成、結論 dropout=0.2 有效（2026-06-27）**：Colab 跑完 0.1/0.2（0.3 best 已鎖、log 停 ep9），結果在 `V6/experimental/result/`（`phase3_A_dropout_sweep_result.json` + 三個 `status_short_A_doXp.json`）。**同 harness 對照（cutoff=2023-12-31、val 580 天 2024–2026，唯一變因 dropout）**：0.1 峰 5d IC **0.0870@ep3** / 0.2 **0.0959@ep4** / 0.3 **0.0961@ep3**。判讀：① **要跟「這次的 0.1 重跑（0.0870）」比、不是歷史 0.0951**——歷史 0.0951@ep8 沒重現（資料切分/seed 差異），同 harness 下 0.2/0.3 比 0.1 高 **+0.009（≈+10%）**、追平/微勝歷史 0.0951；② **val_loss 同步變好**（0.2 谷底 0.12331@ep4 與 IC 峰同 epoch、比 0.1 更低）→ 不是單點噪音；③ **過擬合起點延後**（val_loss 谷底 ep2→ep4、高 IC 平台變寬）＝弱版「峰值延後」達標，**但崩壞速度沒變慢**（峰後下滑 0.2 的 +0.0042 甚至比 0.1 的 +0.0029 快、峰仍偏早單尖）；④ **0.3 對 0.2 平手**（0.0961 vs 0.0959）、邊際遞減 → 選 **0.2**。對照規則「峰值升/延後/崩壞變緩 任一即有效」：峰值升 ✅ + 延後 ✅ 達標。**誠實補充**：改善屬邊際（同 harness +0.009、對歷史只追平）、0.096 仍單 epoch 尖峰，真偽待 Phase 3-E 多 seed 集成確認＝「方向可信、幅度待驗證」。**決定按計畫把 dropout=0.2 帶進趨勢模型、收 A 進 B**（未加跑 weight_decay 變因——+0.009＋延後已達「有效」門檻，不拖長 A）。0.3 那輪未跑完不影響結論（峰值早鎖），可不補
- **Phase 4 設計筆記定稿：產業理解融合（2026-06-27）**：規劃將「產業/供應鏈理解」並行加一層融進系統（不碰 V6.1）。起因＝Threads「股癌資訊系統為何快」討論（方法1 建產業地圖、方法2 追變化）。已實際進站調查櫃買 ic.tpex 產業價值鏈平台 + 查證 FinMind/MOPS 資料源。兩塊：**4-A KG 豐富化（進模型）**＝爬 ic.tpex 30+ 鏈 → 公司×節點對應 → 建「同節點競品/相鄰節點上下游（有向、位置粒度）」邊，重訓生效；對不到 stock_id 者（KY/外國/改名）保留標記不連邊。**4-B 產業變化 Agent（獨立層）**＝守備清單用鏈分組，訊號用 FinMind（月營收/存貨/capex/法人）+ MOPS（法說會簡報PDF/重大訊息），LLM 比對本季 vs 上季語氣、整條鏈同向變化才提醒。付費報價催化劑（集邦TrendForce）有記錄、因成本暫緩。**完整計畫見 `docs/phase4-industry-chain-fusion-plan-2026-06-27.md`**。定位 Phase 4 等級，待 Phase 3 收完接續、與穩定性工作不衝突。尚未動工
- **Phase 3 實驗 B/C/D 檔案備妥 + 預期結果文件（2026-06-25）**：接續 A，再備三支隔離實驗（全推 main 供 Colab pull、皆不改 production、不覆蓋線上 checkpoint、每組落盤 Drive JSON）：
  - **B `phase3_b_listnet_sweep.py`**：短線 listnet_5d sweep {0.0/0.25/0.5基準/1.0}（10d 保 1/2 比例）。weights 走 `train_short_model` 現成參數、dropout 用 monkeypatch 留可選。**0.0 是控制組**＝量出 listnet 在 rank-MSE 之上的邊際貢獻。釐清：Phase 1 的「listnet=Mid↔Long 旋鈕」是多尺度現象，短線單尺度無 gate、此處純排名強度
  - **C `phase3_c_trend_single_scale.py`**：趨勢砍成單尺度 Long-only（`LongOnlyEncoder` 含 padding_mask + GAT + gating + 3頭，與 MarketMambaV6 逐行對齊），monkeypatch `trend_model.MarketMambaV6` 後重用 `train_trend_model`。單跑診斷、對照多尺度基準 20d IC 0.0961、印參數量與判定（gate 已塌 Long → 預期持平、可砍）
  - **D `phase3_d_window_sweep.py`**：短線 window sweep {60基準/90/120}。window 是 `train_short_model` 直接參數（比 A/B 更乾淨）。≤200 不需 padding mask。前提：DS 已證多尺度對 5d 冗餘 → 預期持平、但屬便宜驗證（純切片免重建）
  - **預期結果文件 `docs/phase3-experiment-plan-2026-06-25.md`**：A~F 每個實驗的假設＋推理＋不同結果代表什麼＋把握度（誠實標註待驗證、哪裡沒把握）+ 跨實驗判讀紀律。亦為履歷 case study 素材
- **Phase 3 起手：實驗 A（正則救峰值）檔案備妥、待跑（2026-06-25）**：新增 `V6/experimental/phase3_a_dropout_sweep.py`（隔離、**不改** `short_model.py`，線上 dual 推論零影響）。設計＝**單一變因 dropout sweep**（基準 0.1 峰 0.0951@ep8 → 掃 0.2/0.3，其餘 window60/layers3/LR7e-5/wd1e-4/切分 train≤2023 全凍結）；monkeypatch `ShortModelV6` 固定 dropout 後交給現成 `train_short_model`，checkpoint 用獨立檔名 `v6_short_A_doXX.pt`（**不覆蓋** production `v6_short.pt`）；每組跑完落盤 Drive JSON（Colab 斷線可續）；尾端印峰值 IC／峰值 ep／峰後下滑對照表。**判讀**：峰值 IC>0.0951／峰值延後／崩壞變緩 任一改善＝dropout 有效、帶進趨勢模型。Colab 用法見檔頭 docstring。**尚未實跑**（使用者之後測試）
- **git 善後：收齊已完成卻沒上的 O3／訓練診斷／文件（2026-06-25）**：這批改動一直躺在 working tree（跟本機 56 維 config dirty 檔混在一起沒乾淨 commit）。指定檔案提交（commit `09fca17`，**排除** 56 維 `config.py`）：O3 `Signal_Quality_Raw` 未截斷排序（`run_daily_inference`/`signals.py`/`sim_engine_v3`/`portfolio_checker`，向下相容 fallback；已驗證 df_kelly.csv 確含該欄、後端切過去安全）、`deep_supervision.py` 訓練進度列印+ETA、docs（training-observation/uncertainty-calibration）、補 06-25 dual 歸檔。本機 56 維 `config.py` 維持 dirty、未上
- **修復雙模型前端不更新＝Render dual 快取沒被刷新（2026-06-22）**：症狀＝df_short/trend.csv 正確 push 上 GitHub（06-22、欄位對），但 Vercel 前端 dual 頁顯示舊資料。診斷：`/api/signals`(df_kelly) 回 06-22 ✅、`/api/dual/signals` 卻回 06-18 ❌——兩者讀同一 GitHub raw 來源，差別在刷新機制。**根因**：`dual.py` 與 `signals.py` 用同套 1h TTL 記憶體快取，但 dual 沒有 refresh 端點，且 `signals.py` 的 `/cache/refresh` 只清自己模組的 global（碰不到 dual.py 的 `_cache`），每日自動化 push 完 dual 從沒被刷新→只能等 1h TTL，在 Render free tier 下不可靠。**修法（方案 A，純附加）**：①`app/backend/routers/dual.py` 新增 `POST /api/dual/cache/refresh`（清自己 `_cache`，比照 signals.py，不動既有 `/signals`）；②`PersonalOS/scripts/run_daily.py` 加 `refresh_dual_cache()`，在 `run_full()` 的 `run_wsl_dual()` **成功 push 後**呼叫（須放 dual push 之後，現有 `signals/cache/refresh` 在 dual 之前且碰不到 dual）。已 push 部署
- **每日自動化 + 單日篩選上線（2026-06-19）**：`run_dual_inference.py` 加 `--push`（只 add 兩 CSV）+ **每日歸檔** dated 副本到 `V6/results/archive/`（本機留存、供 sim/穩定性累積歷史）；PersonalOS `run_daily.py` 在 `run_full()` V6.1 推論成功後呼叫 `run_wsl_dual()`（獨立 process、自動 push、失敗不影響 V6.1、休市日自動不跑——交易日 gate `trading_day.py` 完好沒被改）；前端 `DualSignals.jsx` 加「⭐ 精選」分頁（短線∩趨勢共識股）。**後續轉 Claude Code**：剩跨日穩定 filter + dual 模擬機器人（讀 `results/archive/`，待累積幾週資料；比照 `sim_engine_v3` 回放）
- **Phase 2 步驟 5 完成、雙模型 Vercel 前端上線 → Phase 2 全數完成（2026-06-19）**：新增 `app/backend/routers/dual.py`（唯讀 `/api/dual/signals`，1h cache 比照 df_kelly、重用 `GITHUB_RESULTS_URL` 換檔名得 short/trend）+ 前端「🔀 雙模型」新頁（`pages/DualSignals.jsx`，短線/趨勢 tab + rank-score 語意說明）+ `api/dual.js`、`App.jsx` 路由、`AppLayout.jsx` nav。**全程附加，`signals/market/...` 與 V6.1 頁面一個字沒動**。雙模型從訓練→並行推論→前端全鏈路上線、與 V6.1 並存當安全網。未做：每日自動化（run_dual 排程 + 自動 push CSV）留後續
- **Phase 2 步驟 4 `run_dual_inference.py` 跑通、雙模型推論上線（2026-06-19）**：獨立並行推論——自切 59 維 config（不動 V6.1 的 56 維）、個股大表 trim 近 2 年解 OOM（照 V6.1）、MC-dropout 兩段式、`clean_and_scale(macro_norm="ts")`。輸出 `df_short.csv`(5d/10d) + `df_trend.csv`(20d/60d)，rank-score 語意、依 SQ 排序。**健檢過**：各 1948 支、無 NaN、不確定性全正、`load_state_dict` 成功（驗證 59 維切換正確）、短線/趨勢前段名單不同。小觀察：趨勢 Score_20d 平均 +0.05（短線 ~0）= macro 2 年近似的水位平移、**排名不變**（排序對常數免疫），印證「macro 近似對選股 OK」。OOM 根因＝原本沒 trim、把整份 prices 8.7M + inst 32M 丟進 build
- **Phase 2 步驟 3 趨勢模型完成、雙模型齊備（2026-06-15，ep13 峰、Colab ep15 斷線）**：多尺度 `MarketMambaV6` + 20d 主導 + rank。**best 20d IC = 0.0961 @ep13**（60d 0.1030、5d 0.089），比歷史 z-score 20d 的 0.051 **近翻倍**，存 `v6_trend.pt`。過擬合 ep9 起但 IC 峰在 ep13、斷線無妨。**白賺發現：scale_gate ep3 就 100% 塌 Long**——趨勢多尺度也沒加值、實質單尺度 Long（Phase 3 可砍）。**至此雙模型齊備：短線 `v6_short.pt`(5d 0.0951) + 趨勢 `v6_trend.pt`(20d 0.0961)，rank 目標把兩者各自翻倍**
- **Phase 2 步驟 2 短線模型大勝（2026-06-15，ep8 峰）**：單尺度（window 60、1.66M 參數）+ 5d/10d 頭 + listnet + **訓練時 rank 目標**。**best 5d IC = 0.0951 @ep8**（10d 0.0943）——比舊 z-score 目標的 0.049 **近乎翻倍**、追平歷史多尺度 20d 水準，乾淨 out-of-sample（val 2024–26）。功臣＝rank 目標（對齊 Spearman IC + 對台股 ±10% 厚尾離群免疫；z-score 會被暴漲跌股主導 MSE）。ep8 後過擬合（val_loss 升、IC ep10→0.080、ep12→0.067），best=ep8 存 `v6_short.pt`。附帶：單尺度 ~33 分/epoch、比多尺度 ~75 分快一倍。**按「明顯勝直接 production」規則 → 這就是 production 短線模型、免控制組**
- **Phase 1 ② 方案B deep supervision 完成（2026-06-15，ep6 峰、ep8 停）**：每分支輔助頭各自預測 5d。**ep6 三分支 aux IC 全擠在 ~0.05（Short 0.053 / Mid 0.048 / Long 0.055）**——(1) **Short 是被 gate 餓死、不是沒料**（給直接梯度就跟 Mid/Long 一樣好），推翻「Short 死」；(2) **三尺度對 5d 冗餘**（三條 aux IC 完全同步爬升/觸頂/回落，融合主 IC 0.047 沒贏單分支）→ 多尺度對 5d 沒加值、**短線模型單一分支就夠**；(3) gate 照樣全塌 Mid。附帶：DS 主 IC 峰 0.0474 略勝 baseline 0.0434，但沒明顯延後過擬合（DS 當正則只有小 bonus、可選）
- **Phase 1 ① listnet_5d 完成（2026-06-14，跑到 ep12 停）**：baseline + listnet_5d=0.5（runtime monkeypatch `multi_horizon_loss`，production 零修改、無 push）。**best 5d IC 0.0434→0.0487@ep7（幾乎追平 20d 的 0.051）**——5d 加排名損失更好預測。代價：**gate 整個塌到 Long 0.975（Mid 被丟）**，坐實「估值走 Mid、排序走 Long」、0.5 力道太大；val_loss 1.553→1.641 過擬合比 baseline 兇、IC 峰高但脆。帶走：listnet 權重=Mid↔Long 旋鈕（之後可試 0.2）、過擬合是 baseline+本輪共同主題（production 短線模型需處理正則/早停峰值）、Short 兩損失皆≈0 砍定
- **V6.2 第三次重訓（5d 主導）baseline 定稿、Phase 0 完成（2026-06-14）**：跑到 ep11 停。**gate 收斂 = Short 0.004 / Mid 0.80 / Long 0.20**——5d 目標下模型自選 Mid（3 月回看）主 + Long 輔、Short 死；證明 scale gate 會隨 horizon 表態、多尺度（Mid+Long）有效，**推翻「gate 永遠只會塌成單分支」**。**best 5d IC 0.0434@ep5（無 listnet）**，5d 可預測成立。過擬合 ep4–5 起（val_loss 最低 ep4、之後升、train 續降）。Short 真死/餓死留給 Phase 1 方案B 判定
- **V6.2 第二次重訓觀察 → 轉向 5d 主導重訓（2026-06-14）**：
  - **第二次重訓（padding mask + D1 macro ts）實跑至 ep7 後停止檢視**：scale_gate 仍在 **ep4 崩到 Long 0.987、ep7 = [0.0006, 0.0023, 0.997]**（Short/Mid <0.5%）——**`USE_PADDING_MASK=True` 沒能阻止偏 Long，推翻「padding 零值被當訊號」是唯一主因的假設**；val 已現過擬合起點（val_loss 最低 ep4、val_ic 最高 0.051@ep5 後雙雙轉差、train_loss 續降）；best IC 0.051 為 20d 且僅 ep7 中途值。觀察整理 `docs/training-observation-2026-06-14.md`
  - **關鍵領悟（兩條尺度軸）**：Scale Gate 的 Short/Mid/Long 是「看多長**歷史**」（輸入回看 20/60/252 步），5d/20d/60d 是「預測多遠**未來**」（輸出 horizon）。loss 主目標是 20d（趨勢），模型把 gate 押 Long 其實合理——硬加 gate 正則＝跟做對的模型作對。改採「把目標換短 + 讓 gate 自己表態」當診斷
  - **停掉第二次、改第三次重訓（5d 主導實驗）**：LOSS_WEIGHTS 改 `mse_5d=1.0 / mse_20d=0.3 / mse_60d=0.3 / listnet_20d=0.0`（listnet 寫死 20d，留著會把 gate 拉回 Long，故關掉；`listnet_5d` 列第二階段）；`trainer.py:678` val_ic 由 `preds[:,1]`(20d) 改 `preds[:,0]`(5d)，讓 early-stop / 最佳 checkpoint / headline IC 全部追 5d。commit `fb94fe3` 已推 origin/main
  - **git 修復**：移動筆電遺留的 stale `.git/index.lock`（6/13）+ index 損毀，已 `rm lock` + 重建；用 stash 手術只把 2 檔改動疊在 59 維 LF 基底上 commit、push 後還原本機 56 維（trainer.py 本機僅 CRLF 差異，`stash pop` 衝突，取 committed LF+5d 版即可）
- **sim_engine_v3 全量回放驗證 + O3 完成（2026-06-13）**：
  - **回放驗證**：32 個交易日（04-29→06-12）完整跑通，+0.40%、最大回撤 -0.59%、勝率 56.2%、16 筆賣出。四層退場全部正確觸發：L1（M頭×5、型態失敗線×1、Trailing Stop×1）、L2（排名穩定性消失×4、連2天出Top50×3、Uncertainty 2倍×1）、L3（Alpha連3天降×1，因持倉僅100股無法減半而正確升級全出）、L4（未滿倉、正確不觸發）。註：回放中 pattern_signals.json/history_index.json 用的是當前版而非各日歸檔版（歷史日期有輕微 look-ahead），屬已知設計限制
  - **O3**：`run_daily_inference.py` 新增 `Signal_Quality_Raw`（未截斷），排序改用 raw；下游 4 處重排序（sim_engine_v3、portfolio_checker×2、backend signals.py）改為「有 raw 用 raw」向下相容。顯示欄位 Signal_Quality 維持 ±10 截斷不變
  - **Uncertainty 校準分析**：`docs/uncertainty-calibration-2026-06-13.md`——U 與誤差相關 +0.30（27/27 天為正）；SQ Top50 五日去市場報酬 +1.66%/日完勝純 Alpha 排名；**SQ 設計獲實證支持，conformal（U5）優先度調降**
- **V6.2 重訓準備完成、訓練暫停（2026-06-13）**：
  - Colab Cell 3 重建 59 維 feature matrix 成功（8,712,228 rows × 64 cols、2,515 支、2005–2026-06-02），D1 檢查通過（VIX/TWII_Return/FED_Rate 非零，absmax=3.0 為 ±3σ 截斷邊界，預期行為），已快取至 Drive（3.0 GB）
  - Cell 4 首次執行因 `training_status.py` 為新檔案漏推 GitHub 而 ModuleNotFoundError，已補推（commit 2f7ea4e）
  - **GitHub main 目前是 59 維 config（供 Colab 拉取）；本機工作目錄已還原 56 維且不 commit**（V6.1 推論用）。⚠️ 訓練期間本機禁用 `git add -A`/`git commit -a`，否則 56 維會覆蓋遠端、Colab 斷線重連拉到錯的 config
  - **使用者這兩天移動筆電，重訓暫停**，移動完成後從 Cell 0→1→2→3（讀 Drive 快取）→3b→4 重新開始
- **O2 + D1 完成（2026-06-12）**：
  - **O2**：`run_daily_inference.py` 的 Confidence 從固定 bins（0.02/0.05）改當日 Q30/Q70 分位數制（與 scanner 邏輯一致，對分布漂移免疫），印出門檻與三級數量
  - **D1**：`clean_and_scale()` 新增 `macro_norm` 參數——`"cross"`（預設，V6.1 行為不變）/`"ts"`（Group D 12 維改 expanding time-series z-score：shift(1) 無 look-ahead、min 252 天、clip ±3σ，並印出最後交易日 macro z 值）。已驗證：前 252 天歸 0、同日同值、第 N 天 z 值與手算只用前 N-1 天完全一致
  - Colab Cell 3 改用 `macro_norm="ts"` 並加 D1 非零檢查輸出；`_train_meta` 記錄 macro_norm 進 training_status.json
  - **⚠️ 部署 checklist**：推論端維持 `"cross"`（V6.1 checkpoint 的 proj_D 未訓練，提前切換=注入隨機噪音）；V6.2 checkpoint 上線時必須同步改 `clean_and_scale(df, macro_norm="ts")`（程式內已留註解標記位置）
- **P0 推論修復完成並驗證（2026-06-12）**：`V6/run_daily_inference.py`
  - **P0-1 兩段式推論**：Mamba encoder 維持 128 股分批，GAT 改為一次吃完整 cross-section 圖（舊版每批只取批內 KG 邊，跨批邊全丟、GAT 幾乎退化成 identity）
  - **P0-2**：推論傳入 `padding_mask`（對 V6.1 checkpoint 數值零影響，為 V6.2 部署鋪路）+ MC-Dropout 以日期為 seed（`torch.manual_seed(YYYYMMDD)`），同日重跑可重現
  - **P0-3**：新增剔除統計輸出（clean_and_scale NaN 剔除數、cross-section 歷史不足剔除數）
  - **驗證結果（2026-06-12 實跑比對）**：Alpha rank 相關 0.98（模型輸出穩定）；最終 SQ 排名相關 0.92、Top10 重疊 4/10、Top50 重疊 30/50 = GAT 實質貢獻；**Uncertainty 整體 -34%**（Q50 0.046→0.032）——舊版因 GAT 失效而系統性高估不確定性；新進 Top50 呈營建/金融產業群聚，符合 GAT 沿 KG 邊傳播訊號的預期
  - **一次性轉換成本**：排名穩定性判斷（scanner 30 分權重）跨新舊排名計算，買入訊號 3–5 個交易日內可能短暫偏少，不需處理
- **整體架構分析報告（2026-06-12）**：`docs/architecture-analysis-2026-06-12.md`，資料→模型→輸出全面分析。兩大發現：(1) **D1**：`clean_and_scale()` 對 Group D macro 特徵做 per-date cross-sectional z-score，同日全股票同值 → std=0 → 整組 12 維恆為 0，模型 macro 分支無資訊（需重訓修復，改 time-series 標準化）；(2) **M1**：推論時 `INFER_BATCH=128` 只取批內 KG 邊，GAT 圖被切碎與訓練不一致。升級建議 P0~P3 見報告
- **模型訓練狀態記錄 + 模型狀態頁面改版（2026-06-12）**：
  - 新增 `V6/marketmamba/training_status.py`：`dump_training_status()` 將 TrainingHistory 寫成 `training_status.json`（含學習曲線、scale_gates、epoch 耗時、config 快照）
  - `v6_colab_training.py` Cell 4 / 4b：每 epoch 寫 JSON 到 Drive（`MyDrive/MarketMamba_V6/training_status.json`）；訓練完成補 n_parameters 與最終狀態；順手修了 resume 時 scale_gates 曲線遺失
  - **資料流定案**：Colab → Drive（訓練中）→ 訓練完成後手動複製到 `V6/results/` → git push → Render（30 分 TTL，`POST /api/performance/cache/refresh` 可強制刷新）
  - 後端 `performance.py` 整支重寫：刪除全部寫死資料（V5 世代 WF folds、固定學習曲線、math.sin 假累積報酬圖），改讀 `training_status.json` + `ic_analysis.json` 真實資料；`schemas.py` 同步改新結構；`mock_data.py` 移除 MOCK_PERFORMANCE
  - `ModelStatus.jsx`（PersonalOS 與 app/frontend 兩份同步改版）：動態訓練狀態 badge、真實學習曲線、新增 Scale Gate 三分支面板、線上 IC 時序面板、架構摘要由 config 快照動態帶出
- **repo 整理（2026-06-12）**：
  - `HANDOFF.md`（2026-04-27 舊交接文件）、`signal_scanner_plan.md`（V6.1 規劃，已在 V6.2 實作完成）移至 `archive/docs_old/`
  - `obsidian_note/`（含高度個人化內容）改為 `.gitignore` 排除、`git rm -r --cached`，保留本機 Obsidian 使用但不再公開於 GitHub
  - `OVERVIEW.md`/`PROJECT.md` 維持不動（分別供 claude.ai Project 知識庫與外部整合系統使用）
  - `archive/`（33MB 舊 notebook）維持不動，使用者表示僅為紀念用途、空間不大暫不處理
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
- [x] ~~**F6 的 2×2 第四格**~~ **✅ 已完成（2026-08-02）**，見「最近完成」第一條
- [x] ~~**待查：GRU 的 decile spread Sharpe = 2.846**~~ **✅ 已查清（2026-08-02）**：不是 bug、不是 panel 差異，見「最近完成」第二條
- **★ 唯一待跑（很便宜）：把 2×2 最佳格的 checkpoint 產成分數、過 portfolio_lab**。checkpoint 是 Drive 上的 `v6_short_GD_no_macro_gatv2.pt`。用 `V6/experimental/score_mamba_local.py` 在 WSL RTX 3060 前向一次（**約 7 分鐘、不需 Colab、不需重訓**）→ `result/scores/v2_kg_nomacro.parquet` → 丟進 `portfolio_lab`。**在這之前不可宣稱它贏過 GRU 的組合層**（訊號層贏是確定的：+0.1145 vs +0.1016）。⚠️ 產分數時 Group D 那 12 欄必須同樣歸零，否則與訓練時的輸入不一致——這是 `macro_norm`／`fundamentals_v2` 已經踩過兩次的同一種坑
- **F6 下一階段的計畫待討論（2026-08-02 更新）**：這一輪（GAT 三組 + 2×2 + GRU 診斷）已收尾。**下一輪做什麼尚未決定**，討論時可用的素材：
  1. **run-to-run σ 從沒量過**，而現在有**兩個**大結論壓在單一 seed 上（GRU 兩端優勢、Group D 負貢獻）。最便宜的補法：GRU h64 用 2~3 個不同 seed 重跑（WSL、每個約 1~2 小時），一次同時得到 σ 與 GRU 的複驗
  2. **`INPUT_DIM` 59 → 47**（砍掉 Group D）要不要正式落規格
  3. **GRU 的「兩端訊號更耐放」值得追**——它正好落在專案已確立最會賺錢的軸（低頻/省換手）上
  4. 多尺度主模型用 `epochs=15~20` 重跑（§3 那輪排程壞掉、數字不可用）
  5. 中性化用 Mamba 複驗（F5 給的是方向正但不顯著）
- **Cell 4 多尺度（10–16h、約 300–440 元）**：優先度仍低——有三條證據指向 5d 目標下多尺度會退化
- **F6（Colab／GPU）的其餘項目，特徵工程層已凍結**：① ~~GAT 三組消融~~ **✅ 已完成（2026-08-01）** ② ~~Group D 12 維消融~~ **✅ 已完成（2026-08-02，含 2×2 四格）——證實是負貢獻** ③ **中性化用 Mamba 複驗**（F5 給的是方向正但不顯著，唯一還沒做的一項）。**F6 的 config 設定**：`INPUT_DIM=59`（既有 V6.2 設定，**不需 `patch_config_67d`**，連帶避開那個 import 期綁值的陷阱）、`build_features(fundamentals_v2=True, availability_flags=False)`、`clean_and_scale(macro_norm="ts", neutralize="none")`、切分用 `splitters`（purge 60 + embargo 20）、train 起點 2013
- **研究計畫三方向（`planing/研究計畫_主檔.md`）**：**方向一 ✅ 全部完成、方向二 ✅ 全部完成**（Step 2~5 + 交付；`/pipeline` 前端頁待驗收部署）；方向三-C 等 post-P0 樣本 ≥20 天（約 7 月底）重跑複驗；方向三-A/B（事件驅動/Meta-labeling）優先級最低、先做資料可行性評估再決定投入。平行可做：資料基礎升級計畫階段二（資料衛生三項 + baseline_common 序列輸出，見 `planing/資料基礎升級計畫_baseline_common扶正.md`）
- **scanner 1.4 新邏輯待 07-07 推論驗證**：確認 ① BUY 數量合理（分數制 + 機構條件復活，乾跑 07-06 資料為 15 BUY/24 WATCH）② 機構連買明細正常顯示 ③ `condition_analysis.json` 有產出並被 push。條件貢獻分析的機構統計從 07-07 歸檔起才有意義（歷史旗標全 False）
- **Phase 3 實驗暫停中（2026-07-06 使用者決定）**：等真倉（V6.1）先驗證有沒有賺錢再繼續做實驗。目前進度停在：~~A 正則救峰值 ✅（dropout=0.2 有效）~~ → B/C/D 三支檔案已寫好推 main、**尚未在 Colab 執行**（listnet 權重 sweep / 趨勢單尺度簡化 / 短線窗口 sweep，預期結果見 `docs/phase3-experiment-plan-2026-06-25.md`）→ E/F 未設計。恢復時：在 Colab 跑 B、貼結果回來判讀
- **雙模型真實市場效益追蹤已上線（2026-07-06）**，取代原本卡在「archive 累積」的路線圖第②③步——不用再等，改成自動化每天算、頁面自動顯示，樣本量隨時間自然增加，不需要手動介入。目前 5d 樣本僅 4 天，10d/20d/60d 尚無資料
- **資料品質稽核第二輪待討論（2026-07-27）**：第一輪「純程式修改」已完成。剩下的都不是改程式能解的，見下方「下一步」——資料操作三項（清 `Close<=0` 存量、補 14 個停更源、季度全量重抓價格）、歸因一項（P2 少掉的 353 支）、需重訓或動凍結協定兩項（label 起點改 t+1、`*_is_missing` 缺失旗標會動 `INPUT_DIM`）。另有已揭露但暫不處理的：`Open` 欄與 HLC 在冷門股上來源定義不一致（OHLC 違規 322,300 列 / 3.7%）、漲跌停/處置/停牌四類交易狀態欄位完全不存在、存活者偏差未量化

### 下一步
- [x] ~~**① 拆解 F5 那個 −0.0079**~~ **已完成（2026-08-01）**，見上方「最近完成」與 `docs/feature-protocol-v2.md` §9.6
- [x] ~~**② 寫 Group D 消融腳本**~~ **已完成（2026-08-01）**：`V6/experimental/groupd_ablation.py`，改用 mask 設計、控制組沿用 `no_gat`
- [x] ~~**R0d**~~ **決定不跑（2026-08-01 使用者拍板）**：要確認的殘差 ≈ −0.0008 比實務門檻低一個數量級、不改變任何規格決策，成本與價值不成比例。文件已誠實記為「相減得出、未獨立驗證」。`v1univ` 的 chunk 留著（0.8 GB），日後想補只剩 20–30 分鐘，指令在 `docs/f6-training-log-and-readout.md` §6.1（**須在自己的終端機跑**，背景任務三次都被砍）
- [x] ~~補 Mamba 三組的組合層分數~~ **✅ 已完成（WSL 本機，非 Colab）**：`score_mamba_local.py`，各 7 分鐘，並用逐日 IC 對 Colab 記錄驗證通過
- [x] ~~**【優先級 1】外部效度**~~ **✅ 已完成（2026-08-01）**：`wf_scores.py` → 11 年連續 OOS，低頻+緩衝撐住、N=10 證實為過擬合。詳見結果文件 §7e。**剩下可選**：GBDT 的 WF（需逐 fold 重訓，較貴）
- [ ] ~~【優先級 1】原文~~：把 Ridge/GBDT 的 walk-forward 預測餵進 `portfolio_lab`。目前全部結論來自 2024-01~2026-06 的單一 **多頭** 窗口 + 1,500 格網格，這是唯一可能讓整套結論作廢的風險。WF 已有 **46 個 fold（2014–2026）**、涵蓋多個 regime → 可檢驗「20 日 / 緩衝 / N 依賴」這些**結構性**結論是否跨 regime 成立。Mamba 沒有 WF（每次 3.6h 太貴），但結構性結論本來就是關於組合建構、不是關於哪個模型好。~~子期間切分~~ 已做，**測不到趨勢混淆（前後半都是多頭）**，改用大盤區間切才有效
- [ ] **【優先級 2】確認 v1.1 修訂提案**（規格 §8）：A 改 headline、B 分數平滑、~~C 補 10 日（已生效）~~、D 大盤區間報告納入標準輸出
- [ ] **【優先級 3】設計接班系統**（scanner 退役後的替代）——**但必須等優先級 1 通過才動工**。缺口清單在規格 §5b：整張交易資金門檻、漲跌停/停牌可成交性、持倉同步、部分成交、權重漂移再平衡
- [ ] **★ WSL（下次開機第一件事，約 7 分鐘）：把 2×2 最佳格產成分數過 portfolio_lab**。checkpoint `v6_short_GD_no_macro_gatv2.pt`（在 Drive）→ `score_mamba_local.py` → `result/scores/v2_kg_nomacro.parquet` → `portfolio_lab`。**Group D 那 12 欄要同樣歸零**，否則推論輸入與訓練不一致。做完才能回答「訊號層新高有沒有換到錢」
- [x] ~~**WSL：補 GRU 分數**~~ **✅ 已完成（2026-08-01）**：`result/scores/gru.parquet`，並已過 portfolio_lab（N=50/k=1.5/20日：年化 +31.3%、Sharpe 1.401、超額 +13.9%，四階最好）
- [ ] **量 run-to-run σ**（整個 F5/F6 都在用卻從沒量過）：GRU h64 用 2~3 個不同 seed 重跑（WSL、每個約 1~2 小時），一次同時得到 σ 與 GRU 兩端優勢的複驗。**現在有兩個大結論壓在單一 seed 上**
- [ ] **確認 v1.1 修訂提案**（規格 §8）：A 改「貼近實際操作」headline 為 20 日再平衡（我原本定成每日是設計失誤）；B 新增「分數平滑」維度。兩者都是看到數字後才提的 → 採用後所有模型要在 v1.1 下重跑，新舊數字不得混用
- [x] ~~**Colab：跑 Group D 消融**~~ **✅ 已完成（2026-08-02，含 2×2 四格）**
- [x] ~~**③ macro 接進 `run_daily_update`**~~ **✅ 決定不做（2026-08-02）**：原本的前提是「等 Group D 消融證實有貢獻再做」，現在證實**是負貢獻**（2×2 兩列都是），所以不接。V6.1 的 regime 閘門會維持 N/A——但 V6.1 已非紅線，且 Group D 對 V6.1 本來就被 cross z-score 歸零、無影響
- [ ] **`INPUT_DIM` 59 → 47 要不要正式落規格**（砍掉 Group D 12 維）：證據已足（2×2 兩列一致、可加、t=2.93/3.23），但這會動特徵協定與 `FEATURE_GROUPS`，等下一輪計畫確定後一起做
- [ ] **F5 的變體快取可刪**：`baseline_cache_v2_v1like`（~7.5 GB）已無後續用途、確定可刪；`baseline_cache_v2_v1univ/chunks`（~0.8 GB）留著當 R0d 的續跑起點（很小，不急）。D 槽現剩約 207 GB，沒有空間壓力
- [ ] **F6 開跑前的 checklist**（比原「V6.2 部署 checklist」更新，以此為準）：`config.py` 維持 `INPUT_DIM=59` + `FEATURE_GROUPS` 取消 RS 注釋（不做 67/66 維 patch）；`build_features(fundamentals_v2=True, availability_flags=False)`；`clean_and_scale(macro_norm="ts", neutralize="none")`；Colab 的兩個切分點已接 `splitters`（G1 已補做）；Drive 上的舊 matrix 快取要刪除重建
- [ ] **F5 的兩份變體快取可刪**（決策已定案）：`Data/processed_v6/baseline_cache_v2_nofund`（7.3 GB）確定可刪；`baseline_cache_v2_neuind`（7.7 GB）**建議先留**——`NEUTRALIZE` 是延到 F6 的未決項，留著省一次 27 分鐘重建。D 槽現剩 55.3 GB
- [ ] **選配：拆解 R0b−R0 的 −0.0079**（宇宙過濾 vs 外資持股回補），需再建一份「v2 規格 ＋ v1 宇宙規則」的矩陣，約 27 分鐘。不影響 F6，屬「想知道」而非「必須知道」
- [x] ~~`/pipeline` 頁驗收 + 部署~~ **已於 2026-07-15 完成並 push**（commit `3cc7bb1`，已確認在 origin/main 上，Vercel 應已自動部署）——CLAUDE.md 這條之前漏更新，2026-07-23 核對 git log 才發現其實早就做完，補記於此
- [ ] **資料基礎升級計畫階段二**（方向一/二收線後的下一個工程主線，不需 Colab）：資料衛生三項（Close=0 gate、還原接縫、超限複驗）+ baseline_common 序列輸出 (N,252,59) + KG 邊介面 + 協定 v2.0 版本化
- [ ] **~7 月底重跑 `V6/experimental/conviction_c_analysis.py`**：post-P0 5d 屆時 ≥20 天、post-P0 20d 開始有樣本；同步對 `results/archive/df_short_*.csv` 的 SQ_5d/Unc_5d 做同款校準分析（horizon 對齊使用者操作週期，腳本小改即可）——5d 校準反向是否為真在此定奪，屆時再決定 deep ensembles/conformal 要不要做
- [x] ~~**重複寫入來源追查**~~ **已於 2026-07-27 定案並修復**：根因是 `fetcher.py:1044` 多來源合併時 Date 型別不一致（yfinance `Timestamp` vs TWSE/TPEX direct `str`）導致 `drop_duplicates` 失效。已修 + `_append_to_parquet` 加保險絲，當晚推論即自動清掉存量 11,384 列。**但「非交易日不寫入」gate 仍未做**（06-07 週日、06-19 端午的假資料是另一條線：來源在非交易日回了資料，與型別 bug 無關）
- [ ] **每日更新寫入端補「非交易日不寫入」gate**：查 TWSE 交易日曆後才寫入，防止 06-07（週日）、06-19（端午）那類整日假資料再發生
- [ ] `Data/processed_v6/prices_raw_backup_20260712.parquet`（127MB）：推論穩定跑幾天後可刪（與既有兩個 institutional backup 一起）
- [ ] **PersonalOS 同步 K 線圖**（Vercel 版已驗收）：複製 `KLineChart.jsx` + `StockModal.jsx` + `TradingSignals.jsx` 過去（注意 import 路徑差異 `../api/market` → `../../api/mm`），PersonalOS `npm install klinecharts@^9.8.10` + `npx vite build` + 重啟 exe
- [ ] **macro_raw 停在 2026-04-24**：每日更新本來就不含 macro → regime 閘門（TWII vs MA60）恆為 N/A、保守模式從未啟用（scanner 已加 fallback+新鮮度檢查，macro 一更新就自動生效）。對 V6.1 推論無影響（Group D 被 cross z-score 歸零）、對雙模型排名免疫（橫斷面常數）。要啟用需把 macro 加進每日更新
- [x] ~~**prices_raw 每日 ~800 筆重複寫入**~~ **已於 2026-07-27 修復**（同上，型別 bug）。健檢確認「今日 0 列 / 近 90 天 0 列 ✓」
- [ ] **B-3 步驟 3（重抓完成後接續）**：① 用重抓的原始價 + `dividend_raw` 公式建上櫃因子表（記得：`CashIncreaseSubscriptionRate` 不可用、股票股利 ÷10）② 合併官方（上市）+ 自算（上櫃）因子 → 套用累積還原 → **先寫新檔驗證再切換** ③ 2007-07 前上櫃段標記為未還原
- [ ] **接續資料修復主線（優先序）**：① 7 個資料源改直連（`foreign_shareholding`/`dividend`/`futures_inst`/`options_inst`/`holdings`/`business_indicator`/`fear_greed`，模板照 margin/daytrade）+ `run_daily_update` 從 5 個源擴到 14 個。（`dividend_raw` 停在 2026-05-05，但**已不再阻塞 B-3**——上櫃改用 TPEX 官方因子） ② ~~清 `Close<=0`、回補 04-27/04-28~~ 已完成 ③ ~~興櫃排除~~ 已完成（B-2）
- [ ] **B-5：`fetcher.py` 寫入 `stock_info` 時去重**（源頭修）。`industry_category` 多值需先決定保留策略（取第一筆或串接）。在那之前，新程式 join `stock_info` 一律先 `drop_duplicates(subset=["stock_id"])`
- [ ] **清 329 列 `Close<=0` 存量**（2026-04-30 ~ 05-22、122 支）：`_sanitize` 已在讀取時剔除、推論不受影響，但 parquet 內仍在，訓練/回測若直接讀 raw 會踩到（2026 年 515 筆極端報酬中 422 筆源自此）。屬資料操作，需先備份
- [ ] **補 14 個停更資料源**（`margin_raw`/`per_raw`/`market_value_raw`/`daytrade_raw`/`securities_raw`/`fear_greed` 停在 2026-04-24；`foreign_shareholding`/`dividend`/`futures`/`options` 05-05；`holdings` 05-08；`revenue` 04-01；`financials` 03-31；`business_indicator` 02-01）。59 維中約 27 維目前是凍結值或常數（`Day_Trade_Volume` 04-24 後恆為 0 → z-score 後整維歸零）。健檢每天會報，修好會自動變 ✓
- [ ] **P2：股票池 2026-05-25 少掉的 353 支歸因**（2,321 → 1,968，一日之間，非下市）：查該日來源切換，確認是抓不到還是真下市。持續流失的 16 支樣本：2073、2321、3064、3426、3531、3629、4183、4413、4530、4584、5703、6680、6865、8077、8093、8923
- [ ] **定期全量重抓價格以修正除權息還原**：prices_raw 逐日增量寫入 → 未來的除息永遠不會回頭調整已寫入的歷史（2412 於 2026-07-09 跌 6.0 元即疑為未還原除息）。歷史段（2015–2026-05）經 24,357 筆事件檢定確認已正確還原，問題只在增量段。建議季度全量重抓一次
- [ ] 條件貢獻分析累積 20+ 天後回頭校準四條件權重（30/25/25/20）與 70/90 門檻；屆時也評估「發現 3：掃描池擴到 Top200（型態+分數過門檻才進 BUY）」要不要做
- [ ] `Data/processed_v6/institutional_raw_backup_*.parquet` 兩個備份檔（各 ~148MB），scanner 穩定跑幾天後可刪
- 若之後要恢復 Phase 3：見上方「進行中」，直接跑 B
- **Phase 4 產業理解融合（2026-06-27 計畫定稿、待 Phase 3 完成後接續）**：細節全在 `docs/phase4-industry-chain-fusion-plan-2026-06-27.md`。4-A KG 豐富化（產業鏈上下游邊進模型）+ 4-B 產業變化 Agent（FinMind/MOPS 訊號 + LLM 語氣 diff，獨立層不碰 V6.1）。動工前仍須依規則 2 列實作計畫、確認後再動
- **雙模型篩選條件**（原路線圖②，尚未做）：在 dual rank-score 上做 SQ 門檻/低不確定性/short∩trend 交集/跨日排名穩定（需要 history 累積才有「穩定性」可言）；③ 真正的 dual 模擬機器人（比照 sim_engine_v3 跑紙上交易含進退場）仍是之後才做——**目前的雙模型驗證只算 IC/Top50 超額，不是完整模擬交易**，兩者不要混淆
- **雙模型 Roadmap（2026-06-14 定案，不急著取代 V6.1）**：原則＝診斷實驗先做完（隔離在 `V6/experimental/` 副本、零 production 影響、一次一個變因），最後才動 production 雙模型、下游 plumbing 只做一次
  - **Phase 0（✅完成）**：5d baseline ep11 停定稿——gate 收斂 Mid 0.80/Long 0.20/Short~0、best 5d IC 0.0434@ep5、過擬合 ep4–5 起
  - **Phase 1（experimental 副本診斷）**：①✅ `listnet_5d`——5d IC 0.0434→0.0487（近 20d），但 gate 全塌 Long、過擬合加劇 ②✅ deep supervision——**Short 餓死非沒料、三尺度對 5d 冗餘 → 短線模型單分支即可**（主 IC 0.0474 略勝 baseline）③（可選，前提已削弱）3d 測試 ④（降級，B 已證冗餘）窗口階梯 {60,126,252}
  - **Phase 2（雙模型「並行、不動資料流」上線——首發即優化版）**：安全網是 V6.1，雙模型不必先上保守版，故把便宜高把握的改進折進來：**目標改每日 cross-sectional rank** + **listnet**。短線=單尺度 5d/10d+listnet_5d、趨勢=多尺度 20d/60d+listnet_20d。獨立 `run_dual_inference.py` + 獨立輸出檔（自建 59 維），V6.1→df_kelly→dashboard 完全不動、失敗也不影響線上。短線 run 兼當 rank-vs-raw 證據（對照舊 multi-scale raw 0.049）。技術點：56 vs 59 特徵排列不同（RS 插 group A 中間）
  - **Phase 3（進行中，雙安全網下迭代抬 IC，2026-06-25 拍板 A→F 順序、一次一變因）**：
    - **A 正則救峰值（起手，檔案備妥待跑）**：`V6/experimental/phase3_a_dropout_sweep.py`，短線 dropout sweep 0.1/0.2/0.3，看峰值 IC 能否 >0.0951／延後／崩壞變緩。dropout 有效再帶進趨勢；不夠則下一變因試 weight_decay（1e-4→1e-3）
    - **B** listnet 權重 sweep（Mid↔Long 旋鈕，試 0.2/0.5）→ **C** 趨勢單尺度簡化（gate ep3 就塌 Long、多尺度沒加值，砍成單尺度確認 IC 不掉、未來訓練更省）→ **D** 短線窗口 90/120（≤252 純切片免重建）→ **E** 多 seed 集成（對抗 IC 脆弱、部署穩定加成）→ **F** 特徵分離（短線快/趨勢慢特徵，重建 feature matrix 大工程、潛力最大、留最後）
- [ ] 若決定不走 5d 路線：把 LOSS_WEIGHTS / val_ic 改回 20d（remote main 目前是 5d 實驗設定）
- [ ] **本機 git 善後**：第三次重訓 push 後 `git stash pop` 在 trainer.py 留下 CRLF 衝突——需 `git checkout HEAD -- V6/marketmamba/models/trainer.py` + `git restore --staged V6/marketmamba/config.py` + `git stash drop`
- [ ] **V6.2 部署 checklist**：config INPUT_DIM=59 + FEATURE_GROUPS 取消 RS 注釋；`run_daily_inference.py` 的 `clean_and_scale` 改 `macro_norm="ts"`（程式內有註解標記）；**`build_features` 加 `fundamentals_v2=True`（2026-07-27 新增，訓練端與推論端必須同時切；2026-07-28 起這個旗標同時控制 PER/PBR 自算與橫斷面校準，不另加旗標）**；checkpoint 換新
- [ ] 觀察 3–5 個交易日：排名穩定性恢復情況、買入訊號數量是否回歸正常
- [ ] P0 後累積 20+ 天 archive 重跑 Uncertainty 校準分析（`docs/uncertainty-calibration-2026-06-13.md`，結論：SQ 設計獲實證支持、conformal 優先度降低）
- [ ] 下次重訓後驗證模型狀態頁面：Drive JSON → V6/results → push → 頁面顯示
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
- **★ Group D（總經 12 維）不但沒貢獻，還是顯著的負貢獻 → 那三個資料源不用補了（2026-08-02）**：Δ=−0.0186、NW t=−3.12、`no_macro` 十個 epoch 全勝。**機制**：Group D 是每日橫斷面常數，在排序上零資訊，卻給了模型「記住是哪一天」的通道 → **製造過擬合**（`with_macro` val_loss ep2 後單調惡化，`no_macro` 一路降到 ep6）。→ **推翻既有決策「先證實有貢獻再補來源」的前提**——證實了是負的，所以 `fear_greed`/`business_indicator`/`fed_rate` **不用補**，`INPUT_DIM` 應考慮 59 → 47。⚠️ 幅度被排程截斷（峰值在 ep9/共 10），但方向不受影響
- **穩健的結論是「不要每日再平衡」，不是「20 日最好」（2026-08-02，GBDT WF 修正）**：11 年 WF 下 **Ridge 單調到 20 日（+20.7%）但 GBDT 在 5 日就見頂（+26.2% vs 20日 +21.9%）**。兩者唯一一致的是**每日再平衡都是災難**（Ridge −12.1%、GBDT −5.0%）。→ 之前寫的「20 日最好」要降級成「Ridge 上成立、GBDT 上不成立」；跨模型穩健的只有「不要每日」。緩衝在 20 日下對兩者都只有微幅效果
- **存活者偏差其實不存在，長期限制可以劃掉（2026-08-01）**：CLAUDE.md 掛了很久的「存活者偏差未量化」——實測 `prices_raw` 有 **211 支（9.7%）已下市/長期停牌，且它們在下市前的交易日都留在資料裡**（WF 11 年窗內 146 支、582 天窗 19 支）。原因是 **B-3 重抓價格是逐日從交易所抓的**，自然保留到最後交易日。連帶：**下市可由資料推導、不需要找資料源**。剩下的真問題是「持股下市時怎麼認列」——目前 ffill = 「以下市前最後成交價認列」，若改成回收 0，曝險是 Top50 的 0.25%~0.56% → **最壞影響上界 <1pp/年**
- **配權與平滑沒有跨模型一致的贏家 → 基準版維持等權（2026-08-01，推翻我自己稍早的結論）**：v1.1 擴充網格（108 組/模型 × 7 份分數）跑完後，**方向相反**——配權 Sharpe（N=50/k=1.5/20日）wf_ridge 是 inv_vol **1.526** > equal 1.430，但 **v2_kg 是 equal 1.487 > inv_vol 1.310**；平滑同樣不一致（wf_ridge 變差、ridge 582天 變好）。**我稍早說「波動度倒數是 WF 驗證下最好的」是只看 wf_ridge 一個模型的結論**，補齊六個模型後不成立。→ 基準版維持等權（與檢核表原文一致：等權最單純、不引入額外變因）。**這是「多一個模型/多一個窗口就翻盤」的第五個案例**
- **漲跌停不需要新資料源，而且影響是正的（2026-08-01，我預期錯了兩次）**：風控 C 類原本記為「做不了、要另接 TWSE 公告」，但**台股漲跌幅上限可由還原收盤價的日報酬直接推導**（2015-06 起 ±10%、之前 ±7%）。① **第一個預期錯**：以為動能模型會富集漲停股，實測 Top50 的漲停比例只有全市場的 **0.1×（少 8.5 倍）**——模型系統性避開剛漲停的股票（label 是前瞻報酬、漲停後傾向均值回歸）② **第二個預期錯**：以為加上限制會讓報酬下降（回測系統性偏樂觀），實測**定案的低頻設定下反而變好**——WF 11 年 N=50/20日 +20.7%→**+21.0%**、v2_kg +26.0%→**+26.8%**，**空頭區間也不吃虧**（下跌段 −14.7%→−14.0%）；只有高頻（5 日）才吃虧 −1.23pp。機制：被擋掉的買單避開追高、被迫續抱的跌停股吃到均值回歸。→ **「回測系統性偏樂觀」就漲跌停這一項而言不成立，影響在 ±1pp 內**。⚠️ 仍未涵蓋：處置股/注意股/全額交割股/下市，那三項要另接 TWSE 公告、偏誤仍未量化
- **GRU 那組不可比、不要拿來補四階對照（2026-08-01）**：`result/gru_5d.pt` 是 2026-07-14 訓練的，用 **v1 快取 + 舊資料**（在除權息還原與整批資料修復之前），與 Ridge/GBDT/Mamba 三組共用的 `baseline_cache_v2` 不是同一個基礎。要可比得**重訓**（h64-only 約 1–2 小時；原版 12.2 小時裡有 ~11 小時浪費在 h128），不是「評分 10 分鐘」
- **圖的價值在「不要有錯的邊」，不在「有更多邊」（2026-08-01，v3_kg 實測後才看清楚）**：三個數字放一起——加雜訊圖 `B−A = +0.0055`（t=0.91 不顯著）／**把垃圾邊換成合理邊 `C−B = +0.0052`（t=5.17 顯著）**／**在合理邊之上再加 4,504 條相關性邊 `D−C = −0.0002`（無效應）**。逐 epoch |Δ| 最大只有 0.0008，兩條曲線幾乎完全重疊；10d 同樣（+0.1051 vs +0.1054）。→ **直接下修 Phase 4-A 的期望值**，因為 4-A 正是「在 v2 之上再加產業鏈邊」，與 v3 同一種操作。**也推翻了使用者 2026-08-01 的樂觀讀法**（「連這麼陽春的圖都有效果，代表通道是活的、更完整的圖仍有空間」）——那個外推被直接測了，結果是沒有空間。**公平的保留**：相關性邊是從價格學的（模型本來就摸得到一部分），產業鏈邊是模型沒有其他管道知道的外部關係知識，資訊類型不同，不能完全外推。**附帶發現：GAT 對無用的邊是穩健的**——多給 5 個沒訊號的鄰居，注意力把權重壓低，結果沒變差（最大退步 0.0008）
- **Conviction 線（多模型共識選高信心股）已做零成本前測，結果要注意（2026-08-01）**：使用者規劃「訓練多個模型、交叉比對排序找高信心股票」，用集中持股繞開散戶資金限制，同時把整條路走過一次當作品集。**五個模型的分數都在同一批 582 天上 → 不用訓練就能前測。** 前提診斷成立（跨家族 Top50 只重疊 ~20%，有分歧可用），但四種共識法（5模型平均／3家族平均／一致性過濾／同意票數）**在 18 個切面裡只贏 3 次，且全部集中在 N=10/10日 那一格**＝證據強度接近雜訊。**一致的型態是「共識法比較保守」**：下跌段保護明顯較好（N=50/10日：v2_kg −21.5% vs B −9.0% vs C −10.5%）、上升段參與較少，在這段多頭裡淨效果是輸的。⚠️ **這是第二條指向同一方向的證據**——方向三-C（2026-07-11）也量到「低 U 硬門檻交集」輸給「SQ 連續值」。**兩次都是取交集/共識輸給單一連續分數。** → **不是否定計畫**，但這五個模型本來就不是為 ensemble 而建（同特徵、同標籤、同訓練期，其中三個同架構、old_kg↔v2_kg 相關 0.967）→ **重點應放在製造真正的多樣性（不同 seed/horizon/標籤/特徵子集），而不是拿現有高度相關的模型投票**
- **scanner 訊號系統跟 V6.1 一起退役，由組合建構層接班（2026-08-01 使用者決定）**：使用者說明現行的 `scanner.py`（複合分數 ≥70 就 BUY）＋ `signal_conditions.py` 四層退場是「上課前自己想出來的粗糙策略」，上完課後判斷用組合建構那一套會更好 → **不做舊系統的換手稽核，直接讓它退役**。**這改變了 `portfolio_lab` 的定位**：它不再只是「量尺」，而是**接班系統的規格**。連帶要退的下游：`action_signals.json`、`condition_analysis.json`、`portfolio_exit_check.json`、`sim_engine_v3`、dashboard 的 TradingSignals 頁。⚠️ **接班系統需要回測沒有模型的東西**：整張（1,000 股）交易的資金門檻、漲跌停/停牌的可成交性、實際持倉同步、部分成交
- **N 的選擇會被資金門檻限制住（2026-08-01 實測，回測沒有模型這件事）**：等權買滿 N 檔各 1 張的最低資金（2026-06-02 的 `v2_kg` 排名 × 收盤價）——**N=10 約 65 萬、N=25 約 205 萬、N=50 約 365 萬、N=224 約 1,437 萬**。而實測顯示 Mamba 家族是**分散型訊號**（N 越大越好），**與資金門檻方向相反**。→ 定 headline 與上線規格時必須把可投入資金講清楚；盤中零股可買碎股但流動性與價差較差
- **風控層基本版：A/B/E 已實作，C 類是資料缺口做不了（2026-08-01，依使用者提供的《風控層基本版檢核表》）**：A 單股上限（`WEIGHT_CAP`）+ **單一產業佔比改以權重計**（非等權下用檔數會低估）+ 40% 觀察警戒；B 年化波動、滾動 20 日波動、**回撤持續期間**、處於回撤中佔比、回撤 >15% 天數；E **滾動 20/60 日 IC** + 逐年 IC + 換手標準差。**C 類（漲跌停鎖死/處置股/下市）做不了——`prices_raw` 沒有這四類交易狀態欄位**，要另接 TWSE 公告；在那之前回測系統性偏樂觀。C 類唯一可確認的一項 ✅：組合層、訊號層、資料層**共用同一套還原價格序列**，沒有各自維護。**實測重點**：`wf_ridge` MDD −29.4% 聽起來還好，但**最長連續 289 個交易日（約 14 個月）在水面下、80.1% 的日子處於回撤中**——檢核表說「回撤持續期間比 MDD 更有判讀價值」是對的。**兩個要盯的訊號**：① **2026 年滾動 60 日 IC 是 11 年最低**（wf_ridge +0.065 vs 歷年 +0.094~+0.130；v2_kg 也是 2026 最低）——樣本少、列入觀察不下結論 ② 滾動 20 日波動最高衝到 55.6%（中位 9.9%）。**交叉驗證**：`signal_health` 算出 v2_kg mean IC +0.0988 vs `kg_ablation` 記錄的 +0.0991，兩條不同計算路徑對得上
- **統一的故事：換手是唯一的槓桿，低頻／緩衝／分數平滑三種工具都在拉同一根桿子（2026-08-01，WF 11 年驗證）**：緩衝的 Sharpe 增益 3日 +0.38 → 20日 +0.08；分數平滑 5日 0.921→**1.181**、20日 1.430→**1.416（零增益）**。**拉到底（20 日 + 緩衝）之後第三種工具就沒有 Sharpe 上的空間**，但平滑仍把換手從 76% 壓到 60%——那對本回測沒有模型的滑價與容量有意義
- **配權：波動度倒數最好，但一定要設權重上限（2026-08-01）**：原始比例配權在台股會退化成單一持股——實測 `liquidity` 最大單股權重 **84.95%**（整個組合押一檔台積電）、`inv_vol` 28.5%。套 `WEIGHT_CAP=3.0×` 等權上限後，**WF 11 年 inv_vol Sharpe 1.526 > equal 1.430 > liquidity 1.156**（MDD −27.1% vs −29.4%、下跌段 −10.3% vs −14.7%，代價是年化少 3.1pp）。⚠️ **流動性加權是第三個被 WF 抓到的單窗假象**（582 窗 Sharpe 1.402/1.545 看起來最好，11 年卻最差）。**換手算法一併改用權重變動**（檔數比例在非等權下失真）；等權回歸檢查 +10.8%/76% → +10.7%/77%，既有結論不受影響
- **`knowledge_graph_v3.npz` 已建好、待訓練驗證（2026-08-01）**：`V6/experimental/kg_builder_v3.py`。動機是 §7c 的機制發現（GAT 的作用是**下跌段避開一起崩的股票**），而相關性邊直接編碼「誰跟誰一起動」，比產業鏈邊少隔一層。**PIT 安全**：只用 2018~**2023-12-31**（F6 訓練切點）的報酬 → 對 2024+ 零 look-ahead，圖仍靜態、**不需改架構**。**關鍵設計：先減每日橫斷面均值去市場因子**（不做的話兩兩相關全是 0.3~0.6、選出來只是高 beta 股；實測去因子後非對角 median −0.003）。鄰居抽驗**明顯合理**（2330 → 日月光/聯電/華邦電/南亞科/聯發科；2891 中信金 → 華南金/兆豐金/第一金/合庫金），對照舊圖 2330 的鄰居是「電器電纜、綠能環保」。**改用 `additive` 模式（2026-08-01，比原設計乾淨且省一組 GPU）**：`v2` 的 32,083 條邊**逐條保留**、相關性邊加在上面（每節點 ≤5）→ **與 v2 的差異只有「多了相關性邊」一個變因**，不再有「加新邊擠掉舊邊」的混淆。可行的關鍵是實測 **`trainer.build_kg_csr()` 原樣載入 npz、不再截斷** → 超過 `MAX_NEIGHBORS_GAT=15` 的邊會真的被用到，**不需改 config、不需碰受保護的 trainer.py**，也不必為對照再訓練一組 `v2_n30`（省 3.6h / 約 100 元）。**門檻依分布定不依結果定**：殘差相關 p50 −0.003 / p90 0.091 / **p99 0.259** → 初版 0.35 在 p99 以上只覆蓋 33.5%（劑量太小），改用 **0.20（≈p98）** → 覆蓋 **70.2%**、淨增 **+4,504** 條邊（權重分布 0.5/0.8 的數量與 v2 完全相同，證實 v2 邊未被動到）。**驗證亮點：廣宇（2328）本來就是鴻海集團成員但手工表為求準確沒收錄，相關性邊自己把它找回來了**
- **外部效度已驗證：低頻 + 緩衝跨 11 年成立，但 N 的取值沒撐住（2026-08-01）**：新增 `V6/experimental/wf_scores.py` 產生 **2015-04~2026-06 連續不重疊的 OOS 預測**（4,552,371 列 / 2,724 天，expanding WF、每季重解、purge 5+embargo 20、α 固定）。**撐住的**：① 頻率單調（1日 −11.5% → 20日 +20.8%/Sharpe **1.434**，比 582 窗的 1.233 還高）② 緩衝有效但隨頻率降低而遞減（Sharpe 增益 3日 +0.38 / 5日 +0.23 / 10日 +0.10 / 20日 +0.08——**緩衝與低頻攻擊同一個問題：換手**）③ 下跌段保護（976 天真空頭：基準 −32.1%、20日 **−14.6%**）④ **逐年 12 年裡 10 年正超額，2018 空頭年策略 +15.1% vs 基準 −6.8%（超額 +21.9%）**，唯一大輸是 2020（−13.1%，COVID 後急彈跟不上）。**沒撐住的**：**N=10 是 582 窗的過擬合**——WF 最佳落在 **N=25**（Sharpe 1.500），N=10 只有 1.110，且 profile 平坦得多 → §7b「Ridge 是唯一頭部集中型」要打折。⚠️ **Mamba 三組沒有 WF（每次 3.6h），N 依賴仍只有單窗證據**
- **組合層的槓桿比 IC 層大一個數量級（2026-08-01 實測）**：緩衝規則單獨值 **+5.6pp 年化**、再平衡頻率 5 日→20 日值 **+7pp**、分數平滑把換手砍半——而 IC 上量過最大的改動（F6 GAT +0.0107）只值 1–2pp。**年化成本 20.6% 是整個系統最大的漏水口**。這不代表 IC 不重要（decile spread 證明排序確實有效），而是**在修好交易頻率之前，IC 的改進換不到錢**
- **固定用 Top50 當量尺會系統性偏袒特定模型（2026-08-01，機制已量出）**：Ridge 的訊號集中在頭部（N=10 Sharpe 1.126 → N=224 0.524），GBDT 相反（0.264 → 0.551）且整體 decile spread 更強（1.689 vs 1.064）。**四次「IC 與組合層不同調」的機制就是這個。** → 過去所有以 Top50 年化排序做的模型比較都帶這個偏誤，引用時要標明
- **★ 拿掉一個雜訊來源，比加一個訊號來源更能讓其他效應現形（2026-08-02，2×2 實證）**：GAT 效應在 with macro 下 Δ=+0.0107 卻只有 t=1.73（std(Δ)=0.0792），在 no macro 下 Δ 縮到 +0.0075 但 **t 衝到 3.80**（std(Δ)=**0.0273**，小 2.9 倍）。**效應變小、顯著性卻大幅上升。** 這是繼 GAT 消融「C−B 離散度小 5.6 倍」之後的第二次。→ **量新效應之前，先把已知的雜訊源清掉**，比多跑幾組划算；也意味著**過去在 with-macro 基礎上做的所有消融，t 值都被系統性壓低過**
- **★ 同一把尺重算，是跨腳本比較的前置作業，不是可選的嚴謹（2026-08-02，GRU 診斷實證）**：GRU 的 +0.1018 出自 `baseline_rnn`、v2_kg 的 +0.0991 出自 Colab `kg_ablation`——**兩條計算路徑**。用同一份 `Alpha_5d`、同一段日期、同一支程式重算後才發現 GBDT +0.1027 ≈ GRU +0.1016，原本以為的排名並不成立。→ **凡是把不同腳本產出的指標並列成表，先用單一路徑重算一次**；順帶：重算值 +0.0989 對上 Colab 的 +0.0991，這個吻合本身就是「兩邊同尺」的證據，值得每次都留一個這樣的對照點
- **★ 異常值先問「是假象嗎」，用可證偽的清單一條一條殺（2026-08-02，GRU decile Sharpe 2.846）**：四個假象假設（panel 建構、切分洩漏、小型股墊高、少數幾天/下市邊緣列）**全部被自己的量測排除**，剩下的才是真發現。關鍵在每個假設都事先想好「若成立會在哪個數字上留下指紋」：panel → 逐對比對；小型股 → 限制高流動子集後優勢應消失（實際**反而變大**）；少數幾天 → 去極端 N 天。→ 這是「假設要能被自己的實驗殺死」的正向版本：**先列出可殺死它的檢定，再去跑**
- **decile spread Sharpe 對窗長非常敏感、不可跨窗並列（2026-08-02）**：`wf_ridge` 3.007 / `wf_gbdt` 4.136 看似比 GRU 的 2.846 還高，但那是 11 年窗；同一個 Ridge 在 582 天窗只有 1.064。→ 這個指標只在**同窗**內比較有意義，寫進任何表格都要標窗長
- **凍結規格的價值，在它讓我自己的錯誤現形（2026-08-01）**：我事先定死的「貼近實際操作」headline 用了每日再平衡，結果 Ridge −16.3% / GBDT −31.6%。**如果我看完數字才定 headline，我會選 20 日，然後永遠不會發現自己把「每天看 dashboard」誤讀成「每天換股」。** → 失誤如實留在文件裡，修訂走 v1.1 提案並記錄理由，不偷改
- **GAT 採納為有用（2026-08-01 使用者拍板），但幅度小、且對 Phase 4-A 的意義未定論**：`C−B = +0.0052`（t=5.17）在統計上非常確定，幅度卻低於跑前定的 +0.009 門檻。**使用者的採納理由值得記下來，因為它比單看幅度更有資訊**：v2 圖是**刻意做到最陽春的**——只有產業邊 32,047 條（0.5）＋ 集團邊 **36 條**（0.8，集團表只有台塑/遠東/統一/鴻海**四個**），供應鏈邊與動態相關性邊都刻意沒放；**連這種圖都有效果，代表「關係資訊」這個通道是活的**，而那是 59 維特徵矩陣裡完全沒有的東西。→ 文件同時記錄我的悲觀讀法（邊際遞減）與使用者的樂觀讀法（通道是活的、更完整的圖仍有空間），**兩者都是單一數據點的外推，不採納任何一方為結論**。能分辨的實驗：Phase 4-A 建好產業鏈邊後，用同一套乾淨配對（同架構同 seed、只換圖）量 `v2_kg + chain` 對 `v2_kg` 的增量
- **消融的「乾淨配對」比「多跑幾組」值錢得多（2026-08-01，GAT 消融實證）**：同一輪三組，B−A 的 t 只有 0.91、C−A 只有 1.80，唯獨 C−B 到 **5.17**——差別純粹在於 B/C 同架構同 RNG，配對差的離散度小 **5.6 倍**。**如果三組都設計成架構等價，這一輪本來可以得到三個顯著的結論，而不是一個。** → 之後所有消融一律用 mask/換內容，不用「拿掉模組」（`groupd_ablation.py` 已照這個原則寫）
- **量兩份快取的差之前，先查兩份快取的「建立日期」（2026-08-01，代價是一個錯了三週的歸因）**：F5 從 2026-07-30 起就把 `R0b − R0 = −0.0079` 記成「宇宙過濾 + 外資持股回補」，但 R0 的矩陣建於 **07-12**，而 07-27→07-31 之間整批 raw 被翻修（含 07-29 的除權息還原切換）——**最大的變因根本沒被列進候選**。快取檔的 mtime 是一秒就能查的東西，卻沒查。→ **凡是比較兩份快取/矩陣的結果，第一步先列出兩者的建立時間與期間的資料異動**
- **假設要能被自己的實驗殺死（2026-08-01，同一件事上連殺兩個假設）**：① 「殘差主要來自宇宙過濾與外資持股回補」被檔期事實推翻 ② 改猜「來自除權息還原」，也被兩個檢查推翻（事件密度相關 +0.059 方向相反；價格類特徵的橫斷面 ρ 全部 ≥0.985）。**第二個假設聽起來非常合理**（未還原＝除息日假跌幅＝可預測＝虛假 IC），但它忽略了「還原是逐股的單調重新縮放、對橫斷面排名幾乎免疫」。→ 提出機制假設之後，要先問「這個機制會在哪個可觀測量上留下指紋」，再去量那個指紋，而不是直接把假設寫進結論
- **消融一律設計成「所有組架構等價」（2026-08-01 落實到 `groupd_ablation.py`）**：Group D 消融原計畫是砍成 47 維，改成把那 12 欄**值歸零**。理由是砍維度會改 `GROUP_DIMS` → `FactorGroupedEmbedding` sub_dim 重新分配、參數量變、RNG 消耗不同 → head 初始化與 DataLoader 順序都分岔，完全重演 GAT 消融 A vs B 的三個干擾項。**代價（`proj_D` 殘留一個 bias 常數）遠小於干擾項的代價**
- **背景任務不適合跑長時間的矩陣 build（2026-08-01）**：三次都在 `clean_and_scale` 被外部終止且無 traceback、無殘留程序、無系統事件，而同機制下另一份 build 卻跑完 38 分鐘 → 真因在容器內診斷不出來。**不要靠重試碰運氣**（已試三次就停），這類工作直接給使用者可貼的指令自己跑，符合既有的執行分工
- **V6.1 已非紅線（2026-08-01 使用者明講）**：台股走跌、家人看盤興致大減、且他認為 V6.1 參考價值越來越低 → **需要的話可以停掉**。先前所有「附加不改既有／隔離在 experimental／production 檔案逐次授權」的繞路成本因此大幅下降。⚠️ 但他說的是「如果需要」不是「現在停」——**不要主動去停或破壞 V6.1**；遇到「為避開線上而要付明顯額外成本」的設計時，把兩個選項與成本差異攤開來問，不要預設繞路
- **`epochs` 不該同時當「最多跑幾輪」與「OneCycle 排程長度」（2026-08-01，Cell 4 實證）**：`epochs=100` 讓暖身長達 15 個 epoch，模型在 ep7 就過擬合、峰值時 LR 只有 max 的 47% 且還在爬，從頭到尾沒進入退火階段 → 峰值 IC 只有 +0.0526，而同架構家族的單尺度（`epochs=10`、排程走完）是 +0.0884。**有 early stopping 時這兩個角色必須解耦**。連帶：消融的三組**必須用相同的 `epochs`**，只補跑其中一組會讓 LR 曲線不同、結論作廢
- **固定 seed 不等於完全隔離（2026-08-01 實測修正）**：`use_gat=False` 少建 GAT 層 → 少消耗 RNG → 不只 head 初始化不同，**連 DataLoader 的打亂順序都不同**（實測第一個 batch 一個 1,558 支、一個 1,668 支）。架構相同的兩組才是乾淨配對。**下次設計消融時，要嘛所有組架構等價（用 mask 而非拿掉模組），要嘛明確承認這個限制並用多 seed 量 σ**
- **正確性修正與效益改動要分兩套尺，不能混在同一張判讀表（2026-07-30，F5 定案）**：`fundamentals_v2` 與 purge 都讓 IC 下降，但它們拿掉的是 look-ahead——IC 掉是**誠實化的代價**，不是「這個改動不好」。使用者要求把 `fundamentals_v2` 的負 Δ 拆到子修正後證實由 Q4 延遲（t=−4.06）主導，因此與 purge 同列「一律採用、不套 |Δ|≥0.009 門檻」。**判讀規則必須在跑之前定死**，否則看到數字才選規則，等於用結果反推標準
- **旗標這類「不為 IC 而設」的特徵，裁判要選對模型；但假設被推翻就要認（2026-07-30）**：可得性旗標要靠交互作用才有用，線性模型結構上用不到 → Ridge 的 +0.0000 是弱證據。我據此主張「GBDT 才是對的裁判」並跑了三組，**結果 GBDT 也否決**（5d 無效應、20d −0.0060 t=−2.93、gain 佔比 0.32% vs 均分 0.326%）＝「有能力用但不用」。假設被自己的實驗推翻，就照結果走。另一個收穫：**起點後移到 2013 與旗標功能高度重疊**，旗標要標的期間大多已被砍掉——兩個改動的效果不獨立，這是量 delta 時該預期到的
- **不依 test 集 IC 挑特徵子集（2026-07-30）**：「只留 gain 最高的 `Avail_Daytrade`/`Avail_Institutional`」看起來划算，但那是 test-set selection，會讓後續所有數字失去 out-of-sample 意義。規格要照原則定，不照測試期表現定
- **成本不是零的改動，方向對但不顯著時不進規格（2026-07-30）**：中性化四個測量方向全正、去極端日後更強、組合層也最好，但 t=0.76–1.42。決定性理由不是統計而是**它在推論路徑多一個必須同步的步驟**——`macro_norm` 與 `fundamentals_v2` 已經是兩次前例。+0.002（5d 主 horizon）換這個風險不划算，留到 F6 用真正要上線的模型再量
- **量 delta 時「一次一變因」要驗證組合本身自洽（2026-07-30）**：原設計的 R3「fund_v2 關但旗標開」看似合理，但 `Avail_Financials` 是在 `_merge_fundamentals(fundamentals_v2=...)` 之後才依 `notna()` 算的，而舊路徑那三欄是**死常數不是 NaN** → 旗標在該組合下退化成死常數 1.0。**這種不自洽不會報錯**，只會讓 delta 混進第二個變因。實測比對兩份矩陣的旗標統計才抓到
- **背景任務回報 killed 不代表子孫都死了（2026-07-30 操作教訓）**：據此另啟接力腳本，導致兩個 `--build` 同時寫同一個輸出檔。啟接力前要先列 process 確認，不能相信通知。附帶驗證了共用 chunk 的唯讀設計有效——`baseline_cache_v2` 完全未受影響
- **會改變歷史特徵語意的 bug 一律用旗標、預設維持現況（2026-07-27）**：`EPS_Surprise` 季頻修正與 Q4 年報 `available_from` +90 天都是真 bug，但直接改會讓推論端特徵與 V6.1 checkpoint 的訓練語意不一致——這正是 D1 `macro_norm` 踩過的坑。故比照該慣例加 `fundamentals_v2: bool = False`，預設關、Colab 訓練端傳 True，並寫進 V6.2 部署 checklist。**驗收標準是「與 git HEAD 逐位元相同」**：本次實測 60 欄最大絕對差 `0.000e+00`。凡動 `feature_engineer.py` 都應跑這個回歸測試
- **資料衛生必須在 build_features 之前，不能在 clean_and_scale 之後（2026-07-27 教訓）**：舊流程的去重放在 `clean_and_scale` 之後，看起來「有去重」，但時序特徵（Return/MA/RSI/ATR/KD/OBV/Volatility）與當日橫斷面 z-score 早就吃過重複列了——2432 的 Return_5d 因此正負號相反。**清理的位置比清理本身重要**
- **健檢一律 non-fatal 且用 parquet statistics（2026-07-27）**：線上 V6.1 每天要出訊號給家人看，健檢不該成為新的失敗點，故全部 `try/except` 包住、只印警告。取最大日期改用 row-group statistics（不讀資料）、損壞列統計只看近 90 天窗口——`institutional_raw` 有 32.8M 列，整檔載入會吃掉數 GB，本機只有 23.7 GB 且推論同時在跑。實測健檢自身 RSS 僅 0.14 GB
- **`baseline_common` rolling 順序修正 → 方向二協定升 v2.0（2026-07-27）**：rolling 特徵原本建在 `clean_and_scale` 之後的橫斷面 z-score 上（檢查表 G4 的順序顛倒），已改建在 chunk 原始值上、再逐日 winsorize + z-score，與同檔 `Mom_*` 的既有正確作法一致。**lag 特徵（`*_lag1/5/20`）刻意不改**——純位移不是時序聚合、不會混到不同日的尺度，「該股 N 日前的橫斷面排名」本身合理。既有 Ridge/GBDT/GRU 三階結果是舊特徵下的數字，重跑後才可與新結果並列
- **階 3 拍板：GRU 擇一 + window 60（2026-07-14 使用者拍板）**：循環單元 LSTM/GRU 擇一控制多重測試，選 GRU（同級替代、參數較少、3060 較快）；window 60 偏離協定 §3 字面的 252——對齊 5d 對照對象 v6_short 的 window 60，Phase 1 deep supervision 已實證長窗對 5d 冗餘，偏離理由記錄於腳本 docstring 與結果報告。loss 維持純 MSE on rank（不跟進 v6_short 的 listnet_5d，避免引入新自由度、破壞同場對照）
- **Baseline IC 引用需分層、組合基準改等權宇宙、不為此重抓資料（2026-07-13 排查定案）**：0.1015 經 D0–D5 排查非 bug 非資料錯誤，但為「全市場含小型股」數字——對外引用一律附分層（高流動 0.0705 / 純籌碼基本面 0.0717）與存活者偏差未量化聲明；組合層以「等權 eligible 宇宙」為基準（TWII 僅脈絡）；「對 TWII −90%」已證實為基準錯配不再單獨引用。下市股回補維持先不動工（影響絕對水位不影響四階相對比較），等方向二跑完再評估
- **Baseline 對照協定四決定（2026-07-12 使用者拍板）**：單一切分為主（同 Phase 3 harness，避免 36 次 Mamba WF 重訓）+ 便宜階 WF 為輔／rank label（與 production 線一致）／5d 主 horizon（使用者短線操作）／Top50 等權 5 日再平衡、0.15%/0.45% 成本。協定凍結後不中途改 label/切分（改了全部 baseline 重跑）。引用紀律：Mamba 端用同 harness 重跑值（0.0870）比較、不用歷史峰值（Phase 3-A 教訓）
- **Conviction 線 DL 輔助訊號用 20d SQ 連續值、不用低 U 硬門檻交集（2026-07-12，方向三-C 首輪實證）**：20d 校準大致成立且 SQ Top50 贏純 Alpha +1.3%/20d，但「Alpha 前 20% ∩ U 後 20%」交集反而輸給高 U 對照組——U 除權（連續）優於 U 門檻（硬切）。5d 校準 post-P0 暫時反向 → **deep ensembles/conformal 先不做**，等 7 月底樣本夠重跑再定；屆時對齊 horizon 用 v6_short 的 SQ_5d 檢驗
- **倖存者偏差（D3）不為 baseline 對照修復（2026-07-12）**：四階模型吃同一份有偏資料，相對比較仍公平；偏差抬高的是絕對數字（並可能輕微偏袒能利用「倖存者總會反彈」假型態的彈性模型，屬二階效應、已在協定與說明頁揭露）。完整修復需歷史成分股 + 已下市股的籌碼/基本面歷史，後者多半不可得，修了也不完整——成本效益不成立，維持揭露 + 流動性門檻緩解
- **非交易日假資料一律整日刪除（2026-07-12）**：06-07（週日）5,194 筆與 06-19（端午）同模式；非交易日不存在合法資料，直接刪整日、先備份。根治靠寫入端補交易日 gate（待辦）
- **進場標準統一為分數制、以 signal_conditions 為單一事實來源（2026-07-07）**：scanner 的「條件數 ≥2/4」與 sim 的「分數 ≥70」是兩套會分歧的標準（50 分 2 條件股 dashboard 顯示 BUY、sim 卻不買），統一成分數制後 sim 累積的績效才能直接回饋到使用者實際看的 dashboard 訊號。scanner 權重/型態加分一律 import signal_conditions、不再自帶副本（重複實作已實際造成 sim 型態分數重複計算 bug）。模型實驗暫停期間的主軸＝「不動模型，把模型輸出的使用方式調到最優」
- **TPEX 新端點常數放 fetcher.py、不放 config.py（2026-07-07）**：本機 config.py 是刻意保持 56 維、不 commit 的 dirty 檔，改它無法安全推上 remote → 資料源 URL 這類需要上 remote 的常數直接定義在 fetcher.py
- **交易所 API 欄位對映必須數值驗證（2026-07-07 教訓）**：TWSE T86 舊 parser 的投信/自營索引在 19 欄版面是錯位的（抓到外資自營商欄）；修 TPEX 24 欄時用「買-賣=淨、分項加總=合計」恆等式對活躍股驗證後才定案。另：長時間輪詢交易所 API 可能回異常資料（06-19 端午節竟回 1,075 筆），回補後要對照 prices_raw 交易日健檢
- **Phase 3-A：選 dropout=0.2、收 A 進 B、不加跑 weight_decay（2026-06-27）**：dropout sweep 同 harness 對照（基準 0.1 重跑 0.0870，非歷史 0.0951）下 0.2/0.3 各 +0.009 且 0.3 對 0.2 平手（0.0961 vs 0.0959）→ 取較不激進的 **0.2**。判定「有效」＝峰值升 + 過擬合延後（val_loss 谷底 ep2→ep4）兩項達標，雖屬邊際改善，但已過規則門檻、不必在短線端加跑 wd 拖長 A。**dropout=0.2 帶進趨勢/後續模型**。關鍵紀律：**baseline 要用同 harness 重跑值比，不用歷史值**（歷史 0.0951 因切分/seed 沒重現，直接拿來比會誤判 dropout「沒效甚至變差」）。0.096 仍單 epoch 尖峰，幅度真偽留 Phase 3-E 多 seed 確認
- **Phase 3 照 A→F 順序逐一做、一次一變因（2026-06-25）**：使用者拍板照表順序（A 正則→B listnet→C 趨勢單尺度→D 窗口→E 集成→F 特徵分離），而非挑單一最高把握者先做——因「這些都要實驗才知道」、且想累積成可寫進履歷的 case study，故每個實驗都把「結論＋為什麼」整理清楚。診斷一律隔離在 `V6/experimental/`、不改 production、不覆蓋線上 checkpoint；Claude 備程式＋給可貼 Colab 指令，使用者自己跑訓練
- **實驗檔不改 `short_model.py`、改用 monkeypatch（2026-06-25）**：`short_model.py` 被 `run_dual_inference.py` import 做線上推論，故 dropout sweep 不直接加參數到 `train_short_model`，而是在實驗檔內暫時把 `ShortModelV6` 包成固定 dropout 版（functools.partial）、跑完還原，線上零影響。checkpoint 用獨立檔名避免覆蓋 `v6_short.pt`
- **雙模型輸出=rank-score、SQ=Score/Unc；趨勢分數 +0.05 水位偏移（2026-06-19）**：模型用 rank 目標訓練→輸出非報酬而是 rank-score，SQ（Score/Uncertainty）拿來排序選股。趨勢 Score_20d 平均 +0.05（短線 ~0）來自 macro 2 年近似的水位平移、**對排名無影響**（排序對常數免疫）。Phase 3 可選 polish：(a) 分數 per-cross-section 置中讓 SQ 好解讀、(b) macro ts 改完整歷史去掉偏移、(c) 前端標明 rank-score 語意
- **dual inference 個股大表 trim 近 2 年（2026-06-19）**：root cause＝沒 trim 把整份 raw（prices 8.7M + inst 32M）丟 build_features → OOM 被砍（`python|tail` 隱藏了 `Killed`）。照 V6.1 trim 近 730 天解掉。眉角：macro ts 嚴格要完整歷史，但 macro 為橫斷面常數、對排名二階，先用 2 年近似（已實證只造成水位平移、不動排名）
- **目標改純 cross-sectional rank、目標工程+listnet 折進 Phase 2（2026-06-15）**：安全網是 V6.1 不是雙模型，故雙模型首發即用優化版、不做「raw 先上再重訓」的浪費（省掉雙模型 ×2 的重訓）。目標選 A（每日橫斷面 rank、IC 對齊最強），不選 vol-norm；下游 rank-score 語意之後在前端加說明。`Alpha_5d/20d/60d` 改 rank、順便加 `Alpha_10d`。特徵分離等大工程留 Phase 3。短線訓練 run 兼當 rank vs raw 證據（對照舊 multi-scale raw 0.049），明顯勝直接當 production、含糊才補 raw 控制組
- **短線模型放 `V6/experimental/`、不掛 DS 輔助頭（2026-06-15）**：單尺度短線模型類別放 experimental/（延續隔離、不碰受保護 models/），`run_dual_inference` 從那 import，穩了再考慮扶正；短線模型先不保留 deep supervision 輔助頭（B 顯示主 IC 只小升、效果有限），DS 列 Phase 3 可選。listnet_5d 保留（5d IC 0.049）
- **雙模型採「並行、不動資料流」上線、V6.1 留當安全網（2026-06-15）**：雙模型走獨立 inference + 獨立輸出檔，現有 V6.1→df_kelly→dashboard 一律不動、不合併（家人在看），雙模型失敗不影響線上。先求上線（IC ~0.047 就先上）、上線後才在雙安全網下做抬 IC 的資料/架構嘗試（Phase 3）。技術點：59 維特徵需並行路徑自建（RS 插 group A、與 56 維排列不同）
- **短線模型改用單一時間分支（2026-06-15，方案B 實證）**：deep supervision 顯示 20/60/252 三窗對 5d 完全冗餘（三分支 aux IC 同步爬到 ~0.05、融合主 IC 沒贏單分支），多尺度對 5d 沒加值 → 短線模型砍成單分支（省參數/算力）。連帶：④ 窗口階梯降級（連極端窗距都冗餘）、③ 3d 前提削弱列可選。DS 主 IC 略勝 baseline，可選擇在 production 短線模型保留輔助頭當輕度正則
- **加碼窗口階梯實驗 {60,126,252}（2026-06-15）**：Short=20 大機率沒功能，改用「丟 20、補 60→252 空洞」的幾何階梯（間距 2.1×/2.0×，比原本 3×/4.2× 平均），給三分支更有機會各自有用。用 DS 版跑以直接拿每分支 aux IC、與 {20,60,252} 對照；layer 維持 [2,3,3]（一次一個變因）；≤252 純切片免重建資料。即使仍塌成單分支也是「多尺度對 5d 沒加值」的證據（穩賺）
- **診斷實驗一律在 `V6/experimental/` 副本做、不動 production `marketmamba/`（2026-06-14）**：避免改來改去汙染雙模型主線；副本另開等於也不需動受保護的 `models/`、不必每次授權。只有確定上線的東西才回寫正式 package
- **不急著取代 V6.1：實驗全做完 → 雙模型一起練好一起上線（2026-06-14）**：短線（5d/10d）+ 趨勢（20d/60d）兩模型一起在 Phase 2 做，下游合併輸出的 plumbing 只做一次，避免先趨勢後短線各接一遍
- **改用 5d 主導目標、且不強制 gate 均衡（2026-06-14）**：使用者操作偏短線，且 20d 趨勢目標本就讓 gate 合理偏 Long。與其用正則硬壓 gate（違反「讓模型自己篩選」初衷），不如把目標換 5d、用 scale_gate 當「多尺度是否真有用」的試紙——散開＝多尺度成立；仍偏 Long＝可光明正大簡化成單分支
- **本輪 listnet 關掉而非改 5d**：ListNet 在 trainer 寫死只算 20d，留著會把 gate/表徵拉回 20d/Long、汙染實驗；本輪設 0（代價：無排名損失、5d IC 略保守）。`listnet_5d` 列第二階段（要動受保護 trainer.py）
- **trainer.py 例外修改授權（2026-06-14）**：為讓 early-stop/checkpoint/headline IC 追 5d，經使用者明確同意例外改 `marketmamba/models/trainer.py:678`（`preds[:,1]`→`preds[:,0]`）。純驗證指標 horizon、不影響 checkpoint 相容性
- **padding mask 未能解 gate 偏 Long（2026-06-14 實證）**：第二次重訓 `USE_PADDING_MASK=True` 下 gate 仍 ep4 崩到 0.997，推翻「padding 零值被當訊號」為唯一主因；偏 Long 更可能來自 20d 趨勢目標本身偏好長回看
- **training_status.json 採 Drive 手動同步而非 Colab 直接 push GitHub**：使用者不想在 Colab 放 GitHub token；代價是頁面只在訓練完成、手動放入 V6/results 並 push 後更新
- **模型狀態頁面的假資料全面移除**：無資料時顯示空狀態提示，不再用合成數字；WF 面板移除，待 walk-forward 例行化後以真實 fold 結果加回
- **padding mask 只加在 Long branch**：Short 取最後 20 步、Mid 取最後 60 步，在 ≥202 天資料的前提下這兩個 branch 輸入全為真實資料，不需 mask；只有 Long 使用完整 252 步才有 padding 問題
- **scale_gate 改為 `print()` 而非 `logger.info()`**：Colab 預設 logging level = WARNING，`logger.info` 會靜默丟棄；`print(flush=True)` 永遠可見，與其他訓練 log 風格一致
- **claude.ai Project 知識庫改放 OVERVIEW.md 等靜態文件，CLAUDE.md 動態狀態區塊僅供 Claude Code 使用，不需同步到 Project**
- **PR 3 的 rs_20d / rsi 欄位留 null**：df_kelly.csv 目前不含 RS_20d（那是 feature matrix 的中間產物），前端顯示四層時這兩個欄位以 0 fallback，不影響 L1~L4 主要條件判斷；待 V6.2 模型若輸出 RS 相關信號時再補
- **portfolio_exit_check 的 inst_sell_streak 從 prices_raw 計算而非 action_signals**：prices_raw 含原始 Foreign_Buy/Foreign_Sell，精確度更高；action_signals 的 institutional_buy 只是 scanner 的 boolean flag
