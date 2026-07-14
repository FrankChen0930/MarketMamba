# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> 最後更新：2026-07-15（方向二全數完成：Step 4 GRU 5d IC +0.1113 四階最高 + Step 5 對照表定稿；方向一 Step 8/9 完成、`/pipeline` 頁待驗收部署）

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

> 最後更新：2026-07-15（方向二 Step 5 對照表 + 方向一 Step 8/9 完成，研究計畫方向一/二全部收線；Phase 3 模型實驗仍暫停中，等真倉驗證獲利再繼續）

### 最近完成
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
- **研究計畫三方向（`planing/研究計畫_主檔.md`）**：**方向一 ✅ 全部完成、方向二 ✅ 全部完成**（Step 2~5 + 交付；`/pipeline` 前端頁待驗收部署）；方向三-C 等 post-P0 樣本 ≥20 天（約 7 月底）重跑複驗；方向三-A/B（事件驅動/Meta-labeling）優先級最低、先做資料可行性評估再決定投入。平行可做：資料基礎升級計畫階段二（資料衛生三項 + baseline_common 序列輸出，見 `planing/資料基礎升級計畫_baseline_common扶正.md`）
- **scanner 1.4 新邏輯待 07-07 推論驗證**：確認 ① BUY 數量合理（分數制 + 機構條件復活，乾跑 07-06 資料為 15 BUY/24 WATCH）② 機構連買明細正常顯示 ③ `condition_analysis.json` 有產出並被 push。條件貢獻分析的機構統計從 07-07 歸檔起才有意義（歷史旗標全 False）
- **Phase 3 實驗暫停中（2026-07-06 使用者決定）**：等真倉（V6.1）先驗證有沒有賺錢再繼續做實驗。目前進度停在：~~A 正則救峰值 ✅（dropout=0.2 有效）~~ → B/C/D 三支檔案已寫好推 main、**尚未在 Colab 執行**（listnet 權重 sweep / 趨勢單尺度簡化 / 短線窗口 sweep，預期結果見 `docs/phase3-experiment-plan-2026-06-25.md`）→ E/F 未設計。恢復時：在 Colab 跑 B、貼結果回來判讀
- **雙模型真實市場效益追蹤已上線（2026-07-06）**，取代原本卡在「archive 累積」的路線圖第②③步——不用再等，改成自動化每天算、頁面自動顯示，樣本量隨時間自然增加，不需要手動介入。目前 5d 樣本僅 4 天，10d/20d/60d 尚無資料

### 下一步
- [ ] **`/pipeline` 頁驗收 + 部署**：本地 `cd app/frontend && npm run dev` 看 `localhost:5173/pipeline`，OK 後指定檔案 push（`docs/baseline-comparison-table-2026-07-15.md`、`docs/baseline-step4-rnn-2026-07-15.md`、`docs/breadth-pipeline-page-draft-2026-07-12.md`、`app/frontend/src/pages/Pipeline.jsx`、`App.jsx`、`AppLayout.jsx`、`planing/`、`CLAUDE.md`、`V6/experimental/baseline_rnn.py`、`result/baseline_rnn_result.json`；**不要 `git add -A`**）→ Vercel 自動部署
- [ ] **資料基礎升級計畫階段二**（方向一/二收線後的下一個工程主線，不需 Colab）：資料衛生三項（Close=0 gate、還原接縫、超限複驗）+ baseline_common 序列輸出 (N,252,59) + KG 邊介面 + 協定 v2.0 版本化
- [ ] **~7 月底重跑 `V6/experimental/conviction_c_analysis.py`**：post-P0 5d 屆時 ≥20 天、post-P0 20d 開始有樣本；同步對 `results/archive/df_short_*.csv` 的 SQ_5d/Unc_5d 做同款校準分析（horizon 對齊使用者操作週期，腳本小改即可）——5d 校準反向是否為真在此定奪，屆時再決定 deep ensembles/conformal 要不要做
- [ ] **重複寫入來源追查 + 每日更新寫入端補「非交易日不寫入」gate**：07-07 起每日 ~1,550 筆重複；06-07（週日）、06-19（端午）兩次假資料同模式，疑同源
- [ ] `Data/processed_v6/prices_raw_backup_20260712.parquet`（127MB）：推論穩定跑幾天後可刪（與既有兩個 institutional backup 一起）
- [ ] **PersonalOS 同步 K 線圖**（Vercel 版已驗收）：複製 `KLineChart.jsx` + `StockModal.jsx` + `TradingSignals.jsx` 過去（注意 import 路徑差異 `../api/market` → `../../api/mm`），PersonalOS `npm install klinecharts@^9.8.10` + `npx vite build` + 重啟 exe
- [ ] **macro_raw 停在 2026-04-24**：每日更新本來就不含 macro → regime 閘門（TWII vs MA60）恆為 N/A、保守模式從未啟用（scanner 已加 fallback+新鮮度檢查，macro 一更新就自動生效）。對 V6.1 推論無影響（Group D 被 cross z-score 歸零）、對雙模型排名免疫（橫斷面常數）。要啟用需把 macro 加進每日更新
- [ ] **prices_raw 每日 ~800 筆重複寫入**（5/26 起、值完全相同，疑價格更新雙資料源重疊）：存量已清（51,801 筆），來源待查
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
- [ ] **V6.2 部署 checklist**：config INPUT_DIM=59 + FEATURE_GROUPS 取消 RS 注釋；`run_daily_inference.py` 的 `clean_and_scale` 改 `macro_norm="ts"`（程式內有註解標記）；checkpoint 換新
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
