# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> **最後更新：2026-08-11**（`prices_raw` 欄名撞名讓當天三次執行全掛 → 已修資料 + 修根因，
> V6.1／V6.2 都補跑完成；教訓：`to_parquet` 少 `index=False` 只在索引不連續那天才會爆）
>
> **開工先讀本檔最下面「下一步」的 ▶ 區塊。**

---

## 📚 記憶分三層（2026-08-06 整理）

CLAUDE.md 原本累積到 264 KB、超過載入上限，已拆成三層。**找不到東西時照這個順序找**：

| 層 | 位置 | 放什麼 | 進 git？ |
|---|---|---|---|
| **1** | **本檔** | 規則、系統說明、**現在要做什麼**、每天會用到的操作紀律 | ✅ |
| **2** | `obsidian_note/` | **結論與教訓**、實驗史、決策全紀錄 | ❌ 本機（含個人內容） |
| **3** | `docs/`（25 份） | 完整數字、逐 epoch log、規格書 | ✅ |

### 第二層的入口

```
obsidian_note/
├── 🏠 Home.md                          ← 總導覽
├── 01 系統現況/現況整理.md              ← V6.2 上線、三套系統並跑
├── 02 問題追蹤/已知問題清單.md          ← 未解問題、決定不修的限制
├── 03 架構筆記/資料管線與修復史.md      ← 直連化、MOPS、除權息還原、雷區
├── 03 架構筆記/訓練紀錄.md              ← V6.1→V6.2、Phase 0-3
├── 06 研究紀錄/00 研究總覽.md           ← ★ 八模型定稿表 + 引用紀律
├── 06 研究紀錄/01 Baseline 四階對照.md
├── 06 研究紀錄/02 F5 特徵協定.md
├── 06 研究紀錄/03 F6 消融系列.md        ← GAT / Group D / 2×2 / 47 維
├── 06 研究紀錄/04 組合建構層.md         ← portfolio_lab / WF / 風控
├── 06 研究紀錄/05 標籤與再平衡.md
├── 07 決策與教訓/決策紀錄.md            ← ★ 選了什麼、為什麼、誰拍板
└── 07 決策與教訓/方法論教訓.md          ← ★ 可帶走的通則
```

> [!important] 引用任何研究數字之前，先看 `06 研究紀錄/00 研究總覽.md` 的「引用紀律」
> 那裡有雜訊底線（N=50 年化帶 ±2.68pp）、窗長不可跨比、IC 要分層引用等五條，
> 不照著做會把雜訊當成結論。

**維護規則**：本檔的「最近完成」只留**最近一個月**；更早的事情**移進第二層**、不要刪掉。

---

## 標記慣例（2026-08-05 使用者訂定）

**`<預測頭>/<再平衡天數>`**，例如 **`5d/20`** = 用 5 日頭的分數、每 20 個交易日再平衡一次。
討論組合層結果時一律用這個寫法，避免「五日」到底指預測期還是持有期的混淆。

### 標籤 × 再平衡的完整矩陣（N=50 / k=1.5，net 年化）

**Ridge / GBDT —— 越長越好，兩模型五個頻率上幾乎全部成立**

| | 1日 | 3日 | 5日 | 10日 | 20日 |
|---|---|---|---|---|---|
| ridge 5d | −19.8% | 0.2% | 12.9% | 16.1% | 20.6% |
| ridge 10d | 0.2% | 13.1% | 19.4% | 18.6% | **25.1%** |
| ridge 20d | 9.6% | 18.2% | 19.4% | 20.5% | **27.4%** |
| gbdt 5d | −30.2% | 0.7% | 6.0% | 17.3% | 11.2% |
| gbdt 10d | −10.4% | 4.4% | 6.9% | 14.1% | 11.8% |
| gbdt 20d | −4.1% | 10.4% | 14.4% | 22.7% | **21.8%** |

**Mamba（隔離 40 天那一輪＝`head20d_ablation`，唯一三種標籤齊全的）—— 倒 U 形，10d 最好**

| | 1日 | 3日 | 5日 | 10日 | 20日 |
|---|---|---|---|---|---|
| 5d | 29.3% | 36.1% | 45.6% | 38.0% | 36.4% |
| *5d（同設定另一次抽樣）* | *26.7%* | *35.4%* | *46.8%* | *40.4%* | ***30.2%*** |
| **10d** | **46.3%** | **46.1%** | **49.0%** | **46.2%** | **45.9%** |
| 20d | 35.2% | 36.0% | 42.1% | 39.9% | 39.2% |

⚠️ **看那兩列 5d——同樣的東西、兩次抽樣，20 日欄差了 6.2pp**。所以「Mamba 上 10d > 20d」本身也在雜訊邊緣。

**Mamba（隔離 30 天那一輪＝F6 2×2 最佳格）**：5d ✅ / 10d ✅ / **20d ❌ 沒有**
——那是**刻意的**，20d 標籤需要更長 purge，用 30 天會系統性偏袒它。
**兩輪之間隔離天數不同（30 vs 40），不可跨輪並列。**

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

## 每日排程（兩條鏈並跑）

**⚠️ 2026-08-05 起 19:30 → 21:30**：實測 19:35 時 TWSE 的 `margin`/`daytrade` 還「尚未公布」，
21:11 才有當日資料。跑太早會靜默 ffill 昨天的值。

```
平日 21:30  PersonalOS_Daily（Windows Task Scheduler）
  └─ WSL2 → run_daily_inference.py（V6.1, 56 維）
        [1/7] 資料更新（yfinance + 交易所直連 + MOPS）
        [2/7] 特徵矩陣建構 + 資料新鮮度檢查
        [3/7] Mamba+GAT 推論 → df_kelly.csv, df_traj.csv
        [4/7] LLM 市場報告 → market_summary.json
        [5/7] 歸檔（90 天滾動）
        [6/7] 訊號掃描 → action_signals.json
        [7/7] git push → GitHub → Render 快取更新
     └─ run_dual_inference.py（雙模型, 59 維）→ df_short.csv / df_trend.csv

平日 22:15  MarketMamba_V62（2026-08-10 上線）  ── 實測全程約 9 分鐘
  └─ run_hidden.vbs → v62_daily.bat → WSL2 → run_v62_daily.py
        [1/6] 自己抓資料（fetch_data，不再依賴 V6.1）
        [2/6] 當日資料檢查（容許 0 天，缺就 Telegram 告警）
        [3/6] 特徵矩陣建一次（59 維）→ 8 份 Mamba 分數（**先去重再前向**）
              └ 另起 process（MM_PROTOCOL=v2, 66 維）→ run_v62_baselines.py
                → 3 份 baseline 分數（ridge / gbdt / gru）
        [4/6] 組合層狀態機 × **19 個組合**（分數 × 再平衡率，純 CPU）
        [5/6] 前瞻績效彙總（**又一個獨立 process**，MM_PROTOCOL=v2）
              → v62_performance.json
        [6/6] git push → POST /api/v62/cache/refresh → /breadth/portfolio
```

**兩張表的關係**：`run_v62_inference.ARMS`（模型 × 預測頭 → 分數檔）與
`v62_portfolio.PORTFOLIOS`（分數 × n/k/freq → 持股）**是分開的**。
**再平衡率是組合層參數、不是模型參數** → 19 個組合只需要 11 份分數。
`v62_portfolio.py --list` 看全表；`PORTFOLIOS` 是唯一真相，發布成 `v62_arms.json` 給後端讀。

推論進度透過 tkinter 視窗即時顯示（WSLg）。成功 3 秒自動關閉；失敗保持開啟並置頂。
中文字型未裝時**自動改用英文標籤**並印出安裝指令（不會出現豆腐方塊）。

---

## 訊號系統（V6.2）

> [!warning] 這整節將隨 V6.1 一起退役（2026-08-01 使用者決定）
> 使用者說明這是「上課前自己想出來的粗糙策略」，判斷用組合建構層會更好
> → **不做舊系統的換手稽核，直接讓它退役**。
> 連帶要退的下游：`action_signals.json`、`condition_analysis.json`、
> `portfolio_exit_check.json`、`sim_engine_v3`、dashboard 的 TradingSignals 頁。
> 接班規格＝`v2_kg_nomacro` 分數 + N=50 / k=1.5 / 20 日再平衡。
> **在 V6.2 連續跑順幾天之前不要拆。** 以下保留供維護期參考。

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

## Colab CLI（2026-08-04 裝好並測通）

不必再手動操作 Colab 網頁——`colab run` 可以「租一台 GPU → 跑本機腳本 → 自動釋放」。

| 項目 | 值 |
|------|-----|
| 套件 | `google-colab-cli` 0.6.0（PyPI，需 Python ≥3.12、**只支援 Linux/macOS**） |
| 安裝位置 | WSL2 的獨立 conda 環境 `colabcli`（**刻意不裝進 `mamba_env`**） |
| 執行路徑 | `~/miniconda3/envs/colabcli/bin/colab`（直接呼叫，不走 `conda run` 以免輸出緩衝） |
| 認證 | oauth2 貼授權碼流程，token 在 `~/.config/colab-cli/token.json`（**不需要 gcloud**） |
| 實測可用加速器 | **CPU ✅ / T4 ✅ / A100 ✅**（A100 = 40GB、12 CPU、83.5 GiB RAM、torch 2.11.0+cu128） |

```bash
# 一次性作業（new + exec + 自動 stop），推薦
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab run --gpu A100 -s <name> ~/colab_jobs/<script>.py"
# 查有沒有殘留 session（會計費）
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab sessions"
```

**⚠️ 必須記住的五件事**
1. **依賴要鎖版本**：`google-colab-cli` 對 `jupyter-kernel-client` **沒有鎖版本**，而後者 1.0.0（2026-07-26）把 `KernelClient` 改名 → 直接 `AttributeError`。已鎖 **`jupyter-kernel-client<1.0.0`（實際 0.15.0）**。**重裝或 `pip install -U` 會再壞一次**。
2. **沒 stop 的 session 會一直燒 compute units**（只有 24h 上限兜底）→ 一律用 `colab run`，不要用 `colab new`。
3. **`repl` / `console` / `auth` / `drivemount` 需要 TTY，Claude Code 不能代跑**。
4. **`colab run` 與 Drive 互斥**：`run` 是即開即棄的 session，而 `drivemount` 必須人在終端機。
   - **繞開 Drive 是可行的**：需要 TTY 的只有 `repl`/`console`/`auth`/`drivemount` 四個，`upload`/`download`/`install` 都不需要 → `new` → `upload` → `exec` → `download` → `stop` **可以全自動**。
   - **但成本不划算**：`processed_v6.zip` 約 3 GB，`new` 到 `stop` 之間**整段計費**（含上傳等待）。20 Mbps 上傳＝在 A100 上乾等 20 分鐘，比手動掛一次 Drive 貴得多。mamba 的 whl 不是問題（幾十 MB，可在 VM 上直接從 PyPI/GitHub 抓）。
   - **決策（2026-08-04 使用者拍板）**：**維持既有的 Drive + 手動掛載模式**，不做上傳速率實測。CLI 已測通這件事本身是資產，**下個大階段（V6.2 上線收尾之後）再優化，不必重來**。
5. **CLI 的 DEBUG log 會把完整 OAuth token（含長期有效的 `refresh_token`）明文寫進 `~/.config/colab-cli/colab.log`**，權限還是 `-rw-r--r--`。定期清、或回報上游。

指令全貌用 `colab -h`；給 agent 看的操作手冊用 `colab skill`（品質很高，含各指令的坑）。

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

> 最後更新：2026-08-09。**本區塊只留最近一個月**，更早的完整紀錄在第二層（`obsidian_note/`）。

### 最近完成（2026-07-06 ~ 08-11）

#### 2026-08-10/11 — ★★ `prices_raw` 欄名撞名，當天三次執行全掛（已修資料 + 修根因）

**症狀**：08-10 的三次執行**全部在第一行就死**，執行 0 秒：

| 任務 | 時間 | 結果 |
|---|---|---|
| `PersonalOS_Daily` | 21:30 | `0xC000013A`（使用者看它卡住手動關掉） |
| `MarketMamba_V62` | 22:15 | exit 255 |
| 手動 V6.1 | 22:30 | `ArrowInvalid` |

```
pyarrow.lib.ArrowInvalid: Multiple matches for FieldRef.Name(__index_level_0__)
```
`prices_raw.parquet` 的 schema 有**兩個同名欄** `__index_level_0__`
（一個 `double`、一個 `int64`）→ `fetcher.py:3857` 的
`pd.read_parquet(price_path, columns=["Date"])` 無法解析 FieldRef。

**★ 怎麼定位的（依「欄位位置」讀值，不是猜）**
欄名重複時 pandas 讀不了，但 `pq.ParquetFile(p).read()` 可以（**不能用 `pq.read_table()`
——它走 dataset API，欄名重複直接 `Can't unify schema with duplicate field names`**）。
依位置抓出那個 double 欄的值 = **`0 … 8,737,250`、單調遞增、非 NaN 共 8,737,251 個**，
而且**只有 2026-08-10 的 1,959 列是 NaN**。
8,737,251 正好是 08-08 回補後的列數 → **它是回補那次寫入留下的 RangeIndex，
今天之前就躺在檔案裡了**；今天的 append 只是把第二個疊上去。

**★★ 根因：`_append_to_parquet` 的 `df.to_parquet(path)` 沒有 `index=False`**
關鍵在於**它平常不會出事**：索引**連續**時 pandas 只把 RangeIndex 記進 metadata、
**不落成實體欄**；只有 `drop_duplicates` 真的丟掉列、索引變得**不連續**時才會實體化。
檔案裡若已存在同名 stray 欄（讀回來被當成一般資料欄），這一寫就變成兩個同名欄。
→ 同一批被寫的 `institutional_raw` / `margin_raw` 至今乾淨，就是因為沒觸發去重。
**「平常看起來沒事」不代表沒有這個缺陷。**

已修：`to_parquet(path, index=False)` + **寫入前**偵測並丟棄任何 `__index_level_*` 欄並印警告。

**修復與驗收**（`V6/experimental/fix_prices_index_column.py`，預設只檢查、`--apply` 才寫）：
依位置刪掉第 7 欄，其餘原封不動。守門放在**寫入前**（欄名、型別逐欄、列數），
先寫暫存檔驗過才 `os.replace`。驗收：`Open/High/Low/Close/Volume` 全部
**`max|Δ| = 0.000e+00`**、`(Date, stock_id)` 鍵集合 **8,739,210 完全相同**、
`pd.read_parquet(columns=['Date'])` 恢復正常。

**⚠️ 沒查清楚的一點（不偷改成「已解決」）**：committed 版的 `backfill_prices.py:220`
寫的是 `to_parquet(PROD, index=False)`，照那份程式碼**不會**產出 stray 欄，
git 歷史上這行也只有一個版本。中間檔已被覆蓋 → **無法重現**。
實際跑的很可能是 commit 之前的工作區版本，但這點**沒有證據，不下定論**。

**補跑結果**：V6.1 ✅ 24m20s（df_kelly 1,925 檔、型態 830 個、push + Render `200`）；
V6.2 ✅ 18.3 分（11 份分數 × 19 個組合、push + V6.2 快取 `200`，用 `--no-fetch`
沿用 V6.1 剛抓好的 raw parquet）。

**⚠️ 兩個我自己踩到的流程坑**
① **背景任務被砍**（CLAUDE.md 早就寫過，我第一次沒照做）：第一次補跑在
   `[1/10]` 已經通過、跑到 yfinance 時被工具層回收。改用
   **`setsid nohup … &` + `disown` detach** 才活得下來。重啟前先 `ps` 確認無殘留
   （避免兩個程序同時寫同一批 parquet）。
② **監看掛錯檔案**：detached 那次的輸出走 stdout 導向的 `.out`，
   **`inference.log` 一個字都沒寫** → 我掛在 `inference.log` 的 monitor 全程零事件，
   看起來像「沒在跑」。**要監看的是「這次執行實際會寫的那個檔」，不是平常那個。**

#### 2026-08-09（傍晚）— 用 8/6 資料整條跑一次並推上線，當場抓到自己的洞

使用者決定不等週一，直接跑。**exit 0、8.4 分鐘、11 份分數 × 19 個組合、已推送、
Render 快取 200**。10/10 個每日源完整，19 個 arm 各建倉 50 檔。

**★★ 第一天的輸出立刻暴露我前一小時才寫的誤差棒有個洞**

```
v2_kg_nomacro_f20   1   -0.1%   -31.5%   ...   37.3%   -68.8pp   primary
```
**19 個 arm 全被印成「比回測差 45~78pp」，且不帶任何標記。**
根因：n=1 → `sd=0` → `ann_stderr_pp = null`，而下游寫成 `se = m.ann_stderr_pp ?? 0`
和 `"*" if se and abs(diff) < se` —— **null→0 之後每一列都通過**。
效果與設計意圖完全相反。

→ **算不出誤差 ≠ 沒有誤差。`None` 必須當成「不確定性無限大」。**
修成終端印 `±?` 並一律標 `*`、前端 `solid` 要求 `!= null`。

那個 −0.15% **其實只是建倉手續費**（100% 換手 × 0.1425%），19 個 arm 完全相同
——第一天本來就只會有這個。

**★ 這件事的意義**：這個洞在合成測試裡看不到（我的合成紀錄是 30 天、sd>0），
**只有真的跑第一天才會出現**。提早跑不只是「早點看到畫面」，
它是唯一能測到 n=1 這種邊界的方法。

**★ 各 arm 第一天的 Top50 重疊（值得記著，判讀並行紀錄時會用到）**

| 對照 | 與主線重疊 |
|---|---|
| 5d 頭的其他四個頻率 | **50/50**（第一天必然相同，之後才分岔） |
| 10d 頭（同 checkpoint 第二欄） | 34/50 |
| `head10d` / `head20d`（獨立 ckpt） | 32 / 26 |
| **`v2_kg`（只差 Group D）** | **3/50** |
| `no_gat` | **0/50** |

⚠️ `v2_kg` 只差 Group D 卻只重疊 3 檔（隨機期望值 1.3）——因為它是
**獨立訓練的另一顆 checkpoint**，不是推論時把 Group D 遮掉。RNG 全然不同，
加上 Group D 是 F6 量到最大的單一效應。
→ **這些不是「同一個模型的變體」，判讀並行紀錄時不可當成受控對照。**

#### 2026-08-09（下午）— 前瞻績效接上線 + 19 個組合分組呈現

**背景**：`v62_performance.py`（08-08 寫的）**沒有任何人呼叫它**，也沒有後端端點
——它是唯一能回答「哪個組合真的比較好」的工具，卻只能手動跑。三件一起補完。

**① 產出鏈路**：`run_v62_daily.py` 新增 `run_performance()`，**獨立 process**
（與 baseline 同一個理由：`portfolio_lab` 在 import 期就要 `MM_PROTOCOL=v2`，
Mamba 線是 59 維，混跑會靜默算錯）。位置在 `step()` 之後、push 之前
——它讀今天剛寫進 jsonl 的那一列，產出要跟著同一個 commit 上去。
流程從 5 步變 **6 步**（進度視窗也加一格）。非致命：算不出來不影響持股紀錄。

**② `family` 欄位（與 `tier` 正交）**：19 個 arm 平鋪在同一排按鈕看不出結構，
而且**會誤導**——`v2_kg_nomacro_f03`（主線換 3 日再平衡）與 `old_kg_f20`
（KG 壞掉的對照組）**tier 都是 `inferior`**，平鋪長得一模一樣。
→ `family` 說「這是什麼」、`tier` 說「能不能照做」，兩個維度都要看得到。
五組：`main_5d`(5) / `main_10d`(5) / `ckpt`(2) / `ablation`(4) / `baseline`(3)。
加了 import 期守門（family 打錯字會讓整組在前端消失且不報錯）。

**③ ★★ 樣本不足的處理 —— 我原本的門檻擋不住問題**

原設計是「n_days < 20 就不顯示年化」。用合成紀錄端對端測時發現
**30 天算出「年化 −52.3%」，而 20 天的門檻會放行它**
——那個數字會被擺在 37.3% 的回測值旁邊，讀起來像「模型崩了」。

改成兩層：
- 門檻 20 → **60**（一季）。依據是算出來的，不是拍的：年化的標準誤
  ≈ 252 × s_daily / √n，n=20 → ±68pp、n=60 → ±39pp、n=252 → ±19pp
  （對照組合層雜訊底線 **±6pp**）→ **第一年之內年化都排不出名次**
- 超過門檻也**一律附 `ann_stderr_pp`**。只做門檻的話，第 61 天會突然
  冒出一個看似精確的數字。實測那筆 30 天的紀錄是 **−52.3% ±126pp**
- 「差」要**同時跨過雜訊底線與自己的標準誤**才上色，沒跨過標 `*` 保持灰

**驗證**：合成 30 天 jsonl → 跑工具 → **獨立重算累積報酬 −8.4258% vs 工具 −8.4%**，
跑完自動刪除（用不在 `PORTFOLIOS` 裡的 `__smoketest` arm 名，不污染真實紀錄）。
router 煙霧測試 14 項全過；前端 build 乾淨。

**⚠️ 又抓到三個寫死的比較基準**（這是同型的第 4~6 次）：
前端文案「回測從 +38.0% 掉到 −19.9%」（**兩個數字都過時，而且 −19.9% 連正負號
都是錯的**——實際是 37.3% → 25.0%）／router fallback 的 `backtest_ann: 0.380`／
`--list` footer 說 bt_ann 來自 08-03 的 docs。全部改成從單一真相取值或如實描述。

#### 2026-08-09 — 全面重評分：**資料修正沒有推翻任何結論**

回補 + 重建之後，11 個 arm（8 Mamba + 3 baseline）全部在新面板上重評分，
`portfolio_lab` 重跑，19 個組合的 `bt_ann` 全部更新成新面板值。
**口徑：不重訓、只重評分**（沿用 2026-08-05 前例；「有些重訓、有些沒有」會讓跨模型比較失去意義）。

**★ 訊號層：11/11 全部低於 ±0.009 實務門檻**

| arm | 舊 IC | 新 IC | Δ |
|---|---|---|---|
| `v2_kg_nomacro` | +0.1141 | +0.1125 | −0.0017 |
| `head20d`（最大變化） | +0.1385 | +0.1320 | −0.0066 |
| 其餘 9 個 | — | — | −0.0003 ~ −0.0029 |

⚠️ **但 11 個全部是負的**——11/11 同號不會是隨機。合理解釋是
**補回來的以冷門股為主，橫斷面變難排序**：這是誠實化的代價，不是模型退步。

**★ 組合層：11/11 全部在 ±6pp 雜訊底線內**（最大 `no_gat` +4.1pp）。
主線 **38.0% → 37.3%**。狀態機 replay 仍與 `portfolio_lab` 一致（37.28% vs 37.28%）。

→ **缺 15% 的列沒有改變任何比較結論。** 這件事本身有價值：它說明八模型表的
排序對這種資料缺陷不敏感。

**新增 `V6/experimental/compare_panel_ic.py`**：同一份標籤下的新舊 IC 對照，
**own（各自宇宙＝總效應）與 common（共同鍵＝純訊號效應）兩種口徑並列**——
只報一個都會誤導（宇宙變大本身就是修正的一部分）。

**⚠️ 移除一個會誤導的輸出**：`score_window()` 原本印
`對照（舊資料基礎）：mean IC +0.1145`——那是 **`v2_kg_nomacro` 一個 arm 的舊值，
卻對每個 arm 都印同一行**。重評分八個 arm 時 log 會長成
「head10d 從 0.1145 掉到 0.1094」，但 head10d 的舊值根本不是 0.1145。
**看起來像對照，其實是拿別人的基準。** 已移除，改由上面那支腳本做真對照。

**⚠️ 我的 tier 規則本身是錯的（已修）**：第一版只看 `|Δ| > 6pp` 就標 `inferior`
→ `head10d_f20` 比主線**好 8.9pp**，卻被標成「已知明確劣於主線，請勿照做」。
**方向搞反了。** 更根本的是：`head10d`/`head20d` 出自**隔離 40 天**那一輪、
主線是 30 天，**那個比較本身就不成立**。已新增 `incomparable` 分級
（前端用中性灰，**不可與 inferior 同色**，否則會把「不能比」誤導成「比較爛」）。

**⚠️ 寫死比較基準的坑，一次出現在三個地方**：前端寫死「主線是 38.0%」、
router 測試寫死 `0.380`、`score_window` 寫死 `+0.1145`。
重跑後全部過時。前兩個已改成從 manifest / API 取值。

#### 2026-08-08（晚 3）— ★★ `prices_raw` 有真實資料遺失（15%），根因已修 + 已回補

**怎麼發現的**：清備份時做了「刪之前先驗證正式檔不比備份少」的檢查——**那個檢查本身抓到的**。

**證實不是清理、是遺失**：用 TWSE 官方資料判定，4169 於 2026-06-01 確實有交易
（收盤 158.50 / 量 162,506），與備份吻合，**正式檔就是少那一列**。
備份多出的列全部有真實成交量（Volume=0 佔 0%、中位數 30 萬）、OHLC 合理、日期連續。

**規模**（582 天評估窗）：缺 **194,909 列 = 15.15%**、859 支受影響
（**489 支在可交易宇宙內**）、190 支整段消失、缺漏散布在 **98% 的交易日**。
受影響股票偏冷門（日成交中位數 107 萬 vs 未受影響 1,972 萬），
**但有 57 支成交金額高於未受影響的中位數**，其中 5 支日成交 3,000~5,200 萬卻整窗缺席。

**★★ 根因：`_append_to_parquet` 的寫入語意是「整天替換」**
```python
df_old = df_old[df_old["Date"] != date_str]   # 先刪掉該日全部舊列
df = pd.concat([df_old, df_new])              # 再寫入這次抓到的
```
→ **只要某天抓到的股票比已存的少**（端點掛掉、限流、universe 縮水），
差額就被**靜默刪除、不留痕跡**。已改成 **merge 語意**：
只覆蓋新資料也有的 stock_id，舊有的一律保留；要刪必須明確傳 `allow_shrink=True`。
理由：**抓取失敗是常態，而「今天沒抓到」與「今天沒交易」在寫入端分不出來；
分不出來的時候，保留是可回復的，刪掉是不可回復的。**

**回補**（`V6/experimental/backfill_prices.py`）：+441,543 列 → 8,737,251。
**可交易宇宙的受影響股票 489 支 → 6 支。**
驗收：舊鍵全保留 ✅／無重複 ✅／**既有列的值 max|Δ|=0.00e+00** ✅／schema 型別相同 ✅。

**★ 中途兩個被推翻的假設（都是驗證擋下來的）**
① 原以為 `正式 = 備份 × Π(adj_factor)`。**791 萬個重疊鍵上只有 11.4% 對得上**
   （5364 於 2011-11-18：正式 23.8939、備份 23.90 幾乎相同，但因子表說該差 30 倍）
   → **正式檔的還原口徑無法用 `ex_rights_raw` 重現**（但它確實有還原：
   除權息當日跌 >3% 的比例 9.5%，未還原的備份是 13.5%）。
② 改用**經驗比值法**：逐股在重疊區學 `正式/備份`，該比值是分段常數
   （看似數百個相異值其實是備份只有 2 位小數的浮點雜訊）。
   **前後兩側一致才採用**（一致 ⇒ 中間沒發生事件），學不到就不補。
   → **完全不需要知道正式檔用什麼口徑**，只是把它自己的轉換照樣套上去。
   留出法驗證：99.9898% 落在 0.5% 內、median 相對誤差 6.76e-05。

**⚠️ 我犯的順序錯誤**：schema 檢查寫在**寫入之後**，所以型別已經
`large_string → string` 才被抓到（已改回）。**守門要放在寫入前。**

**⚠️ 尚未做：base matrix 與衍生快取都建在舊資料上**
→ 模型現在仍吃缺 15% 的舊面板。**08-17 正式起跑前必須重建**：
`MM_PROTOCOL=v2 python V6/experimental/baseline_common.py --build --force`

#### 2026-08-08（晚 2）— F6 四個消融 arm 併入並行 → **19 個組合 / 11 份分數**

加入 `v2_kg` / `v3_kg` / `old_kg` / `no_gat`（checkpoint 從 Drive 複製到 `V6/checkpoints/`）。
四個驗證全過（2026-03-02，判準 ρ≥0.95 / Top50≥40）：
ρ = 0.9905 / 0.9907 / 0.9916 / 0.9844，加上 `v2_kg_nomacro` 回歸對照 0.9933。

**★ 但這不是「加四列 ARMS」而已——它們 `zero_macro=False`，吃 Group D。**
`run_v62_inference.build_feature_df()` 原本的註解寫「本模型 Group D 一律歸零，
所以 trim 不構成落差」——**那對 `v2_kg_nomacro` 系成立，對這四個不成立**。
已把 macro 全歷史貼回也套到 Mamba 這條線（新增共用模組 `V6/macro_ts_full.py`，
兩條線都用，因為它們在不同 process、不同 `MM_PROTOCOL`）。
實測幅度：`TWII_Return` 窗內自算 **−0.1264 → 全歷史 −0.8985**。

**★★ chunk 只到它建立那天，今天的日期一定要由呼叫端補**：
`base_chunk_*.parquet` 停在 2026-07-29，而每日推論跑的是今天 →
少了 `recent_raw` 參數，今天的 macro `map` 回來會是 **NaN**（比偏掉更糟）。
歷史段取 chunk、近期段取當前窗的未標準化 macro，接起來再算 expanding。
實測 log：`來源 3,815 個交易日：2011-01-03 → 2026-08-06`（chunk 只給到 07-29）。

**⚠️ `no_gat` 的 state_dict 少了 `graph_layer`/`gate`/`norm_fuse`**
（檔案 5.3 MB vs 6.4 MB、參數 1,394,301 vs 1,659,005）→ `Arm` 加 `use_gat` 欄位，
`infer()` 不可再寫死 `use_gat=True`，否則 strict load 當場失敗。

**⚠️ 一個讓我誤判的坑：關鍵步驟不可只走 logger。**
`macro_ts_full.splice()` 第一版只用 `logger.info` → 在 `run_v62_inference` 的驗證 log 裡
**一行都沒出現**（root logger 早被別的 import 設定過，`basicConfig` 變成 no-op），
害我一度以為貼回沒執行。**已改成一律 `print` + logger 附加。**
規則 7 的重點就是這個：**看不見等同沒做，而可見性不該交給 logging 設定。**

**⚠️ `v62_performance.py` 沒設 `MM_PROTOCOL=v2` 會靜默退回 v1**
（載到 2026-07-12 建的舊 baseline_cache，除權息修復之前），只印一行警告不會失敗
→ 已在檔案開頭直接設定，不依賴呼叫端記得帶。

端對端：**9.3 分鐘、11 份分數 × 19 個組合**、exit 0，測試殘留已清。

#### 2026-08-08（晚）— 前瞻績效工具，開跑第一天就抓到線上與回測的分歧

新增 `V6/v62_performance.py`：把 `v62_portfolio_{arm}.jsonl` 的逐日持股算成實際報酬。
**在這之前沒有任何程式做這件事**——15 個組合天天累積紀錄，但「哪個比較好」沒工具回答。
成本／報酬順序／換手定義全部 import 自 `portfolio_lab`，實戰數字才與回測同義。

**★★ 工具寫完立刻抓到一個真的 bug：`step()` 的並列打破與回測不同，而且不具決定性。**

把狀態機在 582 天窗上跑一遍 → jsonl → 算報酬，得到 **38.20% vs 回測 38.02%（差 0.18pp）**。
逐日 diff 定位到 **2024-10-01 的一檔股票**（線上選 `6206`、回測選 `3035`）。
兩者 **score 完全相等**（0.06756592，float32），那天有 **183 組並列**：

| | 並列怎麼打破 |
|---|---|
| 回測 `replay()` | `rank(axis=1, method="first")` → 依 pivot 欄序（stock_id 字典序） |
| 線上 `step()` | `sort_values("score")` → pandas 預設 **quicksort，不穩定排序** |

→ 不只口徑不一致，**`step()` 本身不具決定性**（同一份資料、列序不同就可能選出不同持股）。
修法：`sort_values(["score","stock_id"], ascending=[False,True], kind="mergesort")`。
修完 **38.020% vs 38.020%（0.000pp）**，四項指標全過；
且**故意打亂輸入列序跑三次，持股完全相同**。

**★ 這件事的兩個教訓**：
① **0.18pp 太容易被當成捨入誤差放過** —— 而它其實是「線上與回測是不同東西」的訊號。
   前瞻績效工具的第二個價值就是當**線上/回測一致性的稽核器**，不只是記績效。
② **浮點分數會並列**（float32、183 組/天）→ 凡是「排序取前 N」的地方都要問：
   **並列怎麼打破？兩條路徑打破的方式一樣嗎？排序穩定嗎？**

**順帶對齊的另一個口徑**：`step()` 原本沒有回測的「當日無收盤價＝不可交易」過濾
（`replay()` 有 `rank_df.where(mkt.px.notna())`）→ 已補 `_tradable_on()`。
這次的分歧不是它造成的（兩檔當天都可交易），但它是真的口徑缺口。
⚠️ `prices_raw` 的 `Date` 是 **large_string**，parquet filter 要用字串、不能用 `pd.Timestamp`。

**jsonl 補記 `weights`**：現行規格等權、其實推得回來，記它是為了
**讓紀錄自我描述**——哪天 `WEIGHT_MODE` 不再是 "equal"，所有「假設等權」的重建
會靜默算錯，而舊紀錄裡沒有線索能發現。

#### 2026-08-08 — 多頻率並行 + B 類經典模型上線 + push 鏈路補完

**使用者的需求**：20 日主線中間 19 天不換股，dashboard 沒東西可看 → 同一份分數
同時餵給多個再平衡頻率，中間天數看「參考組合」，但要標清楚那不是最好的。
外加把 Ridge / GBDT / GRU 也放上去並行累積實戰紀錄。

**成果：7 份分數 × 15 個組合，端對端 8.0 分鐘跑通**（`--first-day` 實測，測完已清殘留）。

**★★ 最重要的發現：`v62_daily` 整條鏈原本沒有 git push。**
`run_v62_daily.py` 與 `v62_daily.bat` 都沒有任何 `git add/push`——log 會顯示 ✅ 完成，
但 state 只留本機，**Render 與 dashboard 永遠看不到**。典型的「不會報錯的失敗」。
已補 `push_to_github()`（3 次重試）+ `_refresh_backend_cache()`。
⚠️ 快取要打 **`/api/v62/cache/refresh`**，不是 V6.1 的 `/api/signals/...`——兩個 router
各有各的 cache dict，清錯的完全沒作用。

**★ 再平衡率是組合層參數，所以 12 個 Mamba 組合只需要 4 份分數**（先去重再前向，
否則同一顆 checkpoint 會白白前向 5 次）。GPU 前向實測 **1.5 秒**，組合層純 CPU。
狀態機 5 個頻率 × 2 顆頭 = **10 組全部重現 `portfolio_lab`**（最大差 0.003pp）。

**★ 分級不是憑感覺，是拿回測差距對 ±6pp 雜訊底線量出來的**（`tier` 進 state/jsonl/API）：
20 日=主線；10/5 日與主線差 1.6~3.5pp **在雜訊內 → 老實說法是「分不出優劣」**；
5d 頭的 3/1 日差 14~18pp → 明確較差。
**高頻端要用 10d 頭**：與 5d 頭的差距隨頻率放大（20 日 +1.2pp 在雜訊內 → 1 日 **+9.6pp**
超出底線），機制在成本欄（1 日那格成本 31.1% vs 42.6%）。

**★ 後端 router 改讀 manifest，不自帶 arm 表**。後端跑在 Render（rootDir=`app/backend`），
import 不到 `v62_portfolio.py` → 兩邊的表**連 assert 都對不起來**（不同 process、不同機器）。
唯一真相是 `PORTFOLIOS`，每天發布成 `v62_arms.json` 一起 push。

##### B 類經典模型（Ridge / GBDT / GRU）

新增 `V6/run_v62_baselines.py`（**獨立 process**——Mamba 線 59 維、baseline 線 66 維，
config patch 是 module 級全域，混跑會靜默算錯）。三個 arm 驗證全過：

| arm | ρ | Top50 重疊 | trust |
|---|---|---|---|
| ridge | 0.9923 | 43/50 | reproduced |
| gbdt | 0.9995 | 50/50 | **new_model**（比的是它自己） |
| gru | 0.9969 | 45/50 | reproduced |

**★★ 尾端窗推論的三個落差，用「兩個不同窗長互比」隔離出來**
（第一版拿尾端窗直接對快取，**同時改了窗長與資料版本**、無法歸因，已如實記在
`diag_window_panel.py` 檔頭不偷改）：
① **51/54 個非 macro 欄完全相同**（ρ=1.000000）→ 這條路本身成立
② `Dividend_Yield_Fwd`/`Securities_Balance`/`Avail_Securities` 會偏 → **非價格 raw 一律不 trim**
③ **12 個 macro 欄全部偏很大**（`Oil_Return` max|Δ|=**2.27**、`TNX` 1.07）
→ `clean_and_scale(macro_norm="ts")` 的 expanding 統計量算在**傳進去的日期範圍**上。
解法：全歷史算一次再按 Date 貼回，且**用 `feature_engineer.macro_ts_zscore`**
（為此把那五行抽出來，逐位元回歸通過），不複製第二份實作。
⚠️ **Mamba 線量不到 ③**——上線 arm 把 Group D 整個歸零。Ridge/GBDT/GRU 吃全部 66 維躲不掉。

**★★ GBDT 的可重現性要分兩層講，不可簡化成「不可重現」**：

| 層 | 重建 vs 參考 |
|---|---|
| 訊號層 | ❌ ρ=**0.9203**、Top50 重疊 **25/50** |
| 組合層 | ✅ **11.0% vs 11.2%**、Sharpe 0.653 vs 0.639、換手 79% vs 81% |

**不是設定丟失**——我自己兩次跑（只差 purge 30 vs 60 天）彼此也才 ρ=0.9434、重疊 28/50。
樹的切點是離散決策，訓練窗動 1% 就翻，151 輪 boosting 再放大。
**但換進來的那半個 Top50 與被換掉的一樣好** → 「持股名單對不上」≠「策略不同」。
→ **這正是「比較模型用 decile spread、不用 Top50 年化」那條紀律的機制**，這次直接看到它發生。
（decile Sh 1.714 vs 1.806；對照 Ridge 同樣擾動只動 ρ=0.9970 —— 線性模型對訓練窗微調不敏感。）

**2026-08-03 的 `*_p30` 參考檔無法從 repo 重現**：`baseline_ridge_lasso.py` 當時沒有
`--purge` 也不寫分數 parquet，`label_horizon_report.py` 檔頭明寫「不重跑任何模型」，
而 `gbdt_5d.txt` 是 **07-13 的舊版**（v1 資料、無 purge）。已補上
`--purge/--tag/--save-model/--dump-scores`（**預設維持現況、逐位元不變**）。

**⚠️ 我犯的兩個錯，都已修正並留下防呆**：
① **帶 tag 跑一次就蓋掉正式結果 JSON**（`--tag` 只套在模型與分數上、漏了 `RESULT_PATH`）
→ 靠 git 還原，根因已修。`baseline_rnn.py` 本來就做對了。
② **收尾 log 寫死「已推送」**，`--skip-push` 時是假話 → 已改成如實顯示。

**★ `--purge` 的 stride 陷阱**：訓練用 `day_stride=2` 載入，在「載入後的日期空間」取倒數
第 30 個 ＝ 真實日曆的倒數第 **60** 個。參數名說 30 卻做 60。
已新增 `baseline_common.purge_cutoff()` 一律走真實交易日曆，兩支腳本共用。

##### pandas / numpy 兩邊不同調 → **不改版本，修 2 行**

Windows `pandas 2.2.2 / numpy 1.26.4`、WSL `pandas 3.0.2 / numpy 2.4.3`。
**不降 WSL**（`mamba_ssm`/`causal_conv1d` 是編譯過的 wheel，降 numpy 會斷整條推論鏈）、
**不升 Windows**（九個模型的 `portfolio_lab` 結果都是在 1.26 上算的，升級要重驗整張表）。

實際壞掉的只有 **2 行**（`portfolio_lab.py:286`、`f5_r_series.py:248`），
而修正模式 codebase 裡本來就有——另外 5 處早就寫了 `.copy()`，只是漏了這兩個。

**跨平台數字實測一致**（582 天 replay）：freq=20 → 38.02% vs **38.02%**、
freq=1 → 19.91% vs **19.91%**、換手兩端都相同。
→ **「組合層一律 Windows」這條紀律解除。**
⚠️ 但這只證明 `portfolio_lab` 這條算術路徑等價，不是整個 codebase 等價
（`pd.qcut`、groupby 排序穩定性那類還是可能有差）——別的管線要搬先對一次已知數字。

**環境**：WSL 補裝 `lightgbm==4.6.0`（`--no-deps`，對齊 Windows；numpy/scipy 未動）。
**Windows 的 torch 已損壞**（`c10.dll` 初始化失敗）→ GRU 每日推論只能在 WSL 跑。

#### 2026-08-06 — 47 維 arm 判定：未達標，結案

**Δ = −0.0014、配對 NW t = −0.68 → 維持 59 維 + Group D 歸零，V6.2 上線規格不變。**

| | 47 維（真砍） | 控制組：59 維 + Group D 歸零 |
|---|---|---|
| mean IC（582 天） | 0.1131 | **0.1145** |
| ICIR | 1.386 | 1.340 |
| 參數量 | 1,659,237 | 1,659,005 |
| 峰值 epoch | 8 | 5 |

判準跑前寫死在腳本裡（門檻 +0.009）。配對檢定是我補做的：逐日相關 0.9410、Δ>0 佔 46.0%、
前後半與去極端 5 天都同向 → **不是「47 維比較差」，是「沒有差別」**。

**使用者問「會不會是 10 個 epoch 的限制」——查過了，不是。** 三條證據：
① **LR 已退火到 2.80e-10**（走平是排程跑完，不是被砍）
② 逐 epoch 的 Δ **沒有在收斂**（ep4~ep10 穩定在 −0.0015，最後一點還更寬）
③ 要補 +0.0104 但最後三個 epoch 只漲 +0.0007，且 `val_loss` 谷底在 ep5 之後微升、
`train_loss` 續降 ＝ 過擬合指紋

⚠️ **「跑 20 個 epoch」不是「同一條 run 繼續跑」**——`epochs` 同時決定 OneCycle 排程長度，
改了是新實驗，而且兩組都得重跑（2 × 3.6h）才可比。

**刻意沒做**：沒把它產成分數過 `portfolio_lab` 找翻案（訊號層閘門沒過再去組合層找＝事後搜尋）。

→ 完整判讀：`obsidian_note/06 研究紀錄/03 F6 消融系列.md`

#### 2026-08-05 深夜 — V6.2 上線鏈路全部接完並驗證通過

新增五支程式，**全部純附加，V6.1／雙模型一個字未動**：

| 元件 | 檔案 | 驗證 |
|---|---|---|
| 推論 | `V6/run_v62_inference.py` | 整窗 582 天，**569/582（97.8%）**同時過 ρ≥0.95 與 Top50 重疊≥40；Group A/B **逐位元相同**（34/35 欄 `max|Δ|=0.0000`）＝接線正確 |
| 組合層 | `V6/v62_portfolio.py` | 獨立實作 replay 對 `portfolio_lab`：年化 **38.15% vs 38.15%**（差 0.002pp）、換手 82.7% vs 82.7%、都是 30 次再平衡 |
| 每日 orchestrator | `V6/run_v62_daily.py` | 自帶抓資料、當日資料檢查、Telegram 告警 |
| 後端 | `app/backend/routers/v62.py` | `/breadth/portfolio` |
| 前端 | `app/frontend/src/pages/BreadthPortfolio.jsx` | 分頁名叫「持股組合（20 日）」 |

**`run_v62_inference.py` 補齊與 `run_dual_inference` 的五個落差**——第五個是查程式才發現的：
**宇宙過濾**（`run_dual_inference` 完全沒有，而 base matrix 走 `filter_tradable_universe`）。
ETF 與興櫃留在橫斷面裡會改變 `clean_and_scale` 的 winsorize／z-score **分母**
→ **59 維每一維都會偏、且不報錯**。

**兩個照抄回測、不可自由發揮的口徑**：
① 分數＝`model.eval()` **單次前向**，不是 MC-Dropout 平均（MC 只當顯示用、不參與排序）
② 排序用 **raw score** 不用 `SQ=score/unc`（SQ 是雙模型那條線的設計，混用會讓實盤與回測不同義）

**組合層兩個設計**：再平衡觸發**用交易日曆算距上次幾天，不用「跑了幾次」**（漏跑一天不會永遠差一天）；
中途一度差 1.70pp，查出是**換手定義不同**（`portfolio_lab` 記 `buy_frac`、我記 `(buy+sell)/2`，
差在首次建倉的 0.5 分攤到 30 次 = 1.67pp，與觀測值吻合）。

**orchestrator 三個設計**：每日源**必須有當日資料（容許 0 天）**——既有健檢容許 5 天，
所以 margin 落後 1 天會顯示 ✓，**正是它漏掉真問題的原因**；
**特徵矩陣建一次、三個模型共用**（矩陣 3–5 分是成本大宗、GPU 前向每 arm 只要 1 秒）；
完整性隨當天持股一起落檔（事後補不回來）。

**⚠️ 我的驗證設計失誤**：第一次挑 **2026-06-02** 當測試日 → ρ=0.9398 **未過**。
那是**窗的最後一天**，而 MOPS 補的 2026Q1 財報 `available_from` = **2026-05-15**
→ 我保證挑中受新資料影響最大的那 13 天之一。逐日分布：
**2026-05-15 之前平均 ρ=0.9941、之後 0.9324**，分界線精確吻合。
**單日驗證挑在窗緣＝專挑最差的一天。**

三個模型並行（`v2_kg_nomacro` 5d 頭／`head10d`／`head20d`）。
**`{5d,20d}` 不需要訓練——`v6_short_H_h20.pt` 就是**（`head20d_ablation` 的產物）。
checkpoint 收在 `V6/checkpoints/`（新目錄、已 gitignore；`V6/models/` 受保護不放）。

#### 2026-08-05 — 三項會改變既有判讀的發現

**① 「資料修正後 checkpoint 會 OOD」實測為零 → 原訂的「重建矩陣＋全部重訓」取消**

base matrix 建於 07-30，之後 Group C 被改過（FCF 修正 + MOPS 補齊），
逐欄比對證實落差 **100% 在 Group C**（`Free_Cash_Flow` ρ 只有 **0.0911**、`EPS_Surprise` 0.2587、
`ROE` 0.6892），Group A/B 逐位元相同。

**我當時判斷「可能要重訓」。用現在的資料重評 582 天，實測推翻**：

| | 舊資料 | 現在資料 | Δ | 雜訊底線 |
|---|---|---|---|---|
| 年化 | 38.0% | **38.1%** | +0.1pp | ±2.68pp |
| decile spread Sharpe | 4.999 | **5.007** | +0.008 | σ=0.019 |
| mean IC | 0.1139 | **0.1139** | 0.0000 | — |

→ **每一項都在雜訊底線內**，且缺陷是八個 arm **共用**的。
**連帶取消**：矩陣重建、3 GB 上傳、七個 arm 重訓（≈29h A100）。

**② margin 不是 T+1，是我們抓太早**（更正一條寫錯的記載）

實測 **21:11** 打端點，`Date=2026-08-05` 的 margin **已經公布**；同一天 **19:35** 抓時回「尚未公布」。
`daytrade` 完全相同。→ **訓練資料沒有 look-ahead**，日期標記一直是對的。
**解法是排程 19:35 → 21:30，不需 `shift(1)`、不需重訓。**

完整抵達時間稽核：十個每日源裡八個在 19:35 就有當日資料；`dividend` 落後 1 天是**事件驅動**不是時間問題。

**③ EPS 截斷是局部缺陷、不是全歷史**

「EPS 剛好是整數」的比例逐年 2004→2026 **全部落在 1.0%~1.8%**、2026Q1 的 1.37% 同水位
→ 沒有系統性截斷。那 16 列只是 FinMind 2026Q1 那批的局部缺陷、且已被 MOPS 覆蓋。
**判別法：真實 EPS 幾乎不可能剛好是整數，所以「整數比例」不需要 ground truth 就能驗。**

#### 2026-08-05 — GRU 重訓 + 「同一把尺」稽核

**GRU 補 purge + 換 v2 基礎 + 三個 seed：排名第 2 → 第 4。**
舊表的 `+31.3% / Sharpe 1.401` 是用 **v1 資料基礎 + 完全沒有 purge** 跑的，
修正後 **19.6% / 0.996 → −11.7pp**。

**★ run-to-run σ（本專案第一次量到）**：三個 seed 的 Top50 年化 σ = **2.68pp**、Sharpe σ = 0.126。
**但 decile spread Sharpe 的 σ 只有 0.019 —— 穩定 40 倍。**
→ **比較模型優劣用 decile spread，不要用 Top50 年化**；而且**用 decile 就不需要每個模型都跑多 seed**。

⚠️ purge 必須**同時套用在各 horizon 的訓練列**上——只改 `val_start` 不夠，
被剔除的尾端日子會從 `val_rows` 溜回訓練迴圈。

**「同一把尺」全面稽核**（使用者提出「不然比較性就消失了」後做的）：
purge ✅ 統一／資料基礎 ✅ 全部 v2／標籤口徑 ✅ 一致／
**panel 原本差 0.74%**（已產 `__common.parquet` 重掃，**排名完全不變**）／
**seed ❌ 是剩下唯一真正的破口**。

**定稿八模型表**（共同 panel、全部 purge，**decile 為主要量尺**）：

| model | decile Sh ⭐ | 年化 | Sharpe |
|---|---|---|---|
| **v2_kg_nomacro** | **5.005** | 38.0% | 1.713 |
| gru (p30, v2) | 2.388 | 20.9% | 1.091 |
| v3_kg | 1.924 | 26.8% | 1.522 |
| v2_kg | 1.905 | 26.0% | 1.487 |
| gbdt (p30) | 1.735 | 9.7% | 0.639 |
| no_gat | 1.665 | 12.2% | 0.662 |
| old_kg | 1.229 | 16.6% | 1.036 |
| ridge (p30) | 1.088 | 20.5% | 1.211 |

⚠️ **decile 與年化的排序不一致**——decile 量的是**全市場排序能力**，年化量的是 **Top50 這個特定用法**。

#### 2026-08-05 — head20d 消融：20d 沒有勝出 + 第一次量到雜訊底線

**主結果（N=50/k=1.5/20 日）**：第二顆頭訓練在 10d = **+45.9%**、訓練在 20d = **+39.2%**
→ **20d 輸 6.7pp**，與 Ridge（+6.8pp）、GBDT（+10.6pp）方向相反。

**★★ 但兩個 5d 頭（同標籤、同 seed、同架構）的組合層差了 6.2pp** → **效應與雜訊同量級**。
而且那一對的訊號差異是**兩倍**，組合層差距卻幾乎一樣 → **效應沒有隨訊號差異放大**。

**跨 N 符號會翻轉**：N=10 +36.7pp、N=50 +6.7、**N=100 −0.5、N=224 −2.4**。
效應大的 N=10/25 正是已證實為單窗過擬合的區段 → **跑前定死的規則未達標**。

**結論：不要把上線規格改成 20d 標籤**——不是證明它較差，而是**沒有證據支持改動**。

**★★ 副產品：第一次量到組合層的雜訊底線 N=50 約 ±6pp。**
方法是免費的——**任何雙頭消融裡「兩組共用的那顆頭」天然就是完美的雜訊對照**。
連帶：八模型表裡**小於 6pp 的差距不該當數**。

⚠️ 跑前修掉一個會讓實驗作廢的缺陷（commit `aeeb134`）：
`head20d_ablation` **完全沒有設 seed**（`SEED` 只寫進輸出 JSON）→ 兩組會是髒配對。

#### 2026-08-04 — MOPS 財報直連 + FCF bug + Colab CLI

**★★ MOPS 整批直連上線，覆蓋斷崖補平**（`fetcher.py` **純附加約 750 行**）：
`financials`/`balance_sheet`/`cashflow` 2026Q1 從 216/16/16 支 → **各 1,972 支**、
月營收三個月從 951 → **1,926 支**。**一季兩次請求取代 FinMind 的 ~1,900 次逐股查詢。**
四項驗證全過（單次執行 exit 0、零警告）。

**★ 使用者「偏好 MOPS」的判斷被資料證實**：16 列被覆寫的 EPS，
**FinMind 的值全是整數且為朝零截斷**（14.54→14.0、**0.97→0.0**、**−0.37→0.0**），佔 7.4%，完全不報錯。

**★★ 最不直觀的發現：FinMind 三張財報的單季/累計慣例不一致**
（`financials` 單季、`cashflow` **年初至今累計**、`balance_sheet` 時點值）。
**照直覺寫下去，現金流特徵會只剩實際值的三分之一且不報錯。**

⚠️ 三個踩過的坑：**MOPS 有一般業/金融保險業兩種版面**（第一版整個金融業拿不到 `Book_Value`）；
**我寫的「暫時性失敗保留舊值」fallback 讓 13,340 列悄悄變成錯誤資料**（改成寧可丟棄）；
**V3/V4 第一版是我的參考值過時、不是資料錯**（改成用資料自身近況自我校準，沒有把區間放寬了事）。

**`Free_Cash_Flow` 兩層 bug 修好**（走 `fundamentals_v2`）：
① type 名稱猜錯 → **一直恆等於營業活動現金流、從沒減過資本支出**（22.2% 符號改變、Spearman 0.654）
② 沒處理累計慣例（我原本猜「會抵銷掉」，實測不成立）。
驗收 `max|Δ| = 0.000e+00`。⚠️ **我的第一版驗收腳本測錯了東西**（檢查指標本身也要驗證）。

**Colab CLI 裝好並測通**（詳見上方「Colab CLI」章節）：**CPU / T4 / A100 三種都拿得到**。

#### 2026-08-03 — 統一 purge + 標籤 horizon + 2×2 過組合層 + 資料稽核

**★★ 2×2 最佳格過 portfolio_lab：+38.0% / Sharpe 1.713，八個模型全面最好。**
WSL RTX 3060 前向只要 **7.2 分鐘**，不需 Colab、不需重訓。
對 `v2_kg` 是 **+12.0pp 年化、decile 1.905 → 4.999（2.6×）**，而 IC 只多 +0.015。

→ **「IC 小幅改善換不到錢」這個經驗法則要收窄**：它只在同一模型家族、輪廓形狀沒變時成立。

四項可疑處檢查全過（panel 逐對相同、對 Colab 逐日相關 1.0000、前後半都強、
Group D 輸入探針**雙向都驗**）。

⚠️ **交易限制的代價是全場最大**：擋漲跌停＋處置股後 −5.5pp / −7.3pp，
CLAUDE.md 原記「影響已量化（±1pp 內）」**在這個模型上不成立**（但扣完仍是最高）。

**★★ 統一 purge：既有表格系統性偏袒 Ridge/GBDT/GRU。**
稽核發現隔離處理是**混的**（Mamba 30 天、Ridge/GBDT/GRU **0 天**）。
代價：**Ridge −0.5pp、GBDT −6.7pp** → **GBDT 從第 4 名掉到最後一名**。
→ **`v2_kg_nomacro` 的勝幅是被低估的**。附帶：purge 的影響**在 20 天就飽和**。

**★★ 標籤 horizon：使用者的直覺是對的，但機制不是他想的那個。**
Ridge +6.8pp、GBDT +10.6pp（跑前定死的規則達標）。
**機制不是「預測過期」，是「短標籤製造付不起的換手」**——
如果是前者，長標籤只該在匹配頻率上贏；實際上 **Ridge 的 20d 標籤在五個頻率上全部最好**。
20 日再平衡下 Ridge 換手 81% → **56%**，而**毛報酬同時從 26.6% 升到 31.5%**。

**★ 資料抓取稽核抓到三個缺口**：財報三源覆蓋斷崖（✅ 08-04 MOPS 解決）、
月營收覆蓋腰斬（✅ 08-04 解決）、`trading_status_raw` 不在每日流程（⚠️ **仍未解**）。
✅ 10 個每日源最近 15 個交易日零缺漏。

**三個靜默問題修復**：`_add_alpha_targets` 的 TWII fallback 原本完全無聲（**只加警告不改值**，
四個 Alpha 欄逐位元回歸 `max|Δ| = 0`）；四支腳本用了協定依賴常數卻沒守門
（**修在 `baseline_common` 的 import 期橫幅**，所有消費端含未來新腳本自動受益，
**刻意不 raise**——v1 仍是合法設定，問題是「看不見」不是「不能選」）；
`label_10d.py` 加 `--check-only`。

#### 2026-08-02 — Group D 消融 + 2×2 四格 + GRU 診斷

**★ Group D（總經 12 維）移除讓 IC 顯著上升 —— F6 量到最大的單一效應。**
`with_macro` +0.0884 vs `no_macro` +0.1070 → **Δ = −0.0186、配對 NW t = −3.12**。

**機制清楚**：Group D 是**每日橫斷面常數**（實測抽 5 天，12 維的當日相異值數都是 1）
→ 在橫斷面排序上零資訊，但給了模型「記住是哪一天」的通道 → **它的作用是製造過擬合**。

**★ 沒預期到的發現：拿掉 Group D，連 2026 年的 IC 衰退也一起消失**
（滾動 60 日 IC 逐年 2024 +0.1146 / 2025 +0.1156 / 2026 +0.1117 幾乎持平，
而 `v2_kg` 是 0.1047/0.1061/**0.081**）。
→ **CLAUDE.md 原本把「2026 是 11 年最低」列為要盯的訊號——那多半是 Group D 的過擬合在失效，
不是訊號本身在退化。**

**2×2 四格全齊**：兩個效應**可加**（交互作用 −0.0032、t=−0.59 與 0 無法區分），
最佳格 **+0.1145 是同窗訊號層新高**、ICIR 1.340、IC>0 比例 90.9%。

**★ 移除 Group D 讓 GAT 的比較乾淨了 2.9 倍**：GAT 效應在 with macro 下 Δ=+0.0107 但 t 只有 1.73；
在 no macro 下 Δ 縮到 +0.0075 **t 反而衝到 3.80**。
→ **拿掉一個雜訊來源，比加一個訊號來源更能讓其他效應現形。**

**★ 一個掛著的異常被解掉了**：舊紀錄「`v2_kg` 訊號層贏 `no_gat` 但組合層輸」
——移除 Group D 後**兩層同調**。

**★ GRU 的 decile spread Sharpe 2.846 查清楚了：不是 bug、不是 panel 差異，是真的。**
四個「這是假象」的假設**全部被自己的量測排除**（限制在高流動前 1/3 後優勢**反而更大**）。
**必須先修的比較基礎**：GRU 與 v2_kg 出自**兩條計算路徑**，用同一把尺重算後排名才對。
機制＝**十分位輪廓的形狀**（D9−D0 = 0.0076 vs v2_kg 0.0049）；
**Spearman IC 由整個分布主導、對輪廓斜率不敏感**。

#### 2026-08-01 — F6 GAT 三組 + 組合建構基準版 + 11 年 WF

**★ GAT 三組消融**：`no_gat` +0.0884 / `old_kg` +0.0939 / **`v2_kg` +0.0991**。
**最關鍵的數字腳本沒印出來**：**`C − B = +0.0052、配對 NW t = +5.17`**。
B vs C 是**唯一乾淨的配對**；`std(C−B)=0.0140` vs `std(C−A)=0.0791`，**差 5.6 倍**
→ **不是效應小，是 A 的 RNG 不同讓配對差的雜訊淹掉訊號**。

**★ 組合建構基準版 v1.0 凍結**（規格跑之前凍結）。動機的量化依據：
換手 70–77%/次 × 每年 50.4 次 → **年化成本拖累 20.6%**，而 `v2_kg` 淨年化只有 +8.3%
→ **成本吃掉的比留下的還多**。

十一項發現裡最重要的四項：
① **緩衝有效且單調**（單一條規則值 **+5.6pp**）
② **每日再平衡是災難**（推翻使用者原本偏好每日的直覺；根因是 Top50 名單隔天只留存 47%）
③ **Ridge 與 GBDT 的訊號住在完全相反的排名段** → **固定用 Top50 當量尺會系統性偏袒頭部集中型**
④ **等權 eligible 宇宙年化 +15.2%**（新參照點）——現行頻率下**打不贏等權買全市場**

⚠️ **headline 設計失誤，已如實記錄不偷改**：我把「貼近實際操作」定成「每日再平衡」，
但使用者說的是「**每天看 dashboard**」。
**凍結規格的價值，就在它讓我自己的錯誤現形。**

**★ 11 年 WF 外部效度**（`wf_scores.py`，2015-04~2026-06 連續 OOS）：
**撐住的**——頻率單調、緩衝有效、下跌段保護、逐年 12 年裡 10 年正超額（2018 空頭年超額 +21.9%）。
**沒撐住的**——**N=10 是單窗過擬合**（WF 最佳在 N=25）。

**★ 拆解 F5 那個 −0.0079**：主因不是原本猜的那兩項。
查檔期才發現 **R0 的矩陣建於 07-12，而除權息還原切換是 07-29**
→ **資料修復整包 −0.0053（佔 71%）**。
**我的第二個假設也被自己的實驗推翻**（不是除權息還原——還原是逐股單調重新縮放、
橫斷面排名幾乎不動，ρ 全部 ≥0.985）。真正主因是 **Group B 籌碼欄的正確性修正**。

**★ 存活者偏差其實不存在**：實測 `prices_raw` 有 **211 支（9.7%）已下市/長期停牌且保留到最後交易日**
（B-3 逐日重抓的副產品）→ **這個長期限制可以劃掉了**。

**★ 漲跌停不需要新資料源，而且影響是正的**（我預期錯了兩次）：
可由還原收盤價的日報酬直接推導；Top50 的漲停比例只有全市場的 **0.1×（少 8.5 倍）**；
加上限制後**定案的低頻設定下反而變好**。

**★ v3_kg（加相關性邊）沒有效果 → 下修 Phase 4-A 期望值**：`D − C = −0.0002`。
三個數字放一起：加雜訊圖 +0.0055(t=0.91)／**把垃圾邊換成合理邊 +0.0052(t=5.17)**／
**再加 4,504 條相關性邊 −0.0002**。→ **圖的價值在「不要有錯的邊」，不在「有更多邊」。**

#### 2026-07-30 — F5 R-series 完成、特徵工程層規格凍結

共 **14 級**。**誠實總結論：沒有任何一項達到 +0.009 的實務門檻，唯一顯著的效應是負的。**
起點 2012→2013 +0.0001／旗標 +0.0000／中性化 +0.0025（t=1.06 不顯著）。
**F5 真正的價值不在 IC 上，而在把不該算的 0.003 拿掉。**

`fundamentals_v2` 的負 Δ **完全來自 look-ahead 移除**——
**(a) Q4/年報 `available_from` 45→90 天 = −0.0016（t=−4.06，全 F5 最顯著）**。
→ **與 purge 同列「正確性修正、一律採用、不套效益判讀表」**。

**「換欄」技巧**省掉 4 次 27 分鐘的重建（兩份矩陣可按 (Date, stock_id) 對齊）。

**我的假設被自己的實驗推翻**：主張「GBDT 才是旗標的對的裁判」，結果 GBDT 也否決
（7 個旗標 gain 佔比合計 0.32%，而 307 維均分是 0.326%）。
**非顯而易見的原因：起點移到 2013 與旗標功能高度重疊。**

**方法紀律四條**（寫進腳本檔頭）：存逐日 IC 用配對 NW t／判讀規則跑之前定死／
數值 import 自既有實作不另寫一套／GBDT 刻意不重掃網格。

#### 2026-07-27 ~ 29 — 資料層大修（36 項稽核 + 直連化 + 全歷史還原）

使用者定調「**先把資料完全修好，再談模型**」。**健檢警告 14 → 2 項**，資料損壞類全部歸零。

**★ 關鍵決策：不訂閱 FinMind VIP**——VIP 解決的是錯的問題（**差別在 API 形狀不在速率上限**：
FinMind 強迫逐股 ~2,000 次/3.6 小時，交易所是逐日整批 60 次/1.5 分鐘）。

**★ 三個「不會報錯」的線上問題，根因收斂到 `fetcher.py:1044` 一個型別 bug**
（yfinance 的 `Date` 是 `Timestamp`、direct 是 `str` → concat 後 `drop_duplicates` **完全失效**）。

**★★ 資料衛生必須在 `build_features` 之前**——舊流程的去重放在 `clean_and_scale` **之後**，
但時序特徵與當日 z-score **早就吃過重複列了**。**清理的位置比清理本身重要。**

**★ B-3 全歷史除權息還原重建**（3.84 小時重抓 + 官方因子表 26,385 筆）：
**每一項指標都優於舊資料**——`|報酬|>40%` 850→**471**、報酬 std 0.0817→**0.0349**、
p99.9 **+12.52%→+10.00%**（正好是台股漲跌幅上限，舊值在制度上不可能存在）。
過程中**又抓到一類漏掉的公司行為：減資**（純減資走股票換發、完全不在除權息表裡）。

**切換 production 的關鍵原則：只改值、不改型別**（新檔 `Date` 是 `timestamp[ns]`、
production 是 `large_string`，直接換會讓 `drop_duplicates` 靜默失效 ＝ 重演問題 1）。

**順帶修健檢的量測分母**：法人覆蓋率 96.1% → 89.5% 是**假摔**（分母多了 133 支 ETF）。
**不修會留下永遠不消的假警報——長期假警報會訓練人忽略警報，比沒有警報更危險。**

**14 個資料源改直連**（margin/daytrade/prices/foreign_shareholding/dividend→MOPS/
futures+options→TAIFEX/holdings→TDCC/per/securities/market_value），
每日更新從 5 個源擴到 14 個。過程抓到：**FinMind 對上櫃股把券賣/券買標反**（33.2% 的列）、
`Day_Trade_Volume` **2014–2026 全部 426 萬列都是 0**、
**財報三維死常數**（FinMind 把「淨利歸屬母公司業主」標成 `EquityAttributableToOwnersOfParent`，
且 `df_balance_sheet` 一直是從未被讀取的參數）。

**⚠️ FinMind 額度耗盡與 IP 封鎖**：402 被吞進 `logger.debug` ＝與「這支沒新資料」無法區分；
且**請求速率本身就會觸發封鎖**，不是把每日 600 次用完才擋。

→ 完整記載（含 13 條雷區、每個問題的「怎麼發現/如何解決/產生原因」）：
`obsidian_note/03 架構筆記/資料管線與修復史.md`、`docs/data-source-implementation-traps.md`

#### 2026-07-12 ~ 15 — 方向二 Baseline 四階對照完成

**答案是「用可解釋性換到多少效益？負的。」**
四階訊號層 IC 全部落在 0.089 ~ 0.103：Ridge +0.1015 / GBDT +0.1098 / **GRU +0.1113** / Mamba +0.0870。

**三條收斂證據：Mamba 贏不是架構紅利**——~49K 參數的 2 層 GRU 比 1.66M 參數的 v6_short 高 +0.024。
**⚠️ GBDT 教訓：IC 排名 ≠ 落袋排名**（IC 較高但 Top50 年化低 8pp）
→ **對照表必須訊號層與組合層並排**。

**IC 0.1015 排查（D0–D5）**：不是 bug、不需重抓資料。
最有價值的是 **D2 推翻「反轉主導」**——去掉全部價格技術特徵仍有 **+0.0717**
→ **0.10 是幾十個弱訊號分散疊加**（也解釋了 46/46 fold「乾淨得可疑」）。
**D3 證實基準錯配**（等權全市場不排序對 TWII −87.2%）。
**D4 成本是真正脆弱點**（成本 ×2 → +18.7% 變 −5.4%）。

#### 2026-07-06 ~ 07 — 雙模型效益追蹤 + 進場標準統一

使用者決定**暫停 Phase 3 實驗**，等真倉先驗證有沒有賺錢。
雙模型定位釐清為「**拿真實市場走勢驗證效益**」（真倉仍是 V6.1）。
新增 `dual_ic_analyzer.py` 自動算 IC / Top50 實現超額，前端第三分頁「🔬 雙模型驗證」。

**進場標準統一為分數制**（scanner 1.4）：原本 scanner「條件數 ≥2/4」與 sim「分數 ≥70」
是**兩套會分歧的標準** → 統一後 sim 績效才能直接回答「照 dashboard 買會不會賺」。
**權重/型態加分一律 import `signal_conditions`、不再自帶副本**（重複實作已實際造成 bug）。

**★ 機構資料管線修復（重大）**：2026-04-25 起 institutional_raw 每日只寫進 **7 支水泥股**
——**TWSE T86 缺 `selectType=ALLBUT0999`**（預設只回水泥類）+ TPEX 舊端點只回 HTML。
⚠️ **交易所 API 欄位對映必須數值驗證**（舊 parser 的投信/自營索引在 19 欄版面是錯位的）。

---

### 進行中

#### 🚀 V6.2 上線 — **已經在跑了**（2026-08-09 用 8/6 資料試跑並推上線）

**★ 使用者 08-09 的決定**：不等週一，直接用手上的 8/6 資料跑一次完整流程並推上去。
理由是①不必守在電腦前等 21:30 ②可以先看前端實際長什麼樣
③反正 08-15/16 本來就要清資料重來。

**試跑結果：exit 0｜8.4 分鐘｜11 份分數 × 19 個組合｜已推送｜Render 快取 200。**
資料日期 2026-08-06、10/10 個每日源完整、19 個 arm 各建倉 50 檔。
Render 三個端點都驗過（`/portfolio` `/arms` `/performance`）。

**★ 節奏（08-08 定，未變）**：現在跑的是**測穩定度**，
**08-15/16 把資料清掉、08-17 才正式開始累積實戰紀錄**。
→ **「模型集合定案日」是 08-17。**

##### 🗑 08-15/16 要刪的檔案（清完才是正式起跑）

```bash
cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba
rm V6/results/v62_state_*.json          # 19 份：持股狀態機
rm V6/results/v62_portfolio_*.jsonl     # 19 份：逐日紀錄（前瞻績效的原始資料）
rm V6/results/v62_performance.json      # 1 份：績效彙總
rm V6/results/archive/df_v62*_2026-08-*.csv   # 選配：試跑期間的分數快照
git add -A V6/results/ && git commit -m "v62: 清除試跑資料，08-17 正式起跑" && git push
# 然後 08-17 當天用 --first-day 重新建倉
```

**不要刪**：`v62_arms.json`（manifest，後端要讀）、`df_v62*.csv`（每天覆寫）。

⚠️ **刪 state 但沒刪 jsonl 會產生混合紀錄**：state 沒了會重新建倉，
但 jsonl 是 append 的 → `v62_performance` 會把 8 月的試跑段與正式段接在一起算，
**而且完全不會報錯**。兩類一定要一起刪。

##### 剩下的三件（使用者自己做，非阻塞）

1. **裝中文字型**（需 sudo 密碼）：
   `wsl -d Ubuntu -- bash -lc "sudo apt-get update && sudo apt-get install -y fonts-noto-cjk"`
   沒裝的話進度視窗會**自動改用英文標籤**並印出這行指令（不會出現豆腐方塊）
2. **`V6/.env` 加 Telegram 兩行**（選配）：`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
   （值可從 `PersonalOS/scripts/.env` 複製）。沒設的話告警只印 log、不 raise
3. ~~週一手動跑第一次建倉~~ **08-09 已經跑過了**（見上）。
   排程接下來每天 22:15 會自己跑，**不需要人在電腦前**。

**排程已生效**：`PersonalOS_Daily` 19:30 → **21:30**（V6.1+雙模型）、
新建 `MarketMamba_V62` **22:15**（`run_hidden.vbs` → `v62_daily.bat`，無小黑窗，
含 StartWhenAvailable 補跑 + WakeToRun 喚醒）。

⚠️ **使用者要的「不用登入也能執行」沒能做到**：
① Claude Code 的 shell **不是系統管理員**、建不了 S4U 工作
② 那個模式跑在 Session 0，而 **WSL2 需要使用者 session** → 極可能整條鏈壞掉。
**這一點沒能實測。** 要試的話用提權 PowerShell：

```powershell
Set-ScheduledTask -TaskName "MarketMamba_V62" -Principal (New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" -LogonType S4U -RunLevel Highest)
```

跑完看 `V6/logs/v62_daily.log` 是不是空的。

#### ⚠️ V6.1／雙模型的退役時機

使用者 2026-08-05 已同意拆掉，但**要等 V6.2 連續跑順幾天之後**。
**V6.2 已能自己抓資料**（`fetch_data()`，commit `d38943e`），過渡期兩者排程錯開即可。

#### ⏸ 待跑：`v2_kg_nomacro` 換 seed 量 run-to-run σ

使用者晚點自己跑，**不需改程式、不需 push**。指令與三個驗證檢查在下方 ▶ 區塊。

---

### 下一步

> ## ▶ 下次開工從這裡開始（2026-08-09）
>
> ### 🎯 系統已經在跑，沒有阻塞中的事
>
> **08-09 已用 8/6 資料整條跑過並推上線**（exit 0、8.4 分、19 個組合建倉完成）。
> 排程每天 22:15 自己跑，不需要人在電腦前。
> **下一個時間點是 08-15/16 清資料**——刪哪些檔案見「進行中」的 🗑 區塊
> （**state 與 jsonl 一定要一起刪**，只刪 state 會產生混合紀錄且不報錯）。
>
> **47 維 arm 已判定並結案**：Δ=−0.0014、NW t=−0.68 → 維持 59 維 + Group D 歸零。
> 「峰值在 ep8 會不會是 epoch 不夠」已查清＝不是（LR 已退火到 2.80e-10）。
> **V6.2 上線規格因此完全不變。**
>
> ### 📌 08-17 起跑前要決定的事
>
> - **模型集合定案**（晚加入的少了那段紀錄，永遠無法公平並列）。
>   目前 **19 個組合 / 11 份分數**，`v62_portfolio.py --list` 看全表（已按 family 分組）。
>   四個 F6 消融 arm 已於 08-08 併入（`v2_kg`/`v3_kg`/`old_kg`/`no_gat`，
>   macro 全歷史貼回也驗過了）→ **要不要再加，08-17 前決定**。
> - **前端呈現**：08-09 已做了第一輪（family 分組按鈕 + tier 色點 + 前瞻績效分頁）。
>   還沒處理的是**版面本身**——19 個組合 + 兩個分頁塞在一頁，能用但沒設計過。
>   使用者說「等全部上線、看得見全貌之後再想」。
>
> ### ⏸ 待跑：換 seed，量 Mamba 的 run-to-run σ
>
> 動機：現在有三個大結論壓在**單一 seed** 上，含已經上線的主線 **+37.3%**。
> 換 seed 重跑可同時拿到 ① 上線 headline 的誤差棒 ② dim47 的 −0.0014 落在幾個 σ 內
> （預期遠在雜訊內，但那會變成**量到的**、不是推論的）。
>
> ⚠️ **必須用全新的 runtime**：`dim47_ablation` 會把 `cfg.FEATURE_GROUPS` 的
> `macro_environment` **pop 掉**、`INPUT_DIM` 改成 47。同一個 session 接著跑會**靜默**吃到
> 47 維 config（Colab 坑 ①：模組快取不報錯）。前置照舊 **Cell 0 → 1 → 2 → 3**。
>
> ```python
> from experimental import groupd_ablation as gd
>
> import marketmamba.config as _cfg
> assert _cfg.INPUT_DIM == 59, f"config 是 {_cfg.INPUT_DIM} 維 → 請換全新 runtime 重跑 Cell 0~3"
>
> for _seed in (20260806, 20260807):          # 想先跑一個就刪掉第二個
>     gd.SEED = _seed                          # ⚠️ 一定要設在 gd 上，理由見下
>     print(f"\n■ seed = {_seed}", flush=True)
>     gd.run_groupd_ablation(
>         df,
>         arms=("no_macro",),                  # 只跑這個 arm，不碰 with_macro
>         use_gat=True,
>         kg_file="knowledge_graph_v2.npz",
>         epochs=10, early_stop=5,             # 與控制組同排程（epochs 也決定 OneCycle 長度）
>         cutoff_train_end="2023-12-31", purge=True,
>         val_end="2026-06-02", train_start="2013-01-01",
>         tag=f"_gatv2_s{_seed}",              # 避免覆蓋既有 checkpoint / JSON
>         drive_dir="/content/drive/MyDrive/MarketMamba_V6",
>     )
> ```
>
> **⚠️ 為什麼 seed 必須設在 `gd` 上**：`groupd_ablation.py:101` 是
> `from experimental.kg_ablation import (DROPOUT, SEED, ...)` ＝**值綁定**。
> 改 `kg_ablation.SEED` **完全沒作用且不報錯** → 照樣用 20260730 跑，
> 會得到三個一樣的結果、還以為 σ=0。與 `head20d_ablation` 當初「seed 只寫進 JSON」同型。
>
> **跑完的三個檢查（任一不符＝沒換到 seed）**：JSON 頂層 `seed` = 新值／
> `n_parameters` = **1,659,005**（與控制組相同＝架構沒被動到）／`eval_mean_ic_5d` **≠ 0.1145**。
>
> ⚠️ **用 Colab 網頁版跑，不要用 `colab run` CLI**——`head20d` 就是栽在
> WSL2 半夜自行重啟殺掉 keep-alive daemon。
>
> 跑完把 JSON 給 Claude → 算 σ + 用它標定 dim47 的 −0.0014 + 給上線的 +38.1% 補誤差棒。
>
> ### 上線之後的下一階段（使用者已同意的方向）
>
> - **多模型並行累積實戰紀錄**。三個模型已就緒（`v2_kg_nomacro` 5d 頭／`head10d`／`head20d`），
>   **`{5d,20d}` 不需訓練**——`v6_short_H_h20.pt` 就是
> - **模型集合要在起跑日定案**——晚加入的模型少了那段紀錄，無法公平並列
>   （47 維的結論是**不進來**）
> - **再平衡率是組合層參數、不是模型參數**——一份分數可同時跑 5/10/20 日，零成本
>
> ### 📋 交接單：`docs/session-handoff-2026-08-06.md`（一分鐘上手）

#### 【優先級 1】`v2_kg_nomacro` 的外部效度 —— 現在最大的缺口

它是全場最好的模型（+38.0%、decile 5.005），**證據卻只有單一 582 天多頭窗、單一 seed、無 WF**。
而大盤區間切分顯示它的優勢**全在上升段**（+91.9% vs `v2_kg` +58.2%），
**下跌段反而差 8.4pp**（−24.2% vs −15.8%）→ **對多頭 regime 的曝險比 `v2_kg` 更高**。

真 WF 要付 Colab 重訓的錢（每 fold 重訓 → 很貴），先討論值不值得。
便宜的替代已排上：換 seed 量 σ。

#### 【優先級 2】待決定 / 待清理

- [ ] **GBDT 那格的誤差棒沒人量過**（2026-08-08 新增）：八模型表的 `gbdt (p30) +11.2%`
      是單次跑的值。實測同樣資料、同 seed、只差 30 個訓練日 → 訊號層 ρ 只有 0.92。
      組合層雖然穩（11.0 vs 11.2），但**那是 n=2 的觀察**。要當數用得跑幾個變體量 σ
- [ ] **B 類每日面板每天重建一次（約 4 分）**：`run_v62_baselines.py` 走尾端窗，
      與 Mamba 線的矩陣**不能共用**（59 維 vs 66 維、config 是 module 級全域）。
      目前總時長 8 分可接受；要再壓的話得先確認兩邊協定能否合併，**不要為了省時間去混 config**
- [ ] **`trading_status_raw` 接進每日流程**：資料已補到 2026-08-03，
      但 `V6/experimental/fetch_trading_status.py` 的 `build()` 是**整檔重建**，
      直接排每日會每天重抓 11 年 → **需先加增量路徑**。組合建構的處置股限制要用它
- [x] ~~`{model}__common.parquet` 的處置~~ **2026-08-08 決定：保留**。
      它們是八模型定稿表的證據——**為了省 `--sweep` 的掃描時間刪掉研究證據是壞交易**。
      改成讓成本看得見：`sweep()` 未指定 `--models` 時會先印出「要掃幾個檔、其中幾個是衍生檔」。
- [x] ~~確認 portfolio_lab v1.1 修訂提案~~ **2026-08-08 全部裁決完畢**（規格 §8 有裁決總表）：
      **A 改 headline** ✅ 早在 08-01 就生效了（`portfolio_lab.py:95`，CLAUDE.md 這條待辦是過時的）；
      **C 補 10 日頻率** ✅ 同樣早已生效；
      **D 大盤區間報告** ✅ **採用**——成本近乎零，且直接對準最大未解風險
      （主線優勢全在上升段、下跌段差 8.4pp，而前後半切分測不到）；
      **B 分數平滑** ❌ **不採用進主規格**——原提案說「兩個模型同向」，
      加上 GRU 就不成立（Ridge +4.6pp、主線 +0.4pp 在雜訊內、**GRU −6pp 方向相反**），
      而採用要付主網格 300→900 組的三倍重跑成本。留在 `grid_ext`：算了但不進規格。
- [ ] **`INPUT_DIM` 59 → 47** 要不要正式落規格：證據充分但 **47 維實測未達標** →
      現行 mask 實作已足夠，屬「清理規格」，可無限期延後
- [x] ~~每日更新寫入端補「非交易日不寫入」gate~~ **早就做完了**（`fetcher.py:3844`
      的 `is_trading_day()`，週末不打 API、平日查 TWSE MI_INDEX）。
      2026-08-08 實測五個日期全對，含**端午節**（需要真的打 API 才判得出來）。
- [x] ~~清 329 列 `Close<=0` 存量~~ **實測已經是 0 列**——B-3 全歷史重建時就一起沒了。
- [ ] **P2：股票池 2026-05-25 少掉的 353 支歸因**（2,321 → 1,968，一日之間，非下市）
- [ ] ~~PersonalOS 同步 K 線圖~~ **使用者 2026-08-08 決定不做**
- [x] ~~可刪的備份~~ **2026-08-08 已清 8.4 GB**（23 個備份 728 MB + `baseline_cache_v2_v1like` 7.7 GB）。
      刪之前逐檔驗證「正式檔存在、可讀、鍵集合不少於備份」。
      ⚠️ **保留 6 個未通過驗證的**（`prices_raw_backup_*` ×4、`daytrade_raw_backup_20260728`、
      `holdings_raw_BACKUP`）：正式檔比備份**少 44 萬個 (stock_id, Date) 鍵**，
      缺漏均勻分布在 2007–2026 每一年、881 支各缺 >100 天。
      357 支整支消失的裡面有 352 支是 `filter_tradable_universe` 刻意排除的（興櫃），
      **但剩下那 44 萬筆部分缺漏還沒解釋** → 在解釋清楚之前不刪（那是唯一的副本）。
      ⚠️ 我的第一版檢查有 bug：`holdings_raw_BACKUP.parquet` 大寫沒被 regex 匹配，
      變成拿檔案跟自己比而回報「可刪」。**檢查腳本本身也要驗。**
- [ ] **其他衍生快取要不要刪（約 24 GB，不在原清單上）**：
      `baseline_cache`(v1) 8.1G、`baseline_cache_v2_neuind` 7.8G、
      `baseline_cache_v2_nofund` 7.3G、`baseline_cache_v2_v1univ` 819M。
      都是 F5 R-series 的變體快取、實驗已凍結、可由 raw 重建（但要數小時）。
      **`baseline_cache_v2` 8.0G 是現行協定，不可刪。**
- [x] ~~本機 git 善後~~ `trainer.py` 早已乾淨、stash 是空的。
      **`config.py` 維持 dirty 是刻意的**（V6.1 的 `run_daily_inference.py` 要 56 維），
      V6.1 退役前不要動它。

#### 【優先級 3】要做但不急

- [ ] **接班系統設計**（scanner 退役後）：候選規格已明確＝`v2_kg_nomacro` 分數 + N=50/k=1.5/20 日。
      缺口清單在 `docs/portfolio-construction-baseline-v1.md` §5b
      （整張交易資金門檻、漲跌停/停牌可成交性、持倉同步、部分成交、權重漂移再平衡）。
      **必須等優先級 1 通過才動工**
- [ ] **Phase 3 B~F**（2026-07-06 起暫停中）：B/C/D 三支檔案已寫好推 main、**尚未在 Colab 執行**
      （listnet 權重 sweep / 趨勢單尺度簡化 / 短線窗口 sweep），
      預期結果見 `docs/phase3-experiment-plan-2026-06-25.md`
- [ ] **Cell 4 多尺度用 `epochs=15~20` 重跑**（10–16h、約 300–440 元）：
      優先度低——已有三條證據指向 5d 目標下多尺度會退化
- [ ] **中性化用 Mamba 複驗**（F5 給的是方向正但不顯著，F6 唯一還沒做的一項）
- [ ] **Phase 4 產業理解融合**：計畫在 `docs/phase4-industry-chain-fusion-plan-2026-06-27.md`。
      ⚠️ **期望值已被 v3_kg 下修**（在合理邊之上再加邊 = 無效應）
- [ ] **資料基礎升級計畫階段二**：`planing/資料基礎升級計畫_baseline_common扶正.md`
- [ ] **定期全量重抓價格**：prices_raw 逐日增量寫入 → 未來的除息不會回頭調整已寫入的歷史。
      建議季度全量重抓一次

---

### 決策紀錄（每天會用到的操作紀律）

> **完整決策全紀錄在 `obsidian_note/07 決策與教訓/決策紀錄.md`（含時間、誰拍板、為什麼）。**
> **可帶到別的專案的通則在 `obsidian_note/07 決策與教訓/方法論教訓.md`。**
> 這裡只留「動手之前會用到」的那幾條。

#### 🔒 已定案，不要再重新討論

- **Group D 是負貢獻** → `fear_greed` / `business_indicator` / `fed_rate` **不補**；
  健檢那 3 項警告是預期中的，不是問題
- **47 維未達標** → 維持 59 維 + Group D 歸零（mask）
- **`Alpha_Nd` 沒有減大盤是刻意不修**（rank + Spearman 對當日常數免疫）
- **財報一律以 MOPS 為準**，但英文 `type` 沿用 FinMind 詞彙（含它標錯的）
  ——260 萬列歷史都是那套，改「正確」名稱會讓同一科目有兩套 key，**比一致地錯更糟且是靜默的**
- **標籤 horizon 的機制是「短標籤製造換手」**，不是「預測過期」
- **scanner 訊號系統跟 V6.1 一起退役**，由組合建構層接班
- **不訂閱 FinMind VIP**（差別在 API 形狀不在速率上限）
- **不用 Line Notify**（2025 年 3 月底已停止服務）

#### 🛠 動手之前

- ~~組合層掃描一律在 Windows 端跑~~ **已於 2026-08-08 解除**。原因是 `portfolio_lab.py:286`
  與 `f5_r_series.py:248` 沒有 `.copy()`（pandas 3.0 的 `to_numpy()` 回唯讀陣列），
  而**同一份檔案裡另外 5 處早就寫了 `.copy()`**——是漏改，不是版本不相容。
  修完兩邊數字實測一致（582 天 replay：freq=20 → 38.02% vs 38.02%、freq=1 → 19.91% vs 19.91%）。
  ⚠️ **這只證明 `portfolio_lab` 這條算術路徑等價**，不是整個 codebase 等價
  （`pd.qcut`、groupby 排序穩定性那類仍可能有差）。**別的管線要搬到 WSL，先對一次已知數字。**
- **版本不同調時，先查是不是只有幾行不相容**——Windows `pandas 2.2.2/numpy 1.26.4`
  vs WSL `pandas 3.0.2/numpy 2.4.3`。**不降 WSL**（`mamba_ssm`/`causal_conv1d` 是編譯 wheel，
  降 numpy 會斷整條推論鏈）、**不升 Windows**（九個模型的結果都在 1.26 上算的，
  升級要重驗整張表）。**改版本是治標且會一直追著跑，修相容性是一次性的。**
- **Windows 的 torch 已損壞**（`c10.dll` 初始化失敗）→ 任何要 torch 的東西只能在 WSL 跑。
  WSL 的 `lightgbm` 已補裝 **4.6.0**（`--no-deps`，對齊 Windows；動 numpy/scipy 會波及 mamba_ssm）。
- **背景任務會被砍** → 長工作用**前景分段**（每段 <10 分鐘）。
  且**背景任務回報 killed 不代表子孫都死了**——啟接力前要先列 process 確認
  （曾因此讓兩個 build 同時寫同一個輸出檔）
  - **★ 分不了段的整條管線（推論約 20~25 分）用 detach 跑**（2026-08-10 驗證可行）：
    寫成 `.sh` 用 `setsid nohup python … > out 2>&1 < /dev/null &` + `disown`，
    **從 PowerShell 呼叫 `wsl -d Ubuntu -- bash <腳本>`**
    （Bash 工具會把 `/mnt/c/...` 改寫成 `C:/Program Files/Git/mnt/c/...`；
    多層引號直接寫在指令列也會讓 `&` 後面整段失效、連輸出檔都不會建）。
    重啟前一定先 `ps` 確認沒有殘留，否則兩個程序會同時寫同一批 parquet。
  - **★ 監看要掛在「這次執行實際會寫的那個檔」**——detached 那次的 log 走 stdout
    導向的 `.out`，`inference.log` **一個字都沒寫**，掛在 `inference.log` 的 monitor
    全程零事件，**看起來跟「沒在跑」完全一樣**。沉默不是成功的證據。
- **啟動長跑前要先確認推論不在跑**（2026-08-04 實例：推論結束後排程接著啟動雙模型、RSS 9.7 GB）
- **全量重抓期間只做不同主機的作業**（TWSE 同時被兩邊打會出現 HTTP 307 限流；
  TAIFEX / TDCC / MOPS 各自獨立可並行）
- **git 一律指定檔案 `git add <檔>`，不要 `git add -A`**
  ——本機 `config.py` 是刻意保持 56 維的 dirty 檔，上去會覆蓋遠端的 59 維
- **checkpoint 不必手動下載**——Drive 桌面同步就掛在
  `G:\我的雲端硬碟\MarketMamba_V6\checkpoints\`

#### 🧪 改程式 / 跑實驗之前

- **會改變歷史特徵語意的 bug 一律用旗標、預設維持現況**（如 `fundamentals_v2`）。
  **驗收標準是「與 git HEAD 逐位元相同」**——凡動 `feature_engineer.py` 都要跑這個回歸測試
- **消融一律設計成「所有組架構等價」**（用 mask / 換內容，不用「拿掉模組」）。
  砍維度會改 `GROUP_DIMS` → sub_dim 重分配、參數量變、RNG 分岔
- **固定 seed 不等於完全隔離**——少建一層會少消耗 RNG，
  連 DataLoader 的打亂順序都不同（實測第一個 batch 一個 1,558 支、一個 1,668 支）
- **判讀規則跑之前先定死**。看到數字才選規則就沒意義
- **`epochs` 不該同時當「最多跑幾輪」與「OneCycle 排程長度」**——
  有 early stopping 時必須解耦。**改了 `epochs` 就是新實驗，兩組都得重跑才可比**
- **曲線尾端走平 ≠ 被截斷**：先看 `status_*.json` 的 `lr` 曲線
  （退火到 1e-10 ＝ 排程跑完）+ 逐 epoch Δ 有沒有收斂 + val_loss 谷底位置
- **baseline 要用同 harness 重跑值比，不用歷史值**
- **不依 test 集挑特徵子集**（那是 test-set selection）
- **正確性修正與效益改動要分兩套尺**——移除 look-ahead 讓 IC 下降是**誠實化的代價**，
  一律採用、不套 |Δ|≥0.009 門檻
- **小樣本/快速驗證模式的輸出，檔名必須與正式輸出分開**
  （`--max-days` 曾直接覆蓋正式分數檔，而 `--sweep` 是 glob 整個目錄）
- **`--tag` 要套在該次執行的「每一個」輸出上，特別是結果 JSON**——2026-08-08
  給 ridge/gbdt 加了 `--tag`，只套在模型與分數上、漏了 `RESULT_PATH`，
  **帶 tag 跑一次就蓋掉正式的 `baseline_*_result.json`**（靠 git 才救回）。
  `baseline_rnn.py` 本來就做對了（`_sfx` + 註解「絕不覆蓋既有的正式檔」）。
  **加旗標時要問：這次執行會寫出哪些檔？每一個都帶到了嗎？**
- **`--purge N` 之類的「天數」參數，要確認 N 數的是哪個空間的天**——訓練資料用
  `day_stride=2` 載入，在載入後的日期空間取倒數第 30 個 ＝ 真實日曆的倒數第 **60** 個。
  參數名說 30 卻做了 60。**一律走真實交易日曆換算**（`baseline_common.purge_cutoff()`）。
- **會覆寫共用結果檔的腳本，跑之前先備份**（`portfolio_lab --sweep` 是讀舊檔再合併，
  安全；但確認過才知道，不能假設）
- **★ 「算不出來」的預設值要往保守的方向倒，不可用 `?? 0` / `if x`**（2026-08-09）：
  `ann_stderr_pp` 在 n=1 時是 `None`，下游寫 `se = x ?? 0` 與 `if se and ...`
  → **null 變成「零誤差」，每一列都通過門檻**，效果與設計意圖完全相反。
  問法是：**這個欄位算不出來的時候，哪個方向是安全的？**
  誤差／不確定性 → 無限大；覆蓋率／信心 → 零；筆數 → 不可假設。
  ⚠️ 這類洞**合成測試常常測不到**（合成資料通常「正常」），要靠真的跑邊界。
- **★ 加一個設定欄位時，先問「這個設定在幾個地方被讀？」**（2026-08-08/09 連中三次）：
  `use_gat` 修了 `infer()` **漏了 `score_window()`**（`no_gat` 載權重當場炸）／
  `--tag` 套了模型與分數 **漏了結果 JSON**（蓋掉正式檔）／
  `SEG_WINDOW` 與 rolling 邏輯在兩支檔案**各一份**。
  修完手上那處就當完成，是最常見的漏網型態。
- **★ 比較基準不可寫死**（2026-08-09 一天內踩到**六個**地方）：前端寫死「主線是 38.0%」、
  router 測試寫死 `0.380`、`score_window` 寫死 `對照 +0.1145`、
  前端文案寫死「從 +38.0% 掉到 −19.9%」（**兩個數字都過時，而且 −19.9% 連正負號
  都是錯的**——實際是 37.3% → 25.0%）、router fallback 的 `backtest_ann: 0.380`、
  `--list` footer 說 bt_ann 出自 08-03 的 docs。
  重跑一次全部過時，而且**看起來還是很像正確的對照**。
  基準一律從單一真相（manifest / API / 該 arm 自己的參考檔）取值；
  **真的要寫在說明文字裡就不要寫數字**，講機制就好。
- **★ 分級規則要分「更差」與「不可比」**：只看 `|Δ|` 會把**更好**的組合標成
  「已知明確劣於主線」（`head10d_f20` 好 8.9pp 卻被標 inferior）。
  而跨訓練輪（隔離天數不同）的比較**本身就不成立**，要有獨立的
  `incomparable` 分級，**且視覺上不可與 inferior 同色**——
  否則「不能比」會被讀成「比較爛」。
- **凡是「排序取前 N」，一定要問並列怎麼打破**——分數是 float32，實測**一天有 183 組
  完全相等的分數**。`sort_values` 預設 quicksort **不穩定** → 同一份資料列序不同就可能
  選出不同持股。線上與回測必須用**同一套 tie-break**
  （回測是 `rank(method="first")`＝ stock_id 字典序 → 線上要
  `sort_values([score, stock_id], ascending=[False, True], kind="mergesort")`）。
  2026-08-08 實測：不對齊會造成 **582 天年化差 0.18pp**，且看起來像捨入誤差。
- **上線用的口徑必須逐行照抄回測**，不可「順便改好一點」
- **診斷實驗一律在 `V6/experimental/` 副本**；受保護的 `marketmamba/models/` 不碰。
  修推論一律改 `run_daily_inference.py`，**不要動已棄用的 `models/inference.py`**

#### 📊 判讀數字之前

> 完整的引用紀律（五條）在 `obsidian_note/06 研究紀錄/00 研究總覽.md`。

- **比較模型優劣用 decile spread，不用 Top50 年化**
  ——年化 σ = **2.68pp**，decile Sharpe σ 只有 **0.019（穩定 40 倍）**
- **八模型表裡小於 6pp 的年化差距不該當數**
- **★ 小樣本的年化必須附標準誤，而且「不顯示」的門檻要用算的**（2026-08-09）：
  年化的標準誤 ≈ 252 × s_daily / √n → n=20 是 **±68pp**、n=60 ±39pp、n=252 ±19pp，
  而組合層雜訊底線只有 ±6pp → **第一年之內年化排不出名次**。
  實測 30 天的紀錄算出「年化 −52.3% **±126pp**」——**沒有誤差棒的話，
  那個數字會被擺在 37.3% 的回測值旁邊，讀起來像「模型崩了」**。
  ⚠️ 只設門檻不夠：門檻那天會突然冒出一個看似精確的數字。**門檻 + 誤差棒兩個都要。**
- **「重不重現得出來」要分訊號層與組合層兩層問，不可簡化成一句話**（2026-08-08 GBDT 實例）：
  同一份資料、同一個 seed、只差 30 個訓練日 → **訊號層 ρ=0.9203、Top50 只重疊 25/50**，
  但**組合層 11.0% vs 11.2%、Sharpe 0.653 vs 0.639**。
  → **換進來的那半個 Top50 與被換掉的一樣好**；「持股名單對不上」≠「策略不同」。
  這正是上面那條 decile 紀律的機制。
- **模型對訓練窗微擾的敏感度差一個量級**：同樣多剔 30 天，
  **Ridge ρ=0.9970**（線性閉式解、係數幾乎不動）vs **GBDT ρ=0.9434**
  （切點是離散決策，151 輪 boosting 再放大）。**樹模型的個股名單天生不可重現。**
- **decile spread Sharpe 對窗長極度敏感**，11 年窗與 582 天窗**不可並列**
- **IC 要分層引用**（全市場 / 高流動 / 純籌碼基本面）
- **組合層基準用「等權 eligible 宇宙」（+15.2%），不用 TAIEX**
- **「IC 小幅改善換不到錢」只在同一模型家族、輪廓形狀沒變時成立**
- **系統性型態（頻率單調、N 依賴方向、A<B<C 一致）＞ 任何單一格子**
- **好結果先當可疑處理**；異常值先用可證偽的清單一條一條殺
- **兩輪之間隔離天數不同就不可並列**（30 vs 40 天 purge）

#### 🗄 資料工程

> 完整雷區 13 條：`docs/data-source-implementation-traps.md`

- **缺值看得見、錯值看不見**——寧可丟棄、寧可整季放棄，也不要用推測值填補
- **「抓不到」≠「還沒公布」≠「T+1 才有」**——在不同時間點打一次端點就能分辨
- **對映表要從資料反推、不要手寫**（手寫漏掉整個金融保險業版面且不報錯）
- **同一個資料商的不同表可能有不同的期間慣例，必須逐表實測**
- **交易所 API 欄位對映必須數值驗證**（用「買−賣=淨、分項加總=合計」恆等式）
- **換 production 資料檔：只改值、不改型別，而且守門要放在「寫入之前」**
  ——2026-08-08 我把 schema 檢查寫在寫入之後，型別已經 `large_string → string`
  才被抓到。**寫完再檢查只能發現、不能防止。**
- **★ 資料寫入端「縮小」必須明確授權**（2026-08-08 血淚）：
  `_append_to_parquet` 原本是「整天替換」，只要某天抓到的比已存的少就靜默刪除差額
  ——`prices_raw` 因此掉了 15% 的列、涉及 489 支可交易股票。
  **抓取失敗是常態，而「今天沒抓到」與「今天沒交易」在寫入端分不出來；
  分不出來的時候，保留是可回復的，刪掉不是。**
- **刪任何備份之前，先驗「正式檔不比備份少」**——不是比列數，是**比鍵集合**。
  上面那個 15% 的資料遺失，就是清備份時做這個檢查才發現的。
  ⚠️ 連檢查腳本本身都要驗：我的第一版 regex 沒匹配到大寫的 `_BACKUP.parquet`，
  變成拿檔案跟自己比而回報「可刪」。
- **反推不出來的口徑，就從資料把轉換學回來**：想用 `ex_rights_raw` 的因子重現
  正式檔的還原價，791 萬列只有 11.4% 對得上。改成逐股在重疊區學 `正式/備份` 比值
  （分段常數、兩側一致才採用），留出法 99.99% 通過。
  **不需要知道對方用什麼公式，只需要它自洽。**
- **清理的位置比清理本身重要**（衛生要在 `build_features` 之前）
- **在源頭修 vs 靠消費端各自防禦 → 源頭修一次**，但**源頭修不等於刪資料**
  （帶 PIT 事實的重複列要保留，改提供單一權威入口）
- **比率型健檢的分母要跟著重檢**——長期假警報會訓練人忽略警報，比沒有警報更危險
- **健檢一律 non-fatal**，且用 parquet statistics 取最大日期（不讀資料）
- **驗證的判準不可硬編記憶中的數字**，要用資料自身近況自我校準
- **★ 寫 parquet 一律 `index=False`**（2026-08-10 血淚）：索引沒有語意時落檔只會製造
  `__index_level_0__`。**危險的是它平常不會出事**——索引連續時 pandas 只寫 metadata、
  **不落成實體欄**，只有 `drop_duplicates` 真的丟掉列、索引變不連續那天才會實體化，
  於是跟檔案裡既有的同名 stray 欄撞名，之後**任何** `pd.read_parquet` 都會拋
  `Multiple matches for FieldRef.Name(...)`。08-10 因此讓當天三次執行全部 0 秒失敗。
  → **「平常看起來沒事」不是這個缺陷不存在的證據，只是還沒遇到觸發條件。**
- **★ 欄名重複的 parquet 只能用 `pq.ParquetFile(p).read()` 讀**——`pq.read_table()` 走
  dataset API，會先 unify schema 而直接 `Can't unify schema with duplicate field names`。
  要動這種檔案一律**依欄位「位置」**操作，name-based API 全部會失敗。
- **★ 診斷壞掉的欄，先看它的「值」而不是它的名字**：08-10 那個 stray 欄的值是
  `0 … 8,737,250` 單調遞增、非 NaN 數正好等於某次回補後的列數 → **一眼定位到是哪一次
  寫入留下的**，不必猜。欄名只告訴你它是索引，值才告訴你它是誰的索引。

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
