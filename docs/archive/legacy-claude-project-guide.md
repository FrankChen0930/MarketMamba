# Legacy Claude Project Guide

> Status: `HISTORICAL_ONLY / NON_AUTHORITATIVE`
> Archived: `2026-09-20` during K1 agent-guide integration.
> This file preserves the former 990-line Claude-specific project guide and its historical decisions. It must not be used for current project state, runtime status, model contracts, or next actions. Current authority: `AGENTS.md`, `knowledge/00_Project_Map/Current_State.md`, and `knowledge/00_Project_Map/Authority_Map.md`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MarketMamba — AI 助手指引

> **最後更新：2026-08-11**（過渡期全面盤點：CLAUDE.md 109 KB → 約 62 KB、
> 敘事移進第二層、新增事故簿與遷移地圖、建立 `.claude/agents/` 三個角色）
>
> **開工先讀本檔最下面「下一步」的 ▶ 區塊。**
> **這份檔案是 Project Constitution（規則與 invariant），不是百科全書。**
> 歷史敘事在 `obsidian_note/`、完整數字在 `docs/`。

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
├── 01 系統現況/現況整理.md              ← ★ 現在的 MarketMamba（不是歷史）
├── 01 系統現況/遷移地圖.md              ← ★ 新舊交界、誰依賴誰、移除條件
├── 02 問題追蹤/事故簿.md                ← ★ 14 起事故，六欄固定格式
├── 02 問題追蹤/已知問題清單.md          ← 未解問題、決定不修的限制
├── 03 架構筆記/資料管線與修復史.md      ← 直連化、MOPS、除權息還原、雷區
├── 03 架構筆記/{模型架構,推論流程,訓練紀錄,前端網頁}.md
├── 06 研究紀錄/00 研究總覽.md           ← ★ 八模型定稿表 + 引用紀律
├── 06 研究紀錄/01~05                    ← Baseline / F5 / F6 / 組合層 / 標籤
├── 07 決策與教訓/決策紀錄.md            ← ★ 選了什麼、為什麼、誰拍板
├── 07 決策與教訓/方法論教訓.md          ← ★ 可帶走的通則
└── 07 決策與教訓/外部架構參考 Fincept.md ← 外部參考的蒸餾結論
```

> [!important] 引用任何研究數字之前，先看 `06 研究紀錄/00 研究總覽.md` 的「引用紀律」
> 那裡有雜訊底線（N=50 年化帶 ±2.68pp）、窗長不可跨比、IC 要分層引用等五條，
> 不照著做會把雜訊當成結論。

**維護規則**：本檔的「最近完成」只留**索引**，敘事一律在第二層。
CLAUDE.md 曾在 2026-08-06 從 264 KB 壓到 68 KB、五天內長回 109 KB
→ **它會自然膨脹，超過 60 KB 就該叫 Archivist 檢查。**

---

## 🤖 Agent 工作流程（2026-08-11 建立，同日加入 Advisor 協調層）

### 角色

**可用 Agent tool 派遣的 specialized agent 只有三個**，定義在 `.claude/agents/`，
**身分存在於檔案裡，不依賴對話歷史**——`/clear` 之後可以完整重建。

| 角色 | 做什麼 | 不做什麼 |
|---|---|---|
| **builder** | 實作、除錯、測試、局部重構、技術調查 | 不改專案方向、不建大型抽象、不因難看就重構、不無證據刪 legacy |
| **verifier** | 獨立驗證、找靜默失敗與 regression、挑戰假設 | **不自行大改 production code**（產出是 Finding 不是 patch）、不把 style 當 bug |
| **archivist** | 維護 CLAUDE.md 與 obsidian、更新現況、提煉研究結論 | 不存對話紀錄、不為每個 task 建 log、不改 production code |

**Advisor 不是第四個 agent，不要用 Agent tool 派它。**
Advisor 是**主 session 自己遵循的 orchestration protocol**，定義在
`.claude/skills/advisor/SKILL.md`（需要跨角色協調、判斷該不該派 agent、
或要仲裁 Builder 回報與 Verifier Finding 時載入）。它負責：
理解 task → 判斷是否需要委派 → 建立 delegation brief →
控制 Builder／Verifier 的資訊隔離 → 收集結構化結果 →
把 Findings 壓縮成 decision point → 需要批准的交給使用者。

### 通訊模型：**共享 repo + 由 Advisor 傳遞脈絡**

```
                     使用者（最終決策權）
                            ↓
              Advisor ＝ 主 session（orchestration protocol）
             /              |              \
        builder          verifier        archivist
             \              |              /
                       repository

        builder  ──────X──────  verifier   ← 不得直接溝通
```

**Builder 與 Verifier 不互相溝通、不互相辯護。**
Verifier 的價值來自冷啟動——它不知道 Builder 希望答案是什麼。
**Advisor 是唯一在兩者之間傳遞脈絡的節點，而且只傳客觀事實，不轉發辯護性 reasoning。**
要辯論就升到使用者那裡。

**不建立 agent 之間的對話系統，不建立 `agent-a-message.md` 這類檔案。**
Agent 之間共享的是 code / tests / docs / CLAUDE.md / obsidian，**不是對話歷史**。
delegation brief 寫在 Agent tool 的 prompt 裡，不落檔。

### 記憶檢索紀律（不要把整個 vault 塞進 context）

```
永遠讀：CLAUDE.md + obsidian_note/01 系統現況/現況整理.md
然後依 domain 讀：
  資料/parquet  → 03 資料管線與修復史 + 02 事故簿
  特徵          → 06 研究紀錄/02
  模型/訓練     → 03 模型架構 + 03 訓練紀錄 + 06 研究紀錄/03
  組合/回測     → 06 研究紀錄/04 + 02 事故簿#INC-03
  退役/清理     → 01 遷移地圖
  引用任何數字  → 06 研究紀錄/00（引用紀律五條）
```

### 知識升級管線

```
對話 → docs/（artifact）→ obsidian_note/（canonical memory）→ CLAUDE.md（constitution）
```

**每一階都是提煉不是複製。** 對話不是記憶；不要因為某件事重要就把細節全塞進 CLAUDE.md。

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

**MarketMambaV6**（**1,659,005 參數**，Google Colab A100 訓練，本機 RTX 3060 推論）：

> ⚠️ 參數量在 repo 裡有三個不一致的版本：父層 `ProjectForMe/CLAUDE.md` 寫 11.5M、
> 本檔舊版寫 ~4M、F6 消融實驗的 JSON 寫 **1,659,005**。
> **以 1,659,005 為準**（有實驗直接佐證）。另外兩處是過期文件，見
> `docs/research/project-transition-audit.md` §6.1。

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

## 訊號系統（V6.1，退役中）

> [!warning] 這整套將隨 V6.1 一起退役（2026-08-01 使用者決定）
> 使用者說明這是「上課前自己想出來的粗糙策略」，判斷用組合建構層會更好
> → **不做舊系統的換手稽核，直接讓它退役。**
> **在 V6.2 連續跑順幾天之前不要拆。**

**元件**：`signals/scanner.py`(1.4)、`signals/signal_conditions.py`、`quant/pattern_scanner.py`、`backtest/sim_engine_v3.py`
**連帶要退的下游**：`action_signals.json`、`condition_analysis.json`、`portfolio_exit_check.json`、前端 `/legacy/signals`
**接班規格**：`v2_kg_nomacro` 分數 + N=50 / k=1.5 / 20 日再平衡

**維護期唯一需要記住的 invariant**：
進場標準是**分數制**（≥70 分，保守 ≥90），權重與型態加分**一律 import `signal_conditions`，不得自帶副本**
——2026-07-07 就是因為「同一套規則兩份實作」造成過真實 bug。

→ 完整規格（四條件權重、四層退場、Trailing Stop 四檔、7 種型態）：
`obsidian_note/01 系統現況/遷移地圖.md` 與 git 歷史 commit `da9a016` 之前的 CLAUDE.md

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

### 最近完成（索引；完整敘事已移到第二層）

> **2026-08-11 壓縮**：這一節原本有 770 行的完整事故敘事與逐項數字，佔 CLAUDE.md 的 46%。
> 那是「我們做過什麼」，不是「新 Agent 必須知道什麼」。
> **敘事已移到第二層，萃取出的規則保留在下方「決策紀錄」。**
> 需要細節時：`git log` + `obsidian_note/`。

| 日期 | 做了什麼 | 完整記載在哪 |
|---|---|---|
| **08-10/11** | 🔴 `prices_raw` 欄名撞名 → 當天三次執行**全部 0 秒失敗**。修資料 + 修根因，V6.1／V6.2 都補跑完成 | [[02 問題追蹤/事故簿#INC-01]] |
| **08-09（傍晚）** | 用 8/6 資料整條跑通並**推上線**（exit 0、8.4 分、19 組合建倉）。當場抓到 n=1 時 `ann_stderr_pp=None` 被 `?? 0` 吃掉的洞 | 事故簿 INC-04 |
| **08-09（下午）** | 前瞻績效接上線（`v62_performance.py` 之前**沒有任何人呼叫它**）+ 19 個組合按 family 分組 + 小樣本年化加誤差棒 | `06 研究紀錄/04 組合建構層` |
| **08-09** | 全面重評分：**資料修正沒有推翻任何結論**（11/11 arm 的 IC 變化 <0.009、組合層全在 ±6pp 內）。主線 38.0% → **37.3%** | `06 研究紀錄/00 研究總覽` |
| **08-08（晚 3）** | 🔴 `prices_raw` **真實遺失 15% 的列**（194,909 列、489 支可交易股票）。根因已修 + 已回補 441,543 列 | [[02 問題追蹤/事故簿#INC-02]] |
| **08-08（晚 2）** | F6 四個消融 arm 併入 → **19 個組合 / 11 份分數**；macro 全歷史貼回也套到 Mamba 線 | `06 研究紀錄/03 F6 消融系列` |
| **08-08（晚）** | 前瞻績效工具開跑第一天就抓到 `step()` 與 `replay()` 的並列打破不同（582 天差 **0.18pp**） | 事故簿 INC-03 |
| **08-08** | 多頻率並行 + B 類經典模型（Ridge/GBDT/GRU）上線 + **補上原本完全不存在的 push 鏈路** | `06 研究紀錄/01 Baseline 四階對照` |
| **08-06** | 47 維 arm 判定：**未達標結案**（Δ=−0.0014、NW t=−0.68）→ 維持 59 維 + Group D 歸零 | `06 研究紀錄/03 F6 消融系列` |
| **08-05（深夜）** | V6.2 上線鏈路全部接完並驗證通過（推論 569/582 天過雙判準；組合層 38.15% vs 38.15%） | `01 系統現況/現況整理` |
| **08-05** | 三項改變既有判讀的發現：①「資料修正後會 OOD」實測為零 → **取消全部重訓** ② **margin 不是 T+1，是我們抓太早** ③ EPS 截斷是局部缺陷 | 事故簿 INC-11 |
| **08-05** | GRU 重訓（補 purge + v2 基礎 + 3 seed）：排名第 2 → 第 4。**第一次量到 run-to-run σ**（年化 2.68pp、decile Sharpe 0.019） | `06 研究紀錄/00 研究總覽` |
| **08-05** | head20d 消融：20d **沒有勝出**；第一次量到組合層雜訊底線 **±6pp** | `06 研究紀錄/05 標籤與再平衡` |
| **08-04** | MOPS 財報整批直連（2026Q1 從 16~216 支 → **各 1,972 支**）+ `Free_Cash_Flow` 兩層 bug 修好 + Colab CLI 測通 | 事故簿 INC-07、`03 架構筆記/資料管線與修復史` |
| **08-03** | 統一 purge（**GBDT 掉 6.7pp、從第 4 名變最後一名**）+ 標籤 horizon + 2×2 最佳格過組合層 **+38.0%／Sharpe 1.713 全面最好** | `06 研究紀錄/03、05` |
| **08-02** | **Group D（總經 12 維）證實負貢獻**（Δ=−0.0186、NW t=−3.12）；拿掉後連 2026 年的 IC 衰退也消失 | `06 研究紀錄/03 F6 消融系列` |
| **08-01** | GAT 三組消融（C−B = +0.0052、NW t=+5.17）+ 組合建構基準版 v1.0 凍結 + 11 年 WF | `06 研究紀錄/03、04` |
| **07-30** | F5 R-series 完成、**特徵工程層規格凍結**。誠實結論：14 級**沒有任何一項達到 +0.009 門檻**，唯一顯著的效應是負的 | `06 研究紀錄/02 F5 特徵協定` |
| **07-27 ~ 29** | 資料層大修：36 項稽核 + 14 源直連化 + 全歷史除權息還原重建。健檢警告 **14 → 2 項** | `03 架構筆記/資料管線與修復史`、事故簿 INC-08/09 |
| **07-12 ~ 15** | 方向二 Baseline 四階對照完成。**Mamba 贏不是架構紅利**——49K 參數的 GRU 比 1.66M 的 v6_short 高 +0.024 | `06 研究紀錄/01 Baseline 四階對照` |
| **07-06 ~ 07** | 雙模型效益追蹤 + 進場標準統一為分數制 + 機構資料管線修復（每日只寫進 7 支水泥股） | 事故簿 INC-08 |

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

> ## ▶ 下次開工從這裡開始（2026-08-11）
>
> ### 🎯 系統已經在跑，沒有阻塞中的事
>
> **08-09 已用 8/6 資料整條跑過並推上線**（exit 0、8.4 分、19 個組合建倉完成）。
> 排程每天 22:15 自己跑，不需要人在電腦前。
> **下一個時間點是 08-15/16 清資料**——刪哪些檔案見「進行中」的 🗑 區塊
> （**state 與 jsonl 一定要一起刪**，只刪 state 會產生混合紀錄且不報錯）。
>
> ### 🔴 08-11 盤點發現的最高優先工程問題（等使用者決定是否動手）
>
> **全樹掃描確認：22 個 `to_parquet(` 仍缺 `index=False`**，其中 Tier A（索引必然不連續、
> 且在每日/同步活躍路徑上）包含 `fetcher.py:90`（ticker_universe）、`4151`（macro_raw）、
> `4347/4385/4423/4464/4503`（期貨/選擇權/TRI/股利/外資持股）。
> 08-10 修的只是 `_append_to_parquet` **一個呼叫點**。
> **致命形態需要「先有 stray 欄 + 再寫一次」，而 `V6/scripts/` 有 20+ 支讀寫 parquet 的修復腳本**
> → **下一次資料修復就可能重演。**
> 完整清單與最小 ratchet 計畫：`docs/research/verification-gap-analysis.md`
>
> **三個候選的架構裁決已完成**（`docs/research/architecture-decision-review.md`）：
> 回歸測試+ratchet **ACCEPT**／ResultsStore **ACCEPT WITH LIMITS**（抽象本身 DEFER）／
> SourcePolicy **拆兩半**（寫入守門 ACCEPT、freshness policy 層 DEFER）。
>
> ### 📌 47 維已結案（不必重開）
>
> Δ=−0.0014、NW t=−0.68 → 維持 59 維 + Group D 歸零。
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
