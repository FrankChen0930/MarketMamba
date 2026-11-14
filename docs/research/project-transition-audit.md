# Project Transition Audit — 新舊版本交接期全面盤點

> 日期：2026-08-11 ｜ 盤點者：Claude Code（本輪 **未修改任何 production code**）
> 姊妹文件：`verification-gap-analysis.md`（驗證缺口與 ratchet 提案）、`architecture-decision-review.md`（三個候選的裁決）
> 紀律：**Audit first. Evidence first. Minimal changes. Preserve working behavior.**
> **「看起來舊」不是刪除的理由。本文件只做分類與舉證，不做刪除。**

---

## 0. 為什麼現在做這件事

MarketMamba 同時處在四個交接點上：

1. **V6.1 → V6.2**（模型與選股邏輯）— V6.2 已上線試跑，08-17 正式起跑
2. **scanner 訊號系統 → 組合建構層**（決策方式）— 前者確定退役
3. **人工驗證 → 自動驗證**（品管方式）— 尚未開始
4. **單人開發 → 人 + 多 Agent 協作**（開發方式）— 本輪要建立基礎設施

**四個交接同時發生，是做盤點的最好時機，也是最容易出錯的時機。**

---

# 1. Current Architecture（目前真正 active 的架構）

## 1.1 執行鏈

```
平日 21:30  PersonalOS_Daily（Windows Task Scheduler）
  └─ WSL2 → V6/run_daily_inference.py（V6.1，56 維）
        └─ V6/run_dual_inference.py（雙模型，59 維）
              → git push → Render 快取刷新 → Vercel

平日 22:15  MarketMamba_V62
  └─ run_hidden.vbs → v62_daily.bat → WSL2 → V6/run_v62_daily.py
        [1] fetch_data()（自給自足，不再依賴 V6.1）
        [2] 當日資料檢查（容許 0 天）
        [3] 特徵矩陣建一次 → 8 份 Mamba 分數
            └ 另起 process（MM_PROTOCOL=v2）→ run_v62_baselines.py → 3 份 baseline
        [4] 組合層狀態機 × 19 個組合
        [5] 另起 process → v62_performance.py
        [6] git push → POST /api/v62/cache/refresh
```

## 1.2 分層

| 層 | Active 元件 |
|---|---|
| 資料 | `marketmamba/data/fetcher.py`（215 KB，14 源直連 + FinMind + yfinance）、`hygiene.py`、`merger.py` |
| 特徵 | `feature_engineer.py`（94 KB）、`feature_spec.py`、`macro_ts_full.py` |
| 模型 | `models/architecture.py`（**受保護**）、`models/trainer.py` |
| 推論 | `run_daily_inference.py` / `run_dual_inference.py` / `run_v62_inference.py` / `run_v62_baselines.py` |
| 組合 | `v62_portfolio.py`（狀態機 + manifest）、`experimental/portfolio_lab.py`（研究，規格凍結） |
| 績效 | `v62_performance.py` |
| 發布 | 各 runner 自帶的 push（見 §5） |
| 服務 | `app/backend/`：10 個 router，全部走 GitHub raw + 1h TTL |
| 前端 | `app/frontend/`：三代並存路由（見 §3.3） |
| 記憶 | `CLAUDE.md`（**109 KB**）、`obsidian_note/`（14 篇）、`docs/`（27 份）、`檢查表與實作紀錄/`（11 份）、`planing/`（9 份） |

## 1.3 儲存

**沒有資料庫。** 兩種持久化：

- `Data/`（本機，不進 git）：parquet 面板 + 衍生快取（約 24 GB 待決）
- `V6/results/`（**進 git**，66 檔）：CSV / JSON / jsonl → GitHub raw 供後端讀

> **git repo 被當成資料交換匯流排在用。** 這是一個**沒有被明確命名但很成功**的架構決定：
> 免運維、天生有版本、天生可稽核，而且讓「結果」與「產生它的程式碼」永遠在同一個 commit。
> 任何未來的 storage 提案都必須先說明它如何保留這三個性質。

---

# 2. Legacy Architecture（仍屬舊版本）

| 元件 | 狀態 | 證據 |
|---|---|---|
| **V6.1 推論鏈**（`run_daily_inference.py`，56 維） | **仍每日執行**，已宣告非紅線 | obsidian `01 現況整理`：「2026-08-01 使用者明講已非紅線……但不要主動去停」 |
| **雙模型**（`run_dual_inference.py`） | 仍每日執行，定位是「驗證效益」 | CLAUDE.md 進行中區塊 |
| **scanner 訊號系統** | 仍執行，**已宣告退役** | CLAUDE.md「訊號系統（V6.2）」整節標註「將隨 V6.1 一起退役」 |
| 連帶退役的下游 | `action_signals.json`、`condition_analysis.json`、`portfolio_exit_check.json`、`sim_engine_v3`、前端 TradingSignals | 同上 |
| `archive/`（V4 / V5 / V5.5 / V5_models / V5_scripts / docs_old / Get_Data / notebooks） | 只讀歷史 | CLAUDE.md「不在活躍維護範圍」 |
| `V6/notebooks/not_using/`（4 檔） | 目錄名即宣告 | — |
| `models/inference.py` | **已棄用**（D4），欄位與 `run_daily_inference.py` 已分歧 | CLAUDE.md 注意事項；`models/` 受保護不可改 |

---

# 3. Transitional Components（新舊交界）

## 3.1 `config.py` 的 56 / 59 維分裂 ⚠️ 最高風險交界

- 本機 `config.py` = **56 維**（V6.1 推論用），**刻意保持 dirty 不 commit**
- GitHub main = **59 維**（Colab 訓練 / V6.2 用）
- baseline 線又是 **66 維**（`MM_PROTOCOL=v2`，另起 process）

**證據**：`git status` 顯示 ` M V6/marketmamba/config.py` 至今仍 dirty。

**風險**：`git add -A` 會把 56 維推上去覆蓋遠端 59 維 → **V6.2 與 Colab 訓練同時壞掉**。
CLAUDE.md 已有「一律 `git add <檔>`」的紀律，但**沒有任何機制阻止**。

## 3.2 三套推論並跑，共用同一個資料層

`fetcher.py` 同時服務 V6.1 / 雙模型 / V6.2 三條鏈。
**這代表資料層的任何改動同時影響三條線**——這是為什麼「先做回歸測試再動 `fetcher.py`」是對的順序。

## 3.3 前端三代路由並存（**設計良好，不是債**）

`app/frontend/src/App.jsx` 清楚劃分：

| 路由 | 對應 | 註解原文 |
|---|---|---|
| `/breadth/*` | **V6.2，目前上線中** | 「廣度量化模型 /breadth/*（V6.2，目前上線中的版本）」 |
| `/conviction/*` | 高信念模型 | — |
| `/legacy/*` | **V6.1，已凍結** | 「前一版 V6.1 /legacy/*（已凍結，保留對照用）」 |
| `/compare` | **尚未串接** | 「模型分歧看板（尚未串接，暫時不放進主導覽）」 |

而且保留了 **10 條舊網址重導向**（2026-07-30 與 2026-08-11 兩次頁面樹重整），註解寫「三個舊網址都還活著，不留死連結」。

> ✅ **這是本次盤點中最健康的交接實作。** 版本邊界明確、舊入口不斷、意圖寫在程式碼裡。
> **不要動它，而且應該把它當成其他層的範本。**

## 3.4 `results/` 的雙世代 metadata

- V6.2 產物（`v62_state_*.json` ×19）：有 `last_date` / `data_complete` / `spec` / `n_rebalances` — **設計完整**
- V6.1 產物（`df_kelly.csv`）：只有資料列的 `Date` 欄，無任何 metadata

**判定**：不必補 V6.1，因為它正在退役。**為退役中的線做基礎建設是純虧損。**

---

# 4. Active Execution Paths（真正會被執行的）

| 路徑 | 觸發 | 頻率 |
|---|---|---|
| `run_daily_inference.py` | Task Scheduler 21:30 | 每交易日 |
| `run_dual_inference.py` | 上者串接 | 每交易日 |
| `run_v62_daily.py` → `run_v62_inference` + `run_v62_baselines` + `v62_portfolio` + `v62_performance` | Task Scheduler 22:15 | 每交易日 |
| `fetcher.run_daily_update()` | 被上面兩條各呼叫一次 | 每交易日 ×2 |
| `hygiene.check_data_health()` | 推論內 | 每交易日 |
| `app/backend/*`（10 router） | Render 常駐 | 持續 |
| `notebooks/v6_colab_training.py` | 人工 | 重訓時 |
| `experimental/portfolio_lab.py` | 人工 | 研究時 |

---

# 5. Legacy / Dead Candidates（需要證據才能確認）

> **本輪只舉證，不刪除。**

| 候選 | 證據 | 判定 | 移除條件 |
|---|---|---|---|
| **`marketmamba/deploy/publisher.py`** | ① 寫到 `REPO_ROOT/results/`，該目錄**實測不存在**（`Test-Path results` = False）② docstring 寫「so the **Streamlit** frontend can auto-refresh」— Streamlit 早已不存在 ③ 唯一 import 者是它自己的 `deploy/__init__.py` ④ 實際在用的是 `run_daily_inference.py:1165 _push_to_github()` 與 `run_v62_daily.py:275 push_to_github()` | 🔴 **DEAD（證據充分）** | 確認 `marketmamba/__init__.py` 未透過 `deploy` 匯出後即可刪 |
| `app/backend/mock_data.py` | 仍在 production backend 目錄；CLAUDE.md 記載 `/api/portfolio` 仍是 mock | 🟡 **仍被引用，需查 router** | 待 `/api/portfolio` 接真實資料 |
| `V6/V6_Master_Plan.md` | 標頭「**狀態：規劃中 (Planning)**、最後更新 2026-04-24」，描述的是 V5.5→V6 升級 | 🟡 **OBSOLETE PLAN** | 移進 `docs/` 或 archive；**不要刪**（有歷史價值） |
| `V6/V6_V61_Implementation_Guide.md` | 最後更新 2026-05-12，涵蓋 V6.0→V6.1 | 🟡 **LEGACY DOC** | 同上 |
| `V6/data_check_result.txt`、`quick_check_result.txt` | 一次性輸出檔留在 V6 根目錄 | 🟡 **殘留輸出** | 可移入 `V6/logs/` 或刪 |
| `V6/notebooks/not_using/`（4 檔） | 目錄名即宣告 | 🟢 **已隔離，無害** | 不必動 |
| `backtest/sim_engine.py` / `sim_engine_v2.py` | v3 是現行；v1/v2 未見於每日路徑 | 🟡 **需確認無 import** | V6.1 退役時一併 |
| `backtest/scanner_engine.py`、`condition_analyzer.py` | scanner 系列，已宣告退役 | 🟡 | scanner 退役時 |
| `V6/scripts/` 的 20+ 支一次性修復腳本 | `fix_*` / `backfill_*` / `refetch_*` / `merge_staging_202607` 等 | 🟢 **保留** — 它們是**事故的證據與可重跑的修復程序**，`fix_prices_index_column.py` 的檔頭甚至保留了未定論的疑點 | 不建議刪 |
| `archive/V5.5_repo` | `git status` 顯示 `?` （未追蹤且被 gitignore 規則部分覆蓋） | 🟡 需確認是否該進 gitignore | — |

---

# 6. Documentation Debt

## 6.1 過期 / 矛盾（會直接誤導 Agent）

| 文件 | 問題 | 證據 |
|---|---|---|
| **`OVERVIEW.md`**（22 KB，2026-06-13） | 寫「每日收盤後（**17:00**）」「~2,515 支」 | 實際是 **21:30 / 22:15**（2026-08-05 改），df_kelly 實際 **1,925 檔** |
| **`PROJECT.md`**（6 KB） | 寫「全市場 **2888** 支」 | 與 README 的 ~2,515、實際 1,925 三方不一致 |
| **`README.md`**（14 KB） | Live Demo 指向 `marketmamba.vercel.app` | CLAUDE.md / 父層 CLAUDE.md 都寫 `market-mamba-pi.vercel.app` → **需確認哪個是真的**（NEEDS EVIDENCE，可能是新網域） |
| **父層 `ProjectForMe/CLAUDE.md`** | 寫「推論時間：每日 **17:00**」、「11.5M 參數」 | 實際 21:30、**~1.66M 參數**（CLAUDE.md 本身寫 ~4M，obsidian 寫 1,659,005）→ **三處不一致** |
| `V6/V6_Master_Plan.md` | 「狀態：規劃中」 | 該計畫早已執行完畢 |
| `docs/session-handoff-2026-08-06.md` | `git status` 顯示 ` M`（已修改未提交） | 交接單過期，且 CLAUDE.md 仍指向它 |

> ⚠️ **參數量三處不一致（11.5M / ~4M / 1,659,005）是最容易讓新 Agent 說錯話的一項。**
> 1,659,005 有 F6 消融實驗的直接佐證，應視為權威值。

## 6.2 重複

| 內容 | 出現在 |
|---|---|
| 專案定位一句話 | `CLAUDE.md` / `README.md` / `OVERVIEW.md` / `PROJECT.md` / 父層 `CLAUDE.md` / obsidian Home — **6 份** |
| 每日排程說明 | `CLAUDE.md` / `OVERVIEW.md` / obsidian `Home` + `現況整理` — 4 份，其中 2 份過期 |
| 模型架構圖 | `CLAUDE.md` / `OVERVIEW.md` / `PROJECT.md` / obsidian `模型架構` / `V6_V61_Implementation_Guide.md` — 5 份 |
| 八模型定稿表 | `CLAUDE.md` / obsidian `00 研究總覽` / `docs/baseline-comparison-table` — 3 份 |

**判定**：README/OVERVIEW/PROJECT 是**對外**文件（README 已改成作品集導向），與 CLAUDE.md 的**對內**用途不同 → **重複本身可接受**，但**過期不可接受**。

## 6.3 `docs/` 分類（27 份）

| 分類 | 檔案 | 處置 |
|---|---|---|
| **KEEP（研究證據，長期價值）** | `portfolio-lab-results-2026-08-01.md`(45K)、`f6-training-log-and-readout.md`(30K)、`feature-protocol-v2.md`(26K)、`data-source-implementation-traps.md`(30K)、`label-horizon-vs-holding-period`、`portfolio-construction-baseline-v1.md`、`head20d-ablation-result` | 不動。它們是 obsidian 第二層引用的第三層 |
| **KEEP（本輪新增）** | `research/fincept-terminal-analysis.md`、`research/market-mamba-gap-analysis.md`、`research/future-architecture.md`、`research/architecture-decision-review.md`、本文件、`research/verification-gap-analysis.md` | — |
| **UPDATE** | `session-handoff-2026-08-06.md`（已 dirty 且過期） | 由 Archivist 決定：更新或標記為歷史 |
| **ARCHIVE 候選** | `baseline-experiment-protocol-draft-2026-07-11.md`(draft)、`phase3-experiment-plan-2026-06-25.md`（暫停中）、`phase4-industry-chain-fusion-plan`（期望值已被 v3_kg 下修）、`conviction-c-analysis-2026-07-11.md`、`breadth-pipeline-page-draft-2026-07-12.md`(draft) | 移入 `docs/archive/`；**不刪** |
| **DEFER** | `gap-analysis.md`（11 K，日期不明，與本輪 `research/market-mamba-gap-analysis.md` **名稱高度混淆**） | 需人工確認內容後改名或歸檔 |

> ⚠️ **`docs/gap-analysis.md` 與 `docs/research/market-mamba-gap-analysis.md` 名稱太像**，
> Agent 很可能讀錯一份。這是**新引入的**混淆，必須處理。

---

# 7. Memory Debt

## 7.1 `CLAUDE.md`：已經回到 Encyclopedia 狀態

**實測 111,719 bytes（109 KB）。**

git 歷史 `d93926d`「CLAUDE.md 整理：拆成兩層記憶，264 KB → 68 KB」（2026-08-06）。
**五天之內從 68 KB 長回 109 KB（+60%）。**

結構分析（總 1,672 行）：

| 區塊 | 行數 | 佔比 | 性質 |
|---|---|---|---|
| L1–128 記憶分層 / 標記慣例 / 互動規則 / 協作偏好 | 128 | 8% | ✅ **Constitution** |
| L129–252 專案定位 / 目錄 / 模型架構 / 排程 | 124 | 7% | ✅ Constitution（但與 OVERVIEW 重複） |
| L253–312 訊號系統 V6.2 | 60 | 4% | ⚠️ **整節已宣告退役**，仍完整保留 |
| L313–481 Colab / 資料管線 / 部署 / 環境變數 / Colab CLI / 常見任務 | 169 | 10% | ✅ Constitution |
| **L486–1255 「最近完成」（19 個日期條目）** | **770** | **46%** | 🔴 **這是 Encyclopedia**——完整事故敘事、逐項數字、當時的推理過程 |
| L1256–1325 進行中 | 70 | 4% | ✅ 但**部分過期** |
| L1326–1482 下一步 / 優先級 1–3 | 157 | 9% | ✅ Constitution（現在要做什麼） |
| L1483–1671 決策紀錄（操作紀律） | 189 | 11% | ✅ **最高價值的 Constitution** |

**判定：L486–1255（46%）應該搬到 obsidian。** 它們是「我們曾經做過什麼」，不是「新 Agent 必須知道什麼」。

**但注意**：那 770 行裡混雜著**真正的 invariant**（例如「`to_parquet` 一律 `index=False`」「並列怎麼打破」「`None` 必須當成無限大」）。
**這些已經在 L1483–1671 的「決策紀錄」被萃取過一次**——所以搬移是安全的，只要保留萃取後的規則。

## 7.2 `CLAUDE.md` 內容過期

| 位置 | 過期內容 | 實際 |
|---|---|---|
| 「▶ 下次開工從這裡開始（2026-08-09）」 | 日期是 08-09 | 今天 08-11，且期間有兩個 commit（`89b4617` 修欄名撞名、`da9a016` 前端拆頁） |
| 核心模型架構「~4M 參數」 | 與 F6 實驗的 1,659,005 不符 | obsidian 與實驗 JSON 為準 |
| 「訊號系統（V6.2）」整節 | 已宣告退役 | 應壓縮成「退役中 + 指向 obsidian」 |

## 7.3 `obsidian_note/`：結構成熟，但**現況已過期**

**結構評價：良好，不要重建。** 07 個分區、`🏠 Home.md` 有清楚導覽、note 之間有 `[[wikilink]]`、每篇有更新日期與 tags。

**但 `01 系統現況/現況整理.md`（2026-08-06）已經過期，而且是危險的過期**：

| 它說 | 實際 |
|---|---|
| 「週一 **2026-08-10** 上線」 | **08-09 就已經跑過並推上線**（exit 0、8.4 分、19 個組合建倉） |
| 「上線前使用者要自己做的三件事……③ 週一手動跑第一次建倉」 | **③ 已完成**。一個新 Agent 讀到這裡會叫使用者再跑一次 `--first-day` → **會覆蓋現有狀態機** |
| 「V6.2 🚀 週一上線」 | 已在跑，且 **08-15/16 要清試跑資料、08-17 才正式起跑** |
| 完全沒提 08-08 的 15% 資料遺失、08-10 的欄名撞名 | 兩起重大事故不在第二層記憶裡 |

`🏠 Home.md` 的「🎯 現在最重要的事」同樣停在 08-06。

> 🔴 **這是本次盤點發現的最高優先記憶債**：不是「資訊不足」，是**記憶會主動誤導 Agent 執行破壞性操作**。

## 7.4 缺失的記憶

| 缺什麼 | 為什麼重要 |
|---|---|
| **事故簿**（統一 schema 的 incident ledger） | 事故散落在 CLAUDE.md 敘事、`03 資料管線與修復史`、`02 已知問題清單` 三處，格式不一，無法快速回答「這類問題以前發生過嗎」 |
| **遷移地圖** | 新舊交界目前只存在於 CLAUDE.md 散落的段落與 `App.jsx` 註解 |
| **Fincept 研究的蒸餾結論** | 四份研究在 `docs/research/`（共 ~100 KB），第二層完全沒有對應 note |
| **Agent 工作流程** | CLAUDE.md 有互動規則與協作偏好，但沒有「Builder / Verifier / Archivist 各自該讀什麼」 |

## 7.5 過度細節（應下沉）

CLAUDE.md 的「標籤 × 再平衡的完整矩陣」（L56–84，30 行的完整數字表）
→ obsidian `06 研究紀錄/05 標籤與再平衡` 已有 → **CLAUDE.md 保留結論即可**。

---

# 8. Technical Debt（真正存在的）

依「是否肇事」分級，**不依美觀**：

## 8.1 已肇事

| # | 債 | 證據 |
|---|---|---|
| T-1 | **22 個 `to_parquet` 缺 `index=False`** | `verification-gap-analysis.md` §1 逐行確認 |
| T-2 | **零自動化驗證** | 無 `.github/`、無 pytest |
| T-3 | 線上／回測口徑一致性無守門 | 08-08 的 0.18pp |
| T-4 | 誤差欄位 null 處理無守門 | 08-09 的 19 個 arm 全被誤標 |
| T-5 | 硬編比較基準 | 08-09 一天六處 |

## 8.2 未肇事但結構性

| # | 債 | 說明 |
|---|---|---|
| T-6 | `deploy/publisher.py` 死碼 | §5 |
| T-7 | 後端 4 個 router 各抄一份 TTL 快取；URL 用 `.replace()` 組 | `v62.py:40`、檔頭自承抄自 `dual.py`/`signals.py` |
| T-8 | 回測/模擬引擎 4 套並存 | engine / sim_engine v1,v2,v3 / portfolio_lab |
| T-9 | 前端單檔過大 | `InvestmentSim.jsx` 57 KB、`QuantAnalysis.jsx` 43 KB — **但兩者都在 `/legacy` 或非主線** |
| T-10 | 約 24 GB 衍生快取待決 | CLAUDE.md 優先級 2 |
| T-11 | `trading_status_raw` 未進每日流程 | obsidian R-3 |

## 8.3 明確**不是**債（避免未來被誤判）

| 看起來像債 | 為什麼不是 |
|---|---|
| `fetcher.py` 215 KB 平坦模組 | 它現在是對的。每個 `fetch_*` 對應一個真實端點的真實版面差異 |
| 4 套新鮮度門檻 | **刻意不同**：`run_v62_daily.py:27` 明文說明它必須比健檢更嚴 |
| 11 個 `_catch_up_*` 各自的魔術數字 | 反映各資料源真實的補公布行為 |
| 前端三代路由並存 | §3.3 — 交接實作的範本 |
| `V6/scripts/` 20+ 支一次性腳本 | 事故的證據與可重跑的修復程序 |
| `config.py` 保持 dirty | 刻意的（V6.1 需要 56 維） |

---

# 9. Verification Gaps

完整內容見 `verification-gap-analysis.md`。摘要：

| Gap | 已肇事 | 已修 | **已鎖** |
|---|---|---|---|
| Parquet index 序列化 | ✅ 08-10 全系統停擺 | ⚠️ 只修 1/23 呼叫點 | ❌ |
| 線上/回測並列打破 | ✅ 08-08，0.18pp | ✅ | ❌ |
| null 誤差值方向 | ✅ 08-09 | ✅ | ❌ |
| 寫死比較基準 | ✅ 08-09 ×6 | ✅ | ❌ |
| 寫入端縮小語意 | ✅ 08-08，15% 資料 | ⚠️ 只覆蓋 2 個呼叫端 | ❌ |
| 非交易日寫入 | ✅ 06-07、06-19 | ⚠️ 部分 | ❌ |

**六類全部「已知、多數已修、零自動化」。**

---

# 10. Migration Risks（交接期最可能出問題的地方）

依「後果不可回復程度 × 發生機率」排序：

## 🔴 M-1：08-15/16 清試跑資料時只刪 state 沒刪 jsonl

CLAUDE.md 已有警告，但**沒有腳本保證兩者一起刪**。
`v62_state_*.json` 刪掉會重新建倉，`v62_portfolio_*.jsonl` 是 **append** 的 → `v62_performance` 會把 8 月試跑段與正式段接在一起算，**而且完全不會報錯**。

**後果不可回復**：混合紀錄無法事後拆分，除非從 git 歷史還原。

## 🔴 M-2：obsidian `現況整理` 過期，導致重複執行 `--first-day`

見 §7.3。新 Agent 讀到「週一手動跑第一次建倉」→ 執行 → **覆蓋 19 個狀態機**。
**這是本輪必須立刻修的一項。**

## 🟠 M-3：`git add -A` 覆蓋遠端 59 維 config

`config.py` 仍 dirty。紀律存在，機制不存在。

## 🟠 M-4：下一次資料修復引爆 index 撞名

`V6/scripts/` 有 20+ 支讀寫 parquet 的修復腳本，而 22 個寫入點會製造 stray 欄。
**08-08（製造）→ 08-10（引爆）的組合會重演。**

## 🟠 M-5：V6.1 退役時誤刪仍被 V6.2 依賴的東西

V6.2 已自給自足（`d38943e`），但**兩條鏈共用 `fetcher.py` 與 `feature_engineer.py`**。
退役 V6.1 時若順手清理「V6.1 專用」的資料源，可能斷掉 V6.2。

## 🟡 M-6：08-17 起跑後模型集合再變動

CLAUDE.md 已明確：晚加入的 arm 少了那段紀錄，**永遠無法公平並列**。

---

# 11. Cleanup Decisions（本輪裁決）

| 對象 | 決定 | 本輪是否執行 |
|---|---|---|
| `obsidian_note/` 現有 7 個分區結構 | **KEEP** — 成熟，沿用 | — |
| `01 系統現況/現況整理.md` | **UPDATE** — 危險的過期 | ✅ 已更新 |
| `🏠 Home.md` 的「現在最重要的事」 | **UPDATE** | ✅ 已更新 |
| 事故知識（散在三處） | **MERGE → 新的事故簿** | ✅ 已建立 `02 問題追蹤/事故簿.md` |
| 新舊交界知識 | **新增遷移地圖** | ✅ 已建立 `01 系統現況/遷移地圖.md` |
| Fincept 四份研究 | **PROMOTE（蒸餾，非複製）** | ✅ 已建立 `07 決策與教訓/外部架構參考 Fincept.md` |
| `CLAUDE.md` L486–1255（46%） | **MOVE → obsidian**，保留萃取後的規則 | ✅ 已壓縮 |
| `CLAUDE.md` 訊號系統整節 | **壓縮 + 指向 obsidian** | ✅ |
| `deploy/publisher.py` | **DELETE 候選**，證據充分 | ❌ 不執行（production code） |
| `V6_Master_Plan.md` / `V6_V61_Implementation_Guide.md` | **ARCHIVE 候選** | ❌ 不執行（等指令） |
| `OVERVIEW.md` / `PROJECT.md` / 父層 CLAUDE.md 的過期數字 | **UPDATE 候選** | ❌ 不執行（等指令） |
| `docs/gap-analysis.md` 命名混淆 | **DEFER**（需人工確認內容） | ❌ |
| `docs/` 的 5 份 draft/暫停計畫 | **ARCHIVE 候選** | ❌ |
| `archive/`、`not_using/`、`V6/scripts/` 修復腳本 | **KEEP** | — |
| 前端三代路由 | **KEEP**（範本） | — |

---

# 12. Final Memory Consistency Check

模擬：**今天加入一個全新的 Claude Code Builder，只有 `CLAUDE.md` + 相關 obsidian + task description。**

| # | 問題 | 本輪整理**前** | 本輪整理**後** |
|---|---|---|---|
| 1 | 知道 MarketMamba 是什麼？ | ✅ | ✅ |
| 2 | 知道目前 architecture？ | ⚠️ 知道，但混在 770 行敘事裡 | ✅ |
| 3 | 知道哪些不能碰？ | ✅（`models/`、Line Notify、`git add -A`） | ✅ |
| 4 | 知道過去有哪些重大事故？ | ⚠️ 知道**敘事**，但沒有可查詢的結構 | ✅ 事故簿 |
| 5 | 知道新舊版本交界在哪？ | ❌ **散落且部分過期** | ✅ 遷移地圖 |
| 6 | 知道現在正在做什麼？ | 🔴 **會被誤導**（現況整理說「週一上線、要跑 `--first-day`」，實際已跑） | ✅ |
| 7 | 知道如何驗證自己的工作？ | ⚠️ 知道紀律，但沒有可執行的東西 | ⚠️ **仍未解**（ratchet 尚未建立） |
| 8 | 會不會因不知道歷史決策而錯誤重構？ | ⚠️ 高風險：可能去「統一」4 套刻意不同的新鮮度門檻、或「抽象化」`fetcher.py` | ✅ §8.3「明確不是債」清單 |

**結論：第 7 項是唯一仍為否定的。** 依 §24 原則——**優先修 memory architecture 而不是加更多文件**——
memory 已修完，剩下的是**執行**（ratchet），需要使用者下一個指令。
