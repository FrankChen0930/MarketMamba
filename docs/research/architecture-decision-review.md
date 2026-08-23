# Architecture Decision Review — 三個候選項目的裁決

> 撰寫日期：2026-08-11
> 前置：`fincept-terminal-analysis.md` / `market-mamba-gap-analysis.md` / `future-architecture.md`
> 性質：**決策文件，不是實作計畫。** 本輪未修改任何 production code。
> 判準：每一項都必須指得出 **repository 中現在就存在的證據**。指不出來就 DEFER 或 REJECT。

---

## 摘要：三個裁決

| 候選 | 裁決 | 一句話理由 |
|---|---|---|
| **1. 數值回歸測試 + 三條 ratchet** | **ACCEPT** | 造成 2026-08-10 全系統停擺的缺陷，**只在 1 個呼叫點被修好，另外 ~18 個呼叫點今天仍然帶著它** |
| **2. ResultsStore** | **ACCEPT WITH LIMITS**（嚴格限縮） | 有真實的重複與死碼，但**沒有任何事故證據**。只接受四項小清理；storage abstraction 本身 **DEFER** |
| **3. SourcePolicy** | **ACCEPT WITH LIMITS**（拆成兩半） | 寫入守門有壓倒性證據 → 接受；**宣告式 freshness/fallback policy engine → DEFER**，因為散落的門檻**多數是刻意不同的，不是重複** |

**最重要的一句話**：本次審查推翻了前一輪研究的一個判斷。前一輪把 SourcePolicy 寫成「涵蓋 freshness / priority / fallback / validation 的宣告式政策層」。實際查證後發現**那些規則之所以散落，有一半是刻意的**（不同消費端本來就該有不同容忍度）。真正缺的不是 policy engine，是一張**沒有人維護的資料源登記表**。

---

# 1. Current evidence（現況證據）

以下全部是本次在 repository 中實際查到的，附檔案與行號。

## 1.1 「寫 parquet 忘了 `index=False`」的缺陷**只修好了 1/19**

2026-08-10 讓當天三次執行全部 0 秒失敗的根因，`fetcher.py:4276-4298` 已經修好，而且註解寫得非常完整（包含「為什麼平常不會出事」）。

**但那是一個呼叫點。** 本次掃描 `V6/` 底下所有 `to_parquet(`，在**現行執行路徑上**仍然缺少 `index=False` 的至少有：

| 檔案:行 | 寫什麼 | 索引是否可能不連續 |
|---|---|---|
| `fetcher.py:90` | `ticker_universe.parquet` | **是** — `_fetch_universe_from_finmind()` 有 `^\d{4}$` 過濾 |
| `fetcher.py:2171` | prices FinMind 快取 | 是（concat + 去重） |
| `fetcher.py:2269 / 2296 / 2315` | `run_full_data_sync` 的 prices / institutional / margin 快取 | 是 |
| `fetcher.py:4151` | `macro_raw` | 是 |
| `fetcher.py:4170 / 4186` | revenue / financials 快取 | 是 |
| `fetcher.py:4347 / 4385 / 4423 / 4464 / 4503` | 期貨 / 選擇權 / TRI / 股利 / 外資持股 快取 | 是 |
| `scripts/build_feature_matrix.py:160` | **特徵矩陣** | 是 |
| `notebooks/v6_colab_training.py:329` | **Colab 端特徵矩陣** | 是 |
| `scripts/backfill_institutional.py:99` | institutional 合併 | 是 |
| `scripts/fetch_v61_data.py:121 / 140 / 182`、`refetch_shareholding.py:129` | 各式快取 | 是 |

**這是本次審查最重要的發現。**

`ticker_universe.parquet` 特別值得指出：它是 `load_ticker_universe()` 的持久化快取，CLAUDE.md 已經記載過它一旦損壞會讓 `prices_raw` 長出數萬支非股票工具、`[Dataset init]` 顯示 46,488 stocks。**它現在的寫入路徑就是 `df.to_parquet(cache_path)`，而 df 剛剛經過正則過濾。**

而且缺陷的性質決定了它**不會被日常使用發現**——`fetcher.py:4288` 的註解自己寫得最清楚：

> 索引**連續**時 pandas 只寫 metadata、不落成欄，所以絕大多數日子都相安無事——只有 `drop_duplicates` 真的丟到列的那天才會出事。
> **「平常看起來沒事」不代表沒有這個缺陷。**

→ **同一個判斷，逐字適用於上面那 18 個呼叫點。** 這不是推測，是同一段程式碼註解對自己的診斷。

## 1.2 驗證紀律存在，但沒有一條會自己失敗

repo 內與「驗證」有關的資產：

| 資產 | 性質 | 會不會自動 fail |
|---|---|---|
| `scripts/test_feature_invariance.py`（17 KB） | 一次性腳本 | ❌ |
| `scripts/test_availability_flags.py`（14 KB） | 一次性腳本 | ❌ |
| `scripts/verify_quality_checklist.py`（12 KB） | 一次性腳本 | ❌ |
| `scripts/data_quality_check.py`、`verify_fixes.py` | 一次性腳本 | ❌ |
| `data/hygiene.py:check_data_health`（18 源、各設容許值） | **每日執行** | ❌ **明文 non-fatal** |
| `run_v62_daily.check_freshness()` | 每日執行 | ⚠️ 只發 Telegram 告警 |
| CLAUDE.md「與 git HEAD 逐位元相同」 | **文字紀律** | ❌ 靠人記得 |

- 沒有 `.github/`（**零 CI**）
- 沒有 pytest / conftest（`scripts/test_*.py` 是可執行腳本，不是測試套件）
- `agent-skills/` 底下的 `test_*.py` 是外部 skill pack 的，與本專案無關

→ **驗證的內容都寫過了，但沒有一個是「執行後會回非零 exit code 並擋下改動」的形式。**

## 1.3 Results / artifacts 現況（ResultsStore 審查用）

`V6/results/` 共 **66 個檔**（32 個 JSON + 17 個 `df_v62*.csv` + `df_kelly/df_traj/df_short/df_trend` + archive）。

**metadata 覆蓋情況（本次逐檔查）**：

| 類別 | 有無時間／版本資訊 |
|---|---|
| `v62_state_*.json`（19 份） | 有 `last_date`、`data_complete`、`spec`、`last_rebalance`、`n_rebalances` — **內容其實相當完整** |
| `v62_arms.json` / `v62_performance.json` | 有 `generated_at` |
| `pattern_signals` / `portfolio_exit_check` / `quant_market` | 有 `generated_at` + `date` |
| `action_signals` / `history_index` / `ic_analysis` / `market_summary` / `sim_backtest` | 只有 `date`（資料日，非產生時間） |
| `condition_analysis` / `dual_ic_analysis` | **完全沒有** |
| **`df_kelly.csv` 與 17 份 `df_v62*.csv`** | **只有資料列裡的 `Date` 欄，沒有任何 metadata**（欄位：`Ticker,Date,Exp_Alpha_5d,…,Suggested_Weight`） |

**發布路徑的重複與死碼**：

| 位置 | 狀態 |
|---|---|
| `marketmamba/deploy/publisher.py:97 push_to_github()` | **死碼**。寫到 `REPO_ROOT/results/`，該目錄**不存在**（本次確認 `Test-Path results` = False）；docstring 還寫「so the Streamlit frontend can auto-refresh」——Streamlit 早已不存在。唯一的 import 來源是 `marketmamba/deploy/__init__.py` 本身 |
| `run_daily_inference.py:1165 _push_to_github()` | 實際在用（V6.1） |
| `run_v62_daily.py:275 push_to_github()` | 實際在用（V6.2，含 3 次重試 + `_refresh_backend_cache()`） |

**後端讀取端的重複**：
`routers/v62.py` 的檔頭自承「**完全比照 `dual.py` / `signals.py` 的模式**」，四支 router 各有一份 `_cache` / `_cache_time` / `asyncio.Lock` / `httpx` 抓取。
URL 由 `v62.py:40` 的 `GITHUB_RESULTS_URL.replace("df_kelly.csv", name)` 組出來——**環境變數換一個檔名，整組靜默回 None**。

## 1.4 資料規則散落的實況（SourcePolicy 審查用）

`fetcher.py` 有 **11 個 `_catch_up_*` 函式**（`margin` / `daytrade` / `taifex` / `dividends` / `holdings` / `mops_quarterly` / `mops_revenue` / `trading_status` / `monthly` / `market_value` / `generic`），各自帶不同的 `max_days=15`、`max_months=6`、`max_quarters=6`、`lookback_days=120`。

**新鮮度判斷有 4 套獨立實作，容忍度各不相同**：

| 位置 | 容忍度 | 用途 |
|---|---|---|
| `run_v62_daily.py:62,77 check_freshness()` | **0 天** | 每日管線閘門 |
| `run_daily_inference.py:871-878` | **3 天**（`_stale_days > 3` 才警告） | 推論前提示 |
| `hygiene.py:290` `check_data_health` | **逐源 budget**（18 源各設一個） | 每日健檢 |
| `signals/scanner.py`（regime fallback） | **10 天** | macro 太舊時維持 NORMAL |

`run_v62_daily.py:27` 的註解明白寫出它為什麼另起爐灶：

> 既有健檢的容許值是「停更 5 天內算 ✓」，所以 margin 落後 1 天會顯示 ✓，**正是它漏掉真問題的原因**。

**→ 這是本次最重要的反面證據。** 這四套**不是重複實作，是四個刻意不同的政策**：管線閘門要嚴、健檢要寬（否則假警報會訓練人忽略警報，CLAUDE.md 已明文記載此教訓）。

另外 `feature_spec.py:51-54` 明確標示了一個已知邊界：

> 旗標描述「值是不是真的」，**不描述新鮮度**。新鮮度是另一個維度，若之後證實需要，應該另加欄位而不是改這個的語意。

**這代表使用者已經想過這件事，並且刻意沒做。** 任何 policy 提案都必須尊重這個既有決定。

---

# 2. Problems（問題陳述）

把上面的證據收斂成三個問題，並嚴格區分「已經發生」與「可能發生」。

| # | 問題 | 已發生？ | 證據 |
|---|---|---|---|
| **P-1** | **同一類缺陷修在一個呼叫點，其餘 18 個帶著同樣缺陷繼續運作，而且不會被日常使用發現** | ✅ 已發生一次（08-10 全系統停擺） | §1.1 |
| **P-2** | **所有驗收紀律都是「人要記得跑」的形式，缺陷的發現時間取決於運氣** | ✅ 反覆發生（08-08 清備份時偶然發現 15% 遺失；08-09 第一天跑才發現 n=1 的洞；08-08 寫績效工具時才發現線上/回測分歧） | §1.2 |
| **P-3** | **沒有一個地方寫著「每個資料源應該長什麼樣」**，於是每個消費端各自重新推論一次 | ⚠️ 尚未直接肇事，但已造成 `check_freshness` 另起爐灶 | §1.4 |
| **P-4** | 結果發布/讀取有死碼與四份重複快取邏輯 | ❌ **未肇事** | §1.3 |
| **P-5** | 主要輸出（`df_kelly.csv` + 17 份 `df_v62*.csv`）沒有產生時間與程式版本 | ❌ 未肇事（資料日可從 `Date` 欄與 `v62_state` 取得） | §1.3 |

**P-1 / P-2 是已發生且會再發生的。P-4 / P-5 是「看起來不整齊」。這個區分決定了三個裁決。**

---

# 3. Candidate solutions — 逐項回答 A–E

---

## 候選 1｜數值回歸測試 + 三條 verification ratchet

### A. 解決哪一個**目前已經存在**的問題？

**P-1 與 P-2，而且證據是可數的。**

不是「未來可能有 bug」。是：**`ticker_universe.parquet`、特徵矩陣、macro_raw、以及另外 15 個 parquet 寫入點，今天就帶著造成 08-10 停擺的同一個缺陷。** 它們只是還沒遇到觸發條件（`drop_duplicates` 真的丟到列）。

第二個具體證據：CLAUDE.md 的驗收標準「凡動 `feature_engineer.py` 都要跑逐位元回歸測試」——**這條規則存在，但沒有任何檔案實作它**。`scripts/test_feature_invariance.py` 是為某一次改動寫的一次性腳本，不是可重複執行的守門。

### B. 完全不做會怎樣？

| 期程 | 風險 | 可信度 |
|---|---|---|
| **Immediate（現在 ~ 08-17 起跑）** | 18 個寫入點中任何一個觸發 → **當日推論 0 秒失敗**，與 08-10 完全相同。若發生在 08-17 之後，前瞻紀錄會出現缺口，而 jsonl 是 append 的 → 缺口不可回填 | **高**。已發生過一次，機制完全相同 |
| **Medium（起跑後 1–3 個月）** | 動 `feature_engineer.py` / `portfolio_lab.py` 時靜默改變數值，而**前瞻紀錄的年化在 n<60 時本來就有 ±39pp 誤差** → **數值漂移會被誤讀成模型表現變化**，且分不出來 | **高**。這正是 CLAUDE.md 記載「線上與回測差 0.18pp 太容易被當成捨入誤差放過」的同型情境 |
| **Long（>6 個月）** | 累積一年的前瞻紀錄，若中途有過未被察覺的數值變更，**整段紀錄的可比性作廢**，而且無法事後判定是哪一天開始的 | 中—高 |

**最壞情況不是「有 bug」，是「前瞻紀錄被污染而且不知道從哪天開始」。** 對一個以「結論可不可信」為核心價值的系統，這是最貴的失敗模式。

### C. 最小可行版本

**不是測試框架。是三支腳本 + 兩個黃金檔。**

```
R1  ratchet：掃描 to_parquet( 缺 index=False 的呼叫點數量
      baseline = 今天的實際數字（先量，不猜）
      規則：只能降，不能升
      → 這一支就把 P-1 從「潛伏」變成「可見且不能惡化」

R2  golden：portfolio_lab.replay() 對已知窗 → 年化 37.28% / 換手 82.7%
R3  golden：v62_portfolio.step() replay → 與 R2 差 0.000pp
      → 這兩條鎖住 08-08 才修好的並列打破一致性

一個 check.sh / Makefile 把三者串起來，非零 exit code
```

**刻意不做的**：
- ❌ 不裝 CI（本機能跑就有 90% 價值）
- ❌ 不做特徵矩陣黃金檔（要 27 分鐘重建，成本太高）→ 改成**小樣本**：固定 20 天 × 50 檔的雜湊，跑得動才有人跑
- ❌ 不追覆蓋率
- ❌ 不碰 `fetcher.py`

**R1 是最小中的最小。** 如果只做一件事，做 R1——它是唯一能防止 08-10 重演的東西，而且**不需要動任何 production code**。

### D. 需要新 abstraction 嗎？

**不需要。**

R1 是一支獨立腳本 + 一個 JSON baseline。R2/R3 是呼叫既有函式比對既有數字。**沒有新的類別、沒有新的介面、沒有任何 production 模組被修改。**

（明確拒絕：不要引入 pytest fixture 體系、不要建 `tests/` 的分層結構、不要為了「可測試性」重構 `portfolio_lab`。）

### E. 如何被 automated verification 保護？

**它本身就是那個保護。** 但要回答「誰保護保護者」：

- R1 的 baseline 必須**先量後鎖**（不能憑印象填），且 baseline 檔改動必須是獨立 commit，message 要寫「為什麼這個數字該變」——借用 Fincept arch-ratchet 的紀律：**baseline 只能往下調**
- R2/R3 的黃金值會在**合法的正確性修正**時變動。緩解同上：獨立 commit + 理由
- **R1 自己要有一個 self-test**：CLAUDE.md 已經記載過「我的第一版檢查腳本有 bug，大寫 `_BACKUP.parquet` 沒被 regex 匹配，變成拿檔案跟自己比而回報可刪」。→ **R1 必須附一個「故意寫壞的樣本檔」，確認它抓得到。**

---

## 候選 2｜ResultsStore（**特別審查**）

### 2.1 現有 results / artifacts / logs / checkpoints 狀況

| 類別 | 位置 | 進 git | metadata |
|---|---|---|---|
| 每日結果 | `V6/results/`（66 檔） | ✅ | 部分（見 §1.3） |
| 歸檔 | `V6/results/archive/`、每日目錄（90 天滾動） | ✅ | 檔名帶日期 |
| 逐日持股紀錄 | `v62_portfolio_{arm}.jsonl`（19 份，append） | ✅ | 每列自帶 `weights`（08-08 刻意加的） |
| 狀態機 | `v62_state_{arm}.json`（19 份） | ✅ | **`last_date` / `data_complete` / `spec` / `n_rebalances` 都有** |
| raw 面板 | `Data/processed_v6/*.parquet` | ❌ 本機 | parquet schema |
| 衍生快取 | `Data/baseline_cache*`（約 24 GB） | ❌ | — |
| checkpoint | `V6/models/`（保護）、`V6/checkpoints/`（gitignore） | ❌ | 訓練 JSON 另存 |
| log | `V6/logs/*.log`、`model_tracker.jsonl` | 部分 | — |

### 2.2 哪些資訊目前**無法**被可靠追蹤

誠實列舉，並標明它現在有沒有造成問題：

| 追蹤不到的東西 | 現在有沒有造成問題 |
|---|---|
| **產生時間（wall clock）**——只有資料日 | ❌ 沒有。日頻系統裡「資料日」比「產生時間」更重要，而資料日到處都有 |
| **產生它的程式 commit** | ⚠️ 間接可得：結果與程式在**同一個 git repo、同一個 commit** 被 push。**這是「git 當匯流排」的意外紅利** |
| **`df_kelly.csv` / `df_v62*.csv` 的列數與完整性** | ⚠️ `v62_state` 有 `data_complete`，V6.1 那條線沒有 |
| **哪個 checkpoint 產生了哪份分數** | ⚠️ `v62_arms.json` 的 `spec` 有，V6.1 沒有 |
| **實驗 arm 的可比性資訊**（purge 天數、panel 版本、seed） | ❌ **這是真的缺口**，目前只存在 obsidian 的人類文字裡；08-09 曾因此標錯 tier |

### 2.3 真的需要一個 ResultsStore 嗎？

**不需要。至少現在不需要，而且我要推翻上一輪自己的 P0-3 排序。**

三個理由：

1. **P-4 / P-5 沒有肇事紀錄。** 這是本審查最硬的判準。相較之下 P-1 造成過全系統停擺、P-2 造成過反覆的延遲發現。**把工程時間投在沒出過事的整潔問題上，是在錯誤的地方付費。**
2. **git 已經在做 ResultsStore 該做的一半事。** 版本、時間、與程式碼的對應關係、可回溯——全部免費且已經在運作。引入一個 store 抽象會**削弱**這個性質（多一層之後，「這份結果對應哪個 commit」反而變得間接）。
3. **`v62_state_*.json` 已經是一個設計良好的 artifact**：`last_date` + `data_complete` + `spec` + `n_rebalances`。**要學的模式已經在 repo 裡了**，缺的是把它套到 V6.1 那條線——而 V6.1 正在退役。**為一條正在退役的線做基礎建設是純虧損。**

### 2.4 如果要做，最小應該保存什麼

假設未來真的需要（觸發條件見 2.6），最小集合是**三個欄位**，不是一個 store：

```
每個發布的結果檔旁邊，一份 results_manifest.json：
  { "df_kelly.csv": { "data_date": "2026-08-11", "rows": 1925, "complete": true }, ... }
```

`generated_at` **刻意不放**——它會讓每次 push 都產生 diff（即使內容沒變），污染 git 歷史，而它解決的問題（「這是不是今天的」）`data_date` 已經回答了。

### 2.5 能不能用現有的 filesystem / JSON 完成？

**能，而且已經完成了 80%。**

- `v62_arms.json` 已經是 manifest 模式（且註解裡寫明了為什麼不能在後端自帶一份表）
- `v62_state_*.json` 已經帶了完整性資訊
- git 已經提供版本與時間

**不需要 SQLite。** 引入 SQLite 會同時失去 git diff 可讀性與 GitHub raw 直讀能力——那是現行架構最有價值的兩個性質。

### 2.6 什麼情況下才值得升級成真正的 storage abstraction

三個觸發條件，**任一成立才重新評估**：

1. **git repo 因結果檔膨脹到 clone / push 不可接受**（例如 >2 GB，或每日 push 超過 1 分鐘）
2. **出現第二個消費端且不能用 GitHub raw**（例如手機 App、或需要查詢而非整檔讀取）
3. **真的發生一次「因為追不到 provenance 而做出錯誤判讀」的事件**——注意 08-09 的 tier 標錯**已經很接近**這一條，但當時是人工標記錯誤，不是追不到資訊

### 2.7 A–E 快答

- **A**：解決 P-4/P-5，**而這兩個沒有肇事證據**
- **B**：immediate = 幾乎為零；medium = router 增加時重複再抄一份（成本線性、不加速）；long = 若消費端變多會變成真問題
- **C**：見下方 ACCEPT WITH LIMITS 的四項
- **D**：**不需要新 abstraction。** 現有 filesystem + JSON + git 足夠
- **E**：四項小清理各自的保護見 §7

---

## 候選 3｜SourcePolicy / data-fetching contract（**特別審查**）

### 3.1 現在哪些規則散落在哪裡（要求至少 3 個實例，這裡給 5 個）

**實例 1：新鮮度（4 套實作、4 種容忍度）**
`run_v62_daily.check_freshness`（0 天）／`run_daily_inference.py:873`（3 天）／`hygiene.py:290`（逐源 budget，18 源）／`scanner.py` macro fallback（10 天）。
**判定：一半是刻意的。** `run_v62_daily.py:27` 明講它另起爐灶是因為既有健檢太寬會漏掉真問題。**這不是重複，是不同用途的不同政策。**

**實例 2：補抓視窗（11 個 `_catch_up_*`，各自的魔術數字）**
`max_days=15`（margin / daytrade / market_value）／`max_months=6`（taifex / mops_revenue）／`max_quarters=6`（mops_quarterly）／`lookback_days=120`（trading_status）。
**判定：這些數字反映的是「各資料源多久會補公布」，是真實世界的性質。** 分散的問題不是值不同，是**沒有一個地方列出這 11 個值讓人一眼看到全貌**。

**實例 3：來源優先序 / fallback（各自寫死）**
`fetch_margin_direct()`（TWSE + TPEX 合併）／prices（yfinance + 交易所直連）／`scanner` 的 regime 判斷（prices_raw 的 TWII 找不到 → fallback `macro_raw.TWII_Close` + 10 天新鮮度檢查）。
**判定：真的散落。** 特別是 scanner 那條，它是**業務層在做資料層的 fallback 決策**。

**實例 4：欄位可得性（`feature_spec.py` 的旗標）**
已有制度化的 availability flags，**但檔案自己標明「不描述新鮮度」**（`feature_spec.py:51`）。
**判定：這是刻意畫的界線，不是缺陷。** 任何提案不得越過它。

**實例 5：寫入語意（`_append_to_parquet` vs 其餘 18 個 `to_parquet`）**
一個呼叫點有完整守門（merge 語意、stray 欄丟棄、去重、`index=False`），其餘沒有。
**判定：這是唯一有壓倒性證據、且純粹是缺陷的一項。**

### 3.2 「如果只建立一個最小的 declarative policy layer，它應該長什麼樣子？」

**先講不該長什麼樣子。**

❌ **不要 policy engine。** Fincept 的 `TopicPolicy` 有一個 scheduler 去執行它；MarketMamba 沒有排程器、沒有訂閱者、沒有 in-flight 狀態，日頻批次的執行順序是寫死在 `run_daily_update()` 裡的。**照抄一個沒有執行者的政策層，等於憑空造一個新的抽象層。**

✅ **應該長成一張「登記表」，而不是一個「引擎」。**

```
概念（不是要實作的 code，只表達形狀）

SOURCES = {
  "prices_raw": {
     path, key_cols=("Date","stock_id"),
     cadence="daily",  publish_hour=14,      # 交易所何時公布
     catch_up_days=15,                        # 目前散在 _catch_up_* 的那個數字
     dtypes={...},                            # 寫入前守門用
     allow_shrink=False,                      # 08-08 的教訓
  },
  "margin_raw": { ..., publish_hour=21, ... },   # 08-05 排程 19:30→21:30 的理由
  ... 共 14~18 個源
}
```

**三條設計約束**：

1. **它只是資料，沒有行為。** 誰要用、用多嚴的門檻，由消費端自己決定。
   → `run_v62_daily` 繼續用 0 天、`hygiene` 繼續用各自 budget，**兩者都從 SOURCES 讀「這個源應該多久更新一次」，但各自套自己的容忍度**。
   這樣既解決了「沒有全貌」，又**不強行統一那些刻意不同的政策**。
2. **它從已存在的常數收斂而來，不是新發明。** 那 11 個 `_catch_up_*` 的參數、`hygiene` 的 18 個 budget、`publish_hour`（08-05 稽核已經量過十個源的抵達時間）——**全都已經存在，只是沒有集中。**
3. **第一版是唯讀的。** 先讓 ratchet 與健檢**消費**它，不要讓 `fetcher.py` 依賴它。等它被證明準確，再逐源接上寫入端。

### 3.3 A–E

**A. 解決哪一個已存在的問題？**
拆成兩半，證據強度完全不同：

- **3-a 寫入守門**：解決 P-1。證據 = §1.1 的 18 個呼叫點 + 08-08 的 15% 遺失 + 08-10 的停擺。**壓倒性。**
- **3-b 登記表**：解決 P-3。證據 = `run_v62_daily.py:27` 明文記載它必須另起爐灶，因為看不到既有健檢的容忍度。**中等**——它記錄的是一次「差點漏掉」，不是一次事故。

**B. 不做會怎樣？**

| | 3-a 寫入守門 | 3-b 登記表 |
|---|---|---|
| Immediate | **18 個呼叫點任一觸發即當日推論失敗** | 幾乎無風險 |
| Medium | 靜默資料遺失（08-08 那類）在**沒走 `_append_to_parquet` 的源**上重演；而 `prices_raw` 之外的源**從未被查過** | 新增消費端時再推論一次；容忍度繼續各自漂移 |
| Long | 面板長期缺列 → 所有回測結論的基礎被侵蝕，且**跨期不可比** | 排程時間與交易所公布時間脫鉤（08-05 已發生過一次，19:35 抓 margin 抓到「尚未公布」） |

**C. 最小可行版本**

- **3-a**：**一個函式**，不是一個層。把 `_append_to_parquet` 已經寫好的守門抽成 `write_table(path, df, *, key_cols, allow_shrink=False)`，內容 = 型別/欄名守門 + stray 欄丟棄 + 去重 + `index=False` + 暫存→驗過→`os.replace`。
  **然後一次遷一個呼叫點**，每遷一個跑鍵集合比對（舊 ⊆ 新、既有列 `max|Δ|=0`）——這正是 08-08 回補用過的驗收方法。
  **第一個要遷的是 `ticker_universe`**（風險最高、影響最大、CLAUDE.md 已記載過它壞掉的後果）。
- **3-b**：**一個 dict**，唯讀，先只被 R1 ratchet 與 `hygiene` 消費。

**D. 需要新 abstraction 嗎？**

- **3-a：不需要新 abstraction，只需要「把已有的正確實作變成唯一入口」。** 這是收斂，不是新增。
- **3-b：需要一個新的**東西**，但它是一個 dict，不是一個抽象。** 沒有類別、沒有介面、沒有 protocol、沒有執行者。
  **明確拒絕**：不要 `Source` 基類、不要 `Producer` 介面、不要註冊機制、不要 policy 繼承。

**E. 如何被 automated verification 保護？**

- **R1 ratchet 就是 3-a 的守護者**：「缺 `index=False` 的 `to_parquet` 呼叫點數量只能降」→ 遷移進度自動被量測，而且**回頭加新的違規會 fail**
- 新增一條 **R4 ratchet**：「不經 `write_table` 而直接對 `Data/processed_v6/*.parquet` 寫入的呼叫點數量只能降」
- 3-b 的保護：一條檢查「`SOURCES` 列出的每個源，其 parquet 的最大日期落後不得超過該源宣告的 cadence × 2」——**這是把 `hygiene` 現有的邏輯改成從登記表取參數**，不是新邏輯

---

# 4. Trade-offs

| 決策 | 得到 | 付出 | 被放棄的替代方案 |
|---|---|---|---|
| **只做 ratchet 不做 CI** | 立即可用、零基礎建設 | 靠人記得跑（**這正是要解決的問題**） | 緩解：把 `check.sh` 寫進 CLAUDE.md「改程式之前」紀律；Phase 2 再上 CI |
| **只鎖 3 個數字不做全面回歸** | 成本可控、跑得快所以有人跑 | 覆蓋率低 | 判準：只鎖「已經出過事」與「數字就是產品」的地方 |
| **ResultsStore 只做四項清理** | 不在沒出事的地方付費 | 第 5 支 router 出現時仍會抄一次 | 接受。**線性成本 < 錯誤抽象的成本** |
| **保留 git 當匯流排、不引入 DB** | 免費版本控制 + 稽核 + GitHub raw 直讀 | 結果檔會讓 repo 長大 | 觸發條件已寫在 §2.6 |
| **SourcePolicy 只做 write_table + dict** | 直擊已肇事的部分 | freshness/fallback 仍散落 | **這是刻意的**——證據顯示它們多數是刻意不同的政策 |
| **不統一 4 套新鮮度門檻** | 保留「閘門要嚴、健檢要寬」的正確設計 | 沒有單一入口 | 由登記表提供**共同的參數來源**，容忍度仍各自決定 |
| **逐呼叫點遷移而非一次改完** | 每步可驗證、可回退 | 遷移期會有兩種寫法並存 | 由 R1 ratchet 讓進度可見 |

**一個必須點名的風險**：R2/R3 黃金檔會在合法的正確性修正時變動。如果紀律鬆掉（隨手更新黃金值），**這套東西會退化成一個假的安全感，比沒有更糟**。緩解已寫在 §7。

---

# 5. Recommended decisions

## 5.1 三個裁決

### 候選 1｜數值回歸測試 + 三條 ratchet → **ACCEPT**

**範圍**：R1（`index=False` ratchet）+ R2/R3（`replay` vs `step` 黃金值）+ 一支 `check.sh`。
**不含**：CI、pytest 體系、特徵矩陣全量黃金檔、覆蓋率目標。
**理由**：唯一同時滿足「已肇事」「今天仍存在」「不動 production code」三個條件的項目。

### 候選 2｜ResultsStore → **ACCEPT WITH LIMITS**（限縮到不再是 architecture）

**接受的只有四項，全部是清理不是建設**：

| # | 動作 | 理由 |
|---|---|---|
| C-1 | **刪除 `marketmamba/deploy/publisher.py`** 與 `deploy/__init__.py` 的 import | 死碼：目標目錄不存在、提到已消失的 Streamlit、被兩個實際實作取代 |
| C-2 | `v62.py:40` 的 URL 改成 base + 檔名，**不用 `.replace()`** | 三行改動，消除一個靜默失效點 |
| C-3 | 後端四支 router 的「抓取 + TTL + lock」抽成**一個 helper 函式**（不是 store、不是類別） | 已有四份抄寫，且 `v62.py` 檔頭自承是抄的 |
| C-4 | V6.1 那條線的主要輸出補 `data_date` + `rows`（**照抄 `v62_state` 已有的做法**） | 只在 V6.1 尚未退役且成本 <30 分鐘時才做；**若 V6.1 即將退役則跳過** |

**明確 DEFER 的**：`ResultsStore` 抽象、provenance API、SQLite、`generated_at`、實驗註冊表。

### 候選 3｜SourcePolicy → **ACCEPT WITH LIMITS**（拆成兩半，一半接受一半延後）

| 子項 | 裁決 | 範圍 |
|---|---|---|
| **3-a `write_table()` 寫入守門** | **ACCEPT** | 一個函式 + 逐呼叫點遷移，`ticker_universe` 優先 |
| **3-b `SOURCES` 登記表（唯讀 dict）** | **ACCEPT WITH LIMITS** | 只收斂既有常數；**只被 ratchet 與 hygiene 消費**；不接寫入端 |
| **3-c 宣告式 freshness policy 層** | **DEFER** | 證據顯示四套門檻多數是刻意不同的 |
| **3-d 來源優先序 / fallback 宣告化** | **DEFER** | 只有 scanner 那一處是真問題，而 scanner 正在退役 |
| **3-e normalization / validation 抽象** | **DEFER** | 見 §6 |

---

# 6. Deferred decisions

明確列出**延後**的項目、延後的理由、以及**重新評估的觸發條件**——沒有觸發條件的延後等於偷偷放棄。

| 項目 | 為什麼延後 | 重新評估的觸發條件 |
|---|---|---|
| **ResultsStore 抽象 / provenance API** | P-4/P-5 無肇事證據；git 已提供一半 | §2.6 三條任一成立 |
| **SQLite 儲存** | 會失去 git diff 與 GitHub raw 直讀 | repo >2 GB，或出現不能用 raw 的消費端 |
| **宣告式 freshness policy 層** | 四套門檻多數是刻意的 | 出現**第 5 個**新鮮度實作，或發生一次「兩套門檻互相矛盾導致誤判」 |
| **`parse_*` 純函式抽取 + 樣本測試** | 前一輪列為 P0-1 的一部分，本次**降級**：它要動 `fetcher.py` 的讀取路徑，而讀取端沒有肇事紀錄（肇事的是寫入端） | 3-a 遷移完成之後；或某個交易所改版面導致靜默解析錯誤 |
| **實驗註冊表（purge/panel/seed 可比性）** | 是真缺口，但目前唯一的消費者是人 | 08-17 起跑後，若再發生一次 tier 標錯或不可比並列 |
| **CI（GitHub Actions）** | 本機 `check.sh` 先證明有人會跑 | `check.sh` 連續一個月被實際使用；或出現第二個開發者 |
| **前端新鮮度指示 / formatting 層** | 屬呈現層，不阻塞資料正確性 | C-4 完成之後 |
| **回測引擎退役（sim_engine v1/v2 等）** | 必須等 V6.1 確定停用 | V6.1 排程移除後 |

---

# 7. Verification strategy

**核心問題：未來怎麼讓同類型 regression 自動 fail，而不是靠人發現？**

## 7.1 四條 ratchet（只能降）

| ID | 量什麼 | 防哪一次事故 | baseline |
|---|---|---|---|
| **R1** | `to_parquet(` 呼叫點中缺 `index=False` 的數量 | 2026-08-10 全系統停擺 | **先量後鎖**（預估 ~18） |
| **R4** | 直接寫 `Data/processed_v6/*.parquet` 而未經 `write_table` 的呼叫點數 | 2026-08-08 的 15% 靜默遺失 | 遷移期用來量進度 |
| **R5** | 非測試檔中出現的硬編比較基準字面值（`0.38` / `38.0%` / `0.1145` 之類） | 2026-08-09 一天踩六個 | 先量後鎖 |
| **R6** | 誤差／不確定性欄位上套 `?? 0` / `or 0` / `if x` 的數量 | 2026-08-09 `ann_stderr_pp` 的 null→0 讓 19 個 arm 全被誤標 | 應為 0，直接鎖 |

## 7.2 兩條黃金值（精確比對）

| ID | 比對 | 防哪一次事故 |
|---|---|---|
| **R2** | `portfolio_lab.replay()` 對已知窗 → 年化 37.28% / 換手 82.7% | 組合層口徑漂移 |
| **R3** | `v62_portfolio.step()` replay vs R2 → **0.000pp** | 2026-08-08 並列打破不一致（0.18pp，當時「太容易被當成捨入誤差放過」） |

## 7.3 三條紀律（人的部分，寫進 CLAUDE.md）

1. **baseline 只能往下調。** 任何上調都必須是獨立 commit + 寫明為什麼那個數字該變（Fincept arch-ratchet 的規則）
2. **黃金值變動 = 正確性修正的一部分**，必須與該次修正同一個 commit，且 commit message 要說明數值差多少、為什麼
3. **檢查腳本本身要有 self-test**——CLAUDE.md 已記載過「檢查腳本本身也要驗」（大寫 `_BACKUP` 未被 regex 匹配那次）。R1/R4/R5/R6 各附一個**故意寫壞的樣本**，確認抓得到

## 7.4 遷移期的逐步驗收（3-a 專用）

每遷一個 `to_parquet` 呼叫點：

```
舊檔鍵集合 ⊆ 新檔鍵集合        （不可縮小）
既有列 max|Δ| = 0.000e+00      （值不可變）
schema 型別逐欄相同             （08-08 的教訓：守門要在寫入之前）
```

**這三條不是新發明**——它們就是 08-08 回補 `prices_raw` 與 08-09 重評分時實際用過的驗收，只是變成每次遷移都跑一次。

## 7.5 明確不做的驗證

- ❌ 覆蓋率門檻
- ❌ 對 `fetcher.py` 的網路抓取做 mock 測試（維護成本高於價值）
- ❌ 前端測試
- ❌ 模型輸出的統計檢定自動化（那屬於研究紀律，見 gap analysis P1-5，本輪不裁決）

---

# 8. Proposed implementation order

**原則：每一步都可以停下來，且停下來時系統是完整的。**

```
Step 0 ── 量測（不改任何東西）
   跑一次 R1/R5/R6 的掃描，把真實數字記下來當 baseline
   ⚠️ 這一步的產出同時是「SourcePolicy 該從哪個呼叫點開始遷」的依據
   停在這裡的價值：已經知道 18 個潛在炸彈在哪

Step 1 ── R1 ratchet + check.sh                          【候選 1，最小核心】
   零 production 改動。防止 08-10 重演的最短路徑
   停在這裡的價值：缺陷不會再惡化

Step 2 ── R6（應為 0，直接鎖）+ R5（先量後鎖）
   同樣零 production 改動
   停在這裡的價值：三類已知踩過的坑都被鎖住

Step 3 ── R2 / R3 黃金值                                  【候選 1，完成】
   需要跑一次 replay 取得基準；不改 production
   停在這裡的價值：組合層口徑被鎖死，可以放心改研究程式

──────── 以上全部不動 production code，可在 08-17 起跑前完成 ────────

Step 4 ── C-1 刪 publisher.py + C-2 修 URL               【候選 2，低風險】
   純刪除 + 三行修改。建議在 V6.2 觀察期之外的空檔做

Step 5 ── SOURCES 登記表（唯讀 dict）                     【候選 3-b】
   從 11 個 _catch_up_* 與 hygiene 的 18 個 budget 收斂
   只被 ratchet / hygiene 消費，不接 fetcher

Step 6 ── write_table() + 逐呼叫點遷移                    【候選 3-a，最高風險】
   ⚠️ 必須在 Step 1-3 完成之後
   ⚠️ 必須在 08-17 正式起跑之後（不要卡在起跑前）
   順序：ticker_universe → 特徵矩陣 → macro/revenue/financials → 其餘
   每遷一個跑 §7.4 三條驗收

Step 7 ── C-3 後端 fetch helper                           【候選 2】
   v62 先遷 → 觀察一週 → signals 最後

Step 8 ── C-4（僅在 V6.1 尚未退役時才做）
```

**與 08-17 的關係**：Step 0–3 可以、也應該在 08-17 之前做完（它們不碰執行路徑）。Step 4 之後全部排在起跑之後。

---

# 9. 裁決總表

| 候選 | 裁決 | 範圍界線 |
|---|---|---|
| **1. 數值回歸測試 + 三條 ratchet** | 🟢 **ACCEPT** | R1/R2/R3（+R5/R6）+ `check.sh`。**不含** CI、pytest 體系、覆蓋率、特徵矩陣全量黃金檔 |
| **2. ResultsStore** | 🟡 **ACCEPT WITH LIMITS** | 只接受 C-1~C-4 四項清理。**ResultsStore 抽象本身 DEFER**（觸發條件見 §2.6） |
| **3-a. `write_table()` 寫入守門** | 🟢 **ACCEPT** | 一個函式 + 逐點遷移。**不含** Source 基類 / Producer 介面 / 註冊機制 |
| **3-b. `SOURCES` 登記表** | 🟡 **ACCEPT WITH LIMITS** | 唯讀 dict，只收斂既有常數，只被 ratchet/hygiene 消費 |
| **3-c. 宣告式 freshness policy 層** | 🔵 **DEFER** | 四套門檻多數是刻意不同的政策，不是重複 |
| **3-d. 來源優先序 / fallback 宣告化** | 🔵 **DEFER** | 唯一的真問題在 scanner，而 scanner 正在退役 |
| **3-e. normalization / validation 抽象** | 🔴 **REJECT**（現階段） | `feature_spec.py:51` 已刻意畫界；讀取端無肇事紀錄；Fincept 自己也只在自訂資料源那條路徑用，核心 fetcher 沒走它 |
| **ResultsStore 的 provenance / SQLite / 實驗註冊表** | 🔵 **DEFER** | 各有觸發條件，見 §6 |

---

## 附：本次審查推翻了前一輪研究的兩個判斷

誠實記錄，不偷改：

1. **前一輪把 ResultsStore 列為 P0-3（「成本最低、風險最低」）。本次降級為 ACCEPT WITH LIMITS 並把抽象本身 DEFER。**
   原因：前一輪看的是「有四份重複」這個結構性事實，**沒有查它有沒有肇事**。查完之後發現沒有，而且 `v62_state_*.json` 已經帶了該有的 metadata。**「重複」不等於「問題」。**

2. **前一輪把 SourcePolicy 描述成涵蓋 freshness / staleness / availability 的宣告式政策層。本次拆開，只接受寫入守門與登記表。**
   原因：實際查證發現 `run_v62_daily.py:27` 明文記載它**刻意**用比健檢更嚴的門檻，而理由是對的。**把刻意不同的政策統一，會破壞一個已經正確的設計。**

兩次修正的共同教訓，值得記進方法論筆記：

> **「看起來重複」與「已經造成問題」是兩件事。架構審查的第一個動作應該是查肇事紀錄，不是查結構。**
