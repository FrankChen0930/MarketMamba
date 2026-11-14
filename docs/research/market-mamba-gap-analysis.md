# MarketMamba vs FinceptTerminal — 差距分析與優先級

> 撰寫日期：2026-08-11
> 前置文件：`docs/research/fincept-terminal-analysis.md`（Fincept 架構分析，含證據等級標示）
> 本文件的立場：**Fincept 是參考實作，不是目標。** 凡是不適合 MarketMamba 的，直接說不適合，不勉強套用。

---

## Part A — MarketMamba 現況架構地圖（Phase 1）

以下全部來自 repo 內既有文件與程式碼，沒有臆測。資訊不足處明確標「未知」。

### A.1 專案目的

**一人一市場的量化投資自動化系統。** 每日收盤後對台股全市場（約 1,900–2,500 支）做深度學習推論，輸出 Alpha 排名，經雲端 dashboard 呈現。

**真正的使用者只有兩類**：使用者本人（做決策）、家人（看 dashboard）。這決定了後面所有取捨。

### A.2 系統邊界（三個 repo、五個執行環境）

```
MarketMamba/V6 (WSL2 + RTX 3060)  ── 每日 21:30 / 22:15 兩條鏈
      │  git push
      ▼
GitHub (V6/results/*.csv|json)     ── 事實上的「資料交換匯流排」
      │  GitHub raw + 1h TTL
      ▼
Render (FastAPI, rootDir=app/backend)
      ├──→ Vercel (MarketMamba frontend, React)
      └──→ PersonalOS (Electron, 獨立 repo、頁面鏡像)

Colab A100                        ── 訓練（手動觸發）
Windows Task Scheduler            ── 排程觸發點
```

**關鍵觀察：GitHub repo 本身被當成訊息佇列在用。** 這不是缺點——對單人、日頻、免費方案的組合來說是相當好的取捨（免運維、天生有版本、天生可稽核）。但它是一個**沒有被明確命名的架構決定**，也因此沒有對應的契約（見 Gap 3）。

### A.3 分層現況

| 層 | 現況 | 主要檔案 |
|---|---|---|
| **資料抓取** | 14 個直連源（TWSE / TPEX / TAIFEX / TDCC / MOPS）+ FinMind + yfinance | `data/fetcher.py`（**215 KB / ~100 個 `fetch_*` 函式，平坦模組**） |
| **資料衛生** | 去重、型別、除權息還原 | `data/hygiene.py`、`scripts/apply_ex_rights_adjustment.py` |
| **特徵工程** | 56 / 59 / 66 維三套協定並存 | `data/feature_engineer.py`（94 KB）、`data/feature_spec.py` |
| **模型** | Multi-Scale Mamba + GATv2，~1.66M 參數 | `models/architecture.py`（**受保護，禁改**） |
| **訓練** | Colab A100，手動 | `notebooks/v6_colab_training.py`（46 KB） |
| **推論** | 三條鏈並行 | `run_daily_inference.py`(66 KB) / `run_dual_inference.py` / `run_v62_inference.py` |
| **組合建構** | 19 組合 × 11 份分數，狀態機 | `v62_portfolio.py`(37 KB)、`experimental/portfolio_lab.py`(53 KB) |
| **前瞻績效** | 逐日 jsonl → 年化 + 誤差棒 | `v62_performance.py` |
| **訊號（退役中）** | scanner 加權評分 + 四層退場 + 型態 | `signals/scanner.py`、`signal_conditions.py`、`quant/pattern_scanner.py` |
| **回測** | 多套並存 | `backtest/engine.py`、`sim_engine_v{1,2,3}.py`、`experimental/portfolio_lab.py` |
| **LLM** | 每日市場報告 | `llm/report_generator.py` |
| **API** | FastAPI，10 個 router | `app/backend/routers/*.py` |
| **前端** | Vite + React 19 + recharts + klinecharts | `app/frontend/src/pages/*.jsx`（25 支） |
| **儲存** | **檔案系統**：parquet（本機 Data/）+ CSV/JSON（V6/results/，進 git） | 無資料庫 |
| **測試** | **無 pytest、無 CI**。驗證靠一次性腳本 | `scripts/test_feature_invariance.py` 等 5 支「驗證腳本」 |
| **文件** | **三層記憶制度**，品質很高 | CLAUDE.md（40 KB）/ obsidian_note（14 篇）/ docs（25 份）/ 檢查表（11 份）/ planing（9 份） |

### A.4 目前正在開發的東西

- **V6.2 上線中**：08-09 已用 8/6 資料整條跑通並推上線；**08-15/16 清試跑資料、08-17 正式起跑**（模型集合定案日）
- **V6.1 + 雙模型退役排程中**：等 V6.2 連續跑順幾天
- **scanner 訊號系統退役中**：由組合建構層接班

### A.5 已知技術債（CLAUDE.md 自列 + 本次觀察補充）

**CLAUDE.md 已列的**：
`trading_status_raw` 未進每日流程 / GBDT 誤差棒沒量過 / B 類面板每天重建 4 分鐘 / 約 24 GB 衍生快取待決 / 2026-05-25 少 353 支未歸因 / `INPUT_DIM` 59→47 規格清理 / 56 vs 59 維 config 分裂

**本次觀察補充的**（CLAUDE.md 未列）：

| # | 技術債 | 證據 |
|---|---|---|
| D-a | **零自動化測試、零 CI** | repo 無 `.github/`、無 pytest、無 conftest。`scripts/test_*.py` 是一次性腳本不是測試套件 |
| D-b | **後端 4 個 router 各自抄一份「GitHub raw + 1h TTL + asyncio.Lock」** | `signals.py` / `dual.py` / `v62.py`（註解自承「完全比照 dual.py / signals.py 的模式」）+ 其餘 |
| D-c | **URL 用字串替換組出來** | `v62.py:40`：`GITHUB_RESULTS_URL.replace("df_kelly.csv", name)` — 環境變數換一個檔名整組壞掉，且不報錯 |
| D-d | **`fetcher.py` 215 KB 平坦模組**，沒有 source 抽象 | 100 個 `fetch_*` 函式 + 20 幾個 `_catch_up_*`，各自 parse、各自重試、各自判「今天有沒有」 |
| D-e | **回測/模擬引擎四套並存**（`engine` / `sim_engine` v1,v2,v3 / `portfolio_lab`） | 08-08 才發現 `step()` 與 `replay()` 的並列打破不一致 → 582 天差 0.18pp |
| D-f | **`mock_data.py` 仍在 production backend 裡** | `app/backend/mock_data.py`，`/api/portfolio` 據 CLAUDE.md 仍是 mock |
| D-g | 前端 `InvestmentSim.jsx` 57 KB、`QuantAnalysis.jsx` 43 KB 單檔 | 與 Fincept「5 個 >1,000 LOC 畫面」同型問題 |

### A.6 MarketMamba 明顯**領先** Fincept 的地方（必須先講，才不會把研究做成單向自卑）

| 面向 | MarketMamba | Fincept |
|---|---|---|
| **研究方法論紀律** | 雜訊底線（±6pp / ±2.68pp）、purge 統一、decile > Top50、跑前定死判讀規則、正確性修正不套效益門檻、配對 NW t 檢定 | **文件中完全不存在**這些概念 |
| **失敗的記錄品質** | 每個 bug 記「怎麼發現／根因／為什麼平常不會出事」，含自己犯的錯 | 只有一張 known weaknesses 表 |
| **反例文化** | 「我的假設被自己的實驗推翻」出現多次且保留 | 無 |
| **不確定性表達** | 誤差棒、`incomparable` 分級、`±?` 標記 | 無 |
| **單一真相制度** | `PORTFOLIOS` → `v62_arms.json` manifest | 有同型做法（回測 provider），但只在一處 |

**結論：在「結論可不可信」這個維度上，MarketMamba 領先 Fincept 一個世代。** 需要借鑑的是**工程結構**，不是研究方法。

---

## Part B — 架構對照表（Phase 5）

| Area | MarketMamba | FinceptTerminal | 差異的本質 | 建議 |
|---|---|---|---|---|
| **Data** | parquet 檔案 + git 傳輸；14 源台股專用 | SQLite ×2 + DataHub topic；100+ 源全球 | MM 是**批次面板**，Fincept 是**串流 topic** | 不採用 topic 匯流排（日頻不需要），**採用 TopicPolicy 的「宣告式新鮮度」觀念** → P0-2 |
| **Data providers** | `fetcher.py` 100 個平坦函式，無 source 抽象 | `Producer` 介面（宣告 pattern + refresh）；16 券商靠 `BrokerEnumMap` 資料表 | MM 每加一源＝加一坨函式；Fincept 每加一源＝填一張表 | **抽 `Source` 契約，但只包「寫入 + 新鮮度」，不重寫抓取邏輯** → P0-2 |
| **Data normalization** | 每個 `fetch_*` 內嵌 parse；欄位對映硬編（曾整個金融保險業版面漏掉） | `DataNormalizationService`：JSONPath + transform + schema 驗證，**對映存 DB 是資料**；`raw` 與 `normalized` 同筆保存 | MM 的正規化**綁在 I/O 上，天生不可測** | **抽 `parse_*(raw_text) -> DataFrame` 純函式**（不打網路），讓它可用固定樣本測 → P0-1 |
| **ML** | Mamba+GAT 自研，八模型定稿表，purge 統一，decile 為主尺 | Qlib 全套 + 30+ 演算法 + RL + AutoML，**無隔離紀律** | MM 深、Fincept 廣**且危險** | **完全不借鑑。** Qlib/RDAgent 明確列入不做 |
| **Quant** | 46/56/59/66 維特徵協定、F5/F6 消融系列、11 年 WF | 34 分析模組 + 18 QuantLib + 14 第三方 wrapper | Fincept 是工具箱，MM 是研究線 | 只在「組合最佳化從等權升級」時考慮 pyportfolioopt → P2 |
| **Backtesting** | `portfolio_lab`（規格凍結）+ 狀態機 replay 交叉驗證；**4 套引擎並存** | 6 個 provider，Python 為單一真相，C++ fallback 最小化 | MM 的多套是**歷史累積**，Fincept 的多套是**刻意抽象** | **不加 provider 抽象；改成刪減**：v62 上線後退役 sim_engine v1/v2 → P1 |
| **Portfolio** | 19 組合狀態機 + jsonl 逐日紀錄 + tier/family 分級 + 前瞻績效 | 持倉聚合 + P&L + 配置；`portfolio:*` topic | MM 這塊**比 Fincept 完整**（Fincept 沒有「多組合並行累積前瞻紀錄」的概念） | 不借鑑。維持現況 |
| **Risk** | 四層退場（退役中）、Trailing Stop 四檔、雜訊底線 | VaR / Sharpe / 部位限制 / `RiskManager`（workflow 執行前風控） | Fincept 的風控在**執行路徑**上，MM 的在**策略內** | 只在未來接自動下單時借鑑 `RiskManager` + `ConfirmationService` + `AuditLogger` 三件套 → P2 |
| **AI Agents** | 產品內只有 LLM 日報；開發用 Claude Code | 37 個 persona（= config）+ agentic runtime（budget/reflect/skill/eval） | 完全不同軸 | **不加 persona agent。** 借鑑 `budget.py` 的四層 cap 觀念與 eval harness 概念 → Phase 9 |
| **UI/UX** | React 25 頁，深色，19 組合分 family 呈現 | 54 screens + dock + F-key + command palette + theme token | Fincept 是工作站，MM 是儀表板 | 採用：**新鮮度指示、command palette、共用 formatting 層**；不採用 dock / F-key → P1 |
| **Visualization** | recharts + klinecharts | 自建 Qt Charts 元件庫 + `ui/formatting/` 獨立層 | — | 借 `formatting` 分層（防「寫死比較基準」第 7 次） → P0-3 |
| **Research workflow** | obsidian 三層記憶 + 檢查表 + 決策紀錄 + 引用紀律 | RDAgent 自動假設迴圈（**無隔離紀律**） | **MM 大幅領先** | 不借鑑。反而該把 MM 的引用紀律寫成 CI 檢查 → P1 |
| **Storage** | 檔案系統（parquet + CSV/JSON in git）；無 DB | SQLite ×2（主庫/快取庫）+ forward-only migration + 26 typed repo | MM 沒有 schema 演進機制 | **不引入 DB**（會毀掉 git 稽核性）。改為**parquet schema 守門**（型別逐欄）→ P0-2 |
| **Testing** | **零**。驗證＝一次性腳本 + 人工判讀 | 有限單元測試 + 4 個 CI workflow + **arch ratchet** | **這是最大的結構性差距** | **P0-1，最高優先** |
| **Documentation** | 三層記憶，40 KB CLAUDE.md，品質高 | current/target 標記 + known weaknesses 表 + ADR（制度有、實踐無） | 兩邊各有所長 | 借 **current/target 標記法** 與 **ADR**；MM 的「決策紀錄」已是 ADR 的變體 → P1 |

---

## Part C — MarketMamba **不該**抄的東西（Phase 6）

明確列出「Fincept 有，但 MarketMamba 現階段不應該做」。

### C.1 整個技術棧層級

| 項目 | 為什麼不 |
|---|---|
| **C++20 / Qt6 原生桌面** | 家人要能用手機看。web 是對的選擇，換成桌面等於砍掉主要使用場景 |
| **嵌入式 Python + 雙 venv + PythonRunner 橋接** | MarketMamba 本來就是 Python。橋接是 Fincept 為了「C++ UI + Python 生態」付的稅，MM 沒有這個稅要付 |
| **SQLite + 26 個 repository + migration** | 目前用 git 傳 CSV/JSON 換到了免費的版本控制與稽核軌跡。換 DB 等於**用工程複雜度換掉一個現在正在發揮作用的性質** |
| **ADS dock manager / 多視窗** | 單人使用，沒有多螢幕工作站需求 |

### C.2 功能層級

| 項目 | 為什麼不 |
|---|---|
| **16 家券商整合** | MM 有一家（永豐 shioaji，且已在 PersonalOS）。抽象化 1 個實例是純虧損 |
| **100+ data connector** | MM 只有台股。多加一個海外源不會提升 alpha，只會增加健檢面積 |
| **即時 WebSocket / HFT / 訂單簿 / 造市** | 日頻、每 20 日再平衡。整條與定位無關 |
| **加密貨幣 / 錢包 / 鏈上** | 無關 |
| **地緣政治 19 個 agent / 海事追蹤 / 衛星資料** | 無關，且沒有任何證據顯示它對台股橫斷面排序有 alpha |
| **37 個 persona agent** | 見 fincept 分析 §4.4：那是產品差異化，不是工程。MM 沒有要賣產品 |
| **Qlib 全套 + RDAgent 自動研究迴圈** | **最重要的一條**。MM 已有自己的特徵協定 / 標籤協定 / purge 紀律 / 八模型定稿表。導入 Qlib＝重來一次資料協定；而 RDAgent 的「自動生假設→自動回測→自動排名」在**沒有多重檢定校正**下是假發現產生器。**這與 MM 累積 12 個月的方法論紀律直接衝突** |
| **RL / AutoML / meta-learning / online learning** | 在 ±6pp 雜訊底線之下自動選模型 = 自動過擬合 |
| **Node editor / 視覺化 workflow DAG** | 每日管線 6 步、只有一個人改。程式碼比拖拉節點好 diff、好回溯、好測 |
| **多 LLM provider 抽象（8 家）** | MM 用 Anthropic 一家。抽象 1 個實例＝純虧損 |
| **語音 / STT / TTS / voice trigger** | 無關 |
| **多 provider 回測抽象（6 家）** | `portfolio_lab` 規格已凍結且與狀態機互驗過（38.020% vs 38.020%）。**加 provider 抽象會讓「唯一真相」變成「六份真相」** |
| **外部 MCP server 生態** | 沒有第三方要擴充 MM |

### C.3 制度層級（有價值但現在不划算）

| 項目 | 為什麼現在不 |
|---|---|
| **13 個 bounded context + 依賴方向 CMake 強制** | MM 是 Python 單體、~130 檔。用 import-linter 之類做同件事的成本遠低於這套 |
| **五級授權 + `is_destructive` 工具閘門** | MM 沒有多使用者、沒有自動下單。等真的要接下單再說（P2） |
| **正式 ADR 目錄** | MM 的「決策紀錄」已經是 ADR 的變體，而且**執行狀況比 Fincept 好**（Fincept 的 `docs/adr/` 只有 README）。**不要為了形式改格式** |

### C.4 一條特別的警告

**不要因為 Fincept 有 `services/` 就把 MarketMamba 拆成 service 層。**
Fincept 有 ~50 個 service 是因為它有 54 個畫面 × 13 個 context 要解耦。MM 的 `marketmamba/` 套件已經是按職責分好的（data / models / signals / quant / backtest / llm / evaluation）。**再加一層 service 只會增加轉發程式碼。**

---

## Part D — 高價值改進（Phase 7）

### Priority 0 — 現在就該做

---

#### **P0-1｜數值回歸測試套件 + 三條架構棘輪**

**Why**
MarketMamba 的整部失敗史都是同一類：**不會報錯的錯**。
`prices_raw` 靜默掉 15% 的列、`Free_Cash_Flow` 一直等於營運現金流、`Day_Trade_Volume` 426 萬列全是 0、`step()` 與 `replay()` 並列打破不同差 0.18pp、`ann_stderr_pp` 的 `null → 0` 讓 19 個 arm 全被誤標、`--tag` 蓋掉正式結果 JSON、`score_window()` 對每個 arm 印同一個別人的基準。

**這些全部是人工發現的。** 每一個都可以被一條斷言擋住。而 CLAUDE.md 裡已經明文寫著「驗收標準是與 git HEAD 逐位元相同」——**紀律已經存在，只是沒有被自動化。**

Fincept 的 arch-ratchet 提供了關鍵的**存量債務**解法：不要硬門檻（會被拆掉），要棘輪（只能降不能升）。

**Expected benefit**
- 把「每次改 `feature_engineer.py` 都要人工跑回歸」變成一條指令
- 08-17 正式起跑後，任何靜默的數值漂移當場現形
- **對 AI 協作的槓桿最大**：Claude Code 改完可以自己驗，不必每次都靠使用者貼 log

**具體內容（最小可行，不要一次做大）**

1. **黃金檔回歸**（3 條）
   - `feature_engineer.build_features()` 對固定 20 天 × 50 檔的樣本 → 雜湊比對
   - `portfolio_lab.replay()` 對已知窗 → 年化 37.28% / 換手 82.7% 精確比對
   - `v62_portfolio.step()` replay → 與上一條 **0.000pp** 比對（這條 08-08 才修好，最該被鎖住）
2. **純函式解析測試**：把幾個 `fetch_*` 的 parse 段抽成 `parse_xxx(text) -> DataFrame`，用存下來的真實回應樣本測（**不打網路**）
3. **三條棘輪**（只能降）
   - `to_parquet(` 呼叫點裡沒有 `index=False` 的數量 → **目前應為 0，鎖住它**
   - 原始碼中的硬編比較基準（regex 抓 `0.38` / `38.0%` / `0.1145` 這類字面值出現在非測試檔的次數）
   - `?? 0` / `or 0` 套在誤差/不確定性欄位上的數量

**Complexity**：中低。pytest + 幾個 fixture + 一支 ratchet 腳本 + 一個 GitHub Actions workflow（或先只做 local `make check`）
**Dependencies**：無。可完全獨立於 08-17 起跑
**Risk**：**低，但有一個真實風險**——黃金檔會在「正確性修正」時合法變動。緩解：黃金檔更新必須是獨立 commit 且 commit message 要寫「為什麼這個數字該變」。這正是 arch-ratchet 那句「only ever regenerate to LOWER a baseline」的同型紀律
**影響現有架構**：**否。純附加。**

---

#### **P0-2｜資料寫入端的宣告式契約（Source policy）**

**Why**
`_append_to_parquet` 的兩次事故（08-08 整天替換語意掉 15% 列、08-10 缺 `index=False` 讓當天三次執行全掛）**根因相同**：寫入端的語意是隱含的、分散的、每個呼叫點各自負責。

Fincept 的 `TopicPolicy` 給了答案的形狀：**把「這個資料源的行為規則」變成一個純資料 struct，而不是散在程式裡的 if。**

**Expected benefit**
- 「今天沒抓到 ≠ 今天沒交易」這條規則寫一次，14 個源自動受益
- 新鮮度檢查從「每個 catch_up 各判一次」變成宣告
- **`refresh_timeout` 的觀念可以直接救 08-10 那類卡死**

**具體內容**

```python
# 概念示意，不是要抄 Fincept 的程式碼
@dataclass(frozen=True)
class SourcePolicy:
    name: str
    allow_shrink: bool = False          # 預設不准縮小（08-08 的教訓）
    max_staleness_days: int = 0         # 每日源＝0 天容忍（08-05 的教訓）
    key_cols: tuple[str, ...] = ("Date", "stock_id")
    expected_dtypes: dict[str, str] = ...  # 寫入前逐欄守門（08-08 的教訓）
    availability_hour: int = 21          # 幾點之後才會公布（margin/daytrade 的教訓）
```

搭配**一個**寫入函式（取代目前分散的 append 邏輯），該函式負責：
寫入前型別/欄名/鍵集合守門 → 寫暫存 → 驗過才 `os.replace` → `index=False` 強制 → 縮小需明確授權。

**這些邏輯 08-08/08-10 已經全部寫過了**（`fix_prices_index_column.py`、`backfill_prices.py`），只是各自一份。這個工作是**收斂既有正確做法**，不是發明新東西。

**Complexity**：中。風險在於要動到 `fetcher.py` 這條線上的 production 程式碼
**Dependencies**：**建議 P0-1 先做**——有回歸測試才敢動 `fetcher.py`
**Risk**：中。緩解：`SourcePolicy` 先以 **opt-in** 方式導入（新寫入路徑用，舊路徑保留），逐源遷移，每遷一源跑一次鍵集合比對
**影響現有架構**：**是**，但是收斂型的。建議排在 **08-17 起跑之後、V6.1 退役期間**，不要卡在起跑前

---

#### **P0-3｜結果發布契約統一（`ResultsStore` + manifest 擴大）**

**Why**
後端 4 個 router 各自抄一份「GitHub raw + 1h TTL + `asyncio.Lock`」，且 URL 是 `GITHUB_RESULTS_URL.replace("df_kelly.csv", name)` 組出來的。這是 Fincept 明列的 **「3-way caching → drift, double-fetch」** 弱點的 MM 版本，而且 MM 的版本更脆：**環境變數改個檔名，整組靜默壞掉。**

同時，MM 已經在 `v62_arms.json` 上做對了正確的事（manifest 是唯一真相、fallback 只放主線一個）。**這個模式應該擴大，而不是只活在 v62 一支 router。**

**Expected benefit**
- 快取邏輯一份、URL 組法一份、錯誤處理一份
- **每個結果檔帶 `generated_at` + `data_date` + `pipeline_version`** → 前端可以顯示新鮮度（見 P1-1）
- 未來換發布方式（例如改成 GitHub Release 或物件儲存）只改一處

**具體內容**
1. `app/backend/results_store.py`：`async def get(name, *, as_json) -> tuple[data, meta]`，內含 TTL + lock + last-known-good
2. URL 用 **base + 檔名**，不用 replace
3. 發布端（`deploy/publisher.py`）在每個結果檔旁寫一份 `results_manifest.json`：檔名 → `{generated_at, data_date, rows, pipeline_version}`
4. 舊 router 逐一改用，**一次一支**，改完比對回傳 payload 逐欄相同

**Complexity**：低
**Dependencies**：無
**Risk**：**低但要小心「線上是紅線」**。作法：先加 `results_store` 並讓 `v62.py` 用它（v62 尚在試跑期），驗一週後再遷 `signals.py`
**影響現有架構**：是，但只在 backend，且可逐支遷移

---

### Priority 1 — 下一階段（08-17 起跑之後）

---

#### **P1-1｜前端資料新鮮度指示**

**Why**
Fincept 的 `last_publish_ms` 註解說得最好：**「顯示一個沒有時間標記的價格，是主動誤導」**。MM 的整部失敗史都是靜默的舊資料。而 08-09 的誤差棒事件證明：**沒有標記的數字會被當成精確的數字讀。**

**Expected benefit**：家人看 dashboard 時能區分「今天的」與「上週五的」。使用者能一眼看出管線是不是掛了，不必去看 log
**Complexity**：低（P0-3 的 manifest 一做完就幾乎免費）
**Risk**：低
**影響架構**：否

#### **P1-2｜共用 formatting 層**

**Why**：「寫死比較基準」在 2026-08-09 一天內踩到**六個地方**。根因不是粗心，是**百分比 / pp / ±誤差棒 / tier 顏色的呈現規則沒有唯一的家**。
Fincept 把 `ui/formatting/` 拉成獨立一層正是這個對策。

**具體**：`formatPct` / `formatPP` / `formatWithErrorBar`（`null → "±?"` 且強制標 `*`）/ `tierColor`（`incomparable` 必須與 `inferior` 不同色）。
**規則**：任何比較基準只能由 API 帶進來，formatting 層**不接受字面值常數**。
**Complexity**：低 | **Risk**：低 | **影響架構**：否（前端內部）

#### **P1-3｜回測/模擬引擎退役計畫**

**Why**：4 套引擎並存已經造成一次真實 bug（0.18pp）。V6.1 + scanner 退役時，`sim_engine` v1/v2、`scanner_engine`、`condition_analyzer` 應一併退役，留下 `portfolio_lab`（研究）+ `v62_portfolio.step()`（線上），且兩者已互驗過。
**Complexity**：低（多半是刪除）| **Risk**：低，但**必須在 V6.1 確定退役後**才動 | **影響架構**：是（減法）

#### **P1-4｜Command palette（前端）**

**Why**：19 個組合 × family 分組 × 多分頁，選單一定爆。搜尋比階層好擴展。
**Complexity**：低—中 | **Risk**：低 | **影響架構**：否

#### **P1-5｜引用紀律的機械化**

**Why**：`06 研究紀錄/00 研究總覽.md` 的五條引用紀律目前**只存在於人的腦中**。可機械化的至少兩條：
- 年化差 < 6pp 卻被標成優劣分級 → 檢查腳本
- `n_days < 60` 卻顯示年化而不帶 `ann_stderr_pp` → 檢查腳本

這是 P0-1 棘輪的延伸，但因為要碰研究產出的語意，排在 P1。
**Complexity**：中 | **Risk**：低 | **影響架構**：否

#### **P1-6｜文件加 current / target 標記**

**Why**：CLAUDE.md 目前把「現況」「決定了但沒做」「打算做」混在一起（例如 §訊號系統 V6.2 整節標了「將隨 V6.1 退役」但仍完整保留）。Fincept 在 ARCHITECTURE.md 開頭教讀者怎麼讀那份文件，是很低成本的改進。
**Complexity**：極低 | **Risk**：無

---

### Priority 2 — 長期

| # | 項目 | 觸發條件 |
|---|---|---|
| P2-1 | **執行路徑三件套**（風控前置 / 人工確認 / 稽核軌跡） | 只在真的要接自動下單時。Fincept 的 `RiskManager` + `ConfirmationService` + `AuditLogger` 是正確的形狀 |
| P2-2 | 組合最佳化庫（pyportfolioopt / skfolio） | 等組合層從等權走向風險模型。**先要有一季前瞻紀錄證明現行規格站得住** |
| P2-3 | 標準績效報告（QuantStats 式） | 等前瞻紀錄 > 一季 |
| P2-4 | 資料源 schema 版本化 / migration | 只在 parquet schema 開始頻繁演進時 |
| P2-5 | 常駐推論 worker（消滅冷啟動） | 只在每日總時長變成瓶頸時。目前 V6.2 全程 9–18 分鐘，不是瓶頸 |

---

### Ignore — 現階段不值得做

Qlib / RDAgent / RL / AutoML / online learning、37 個 persona agent、多 LLM provider 抽象、多回測 provider 抽象、node editor、券商抽象層、即時串流 / HFT、加密貨幣、地緣政治與海事資料、SQLite 遷移、C++ 或桌面化、dock manager、五級工具授權、外部 MCP 生態、多使用者 / 帳號 / 計費。

理由統一：**它們解決的問題 MarketMamba 沒有；而它們帶來的維護面積 MarketMamba 付不起（一個人）。**

---

## Part E — 七個總結問題的答案

### 1. FinceptTerminal 對 MarketMamba 最有價值的 5 個地方

1. **`TopicPolicy` 式的宣告式新鮮度／寫入政策**（純資料 struct，不是散落的 if）—— 直擊 MM 兩次最嚴重的事故根因
2. **`arch-ratchet`：對存量債務用棘輪不用門檻**——「人們會保留的棘輪勝過人們會刪掉的門檻」
3. **MCP 工具層的「schema 驗證跑在 handler 之前」**——把防禦寫在一個地方，而不是每個消費端各寫一次（MM 已經被「兩份實作」咬過）
4. **`DataNormalizationService` 的四個決定**：對映是資料 / `raw` 與 `normalized` 同筆保存 / `errors` 是欄位不是例外 / 有不打網路的純函式版供測試
5. **文件的 current-vs-target 標記 + known weaknesses 帶 owner**——低成本、立即可用

### 2. 哪 5 個東西不應該學

1. **Qlib + RDAgent 自動研究迴圈**——與 MM 的隔離／多重檢定紀律直接衝突，是假發現產生器
2. **37 個 persona agent**——那是行銷指標（roadmap 寫「50+ agents」自證），不是工程
3. **多 provider 抽象（回測 6 家 / LLM 8 家 / 券商 16 家）**——MM 各只有 1 個實例，抽象化純虧損，且會把「唯一真相」變成「六份真相」
4. **SQLite + 26 repository + migration**——會毀掉「git 當資料匯流排」現在正在提供的版本控制與稽核性
5. **Node editor / 視覺化 workflow**——程式碼比拖拉節點好 diff、好回溯、好測；MM 只有一個人改管線

### 3. MarketMamba 現在最大的 architecture gap

> **驗證是人工的。**

不是資料抽象、不是前端、不是 agent。MM 有整個專案裡最好的**驗證紀律**（逐位元回歸、鍵集合比對、雙路徑互驗、跑前定死判準），但**這些紀律沒有一條被自動化**。每一次都要靠人記得去跑、靠人判讀 log。

證據是量化的：CLAUDE.md 記錄的重大事故裡，**每一個都是靠人在做別的事時偶然發現的**——清備份時發現 15% 遺失、寫誤差棒後第一天跑才發現 n=1 的洞、寫前瞻績效工具時才發現線上與回測分歧。

**紀律存在但不自動 ⇒ 缺陷的發現時間取決於運氣。** 08-17 之後要累積前瞻實戰紀錄，運氣不是可接受的品管機制。

### 4. 現在最值得做的 3 個 improvement

1. **P0-1 數值回歸測試 + 三條棘輪**（把已有的驗收紀律變成一條指令）
2. **P0-3 `ResultsStore` + manifest 擴大**（成本最低、風險最低、且是 P1-1 新鮮度指示的前置）
3. **P0-2 `SourcePolicy` 寫入契約**（收斂 08-08/08-10 已經寫過的正確做法）

順序建議：**P0-1 → P0-3 →（08-17 起跑）→ P0-2**。
理由：P0-2 要動 `fetcher.py` 這條 production 線，應該在有回歸測試之後、且不要卡在起跑前。

### 5. 哪些事情應該延後

- 任何**組合最佳化升級**（等一季前瞻紀錄）
- 任何**執行/下單路徑**（風控三件套等真的要下單）
- 任何**架構分層重構**（service 層、bounded context、依賴強制）——MM 規模不到
- **前端版面重設計**——使用者自己說「等全部上線、看得見全貌之後再想」，這是對的
- **P0-2 之外的 `fetcher.py` 重構**——215 KB 很難看，但它現在是對的；**不要為了美觀動一個運作中的資料層**

### 6. 如果只能借鑑 Fincept 的一個部分

> **`TopicPolicy` + `Producer` 所代表的那個觀念：把「資料的行為規則」變成宣告式的純資料，讓平台去執行它，而不是讓每個呼叫點各自負責。**

理由：MarketMamba 到目前為止最嚴重的三次事故——15% 靜默資料遺失、欄名撞名讓三次執行全掛、margin 抓太早靜默 ffill——**全部是「規則存在於人的腦中與分散的程式碼裡，而不是存在於一個地方」的後果**。
而 `TopicPolicy` 只有 60 行。**這是整份研究裡投入產出比最高的一個觀念。**

### 7. Fincept 的存在是否改變我對 MarketMamba 長期定位的看法

**改變了，而且是往收窄的方向。**

三個具體理由：

**(a) 「通用金融終端」這條路已被證明是資金消耗戰。**
Fincept 有 star、有 trendshift、有 16 家券商、有 100+ 資料源、有 AGPL+商業雙授權與 $50,000 起跳的違約金條款——**然後在 2026-06 因資金限制改成每月一次更新**。一個人的專案不可能贏這種廣度競賽，而且**不該去比**。

**(b) 廣度是可複製的，MarketMamba 的東西不是。**
54 個畫面、37 個 agent、100 個 connector，本質上都是「多做幾次同樣的事」。而 MM 累積的東西——台股 11 年的乾淨面板（含官方除權息因子表、MOPS 直連、減資處理）、八模型定稿表、雜訊底線的實測值、purge 的飽和點、「短標籤製造換手」這個機制的證據——**這些是不可複製的資產，而且沒有任何 Fincept 式的廣度能替代。**

**(c) Fincept 缺的正好是 MM 有的，這是定位的答案。**
Fincept 的 AI Quant Lab 能自動生假設、自動回測、自動排名，但沒有任何一句話提到隔離、多重檢定、雜訊底線。**在 AI 讓「產生策略」變得極便宜的時代，稀缺的不是策略，是「這個結論可不可信」的判斷力。**

→ **建議的長期定位：不是「台股版 Bloomberg」，而是「一套結論可稽核的台股研究系統」。**

具體含義：
- 產品面繼續**只做一件事**（台股橫斷面選股 + 組合建構），不擴資產類別、不擴市場
- 工程面的投資**優先投在「可信度基礎建設」**（回歸測試、資料契約、誤差棒、新鮮度、稽核軌跡），而不是投在功能廣度
- 對外若有一天要呈現，賣點是**「每個數字都能追到它是怎麼來的」**，而不是「有幾個模型／幾個指標」

**這個定位與 Fincept 完全不重疊，也因此不需要跟它競爭。**
