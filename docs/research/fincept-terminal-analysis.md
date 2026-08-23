# FinceptTerminal 架構分析（reference implementation 研究）

> 研究日期：2026-08-11
> 研究對象：`github.com/Fincept-Corporation/FinceptTerminal`，`main` HEAD，`--depth 1` clone 到暫存目錄
> 研究性質：**只讀**。沒有修改 Fincept，沒有複製任何程式碼進 MarketMamba
> 授權注意：Fincept 是 **AGPL-3.0 + 商業雙授權**，且 README 明寫「fork 並替換資料源不解除授權義務」
> → **本專案不得抄任何一行 Fincept 程式碼**。以下全部是「架構觀念」層級的閱讀筆記

---

## 0. 先講三件會影響判讀的事實

### 0.1 這個專案正在減速維護

README 第一段（2026-06）自述：因資金限制，公開版改成**每月一次更新、不再每日維護**，團隊轉向訂閱制私有版與新專案 Quantcept。

→ 判讀時要小心：**docs 裡寫的「target state」有相當比例不會被實作完**。ARCHITECTURE.md 自己用 `(current)` / `(target)` 標記區分，這是誠實的做法，但也代表引用它的設計時要先確認那一段是不是已實作。

### 0.2 最有價值的三份文件當中，有兩份不在 repo 裡

`.gitignore` 明確排除：

```
agents.md
.claude/
CLAUDE.md
plans/
**/plans/
fincept-qt/DESIGN_SYSTEM.md
*.md
!README.md
!**/README.md
```

被排除的包括 `fincept-qt/CLAUDE.md`（MCP guide 稱之為 "Project rules"）、`plans/mcp-refactor-phase-{1..6}-*.md`、`REFACTOR_PLAN.md`、`DESIGN_SYSTEM.md`、以及 `docs/agentic-research/design/*`（README 引用了，但 repo 裡只有 README 本身）。

→ **Phase 3 / Phase 9 的問題（他們的 Claude Code workflow 長什麼樣）無法從 repo 直接回答。** 底下關於這部分的結論一律標明是「從可見證據反推」，不是讀到的。這是本研究最大的證據缺口，不補假設。

### 0.3 規模差距是量級的，不是程度的

| | FinceptTerminal | MarketMamba |
|---|---|---|
| C++ | ~1,626 檔 / ~342,000 行 | 0 |
| Python | ~1,400 檔 | ~130 檔（不含 archive） |
| 前端 | Qt6 Widgets 原生（54 screens） | React 頁面 ~25 支 |
| 資料源 | 100+ connectors | 14 個直連源（台股專用） |
| 目標使用者 | 全球零售／專業分析師 | **一個人 + 家人看 dashboard** |
| 資產類別 | 股 / 債 / 衍生品 / 加密 / 另類 | 台股單一市場 |

**這不是「大小之分」，是「物種之分」。** Fincept 是通用金融工作站（廣度優先），MarketMamba 是單一市場的量化研究系統（深度優先）。後面所有建議都建立在這個前提上。

---

## 1. 整體架構（Phase 2.1）

### 1.1 一句話回答「Fincept 怎麼把資料 → 分析 → AI → 視覺化 → 組合/交易串起來」

> **靠一條 in-process 的 pub/sub 資料平面（DataHub），把「誰去抓」和「誰要看」完全解耦；上層任何一個消費者（畫面、AI agent、node editor、報表）都只認 topic 字串，不認資料源。**

這是整份 Fincept 架構裡唯一真正原創、也唯一真正值得 MarketMamba 借鑑的東西。其餘（54 個畫面、16 家券商、37 個 agent）都是**在這條骨幹上的量產**。

### 1.2 分層與依賴方向

```
PRESENTATION   Screens(54) / DashboardWidgets(13) / DockManager
      ↓
APPLICATION    13 個 bounded context（Markets / News / Trading / Agents / …）
      ↓
DATA PLANE     DataHub（topic pub/sub）+ CacheManager（SQLite TTL）
      ↓
ADAPTERS       Broker / MCP / PythonRunner / HttpClient / WebSocket
      ↓
INFRASTRUCTURE Logger / EventBus / SessionManager / SecureStorage / Repositories
      ↓
PLATFORM       Qt6
```

**四條硬規則**（ARCHITECTURE.md §2）：

1. 依賴方向單向，永不反向
2. Adapter 是葉節點：兩個 service 可共用一個 adapter，adapter 不可呼叫 service
3. **跨 context 一律走 DataHub topic 或 typed event，不可直接 include**
4. Infrastructure 沒有業務知識——「它不知道什麼叫 watchlist」

第 3 條是關鍵。它把「Markets 要用 Trading 的資料」這種最常見的耦合來源，變成「Markets 訂閱 `broker:*:orders`」。**兩邊沒有編譯期依賴，也就不可能循環。**

### 1.3 C++ / Python / Qt 的分工

界線劃得很清楚，而且是**依「變更頻率」劃的，不是依「語言擅長什麼」劃的**：

| 層 | 語言 | 理由（我的解讀） |
|---|---|---|
| UI、狀態機、生命週期、資料平面 | C++20 / Qt6 | 需要低延遲與型別安全；變更慢 |
| 分析、模型、資料抓取 | Python（bundled 3.11.9） | 生態系在這裡；變更快；貢獻者門檻低 |
| 橋接 | `PythonRunner`（QProcess + JSON on stdout） | 進程隔離：Python 掛掉不會拖垮 app |

`PythonRunner` 的幾個設計值得記：

- **兩個 venv 並存**：`venv-numpy1`（vectorbt / gluonts 這些還卡在 NumPy 1.x 的套件）與 `venv-numpy2`（預設）。
  → 這正是 MarketMamba 的 **Windows pandas 2.2.2 / WSL pandas 3.0.2 分裂**問題的另一種解法：**不統一版本，改成明確地維護兩套並宣告誰用哪套**。
- **並行上限 3 個 process，超額排隊**
- **冷啟動 0.5–1.5 秒／次**，而且他們把這列在「已知弱點」表上（Phase 11 要改成常駐 worker pool）
- **環境變數白名單**：只注入 22 個已知 API key，其他所有 `*_API_KEY` / `*_SECRET` / `*_TOKEN` **主動從子行程環境剝掉**，防止經由 `/proc/<pid>/environ` 外洩

最後這條是很成熟的安全思維：**不是「不要傳敏感值」，而是「預設剝除，白名單放行」**。

### 1.4 模組邊界的可執行化

Fincept 沒有停在「文件寫了依賴方向」。他們有兩個機制讓規則**自己執行**：

**(a) CMake 目標切分（target state）**：把 `fincept_core` / `fincept_ui` / `fincept_datahub` … 各自變成獨立 library target，依賴方向寫進 CMake → 循環依賴變成**編譯失敗**，不是 code review 靠人看。
（現況仍是單一 3,300 行 `CMakeLists.txt`，這條是 target 不是 current。）

**(b) `.github/workflows/arch-ratchet.yml`（已實作）**：三個「只能降不能升」的計數器：

- inline `setStyleSheet` 總數
- 迴圈體內的 `setStyleSheet`（每列每 tick 重新 parse 一次 CSS）
- 未經 `tr()` 包裝的 UI 字串（`lupdate` 看不到 → 永遠無法翻譯）

workflow 註解裡的一句話值得整段抄進方法論筆記：

> Ratchets rather than hard gates: there are ~5,200 / ~460 / ~120 existing sites, so an outright ban would fail every PR on day one and the check would be deleted.
> **A ratchet people keep beats a gate they remove.**

→ **對存量債務，硬門檻會被拆掉，棘輪才活得下來。** 這是可直接搬到 MarketMamba 的觀念（見 gap analysis P0-1）。

### 1.5 設定系統

- `AppConfig` 單例存全域常數（URL、版本）
- 使用者設定走 `SettingsRepository`（SQLite）
- 憑證走 `SecureStorage`（SQLite + AES-256-GCM，key 由 `machineUniqueId` 導出）
- **刻意不用平台 keychain**（Keychain / DPAPI / libsecret）——理由沒寫，推測是跨平台一致性 > 安全上限

一條明確的紀律：**`AuthManager::session()` 是憑證的唯一真相，`SettingsRepository` 只是 fallback 副本，任何地方都不准另外快取憑證。**
這與 MarketMamba 的「`PORTFOLIOS` 是唯一真相，manifest 每天發布」是**同一個模式**。

### 1.6 Plugin / extension 機制

三個層次，由鬆到緊：

1. **外部 MCP server**（`McpManager` → JSON-RPC over stdio）：真正的第三方擴充點
2. **內部 MCP tool**（`McpProvider`）：C++ 內建工具，靠 schema 註冊
3. **Python script**：目前是硬編路徑（列為弱點，target 是 `ScriptCatalog` 具名解析）

**沒有真正的 plugin ABI。** 對桌面單體來說這是對的取捨——他們明講 microservices 是 anti-goal。

---

## 2. 資料架構（Phase 2.2）

### 2.1 DataHub：這份研究的核心發現

`src/datahub/`（只有 6 個檔、約 70KB）撐起整個 app 的資料流。

**Topic 格式**：`domain:subdomain:id[:modifier]`，例如 `market:quote:AAPL`、`broker:zerodha:orders`、`agent:hedgefund:run:42`。
**關鍵設計**：topic 字串**由內容版本化**——`:1d` 和 `:1m` 是兩個不同 topic，不是同一個 topic 的參數。

**訂閱端 API**（`DataHub.h`）：

```cpp
subscribe(owner, topic, slot)            // owner 死掉自動退訂（QPointer 生命週期綁定）
subscribe<T>(owner, topic, slot)         // typed，自動 unwrap
subscribe_pattern(owner, "market:quote:*", slot)
subscribe_errors(owner, topic, slot)     // 錯誤也可以單獨訂閱
peek(topic)                              // 讀快取、不觸發抓取、過期回無效值
peek_raw(topic)                          // 讀 last-known-good，不管 TTL
last_publish_ms(topic) / age_ms(topic)   // O(1) 新鮮度查詢
```

`last_publish_ms` 的註解直接寫出設計理由，值得整段引用：

> A financial terminal showing a price with no indication of its age is actively misleading: the user cannot distinguish a live quote from one frozen twenty minutes ago by a dead producer.

**→ 「顯示了一個沒有時間戳的數字」比「顯示不出來」更危險。** 這條與 MarketMamba 08-05 的 macro 新鮮度檢查、08-09 的誤差棒教訓完全同構。

### 2.2 `TopicPolicy`：把「新鮮度」變成宣告式資料

`TopicPolicy.h` 只有 60 行，但它是整個資料平面最精華的部分。每個 topic 掛一組**純資料**的政策：

| 欄位 | 預設 | 意義 |
|---|---|---|
| `ttl_ms` | 30,000 | 快取視為新鮮的時間 |
| `min_interval_ms` | 5,000 | 不管幾個訂閱者要求，refresh 頻率下限 |
| `refresh_timeout_ms` | 30,000 | producer 逾時未 publish/publish_error → **清掉 in_flight 並記警告** |
| `push_only` | false | WebSocket 類：scheduler 完全不碰 |
| `coalesce_within_ms` | 0 | 高頻推送的背壓：窗內只送最後一筆 |
| `drop_on_idle` | false | 最後一個訂閱者離開就丟掉整個 TopicState（給 agent run 這種無界 topic 用） |
| `pause_when_inactive` | false | 視窗最小化時暫停 fan-out，但**快取照更新** |

四個值得學的點：

1. **`refresh_timeout_ms` 防的是「靜默卡死」**——producer 掛了不會永遠 pin 住 topic。這正是 MarketMamba 08-10 三次執行 0 秒失敗那類問題的通用解。
2. **`drop_on_idle` 是為「topic 集合無界」設計的**。MarketMamba 的 19 個 arm × 逐日 jsonl 是有界的，暫時用不到，但概念要記。
3. **`pause_when_inactive` 的註解特別警告「少用」**——只給高頻串流；低頻家族留 false，否則使用者切回視窗要等一次 round-trip。**預設值選在「零行為變更」那一側。**
4. 整組是**純資料 struct**，沒有 virtual、沒有繼承。政策可以被序列化、被 diff、被測試。

### 2.3 Producer：新增資料源的唯一介面

`Producer.h` 只有 1.2KB。一個新資料源要做的事：

1. 宣告自己擁有哪些 topic pattern
2. 實作 `refresh(topics)`
3. 抓到之後呼叫 `hub.publish(topic, value)`；失敗呼叫 `hub.publish_error(topic, msg)`

**Hub 負責的（producer 完全不用管）**：排程、TTL、rate limit、request 合併（100ms 窗）、逾時、fan-out、last-known-good 保留、錯誤廣播。

`publish_error` 的語意特別重要：

> Does NOT update the cached value — **last-known-good stays visible.**

→ **失敗時保留舊值 + 明確廣播錯誤**，而不是「清空」或「靜默保留」。使用者看得到「這是舊的，而且我知道它為什麼舊」。

### 2.4 「不同 provider 的格式怎麼避免污染上層」——正面回答

Fincept 用**三道閘**，一道比一道正式：

**第一道：Producer 邊界（每個 provider 各自負責 parse）。**
C++ 端每個 producer 自己把 provider 回應轉成 `QVariant` 型別化結構。這是「靠慣例」的一道，不強制。

**第二道：Canonical vocabulary（`Instrument`）。**
`trading/instruments/InstrumentTypes.h` 定義跨券商的符號模型。ARCHITECTURE.md 明列為 public contract：

> All cross-broker code uses this — **never raw broker strings.** `Instrument::canonical_topic_id()` builds DataHub keys.

搭配 `InstrumentSource` + `SymbolResolver`：新增券商要**註冊一個 source**，`InstrumentService` 透過 resolver 分派，**不准寫 if-chain**。
另有 `BrokerEnumMap<T>`：把每家券商的 order_type / side / product 對映從 switch 換成**資料表**（16 家已遷移 14 家）。

**第三道（最正式）：`DataNormalizationService` — 宣告式 provider 對映。**

`services/data_normalization/DataNormalizationService.h` 的資料流：

```
DataMapping（存在 SQLite，使用者可設定）
   ├─ base_url + endpoint + auth_type
   ├─ JSONPath 欄位抽取（$.key、$[0].key、$[*][N]）
   └─ transform 名稱（to_number / unix_ms_to_iso / upper / abs_value …）
        ↓
   fetch_and_normalize()
        ↓
   NormalizedRecord {
       normalized  ← 通過 schema 驗證的資料
       raw         ← 原始 API 回應（保留！）
       errors      ← 驗證錯誤清單（空 = 乾淨）
       extracted_at
   }
        ↓
   persist 到 normalized_data table
```

四個設計決定值得學：

1. **對映是資料不是程式**——存在 SQLite，可被 UI 編輯，新資料源不必重編譯
2. **`raw` 永遠保留在同一筆記錄裡**——正規化錯了可以回頭查，不用重抓
3. **`errors` 是欄位不是例外**——驗證失敗的記錄照樣落盤並標記，而不是丟掉
4. **有 `normalize_raw()` 純函式版**（不打 HTTP），明講是給測試用 → **正規化邏輯天生可測**

**誠實的限制**：這套只覆蓋「使用者自訂資料源」那條路徑，核心的 100+ Python fetcher **沒有走它**。所以答案是：**Fincept 有這個抽象，但沒有全面套用。** 它是 target，不是 current。

### 2.5 快取、歷史、即時

| 面向 | 做法 |
|---|---|
| 快取 | `CacheManager`（獨立的第二個 SQLite 檔 `CacheDatabase`）；官方唯一快取 API |
| 歷史 | Python fetcher → pandas → JSON；沒有統一的歷史面板概念 |
| 即時 | WebSocket（Kraken / Hyperliquid），每個 exchange 一條專屬 thread，`push_only=true` |
| 另類資料 | Adanos 情緒（Reddit / X / 新聞 / Polymarket）；**未設定時整個功能休眠、其餘行為不變** |
| 總經 | FRED / DBnomics / IMF / World Bank / OECD，各自一支 Python |
| 新聞 | 聚合 + 分群 + 去重 + 偏離偵測（`news:*`） |
| 儲存 | 兩個實體 SQLite（主庫 + cache 庫）+ forward-only migration + 26 個 `BaseRepository<T>` |

**已知弱點（他們自己列的）**：「3-way caching：`CacheManager` + 畫面自帶的 `QHash` + service 層 debouncer」→ drift 與重複抓取。
反面規則寫得很精確：畫面裡的 `QHash` **只有兩種情況允許**——(a) 隱藏時清空的即時 feed 分派表、(b) DataHub 已擁有資料的視圖／索引。

---

## 3. Quant / Research 架構（Phase 2.3）

### 3.1 有什麼

| 區塊 | 內容 |
|---|---|
| Analytics（34 模組） | 股權投資 / 組合管理 / 衍生品 / 固收 / 公司金融 / 經濟 / quant / 另類 / 回測 / 財報分析 / 技術分析 |
| 第三方 wrapper | pyportfolioopt、skfolio、riskfolio、quantstats、statsmodels、gluonts、pmdarima、functime、tsmoothie、py_vollib、fortitudo、pypme、vnpy、finrl |
| QuantLib suite | 18 個模組（定價 / 風險 / 隨機 / 波動率 / 固收） |
| 回測 | **6 個 provider**：BT、VectorBT、Backtesting.py、FastTrade、Zipline、Fincept 自有 |
| AI Quant Lab | Qlib 14 模組（宣稱 100% 覆蓋）+ RDAgent 4 模組 |
| Node editor | 視覺化 workflow DAG + 排程 + MCP 工具節點 |

### 3.2 回測 provider 的整合流程（值得學的部分）

`docs/backtesting-provider-process.md` 是一份**六步驟、必須照順序、一次只做一個 provider** 的檢查表。核心決定：

> **Python is now the single source of truth.** `default_strategies()` and `all_indicators()` return `{}`. On provider switch, C++ calls `load_strategies()`, `get_indicators`, `load_command_options()`. All combos are populated dynamically.

C++ 端保留的 fallback 清單被定義為「離線／錯誤狀態用，保持正確但**保持最小**」。

→ 這與 MarketMamba `v62.py` 的 `FALLBACK_ARMS` 設計**完全一致**（「只放主線一個——放多了就等於偷偷把重複的表又寫回來」）。**兩邊各自獨立想到同一件事，這條可以視為已驗證的模式。**

Step 3 的五項交叉檢查（commands / strategy IDs / categories / indicator IDs / **param 名稱逐字對得上**）是在防「兩份表各自維護」——MarketMamba 已經因為同型問題出過 bug（scanner 進場條件 vs sim_engine 進場分數）。

### 3.3 AI Quant Lab 的定位（要看清楚）

Qlib + RDAgent 的組合是「**自動化研究迴圈**」：LLM 生假設 → 造因子 → 訓練 → 回測 → 組合最佳化 → 評估 → 存進 knowledge base → 再生假設。

**但 README 裡完全沒有出現這些字**：purge、embargo、walk-forward 的隔離天數、run-to-run 變異、多重檢定校正、雜訊底線。
「Evaluation Metrics」只列 Sharpe / Sortino / Calmar / 勝率 / 換手。

→ **這是 Fincept 相對於 MarketMamba 最明顯的弱項。** 一個「自動生成假設 → 自動回測 → 自動排名」的迴圈，在**沒有隔離與多重檢定紀律**的情況下，就是一台大規模製造假發現的機器。

MarketMamba 的 `06 研究紀錄/00 研究總覽.md` 引用紀律五條（雜訊底線 ±2.68pp、decile > Top50 年化、窗長不可跨比、跑前定死判讀規則、正確性修正不套效益門檻）——**Fincept 沒有任何對應物**。

**這一點必須明確寫下來：在研究方法論上，MarketMamba 領先 Fincept，不是落後。**

### 3.4 三級分類

#### Core capability — MarketMamba 現在真的需要

| 項目 | 為什麼 |
|---|---|
| **回測 provider 的「單一真相 + 最小 fallback」模式** | 已經在 `v62_arms.json` 用了，值得推廣到 df_kelly / scanner 那條線 |
| **`normalize_raw()` 純函式版供測試** | MarketMamba 的正規化目前綁在 fetch 裡，難測 |
| **`raw` 與 `normalized` 同筆保存** | 08-08 的 15% 資料遺失、08-10 的欄名撞名，都是「回頭查不到原始樣貌」放大了診斷成本 |

#### Useful later — 現在不需要，之後可能有價值

| 項目 | 觸發條件 |
|---|---|
| 組合最佳化庫（pyportfolioopt / skfolio） | 等組合建構層從等權走向風險模型時 |
| QuantStats 式標準績效報告 | 等前瞻紀錄累積 > 一季，需要對外呈現時 |
| 多 provider 回測抽象 | 只有在要拿第三方引擎交叉驗證 `portfolio_lab` 時才划算 |
| Node editor / workflow DAG | 等每日管線超過 6 步、且需要非工程師改動時 |

#### Over-engineering — 現階段不值得

| 項目 | 為什麼不 |
|---|---|
| Qlib 全套 + RDAgent | MarketMamba 已有自己的特徵協定、標籤協定、purge 紀律、八模型定稿表。導入 Qlib 等於**重來一次資料協定**，而它帶來的自動化正好是最需要紀律把關的那部分 |
| RL 交易（PPO/DQN/SAC/TD3） | 單一市場、日頻、每 20 日再平衡的問題上，RL 的樣本效率是災難 |
| HFT / 訂單簿 / 造市 | 與短線日頻定位無關 |
| Meta-learning / AutoML | 在 ±6pp 雜訊底線之下自動選模型 = 自動過擬合 |
| Online learning / drift detection | 有意義，但要先有前瞻紀錄才能定義 drift |
| 18 個 QuantLib 模組 | 台股現貨選股用不到債券 / 隨機波動率定價 |

---

## 4. AI Agent 架構（Phase 3）

### 4.1 兩套完全不同的東西，不可混談

Fincept 裡「agent」這個字指兩件事，架構位置差很遠：

**(A) Persona agents（37 個，人格模擬）**
`scripts/agents/` 底下：19 個地緣政治（Grand Chessboard / Prisoners of Geography / World Order）、8 個對沖基金（Bridgewater / Citadel / RenTech / Two Sigma / …）、Buffett / Graham 等投資人、經濟分析。

實作方式：**同一個框架 + 不同的 config JSON + 不同的 system prompt**。
證據：`finagent_core/configs/*.json` 共 16 支、`persona_registry.py`、`persona_runtime.py`、以及 `_tools/apply_geopolitics_batch{1..4}.py` 這種「批次生成 agent」的腳本。

→ **這 37 個不是 37 個工程系統，是 37 筆設定資料。** 邊際成本接近零，所以數量才會長成這樣。

**(B) Agentic runtime（真正的工程）**
`finagent_core/agentic/` 底下才是有內容的部分：

| 檔案 | 職責 |
|---|---|
| `runner.py` | ResumableTaskRunner |
| `task_state.py`（上一層） | SQLite checkpoint：`(thread_id, step, state_blob)` |
| `budget.py` | **四層獨立上限**：tokens / USD / wall-clock / steps，任一破就 `budget_stop` |
| `reflector.py` + `reflexion_store.py` | 反思 / 自我修正（Reflexion） |
| `skill_library.py` | 跨任務技能累積（Voyager） |
| `archival_memory.py` | 長期記憶（MemGPT/Letta） |
| `eval_harness.py` + `eval_cases_example.json` | 評測 |
| `events.py` | 逐步事件串流 |
| `scheduler.py` / `daemon.py` | 排程與常駐 |

另有 `deepagents/`（orchestrator + subagents + backends）與 `rdagents/`（含自己的 `mcp_server.py`）。
**`finagent_core/tests/` 有 8 支測試**，包含 `test_agentic_memory_isolation.py`、`test_isolation_integration.py`——**記憶隔離有測試**，這是成熟訊號。

### 4.2 `budget.py` 值得單獨講

四個 cap 的設計哲學寫在檔頭：

> The cost table is intentionally conservative — **under-bills slightly so we never block at the "$0.999999" boundary.**

Token 計數「best-effort 從 Agno 的 metrics 拿，拿不到就 char/4 估算」。

→ 這是**「算不準的時候往哪邊倒」**的教科書案例，與 MarketMamba 08-09 的 `ann_stderr_pp` 教訓（`None` 必須當成不確定性無限大，不可 `?? 0`）同一個問題、**相反方向的正確答案**：
- 預算守門估不準 → 往**低估**倒（寧可讓它多跑一點，也不要在邊界誤殺）
- 誤差棒算不出 → 往**無限大**倒（寧可標記不可信，也不要看起來精確）

**共同規則：預設值要往「錯了也不會產生錯誤決策」的方向倒，而不是往 0 倒。**

### 4.3 Agent 與金融資料的互動：MCP 工具層

`docs/MCP_TOOLS_GUIDE.md` 是整個 repo 品質最高的文件。核心設計：

```
LlmService（provider-agnostic）
   ↓ ToolDispatcher（多輪迴圈、平行 fan-out）
McpService::execute_*_async     ← 統一入口
   ├─ internal: McpProvider     ← src/mcp/tools/*.cpp（40+ 工具）
   └─ external: McpManager      ← JSON-RPC over stdio
```

五個值得學的設計：

**(1) Schema 是強制的，且驗證在 handler 之前跑**

```cpp
t.input_schema = ToolSchemaBuilder()
    .string("symbol", "Ticker").required().pattern("^[A-Z0-9._-]{1,16}$")
    .string("side", "Order side").required().enums({"buy","sell"})
    .integer("limit", "Max items").default_int(20).between(1, 100)
    .build();
```

> Handlers receive a normalised `QJsonObject` and **don't need defensive `.toString("default")` calls.**

→ **把防禦寫在一個地方，而不是每個 handler 各寫一次。** 這正是 MarketMamba「同一套規則兩份實作」那類 bug 的通用解。

**(2) 同步／非同步的判準是「時間」不是「型態」**

| sync | async |
|---|---|
| registry / cache / DB 查詢（微秒） | Python 腳本執行 |
| 純計算 | HTTP |
| **< 1 ms** | **> 100 ms** |

**(3) 授權分五級 + `is_destructive` 正交**

`None` / `Authenticated` / `Verified` / `Subscribed` / `ExplicitConfirm`，而 `is_destructive = true` **不管授權等級一律跳確認 modal**。

→ **「你有沒有權限」與「這件事可不可逆」是兩個維度。** MarketMamba 目前沒有任何工具授權概念（也還不需要），但這個二維切分在未來加自動交易時是必須的。

**(4) 改名保留 `legacy_aliases`**

```cpp
t.name = "markets.get_quote";
t.legacy_aliases = {"get_quote"};   // 存檔的聊天紀錄 / workflow 仍可解析
```

命中 alias 時**記一筆 deprecation log**。

**(5) 有 meta-tool 讓 LLM 自己探索工具**：`tool.list()` / `tool.list({search:"quote|price"})` / `tool.describe({name})` / `mcp.health()`。
→ 工具多到一定程度時，**不要把 40 個 schema 全塞進 context，讓模型自己查**。

### 4.4 正面回答：「Fincept 的 agent specialization 到底在解決什麼問題？」

拆成兩半，答案完全不同：

**Persona agents（37 個）解決的是「產品差異化」，不是「工程問題」。**
證據：它們共用同一個 runtime、同一組工具、同一個記憶體系統，差別**只在 config JSON 與 prompt**。README 的行銷語言（AUM 數字、「Buffett 的護城河哲學」）也指向這個定位。roadmap 寫「Q2 2026：50+ AI agents」——**把 agent 數量當成 KPI 本身就是它是行銷指標的證據。**

真正的工程問題只有一個，而且被解在別的地方：**長時間任務的耐久性**。
`docs/agentic-research/README.md` 的自述極為誠實：

> **Reframed scope:** Original framing of "build true agentic from scratch" was wrong. Real work = 2–3 days for MVP... The work is **wiring, exposure, and adding the reflection + budget layers.**
> What we have: ≈80% of agentic primitives already in place.

→ **他們自己做完研究之後，把「多 agent」這個問題重新框定成「已有的東西沒接上線」。** 而且引用的文獻裡包含 Cognition 的 *Don't Build Multi-Agents*。

**這是整份研究對 MarketMamba 最有價值的一句話：連一個以「37 個 agent」為賣點的專案，自己的內部研究結論都是「問題不在 agent 數量，在耐久性、預算、反思、可觀測性」。**

### 4.5 對照 MarketMamba 的 AI 使用方式

兩者在**不同軸**上：

| | Fincept | MarketMamba |
|---|---|---|
| AI 在產品裡 | 37 個 persona agent 是**功能** | LLM 只做每日市場報告（`report_generator.py`）＋新聞分析 |
| AI 在開發裡 | 不可見（`CLAUDE.md`、`plans/` 被 gitignore） | **這才是主戰場**——CLAUDE.md 40KB + obsidian 三層記憶 + 決策紀錄 |

→ **Fincept 把 AI 當賣點，MarketMamba 把 AI 當同事。** 這兩件事幾乎不重疊，所以「Fincept 有 37 個 agent」對 MarketMamba 的直接可借鑑度**接近零**；有價值的是它 agentic runtime 那七個模組所代表的**問題清單**（預算、反思、記憶隔離、評測、事件串流、checkpoint、排程）。

---

## 5. UI / UX 架構（Phase 4）

### 5.1 導覽與資訊架構

三層並存，這是 Bloomberg 式終端的標準做法：

1. **F-key bar**（功能鍵列）——專家的肌肉記憶路徑
2. **Command palette**（`ui/command/`）——打字跳任意畫面／執行任意動作
3. **Dock manager**（ADS）——多視窗、多面板、可任意拆併

**`DockScreenRouter` 用 lazy factory 註冊 54 個畫面**（`WindowFrame_Setup.cpp`）——畫面只有被開啟才建構。
但已知弱點寫得很直白：**「Screens never unload — 100–250 MB resident per window」**。lazy 只解決了啟動時間，沒解決長時間駐留。

### 5.2 畫面契約

> A screen is a `QWidget` subclass that renders state and accepts user input. It **does not** call `HttpClient` directly, **does not** own caches, and **does not** contain business logic (deduplication, deviation detection, risk calculation, etc.).

而且他們**誠實記錄違規現況**：6 個畫面違反 no-HTTP、約 3 個（News、Derivatives）帶了該在 service 的領域邏輯，全部標進 refactor plan。

→ **契約 + 已知違規清單 + 負責的 phase。** 這種「規則 / 現況 / 誰負責收」三欄式的呈現，比只寫規則有用得多。

### 5.3 狀態與版面

- `IStatefulScreen` + `ScreenStateManager`：畫面自願實作才有持久化
- `TabSessionStore`：每個 tab 的 UI 狀態
- `SessionManager`：frame / panel / 最後畫面
- `workspace/`：多視窗版面

**分層很清楚：暫態 UI 狀態（tab）／畫面狀態（screen）／版面狀態（workspace）三者各有其家。**

### 5.4 主題與視覺

- `ui/theme/Theme.h` 的 token（Obsidian design system）
- 反面規則：**「Don't bake colors; use tokens.」**
- 反面規則：**「不要在 widget 自己的 event handler 裡呼叫 `qApp->setStyleSheet(...)` — Wayland 會 crash。」** 主題變更要合併並用 `Qt::QueuedConnection` 分派
- arch-ratchet 用 CI 盯著 inline `setStyleSheet` 的數量只能降

`DESIGN_SYSTEM.md` 被 gitignore，所以設計 token 的實際內容看不到。

### 5.5 圖表與表格

`ui/charts/`（含 `layers/`）、`ui/tables/`、`ui/markdown/` 都是自建元件庫，基於 Qt6 Charts。
`ui/formatting/` 獨立成一層——**數字格式化是共用關注點，不是每個畫面各寫一次**。

### 5.6 Node editor / workflow

`services/workflow/` 有 11 對檔案，值得注意的是它不只有 executor：

`NodeRegistry` / `WorkflowExecutor` / `ExpressionEngine` / `ParameterProcessor` / `WorkflowCache` / `Extensions`
＋ **`RiskManager` / `ConfirmationService` / `AuditLogger` / `ExecutionHooks`**

→ **後面四個才是重點：一旦 workflow 可以下單，就必須有風控、人工確認、稽核軌跡、執行 hook。** 視覺化編輯器是表面，可稽核的執行才是內容。

### 5.7 正面回答：「MarketMamba 若要從 ML 研究專案走向金融研究平台，哪些 UI/UX pattern 值得採用？」

先講清楚 MarketMamba 的使用者是誰：**使用者本人（每天看、做決策）+ 家人（只看結果）**。不是專業交易員，不是多帳戶操作者。

**值得採用（依價值排序）**

| Pattern | 為什麼對 MarketMamba 成立 |
|---|---|
| **1. 資料新鮮度指示（`age_ms` / 「這個數字幾點算的」）** | MarketMamba 的失敗史幾乎全是「靜默用舊資料」。這是 UI 層的直接對策，成本極低 |
| **2. 「規則 / 現況 / 誰負責」三欄式契約呈現** | 已經在 CLAUDE.md 用了，但沒進 UI。前端的 tier / family / caveats 就是這個模式的雛形 |
| **3. Command palette（打字跳頁）** | 19 個組合 × 多個分頁，選單一定爆。搜尋框比階層選單擴展性好太多 |
| **4. `ui/formatting/` 式的共用格式層** | 目前百分比／pp／±誤差棒的格式散在各頁，正是「寫死比較基準」踩六次的土壤 |
| **5. 版面狀態三層分家（tab / page / workspace）** | 等前端頁面再長就會需要 |

**不值得採用**

| Pattern | 為什麼不 |
|---|---|
| ADS dock manager / 多視窗拆併 | 單人 + 家人瀏覽，沒有多螢幕工作站需求 |
| F-key bar | 需要肌肉記憶的使用頻率門檻，MarketMamba 一天用一次 |
| 54 個畫面的 lazy router | 頁面數量差一個量級，React Router 的 lazy 已足夠 |
| Node editor | 每日管線 6 步且只有一個人改，YAML/程式碼比拖拉節點好維護 |
| 原生桌面 | 家人要能用手機／瀏覽器看 → web 是對的選擇 |

---

## 6. 測試與文件（補充觀察）

### 6.1 測試

- C++：`tests/mcp/` 等目錄（MCP guide 提到「logic 非 trivial 就寫 unit test」）
- Python：`finagent_core/tests/` 8 支（含記憶隔離、persona registry、real config smoke test）
- CI：`build-cpp.yml` / `lint.yml` / `pr-gate.yml` / **`arch-ratchet.yml`** / `test-setup.yml` / `release.yml`

**覆蓋率不高，但「該有防線的地方有防線」**：架構規則有棘輪、agent 記憶隔離有測試、schema 驗證是集中的。

### 6.2 文件

三個值得學的文件習慣：

1. **`(current)` / `(target)` 明確標記**——ARCHITECTURE.md 開頭就教你怎麼讀這份文件
2. **「Known weaknesses」表格帶 owner phase**——13 條弱點，每條有影響與負責的 refactor phase。**不藏短處**
3. **「Patterns / Anti-patterns」對照列表**——正例與反例並列，反例還附原因（Wayland 會 crash）

**ADR**：`docs/adr/README.md` 存在，規則是「反轉或細化 ARCHITECTURE.md 任一節時，加一份 ADR」。但 `docs/adr/` 底下**只有 README**——**規則有了，實踐沒跟上**。這一點要如實記下：ADR 制度本身值得學，Fincept 的執行狀況不值得學。

---

## 7. 本研究的證據等級標示

| 結論 | 證據等級 |
|---|---|
| DataHub / TopicPolicy / Producer 的設計 | **直接讀原始碼**（DataHub.h、TopicPolicy.h、Producer.h） |
| MCP 工具層設計 | **直接讀文件**（MCP_TOOLS_GUIDE.md，內容與 `src/mcp/` 檔案結構一致） |
| DataNormalizationService | **直接讀 header** |
| arch-ratchet | **直接讀 workflow yaml** |
| budget.py 的四層 cap | **直接讀原始碼** |
| 37 個 persona 是 config 不是工程 | **從檔案結構反推**（configs/*.json + persona_registry + 批次生成腳本），未讀 runtime 全部程式碼 |
| 「Fincept 缺乏研究方法論紀律」 | **從缺席反推**（README + ai_quant_lab README + 回測文件均無 purge / 多重檢定 / 雜訊底線字樣）。**未窮舉 1,400 支 Python**，可能有但未被文件化 |
| 他們的 Claude Code workflow | **不可知**（CLAUDE.md / plans/ / agents.md 全被 gitignore）。相關推論一律標明 |
| C++ 執行期實際行為 | **未編譯、未執行**。全部來自原始碼與文件閱讀 |

---

## 8. 一句話總結

> **Fincept 的價值不在它做了什麼，而在它把「資料源的多樣性」壓縮成一條 topic 字串這件事上做對了。**
> 其餘的 54 個畫面、16 家券商、37 個 agent、100+ connector，都是那條骨幹上的量產品——**量產品不可移植，骨幹可以。**
> 而在 MarketMamba 最在乎的那件事（研究結論的可信度）上，Fincept 沒有任何可借鑑之處，反而是 MarketMamba 領先。
