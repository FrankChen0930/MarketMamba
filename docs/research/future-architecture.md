# MarketMamba — 6–12 個月架構方向

> 撰寫日期：2026-08-11
> 前置文件：`fincept-terminal-analysis.md`（Fincept 架構分析）、`market-mamba-gap-analysis.md`（差距與優先級）
> 性質：**方向建議，不是承諾。** 每一階段都可以在該階段的入口重新評估要不要走下一步。

---

## 0. 這份文件的三條自我約束

1. **不為了模仿 Fincept 而擴張。** 每一項變更都必須指得出 MarketMamba 自己歷史上的一個具體問題。
2. **不動運作中的東西，除非有回歸測試護著。** V6.1 雖已不是紅線，但 V6.2 剛上線、08-17 才正式起跑——**這段期間的第一原則是別弄壞。**
3. **每階段都要能停下來。** 如果第 2 階段做完覺得夠了，第 3 階段不做也不會留下半成品。

---

## 1. 起點：現在的 MarketMamba（2026-08-11）

```
┌──────────────────────────────────────────────────────────┐
│ 研究層（人 + Claude Code + Colab）                        │
│   experimental/ 消融、portfolio_lab、obsidian 三層記憶     │
│   紀律：purge / 雜訊底線 / decile / 跑前定死判準           │
│   ⚠ 全部人工執行、人工判讀                                │
├──────────────────────────────────────────────────────────┤
│ 資料層  fetcher.py (215KB, 14 源) → parquet               │
│   ⚠ 寫入語意隱含、分散；正規化綁在 I/O 上不可測            │
├──────────────────────────────────────────────────────────┤
│ 模型層  Mamba+GAT (保護)｜訓練 Colab｜推論 WSL2            │
├──────────────────────────────────────────────────────────┤
│ 組合層  19 組合 × 11 分數 狀態機 + jsonl + 前瞻績效         │
│   ✅ 這一層是全系統最成熟的部分                            │
├──────────────────────────────────────────────────────────┤
│ 發布層  git push → GitHub raw                             │
│   ⚠ 契約隱含（URL 字串替換）、無 manifest（v62 除外）      │
├──────────────────────────────────────────────────────────┤
│ 服務層  FastAPI 10 router，各自抄一份 TTL 快取             │
├──────────────────────────────────────────────────────────┤
│ 呈現層  React 25 頁 + PersonalOS 鏡像                     │
└──────────────────────────────────────────────────────────┘
```

**一句話**：研究層與組合層很強，**資料契約層與驗證層幾乎不存在**，發布層是隱含契約。

---

## 2. 演進路徑總覽

```
現在（2026-08）
   │  ── 護欄期：不加功能，只把已有的紀律變成機器 ──
   ▼
Phase 1  可驗證的 MarketMamba          （約 2–4 週，與 08-17 起跑並行）
   │
   ▼
Phase 2  有資料契約的 MarketMamba      （約 4–8 週，V6.1 退役期間）
   │
   ▼
Phase 3  可稽核的研究平台              （約 3–6 個月，需要一季前瞻紀錄先到位）
   │
   ▼
（可選）Phase 4  對外可呈現的研究系統
```

**刻意不設終點為「financial research platform」**——那是 Fincept 那條路。MarketMamba 的終點應該是「**一套結論可稽核的台股研究系統**」，見 gap analysis Part E §7。

---

## Phase 1 — 可驗證的 MarketMamba

**時間**：2026-08 中～09 上旬，**與 08-17 正式起跑並行**（純附加，不碰起跑路徑）

### Objective

> 把已經存在於 CLAUDE.md 與使用者腦中的驗收紀律，變成一條可重複執行的指令。

**不新增任何功能。** 這一階段的產出全部是護欄。

### Architecture changes

```
新增（全部純附加）：
  tests/
    ├── golden/                    ← 固定樣本與已知數字
    │     ├── features_20d_50stk.parquet
    │     ├── portfolio_lab_582d.json      （37.28% / 82.7%）
    │     └── v62_step_replay.json         （與上一條 0.000pp）
    ├── test_feature_regression.py
    ├── test_portfolio_consistency.py
    └── test_parse_functions.py    ← 用存下來的真實 HTTP 回應樣本，不打網路
  scripts/ratchet.py               ← 三條只能降的計數器
  Makefile 或 check.sh             ← 一條指令跑完
```

### New capabilities

| 能力 | 直接對應的歷史事故 |
|---|---|
| `make check` 一條指令跑完全部回歸 | 「逐位元相同」的驗收目前要人工記得跑 |
| 特徵矩陣黃金檔 | `fundamentals_v2` 旗標、除權息切換 |
| `replay()` vs `step()` 精確一致鎖定 | 2026-08-08 的 0.18pp 分歧 |
| 棘輪：`to_parquet` 無 `index=False` 計數 = 0 | 2026-08-10 三次執行全掛 |
| 棘輪：硬編比較基準字面值計數 | 2026-08-09 一天踩六個 |
| 棘輪：誤差欄位上的 `?? 0` / `or 0` 計數 | 2026-08-09 `ann_stderr_pp` 的 null→0 |
| 純函式 `parse_*` 可用固定樣本測 | FinMind 標反券賣券買、金融保險業版面漏掉 |

### Migration cost

**低。** 全部是新檔案，沒有一行 production 程式碼被修改。
唯一的實質工作是**把幾個 `fetch_*` 裡的 parse 段抽成純函式**——而抽出的函式必須通過「與抽出前逐位元相同」的驗收（這正是 MM 已經在用的標準）。

### Risks

| 風險 | 緩解 |
|---|---|
| 黃金檔在合法的正確性修正時會變動 | 更新黃金檔必須是**獨立 commit**，message 要寫「為什麼這個數字該變」。借用 arch-ratchet 的紀律：baseline 只能往下調 |
| 棘輪一開始就 fail | 先跑一次記錄現況當 baseline（`to_parquet` 那條應該已經是 0，直接鎖住） |
| 抽 parse 函式動到 production | 抽完的驗收是逐位元相同；且**不要一次抽 14 個源，先抽 2–3 個踩過雷的** |

### 這階段**不要**做的事

- ❌ 不要建 CI（GitHub Actions）——先在本機能跑就有 90% 的價值；CI 是 Phase 2 的事
- ❌ 不要追求覆蓋率——只鎖「已經出過事」與「數字是產品本身」的地方
- ❌ 不要重構 `fetcher.py`
- ❌ 不要碰前端

---

## Phase 2 — 有資料契約的 MarketMamba

**時間**：V6.1 + 雙模型 + scanner 退役期間（V6.2 連續跑順之後）

### Objective

> 把「資料源的行為規則」從分散的程式碼收斂成宣告式的純資料；把「結果發布」從隱含契約變成明確契約。

這階段對應 gap analysis 的 **P0-2 + P0-3 + P1-1 + P1-2 + P1-3**。

### Architecture changes

```
資料層
  V6/marketmamba/data/
    ├── source_policy.py     ← SourcePolicy dataclass（純資料）
    ├── writer.py            ← 唯一的 parquet 寫入路徑
    │     寫入前：欄名 / 型別逐欄 / 鍵集合守門
    │     index=False 強制｜縮小需 allow_shrink=True｜暫存→驗過→os.replace
    └── fetcher.py           ← 逐源遷移到 writer（opt-in，不一次改完）

發布層
  V6/marketmamba/deploy/publisher.py
    └── 每次發布同時寫 results_manifest.json
          檔名 → {generated_at, data_date, rows, pipeline_version}

服務層
  app/backend/results_store.py   ← 唯一的 GitHub raw + TTL + lock + last-known-good
    └── router 逐一遷移（v62 先，signals 最後）

呈現層
  src/lib/format.js       ← 唯一的 %/pp/±誤差棒/tier 顏色實作
  <FreshnessBadge/>       ← 讀 manifest 的 generated_at / data_date

退役（減法）
  sim_engine v1/v2、scanner_engine、condition_analyzer、mock_data.py
```

### New capabilities

- **「今天沒抓到 ≠ 今天沒交易」寫一次，14 個源受益**
- **寫入前守門**（08-08 的教訓是「寫完再檢查只能發現，不能防止」）
- **前端能顯示資料是哪一天算的** —— Fincept 那句「顯示沒有時間標記的數字是主動誤導」
- **誤差棒格式只有一個實作**，`null → "±?"` 且強制標 `*` 不可能再被 `?? 0` 繞過
- **`incomparable` 與 `inferior` 顏色不同**由型別保證，不靠記得

### Migration cost

**中。** 動到 production 的資料寫入路徑與線上 backend。
但每一步都是可逐項驗證的：

- 每遷一個資料源 → 跑鍵集合比對（舊檔 ⊆ 新檔、既有列 `max|Δ| = 0`）
- 每遷一支 router → 比對回傳 payload 逐欄相同

**這正是 08-08 回補與 08-09 重評分用過的驗收方法**，不是新流程。

### Risks

| 風險 | 緩解 |
|---|---|
| 動 `fetcher.py` 弄壞每日管線 | **必須有 Phase 1 的回歸測試**；opt-in 遷移；先遷已經出過事的源（prices） |
| 動 backend 弄壞線上 dashboard | `v62.py` 先遷（試跑期）→ 觀察一週 → 再遷 `signals.py` |
| 退役刪錯東西 | 退役清單先寫進 CLAUDE.md 並確認 V6.1 真的停了 |

### 這階段**不要**做的事

- ❌ 不要引入 SQLite / 任何資料庫——git 現在提供的版本控制與稽核性是資產
- ❌ 不要為 `fetcher.py` 做完整的 provider 抽象——只包**寫入 + 新鮮度**，抓取邏輯留在原地
- ❌ 不要重寫前端版面（使用者已明確說等看得見全貌再想）
- ❌ 不要加新資料源

---

## Phase 3 — 可稽核的研究平台

**時間**：需要**至少一季前瞻紀錄**（08-17 起算 ≈ 2026-11 之後）才有意義

### 為什麼前置條件是「一季」

因為 gap analysis 引用的那條算術：年化的標準誤 ≈ 252 × s_daily / √n → n=20 是 ±68pp、n=60 是 ±39pp。
**在第一年之內，年化排不出名次。** 所以這一階段的目標**不能**是「用實戰紀錄挑最好的組合」，而是「讓實戰紀錄本身變得可稽核」。

### Objective

> 讓任何一個呈現在 dashboard 上的數字，都能被追到它是從哪些輸入、經過哪個版本的程式、在什麼時間算出來的。

### Architecture changes

```
研究層
  ├── 引用紀律機械化（P1-5）
  │     ・年化差 < 6pp 卻標了優劣分級 → fail
  │     ・n_days < 60 卻顯示年化而不帶 stderr → fail
  │     ・跨訓練輪（隔離天數不同）被並列 → fail
  └── 實驗註冊表：每個 arm 的 (checkpoint, 協定版本, purge 天數, seed, panel 版本)
        → 「不可並列」由資料判定，不靠人記得

發布層
  └── manifest 擴充成 provenance：
        結果檔 → {輸入 raw 檔的 hash, feature 協定版本, checkpoint id,
                  程式 commit, 執行時間, 資料完整性}

服務層
  └── /api/provenance/{artifact} —— 任一數字可回溯

呈現層
  ├── Command palette（P1-4）
  └── 每個數字可點開看「它是怎麼來的」
```

### New capabilities

- **「這個 37.3% 是哪個 checkpoint、哪個 panel、哪次 purge 算的」變成一次查詢**，而不是翻 obsidian
- **`incomparable` 由資料判定**——目前它是人工標的，08-09 就標錯過一次（把更好的 `head10d_f20` 標成 inferior）
- **CI 擋住違反引用紀律的呈現**

### Migration cost

**中—高。** 但這階段的每一項都可以**單獨**做，沒有互相依賴。可以只做實驗註冊表，不做 provenance API。

### Risks

| 風險 | 緩解 |
|---|---|
| **過度工程化的最大風險點就在這一階段** | 判準：**每一項都要指得出一次真實的誤讀事件**。指不出來就不做 |
| 引用紀律機械化誤擋合法情況 | 一律做成**警告 + 需明確標註豁免**，不做硬 fail（arch-ratchet 的教訓） |

### 這階段**不要**做的事

- ❌ 不要開始做「自動化研究迴圈」（RDAgent 式）——那是假發現產生器
- ❌ 不要為了 provenance 改資料格式——manifest 帶 hash 就夠
- ❌ 不要引入實驗追蹤平台（MLflow / W&B）——MM 的實驗量級用 JSON + git 就夠，引入平台等於多一個要維護的系統

---

## Phase 4（可選）— 對外可呈現的研究系統

**只在使用者真的想對外呈現時才進入。** 觸發條件是需求，不是時間。

**如果進入，賣點應該是**：「每個數字都能追到它是怎麼來的」，不是「有幾個模型 / 幾個指標」。
**如果進入，仍然不做的**：多市場、多資產、券商整合、即時串流、多使用者。

---

## 3. 三張「現在不要建」的清單

| 現在不要建 | 什麼時候重新評估 |
|---|---|
| 資料庫（SQLite / Postgres） | 只有在 git repo 因結果檔膨脹到無法接受時 |
| Service 層 / bounded context / 依賴強制 | 只有在 `marketmamba/` 超過 ~300 檔或有第二個開發者時 |
| 多 provider 抽象（回測 / LLM / 券商） | 只有在真的有第 2 個實例時。**1 個實例的抽象是純虧損** |
| 自動下單路徑 | 只有在一季以上前瞻紀錄證明規格站得住之後，且必須同時建風控前置 + 人工確認 + 稽核軌跡三件套 |
| Node editor / 視覺化 workflow | 只有在有非工程師要改管線時 |
| 常駐推論 worker | 只有在每日總時長變成瓶頸時（目前 9–18 分鐘，不是） |
| CI（GitHub Actions） | Phase 2。本機 `make check` 先跑順再說 |
| 任何新資料源 | 只有在有假設要驗證時，不為了覆蓋率 |

---

## 4. Claude Code / Agent Workflow（Phase 9）

### 4.1 先回答：Fincept 能給什麼啟發？

**證據限制先講清楚**：Fincept 的 `CLAUDE.md`、`plans/`、`agents.md` **全被 gitignore**，他們實際的 Claude Code workflow **看不到**。以下全部是從**可見證據**反推的。

可見證據有三項，每項都給了一個真實的啟發：

**(a) `docs/agentic-research/README.md` 的自我修正**

> Original framing of "build true agentic from scratch" was wrong. Real work = 2–3 days for MVP...
> The work is **wiring, exposure, and adding the reflection + budget layers.**

一個以「37 個 agent」為賣點的專案，自己做完研究後的結論是「**問題不在 agent 數量，在耐久性、預算、反思、可觀測性**」，而且引用了 Cognition 的 *Don't Build Multi-Agents*。

→ **啟發 1：不要用 agent 數量衡量進展。**

**(b) `finagent_core/agentic/` 的模組清單**

`budget` / `reflector` / `skill_library` / `archival_memory` / `eval_harness` / `events` / `task_state`。

→ **啟發 2：長時間 AI 任務真正需要的東西是這七項，而不是更多角色。** 對 MarketMamba 而言，其中三項可以直接對應：
- `budget`（四層 cap：token / $ / 時間 / 步數）→ 對應 MM 的「背景任務會被砍 → 長工作分段」紀律
- `eval_harness` → 對應 Phase 1 的回歸測試
- `task_state`（SQLite checkpoint）→ 對應 MM 的 `v62_state_*.json` 與 `sim_state.json`（**MM 已經在做對的事**）

**(c) `arch-ratchet.yml`**

> A ratchet people keep beats a gate they remove.

→ **啟發 3：AI 協作最大的槓桿不是給 agent 更多能力，是給它更多可自我驗證的機制。** 一個能自己跑 `make check` 的 Claude Code，比三個互相審查的 agent 有用得多。

### 4.2 MarketMamba 真的需要多個 specialized agent 嗎？

**不需要。而且理由是可量化的。**

三條：

1. **每個 subagent 都是冷啟動。** MarketMamba 的關鍵脈絡（雜訊底線 ±6pp、purge 30 vs 40 不可跨比、56 vs 59 維 config 分裂、`prices_raw` 的兩次事故）在 CLAUDE.md 有 40 KB，在 obsidian 有 14 篇。**一個沒有讀過這些的 subagent，會給出看起來合理但違反紀律的建議**——例如把 head10d_f20 標成 inferior（這個錯誤主 agent 自己就犯過一次）。
2. **MarketMamba 的工作極少是可平行的獨立子任務。** 資料 → 特徵 → 模型 → 組合 → 發布是一條嚴格的因果鏈，且每一環的驗收都要拿另一環的數字比對。**平行 fan-out 的前提（任務獨立）在這裡不成立。**
3. **使用者的工作模式是「計畫先行、確認後執行」。** 多 agent 平行執行與這個模式直接衝突——使用者無法逐一確認三條同時進行的線。

### 4.3 最小可行的 agent 架構

**三個角色，而且其中兩個大部分時候由主 session 兼任。**

```
                  ┌────────────────────────────────┐
                  │  主 session（預設，做 90% 的事）  │
                  │  讀 CLAUDE.md + obsidian        │
                  │  計畫 → 確認 → 執行 → 記錄        │
                  └───────────┬────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌────────────────┐
│ Explore       │    │ （主 session）   │    │ Verifier       │
│ 唯讀、扇出搜尋  │    │  實作            │    │ 數值對抗        │
│ 只在「不知道東西 │    │                 │    │ 只在「數字太漂亮」│
│ 在哪」時派      │    │                 │    │ 或改動核心時派   │
└───────────────┘    └────────────────┘    └────────────────┘
```

**只有兩種情況值得開 subagent**：

| 情況 | 角色 | 為什麼值得 |
|---|---|---|
| 「這個東西在 130 個檔案裡的哪裡？」 | **Explore（唯讀）** | 扇出搜尋會吃掉大量主 session 的 context，而結論只有一句話。這是 subagent 的**典型正確用法** |
| 「這個結果好得可疑，幫我找它為什麼可能是錯的」 | **Verifier（對抗）** | 主 session 剛做完這件事，**有確認偏誤**。冷啟動在這裡是優點不是缺點——它不知道你希望答案是什麼 |

**不值得拆成 agent 的**：

- 寫程式（主 session 有全部脈絡）
- 判讀實驗結果（需要引用紀律，冷啟動會誤讀）
- 更新文件（需要知道這次改了什麼）
- 「Reviewer agent」做通用 code review（MM 的 bug 幾乎全是**數值語意**問題，通用 review 抓不到 `index=False` 或 `?? 0`——那要靠 P0-1 的棘輪）

### 4.4 建議的工作流

```
Research  ──→  Spec  ──→  Plan  ──→  Implementation  ──→  Verify  ──→  Record
   │            │          │              │                 │            │
   │            │          │              │                 │            └─ CLAUDE.md Current Status
   │            │          │              │                 │               + obsidian 決策紀錄／方法論教訓
   │            │          │              │                 └─ make check（Phase 1 之後）
   │            │          │              │                    + 該次改動專屬的驗收
   │            │          │              └─ 主 session，一次改一個變因
   │            │          └─ **使用者確認**（互動規則 2，不可跳過）
   │            └─ 判讀規則／驗收標準**跑之前先定死**
   └─ graphify query（專案規則）→ 必要時 Explore agent
```

**兩個 MarketMamba 特有的環節，一般 workflow 沒有**：

1. **Spec 階段必須先定死判讀規則。** 這是 MM 已有的紀律（「看到數字才選規則就沒意義」），應該明確寫進 workflow 而不是靠記得。
2. **Record 階段是必要環節不是可選的。** MM 的三層記憶制度是整個專案最大的資產；跳過 Record 等於讓下一個 session 冷啟動。

### 4.5 哪些資訊該放 project documentation、哪些該放 agent memory

**判準：會不會影響一個未來的決定？**

| 放哪 | 內容 | 現況 |
|---|---|---|
| **CLAUDE.md（第一層）** | 動手前一定會用到的規則、現在要做什麼、每天用到的操作紀律 | ✅ 已在做 |
| **obsidian（第二層）** | 結論、教訓、決策的理由與拍板者、實驗史 | ✅ 已在做 |
| **docs（第三層）** | 完整數字、逐 epoch log、規格書 | ✅ 已在做 |
| **程式碼註解** | **為什麼這樣寫**（尤其是「為什麼不用直覺的做法」） | ✅ 已在做，`v62.py` 的註解是範例 |
| **測試／棘輪** | **不可違反的規則**——比文件強，因為它會擋人 | ❌ Phase 1 要補 |
| **agent memory** | 只放「這個使用者的工作偏好」（計畫先行、指定檔案 git add、繁中回應） | ✅ 已在做 |

**明確不該放 agent memory 的**：任何數字、任何檔案路徑、任何實驗結論。
理由：agent memory 反映的是寫入當下的狀態，**而 MM 的數字每次重跑都會過時**——這正是「比較基準不可寫死」踩過六次的同一個陷阱，換一個載體而已。

### 4.6 各階段最適合的 Skill

| 階段 | Skill | 說明 |
|---|---|---|
| Research | `graphify`（專案規則強制）、`firecrawl-search`（外部文獻） | graphify 先跑，raw 檔後讀 |
| Spec | `brainstorming` | 只在「要做什麼」還不明確時；規格清楚就跳過 |
| Plan | `writing-plans` / `planning-and-task-breakdown` | **但最後仍要走互動規則 2：列計畫等使用者確認** |
| Implementation | `test-driven-development` | 對 MM 特別合適——因為驗收標準（逐位元相同、鍵集合、max\|Δ\|=0）**本來就是先定義好的** |
| Verify | 自建 `make check` + `verification-before-completion` | Phase 1 的產出就是這一格 |
| Record | 無現成 skill；靠 CLAUDE.md 互動規則 8 | 可考慮寫一個專案自有 skill 把三層記憶的更新流程固化 |

**唯一建議新增的 skill**：一支 `mm-verify`，把「這次改動要跑哪些驗收」標準化——包含 `make check`、鍵集合比對、以及「這個改動有沒有可能讓某個數字靜默改變」的檢查清單。**這比任何新 agent 都有用。**

---

## 5. 收束：三個階段各自的「成功長什麼樣」

| 階段 | 成功的定義 |
|---|---|
| **Phase 1** | 改完 `feature_engineer.py` 之後，**一條指令**就能知道有沒有弄壞東西；`to_parquet` 少 `index=False` 不可能再進 main |
| **Phase 2** | 新增一個資料源時，「今天沒抓到 ≠ 今天沒交易」**不需要重新想一次**；dashboard 上每個數字旁邊都有它是哪天算的 |
| **Phase 3** | 任何一個呈現的數字，**點一下就知道它是從哪來的**；違反引用紀律的呈現進不了 main |

三個階段沒有一個是「加了什麼新功能」。**這是刻意的**——MarketMamba 現在缺的不是功能。
