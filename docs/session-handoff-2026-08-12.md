# Session Handoff — 2026-08-12（Advisor operational handoff）

> **給下一個 Advisor session 的接手文件。** 讀完這份 + `CLAUDE.md` + `obsidian_note/01 系統現況/現況整理.md`
> 就能安全接手，**不需要讀上一段對話**。
>
> **標記約定**
> `FACT` = 已被工具實際查證　`DECISION` = 使用者已拍板
> `PENDING` = 仍需使用者決定　`OUT-OF-SCOPE` = 已明確排除，不要重開
>
> ⚠️ **本檔不含任何 token / credential 本體**，也不得在後續被寫入任何憑證值。
>
> **本版取代同日稍早的所有版本**（v1 停在「Builder PLAN_ONLY、等待批准」；
> v2 停在「Verifier PASS，但 `.env` 尚未輪替」。**本版為 session closure 版**。）

---

## 0. 你的第一個動作

**沒有阻塞中的事。credential remediation 這一輪已 CLOSED。**

### 0.1 本輪 CLOSURE 摘要（一眼版）

| # | 項目 | 狀態 |
|---|---|---|
| 1 | FinMind token 在 FinMind 後台輪替 | ✅ **RESOLVED**（使用者完成） |
| 2 | 三個 active 檔案的 credential 字面值移除 | ✅ **RESOLVED** |
| 3 | Builder 實作 + 兩輪冷啟動 independent Verifier | ✅ **RESOLVED** |
| 4 | Windows / Git Bash `< /dev/null` hang defect | ✅ **RESOLVED**（differential verified） |
| 5 | `V6/.env` 寫入輪替後的新 token | ✅ **RESOLVED BY USER**（2026-08-12 使用者回報） |
| 6 | 本輪 verdict | ✅ **PASS — NO BLOCKING FINDINGS / ROUND CLOSED** |

🔴 **「`V6/.env` 還是舊 token」已不再是 pending item，不要再列進待辦、不要再去驗。**

**不要重新調查本輪 incident，除非出現新 evidence。**

不要重跑本輪的掃描、不要重驗這三個檔案、不要重開已列為 OUT-OF-SCOPE 的項目。

下一階段的起點見第 7 節。

---

## 1. Current project state

`FACT`（來源：`obsidian_note/01 系統現況/現況整理.md`，2026-08-11 版）

- **V6.2 自 2026-08-09 起實際運行中**。排程每天 22:15（`MarketMamba_V62`），V6.1 + 雙模型 21:30（`PersonalOS_Daily`）。
- 下一個時間點：**08-15/16 清試跑資料 → 08-17 正式起跑**。清資料時 `v62_state_*.json` 與
  `v62_portfolio_*.jsonl` **必須一起刪**（只刪 state 會產生混合紀錄且不報錯）。
- 🔴 **不要執行 `--first-day`**（會覆蓋 19 個狀態機）。
- 本機 `V6/marketmamba/config.py` **刻意保持 56 維 dirty，不可 commit**。
- **git 一律指定檔案 `git add <檔>`，禁止 `git add -A`。**

---

## 2. 本輪已完成：FinMind credential remediation ✅ CLOSED

`DECISION` 使用者已確認可安全收尾，本輪標記為 **PASS — NO BLOCKING FINDINGS**。

### 2.1 完成了什麼

`FACT` 三個 active 檔案的 hard-coded credential 字面值已全數移除，改為執行期載入：

| 檔案 | 改了什麼 |
|---|---|
| `V6/scripts/test_finmind_api.py` | 移除完整三段 JWT → `load_dotenv(V6/.env, override=False)` + `os.getenv("FINMIND_TOKEN")`；空值時 stderr 明確報錯 + `sys.exit(1)`，**在任何 HTTP 呼叫之前**。L20/40/60 的 `"token": TOKEN` 依計畫未動 |
| `V6/notebooks/V6_Colab_Training.py` | Cell 0 改寫為 `_can_prompt()` / `_load_secret()`；移除 FinMind 與 `ANTHROPIC_API_KEY` 兩個字面值 |
| `V6/notebooks/V6_Colab_Training.ipynb` | `cells[0]` 與上檔逐字同步、`outputs=[]`；**其餘 9 格未動**（`cell7` 的 3 個 output 保留），單一 hunk |

`_load_secret()` 語意（`DECISION` 定案，不再重新設計）：
載入順序 **`os.environ` → Colab Secrets → interactive hidden input**；
`FINMIND_TOKEN` **required**、`ANTHROPIC_API_KEY` **optional**；成功時**只印來源，不印值/長度/prefix**。

`DECISION` `ANTHROPIC_API_KEY` 的 75 字元字面值一併移除，但**不定性為第二起 credential exposure**
——證據不足以證明它是 Anthropic credential，依最小範圍原則處理即可。**不要為它開新調查。**

### 2.2 `_is_interactive()` liveness defect —— 已修復

`FACT` 第一版 gate 用 `sys.stdin.isatty()`。Windows / Git Bash 下 `python < /dev/null` 時
`isatty()` **仍回傳 `True`** → 控制權交給 `win_getpass` 而**無限等待**，不是 fail loud。

`DECISION` 修正方向：gate 判準改為「**是否身處 IPython / Colab kernel**」（`_can_prompt()`），
不再讀 `sys.stdin` / `isatty`。

`FACT` **differential test 證明**（先證明測試抓得到舊 bug，才承認新 gate 通過）：

| 情境 | 舊 gate | 新 gate |
|---|---|---|
| `< /dev/null` + 無 token | **`EXIT=124`（重現 hang）** | `RuntimeError` / **`EXIT=1`**（實測 0.166 秒） |
| 真 pipe（`isatty=False`）+ 無 token | `EXIT=1` | `EXIT=1` |
| 模擬 kernel 環境 | — | **提示路徑仍可達**（排除「乾脆拿掉 getpass」的退化解） |
| 有 credential | — | 走既定順序、`EXIT=0`、不退化 |

`DECISION` **接受隨之而來的行為變更**：純 CPython 直接執行
`V6/notebooks/V6_Colab_Training.py` 時**不再提供 getpass fallback**，改為立即 fail loud。
理由 —— 本檔主要用途是 Colab，credential 缺失時「立即 fail loud」優先於本機互動輸入便利性。
Colab / IPython kernel 的 interactive fallback **保留**。
（`TerminalInteractiveShell` 即終端機版 ipython 也不再提示，這是 gate 定義的必然結果，非 bug。）

### 2.3 驗收結果

`FACT` **Builder + 兩輪冷啟動 Verifier 獨立驗證，無 blocking findings。**

| 項目 | 結果 |
|---|---|
| Differential test（舊 gate 可重現失敗 / 新 gate 通過） | PASS |
| Fail-loud（真 pipe / `< /dev/null`） | PASS |
| 缺 credential 時零 HTTP request（攔截器有正控制） | PASS |
| Loader 順序、required/optional 語意 | PASS（`env-beats-colab` 直證 env 命中即短路） |
| 互動路徑未被移除（退化解排除） | PASS |
| Active secret scan（三檔，含 `.ipynb` source/outputs 分通道） | PASS，`jwt3seg=0 cred_literal=0` |
| 掃描器 synthetic 正控制（`alg`-first / `typ`-first 兩種鍵序） | PASS，非 vacuous |
| `.py` / `.ipynb` Cell 0 byte-identical | PASS，**sha256[:16] = `d4c70482e0626925`** |
| Notebook 結構（10 格、`cell0 outputs=0`、`cell7 outputs=3`） | PASS |
| Scope / 未預期修改 | PASS |
| 不印 credential 值 / 長度 / prefix | PASS |
| 真實 Colab runtime | **UNVERIFIED**（見第 6 節） |

`FACT` **本輪沒有 `git add` / `commit` / `push`，也沒有修改任何 baseline dirty 檔案。**
三個檔案的改動全部 unstaged，`git checkout -- <三檔>` 可完整還原。

### 2.4 `V6/.env` ✅ RESOLVED BY USER

`DECISION` `FACT`（**使用者於 2026-08-12 明確回報**）
**FinMind 後台 token 已 rotation，且 `V6/.env` 已由使用者手動更新為輪替後的新 token。**

→ **舊 handoff 的「`V6/.env` 還是舊 token，排程會靜默缺資料」這一條 pending 已消解，
不得再出現在任何 PENDING 清單裡。**

🔴 **不要讀取、輸出、hash、decode、驗證或再次檢查該 token 本體。不要呼叫 FinMind API 去測它。**
本輪的紀錄依據**只有使用者的口頭回報**，這是刻意的——驗證它需要接觸憑證值，成本高於價值。

`FACT` `V6/.env` 未被 git tracking（`V6/.gitignore:6:*.env` 命中）。保護狀態正確，不需動 `.gitignore`。

---

## 3. 原 incident 的已驗證事實（保留供參，不需重查）

`FACT`
- 相異 token：**3 支**（SHA-256 指紋辨識，涵蓋 HEAD + `git log -S` 找出的全部 7 個曾動過 JWT 的 commit + working tree untracked 副本；歷史中沒有第 4 支）
- 歷史 commit：**7 個** —— `8b82a2f`(04-12)、`35976ad`(04-13)、`bbb8469`(04-24)、
  `c6ccec8`(04-25)、`cacba84`(04-30)、`677260c`(05-25)、`86a9e9b`(06-02)
- **三支 token 的 payload 都沒有 `exp` 到期宣告** → 不會自行失效；**輪替是唯一有效補救，且已完成**
- 「從 HEAD 移除」與「從 history 移除」是兩件事 —— 刪當前檔案後 `git show <commit>:<path>` 仍取得回完整 token
- 其他 credential 掃描：`sk-ant-api` / `ghp_` / `github_pat_` / `AKIA` / `AIza` / `xoxb-` /
  PEM private key **HEAD 全部零命中**。`PROJECT.md` 與 `archive/docs_old/HANDOFF.md` 的
  `SINOPAC_SECRET_KEY=...` 是**字面佔位符、非真值**
- root `.gitignore:9` 註解記載：**2026-06-12 曾發生玉山證券 API 憑證外流到 git 歷史** —— 本專案第二次憑證外洩

### 已知偽陽性（掃描時必須排除）

`FACT`
1. `V6_Colab_Training.ipynb` `cell[7]` 的 13 字元 `eyJ` 片段 —— 位於 `image/png` base64 blob 內
2. `V6/notebooks/not_using/V6_Training.py:285` 的 `[:30]` 前綴列印 —— 落在所有 HS256 JWT 共用的 header 內

### 掃描方法論（不要重蹈）

`FACT` 早期掃描 pattern `eyJ0eXAiOiJKV1Qi` **只認得 `typ` 排在 `alg` 前的鍵序**，且第一次漏掉全部 `.ipynb`。

→ **secret 掃描一律用通用三段式 pattern**（`eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}`），
**必須涵蓋 `.ipynb` 的 `cells[].source` 與 `cells[].outputs` 兩處**，
且**必須有 synthetic 正控制** —— 否則「掃不到東西」也可能只是掃描器壞了。
這是本專案第五次「檢查腳本自己有 bug」。

---

## 4. 目前 git state

`FACT`（本輪最後一次唯讀檢查）

```
HEAD    = a4a88411d3a91d9c89fd8af79c433f3adf609739
branch  = main...origin/main [ahead 1]
staged  = (empty)
```

`git status --short`：

```
 M CLAUDE.md                              ← baseline（08-11 壓縮 960 行 + Advisor 架構 36 行）
 M V6/marketmamba/config.py               ← baseline（刻意 56 維 dirty，不可 commit）
 M V6/marketmamba/data/fetcher.py         ← baseline
 M V6/notebooks/V6_Colab_Training.ipynb   ← ★ 本輪
 M V6/notebooks/V6_Colab_Training.py      ← ★ 本輪
 M V6/scripts/test_finmind_api.py         ← ★ 本輪
 ? archive/V5.5_repo
 M docs/session-handoff-2026-08-06.md
?? V6/results/archive/df_short_2026-08-11.csv
?? V6/results/archive/df_trend_2026-08-11.csv
?? agent-skills/
?? docs/research/
?? docs/session-handoff-2026-08-12.md     ← 本檔
?? graphify-out/
```

`FACT` 本輪 diffstat：156 insertions / 14 deletions，三檔。
⚠️ 兩支 `.py` 是 LF 工作副本、`core.autocrlf=true` → commit 時可能出現行尾 diff。
`.ipynb` 已確認會被正規化回 LF，diff 維持單一 hunk。**non-blocking。**

`FACT` 另有一個未 push 的 commit `a4a8841`「agents: 把 Agent/Advisor 角色定義納入版控」
（`.gitignore` + `.claude/agents/{builder,verifier,archivist}.md` + `.claude/skills/advisor/SKILL.md`），
經 Verifier 六項驗收全 PASS。

---

## 5. Advisor orchestration 狀態

| 角色 | 狀態 |
|---|---|
| Builder | 已完成兩輪並停止。agentId `a1992a292995a993e` |
| Verifier（第一輪） | 已完成。agentId `aed07a33188cf19b7` |
| Verifier（第二輪，冷啟動） | 已完成。agentId `a7396c0e57bdd7552` |
| Archivist | 本 session 從未派遣 |

`FACT` 本輪實測到的機制事實：
- **`/clear` 之後，上一個 session 的 agent 無法用 `SendMessage` 續談**（`No transcript found`）
  → 跨 session 的委派必須把已批准的計畫**逐字附進新 brief**，否則冷啟動會重新設計方案
- `SendMessage` 續談的 agent **一律在背景恢復**，主 session 無法讓它前景阻塞
- **隔離確實有效**：Builder 主動申報 probe harness 有行號假象（traceback 印 `line 61 getpass`
  實為檔案第 70 行的 `raise`）。Advisor **未轉發該解釋**，冷啟動的 Verifier 獨立撞上同一件事、
  自行算出偏移恆為 9 並排除 → 兩邊在互不知情下收斂到同一結論

---

## 6. PENDING / OUT-OF-SCOPE（不要重新開 scope）

### PENDING（需使用者動作或決定）

> ✅ **不在此清單者已結案**，特別是：FinMind token rotation、`V6/.env` 更新、
> 三檔 credential 移除、hang defect 修復、Builder/Verifier 驗收。**不要把它們寫回來。**

**按優先度排序：**

| 優先 | 項目 | 說明 |
|---|---|---|
| **P1** | **這三個 remediation 檔案怎麼 commit** | 目前全部 unstaged。`git add <檔>` 逐檔加，**禁止 `git add -A`**。注意下述 CRLF/LF 觀察 |
| **P1** | **repo 是 public 還是 private** | 仍未確認（`gh` 未認證）。這決定 history cleanup 的緊急度。**不要為了確認而額外登入／認證** |
| **P2** | **git history 要不要 rewrite** | C1 = 接受失效 token 留在 history（Advisor 傾向）／C2 = `filter-repo` + force push（**不可逆**）。**獨立 decision point，不屬本 remediation round** |
| **P2** | **首次 Colab runtime verification** | 需先在 Colab UI 建 `FINMIND_TOKEN` secret（agent 做不到），第一次跑 Cell 0 即為驗證動作 |
| **P3** | **`a4a8841` 要不要 push** | ⚠️ push 會觸發 Render / Vercel 自動部署 |
| **P3** | **`CLAUDE.md` 兩批 dirty 改動怎麼拆 commit** | 只能 `git add -p`（需互動終端，agent 代跑不了） |
| **P3** | **`docs/session-handoff-*.md` 要不要納入版控** | 本檔目前 untracked。不含 credential 值，但描述了 secret 曾在 repo 中的位置 |
| **P3** | 兩支 `.py` 的 LF / `core.autocrlf=true` | commit 時可能產生無意義行尾 diff。**non-blocking** |

以下為技術性未驗項（**non-blocking，不需動作**）：

| 項目 | 說明 |
|---|---|
| **真實 Colab runtime 未實測** | `_can_prompt()` 的 kernel 偵測只用 `ColabShell(ZMQInteractiveShell)` 模擬。**第一次實際跑 Cell 0 時視為驗證動作。** 需先在 Colab UI 建 `FINMIND_TOKEN` secret（agent 做不到）。失敗模式是 loud 的 `RuntimeError`，不會 hang、不會靜默 |
| `google.colab.userdata` 分支未實測 | 同上，無 Colab 環境 |
| `test_finmind_api.py` happy path 未測 | 要測必然得呼叫 FinMind，紅線禁止。只驗了 negative path + 語法 |
| 純 CPython 終端機無 getpass fallback | **`DECISION` 已接受的行為**（見 §2.2）。**不要重開 implementation round 去「修」它** |

`FACT` 相關：`CLAUDE.md` 有三處引用 untracked 的 `docs/research/*.md` —— 若只 commit `CLAUDE.md`
會產生指向不存在檔案的連結。commit 策略時一併考慮。

### OUT-OF-SCOPE（已明確排除，**不要升級成新 scope**）

- ❌ `V6/notebooks/not_using/V6_Training.ipynb` 的**已知**舊 JWT ——
  已在 §3 的盤點清單內、token 已輪替失效。Verifier 曾把它當新發現回報，**那是因為 brief 未附完整清單，不是新資訊**
- ❌ `archive/**` 與 `not_using/**` 的其餘 token（共 7 個檔案含完整 JWT）
- ❌ untracked 的 `archive/V5.5_repo/config.py`、`archive/V5.5_repo/marketmamba/config.py`
- ❌ `V6/marketmamba/config.py`、`V6/marketmamba/data/fetcher.py`
- ❌ `.gitignore`（雙層保護狀態正確）
- ❌ 呼叫 FinMind / Anthropic / 任何外部 API
- ❌ Render / Vercel / Google Drive / Colab 等外部環境
- ❌ 把「移除 token」與「重構 FinMind integration」合併

---

## 7. ▶ 下一階段的起點

> **credential remediation 已結束。下一個 session 不需要重新調查本輪 incident，除非出現新 evidence。**

回到專案主線：

1. **08-15/16 清 V6.2 試跑資料 → 08-17 正式起跑**
   刪檔清單見 `CLAUDE.md`「進行中」的 🗑 區塊。
   ⚠️ `v62_state_*.json` 與 `v62_portfolio_*.jsonl` **必須一起刪**，只刪 state 會產生混合紀錄且不報錯。
2. **08-17 前決定模型集合定案**（目前 19 個組合 / 11 份分數；晚加入的模型少了那段紀錄，無法公平並列）。
3. **待跑：換 seed 量 Mamba 的 run-to-run σ**（指令與三個驗證檢查在 `CLAUDE.md` 的 ▶ 區塊；
   ⚠️ seed 必須設在 `gd` 上，且必須用全新 runtime）。
4. **`index=False` ratchet**：全樹 22 個 `to_parquet(` 仍缺 `index=False`，08-10 只修了一個呼叫點。
   完整清單與最小 ratchet 計畫在 `docs/research/verification-gap-analysis.md`。

### 待 Archivist 收錄 —— **PROPOSED ONLY，尚未寫入任何正式文件**

`FACT` 依 `.claude/agents/archivist.md §0`，Archivist 只在**已確認的 decision** 上動筆，
不得靠自行推測補寫。使用者尚未批准這三條進 `方法論教訓.md`
→ **本輪只提出提案，未修改 `obsidian_note/` 任何檔案。**

目標檔案：`obsidian_note/07 決策與教訓/方法論教訓.md`
建議歸入既有的 **「一、驗證與量測」** 節（該節已有「檢查指標本身也要驗證」「驗證的判準不可硬編」等同族條目）。

**Proposed entry 1 — `isatty()` 不能當「有人在」的判準**

> Windows / Git Bash 下 `python x.py < /dev/null` 時 `sys.stdin.isatty()` **仍回傳 `True`**。
> 用它當 fail-loud 的 gate，程式會**卡住而不是失敗**——而 hang 與「還在跑」在監看端長得一模一樣。
> → 判斷「能不能要求互動輸入」要問**執行環境本身**（是不是 IPython/Colab kernel），不要問 stdin。
> 同族於既有紀律「沉默不是成功的證據」。踩到的地方：2026-08-12 FinMind credential remediation。

**Proposed entry 2 — differential test 要先證明抓得到舊 bug**

> 只證明「新版沒有 hang」不算數：一個根本沒觸發到 gate 的測試情境也會全綠。
> 順序必須是 ①先用舊版跑出 `EXIT=124`（重現失敗）②再證明新版 `EXIT=1`。
> 另需一格排除**退化解**——把功能整段刪掉也會讓所有負向測試通過。
> 這是「檢查腳本自己有 bug」家族的第六個實例。

**Proposed entry 3 —（較大的 recurring lesson）scanner 必須有正負控制**

> **「0 findings」只有在掃描器已證明抓得到 synthetic known-positive 時才有意義。**
> 本輪早期 pattern `eyJ0eXAiOiJKV1Qi` 只認得 `typ` 排在 `alg` 前的鍵序——
> **對 JWT header 的 JSON key ordering 作了錯誤假設**，且第一次漏掉全部 `.ipynb`。
> → secret 掃描一律用通用三段式 pattern，涵蓋 `.ipynb` 的 `source` 與 `outputs` 兩通道，
> 並同時放入 **positive control（兩種鍵序都要）＋ negative control**。
> 本專案已第六次出現「檢查腳本自己有 bug」（前例：備份比對的 regex 漏大寫、FCF 驗收測錯東西、
> MOPS 判準硬編記憶數字）→ **這已經不是偶發，應視為本專案的系統性弱點。**

原兩條的簡述保留如下（與上方提案同義，勿重複收錄）：

1. **Windows 的 `sys.stdin.isatty()` 對 `< /dev/null` 仍回傳 `True`**
   → 不能用它判斷「有沒有人在」。用它當 fail-loud 的 gate 會讓程式**卡住而不是失敗** ——
   而 hang 與「還在跑」在監看端長得一模一樣。同型於既有紀律「沉默不是成功的證據」。
2. **differential test 必須先證明能重現舊 bug，才算數**
   → 只證明「新版沒 hang」是不夠的：一個根本沒觸發到 gate 的測試情境也會全綠。
   同時要有一格排除**退化解**（把功能整段拿掉也會讓所有負向測試通過）。
   這是「檢查腳本自己有 bug」家族的第六個實例。

---

## 8. Scope / safety constraints（沿用，未變）

**Credential 相關**
- 🔴 不得讀取 / 輸出 / echo / hash / decode / 傳遞 `V6/.env` 的任何值
- 🔴 不得把任何 token 本體放進 prompt、agent brief、log、commit message、檔案或 scratchpad
- 🔴 不得索取 token；不得呼叫 FinMind 或任何外部 API 驗證憑證
- 掃描 secret 時只輸出「檔案路徑 + 是否命中 + 命中數量」，不輸出命中字串

**專案常設紅線**（`CLAUDE.md` / `.claude/agents/builder.md §2`）
- 禁止修改 `V6/models/` 下任何檔案（含 `.pt`）
- 禁止 `git add -A` / `git add .` —— `config.py` 刻意 56 維 dirty
- 禁止執行 `v62_daily.bat --first-day`
- 禁止刪 `v62_state_*.json` 而不刪 `v62_portfolio_*.jsonl`
- 禁止 Line Notify
- 禁止改 `models/inference.py`（已棄用，修推論改 `run_daily_inference.py`）
- 禁止未經授權在 production 資料檔上做「縮小」操作

**Advisor 協定**（`.claude/skills/advisor/SKILL.md`）
- Builder ✕ Verifier 不直接溝通；Advisor 是唯一傳遞節點，**只傳客觀事實，不轉發辯護性 reasoning**
- 不代替使用者批准 Builder 的計畫
- 不因 Verifier 提 Finding 就接受；不因 Builder 說「這是刻意的」就採信

---

## 9. 已知的 protocol observed issues（記錄用，未修正）

`FACT`
1. `SKILL.md §2`（主 session 直接做 90% 小事）與 `§8` 紅線（Advisor 不寫 production code）
   **在「主 session ＝ Advisor」前提下互斥**。邊界未定義。
2. `builder.md §0` 第 6 步仍寫「等**使用者**確認」，與 `§0.1` 的 Advisor 轉呈措辭不一致（潛在，未肇事）。
3. `docs/research/future-architecture.md §4.2/4.3` 主張「不需要多 agent、三角色、無協調層」，
   與現行架構矛盾。`DECISION` 使用者已指示**留待實戰後由 Archivist 處理**，本輪不動。
4. `SendMessage` 恢復的 agent 只能背景執行 —— 原生機制限制，`§3` 未載明。
5. **`/clear` 之後 agent transcript 消失**，跨 session 續談不可行 —— `§3`「續談用 SendMessage」
   需補上這個前提條件。

---

## 10. 本 handoff 的元資訊

- 建立時間：2026-08-12（**session closure 版**，取代同日稍早的兩個版本）
- 建立者：Advisor（主 session）
- **本輪 verdict：PASS — NO BLOCKING FINDINGS。ROUND CLOSED。**
- 本 closure 動作**只寫了本檔一個檔案**：未 git add / commit / push、
  未修改任何 production code、未修改任何 baseline dirty 檔案、
  未寫入 `obsidian_note/`、未呼叫任何外部 API、未讀取 `V6/.env` 的任何值
- 本檔目前為 **untracked**，是否納入版控由使用者決定（內容不含任何 credential 值）
- **下一個 session 的閱讀順序**：本檔 §0 → §6 PENDING → §7 起點。§2/§3 只在需要時回查。
