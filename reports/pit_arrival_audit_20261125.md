# 最小每日版本帳本與到檔時間稽核 — 2026-11-25

**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。Human 核准計畫 §J 的第一步；builder 自查不是獨立驗收。本案不讀報酬／標籤或 blind、不做模型／事件 IC、不改 P2／P4d／production authority。

## 範圍與建立前偵察

已讀 AGENTS.md、CLAUDE.md、Current_State、Authority_Map、Data／Research／Operations domain、ml-leakage-audit、計畫 §D／I／J、scheduler mitigation snapshot 及 fetcher 相關讀寫／補齊路徑。既有 `V6/research_lab/pit_snapshots.py` 是月末交易所報價／universe HTTP collector，不適合四表快取版本帳；不能沿用其網路／月份語義。沿用既有 `tools/`、`tests/`、`reports/`；工具放 `tools/pit_ledger.py` 與 `tools/pit_arrival_audit.py`，避免讓 P4a 的 research_lab 來源 inventory 納入本機新能力，不改任何舊 source-policy gate。

另核對既有 `V6/experimental/v7_source_snapshot.py`（HTTP source／response immutable envelope）與 `v7_forward_vintage.py`（PROPOSED observation verifier，raw上限2 MiB），兩者沒有四表業務鍵diff／數千萬列重播介面，不能拿它們的既有source／admission契約承載此帳本；本次不修改它們。

新版帳本使用主 checkout `/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2/`，不落在 worktree；既有 `.artifacts/` 可承載研究證據。Claude 審查回報舊 v1 約9.8 GB（本輪未重測），逐列 JSON 重複存於 events/current，不能滿足每日容量契約，因此使用 Human 指定的獨立 v2 namespace；不是搬遷或覆寫舊證據。分類 **EXTERNAL_LARGE_DATA／本地觀測證據**，非可丟棄 cache；完整 ledger 是唯一此次實收版本副本，禁止當暫存清除。報告及來源為待審查候選；JSON 摘要為可由帳本重建的 GENERATED projection。未建立新 top-level tree、worktree 或 memory 系統。

指派 worktree：`/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww5b81x-8af923/builder_pit_step1b`；detached HEAD `fb32e4c1a99fafb05abf80fd7b041b08859f5f24`，起始 clean。owner/task=`job-run-mww46oz9-6edacd/builder_pit_step1`，created date=本輪開始前已存在／確切 UNKNOWN。WSL Linux CPU，Python 3.12.3：`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`。`MARKETMAMBA_DATA_ROOT` 未設；worktree `Data` realpath 是本地同名普通目錄，沒有讀其資料。實際資料明確取 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/`，不推定兩 checkout 等價。

## 觀測與重播契約

業務鍵：營收 stock_id／revenue_year／revenue_month；三表 stock_id／Date／type。欄位依名稱排序，日期 ISO、數值整數化等價的 1.0／1、NaN→null、拒絕 infinity；股票字串保留前置零。每列完整標準化 JSON 保存原欄位值，不只雜湊；來源 Arrow schema 保留型別及欄位資訊。SHA-256 包含整列（含 metadata），因此 metadata 變動亦是 REVISED，不能直接當成經濟數值重編率。

`events`／`runs` 僅 INSERT，UPDATE／DELETE 有 DB trigger 拒絕。v2 不保留 current 或 INITIAL_STOCK JSON 事件；初始來源 parquet 逐檔複製至以完整 SHA-256 命名的 blobs，fsync／驗證後設唯讀，使用同 filesystem hard-link 原子 no-clobber 發布。每鍵 canonical JSON 鍵與完整256-bit列 SHA 另存 zstd parquet 索引。SQLite quick_check、schema v2、event count、每 run event digest、run chain 與起始 base/index 檔案 SHA 不符即 STOP。每次執行含 run_id、Asia/Taipei 時間、各來源 path／SHA／列數／schema、解析器 SHA、Python 版本、事件計數。來源掃描前後及四表 commit 前均驗 SHA，拒絕讀取過程變動；不聲稱四檔是上游同一批原子更新。

首次 INITIAL_STOCK 計數來自索引，完整欄位保存在起始 parquet 副本，不逐列存 JSON；**不以原 first_observed_at／create_time 倒填帳本時間**。後續新出現 active key 為 FIRST_SEEN（含曾消失後再出現；舊史不刪）、有內容差異為 REVISED 並保存 previous_hash／previous_json，消失為 DISAPPEARED 並保存前值。四表來源 SHA 皆相同只記一筆 NO_CHANGE run，完全不解碼來源列，零內容事件；回到更早歷史 SHA 若相對上一次不同仍正常記錄變動。DISAPPEARED 僅表示快取缺鍵，不能證明下市、撤回或合法刪除。

單一 SQLite FULL synchronous transaction 涵蓋四表，任一失敗全部 rollback；初次 DB 先寫獨立 temp，commit 後 rename＋directory fsync 發布；後續沿用 SQLite atomic commit。flock 排除重疊 writer。不可變 base/index 與 diff events／runs 共同構成本次本地觀測機器證據；run_id.summary.json 原子寫入；若 commit 後 summary 發布中斷，可從 runs.summary_json 重建，不能重造內容事件。DB 已損毀或 IDENTITY 還在而 DB 消失即 STOP，不默默重建。每日多次執行會有多個 run_id 摘要，依 observed_at 可彙整同日；不覆寫昨日。

所有 fresh／公司刷新覆蓋標 UNKNOWN；已有 fetcher 是 revenue 每日最多80股、financials 40股輪詢及 MOPS 補齊，不能將「看見未變快取」解讀為「今日重新查詢全部公司」。三表與營收來源檔時間也不等於公告時間。每日一次可能漏掉日內中間版；本帳本從實際啟動起才有本地版本證據，不恢復歷史 publication PIT。

執行（本輪真實初始化與一次 NO_CHANGE 驗證；未啟用每日工作）：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.pit_ledger --ledger /home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2
```

到檔稽核數字沿用前輪唯讀結果，原解析器 SHA 保留於 JSON；本輪只補解讀，未重新取得或量測。

## 到檔描述結果與限制

[機器報告](pit_arrival_audit_20261125.json) 保存資料／日曆 SHA、全部月別分布與分母。459,119 列中 5,942 筆 create_time 可解析（1.2942%）；其餘 453,177 筆不納到檔分布，也不能視為準時。

`create_time − 次月10日00:00（台北）` 的曆日分布：min −8、Q25 0、中位 11、Q75／P90／P95／P99 42、max 60 日；1,188 筆早於10日。這是以期限日零時作距離原點，**不是法律逾時判定**。遲於＋2交易日的判定採第二個嚴格晚於期限日的 session **整日結束後**（next day 00:00 起）；期限日不算第一個緩衝日。未核實假日法定順延與展延。

- W34 官方開發日期（已驗 pinned SHA，僅用 OFFICIAL session 子集）到 2026-10-22：5,711 筆可計，231 筆需11月日曆而 UNKNOWN；晚於一般10日＋2交易日 **3,971／5,711＝69.5325%**。
- 補充採 fetcher `_trading_calendar()` 既有口徑（prices_raw 只讀 Date union，未 import fetcher／未讀價值）：**3,977／5,942＝66.9303%**；完整月別數字在 JSON。此日期 union **不是合格交易所日曆**，與官方覆蓋口徑分開呈現，不能升格 strict gate。
- 本輪未取得逐公司保險／有保險子公司／其他期限例外的歷史適用證據，**全部 legal_applicability=UNKNOWN；正式法律逾時比例=null**。上述比例僅一般10日代理的壓力描述，不把未知例外默認成一般公司。
- create_time 語義未經來源公告驗證，可能是供應商建檔／回補／最後更新日期；naive 值假設 Asia/Taipei。不能據此分辨公司遲報、收集器遲到或事後更正；5942筆亦可能有非隨機覆蓋偏誤。不讀報酬、不據績效調緩衝、不因此放行歷史期限代理。

create_time 分布（中位 +11 日、大量集中 +42 日）顯示它**很可能是供應商建檔／回補時間**，不能作為遲交比例的上限依據；因此歷史 PIT 的遲交上限需要其他來源，例如 TEJ 營收／盈餘發布日，待 Human 確認授權。本輪不連網、不取得 TEJ。

## 獨立排程提案：只寫指令，未註冊

[2026-08-25 snapshot](../docs/operations/scheduler-mitigation-snapshot-2026-08-25.md) 記錄專用 `\MarketMamba_DataFetch`：週一至五22:30，wscript.exe → run_hidden.vbs → v62_daily.bat --fetch-only → WSL Ubuntu／run_v62_daily.py --fetch-only。既有 fetch-only 檢查每日10源並停止於推論前；本工具不修改其成功條件，該條件也不證明四份月／季表當日全部刷新。

**目前 runtime UNKNOWN**。前輪在台北2026-11-25 20:48左右唯讀呼叫 Get-ScheduledTask／Get-ScheduledTaskInfo 失敗：`WSL ERROR: UtilBindVsockAnyPort:317: socket failed 1`。這不是 Windows runtime observation；沒有刷新 canonical runtime observed_at，舊 snapshot 超過7日 freshness。啟用前由 Claude／Human 唯讀驗證 task identity、觸發、Operational journal 是否啟用、last result 及 source commit／path。下列使用本輪具體 worktree，將來若搬遷須另審 path／artifact dependency，不可刪掉仍被 action 引用的 worktree。

方案：**獨立 Windows event-trigger task**，監聽 DataFetch 完成 EventID102，不採固定23:00的競速假設。抓取不執行／未完成則不跑快照；抓取失敗則新 task STOP。新 task 的退出碼不回傳、不改既有抓取。無 WakeToRun、無補跑、無通知、無 recurring timer；它是未啟用的每日抓取完成後續方案。正式啟用及第一次端到端驗證仍需 Claude 審查／另案操作。

確切註冊指令（**僅文件，未執行**；PowerShell，登入的原操作使用者；不得 -Force 覆寫已存在 task）：

```powershell
$taskName = 'MarketMamba_PITLedger'
if (Get-ScheduledTask -TaskPath '\' -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already exists: STOP' }
$upstream = Get-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_DataFetch' -ErrorAction Stop
$log = Get-WinEvent -ListLog 'Microsoft-Windows-TaskScheduler/Operational' -ErrorAction Stop
if (-not $log.IsEnabled) { throw 'Operational event journal disabled: STOP; do not enable automatically' }
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$script = @'
$ErrorActionPreference = 'Stop'
$t = Get-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_DataFetch'
$i = $t | Get-ScheduledTaskInfo
if ($t.State -eq 'Running' -or $i.LastTaskResult -ne 0 -or $i.LastRunTime -lt (Get-Date).AddHours(-6)) { throw 'Fetch completion not verified: STOP' }
& wsl.exe -d Ubuntu -- bash -lc 'cd /home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww5b81x-8af923/builder_pit_step1b && PYTHONPATH=. CUDA_VISIBLE_DEVICES= /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.pit_ledger --ledger /home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2'
exit $LASTEXITCODE
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
$subscription = '<QueryList><Query Id="0" Path="Microsoft-Windows-TaskScheduler/Operational"><Select Path="Microsoft-Windows-TaskScheduler/Operational">*[System[EventID=102]] and *[EventData[Data[@Name="TaskName"]="\MarketMamba_DataFetch"]]</Select></Query></QueryList>'
$escSubscription = [Security.SecurityElement]::Escape($subscription)
$escIdentity = [Security.SecurityElement]::Escape($identity)
$xml = @"
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Independent research-only PIT cache ledger after DataFetch; no source writes</Description></RegistrationInfo>
  <Triggers><EventTrigger><Enabled>true</Enabled><Subscription>$escSubscription</Subscription></EventTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$escIdentity</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>false</StartWhenAvailable><WakeToRun>false</WakeToRun><Enabled>true</Enabled><ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings>
  <Actions Context="Author"><Exec><Command>powershell.exe</Command><Arguments>-NoProfile -NonInteractive -EncodedCommand $encoded</Arguments></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskPath '\' -TaskName $taskName -Xml $xml -User $identity -ErrorAction Stop
```

還原／撤銷指令（只撤銷新 task，不觸碰 upstream，也不刪帳本）：

```powershell
Disable-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_PITLedger' -ErrorAction Stop
Unregister-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_PITLedger' -Confirm:$false -ErrorAction Stop
```

排程 failure 不影響 upstream，但目前 source-side 無 writer lock／四表共同完成 marker。快照前後 SHA 可偵測讀取當下的改寫，不能排除兩次掃描間上游已經混批；現階段僅本地快取觀測。事件接線、睡眠／登出／停電與 filesystem durability 尚未現場驗收；PowerShell／XML 亦未在 Windows 執行驗證。

## 前輪 v1 驗證紀錄（歷史；不代表新版通過）

RED：新增 tests 後模組尚不存在，collection `1 error in 1.17s`（ImportError），exit2。GREEN focused `10 passed in 3.88s`；含首次存量、新鍵、修訂前後值、消失、重新出現、同SHA零事件、physical corruption、缺DB、derived-index損毀、重鍵／缺表、時區、非有限值、日曆缺口、＋2日邊界、append-only拒絕、讀取中改寫rollback。兩項 in-memory mutations：將營收 row hash 固定化使未變列也誤判 REVISED；將第二交易日門檻換成期限日使11/11誤判逾時；兩者均被驗收 assertion killed（不寫入來源程式）。

40,000合成列實測1.740861秒，線性估13,760,451列約598.876秒。這只是小檔CPU／I/O抽樣，非上界；真實執行含不同JSON長度、SQLite索引及並行CPU測試，以下真實數字優先。

營收月份分布（一般10日代理，W34官方覆蓋）：

| 營收月 | 可解析列 | 可計日曆列 | UNKNOWN日曆列 | 晚於＋2日 | 比例 |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 1 | 1 | 0 | 1 | 100.0000% |
| 2026-02 | 1958 | 1958 | 0 | 1958 | 100.0000% |
| 2026-03 | 1958 | 1958 | 0 | 1958 | 100.0000% |
| 2026-04 | 119 | 119 | 0 | 0 | 0.0000% |
| 2026-05 | 119 | 119 | 0 | 0 | 0.0000% |
| 2026-06 | 394 | 394 | 0 | 14 | 3.5533% |
| 2026-07 | 396 | 396 | 0 | 14 | 3.5354% |
| 2026-08 | 357 | 357 | 0 | 11 | 3.0812% |
| 2026-09 | 409 | 409 | 0 | 15 | 3.6675% |
| 2026-10 | 231 | 0 | 231 | 0 | UNKNOWN |

可解析 create_time 實測範圍 2026-04-08～2026-11-17，**5,942筆全為00:00:00（非零時分秒0筆）**；這是日期精度，不驗證18:00截止，也不能推出當日第一次公告的時刻。本次以第二緩衝日整日為比較單位；不能將此日期代理直接接上交易訊號。

## V2 重播、容量與本輪驗證

v1 的純 events SQL 不適用 v2。新版使用經驗證的起始 parquet 加截至目標 runs.seq 的 diff：

```python
from tools.pit_ledger import replay
rows = replay('/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2', target_run_id, 'revenue_raw')
# rows 逐列產生完整 canonical JSON；NO_CHANGE run 也能重播。
```

任一日的視圖取該日台北截止前最後一筆 runs.seq，再以 run_id 重播；沒有更早觀測則 UNKNOWN，不能回填。重播包括舊 run、NO_CHANGE、全表消失、重新出現；完整值只在 base 或 FIRST_SEEN／REVISED／DISAPPEARED 的差異中保存。DISAPPEARED 保存前值，REVISED 保存前後值。重跑相同來源不重造事件，僅追加觀測摘要。

<1GB數值只指持久帳本；暫存磁碟及記憶體峰值未量測。初始化與內容變更表仍須完整掃描與暫存磁碟 hash 索引；來源 SHA 未變時只讀 SHA／metadata，不解碼列。完整性驗證仍重新 hash 不可變 base/index 及歷史事件，不以「不重掃來源列」解讀為零 I/O。diff／overlay 的記憶體與儲存量隨變更鍵增加；大量全面回補不保證 MB，日常稀疏變更才是 MB 級目標。失敗可能留下已驗 immutable orphan blob，無 run 引用即非已發布觀測；本輪不自動刪除。

舊帳本仍在 `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww46oz9-6edacd/builder_pit_step1/.artifacts/pit-ledger/`，本輪未寫入／刪除，由 Claude 另案處理。新帳本獨立起算，不把舊 v1 觀測時間貼到新版。初次執行 parser SHA 以帳本 summary 為準；啟動後補強動態 git identity 與唯讀模式驗證，最終 source SHA 另列，沒有重標已發布 run。因Git base未含未提交工具，JSON另保留final→initial的精確producer差異，已重建並驗回initial parser SHA；不另建備份樹。

本輪最終實測如下；完整 pytest 輸出另列。完成不是 Human acceptance；獨立審查、現場 Windows 事件接線、上游四表共同刷新證據、外部備份與每日運維均未驗收。

### 真實初始化與 NO_CHANGE（本輪已執行）

- 持久位置：`/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2/`；初始 run `4eebefd111c34d3f9f6f87ff6640661f`，觀測時間 `2026-11-25T21:18:06.516630+08:00`；source commit `fb32e4c1a99fafb05abf80fd7b041b08859f5f24`，未提交程式 SHA 見 JSON。
- 四表 13,760,451 列：營收459,119、損益2,625,875、資產負債8,307,996、現金流量2,367,461。INITIAL_STOCK 都由索引保存，SQLite events **0列**，無current表。
- 真實初始總大小 **556400149 bytes（約 556.400 MB）**；目標 `<1,000,000,000 bytes` 的實際 assertion 通過。其中來源副本78,493,586 bytes、索引477,885,423 bytes，其餘為DB／摘要。初始化 commit 前實測 **614.601970秒**，不是合成外推。
- 第二次 run `2b2dc1b9b3bb404c8e06f120e255633e`，status=NO_CHANGE，實測 **NO_CHANGE_614.601970秒**；只增加 **11913 bytes**，最終 **556412062 bytes**。零新增內容事件；已重新驗 schema／SQLite quick_check／run chain／event count與digest／base與index SHA及唯讀模式。
- 四份原parquet的副本 SHA 與來源相同；8個唯讀base/index blobs以及DB／summary完整size與SHA inventory在JSON。DB SHA：`bfbb7a6296ad12b7e14b4d7ac561ebbcb19a9f230c1383e2e18d4743c42d0f5b`。舊9.8GB帳本未寫刪，本輪不主張已有外部備份或production部署。
- 環境：WSL CPU，Python3.12.3；Pandas 3.1.1／PyArrow 26.0.1／SQLite 3.45.1；`CUDA_VISIBLE_DEVICES=''`、`MARKETMAMBA_DATA_ROOT` unset。來源實體路徑 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`；本worktree Data為普通目錄、未作來源。

### 工程驗證

RED：新增「初始 events 必須為空＋完整重播」測試，舊版實際 **1 failed in 0.89s**，因為保存了8筆INITIAL_STOCK JSON；修後GREEN。最終focused **15 passed in 20.10s，exit0**，無新增skip，驗初始／NO_CHANGE／修訂／新鍵／消失／空表／重新出現的逐表完整重播、byte損毀／缺base／缺DB／event竄改STOP、重鍵／缺來源、時區／非有限值、append-only及來源改寫rollback。

兩項 in-memory mutations，未改寫來源：營收row hash固定化由previous-value驗證STOP殺死；第二交易日改成期限日，由晚到計數assertion殺死。不是實作者獨立審查。

合成100,000列：初始2,295,917 bytes（同內容表可依SHA去重）；未去重索引總3,297,676 bytes，線性預算估13,760,451列＋85MB來源為539,775,090 bytes，assertion `<1GB` 通過。四表各修訂一列，增量11,604 bytes，`<1MB` assertion通過；真實測量優先於外推。NO_CHANGE禁止解碼records的fixture亦通過，額外增量`<100KB`。

語法3檔、JSON解析、報告與四份knowledge的本機link/reference、`git diff --check`均已驗證。Knowledge impact含storage／replay語義、持久path與worktree責任變更，因此同輪更新Current State／Authority Map／Data／Worktree Register；未改authority-map.json、source admission或外部observed_at。knowledge health為WARNING、無structural FAIL；既有8筆AUTHORITY_REPIN_CANDIDATE及legacy文件drift仍在，本任務不擴張修理範圍。依no-commit指示保留4 modified＋5 untracked，無staged，不能宣稱clean worktree。

### 指定全套 pytest 實際輸出

```bash
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

退出碼 **0**，**0 failed**；先前較早collection已中止（exit130），以下為包含全部15項帳本測試的最終執行，未刪測試／新增skip或放寬來源gate。

```text
........................................................................ [  5%]
........................................................................ [ 11%]
........................................................................ [ 16%]
........................................................................ [ 22%]
........................................................................ [ 27%]
........................................................................ [ 33%]
........s............................................................... [ 38%]
.................................s...................................... [ 44%]
........................................................................ [ 50%]
........................................................................ [ 55%]
........................................................................ [ 61%]
........................................................................ [ 66%]
........................................................................ [ 72%]
..........s............................................................. [ 77%]
........................................................................ [ 83%]
........................................................................ [ 88%]
...................................................................s.... [ 94%]
........................................................................ [100%]
=============================== warnings summary ===============================
research_lab/tests/test_p4.py::test_zip_inventory_and_payload_gates[duplicate]
  /usr/lib/python3.12/zipfile/__init__.py:1620: UserWarning: Duplicate name: 'folds/fold_2021/X.npy'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-315/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1292 passed, 4 skipped, 5 warnings in 944.62s (0:15:44)
```

Knowledge impact 的 REVIEW_REQUIRED 指向 Operations／GOVERNANCE：已檢閱，全球操作／安全政策與既有排程未改，兩檔無需更新；持久路徑與本worktree責任已在本輪四份knowledge及報告記錄。暫存磁碟與RAM峰值、Windows接線、長期每日運作及獨立複審仍未驗收。
