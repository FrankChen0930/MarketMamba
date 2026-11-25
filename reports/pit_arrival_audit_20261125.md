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
  <Triggers><EventTrigger><Enabled>false</Enabled><Subscription>$escSubscription</Subscription></EventTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$escIdentity</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>false</StartWhenAvailable><WakeToRun>false</WakeToRun><Enabled>false</Enabled><ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings>
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

<1GB數值只指持久帳本；暫存磁碟及記憶體峰值未量測。初始化與內容變更表仍須完整掃描；本輪修補後 SQLite 暫存 hash 表固定在記憶體，parquet 索引暫存固定在帳本目錄；來源 SHA 未變時只讀 SHA／metadata，不解碼列。完整性驗證仍重新 hash 不可變 base/index 及歷史事件，不以「不重掃來源列」解讀為零 I/O。diff／overlay 的記憶體與儲存量隨變更鍵增加；大量全面回補不保證 MB，日常稀疏變更才是 MB 級目標。失敗可能留下已驗 immutable orphan blob，無 run 引用即非已發布觀測；本輪不自動刪除。

舊帳本仍在 `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww46oz9-6edacd/builder_pit_step1/.artifacts/pit-ledger/`，本輪未寫入／刪除，由 Claude 另案處理。新帳本獨立起算，不把舊 v1 觀測時間貼到新版。初次執行 parser SHA 以帳本 summary 為準；啟動後補強動態 git identity 與唯讀模式驗證，最終 source SHA 另列，沒有重標已發布 run。因Git base未含未提交工具，JSON另保留final→initial的精確producer差異，已重建並驗回initial parser SHA；不另建備份樹。

本輪最終實測如下；完整 pytest 輸出另列。完成不是 Human acceptance；獨立審查、現場 Windows 事件接線、上游四表共同刷新證據、外部備份與每日運維均未驗收。

### 真實初始化與 NO_CHANGE（本輪已執行）

- 持久位置：`/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2/`；初始 run `4eebefd111c34d3f9f6f87ff6640661f`，觀測時間 `2026-11-25T21:18:06.516630+08:00`；source commit `fb32e4c1a99fafb05abf80fd7b041b08859f5f24`，未提交程式 SHA 見 JSON。
- 四表 13,760,451 列：營收459,119、損益2,625,875、資產負債8,307,996、現金流量2,367,461。INITIAL_STOCK 都由索引保存，SQLite events **0列**，無current表。
- 真實初始總大小 **556400149 bytes（約 556.400 MB）**；目標 `<1,000,000,000 bytes` 的實際 assertion 通過。其中來源副本78,493,586 bytes、索引477,885,423 bytes，其餘為DB／摘要。初始化 commit 前實測 **614.601970秒**，不是合成外推。
- 第二次 run `2b2dc1b9b3bb404c8e06f120e255633e`，status=NO_CHANGE，實測 **1.748361秒（約1.75秒）**；只增加 **11913 bytes**，最終 **556412062 bytes**。零新增內容事件；已重新驗 schema／SQLite quick_check／run chain／event count與digest／base與index SHA及唯讀模式。
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


## 獨立審查 REQUIRED 修補 — 2026-11-25

本節為 `job-run-mww6g0bv-340962/builder_pit_fix1`、detached HEAD/base
`47bf4659e1143269b622bfd80ebed687f3538973` 的未提交修補；起始 clean。
**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；可註冊每日排程仍為 NO_GO**，需獨立複審與下列現場驗收。
僅改工具、測試與相關報告／knowledge；不讀 blind、不用網路／GPU、不寫正式 Data 或主 checkout 實際帳本，不改抓取／P4d／Windows runtime。
以下為本輪合成檔案工程證據；上方真實初始化／inventory 是前輪歷史量測，本輪沒有重新量測或改寫它們。

- 業務鍵先 `normalized()` 再驗證；None／NaN／空字串／純空白全部 STOP，非鍵欄位仍可 NaN→null，前置零股票字串保留。
- `validate()` 將 sqlite_master 的必要 tables、constraints、四個 append-only triggers 與預期 SCHEMA 逐物件比對正規化 SQL SHA-256；缺少、被替換、額外 schema 或 table schema 改變即 STOP，`user_version` 不代替定義驗證。
- 寫入根目錄只允許本 checkout `.artifacts/pit-ledger-v2` 或明列的主 checkout 同名 namespace（`ALLOWED_LEDGER_ROOTS`）；沒有任意 CLI allowlist 開關。測試以 monkeypatch 明確允許各自 tmp_path/ledger。來源與帳本互不包含、Data 禁入；lexical/resolved 不一致、root／祖先／既有後代（含 dangling link／lock／DB／blobs／marker）symlink 均 STOP，檢查在首次寫入前。
- parquet 索引 NamedTemporaryFile 使用 `dir=ledger`；SQLite `temp_store=MEMORY`，不再讓 FILE temp 落到系統暫存區。這會增加 RAM 需求，真實13,760,451列的新記憶體峰值未量測，不能沿用前輪容量量測宣稱新峰值通過。
- 路徑檢查需要本機可信、穩定且不被其他程序惡意改名的 namespace；不是對敵意並行 symlink 替換的 OS sandbox。原每日一次／共同刷新／publication UNKNOWN 不變。

### RED、GREEN 與新 mutation

命令：`PYTHONPATH=. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest tests/test_pit_ledger.py -q -s -p no:cacheprovider`。
先加回歸再改實作：**21 failed、18 passed，17.32秒**；NaN／空鍵、trigger 缺少與定義改寫、table schema、越界與 symlink、暫存設定的原始缺陷被重現。
最終 focused：**43 passed in 17.47s，exit0**，無新增 skip。
兩項新 mutation 在記憶體編譯函式，未修改磁碟工具；baseline oracle 先驗 STOP，變異後由相同 oracle 的 assertion 拒絕：

```text
NEW_MUTATION normalized_key KILLED
NEW_MUTATION schema_definition KILLED
```

### 實體 SQLite 故障演練（可重跑）

`test_real_sqlite_kill_and_file_size_limit_recovery` 使用 spawn 子程序、tmp_path 真實 parquet／SQLite DB／rollback journal；`:memory:` 只用於建構預期 schema，完全不作 durability 替身。
SIGKILL 在實際 INSERT 後，將未提交 dirty pages spill 至 DB 並確認 journal 非空，再由父程序殺死 writer。
RLIMIT_FSIZE 在子程序限制每個實體檔案大小，使用大值 diff 觸發 OS 拒寫，非 monkeypatch 例外注入；這是檔案大小限制模擬，沒有填滿主機實體磁碟。
重新開啟後 validate、四表完整重播、run count 與 retry／NO_CHANGE／事件數均驗證：

```text
REAL_SQLITE fault=kill message=WRITING_WITH_JOURNAL recovered_runs=1 retry=CHANGED repeat=NO_CHANGE events=12
REAL_SQLITE fault=full message=STOP:ledger/source validation STOP: disk I/O error recovered_runs=1 retry=CHANGED repeat=NO_CHANGE events=8
```

兩個受控 fault 都在 commit 前，恢復為完整舊狀態，沒有 partial run 或 orphan event；重試各四表完整新狀態，再跑不增內容事件。
這沒有驗證每個 commit 邊界、初始化發布中斷、實體裝置斷電、Windows／跨檔案系統或長期運維；RLIMIT_FSIZE 的 SQLite 錯誤為 disk I/O error，不能改稱實測 ENOSPC。

### 註冊前驗收程序（只供 Human／Claude 另案執行）

1. 維持 NO_GO，先完成獨立複審並由 Human 確認允許執行；確認實際 Windows task／WSL runtime、帳本允許路徑、可用磁碟及 RAM、持久備份與可用的固定 checkout，替換上方示例中的舊 builder worktree。記錄 `observed_at`，不靠舊 snapshot 推定現況。
2. 用 PowerShell parser 驗證 `$script`、`[xml]$xml` 解析 XML，檢視 action 與 upstream task 精確名稱。上方提案已改為 trigger／task 都 `Enabled=false`，且 `WakeToRun=false`；只以停用狀態註冊新獨立 task，確認無 recurring trigger，既有 DataFetch 不變。
3. 待一次可驗證的成功 DataFetch 完成後，以同一 identity 手動執行新 task 的 action（直接呼叫 `$script` 的 PowerShell／WSL 命令，停用 task 不用 Start-ScheduledTask）。保留 stdout／stderr／退出碼；必須 exit0，摘要含四表與正確 run_id／observed_at。STOP／非零退出碼即保留停用並調查。
4. 在主 checkout 已納入本輪候選、步驟3成功且取得另案寫入授權後，於 WSL 執行以下完整命令。此程式先以 `mode=ro`／`query_only=1` validate 並完整 replay 四表，印出各表重播列數、來源 SHA 及 runs／events；接著手動 snapshot 一次，再用新的唯讀連線重驗。若上游來源在兩次間改變、出現 STOP 或計數不符，即停止驗收，保持 NO_GO。這段是另案驗收操作，本輪未執行。

   ```bash
   cd /home/frank/projects/MarketMamba
   PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python - <<'PYVERIFY'
   import sqlite3
   from pathlib import Path
   from tools import pit_ledger as p

   ledger = Path('/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2')
   source = Path('/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6')

   def verify(label):
       with sqlite3.connect(f'file:{ledger / "ledger.sqlite3"}?mode=ro', uri=True) as c:
           c.execute('PRAGMA query_only=1')
           _, last, _ = p.validate(c, ledger)
           counts = tuple(c.execute('SELECT (SELECT count(*) FROM runs), (SELECT count(*) FROM events)').fetchone())
           print(label, 'schema=', c.execute('PRAGMA user_version').fetchone()[0],
                 'runs/events=', counts, 'run_id=', last['run_id'])
       for name in p.KEYS:
           rows = sum(1 for _ in p.replay(ledger, last['run_id'], name))
           assert rows == last['tables'][name]['rows'], (name, rows)
           assert p.file_hash(source / f'{name}.parquet') == last['tables'][name]['sha256'], name
           print(label, name, 'replay_rows=', rows, 'source_sha256=', last['tables'][name]['sha256'])
       return counts, last

   before, previous = verify('BEFORE')
   result = p.snapshot(source, ledger)
   assert result['status'] == 'NO_CHANGE', result['status']
   after, latest = verify('AFTER')
   assert latest['run_id'] == result['run_id']
   assert after == (before[0] + 1, before[1]), (before, after)
   assert all(sum(t['counts'].values()) == 0 for t in result['tables'].values())
   print('NO_CHANGE VERIFIED: runs+1 events+0', before, '->', after)
   PYVERIFY
   ```

   預期相對值是 `runs+1、events+0`，不是 events 必須為0；若仍是前輪唯讀觀測的 `(2, 0)`，本次應為 `(3, 0)`，但不得把舊觀測當成現況。v2 首次成功續寫會同交易升至 schema v3、加入封存業務鍵稽核標記；原有 run_id／blobs／events 不改。命令列出的主 checkout 必須先含本候選，不能用舊 HEAD 代替驗收。最後另核對 upstream 原始退出碼／結果未變。
5. Human 另行批准後才同時啟用 task 與事件 trigger；驗一次真正 EventID=102 接線、退出碼與無重疊執行，失敗即停用新 task。睡眠／登出／停電與備份恢復仍須另驗，本輪沒有註冊、Run、啟用、wakeups 或任何排程操作。

### 修補 identity 與 knowledge gate

- 工具 SHA-256：`8604e766ce75df2475747b7a6bba0b4358d3ee364b086250c593081d1f8335fc`。
- 測試 SHA-256：`b73c516be0b1bd91538618fec93fc684b8ab33ef11befec0acbc65063837c8ab`。
- 環境沿用 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`、Python3.12.3、WSL Linux CPU；本worktree Data realpath為同名本地目錄，`MARKETMAMBA_DATA_ROOT`未設；新增測試來源／帳本均pytest暫存目錄，未接觸正式來源或實際帳本。
- Knowledge-impact 判斷：Semantics／Correctness gate／worktree責任與next action＝YES，因此更新Current State／Authority Map／Data；Authority／subsystem boundary／active runtime／canonical artifact supersession／deprecation＝NO。machine authority、歷史證據與外部observed_at不變。
- `check_knowledge_impact.py`＝WARNING：要求檢閱GOVERNANCE；已閱讀，全球治理政策未改，無需修改。`knowledge_health.py --ci`＝exit0／Overall WARNING；Project Map、Agent Guides、Domains、Runtime Freshness、Generated Index皆PASS；Governance／Authority Drift的8項AUTHORITY_REPIN_CANDIDATE及Legacy Docs drift保留，無structural FAIL。
- AST語法2檔、原JSON解析、修補區7個本機reference及 `git diff --check` PASS。起始無dirty／staged／untracked；本輪依no-commit保留6個指定相關modified，沒有staged／untracked，不宣稱clean-worktree完成。


### 修補後指定全套 pytest 實際輸出

```bash
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

**exit0；1320 passed／4 skipped／5 warnings／0 failed，709.05秒**。未新增skip，沒有放寬來源或接受門檻；完整實際stdout如下：

```text
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 16%]
........................................................................ [ 21%]
........................................................................ [ 27%]
........................................................................ [ 32%]
........s............................................................... [ 38%]
.................................s...................................... [ 43%]
........................................................................ [ 48%]
........................................................................ [ 54%]
........................................................................ [ 59%]
........................................................................ [ 65%]
........................................................................ [ 70%]
..........s............................................................. [ 76%]
........................................................................ [ 81%]
........................................................................ [ 87%]
...................................................................s.... [ 92%]
........................................................................ [ 97%]
............................                                             [100%]
=============================== warnings summary ===============================
research_lab/tests/test_p4.py::test_zip_inventory_and_payload_gates[duplicate]
  /usr/lib/python3.12/zipfile/__init__.py:1620: UserWarning: Duplicate name: 'folds/fold_2021/X.npy'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-320/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1320 passed, 4 skipped, 5 warnings in 709.05s (0:11:49)
```

Completed：REQUIRED修補候選與本機工程驗證完成，非Human acceptance。
Changed：tools/pit_ledger.py、tests/test_pit_ledger.py、本報告與Current State／Authority Map／Data note，共6檔。
Decisions：排程NO_GO保留，SQLite暫存MEMORY；歷史帳本／machine authority不改。
Evidence：RED21 failures、focused43 passed、兩項新mutation killed、實體fault恢復、全套1320 passed／0 failed；identity及限制見本節。
Remaining／Blockers：獨立複審、全量RAM峰值、Windows／事件接線／PowerShell／XML及斷電驗收仍待。
Next recommended action：獨立複審後，由Human另案進行註冊前程序；本輪不註冊或啟用。
Project memory updated／Authority map updated：YES，候選狀態與邊界；machine authority未變。
Git status：6 modified、0 staged、0 untracked；依no-commit指示保留dirty patch。
Commit／Push：未執行。
Worktree：/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww6g0bv-340962/builder_pit_fix1，detached HEAD 47bf4659e1143269b622bfd80ebed687f3538973。
Runtime/environment：WSL Linux CPU／Python3.12.3，測試均獨立暫存來源／帳本；無正式Data／實際帳本寫入。

## 複審 Findings 修補 — 2026-11-25

本節 supersede 上輪「額外 schema 皆 STOP」的過度宣稱；上輪全套輸出只代表其當時環境。複審另一次全套為82 failed／1238 passed，不能寫成通過。本輪從 detached HEAD `04e724dd09ef0d44974dfae9e235e777937e6784`、clean worktree開始，指派 `job-run-mww7h6b0-bba9f3/builder_pit_fix2`，只完成有界候選修補，**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。註冊／排程 NO_GO 保留。

- P1：移除 `LIKE 'sqlite_%'`，完整 sqlite_master 所有物件的 `(type,name,tbl_name)` 集合及正規化 SQL SHA-256 必須精確相同；自動索引 SQL 為 NULL 時也保留於集合。額外 trigger／table／index／view 一律 STOP。`sqliteXdrop BEFORE INSERT ... RAISE(IGNORE)` 回歸確認 validate／snapshot 均拒絕，runs仍1、events仍0。
- P2：舊 schema v2 validate 完整解碼不可變索引的業務鍵，拒絕 None／NaN／空字串／ASCII空白／全形空白／NBSP；同 SHA NO_CHANGE 不可略過。欄位集合或 JSON 不合法亦 STOP。合成估時超過30秒，因此採核准備案：成功 snapshot 才同 SQLite transaction 升至 `PRAGMA user_version=3`，在受 run digest 封存的 `business_key_audit` 記下四份初始索引 SHA。新建帳本由現行 records gate驗鍵後建立標記；後續 validate仍逐次核 blob SHA、run chain與標記 identity，不重複解碼已稽核索引。v2唯讀 validate不寫標記，故直到一次成功續寫前仍會重掃；v1／v3缺失或不符標記 STOP。失敗續寫 rollback保留v2／原run，重試可正常升級；舊run／events／blobs不重寫。資料格式 VERSION仍為 `pit-local-observation-v2`；schema3是新增驗證標記契約，舊版工具會拒絕schema3，後續操作須使用本候選或相容版本。
- P3：**已知限制：單機無惡意者威脅模型下接受**。symlink 檢查與 pathname 寫入間 TOCTOU 競態未修；靜態祖先／後代拒絕不能證明不跟隨 symlink 的安全寫入邊界。使用者明確指定威脅模型為意外損壞／程式錯誤、要求不實作競態防護；不得再宣稱競態已防護。
- 第4步已改為可複製的主 checkout／指定 Python 命令，先後各以唯讀 validate＋四表完整 replay，確認來源 identity／列數，明確斷言 NO_CHANGE時runs+1／events+0。此命令有另案snapshot寫入，本輪完全未執行。

### 本輪 RED → GREEN 實際輸出

命令均使用 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:.` 與 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，不設定外部TMPDIR，pytest停用cacheprovider。所有來源／帳本為獨立暫存合成資料。

```text
# 修實作前：pytest tests/test_pit_ledger.py -k 'extra_schema or legacy_v2' -q -p no:cacheprovider
10 failed, 43 deselected in 3.64s
# 一次性升級實作前：pytest tests/test_pit_ledger.py -k clean_v2_upgrade -q -p no:cacheprovider
1 failed, 53 deselected in 1.55s
# 最終 focused：pytest tests/test_pit_ledger.py -q -p no:cacheprovider
........................................................                 [100%]
56 passed in 41.89s
```

新增13個參數化案例涵蓋四種額外schema物件、六種舊無效鍵、乾淨v2一次稽核及NO_CHANGE計數／歷史replay、v3缺標記與v2升級交易rollback。原SIGKILL／RLIMIT_FSIZE及mutation回歸亦在56項內；仍未證明真正ENOSPC或實體斷電耐久性。

### 合成性能及 identity

本機CPU、Python3.12.3／SQLite3.45.1／既有venv，與全套同時量測，非正式Data效能保證。兩份各100萬唯一業務鍵、batch32768、zstd索引（value_hash為固定32-byte合成佔位）；僅估計完整鍵語義稽核CPU成本，不代表正式hash欄位的壓縮率／I/O：

```text
INDEX_AUDIT name=revenue_raw rows=1000000 bytes=985880 sha256=a41a3fe72dc9a8de24d1085398ec9f20a00f37e72a73b61d44da1ac8df907b3c seconds=2.784322 projected_13760451_seconds=38.313528 hash_seconds=0.001327
INDEX_AUDIT name=financials_raw rows=1000000 bytes=1124865 sha256=9190c9d785ea80f3624caa6bd9ce546a6128c822bb37da74d75cb5ca86419fac seconds=2.814704 projected_13760451_seconds=38.731600 hash_seconds=0.001406
UPGRADE_PERF rows=100000 blobs_bytes=2274957 upgrade=NO_CHANGE upgrade_seconds=0.443616 repeat=NO_CHANGE repeat_seconds=0.176507 v3_validate_seconds=0.005552 projected_v3_validate_13760451_seconds=0.763963 runs_events=(3, 0) db_sha256=aa15a5f238af53e73f38648d5cdead7b45eea5b90e56ccde835e2cb2a8f2b951
```

前兩次超過30秒目標而採一次性升級；最後一次為四表各25000鍵、正式hash索引生成流程的隔離帳本。0.764秒只為小型合成validate的線性投影，不包含正式全量磁碟／日後event chain增長成本。未讀寫主checkout實際帳本／formal Data，也未新增production authority或外部runtime observation。

工具 SHA-256：`4206397d9b501ae028b9d4c63aef99a2dd3dafd93009a945fe223362dec3a171`；測試 SHA-256：`59dab69553112caf939fbbd3a011edba1949a22e6295f811ec41d7944477ead4`。

### 本輪 knowledge-impact 與邊界檢查

Semantics／Correctness gate／worktree責任＝YES（舊帳本鍵稽核、schema3升級標記）；Authority／subsystem boundary／active runtime／canonical artifact supersession／deprecation＝NO。Major blocker／next decision仍為獨立複審及另案Human驗收，未因builder測試解除。已同change更新Current State／Authority Map／Data；`check_knowledge_impact.py` 初次PASS，加入knowledge後WARNING要求檢閱GOVERNANCE，已閱讀，無需改全球政策。`knowledge_health.py --ci` exit0：Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，Overall WARNING，無structural FAIL。AST兩檔、報告驗收命令AST、新增本機links／section references及`git diff --check` PASS。

Runtime為本指派WSL worktree；Data realpath是worktree下同名普通目錄，`MARKETMAMBA_DATA_ROOT`未設定。使用指定既有Python3.12.3 venv；來源／帳本／benchmark均獨立TEMPORARY合成artifact，沒有正式Data／主checkout帳本寫入，沒有網路／GPU／agents／P4d／protected V6/models／排程／production／commit／push／merge。本輪builder自查使用TDD、verification-before-completion、code-review-and-quality及專案ml-leakage-audit；未冒稱獨立審查。

### 本輪指定全套 pytest 實際輸出（最終 source，exit0）

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

```text
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 16%]
........................................................................ [ 21%]
........................................................................ [ 26%]
........................................................................ [ 32%]
........s............................................................... [ 37%]
.................................s...................................... [ 43%]
........................................................................ [ 48%]
........................................................................ [ 53%]
........................................................................ [ 59%]
........................................................................ [ 64%]
........................................................................ [ 70%]
..........s............................................................. [ 75%]
........................................................................ [ 80%]
........................................................................ [ 86%]
...................................................................s.... [ 91%]
........................................................................ [ 96%]
.........................................                                [100%]
=============================== warnings summary ===============================
research_lab/tests/test_p4.py::test_zip_inventory_and_payload_gates[duplicate]
  /usr/lib/python3.12/zipfile/__init__.py:1620: UserWarning: Duplicate name: 'folds/fold_2021/X.npy'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-327/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1333 passed, 4 skipped, 5 warnings in 749.61s (0:12:29)
```

本次0 failed；4 skipped及5 warnings如上，不能稱零skip／零warning。一次性升級調整前啟動的中間全套已中止（exit130），不作最終驗證依據；以上完整執行是在最終source／tests凍結後啟動。

Completed：複審三項Findings依核准範圍處理；builder工程驗證完成，不是Human acceptance。
Changed：tools/pit_ledger.py、tests/test_pit_ledger.py、reports/pit_arrival_audit_20261125.md、knowledge/00_Project_Map/Current_State.md、knowledge/00_Project_Map/Authority_Map.md、knowledge/01_Domains/Data.md。
Decisions：完整schema集合；舊v2無效鍵STOP，採一次稽核後原子schema3升級與封存標記；symlink競態依單機無惡意者模型接受為已知限制；排程NO_GO不變。
Evidence：RED10 failed、升級RED1 failed；最終focused56 passed；全套1333 passed／4 skipped／5 warnings／0 failed、749.61秒、exit0；性能及hash如本節；AST／links／diff檢查通過，knowledge health WARNING／exit0，無structural FAIL。
Remaining：獨立複審、Human另案主帳本驗收、真實全量耗時／RAM、Windows接線、真正ENOSPC／斷電耐久性未驗。
Blockers：本輪有界實作無阻塞；註冊／啟用仍NO_GO待獨立審查與另案Human接受。
Next recommended action：獨立複審此patch，再另案依報告第4步驗收；不直接註冊或啟用。
Project memory updated：YES，Current State與Data note。
Authority map updated：YES，候選gate與接受限制；machine authority未改。
Git status：起始clean；最終6 modified、0 staged、0 untracked；依no-commit要求保留dirty patch，不宣稱clean worktree。
Commit：未執行（明確禁止）。
Push：未執行。
Worktree：/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mww7h6b0-bba9f3/builder_pit_fix2，detached HEAD 04e724dd09ef0d44974dfae9e235e777937e6784。
Runtime/environment：WSL Linux CPU；/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python，Python3.12.3／SQLite3.45.1；MARKETMAMBA_DATA_ROOT未設；獨立合成暫存資料，主帳本／formal Data未寫入。
