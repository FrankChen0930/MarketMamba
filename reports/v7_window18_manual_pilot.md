# Window18 Limited Real Forward Collector Pilot

Completed / STOP / UNMERGED / NOT_DEPLOYED。僅一次有限manualpilot批准；daily／researchadmission未批准。

W17 main ff-only：`000b0b9c40448b11a41d05396dffc6e0f3ea1235` →
`96e8c0c4f2f7f78c508048e31f0f163d6c707128`。38clean worktrees／19sourcehashes／9tests與既有GO重驗。
W18 base同後者，branch `development/v7-window18-manual-collector`，finalHEAD為含本收尾commit。
W06／W12未整合；W14/W16/W17原始資料及W12HALTED保持，129舊protectedfiles、5W16durable及8W17officialfiles hashes不變。

| Approved CSV | HTTP | Observation session | Rows / securities | Raw bytes | 修後離線基本QC |
|---|---:|---|---:|---:|---|
| TWSE11549 | 200 | 2026-10-22 only | 1383 / 1383 | 137956 | 1368PASS、15MISSING_OHLCV_NO_FILL |
| TPEx11371 | 200 | 2026-10-22 only | 1017 / 1017 | 141480 | 990PASS、27MISSING_OHLCV_NO_FILL |

- 各1GET，0retry／redirect／fallback；最大2MiB，25sec socket及40sec實際timeout程序。額度先持久記帳，已耗盡。
  精確URLs及OGDL1.0顯名沿用[W15 Matrix](../research/v7/window15-license-matrix.json)，不要求額外官方書面回覆。
- [首次真實結果](../research/v7/window18-capture-results.json)保留UTC／Taipei receipt、HTTPmetadata、schema、raw／manifest／record路徑與完整SHA。
  日期及body與W16相同不等於重新取得原歷史vintage；publication維持UNKNOWN。
  分類`MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED`，非VERIFIED_FORWARD_VINTAGE／正式研究資料。
  TWSEfreshness僅W17PROVISIONAL boundedcalendar match；TPExcalendar未知，不以牆上日期判stale。
  商品不自動等於普通股Universe；不將停牌／無成交／missing補零，不認證adjustment／currency／可交易性。
- Raw於 `/home/frank/projects/MarketMamba/.artifacts/window18-manual-pilot/snapshots/raw/`：
  TWSE `79bdd059e75a0a829dfef1e5c1330cb48a84b302e04ce040e77ee0dd35e5ec6a.bin`；
  TPEx `c57a72d7864203d633c76716e56c35b689c0134fffbed7f069e41f2646372d1a.bin`。
  同root的`snapshots/manifests`與`captures`保存實際Bytes／來源與時間證據；初始/source-reviewed程式分開保存。
  [實體Inventory](../research/v7/window18-local-artifacts.json)列全部檔案Hash；Git不包含CSV，沒有只有Hash的假還原。
- 兩次獨立CLI子程序[初次Restore](../research/v7/window18-initial-restore.json)及修後兩次
  [Restore](../research/v7/window18-reviewed-restore.json)均驗279436bytes、hash、1383/1017rows。
  Restore目錄分別`.artifacts/window18-manual-restore`／`.artifacts/window18-reviewed-restore`，獨立於W18工作樹，
  仍同機同filesystem，**BACKUP_PENDING**。無已核准獨立備份位置；未使用雲端／Windows新位置。
- 原TPExrecord為`RAW_CAPTURED_WITH_QC_FAILURE`、27INVALID；已知漏辨識`----`。
  原capture/QC/source及首次退出碼2不改寫；[新離線QC](../research/v7/window18-qc-reassessment.json)另存27MISSING。
  reader明確重播已知舊QC版本，Restore同時列原QC與修後QC。兩實際回應未保存Content-Length，
  framing明示UNKNOWN，不能宣稱核對來源宣告長度或完整市場coverage。
- 最終25targetedtestsPASS（16collector/review＋9W17）；integration9PASS。一次獨立review Critical0／Required3，
  初判NO-GO保留；唯一修復輪R1token／R2短讀／R3已發布raw不得隱性修補均RED→GREEN及實際Restore自行驗收。
  **沒有第二次獨立GO**。[審查與修復](../research/v7/window18-review.json)、[測試](../research/v7/window18-targeted-tests.txt)。
- Manual CLI能力已由真實流程驗證，現W18grant不可再下載。未來新授權須另定run額度，不能移除本次ledger重試。
  CLI：`python -B -m V6.experimental.v7_manual_collector {live,offline,restore} --help`。
  Offline需`--root --dataset --input --retrieved-at`；Restore需`--root --record --record-sha --destination`。
  W15adapter仍Synthetic-only；本入口重用v1Snapshot／W09safe+lock／W07clocks／W17calendar，無新平台。

下一次每日啟用最少Human決策：[候選](../research/v7/window18-daily-activation-candidate.json)的獨立WSL環境／排程、
每實際交易日一次與19:00候選（尚未凍結、更新時刻未驗證）、各市場有效日曆、來源更新／QC／缺漏政策、
有界retry／停止恢復撤銷、指定獨立backup與驗收、監測通知及missed-day backfill。兩個exactCSV維持primary，
fallback空；晚補不冒充準時forward。WindowsV6排程隔離，不啟用任何新任務。
HistoricalStudy PROPOSED／StrictSTOP／E1BLOCKED／H2NOT_APPROVED／GPU_SAFETY_HALTED，無訓練或RankIC。

恢復：`cd /home/frank/projects/MarketMamba/.worktrees/v7-window18-manual-collector`，
`git status --short --branch`、`git rev-parse HEAD`，讀Current_State → Authority_Map → 本報告／
[唯一Backlog](../research/v7/research-program.md#window18-limited-real-forward-collector-pilot)／[evidence](../research/v7/window18-evidence.json)。
ProjectMemory／AuthorityMap已更新；Runtime W11CPUoverlay Python3.12.3，DataRoot env UNSET，Windowsoperational不讀payload。
Push/Deploy/production變更 N/A；不mergeW18，不開下一窗口，GET不可再執行。
