# PIT ledger daily task (registered 2026-11-25)

- Windows task `\MarketMamba_PITLedger`, registered 2026-11-25 22:5x Asia/Taipei by Claude with Human approval.
- Trigger: Task Scheduler Operational EventID 102 for `\MarketMamba_DataFetch`, delay 2 min. Interactive logon only, no wake, IgnoreNew, 2 h limit.
- Action: `D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.ps1` (copy: [pit_ledger_task.ps1](pit_ledger_task.ps1)). Skips (exit 2) unless DataFetch is not running, LastTaskResult=0 and ran within 6 h; then runs `python -m tools.pit_ledger --ledger .artifacts/pit-ledger-v2` in WSL distro `Ubuntu-24.04` from the main checkout. Log: `D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.log`.
- Registration script copy: [register_pit_ledger_task.ps1](register_pit_ledger_task.ps1). Revoke: `Unregister-ScheduledTask -TaskName MarketMamba_PITLedger -Confirm:$false` (does not touch DataFetch or the ledger).
- Acceptance 2026-11-25: manual snapshot after successful DataFetch → CHANGED (financials FIRST_SEEN 19, REVISED 21), runs/events 2/0 → 3/40, schema 2→3, four-table replay rows equal source, 551 s incl. one-time upgrade, max RSS 1.6 GB. Task test run via Start-ScheduledTask → NO_CHANGE, LastTaskResult 0, 2.4 s.
- Evidence class: local cache observation only (FinMind/MOPS ingest time as seen locally), not issuer publication time. Event-trigger path from a real DataFetch completion is first exercised 2026-11-26 22:30.

## 外部公開資料每日觀測（2026-11-28 新增）

在 PIT ledger 之後，同一個排程會再執行 `python -m tools.external_snapshot`，對 `.artifacts/external/` 底下的 git clone（目前只有 Sinotrade/shioaji-data 的關係資料）做 `pull --ff-only`，並在 `.artifacts/external/observations.jsonl` 追加一筆紀錄，內容包括我方觀測時間、上游 HEAD 與主要檔案的 SHA。上游的 git 歷史會保留每個版本；我方的觀測時間則是點時（PIT）使用時，判斷某個版本「最早可見」的依據。

- 這一步失敗只會寫入 log，不影響 ledger 的 exit code。
- 第三方研究資料的授權仍是「待定」，只保存在本機 artifacts，不進入公開 repo。
