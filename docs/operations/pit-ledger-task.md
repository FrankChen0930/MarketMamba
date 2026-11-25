# PIT ledger daily task (registered 2026-11-25)

- Windows task `\MarketMamba_PITLedger`, registered 2026-11-25 22:5x Asia/Taipei by Claude with Human approval.
- Trigger: Task Scheduler Operational EventID 102 for `\MarketMamba_DataFetch`, delay 2 min. Interactive logon only, no wake, IgnoreNew, 2 h limit.
- Action: `D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.ps1` (copy: [pit_ledger_task.ps1](pit_ledger_task.ps1)). Skips (exit 2) unless DataFetch is not running, LastTaskResult=0 and ran within 6 h; then runs `python -m tools.pit_ledger --ledger .artifacts/pit-ledger-v2` in WSL distro `Ubuntu-24.04` from the main checkout. Log: `D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.log`.
- Registration script copy: [register_pit_ledger_task.ps1](register_pit_ledger_task.ps1). Revoke: `Unregister-ScheduledTask -TaskName MarketMamba_PITLedger -Confirm:$false` (does not touch DataFetch or the ledger).
- Acceptance 2026-11-25: manual snapshot after successful DataFetch → CHANGED (financials FIRST_SEEN 19, REVISED 21), runs/events 2/0 → 3/40, schema 2→3, four-table replay rows equal source, 551 s incl. one-time upgrade, max RSS 1.6 GB. Task test run via Start-ScheduledTask → NO_CHANGE, LastTaskResult 0, 2.4 s.
- Evidence class: local cache observation only (FinMind/MOPS ingest time as seen locally), not issuer publication time. Event-trigger path from a real DataFetch completion is first exercised 2026-11-26 22:30.
