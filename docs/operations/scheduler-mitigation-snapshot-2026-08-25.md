# Windows Task Scheduler mitigation snapshot — 2026-08-25

Purpose: preserve the pre-mitigation scheduler state before temporarily suppressing
Market Mamba inference because of a suspected Windows/WSL host-memory-pressure incident.
This is a mitigation record, not a root-cause conclusion.

Snapshot source: live Windows Task Scheduler queries (`Get-ScheduledTask` and
`Get-ScheduledTaskInfo`) on 2026-08-25 Asia/Taipei. No task had been changed when this
snapshot was taken.

## Tasks found

### `\MarketMamba_V62`

- Classification: Market Mamba data fetch + V6.2 inference/model/portfolio/publish pipeline (combined)
- State: `Ready`
- Enabled: `true`
- Last run: `2026-08-25 22:15:00 +08:00`
- Last result: `255`
- Next run at snapshot: `2026-08-26 22:15:00 +08:00`
- Trigger: weekly, Monday-Friday, start boundary `2026-08-05T22:15:00+08:00`
- Trigger enabled: `true`
- `StartWhenAvailable=true`, `WakeToRun=true`, `MultipleInstances=IgnoreNew`
- Execution time limit: `PT3H`
- Principal: user `Master`, interactive-token logon, least privilege
- Action executable: `wscript.exe`
- Action arguments:
  `"D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\run_hidden.vbs" "D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\v62_daily.bat"`
- Working directory: unset
- Exact WSL command reached through the VBS and BAT wrappers:

  ```text
  wsl -d Ubuntu -- bash -lc "source /home/frank/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_v62_daily.py 2>&1 | tee -a V6/logs/v62_daily.log; exit ${PIPESTATUS[0]}"
  ```

- Snapshot-time Python behavior: `run_v62_daily.py` called the shared `run_daily_update()` fetcher,
  then performs data checks, feature construction, Mamba inference, baseline model execution,
  portfolio updates, performance aggregation, git push, and cache refresh. It supports
  `--no-fetch`, but had no fetch-only option at the time this pre-mitigation snapshot was
  taken.

### `\PersonalOS_Daily`

- Classification: combined PersonalOS daily orchestration containing Market Mamba V6.1 fetch + inference,
  dual-model inference, plus non-Market-Mamba portfolio sync and financial-news work
- State: `Ready`
- Enabled: `true`
- Last run: `2026-08-25 21:30:01 +08:00`
- Last result: `3221225786` (`0xC000013A`)
- Next run at snapshot: `2026-08-26 21:30:00 +08:00`
- Trigger: daily, start boundary `2026-08-05T21:30:00+08:00`
- Trigger enabled: `true`
- `StartWhenAvailable=false`, `WakeToRun=false`, `MultipleInstances=IgnoreNew`
- Execution time limit: `PT72H`
- Principal: user `Master`, interactive-token logon, least privilege
- Action executable: `D:\Desktop\work\ProjectForMe\PersonalOS\scripts\run_daily.bat`
- Action arguments: unset
- Working directory: unset
- BAT command:

  ```text
  C:\Users\Master\anaconda3\python.exe scripts\run_daily.py
  ```

- Exact WSL commands reached on an open trading day:

  ```text
  wsl -d Ubuntu -e bash -c "source /home/frank/miniconda3/etc/profile.d/conda.sh && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && conda run -n mamba_env python V6/run_daily_inference.py 2>&1 | tee -a V6/logs/inference.log; exit ${PIPESTATUS[0]}"

  wsl -d Ubuntu -e bash -c "source /home/frank/miniconda3/etc/profile.d/conda.sh && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && conda run -n mamba_env python V6/run_dual_inference.py --push 2>&1 | tee -a V6/logs/dual_inference.log; exit ${PIPESTATUS[0]}"
  ```

- Actual Python behavior: `run_daily_inference.py` calls `run_daily_update()` first, then
  sanitizes/builds features and runs V6.1 inference and downstream publishing. There is no
  fetch-only option. `run_daily.py` only launches dual inference after V6.1 succeeds. The
  same `PersonalOS_Daily` task also runs portfolio sync and financial-news jobs, so disabling
  the whole task would affect unrelated PersonalOS work.

### `\PersonalOS_Morning`

- Classification: non-Market-Mamba morning financial-news task; explicitly excluded from mitigation
- State: `Ready`
- Enabled: `true`
- Last run: `2026-08-25 08:30:01 +08:00`
- Last result: `0`
- Next run at snapshot: `2026-08-26 08:30:00 +08:00`
- Trigger: daily at 08:30, start boundary `2026-05-04T08:30:00`
- Action executable: `D:\Desktop\work\ProjectForMe\PersonalOS\scripts\run_morning.bat`
- BAT command: `C:\Users\Master\anaconda3\python.exe scripts\run_daily.py --mode morning`
- No Market Mamba fetch or model inference is invoked.

## Scope exclusions

- No MAS task was queried by name for mutation, changed, disabled, or enabled.
- No unrelated Windows scheduled task was changed.
- `PersonalOS_Morning` was inspected only because the broad candidate search matched
  `PersonalOS`; it is not part of the Market Mamba mitigation.

## Restore reference

At snapshot time both inference-bearing tasks were enabled. If a later mitigation disables
them, the scheduler-only restoration commands are:

```powershell
Enable-ScheduledTask -TaskPath '\' -TaskName 'PersonalOS_Daily'
Enable-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_V62'
```

Any temporary fetch-only action or orchestrator flag must be reverted separately according
to the eventual implementation record; enabling the original combined tasks before reverting
that change could duplicate data fetching or re-enable inference unexpectedly.

## State after user mitigation

After the initial snapshot, the user manually disabled both inference-bearing tasks in
Windows Task Scheduler. A live read-only query subsequently verified:

- `\MarketMamba_V62`: `State=Disabled`, `Enabled=false`; original action unchanged.
- `\PersonalOS_Daily`: `State=Disabled`, `Enabled=false`; original action unchanged.

No scheduler task was changed by Codex. `\PersonalOS_Morning`, MAS tasks, and all unrelated
tasks remain outside the mitigation scope.

## Dedicated fetch-only task created

With user approval, a new task was registered after the two combined tasks had been disabled:

### `\MarketMamba_DataFetch`

- State after registration: `Ready`
- Enabled: `true`
- First/next run: `2026-08-26 22:30:00 +08:00`
- Trigger: weekly, Monday-Friday at 22:30 (`DaysOfWeek=62`)
- `StartWhenAvailable=true`, `WakeToRun=true`, `MultipleInstances=IgnoreNew`
- Execution time limit: `PT3H`
- Principal: user `Master`, interactive-token logon, least privilege
- Action executable: `wscript.exe`
- Action arguments:
  `"D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\run_hidden.vbs" "D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\v62_daily.bat" --fetch-only`
- Last run: never (Task Scheduler's uninitialized sentinel was returned)
- The task was not manually started during setup.

Post-registration verification also confirmed that `\MarketMamba_V62` and
`\PersonalOS_Daily` remained disabled with their original actions unchanged.

To temporarily disable or later restore only this fetch task:

```powershell
Disable-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_DataFetch'
Enable-ScheduledTask  -TaskPath '\' -TaskName 'MarketMamba_DataFetch'
```

## Repository-state convergence record — 2026-08-29

This section records the repository-visible state after the temporary mitigation was made
reviewable. It does not decide the long-term policy, restore full V6.2 inference, or mark
the host-memory incident root cause as resolved.

- `V6/run_v62_daily.py` now has an explicit `--fetch-only` mode. That mode calls
  `fetch_data()`, runs the strict daily-source freshness check, returns non-zero when any
  daily source is missing, and stops before importing `run_v62_inference` or
  `v62_portfolio`.
- `V6/scripts/v62_daily.bat` forwards one optional argument to `run_v62_daily.py`, which
  is sufficient for the scheduler action currently used by `\MarketMamba_DataFetch`:
  `--fetch-only`.
- `V6/scripts/test_v62_fetch_only.py` is the regression entrypoint for this mitigation
  path. It mocks `fetch_data()` and asserts that fetch-only does not import the heavy
  inference or portfolio modules.
- Current publication state remains intentionally separate from ingestion state:
  `V6/results/v62_state_*.json`, `V6/results/v62_portfolio_*.jsonl`, and `df_v62*.csv`
  still reflect `2026-08-12` publication artifacts, while `V6/logs/v62_daily.log` shows
  fetch-only ingestion succeeded for `2026-08-28` with 10/10 daily sources complete.

Verification performed for this convergence record:

```text
python3 V6/scripts/test_v62_fetch_only.py
Ran 2 tests in 0.004s
OK
```
