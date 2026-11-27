# Runs the research-only PIT ledger snapshot after MarketMamba_DataFetch succeeds.
$ErrorActionPreference = 'Stop'
$log = 'D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.log'
function Log($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format o), $m) }
$t = Get-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_DataFetch'
$i = $t | Get-ScheduledTaskInfo
if ($t.State -eq 'Running' -or $i.LastTaskResult -ne 0 -or $i.LastRunTime -lt (Get-Date).AddHours(-6)) {
  Log ("SKIP fetch not verified: state={0} result={1} last={2}" -f $t.State, $i.LastTaskResult, $i.LastRunTime); exit 2
}
$out = & wsl.exe -d Ubuntu-24.04 -- bash -lc 'set -o pipefail; cd /home/frank/projects/MarketMamba && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.pit_ledger --ledger /home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2 2>&1 | tail -c 2000'
$code = $LASTEXITCODE
Log ("exit={0} {1}" -f $code, (($out | Out-String) -replace "`r?`n", ' | '))
# External public datasets (research-only): independent of the ledger result; never changes its exit code.
$ext = & wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /home/frank/projects/MarketMamba && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.external_snapshot 2>&1 | tail -c 1000'
Log ("external exit={0} {1}" -f $LASTEXITCODE, (($ext | Out-String) -replace "`r?`n", ' | '))
exit $code
