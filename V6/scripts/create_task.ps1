$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c "D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\daily_inference.bat"'

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '17:00'

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -WakeToRun:$false

$principal = New-ScheduledTaskPrincipal -UserId 'Master' -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName 'MarketMamba_DailyInference' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'MarketMamba V6 daily inference at 17:00 on weekdays via WSL2' `
    -Force

Write-Host "Task 'MarketMamba_DailyInference' created successfully."
Write-Host "Next run: weekdays at 17:00"
