$ErrorActionPreference = 'Stop'
$taskName = 'MarketMamba_PITLedger'
if (Get-ScheduledTask -TaskPath '\' -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already exists: STOP' }
$null = Get-ScheduledTask -TaskPath '\' -TaskName 'MarketMamba_DataFetch'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$sub = '<QueryList><Query Id="0" Path="Microsoft-Windows-TaskScheduler/Operational"><Select Path="Microsoft-Windows-TaskScheduler/Operational">*[System[EventID=102]] and *[EventData[Data[@Name="TaskName"]="\MarketMamba_DataFetch"]]</Select></Query></QueryList>'
$esc = [Security.SecurityElement]::Escape($sub); $escId = [Security.SecurityElement]::Escape($identity)
$xml = @"
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>MarketMamba research-only PIT ledger snapshot after successful DataFetch. Reads Data, writes only .artifacts/pit-ledger-v2. Log: D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.log</Description></RegistrationInfo>
  <Triggers><EventTrigger><Enabled>true</Enabled><Subscription>$esc</Subscription><Delay>PT2M</Delay></EventTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$escId</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>false</StartWhenAvailable><WakeToRun>false</WakeToRun><Enabled>true</Enabled><ExecutionTimeLimit>PT2H</ExecutionTimeLimit><Hidden>false</Hidden></Settings>
  <Actions Context="Author"><Exec><Command>powershell.exe</Command><Arguments>-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "D:\Desktop\work\ProjectForMe\MarketMamba-ops\pit_ledger_task.ps1"</Arguments></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskPath '\' -TaskName $taskName -Xml $xml -User $identity | Select-Object TaskName,State | Format-List
