' ============================================================
' run_hidden.vbs — 隱藏 Task Scheduler 跑 .bat 時的小黑窗
' ============================================================
' 問題：Task Scheduler 用 `cmd.exe /c xxx.bat` 執行時會跳出一個
'       什麼也沒顯示的主控台視窗。它沒有任何作用，但每天彈一次很干擾。
'
' 為什麼不用 Task Scheduler 內建的「不論使用者登入與否都執行」來隱藏：
'   那會讓工作跑在沒有互動式桌面的 session 下 → **WSL2 起不來**
'   （daily_inference.bat 的檔頭已經記過這個坑），而且 WSLg 的進度視窗
'   也會一起看不到。我們要隱藏的只有那個沒用的黑窗，不是整個工作。
'
' 用法（Task Scheduler 的 Action 改成這樣）：
'   Program/script : wscript.exe
'   Arguments      : "D:\...\V6\scripts\run_hidden.vbs" "D:\...\V6\scripts\v62_daily.bat" --first-day
'
' ⚠️ 刻意用 bWaitOnReturn=True 並把 exit code 回傳給 Task Scheduler。
'    設 False 的話工作會「立刻成功」——排程紀錄永遠是綠的，
'    失敗完全看不出來。那比小黑窗糟糕得多。

Option Explicit

Dim sh, args, i, cmd, rc

If WScript.Arguments.Count = 0 Then
    WScript.Echo "用法：wscript run_hidden.vbs <要執行的程式> [參數...]"
    WScript.Quit 2
End If

Set sh = CreateObject("WScript.Shell")

' 第 0 個參數是要跑的程式，其餘原樣轉傳（各自加引號，路徑含空白才不會被拆開）
cmd = """" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
    cmd = cmd & " """ & WScript.Arguments(i) & """"
Next

' 0 = 視窗隱藏；True = 等它跑完才回來（才拿得到 exit code）
rc = sh.Run(cmd, 0, True)

WScript.Quit rc
