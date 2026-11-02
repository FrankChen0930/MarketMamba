# Window35 — W19 Five-session Closure / Data Recovery v0.4 Offline

Status: COMPLETE / STOP / UNMERGED / NOT_DEPLOYED。主要Review原NO_GO保留；唯一R1已以單輪修復及Independent Closure關閉，工程GO（僅離線Recovery）。15tests PASS，Critical0／Required0；沒有Collector啟用授權。

Base main `c57ac3d7822f229ae08849e56581ebb3c969441c`; branch `research/v7-window35-recovery-v04`, isolated worktree of same name. W19cf1 diverges at87695: no merge or stale settings adoption. Main Snapshot/QC/verified_record/restore can already verify all W19 scheduled records, so no W19 runtime port is necessary for evidence closure. A live normal/recovery dispatcher remains unported/unwired and needs a bounded future adapter before activation. W34fixed82cd is only a read-only64-ID reference; no model was loaded/scored/modified/integrated.

## 五日實際驗收

| Session | TWSE rows / PASS / Missing | TPEx rows / PASS / Missing |
|---|---|---|
| 2026-10-27 | 1386 / 1375 / 11 | 1020 / 993 / 27 |
| 2026-10-28 | 1386 / 1378 / 8 | 1020 / 991 / 29 |
| 2026-10-29 | 1388 / 1365 / 23 | 1017 / 989 / 28 |
| 2026-10-30 | 1389 / 1375 / 14 | 1018 / 995 / 23 |
| 2026-11-02 | 1389 / 1373 / 16 | 1018 / 980 / 38 |

5/5 COMPLETE_NEW_SESSION、10/10 sources、10真實GET已在W19原授權內完成；本輪GET0。Raw重建12031rows/11814basicPASS/217missing/Fatal0；Basic PASS不等於研究准入。8個invocations=5complete+2weekend NON_TRADING_DAY+1授權起點前NOT_STARTED；額外啟動沒有行情GET。

完整Raw而非例外sample：TWSE72missing vs69exceptions，TPEx145 vs100；補回被截斷的48筆明細。涉及121個market/security：缺1天69個、2天23、3天17、4天9、5天3。五日持續者TWSE020011、TPEx020040、TPEx3064，均不因連續性推定停牌/合法無成交。

217筆都缺open/high/low/close，其中45筆另缺volume。35個證券只有code/name特殊商品樣式提示、86個一般四碼類型仍UNKNOWN；樣式不是正式Security Master認證。完整逐證券日期/欄位/原因見window35-missing-distribution.json。沒有足夠文件確認合法無成交或停牌，兩種已確認數均0，原因UNKNOWN；Source-level failure0。Scope與cause分開：全部217筆均OUT_OF_RESEARCH_UNIVERSE，不等於其交易原因已明。

## 固定64檔影響

W34固定TWSE64檔五天每天present64/basicOHLCVvalid64/Feature-inputeligible64，共320個輸入日股票。首日lookback使用已保存W16官方Oct22資料，後續使用前一個實際Pilot Session；重用W29同一feature_at與W34相同兩Feature順序，不讀未來Label，不產生Ranking Score。公式不用volume，但既有input gate要求前後兩日有效OHLC；volume-only缺失不自動當必要Feature缺失。

目前217筆對此Cohort沒有直接Feature阻塞，因此沒有值得為它們執行的內容補抓。若未來研究Universe改變，須重新分類；不把所有非64檔商品永久排除或宣稱都非普通股。

## 原任務期限與保存

唯讀Task觀測2026-11-02T21:27:33.8781610+08：registered/Ready/enabled、lastrun19:00:01/result0、NextRun=null（未回傳值，不等於已停用），Interactive/IgnoreNew/StartWhenAvailable。EndBoundary仍2026-11-02T23:59:59+08，觀測時尚未到期，狀態EXPIRING_PROTECTED_EVIDENCE。政策/code另檢期限、slots、每來源session claim一次；10GET不重置。沒有停止/觸發/延長原Task，也沒有修改Windows權限或ExecutionPolicy。原始.ps1因簽章政策未執行，最後只用null-safe read-only inline查詢成功。

W19 Root原址保留；原始bytes/manifests/records與same-host restore逐一驗證。W35自己的唯讀replay/restore副本不宣稱獨立備份。BACKUP_PENDING：同一實體裝置故障仍可能同時遺失所有Vintage。

## Recovery v0.4 — Offline only

最小新增classify/decide/append-only claims/outcomes與outcome比較，重用source binding、QC、no-clobber writer及file lock；沒有網路dispatch/Collector/Registry/Trainer。狀態含VALID_DATA、LEGITIMATE_NO_PRICE、KNOWN_SUSPENSION、OUT_OF_RESEARCH_UNIVERSE、RECOVERABLE_MISSING_CANDIDATE、SOURCE_LEVEL_FAILURE、UNKNOWN_MISSING、UNRECOVERABLE_AFTER_POLICY_LIMIT；已知停牌/無價必須有匹配市場/日期/證券的來源證據，不能由零量或持續性推定。

必要Feature受影響＋可追溯的合理更新理由＋合法來源與可用配額＋未用內容cycle，才產生補抓計畫；所有decision execution_authorized=False。Whole CSV按market/resource/target-observation-session計數，不能按stock乘大預算；一個額外content cycle上限1。離線參考network retries上限2，與normal1/content1分開，合計physical參考cap4；共享於同resource/session。這只是explicit OFFLINE_REFERENCE，正式Retry間隔/數量/範圍仍待Human凍結。

402/403/429/Source halt持久阻擋同來源後續Session；禁止未核准fallback。回傳不同Session拒絕套用。相同內容不冒稱修復；新內容另存新Vintage，原始不覆寫；仍缺/部分修復後停止，Backfill Queue僅標記待合法歷史來源，不承諾latest CSV可補所有歷史。Backfill/Offline replay永不冒充原始Forward Capture。Synthetic2030時鐘與實際archive時間分開。

## 未來啟用最小方案

候選Normal19:00／Conditional Recovery07:00，Windows Task→固定WSL distro/user/approvedcommit，新V7任務/namespace與V6完全隔離。先確認原W19End已過及budget已消耗，再考慮新限期任務，不能重啟/覆寫W19。Interactive需使用者登入；登出不保證執行，關機/睡眠保留Missing Execution，恢復的實際時刻不得回填成準時Forward。07:00取「前交易日」或「下一民用日」及週末規則須先凍結，不用日曆昨天推定。

不新增service account/權限/喚醒電源設定；新的stop/disable/resume只限未來獨立任務，保留STOP原因及immutable attempts、不重設舊預算。Monitoring沿run/invocation/QC/restore與新attempt ledger，本機摘要即可；不建立Discord webhook。

本次5session原始兩來源共1396903bytes，平均279381bytes/session。按250sessions/year情境，normal raw約69.85MB/year、normal＋一次content約139.69MB/year；若四次physical參考均留完整response，上界情境約279.38MB/year，另加metadata/log/primary+backup冗餘。不是完整年度實測或保留期限政策。

主存仍沿.artifacts治理、新任務另具獨立namespace；真正備份目的地須Human指定不同實體磁碟或授權NAS，不能假設/mnt/d是另一裝置。需size/SHA＋獨立程序restore驗證；既有原始資料全保留，不設自動GC，未來retention另批。

下一Human最少審閱：W35工程整合、用既有one-shot建立有界live-dispatcher接合與驗收、限期新Task/runtime/操作及calendar/retry/update-evidence規則、主存/獨立備份與本機監測停止程序。現有離線程式不能直接宣稱長期Collector Ready。W35不啟用任何長期Collector或Shadow。StrictSTOP/FormalAdmissionNOT_APPROVED/W34OFFLINE_PREPARATION_ONLY/W12SAFETY_HALTED。

R1修復：在file lock內以immutable parent claim SHA檢查其是否已有retry child；success/pending也不允許從舊parent分叉，等值event冪等，線性新failure鏈可再retry。window35-r1-red.txt→15tests GREEN；window35-review-closure.json獨立確認。新window35-replay-closure.json於修後code獨立重建12031rows/217missing、Synthetic修復與停止決策，原replay及NO_GO皆未改寫。
