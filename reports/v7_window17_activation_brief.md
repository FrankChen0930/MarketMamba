# Window17 Forward Vintage Activation Brief

Status: WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Package PROPOSED, collector NOT APPROVED.

W16 main ff-only：27dcb36b5431b70e8aa831edb6cb9234cb3a40b3 → 000b0b9c40448b11a41d05396dffc6e0f3ea1235；
W06／W12 未整合。W17 base 同後者，branch `research/v7-window17-forward-readiness`，finalHEAD為含本文件的commit。

1. **資產保護**：W16原CSV 279436bytes＋2manifest＋request ledger原地保留；根目錄
   `/home/frank/projects/MarketMamba/.artifacts/window16-durable-evidence/` 保存不可覆寫持久副本。
   獨立Python程序重驗5files／2snapshotmanifest／完整rawhash PASS；不是只有hash。
   它獨立於W16/W17工作樹，仍是同機同檔案系統，**BACKUP_PENDING**，不是異地／獨立裝置備份。
   [Inventory](../research/v7/window17-asset-protection.json)記錄原路徑、持久路徑與完整SHA-256。
2. **日曆**：TWSE[官方公告](https://www.twse.com.tw/zh/about/news/news/content.html?8a8216d6a0fa351b01a1638507a10203)
   的標題、ID及2026-10-22 20:10發布時刻確認10/23休市；PDF正文返回404HTML，未認證颱風原因正文。
   [官方年度日曆](https://www.twse.com.tw/zh/trading/holiday.html)及其頁面JSON確認10/26補假。
   10/22 W16實際觀測，23臨時休市、24/25週末、26補假，不計漏抓；27只列預定恢復。
   TPEx年度PDF403／直接臨時公告未取得，不能借用TWSE當合格TPExcalendar。
   [六日面板](../research/v7/window17-calendar-panel.json)僅bounded diagnostic，不是市場日曆准入；
   修訂鏈／asof／unknown／時間區分以Synthetic測試驗證，不回填成10/22本機已知證據。
3. **來源與頻率**：[11549](https://data.gov.tw/dataset/11549)、[11371](https://data.gov.tw/dataset/11371)
   均列每1日更新、OGDL1.0；這不等於numericquota、許可任意自動化、固定發布時刻或SLA。
   本輪TWSE條款timeout、TPEx條款403，未繞過；正式定期服務條件仍 **BLOCKED**。
   保持W15兩個exactCSV URLs，無市場payload、retry探測或其他provider請求。
4. **Capture／QC**：新增離線guard只處理日曆狀態與storage/backup分離；沿用W07 clocks、W14/W15
   Synthetic capture及W09 integrity，沒有網路collector或第二套平台。W16 missing分類保留：
   TWSE15／TPEx27 `MISSING_OHLCV_NO_FILL`；不補零、不以全資源證券作可交易股票Universe。
   32必要定點驗收（整合3＋最終29），其中新9測試；6面板PASS。Raw收存不是QC／PIT／research通過。
   Disk-write/error-log失敗不返回成功；實際獨立備份寫入與失敗驗收尚未完成，須選定位置後再驗。
5. **Fallback**：沿用[W16](../research/v7/window16-fallback-policy.json) PROPOSED。Same-endpoint retry
   和替代endpoint／獨立provider分開；後兩者名單皆空。無合格路徑就保留failure／missing紀錄，
   不產生合格Snapshot。休市不算missing；後補不冒充準時forward；不得用receipt代publication。
6. **下一次最少Human決策**：先取得exact resource定期條件／更新窗口及各市場有效日曆，
   然後批准[Activation Package](../research/v7/window17-activation-package.json)的runtime、每session
   時刻／頻率／有限retry timeout、QC及缺漏政策、儲存與獨立備份、監測通知、missed-day backfill、
   停止恢復撤銷。建議先有界人工監督pilot；正式adapter實作／真實capture仍須新授權。
   279436bytes/pair×250sessions約70MB raw只是單次樣本量級，不含revision／manifest／備份，非容量保證。
7. **Historical Track**：PROPOSED／historicalcoverage BLOCKED。下次僅查W14兩historicalroutes的官方
   授權和coverage文件，或一項Human指定的合法歷史產品；不抓payload、不全面PIT搜尋、不訓練／RankIC。
8. **驗收／恢復**：[Evidence](../research/v7/window17-evidence.json)、[tests](../research/v7/window17-targeted-tests.txt)、
   [唯一Backlog](../research/v7/research-program.md#window17-forward-vintage-operational-readiness)。
   新session：`cd /home/frank/projects/MarketMamba/.worktrees/v7-window17-forward-readiness`，
   `git status --short --branch`、`git rev-parse HEAD`，讀Current_State、Authority_Map、本brief與Backlog。
   無授權下一輪工程；不mergeW17、不push/deploy/正式收集/排程/GPU/訓練。

獨立Review一次GO，Critical0／Required0／Optional0；獨立新9testsPASS5.920s，0review修復。
ProjectMemory及AuthorityMap已更新；原129protectedfiles不變。Runtime existingW11CPUoverlay，
Data root env UNSET，Windowsoperational realpath僅辨識不讀payload；Push/Deploy N/A。
