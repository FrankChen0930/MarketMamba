# Window21 FinMind Historical Data Proof

COMPLETE/STOP/UNMERGED/NOT_DEPLOYED；retrospective sample，不是Dataset／Study Admission。
W20固定6a34520已ff整合main（前87695e4），11deliverablehash／8HTML及manifest／既有ReviewGO重驗；
W21隔離research/v7-window21-finmind-proof，base6a34520，finalHEAD為含本收尾commit。
W19工作樹／排程／Artifact／預算完全不操作；W06/W12/W19不整合，既有production/frozen contracts不變。

1. **來源適用性：有條件保留**。精確匿名FinMind v4/data實際HTTP200，18requests＝14price＋3metadata＋1精確重取，
   未用4名額；0Token／帳戶／retry／redirect／fallback。先1個2010window及獨立restore成功，再Metadata／其餘窗口。
   6檔2330/8069/2499/6515/8272/9929，116筆normalprice／48不同日期，2010-01-04～2025-08-01，
   觀察年份2010/2020/2022/2023/2025。只證明這些窗口，不能宣稱全市場、整段連續或最早1994已驗證。
2. **歷史角色**：2499 provider下市表報2020-11-10，2010取回10筆；下市附近及2022窗口空，
   不推斷最後交易日／停牌原因／確定漏資料。6515有文件指定2020-03-19興櫃語意例外及現metadataTWSE，
   8272有2023-10-17首日open0例外及emerging/TPEx多列metadata；有初步跨狀態線索，確切轉板／上市日期仍UNKNOWN。
   「較短上市歷史」角色未取得正式上市日期，保留UNVERIFIED，不虛構分類。
3. **價格分類**：100 VALID_PRICE僅基本數值／OHLC關係；3 NO_PUBLISHED_PRICE、2 MARKET_SEMANTIC_EXCEPTION、
   11 UNVERIFIED。9929三個OHLC全零日volume分別177(7/25)、0(7/30)、451(7/31)，
   實證有量不代表有效公告價（依FinMind編碼語意，非原交易所公告獨立認證）。不ffill／不補有效0價／不混PriceAdj。
   興櫃例外僅套文件明確指定日期；其他無充分歷史市場背景的open異常保留UNVERIFIED。
   空query為MISSING_DATA但不等於已證實MissingTradingSession；無SOURCE_ERROR或存取拒絕。
4. **Metadata重大限制**：StockInfo4367rows／Delisting727rows／TradingDate6936rows。
   Calendar1999-01-05～2026-12-31，含未來日期，也含2026-10-23，與既有W17官方休市身份相衝。
   不能作已發生交易日／label成熟／交易缺漏的合格authority；這一缺口不回寫或影響W19。
   StockInfo含2330多industry列，已報下市2499仍有2026updated-date列；日期是更新，不是上市／仍在市證明。
   不能只以現在codes／latestrow形成歷史Universe。官方市場日曆／暫停交易／生命週期及代碼再用仍缺。
5. **Raw與還原**：根目錄`.artifacts/window21-finmind-proof`保存真正Raw、Query Snapshot Manifest、record、預先claims及license binding；
   原始FinMind responses不入Git、不對外散布。18response received838854bytes；content-addressed去重後836995rawbytes。
   原始Query／HTTP／receiptUTC／schema／QC／Hash完整；Publication UNKNOWN，今天歷史版本≠當時vintage。
   另一Python程序逐18份verify/copy/hash/schema/records/query驗證PASS；
   `.artifacts/window21-finmind-restore`仍同機副本，BACKUP_PENDING，不能宣稱獨立備份。
   序號1與18的2330同查詢Raw SHA相同，只是短間隔一致，不證明永久不修訂。
6. **下一步建議**：值得提出獨立少特徵、非商業私有、RETROSPECTIVE_ONLY dataset候選；
   不是48-feature E5重現、不是HistoricalAsOf或Executable evidence。首次真實CPUStudy仍需Human分別批准：
   精確dataset／股票類型與歷史Universe、資料保存範圍、補齊合格calendar/lifecycle、zero/missing／公司行動處理、
   Label entry/exit與5d/10d成熟、fold/purge/selection/refit/evaluation政策，以及有限CPUfit計算範圍。
   W07–W11元件可重用，本輪不重寫／訓練／RankIC。資料儲存權利不得因Human批准而超越FinMind方案限制。
7. **何時降級**：必需的下市/市場歷史資料無法取得、正價／時間身份無法解釋、不能合規保存、或固定區間回應截斷／不一致，
   可阻擋相應Study。現有結果不排除Price來源，但Calendar與StockInfo不能直接成為合格研究基礎。
8. **成本估算，不執行**：本樣本normalprice21057bytes／116rows約181.53bytes/row；
   N2000×16年×250sessions＝800萬rows，約1.45GB未壓縮JSON（非全市場實測，metadata/versions/restore/矩陣額外）。
   若每檔全範圍完整則約2000requests（未驗range limit），需年切則約32000；300/hr為文件配額非實測吞吐。
   探索樣本低量，不足以認證長區間、永久保存、跨期完整性或總費用。

逐Query資料範圍／QC／Hash見[Summary](../research/v7/window21-proof-summary.json)，
[獨立Restore](../research/v7/window21-independent-restore.json)，[LicenseBinding](../research/v7/window21-source-license-binding.json)。
本輪沒有DataAdmission，StrictSTOP／E1BLOCKED／H2NOT_APPROVED／HistoricalPROPOSED／W12SAFETY_HALTED不變；W19PROTECTED。
Review／tests／最終Git identity及恢復入口收尾於window21-evidence與唯一Backlog。

審查收尾：一次主要Review Critical0／Required2（原NO-GO保留）。單輪修復：dispatch前驗既有records/依賴，
IncompleteRead/未完成claim持久STOP，repeat僅允許成功完整query；4個regression先RED後GREEN，15focusedPASS。
修後再以獨立程序還原18份PASS；原records/source不重寫，不再API，沒有第二次獨立GO。
CLOSED marker阻止未用4名額；license source已從W20既有文件複製至W21root，不新增網路。
[Review](../research/v7/window21-review.json)、[tests](../research/v7/window21-targeted-tests.txt)、
[實體bytes inventory](../research/v7/window21-local-artifacts.json)保存必要證據。
恢復：`cd /home/frank/projects/MarketMamba/.worktrees/v7-window21-finmind-proof`，
`git status --short --branch`、`git rev-parse HEAD`，讀Current_State／Authority_Map／
[唯一Backlog](../research/v7/research-program.md#window21-finmind-historical-data-proof)與本報告。
ProjectMemory/AuthorityMap/DataNote已更新；W11CPUoverlay Python3.12.3，DataRoot env UNSET，Windowsoperational無payload讀取。
Git main只含W20ff，W21localcommit未整合；Push/Deploy/訓練/RankIC/Windows/排程修改N/A。工程與API均STOP。
