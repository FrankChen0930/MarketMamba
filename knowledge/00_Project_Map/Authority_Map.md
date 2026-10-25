# MarketMamba Authority Map

> Status: `CANONICAL_CURRENT_PROJECT_MAP`
> Owner: `Human + project agents`
> Semantic change owner must update this map in the same change.
> Prior research/Phase A map accepted `2026-09-20`; 2026-09-22 runtime additions are agent-observed, pending separate human review. Canonical means current navigation authority, not immutable final state.

## Window27 — Label Safety Policy / Mask frozen; diagnostic pending

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已本機 ff-only 整合；38 pins／3 focused tests／獨立 Closure GO 核對。
W27 branch `research/v7-window27-label-safety`，base 同固定 SHA；UNMERGED / NOT_DEPLOYED。
[Development-only Policy](../../research/v7/window27-policy.json) 與 [Mask Manifest](../../research/v7/window27-mask-manifest.json)
已在新 metric calculation 前固定，限 Human Window27 frozen-prediction sensitivity，不是正式 Label／Data Admission。
Feature、Entry/Exit、holding path 分別檢查，不補值；confirmed unit-event 優先排除，重疊 reasons 用 disjoint primary 計數；UNKNOWN 不變 CLEAN。
Unit1 Mask complete；Unit2 paired diagnostic / independent replay / review PENDING。0 API／fit／new prediction／GPU／V6／W19 修改。
Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED；W19 不操作。

## Window26 Label Integrity & Next Study — COMPLETE / STOP

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327` 本機 ff-only 整合；
4 focused tests、12 private artifacts／21 deliverable pins 與 Review GO 已核對。W26 base 等於該固定 SHA。
W26 branch `research/v7-window26-label-contract`，worktree `.worktrees/v7-window26-label-contract`；UNMERGED / NOT_DEPLOYED。
2321 已確認 68% 虧損減資；2023-02-17 市場參考價單位變更與 2022-12-08 會計減資基準日分離。
三筆既有 10d Label 算式／市場索引吻合，但跨股數單位變更，不可作未調整持有期報酬。
六個缺行情日期是市場交易日，非市場休市；原始官方停牌公告尚未核實，精確停牌區間保留 UNKNOWN。
其他公司行動覆蓋／historical availability／vintage 未認證；不重寫 W24／W25 歷史結果，不重新評分。
兩路 Label Policy、新 Study、L0 正反向對照與所有日期／閾值／資源額度均為 PROPOSED，待 Human 決定。
曝光帳本核對 E5／E1 machine contracts；2022／2023 已曝光，未列期間為 EXPOSURE_UNKNOWN，不自動成為獨立測試。
FinMind 非價格 1/6、行情 0；官方文件 loads/attempts 4/4；全部預算 CLOSED。0 fit/refit/new prediction/RankIC/GPU。
W26 private root `.artifacts/window26-label-integrity/`；15 files／4,785,564 bytes 獨立同機 Restore PASS；
W24 191＋W25 12 files 原 Hash 保持。BACKUP_PENDING，不宣稱正式備份或完整重現。
[Report](../../reports/v7_window26_label_integrity.md)、[Evidence](../../research/v7/window26-evidence.json)、
[事件矩陣](../../research/v7/window26-event-matrix.json)、[Study Candidate](../../research/v7/window26-next-study-candidate.json)、
[Exposure Ledger](../../research/v7/window26-exposure-ledger.json)、[唯一 Backlog](../../research/v7/research-program.md#window26-label-integrity-and-next-study-contract)。
Review／tests 狀態以 window26-evidence.json／window26-review.json 為準；本窗口無剩餘已授權工程，STOP。
下一 Human：Label 路線及未知／停牌／除息／下市政策、資料與 Universe 資格、Development-only 或 prospective、
Fold／Selection／Refit／評估／資源上限；本次未批准其執行。Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED，W19 不操作。

## Window25 Ranking Robustness & Leakage Audit — complete / STOP

W24fixed `90b1327560f717727ad3fd53dfe86cb421c70f5b` 已LOCAL_INTEGRATED；main由87af76b ff-only，
26tests／191sourcefiles+33deliverablepins／GO與rawrestore驗證。W25 `.worktrees/v7-window25-robustness-audit`，
branch `research/v7-window25-robustness-audit`，base90b1327；LOCAL_INTEGRATED_BY_W26_FIXED27055f7/NOT_DEPLOYED。
單一read-onlyanalysis，0GET/fit/refit/newpredictions/GPU；既有W24policy/predictions/masks/data不改。
44266sourceformulas/timing檢查0確認跨期數值錯誤；TRAINhash/mean/std吻合，historicalavailability/maturity/vintage仍UNKNOWN。
四組RidgeMean/Median正，positive約59–62%；LOSO33×4=132cases/31020paired-days不翻負，非排除群組或TRAIN影響。
固定10/20circularblocks敏感度：2022relativecomparison脆弱/部分bounds跨0；2023direction一致但EXPOSED。
公司行動未知；2321約+224%10dproxy跨6缺行情session，來源/停牌/公司行動原因UNKNOWN，無發明過濾門檻或改寫結果。
分類ENGINEERING_VALID_WITH_EXPLORATORY_SIGNAL僅條件rawproxy；2022comparisonSIGNAL_FRAGILE，actions/asofINSUFFICIENT_EVIDENCE；非Humancorrectness/admissiongate。
Root.artifacts/window25-robustness-audit保存LOSO/casetrace/source；12files獨立同機restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window25_robustness_audit.md)、[matrix](../../research/v7/window25-leakage-matrix.json)、
[evidence](../../research/v7/window25-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window25-ranking-signal-robustness-and-leakage-audit)。
10focused/26mainpost/4independent新testsPASS；ReviewGO0CriticalRequired，0rework；16CBBbounds/264LOSOdailycase獨立核對。
下一Human：公司行動/停牌/adjustment與2321有界補證、universe/PIT或proxy政策、matchedbaseline方向與新未曝光資料/正式統計設計。
本輪無剩餘READY；STOP。StrictSTOP/FormalAdmissionNOT_APPROVED/E1BLOCKED/H2NOT_APPROVED/W12HALTED，W19完全不操作。

## Window24 Expanded Retrospective Pilot — complete / STOP

W23fixed `87af76b038fcbbda9e1e4b3a4349e693a3b1ba48` 已LOCAL_INTEGRATED；main由7027f03 ff-only，
26tests／55priorlocalfiles+25deliverables／GO與rawrestore重驗。W24 `.worktrees/v7-window24-expanded-pilot`，
branch `research/v7-window24-expanded-pilot`，base87af76b；LOCAL_INTEGRATED_BY_W25_FIXED90b1327/NOT_DEPLOYED。
下載前固定36檔HINDSIGHT_SELECTED cohort（currentcapital非historicalmarketcap）；31新FinMindGET+5舊raw，
0metadata/retry/redirect/token/fallback，unused14CLOSED。24511rows，QC24160VALID/346zero/5UNKNOWN；不fill或調整價混用。
TWSE2021–2023既有729calendar；52488三分mask，engineering46291／本輪exploratoryrawproxy44266／strict0。
33current/new股票入比較；3下市startUNKNOWN僅保留診斷，不能宣稱已消除存活者偏誤。
固定L0／Ridge各horizon一次（2realRidgefits），Train2021/Val2022/exposed2023；0search/selection/refit。
940core日期×horizon全部N27–33，預先固定min20未降；1880daily/940paired與scores獨立ReplayPASS。
公司行動/publication/vintage/可交易資格仍UNKNOWN，所有Label為獨立rawdiagnosticproxy，非收益或模型優劣證據。
本輪LIMITED_EXPLORATORY授權已行使；FormalResearchAdmission仍NOT_APPROVED／StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED。
Root.artifacts/window24-expanded-pilot/private原始與完整predictions，193files+5舊raw獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window24_expanded_pilot.md)、[evidence](../../research/v7/window24-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window24-expanded-retrospective-ranking-pilot)。
50focused/26mainpost/44independenttestsPASS；ReviewGO0CriticalRequired，0rework；191pins/restore驗證。
下一Human：正式raw/adjusted/actions政策與historicaluniverse/calendar/Label/eval/compute批准；本輪無剩餘READY，STOP。
W19工作樹/task/artifacts/budget/runtime不操作不干擾，不新增runtime觀測；不mergeW06/W12/W19或W24，不push/deploy。

## Window23 Historical Calendar & CPU Smoke — complete / STOP

W22 fixed `7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470` 已LOCAL_INTEGRATED，main由428a5fa ff-only；
23tests／78localfiles+23deliverablehashes／既有GO重驗。W23 `.worktrees/v7-window23-calendar-smoke`，
branch `research/v7-window23-calendar-smoke`，base7027f03；LOCAL_INTEGRATED_BY_W24_FIXED87af76b/NOT_DEPLOYED。
官方documentloads10已CLOSED（8directGET+2webopen），FinMind0。TWSE2019/20tableempty，2021/22schedule244/246，
2023schedule加官方0803臨時休市239，與observedreference相同；不提升strictPIT／完整TPEx日曆資格。
十檔日期membership僅listinginterval/knownboundary；8272不套用2024TPEx身份，停牌/可交易資格UNKNOWN保留。
W22原19728strictmask不改，Stricteligible0；A195價格幾何失敗／B7009獨立工程診斷／C12524其他阻塞。
Human本輪LIMITED_ENGINEERING_SMOKE_ONLY：五檔TWSE，Train範圍masked到2021、Validation2022、Forward-style2023；
固定L0／兩horizons各一次Ridge，0search/selection/refit；1880dailyrows，1864defined、16minN不足跳過，非模型性能結論。
公式/feature/fold/policy於fit前封存；rawreturn非可執行收益，actions/publication/vintage仍UNKNOWN，不是研究admission。
唯一rework修復JSONmap排序造成replaylist順序不同；原bundle/models/metrics及run-code留存，無重新fit，獨立ReplayPASS。
Root.artifacts/window23-calendar-smoke及同機window23-restore保存資料，56files獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window23_calendar_smoke.md)、[evidence](../../research/v7/window23-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window23-historical-calendar-and-cpu-smoke)。
正式study仍PROPOSED/DataAdmissionNOT_APPROVED；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED；W19完全不操作。
58focused/23mainpost/26independenttestsPASS；獨立GO0CriticalRequired，唯一replay修復無refit。
下一Human集中決定正式historicalcalendar/universe/action/label/availability/evaluation/compute；無本輪剩餘工程READY，STOP。

## Window22 Continuous Historical Panel — complete / STOP

W21兩Required獨立複核CLOSED，15testsPASS/Critical0Required0；133localfiles驗Hash。
main由6a34520 ff-only至 `428a5fa5cb7b771db39a053f88c7046501399f7e`；沒有提前main文件commit。
W22 `.worktrees/v7-window22-historical-panel` / `research/v7-window22-historical-panel`，base428a5fa；LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED。
10FinMindpriceGET、0metadata/diagnostic，均HTTP200，unused20CLOSED；10檔9864rows/1762931rawbytes，2019–2023。
6檔同1216reference日期（不是合格officialcalendar），9725structuralVALID/131UNVERIFIED/8zero，不fill或算return/RankIC。
官方18419/25036/11543CSV各1GET；TPExdelisted403立即停，共4resourceattempts；cataloguedocs另3GET。
6515TWSE上市20210120、8272TPEx上櫃20241023、2499終止上市20201110有officialCSV證據；逐日suspension/membership未知。
實際5d/10d eligible0：calendar/actions/availability/labelmaturity仍UNKNOWN，historicalstudyPROPOSED，不作admission。
Root.artifacts/window22-historical-panel與同機restore保存raw/panel/mask；15產物獨立程序restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window22_historical_panel.md)、[coverage](../../research/v7/window22-panel-summary.json)、
[唯一Backlog](../../research/v7/research-program.md#window22-continuous-historical-panel)。
23focusedtestsPASS；一次主要獨立ReviewGO/Critical0Required0，0rework；15份restore皆PASS。
下一步只建議精準calendar/lifecycle補證與Human統一data/label/eval/CPU授權；本輪完成後STOP。
W19工作樹/排程/Artifacts/budget不讀寫不觸發不整合；W12SAFETY_HALTED/StrictSTOP/E1BLOCKED/H2NOT_APPROVED不變。

## Window21 FinMind Historical Proof — complete / STOP

W20fixed `6a34520d8f9c1166172af3503cad531e23f5ecec` 已LOCAL_INTEGRATED，main由87695e4 ff-only；
11deliverable／8HTML+manifest／既有ReviewGO驗證。W21 research/v7-window21-finmind-proof，base同上，
`.worktrees/v7-window21-finmind-proof`，COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED；固定428a5fa。
18anonymousHTTP200＝14price+3metadata+1repeat；4名額未用且CLOSED，無Token/retry/redirect/fallback。
6檔116price／48dates／2010-01-04..2025-08-01；100structuralVALID、3NO_PUBLISHED_PRICE（含有量零價）、
2documented-date市場例外、11UNVERIFIED；零ffill／PriceAdj混用／研究准入。2499有2010price及provider下市身份，近/後窗口空不推論漏資料。
Snapshot收到838854bytes，去重後836995rawbytes。root.artifacts/window21-finmind-proof獨立於worktree，
18份初次／修後獨立程序Restore與reviewerRestorePASS；原records/code保留。Backup仍同機BACKUP_PENDING。
VendorCalendar含future及已知20261023休市日期，禁止當actualcalendar；StockInfoupdated不等於active/listing，Universe/lifecycle仍BLOCKED。
來源僅CONDITIONAL_RETROSPECTIVE_CANDIDATE，多年份小窗口≠全市場連續性／StrictPIT／Executable證據。
W21原一次Review 2Required及單輪修復證據保留；W22已針對性獨立複核CLOSED/GO，見Window22integration記錄。
[Proof](../../reports/v7_window21_finmind_proof.md)、[evidence](../../research/v7/window21-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window21-finmind-historical-data-proof)。
下一Human：獨立少特徵retrospective dataset身份、合格calendar/lifecycle、missing/actions/Label/maturity與eval/CPUcompute批准；
本輪無剩餘READY。W19工作樹/排程/artifact/budget完全不操作、不整合；無新runtime觀測。
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/W12SAFETY_HALTED。

## Window20 Historical Source Decision Sprint — complete / STOP

Branch `research/v7-window20-historical-sources`，base/main `87695e4d5dc6e047f5ab39ca3cd8294b971bbc3a`；
`.worktrees/v7-window20-historical-sources`，UNMERGED/NOT_DEPLOYED。僅最多三類來源的兩單元文件調查，沒有行情API或工程實作。
FinMind TaiwanStockPrice為有條件主要候選；富邦HistoricalCandles為帳號/使用權BLOCKED的備用；
W14官方歷史route使用資格未釐清，未套用W15CSV授權。多年份日期皆DOCUMENTED宣稱，不是假定Coverage。
非商業/顯名/禁止raw再散布與匿名300/hr文件已核對；永久保存條件UNKNOWN，0價/興櫃open/調整價/下市生命週期須驗證。
建議下一次無Token資格Pilot至多22requests（18price+3metadata+1重取）、6檔角色×3小窗口，PROPOSED未批准。
不是5d/10d Label／Dataset／研究admission，沒有L0/Ridge/LightGBM訓練或RankIC。
W19固定cf1a84f的限期Task授權獨立保留；本輪不整合/改動/停用/觸發，無新增runtime觀測；以其固定來源及timestamped記錄為準。
[Decision brief](../../reports/v7_window20_source_decision.md)、[matrix](../../research/v7/window20-source-matrix.json)、
[evidence](../../research/v7/window20-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window20-historical-data-source-decision-sprint)。
一次獨立審查GO／Critical0／Required0，8官方HTML924253bytes／manifest及分類預算驗證PASS；無實作或套件重跑。
下一Human只需決定是否批准有限無帳號非商業來源資格驗證及保存範圍；富邦另需帳號/行情權限，付費並非首輪必要。
完成後STOP，沒有本輪剩餘READY；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/GPU_SAFETY_HALTED。

## Window18 手動 Forward Collector Pilot — complete / STOP

W17 fixed `96e8c0c4f2f7f78c508048e31f0f163d6c707128` 已 LOCAL_INTEGRATED/NOT_DEPLOYED，
main 由000b0b9 ff-only至固定候選；38treesclean／19deliverablehashes／9CPUtests確認；W06/W12未整合。
W18 `development/v7-window18-manual-collector`，base同上，`.worktrees/v7-window18-manual-collector`，
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED；finalHEAD為含本收尾commit。僅本輪manualpilot批准，daily仍NOT_APPROVED。
TWSE/TPEx exactlicensedCSV各1GET HTTP200，0retry/redirect/fallback；均2026-10-22：1383/1017rows，
137956/141480bytes，raw合計279436。原record/QC/code保留；TPEx初判27INVALID，修後離線27MISSING，TWSE15MISSING；不補零。
本機根目錄`.artifacts/window18-manual-pilot`保存raw+manifest+provenance，獨立程序初次／修後restore均Hash+rowsPASS。
同機restore不是獨立backup：BACKUP_PENDING。無Content-Length證據明確UNKNOWN，不宣稱來源宣告長度驗證。
Capture MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED，publicationUNKNOWN，非VERIFIED_FORWARD_VINTAGE／ResearchDataAdmission。
W17calendar僅PROVISIONAL TWSEboundedmatch；TPEx日曆未知，不以牆上日期判stale，不將商品全列入股票Universe。
新增manualCLI/CSVQC，共重用v1snapshot/W09safe+lock/W07clock/W17calendar；W15adapter仍Synthetic-only，沒有第二套平台。
25targetedtestsPASS（含原W17 9）、integration9PASS；一次review Critical0/Required3→唯一修復輪全部以RED/GREEN＋
獨立程序實際restore自行驗收。沒有第二次獨立GO，不把首次NO-GO改寫。W14/16/17原bytes與W12HALTED不變。
[Brief](../../reports/v7_window18_manual_pilot.md)、[evidence](../../research/v7/window18-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window18-limited-real-forward-collector-pilot)。
下次Human：daily runtime/scheduler、19:00候選與更新/日曆/QC、有限retry、獨立備份、監測通知與backfill/停止撤銷。
本輪GET額度已用完；無剩餘授權執行。StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/GPU_SAFETY_HALTED，
FairEval/PreNextOpen PROPOSED／InferenceRecovery DEFERRED。不得自動重跑/排程/訓練或開下一窗口。

## Window17 Forward Vintage 操作準備 — complete / STOP

W16 固定 `000b0b9c40448b11a41d05396dffc6e0f3ea1235` 已 LOCAL_INTEGRATED／NOT_DEPLOYED；
main 由 `27dcb36` ff-only 至固定候選，原有 W16 UNMERGED 記錄僅保留歷史意義。W06／W12 未整合。
W17 `research/v7-window17-forward-readiness`，base 同上，`.worktrees/v7-window17-forward-readiness`；
WINDOW_COMPLETE／STOP／UNMERGED／NOT_DEPLOYED，最終 HEAD 為含本收尾的 local commit。
W16 CSV 279436 bytes＋2 manifests＋request ledger 已獨立程序驗證並保留至根目錄
`.artifacts/window16-durable-evidence/`，原工作副本及 Restore 不動；同機同檔案系統，BACKUP_PENDING。
獨立Review一次GO／Critical0／Required0／0rework。32 次必要定點驗收（整合3＋最終29）、6日離線面板；新9測試僅工程／Synthetic，不是歷史 PIT 准入。
TWSE 官方10/23休市標題／ID／發布時刻及10/26年度補假確認；PDF正文未取得。
TPEx直接公告仍缺，不能以TWSE推定；10/27預定交易、非已實際交易。日曆修訂須有鏈與asof身份。
兩資源每1日更新不等於自動存取配額／SLA；定期服務條件與正式啟用仍 BLOCKED。
新元件限離線calendar／storage狀態guard，重用W07/14/15及Snapshot；沒有網路Collector或Calendar Platform。
[Activation brief](../../reports/v7_window17_activation_brief.md)、[evidence](../../research/v7/window17-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window17-forward-vintage-operational-readiness)。
下一Human：先確認兩Resource定期服務條件及各市場有效日曆，再批准runtime／時間頻率／有限重試／
QC缺漏／儲存與獨立備份／監測通知／backfill及停止恢復撤銷。沒有剩餘本輪READY或啟用權限。
StrictSTOP／E1BLOCKED／H2NOT_APPROVED／HistoricalPROPOSED／CollectorNOT_APPROVED／GPU_SAFETY_HALTED；
FairEval／PreNextOpen PROPOSED、InferenceRecovery DEFERRED。不啟動下一窗口。

## Window16 Real Resource Proof — complete / STOP

W15fixed27dcb36b5431b70e8aa831edb6cb9234cb3a40b3 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from4da1bcf;25targetedCPUtestsPASS,source/review/hash/36cleantrees verified.
W14 raw/W15 evidencebytes remain originaltrees;not moved to main. Mainheld27dcb36;
W06/W12/othercandidates notmerged. OlderW15UNMERGED text historical for integration only.
W16 research/v7-window16-resource-proof,base27dcb36,.worktrees/v7-window16-resource-proof,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2units only,no newvalidator/platform. Two exactlicensedCSV GETs,
1each/0retry/redirect/fallback. BothHTTP200 UTF8,only2026-10-22/1session:
TWSE1383rows/1383securities/137956bytes,1368basicQCpass+15missing;
TPEx1017/1017/141480bytes,990basicQCpass+27missing. Nozero-fill orcalendar/universe/
pricecurrency/adjustment/tradabilityadmission. HistoricalTrainingCoverageBLOCKED;
no hiddenhistorysearch. Forwardrawresource CANDIDATE_ONLY, activationNOT_APPROVED.
IndependentPythonrestore2records/279436bytes/hash+rowsPASS;3focusedSnapshot testsPASS.
Raw+restore both localuntrackedcopies in W16tree,not certifiedbackup;tree removalrisk.
[Brief](../../reports/v7_window16_resource_proof.md),[evidence](../../research/v7/window16-evidence.json),
[sole queue](../../research/v7/research-program.md#window16-real-resource-coverage-and-capture-proof).
SingleindependentreviewGO0Critical/Required,0repair;28targetedtests, no fullsuite repeat.
Fallbackpolicy PROPOSED,whitelistempty,no provider switch/collector/fit/RankIC/GPU.
Old120files+4Frozenhashes unchanged. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/
ExploratoryPROPOSED/CollectorNOT_APPROVED/GPU_SAFETY_HALTED. No nextwindow.

## Window15 Licensed Foundations / Dual-track — complete / STOP

W14fixed4da1bcfa81c8ff786de82d29ec461c0851d411d5 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only fromc78098f;pre/post150CPUtestsPASS,independent0Critical/Required.
W14original337221rawbytes/4snapshotfiles remain in W14tree, NOT moved to main.
Mainheld4da1bcf; W06/W12/othercandidates notmerged. OlderW14UNMERGED prose historical.

W15 research/v7-window15-dual-track,base4da1bcf,.worktrees/v7-window15-dual-track,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Humanconfirmed dualtrackdirection only.
U1 COMPLETE_REVIEWED:govdatasets11549/11371 exactCSV scopedOGDL1.0+attribution confirmed;
periodicservice/rate/activation conditional; oldhistoricalroutes permissionUNCLEAR.
3officialmetadata/licenseHTML1547650bytes locallyretained/independentverified;0market
payloadrequests. Licensing never implieshistoricalPIT/dataset/researchadmission.
U2 COMPLETE_REVIEWED:independentexploratoryStudy/Dataset/Feature drafts PROPOSED;
7synthetictests. Lag/dates/parameters/universe/calendar/labels/bytes/compute unapproved,
DATA_READY=false; oldsyntheticadapters require separatelyreviewed realdata scope.
U3 COMPLETE_REVIEWED:SYNTHETIC-onlythinarchive adapter reusesSnapshot/W14/W09;
18tests,2boundedrepairs,0remainingCritical/Required. Idempotentraw/newrevision,
currentQC/late vsfirstcapture separated; missingintent/binding/snapshotNOT_RESTORABLE,
no hiddenretryrepair. Initialdryrun+oldsource retained, finalofflinevalidation separate.
175jointCPUtestsPASS; allthree reviews0Critical/Required, U1/U2rework0. No livecollector/
fits/realRankIC/GPU. W11/W12/W14old92files+4Frozenhashes unchanged; no cleanup.
[Brief](../../reports/v7_window15_dual_track_brief.md),
[evidence](../../research/v7/window15-evidence.json),
[sole queue](../../research/v7/research-program.md#window15-licensed-dual-track-foundations).
NextHuman:boundedlicenseddata/coverage/restore qualification, newstudy/eval/admission
policy, separateTrackBoperation/storage/frequency/QC/notice authorization. No READY
execution remains in this completedgrant. FinalHEAD containingclosurecommit.
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/InferenceRecoveryDEFERRED remain. No nextwindow activation.

## Window14 Evidence-first Data Pilot — complete / STOP

W13fixedc78098f3fdddb74e461bc54d700bde5baac0fcc1 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from8db5b911;pre/post140CPUtestsPASS,independentGO0Critical/Required;
12artifact/24source/88protectedhashes verified,34trees preflightclean. Mainheldc78098f;
W06/W12/othercandidates notmerged. OlderW13 UNMERGED text historical for integration only.
W14 research/v7-window14-data-admission,basec78098f,.worktrees/v7-window14-data-admission,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2boundedunits,each1reviewrepair.
U1COMPLETE_REVIEWED:2dailyofficialrequests TWSE20220901/TPEx20220902,337221bytes,
4selectedstocks,currentstructuralQC. Both proposedRETROSPECTIVE_ONLY;0asof/0proxy
qualified. Historicalpublication/revision/calendar/universe/labels and rights remain
unresolved. Contentclass != usagepermission != researchadmission;no moredatarequests.
U2COMPLETE_REVIEWED:PROPOSED forwardcontract+readonlyprototype reusesSnapshot/W07/W09;
10synthetictests,150jointCPUtestsPASS;both0Critical/Required. No collector/scheduler/
newplatform. PROTOTYPE_TIMELY neverauthenticates callerclock/pins oradmitshistoricaldata.
[Brief](../../reports/v7_window14_decision_brief.md),
[evidence](../../research/v7/window14-evidence.json),
[sole queue](../../research/v7/research-program.md#window14-evidence-first-admission-pilot).
NextHuman:permitdata route/rights, qualify asof/vintage or distinctlag-proxy/forward
proposal, then calendar/universe/labels/restore+research/eval/CPUcompute. No trueCPUstudy
ready or remainingauthorizedexecution. Raw4snapshotfiles retained locally;hash-only
restore impossible. ProtectedW11 83bytesfiles/W12 5diagnostics and4Frozenhashes unchanged.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED. FinalHEAD containingclosurecommit;no newwindow.

## Window13 Thermal / Admission Preparation — complete / STOP

Branch research/v7-window13-admission-preparation,base/main8db5b9111b98d4965a9f56f10a51c5157284b900,
.worktrees/v7-window13-admission-preparation,WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
Main unchanged; W11 localintegration verified (olderUNMERGED text historical).
W06 fixed96c60564/ProtocolAudit8601c737/Recoverye181f366/W12 fixed16e444f9 only
UNMERGED_REFERENCE_ONLY; no merge or authority promotion.

U1 COMPLETE_REVIEWED,sourcef26dbf5: HUMAN_CONFIRMED_DESIGN_DIRECTION/GPU_RETEST_DEFERRED;
RTX3060 local-first candidate,WindowsHost target87C reference/futurestop87C;
83C warningPROPOSED, not NVIDIA longtraining certification. OldW12 80C/HALTED/
32.106942s unchanged. GPUkernel/F-B/resume/AMP/fullmodelshape NOT_VERIFIED.
U2 COMPLETE_REVIEWED: [Human Decision Brief](../../reports/v7_window13_decision_brief.md),
DT1–4/EV1–6 options PROPOSED only. W05 audit reused0VERIFIED/21CONDITIONAL/27UNKNOWN;
2TPEx oddlot archives176096bytes currentlyverified, no historicalavailability/QC/
calendar/universe/labels/full48panel proven. No realRankIC or dataadmission.
ENGINEERING_READY scopedW07–11 tools; POLICY_REQUIRED/DATA_BLOCKED/RUNTIME_BLOCKED
separate. 140mainCPUtests+13fixedW06prototypePASS; bothreviewCritical0/Required0,
0rework. ProtectedW11 83files+W12 5diagnostics unchanged; no newsource/framework.
[Decision](../02_History/Decisions/20261024-window13-thermal-admission-preparation.json),
[evidence](../../research/v7/window13-evidence.json),
[sole queue](../../research/v7/research-program.md#window13-thermal-and-admission-preparation).
NextHuman: freeze timing/missing/source/evidenceclass, identify qualified bytes or
approve futurevintage collection, then evaluation/compute contract. No READY execution
in this completedgrant; new boundedauthorization required. Do notclearHALTED/retestGPU.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED unchanged. FinalHEAD containingU2closurecommit.

## Window11 L2 & CPU Resume — complete / STOP

Window10 fixed `f25181637e40138c84dc3722292647d4325882fc` LOCAL_INTEGRATED/NOT_DEPLOYED
ff-only from `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9`;pre/post167testsPASS,
40hashes/independent26 GO,0Critical/Required. Main heldf251816; no main closure commit.
Older Window10 UNMERGED/STOP entries historical, superseded for integration only.

Window11 `development/v7-window11-l2-cpu-resume`,basef251816,
`.worktrees/v7-window11-l2-cpu-resume`, WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
U1 actualLightGBM4.6.0 fit/refit/forward/common-support/independentReplay PASS;
5officialwheels pinned in dedicated overlay, baseCPU Torch/NumPy/SciPy unchanged.
U2 reuses E1 actual train loop/store/adapter, safe weights_only=True probe codec.
Three new-process exits0/75/0,step4(epoch1batch1) afterwarmup ->12steps/4epochs;
23nontelemetryfields exactlyequal forLATEST/PREVIOUS/BEST. CPU_STEP_BOUNDARY_RESUMABLE
and CPU_MID_EPOCH_RESUMABLE only deterministic daily cursor/0workers/FP32/1CPUthread.
No separateepochinterrupt/GPU/AMP/shuffleworker/generalExactResume/full reproduction.
211jointtestsPASS. U1 environment setupcorrection1,reviewrework0;U2rework2,0remaining
Critical/Required. Source U1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b.
Final HEAD is containing artifact/closure commit; no further engineering/Window12.

CPUbytes83files/~1.79MB retained `.artifacts/window11-cpu-resume-proof` under this
worktree; exactinventory in CPUevidence. Hash-only Git metadata cannot restore lost
checkpoints. Preserve this tree/basevenv/overlay/cache until separately reviewed.
Current runtime-bound L2 source snapshot uses U2commit and fresh run identity;
initialsourcebundle/report retained unchanged, requires original source to replay.
[CPU proof](../../research/v7/window11-cpu-resume-evidence.json),
[current L2](../../research/v7/window11-l2-runtime-evidence.json),
[evidence](../../research/v7/window11-evidence.json),
[sole Backlog](../../research/v7/research-program.md#window11-l2-and-cpu-resume).
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. No realresearch/GPU/production/admission/othermerge.
Resume namedtree HEAD/clean/maps/Charter/evidence; new bounded authorization required.

## Window10 Classical Synthetic Foundation — complete / STOP

Window09 fixed `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `bdea30d877fa2c5d02308f343fa5591a35616aec`.
Pre/post141CPUtests PASS;39hashes/independent40tests GO,0Critical/Required;30trees
clean. Main held91fa82c, all new closure/development documents stay isolated.
Older Window09 UNMERGED/STOP entries superseded for integration only.

Window10 branch `development/v7-window10-classical-e2e`, base91fa82c, worktree
`.worktrees/v7-window10-classical-e2e`. UNMERGED/NOT_DEPLOYED; two units only,
<=2rework/unit. Human now permits tiny CPU synthetic fits, not real-data research.
U1 COMPLETE_REVIEWED:13tests/independent13PASS,1rework (full selection run hash and
null containers). L0 explicit toy factor; L1 NumPy train-only scaled Ridge; L2
adapter code present, dependency BLOCKED (lightgbm absent; no install performed).
U2 COMPLETE_REVIEWED:13tests/independent53 including40oldW09testsPASS,1rework
for fit/prediction/config lineage; validation+toyselection and forward pair replay
verified in independent CPU process. Final167joint testsPASS. U1 source 4ec22a4330ed6db7df0b3dafb4106cea820516e2;
U2+closure final HEAD is this containing commit. WINDOW_COMPLETE/STOP.
W09 v1 no-fit profile unchanged; v2 honest tiny-classical-fit profile added. No new
split/evaluator/platform. METADATA/EVALUATION replay proven; TRAINING_RESUMABLE/
FULL_REPRODUCTION_READY unproven. L2 actual fit remains dependency BLOCKED.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No Frozen E5/E1 or production mutation; no other branch
integration, GPU/Colab/search/real research. [Evidence](../../research/v7/window10-evidence.json),
[sole backlog](../../research/v7/research-program.md#window10-classical-synthetic-foundation).
Resume named branch HEAD/clean/maps/Charter/evidence/queue; final candidate is this
containing U2/closure commit. No READY unit remains in this grant; no Window11.

## Window09 Durable Artifacts & Offline Replay — complete / STOP

Window08 fixed `bdea30d877fa2c5d02308f343fa5591a35616aec` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `1631020bce8223ac4337c7e97473a2a3d9a555f0`.
Pre/post101CPUtests PASS;19hashes/24cases reverified; independent34tests GO,
Critical0/Required0. Main remains bdea30d; no extra main closure commit.
Older Window08 UNMERGED/STOP entries are historical, superseded for this integration only.

Window09 isolated `development/v7-window09-artifact-replay`, base bdea30d, path
`.worktrees/v7-window09-artifact-replay`. UNMERGED / NOT_DEPLOYED. Human authorizes
only two CPU/SYNTHETIC units, <=2evidence-based rework/unit, then STOP.
U1 COMPLETE_REVIEWED:20focusedPASS;2bounded rework rounds corrected traversal,
metadata symlinks/JSON overflow, publication-null/sources containers. Immutable bundle profiles, required
artifacts, staging+diagnostics, fsync/flock/rename, idempotence and verified external
bytes references implemented. U2 COMPLETE_REVIEWED:20testsPASS,1rework for nested
containers and actual maturity recheck; independent final40testsPASS,0Critical/Required.
Final141joint tests;2schemas/3outputs validated. METADATA_VERIFIED and
EVALUATION_REPLAYABLE proven; TRAINING_RESUMABLE/FULL_REPRODUCTION_READY not proven.
WINDOW_COMPLETE/STOP. U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final U1 container repair
and U2 source/evidence are in this containing closure commit.
No tracking platform, retention/GC policy or training/research admission added.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No W06/recovery/audit integration or original matrix search.
[Evidence](../../research/v7/window09-evidence.json),
[backlog](../../research/v7/research-program.md#window09-durable-artifacts-and-offline-replay).
Resume named worktree cleanHEAD/maps/Charter/evidence/sole queue; final HEAD is
containing U2/closure commit. No remaining work in this grant; no Window10 grant.

## Window08 Common Ranking Evaluation — complete / STOP

Window07 fixed `1631020bce8223ac4337c7e97473a2a3d9a555f0` LOCAL_INTEGRATED /
NOT_DEPLOYED by ff-only from `465dca229fc1452aaf1b46572b06886a7d8a2c8f`.
Pre/post67tests PASS;10artifact/frozen hashes and16cases verified; independent24tests,
Critical0/Required0 GO. Main remains exactly1631020; no main closure-doc commit.
Window07 STOP/UNMERGED prose below is historical, superseded only for this integration.

Window08 branch `development/v7-window08-ranking-evaluation`, base1631020, worktree
`.worktrees/v7-window08-ranking-evaluation`. UNMERGED / NOT_DEPLOYED. Human grant
permits only common evaluator and paired comparison, synthetic CPU fixtures, max2
rework/unit. U1 COMPLETE_REVIEWED:826b8ddf798e752584d180181d24cc1db967e019;19tests/independent19PASS,1rework.
U2 COMPLETE_REVIEWED in containing closure commit;15tests,0rework. Both Critical0/
Required0;101joint CPUtests,24explicitcases,2schemas/3outputsPASS. WINDOW_COMPLETE/STOP.
Existing E5 average-tie RankIC, canonical_hash and W07 calendar/fold/role guards reused.
Policy remains named SYNTHETIC_NOT_ADMITTED; minN/weighting explicit, no statistics
or acceptance policy frozen. Metadata-first label gating, raw references and daily
rejection ledger are engineering evidence, not real historical PIT certification.

No W06/recovery/audit branch integrated. Strict Phase0 STOP; E1 Frozen Matrix BLOCKED;
H2 NOT_APPROVED; Fair Evaluation and Pre-next-open PROPOSED; Inference Recovery DEFERRED.
[Evidence](../../research/v7/window08-evidence.json),
[unique backlog](../../research/v7/research-program.md#window08-common-ranking-evaluation).
Final HEAD is this containing U2/closure commit. Resume by HEAD/clean, this entry,
Authority Map, Charter, evidence and backlog; a new bounded grant is required.
No automatic Window09, real-data evaluation, training or main merge of Window08.

## Window07 temporal/isolation foundation — completed / STOP

Human approved two CPU-only synthetic engineering units, no formal study config.
Branch `development/v7-window07-temporal-isolation`, worktree `.worktrees/v7-window07-temporal-isolation`,
base/main `465dca229fc1452aaf1b46572b06886a7d8a2c8f`. UNMERGED / NOT_DEPLOYED.
U1 commit5d736e9:13focused/independent13PASS,1owner-binding repair.
U2 complete:11focused/independent11PASS,
1source-hash repair. Final67related tests/16expectedcasesPASS;0remainingCritical/Required.
WINDOW_COMPLETE / STOP. No further work under this grant. Containing Unit2/closure
commit is final checkpoint. Resume via cleanHEAD here/maps/Charter/uniquequeue;
new bounded authorization required. CommonEvaluator synthetic foundation can be
next scope, not approved policy/data/Window08. [Evidence](../../research/v7/window07-evidence.json),
[decision](../02_History/Decisions/20261023-window07-temporal-isolation.json),
[unique backlog](../../research/v7/research-program.md#window07-temporal-fold--research-isolation).
No fixed validation/forward months/origins/30purge or metrics adopted; unknown policy/
data failclosed. No realPIT certification, no formalFeatureLabel/E5E1/admission change.
Window06/audit/recovery branches only pinned references, not merged or promoted.
StrictSTOP/E1BLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. This grant does not authorize a Window08.

## Window06 — Window05 integrated; decision-time preparation authorized

Fixed `65462ec185783eedb2ba81b66cfa04783f86c6e2` integrated ff-only from main
`2ee4ca83e709af64be4dd63032ba4fb47903d769`; pre/post87 tests PASS,13source/output hashes
verified, independent13tests/0Critical/Required. LOCAL_INTEGRATED / NOT_DEPLOYED.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window06-integration.json).

Same Human grant authorizes isolated `research/v7-window06-decision-time` at
`.worktrees/v7-window06-decision-time` from this closure: VERSIONED PROPOSAL,
<=3source families/~10sessions diagnostic panel and synthetic eligibility tests.
New pre-next-open proposal does not replace frozen E5/E1. No formal cutoff/time,
feature/label/admission or execution policy selected. StrictSTOP/E1BLOCKED/H2unapproved/
FairEvaluationPROPOSED remain. No F2 repeat search/training/production changes.
Read local Current State, Authority Map, Charter and unique Backlog on that branch
for subsequent progress; all pending documents stay isolated. Older STOP entries
are historical for their windows; no indefinite authorization.

## Window05 Feature Integrity and Data Research — complete / STOP

Human authorized F0/F1/F2 only: read-only source audit, synthetic candidate formulas,
and bounded monthly-revenue PIT feasibility. Branch `research/v7-window05-feature-integrity`,
worktree `.worktrees/v7-window05-feature-integrity`, base/main `2ee4ca83e709af64be4dd63032ba4fb47903d769`.
UNMERGED / NOT_DEPLOYED. Source-derived macro Date joins do not prove t-close
availability; missing source values may become defaults. F0 identified material
unquantified risks; no real factor/performance analysis may depend on these
unverified payloads. Historical conclusions are preserved; no result reclassification.

F0 source audit complete:48ordered, VERIFIED0/CONDITIONAL21/UNKNOWN27/INVALID0,
all empirical quality UNKNOWN; F0 commit9fbf1bd,4tests/independent4 PASS.
F1 completed PROPOSED shortlist/formulas,9tests/independent9 PASS,1timezone rework;
source commit1a6a452, real diagnostics BLOCKED. F2 complete NOT_CURRENTLY_FEASIBLE
for broad historical PIT recovery;4historical metadata blobs independently verified,
0newqualified rows,0rework. All independent reviews0Critical/Required;87related tests
PASS. Window complete: STOP / UNMERGED / NOT_DEPLOYED, no automatic continuation.

The containing F2/closure commit on this branch is the final resume checkpoint.
Verify HEAD/clean in the named worktree, then read this section, Authority Map,
`knowledge/GOVERNANCE.md` and the local unique Backlog's Window05 section.
[Backlog](../../research/v7/research-program.md#window05-feature-integrity-and-cost-effective-data-research),
[evidence](../../research/v7/window05-evidence.json). Main stays2ee4ca8; pending
integration documents stay here. No formal manifest/cutoff/label/admission changed.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; Fair Evaluation and all
new feature/timing/forward-vintage candidates PROPOSED. Human timing/data identity/
policy choices precede real-data diagnostics; a new bounded grant is required.
Earlier window execution pointers below are historical; this entry owns latest state.

## Window04 W04B-LOG integration closure — STOP

Human-approved fixed `97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9` integrated by local ff-only
from `31f7accc475b5b735684d5a06725a237d85d5693`. Engineering source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`;
no new merge commit. This containing main commit adds closure documentation only.
LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED / WINDOW_COMPLETE / STOP.
Pre/post274 tests each PASS; independent runtime6 PASS,0 Critical/Required.
Both source hashes match;23 existing worktrees clean before integration and preserved.
[Closure evidence](../../research/v7/window04-log-integration-closure.json).

Current continuation entry is WSL main: verify HEAD/clean, read this section,
Authority Map, `knowledge/GOVERNANCE.md` and the local unique
`research/v7/research-program.md`. Older branch-local/unmerged pointers below are
historical, superseded only for W04B-LOG local integration. No new work authorized.
No additional high-value READY item was established by the prior bounded discovery;
future concrete CPU defects can be considered under a new bounded grant.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; research options PROPOSED.

## Window04 Stage B — completed / STOP

Branch `development/v7-window04-cpu-continuation`, base/main `31f7accc475b5b735684d5a06725a237d85d5693`,
worktree `.worktrees/v7-window04-cpu-continuation`. UNMERGED / NOT_DEPLOYED.
WINDOW_COMPLETE / STOP: W04B-LOG source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`; final274 tests PASS.
No additional high-value independent READY unit established. A new bounded grant
is required for further development; this grant is exhausted. Resume by verifying
HEAD/clean here, reading this section, Charter and the unique backlog; do not
continue automatically. The containing docs-only closure commit is final checkpoint.
W04B-LOG fixes admitted E1 runtime reader-error propagation: log write/decode
failure reaches caller with original cause/path and stops a still-running child;
a failure observed after child exit also cannot return success. CPU fixtures only,
no Colab/Drive certification or training readiness claim. Focused6 tests PASS,
independent6 PASS,0 Critical/Required,0 review rework rounds (one initial fix).
[Unique backlog](../../research/v7/research-program.md#window04-stage-b--cpu-continuation)
records evaluated alternatives; [evidence](../../research/v7/window04-cpu-continuation-evidence.json).
Strict Phase0 STOP, E1 matrix BLOCKED, H2 NOT_APPROVED and proposals PROPOSED.
This section owns branch-local progress; earlier integration/continuation pointers
below are historical. Stage B source/closure stays here, main remains unchanged.

## Window04 integrated; bounded CPU continuation authorized

Human-approved fixed `98f270dd63f05831b2a42695d8005638fda2bbf5` was ff-only integrated
from main `c733a53f81ff7bd572739b5d75f8baabf1189976`. LOCAL_DEVELOPMENT_INTEGRATED /
NOT_DEPLOYED. Pre/post272 tests each PASS; independent metadata5 PASS,0 Critical/Required.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window04-main-integration.json).

The same Human grant authorizes Stage B, at most four valuable CPU engineering
units with at most two evidenced rework rounds each. Use isolated
`.worktrees/v7-window04-cpu-continuation`, branch `development/v7-window04-cpu-continuation`,
created from this closure; its `research/v7/research-program.md` owns subsequent
selection and completion state. All pending development/closure stays on that branch.
No Stage B merge/push/deploy/training. Strict Phase0 STOP, E1 matrix BLOCKED,
H2 NOT_APPROVED; comparison proposals remain PROPOSED. Earlier awaiting-approval /
no-StageB text below is historical and superseded solely by this explicit grant.

## Window04 Git reconciliation — current candidate

Human authorized isolated version reconciliation only. Branch
`integration/v7-window04-reconciliation`, worktree `.worktrees/v7-window04-reconciliation`,
base/main `c733a53f81ff7bd572739b5d75f8baabf1189976`; second parent fixed
`489a127cf951f93f7bfc873e9333f05113f5b8e9`. Both histories are retained without rewrite.
Original STOP audit `3330427a1d774d30e65216ddab89d668d97c187c` remains separate evidence,
not a third merge parent. This section supersedes older continuation pointers below;
they describe prior checkpoints, not the current working location.

Status: GO_CANDIDATE_ONLY / AWAIT_HUMAN / UNMERGED_IN_MAIN / NOT_DEPLOYED.
Main is unchanged. No Stage B work or research planning is authorized here.
Merge commit `225efa9bb13b67bd5258faba79a47666c8906148` retains exact two parents above.
272 joint tests PASS; independent metadata5 PASS,0 Critical/Required; no unresolved
conflicts. [Integration evidence](../../research/v7/window04-reconciliation-evidence.json). The containing docs-only
closure commit is the final candidate SHA; main and original candidate are ancestors.
Resume by verifying branch/HEAD/clean here, reading this section and the evidence,
then await Human approval of that fixed SHA. Do not start Stage B.
Strict Phase 0 STOP; E1 Frozen Matrix BLOCKED; H2 NOT APPROVED; comparison options
remain PROPOSED. Read the local unique `research/v7/research-program.md` first
for reconciliation checkpoint; original preparation artifacts are unchanged.

## Fair evaluation preparation complete — fixed continuation pointer

Window02/03 source is integrated in local main at approvedeba6b649; this main
commit adds navigation only. New Window04 work remains UNMERGED/NOT_DEPLOYED on
`development/v7-fair-evaluation-decision@489a127cf951f93f7bfc873e9333f05113f5b8e9`, base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`.
Worktree `/home/frank/projects/MarketMamba/.worktrees/v7-fair-evaluation-decision`.
WINDOW_COMPLETE / STOP. Readiness tool2a9de32 and decision package489a127 completed;
64 related non-training tests PASS,15 source hashes/cost arithmetic verified,
independent reviews0 Critical/Required. New comparison options remain PROPOSED;
no dates/thresholds/training/H2 approved. Original matrix BLOCKED, Strict Phase0 STOP.

Latest unique backlog is that pinned commit's `research/v7/research-program.md`
Window04 section; report `reports/v7_fair_evaluation_decision_package.md`, evidence
`research/v7/fair-evaluation-preparation-evidence.json`. New audit is not merged
into main. To resume, cd named worktree, verify HEAD/clean, read Current State,
Authority Map and that queue. No full historical rediscovery or repeated matrix
search. New bounded execution needs Human grant; no background continuation.
This pointer supersedes earlier Window04 authorized/READY text below.

## Fair evaluation preparation — completed checkpoint

Isolated `development/v7-fair-evaluation-decision` from integrated main closuref31fcbb.
UNMERGED/NOT_DEPLOYED for new work only. Existing-contract read-only E1 metadata
audit completed,5tests/independent5 PASS,0 Required; always execution_ready=false.
Decision package completed:15 source hashes, exposure ledger, fit/cost formulas and
Colab stop conditions independently verified;0 Critical/Required.64 related tests
PASS. WINDOW_COMPLETE/STOP, all new options PROPOSED. New execution requires
Human bounded grant. Resume at this branch via main pinned closure pointer.
[Unique backlog](../../research/v7/research-program.md#fair-evaluation-preparation-window-04),
[evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window02/03 integrated; fair evaluation preparation authorized

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved fixedeba6b649;
main ff-only9956446→eba6b649, pre/post267 tests each PASS, independent0 Critical/
Required.19 worktrees clean before merge;23 source/2 parser/1 report/7 result hashes
verified, protected contracts/data/models/lock unchanged. Fixed candidates retained.
[Integration evidence](../../research/v7/window0203-integration-20261021.json),
[Human scope](../02_History/Decisions/20261021-engineering-integration-fair-preparation.json).
This supersedes UNMERGED/no-main-merge text below only for admitted Window02/03.
No deployment, policy or research classification change. StrictSTOP/E1BLOCKED/H2
unapproved persist. A separate bounded CPU preparation window is authorized;
its current queue is [the unique program](../../research/v7/research-program.md).

## Data reliability / evaluation preparation window03 — 2026-10-21

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Current Human grant superseded the
Window02 STOP solely for this separate window. Main remains9956446; candidate
`development/v7-data-evaluation-preparation` joins main and fixed Window02f34f9f3
in isolation. Five navigation conflicts retain Window02 content, with this new
checkpoint taking precedence; main's previous pointer remains in parent history.
[Decision](../02_History/Decisions/20261021-data-evaluation-window03.json),
[unique backlog](../../research/v7/research-program.md#data-evaluation-window-03),
[evidence](../../research/v7/data-evaluation-window03-evidence.json).
Stage1 isolated integration GO at9bb4247 (259 tests, independent0 Required).
Parser complete: TWSE response date binding and TPEx unique header mapping,
8 tests + independent8 PASS; MOPS source/policy unchanged. No provider certification.
Fairness/readiness completed:12 source identities,7 result hashes,2seed identity
chains; independent0 Required. New designs PROPOSED; NOT_RUN_READY without matrix.
Runtime correction: candidate Data/contains3 inherited tracked historical files,
unchanged and parquet contents unread; no processed_v6 mount. Do not reuse older
Data-absent assumption. Final267 tests PASS, health WARNING/no FAIL. Main stays9956446; next session starts
at `.worktrees/v7-data-evaluation-preparation`, reads this queue/evidence and waits
for a new bounded grant. No main merge authorized. Strict Phase0 STOP/E1 matrix BLOCKED unchanged.

## Cross-domain window02 final checkpoint — STOP

Isolated `development/v7-cross-domain-window-02` from main closurec1179fe,
UNMERGED/NOT_DEPLOYED. W02-DATA/PREP/EVAL/MODEL completed; this bounded Human grant is exhausted;
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
owns acceptance and state. No production parser edits or training permitted.
W02-DATA completed7-case offline diagnostic with independent review0 Required;
TWSE/TPEx source fixes and MOPS policy are now explicit Human decision items.
W02-PREP completed frozen contract/matrix metadata guards;34 tests and independent14
PASS. Metadata-only, no training readiness/admission; original matrix remains BLOCKED.
W02-EVAL rejects missing seeds/nonfinite gate values;6 tests + independent6 PASS.
Acceptance thresholds unchanged; historical evidence not recomputed.
W02-MODEL completed source-level selection/evaluation reuse audit;11 pinned sources
independently verified. New research choices remain PROPOSED/Human-only.
Final259 tests PASS, four independent reviews0 Critical/Required. No additional
high-value READY item demonstrated; see queue for P1/P2 decisions and recovery.
[Evidence](../../research/v7/cross-domain-window02-evidence.json).

## Charter integrated and window 02 authorized — 2026-10-21

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved exact
`f7af348710467ac381b2a84e8d192670ced89932`; WSL main ff-only from
`6d74d7384caa5fa90b1b050ce507bfb8282f19df` to that SHA. This containing
closure commit is docs/evidence only. [Integration evidence](../../research/v7/charter-integration-20261021.json)
records pre/post254 tests each (PIT49, non-training E1 31, portfolio67, ops81,
fetch3+governance23), 6 source/4 authority hashes,75 protected blobs unchanged,
17 clean worktrees and independent0 Critical/Required. No tiny training or GPU probe run.
Window01's UNMERGED wording below is historical and superseded only for integration.

[New Human decision](../02_History/Decisions/20261021-charter-integration-window02.json)
authorizes one next isolated CPU window, maximum3–5 meaningful units. Current
selection/checkpoint remains the [unique program](../../research/v7/research-program.md).
Discover Data, Model Research and Experiment Preparation before optional system
work; do not change V6 production. StrictSTOP/E1BLOCKED/Human research gates persist.
The fixed candidate branch remains unchanged. No push/deploy/Windows/MAS activity.

## Autonomous charter first window — 2026-10-21

**WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW / UNMERGED / NOT_DEPLOYED.**
Human's [Charter](../GOVERNANCE.md#autonomous-development-charter--2026-10-21)
is persisted in existing governance, and the
[single backlog/checkpoint](../../research/v7/research-program.md#autonomous-charter-window-01)
records the complete first window. Branch `development/v7-autonomous-charter`
starts at unchanged main `6d74d7384caa5fa90b1b050ce507bfb8282f19df`.
Five units: Charter `be7998d`, E1 split completeness `3f07f80`, research decision
preparation `5847afc`, checkpoint fallback `2c60415`, snapshot pre-publication
validation in the containing closure commit. No further work starts in this window.

[Evidence](../../research/v7/autonomous-window-01-evidence.json): E1 33 + PIT49 +
governance23 =105 PASS; independent reviews0 Critical/Required after checkpoint's
second/final repair; snapshot one round; four research sources verified. Source
and output hashes/JSON/syntax/diff checks pass; knowledge health WARNING, no FAIL.
CPU synthetic engineering and PROPOSED research preparation only; no historical
checkpoint payload/matrix/data reads, external network, new training, GPU or MAS.
Existing E1 unittest includes a one-step synthetic CPU smoke. Loader remains
trusted-local and unrestricted: no historical-pickle safety certification.

This supersedes older no-next-program or in-progress statements solely for this
completed window. H0 remains locally integrated; Strict Phase0 STOP, E1 Frozen
Matrix BLOCKED, H2 and production/research gates unchanged. Another session reads
Charter + this backlog + fixed branch evidence, verifies Git/runtime, and waits
for a new bounded window grant before executing. No merge/push/deploy authorized.
No confirmed unexecuted READY item remains in this window; do not infer global
absence of defects or repeat E1 searches without new evidence.

## Portfolio History H0 completed — 2026-10-21

**LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED.** Human accepted the limited
fixture Product Slice and [approved exact4a66928](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
WSL main fast-forwarded from `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` to
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; subsequent containing commit is
**documentation/evidence only**, not a different product candidate. Pre/post CPU,
API/HTTP, frontend and real Chrome font/selection/error checks passed.
[Integration report](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](../../research/v7/portfolio-h0-integration-closure.json).
CPU01/02, PORT01–04, UX01–04, final UX and font correction are in local main.
This supersedes historical UNMERGED/H0-pending/UX-review-pending prose below for
these units only. Original reports, fixed branches and all16 worktrees are retained;
audit7ecb920 remains separate historical failed-preflight evidence, not merged.
No Required/Critical remains within this tested slice. Main is a valid source
baseline for future separately authorized CPU work; no new Program starts now.
Strict Phase0 STOP, E1 Frozen Matrix BLOCKED, H2 unapproved, research correctness
unapproved. No push/deploy/Windows operational change/training. STOP after closure.
The existing UI's static UNMERGED badge is historical candidate wording, not a live
Git integration indicator; canonical status above owns integration authority.

## Final H0 browser blocker resolved — 2026-10-21

Human [authorized only the late-font visibility repair](../02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
New branch `fix/v7-timeline-font-visibility` starts from exact `6c894794` product
candidate; `audit/v7-h0-preflight@7ecb920` is evidence only, NOT an ancestor.
True-font RED reproduced desktop3.5px/mobile2.5px clipping. Existing ResizeObserver
now observes all event rows as well as the fixed-height container; GREEN confirms
font-driven layout correction without page movement, focus loss or data change.
[Report](../../reports/v7_portfolio_font_visibility.md),
[evidence index](../../research/v7/portfolio-font-visibility-evidence.json).
Original Required finding RESOLVED within tested fixture scope; bounded preflight
engineering GO, pending Human approval of the new fixed SHA. Final candidate is the
containing commit, not the audit commit. Main remains `83a6a2e`; no merge occurred.
Human limited UX product acceptance remains in force; earlier pending-UX-review
language below is historical. Old H0 authorization does not extend to this new SHA.
UNMERGED / NOT_DEPLOYED; H2 unapproved, Strict Phase0 STOP, E1 Frozen Matrix BLOCKED.
STOP here: no further UX/research work authorized. Native Goal creation was rejected
because the previous H0 goal remains paused; no automatic continuation claim.

## CPU window 02 implementation boundary

`development/v7-cpu-window-01` continues from `7eb9cd9`, UNMERGED / NOT_DEPLOYED.
[Window 02 checkpoint](../../tasks/v7-cpu-development-window-02.md) owns current
engineering planning only. Window complete: normalized collision refusal at
`735ae6f`, duplicate JSON member refusal in this closure commit; both are
reviewed synthetic engineering, not Human research approval. Existing portfolio spec/engine duplicate-refusal and
journal integrity contracts remain authoritative; no research/production promotion,
no replacement of historical window 01 evidence or E1 frozen artifacts.

## Strategic planning ownership — 2026-10-21

The dated current section in [research-program.md](../../research/v7/research-program.md#current-development-program--2026-10-21)
is the **single current development backlog**. This is planning/navigation authority
under the present Human request, not approval of proposed execution/research.
The original umbrella research contract remains historical context in that file;
S5–S9 frozen machine contracts/results retain research authority. Old tasks/plan,
todo and CPU-window queues remain dated evidence, not parallel active roadmaps.

Candidate `integration/v7-cpu-roadmap` from `4c388de` is independently reviewed
GO for local development consideration only, UNMERGED / NOT_DEPLOYED. Main remains
`83a6a2e`; no main merge or source cherry-pick occurred. [Evidence report](../../reports/v7_strategic_roadmap_recovery.md).
PORT-01–04 have [explicit H1 approval](../02_History/Decisions/20261021-portfolio-history-h1.json)
for fixture-only CPU read-only engineering; completion evidence is
[portfolio-history-evidence](../../research/v7/portfolio-history-evidence.json).
PORT-01–04 are COMPLETED_VERIFIED within fixture engineering, with
[acceptance report](../../reports/v7_portfolio_history_acceptance.md).
Human Product Slice Review is pending; engineering completion is not result approval.
New implementation is branch-local UNMERGED / NOT_DEPLOYED. H0 main integration and H2 research
admission remain unapproved and separate. MAS Core Live is not certified or activated.

## UX product direction — 2026-10-21

The subsequent [Human final-closure decision](../02_History/Decisions/20261021-portfolio-ux-closure-decision.json)
accepts the overall dual-mode direction and authorizes only selection visibility
and event readability. Both finishing tasks are engineering-verified; final Human
closure review remains pending. Mobile advanced requires basic access only; AI
should prefer the structured API/JSON. This decision grants no H0/H2 or new research authority.
The following original UX-01–04 record remains historical scope evidence.

[Human UX decision](../02_History/Decisions/20261021-portfolio-ux-human-decision.json)
authorizes only UX-01–04 presentation engineering on `5a2bc6f` ancestry.
The fixture UX slice is engineering-verified; [UX acceptance](../../reports/v7_portfolio_ux_acceptance.md)
is ready for Human review, not Human acceptance or H0/H2 approval.
Pinned project-local frontend skills are subordinate guidance, never research,
production or project authority. No H0/H2 admission or accounting changes.

## Source-of-truth Hierarchy

1. Versioned machine contracts, manifests, hashes, and executable tests.
2. Completed run-local final artifacts and validated checkpoint manifests.
3. Implementation on an explicitly named branch and commit.
4. Timestamped runtime observations.
5. Human reports and guides.
6. Historical notes and legacy artifacts.

An authority class describes evidence strength, not deployment status. Branch-local code and untracked candidates are never promoted to `CANONICAL` merely because they exist.

## Authority by environment and evidence class

- **Operational authority:** implementation plus a timestamped external action/observation. `AGENTS.md` and a runnable script alone do not prove a live schedule. A stale observation becomes `STALE_OBSERVATION`, never a fresh fact by documentation update.
- **Research state:** Current State summarizes admitted branch-qualified results. A main file can lag an accepted worktree result; the frozen experiment contract controls the question, completed run manifest/hashes control the result, and neither promotes production.
- **Artifacts:** a result filename or copied ZIP is insufficient without source commit, contract, input/matrix identity, environment, split/support, model/checkpoint identity, outputs and evidence class. Mutable Parquet caches need separate provenance snapshots. Untracked local files are candidates, not canonical authority.
- **Runtime:** WSL development, Windows Scheduler checkout, Colab Drive and deployed API/UI are distinct. Record `observed_at` for Windows/service facts; use realpath and source commit for cross-environment identity. No inferred equivalence.
- **Branch/worktree:** a versioned `commit:path` remains valid historical/research authority without being merged or deployed. A branch label aids navigation but can move. Worktree checkout name is not evidence of its current branch.

## Freshness Rules

- `RUNTIME_OBSERVATION`: requires `observed_at`; after 7 days it becomes `STALE_OBSERVATION` until rechecked. It is not deleted automatically.
- Repository contracts: no time expiry. They change only through commit identity or explicit supersession.
- Completed-run artifacts: remain evidence for that run; they do not automatically describe a newer run or current production.
- `UNKNOWN`: has no primary authority. Secondary evidence may explain what is known without filling the gap.

## Topic Map

| Topic | Primary Authority | Authority Class | Status | Supersedes | Secondary Evidence | Conflicts / Notes |
|---|---|---|---|---|---|---|
| Worktree responsibility | [Worktree Register](Worktree_Register.md) | `CURRENT_REFERENCE` | `FOUR_ACTIVE_K_BRANCH_HISTORY_RETAINED_2026_09_22` | Ad-hoc worktree cleanup assumptions | Git worktree list, branch HEAD and status | Registration does not authorize removal; ownership/creation dates may be UNKNOWN. |
| Development runtime contract | [Current State](Current_State.md) and [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | `WSL_NATIVE_DEVELOPMENT` | Old assumption that `/mnt/d` checkout is canonical development | `docs/operations/wsl-migration-2026-09-07.md` | Operational Windows/WSL bridge is separate; no runtime migration this change. |
| Mutable data cache | [`V6/marketmamba/config.py`](../../V6/marketmamba/config.py) and source manifests | `CURRENT_REFERENCE` | `MUTABLE_NON_PIT_CACHE` | Treating cache as frozen research evidence | `MARKETMAMBA_DATA_ROOT`, symlink realpath, V7 source registry | Raw caches can change; authority requires snapshot/identity. |
| Colab runtime | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CURRENT_REFERENCE` | `SEPARATE_LOCKED_TRAINING_RUNTIME` | Local Python environment inferred as formal Colab | E1 bundle and exact requirements lock | Formal result needs environment/bundle/run identity. |
| Local untracked candidates | None | `UNKNOWN` | `NOT_ADMITTED` | File-existence/current-version inference | Main untracked research files | No branch commit/contract or deployment authority from untracked source. |
| Early V7 local historical source | [`2026-09-22 preservation register`](../../research/v7/historical-local-source-register-20260922.md) plus same-commit source files | `HISTORICAL_ONLY` | `PRESERVED_NOT_CURRENT_E5_OR_E1` | Untracked early V7 source of uncertain ownership | Dated deliveries and research notes linked in register | Registration preserves identity and source; it does not validate results or promote an old architecture. |
| Dated V7 research/task notes | [`2026-09-22 notes register`](../../research/v7/historical-notes-register-20260922.md) plus same-commit notes | `HISTORICAL_ONLY` | `PRESERVED_SUPERSEDED_AS_CURRENT_STATE` | `research/v7/current-state.md` and task plan/todo as active instructions | Original note hashes and branch-local plan/todo blob IDs | Old paths retained for historical Phase 0 references; canonical Current State and frozen contracts control present decisions. |
| Current project state | [Current_State.md](Current_State.md) | `CANONICAL` | `CANONICAL_CURRENT_PROJECT_MAP` | README/OVERVIEW current-state prose and old handoffs | [K0 report](../../reports/marketmamba_k0_project_knowledge_discovery.md) | Accepted 2026-09-20; update or supersede when semantics change. |
| Domain semantics/navigation | [Domain notes](../01_Domains/) | `CURRENT_REFERENCE` | `CANONICAL_DOMAIN_NOTE` / `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN` | Scattered README/guide summaries | [Knowledge index](../README.md), machine contracts linked per note | Notes summarize boundaries and point to authority; they never override machine contracts. |
| Knowledge governance | [GOVERNANCE.md](../GOVERNANCE.md) | `CANONICAL` | `ACTIVE_GOVERNANCE_V1` | K0 governance proposals and manual-only K1 maintenance | `knowledge/knowledge-impact-map.json`, governance tools/tests | Governance detects and records; it never rewrites authority or business semantics automatically. |
| Runtime observation store | [`runtime-observations/`](../runtime-observations/) | `CURRENT_REFERENCE` | `TIMESTAMPED_EVIDENCE_WITH_TTL` | Runtime facts copied only into prose | Runtime status tool and K0 observation evidence | Recording requires an actual observation; stale evidence remains stale. |
| Production runtime | [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) in Windows `main@8b82a68` plus [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `CURRENT_REFERENCE` | `FETCH_ONLY_ACTIVE`; full path `DISABLED` | README/OVERVIEW implication of full scheduled daily inference | D: batch and [operational path](../runtime-observations/operational-path-20260922T132521+0800.json) | Scheduler selects Windows checkout, not WSL development HEAD. |
| Scheduler state | [2026-09-22 Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `RUNTIME_OBSERVATION` | Active fetch-only / disabled full V6.2; last run 2026-09-21 result 0 | 2026-09-20 observation and older snapshots | [K0 runtime map](../../research/project-knowledge/k0/runtime-map.json), local logs | Becomes `STALE_OBSERVATION` after 2026-09-29 13:21:35 +08:00. |
| Operational checkout and WSL environment | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | Windows `main@8b82a68` → distro `Ubuntu` → `mamba_env` Python 3.11.15 | Assumed WSL-native development checkout is scheduled | Batch content, `wsl.exe --list --verbose`, Git | Development `Ubuntu-24.04` checkout is distinct `main@3b0df0f` at observation. |
| Operational data root and freshness | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | D: physical `Data/processed_v6`; ten daily sources max 2026-09-21 | Mtime-only freshness or guessed path equivalence | Parquet metadata, `v62_daily.log` | Date check 10/10 does not certify source validity; universe delta warning remains. |
| Data provenance policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/source-reliability.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Mixed-source OHLCV admission | Provenance lineage, source index and mismatch inventory in the same directory | Source classification must precede proxy classification. |
| Reusable PIT/provenance code | [`source adoption register`](../../research/v7/source-adoption-register-20260922.json) and its main destinations | `CURRENT_REFERENCE` | `IN_MAIN_RESEARCH_ONLY` | Branch-only implementation for the six adopted primitives | 33 focused CPU tests; source blob IDs in register | Frozen corrected baseline result/readiness and superseded 32/8 architecture draft remain branch-qualified. |
| PIT policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Current-world/backfilled data assumptions | Executable builders and Phase 0 tests | Missing lifecycle/publication/provenance evidence fails closed. |
| Feature contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` | `CANONICAL` | `48_FEATURES` | Legacy V6.1 56-feature, V6.2 59-feature and baseline 66-feature contracts | Corrected baseline builder/tests | Counts belong to separate contracts; they are not interchangeable. |
| Feature list | Same versioned feature manifest and its order hash | `CANONICAL` | `ORDER_PINNED` | Legacy feature orders and removed fundamentals | [K0 feature map](../../research/project-knowledge/k0/feature-map.json) | No current-industry backfill, graph features, or industry neutralization. |
| Label contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/label-manifest.json` | `CANONICAL` | `NEXT_OPEN_NO_ROLL` | Legacy close-to-close semantics | Policy plus historical label materializer tests | Entry is next frozen session open; exit is h-th holding-session close. |
| Universe contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/universe-manifest.json` | `CANONICAL` | `VERIFIED_LIFECYCLE` | Current constituent backfill | Historical security-master builder/tests | Membership and tradability remain separate concepts. |
| Historical simulation proxy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json` | `CANONICAL` | `P0_FROZEN`; simulation readiness `PASS` | Ad-hoc execution heuristics | Proxy validation and provenance-remediation readiness | Must never be called `VERIFIED_EXECUTABLE`. |
| Strict executable evidence | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` | `CANONICAL` | `STOP`; verified labels 5d=0, 10d=0 | Any claim that proxy PASS implies trading readiness | Corrected Phase 0 audit | Strict STOP and simulation PASS are intentionally simultaneous. |
| Corrected E5 architecture | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json` | `CANONICAL` | `d_model=64`, `d_state=32`, 48 features, no graph | `architecture-config.json` draft values 32/8 | Completed E5 result identity and final runner | Completed result authority outranks the earlier draft. |
| Training recipe | Same incumbent contract; E1 freezes it in `experiment-contract.json` | `CANONICAL` | `FP32_ADAMW_20_EPOCH_MAX` | Earlier notebook-only assumptions | Exact environment lock and final runner source | E1 changes split, not recipe. |
| Reusable corrected E5 and E1 source | [Source adoption register](../../research/v7/source-adoption-register-20260922.json) and main research modules | `CURRENT_REFERENCE` | `IN_MAIN_RESEARCH_ONLY`; E5/E1 core, frozen contract copies, checkpoint telemetry fix, Colab log wrapper | Branch-only source for adopted slices | 33 E5, 22 E1 and 27 recovery focused CPU tests on main | Result manifests, bundles, diagnostic output, operational runtime and deployment remain branch/external authority. |
| Corrected E5 checkpoints | Validated run-local checkpoint manifests for each seed | `RESEARCH_EVIDENCE` | Seeds 17/29 complete; seed43 `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Filename/pointer-only inference | Diagnostic resume forensics and telemetry | Seed43 pointer metadata named missing generations and recorded `best_epoch=8`; the surviving orphan is epoch 15/step 39,915 and is convergence evidence only, not promotable or directly resumable. |
| Diagnostic v1 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v1/summary.json` | `RESEARCH_EVIDENCE` | `SUPERSEDED_FOR_CURRENT_CONCLUSION` | Raw partial-run impressions | v1 component analyses | Extended diagnostic v2 is the current conclusion. |
| Extended diagnostic v2 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Diagnostic v1 as final conclusion | v2 head, drift, regime and equal-window analyses | Research evidence only; not production/trading authority. |
| E1 experiment specification | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CANONICAL` | `FROZEN_EXPERIMENT_SPECIFICATION` | Informal experiment proposals | Contract diff, runner, checkpoint store, generated notebook | The pre-run decision field remains historical specification, not completed-result authority. |
| E1 completed result | `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` | `RESEARCH_EVIDENCE` | `PASS_REFRESH_HYPOTHESIS` | External-only/unverified result state | Final evaluation, run summary, seed manifests and hashes in the same directory | Historical-simulation proxy only; strict Phase 0 `STOP`; no portfolio admission. |
| E1 post-run diagnostic | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | Undiagnosed post-run state | Month and leave-one-month-out analyses, checkpoint inventory, logged validation history | July-sensitive 10d absolute level; no final-support epoch replay or causal regime series. |
| Portfolio semantics | [`V6/experimental/v7_portfolio_contract.py`](../../V6/experimental/v7_portfolio_contract.py) plus engine/journal tests | `CURRENT_REFERENCE` | `RESEARCH_ONLY` | V6.2 JSONL as V7 semantic authority | `v7_portfolio_engine.py`, `v7_portfolio_journal.py` | No live broker or scheduler integration was discovered. |
| Production-system architecture | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and roadmap | `CURRENT_REFERENCE` | Audit complete; roadmap Phase A–F | Earlier production prose | Production-system machine contracts | Branch implementation may be newer than prose. |
| Phase A implementation | `89f9496:V6/marketmamba/ops/` | `CURRENT_REFERENCE` | `IN_DEVELOPMENT_MAIN`, `NOT_DEPLOYED`; Task 4/5 scoped approvals retained | Earlier candidate-only status; original branches retained | Explicit local integration decision and closure record | Local merge does not grant production or research authority. Shadow evidence does not own legacy fetch success. |
| V7 status API/UI source | [`app/backend/routers/v7.py`](../../app/backend/routers/v7.py), [`app/frontend/src/pages/V7Status.jsx`](../../app/frontend/src/pages/V7Status.jsx), focused tests | `CURRENT_REFERENCE` | `COMMITTED_NOT_DEPLOYED` in WSL development main | Local untracked V7 status candidate | Health summary builder and frontend contract tests | Additive status route only; no model inference, portfolio or scheduler authority. |
| API/UI live deployment | None for live deployment state | `UNKNOWN` | `UNKNOWN` | Claims inferred from source presence | Source paths above | Source capability does not prove deployment; provider revision remains unverified. |
| Agent instructions | [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | Navigation/policy only | Stale duplicated assumptions in `CLAUDE.md` | K0 agents/CLAUDE audits | Machine contracts and this map outrank project-state prose in agent guides. |

## Authority Classes

CPU-only window 01 is branch-local engineering under the existing approved
portfolio engine contract, not a research authority replacement. See
[scope/backlog](../../tasks/v7-cpu-development-window-01.md).
Main stays `83a6a2e`; E1 evidence remains pinned to `11ae71d`.
Neither test success nor Agent work selection grants research acceptance.

E1 recovery observation (2026-10-21):
[inventory](../../research/v7/e1-rolling-origin-refresh-v1/evidence-recovery-20261021.json)
records surviving byte integrity and limited local discovery. It does not
supersede eec63c5 completed-result or ef020e5 diagnostic research conclusions.
Missing matrix identity and unvalidated payload semantics stop replay.

Subsequent explicit Human integration decision authorized local WSL main only:
`89f9496:research/production-system/phase-a/` is now reachable from main.
The [closure record](../../research/production-system/phase-a/integration-closure.json)
records pre/post identity and verification. This supersedes the candidate's
unmerged status, not research/production authority or either scoped approval.

2026-10-21 explicit Human decision supersedes the prior Task 4 approval-pending
state only: [decision record](../02_History/Decisions/20261021-task4-human-approval.json)
binds limited engineering correctness to fixed `1e119ef` and its verified index.
Agent integration/test judgments do not extend that approval. The subsequent
[Task 5 Human decision](../02_History/Decisions/20261021-task5-human-approval.json)
approves only limited offline engineering at fixed `1c5250e`. Historical
acceptance artifacts remain unmodified. Final Phase A acceptance and integration
are still Human decisions; Agent GO is only an engineering recommendation.
Strict Phase 0 STOP,
research authority and production boundaries remain unchanged.

Historical 2026-10-20 bounded Task 4 follow-up (superseded approval status):
`fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
is an unmerged artifact-summary fix/evidence candidate derived from `f5b0ff3`,
not a replacement authority or Human correctness decision. Its evidence index
records 70 ops + 3 mocked fetch-only passes and remaining concurrent snapshot
isolation evidence gaps. Task 5 remains not started; legacy fetch owns outcomes.

Same-day follow-up `fix/phase-a-task4-evidence@1e119ef` closes the two listed
isolation evidence gaps with real-process tests (75 ops total). Its Task 4
review package is ready for Human decision, **NOT_APPROVED**. This supersedes
only the missing-isolation-evidence observation, not authority or research state.

Branch-local recovery candidate (2026-10-20): `fix/v7-snapshot-immutability`
changes only the snapshot writer/reader and tests from main `50c5b78`, with its
[checkpoint](Current_State.md#bounded-recovery-checkpoint--2026-10-20).
It is `UNMERGED_IMPLEMENTATION / NOT_DEPLOYED`, not a replacement of frozen
PIT contracts, source-adoption history, research results or Phase A approval.
The 2026-09-22 byte-identical adoption claim describes that adoption, not this
subsequent candidate. No authoritative evidence has been superseded.

- `CANONICAL`: accepted contract or project-map authority for its explicitly stated scope.
- `CURRENT_REFERENCE`: best current implementation/report, but not necessarily immutable or deployed.
- `RESEARCH_EVIDENCE`: valid research conclusion for its run/evidence class.
- `HISTORICAL_ONLY`: retained context that must not drive current decisions alone.
- `RUNTIME_OBSERVATION`: external state valid only at its observation time and freshness window.
- `UNMERGED_IMPLEMENTATION`: branch-local implementation, neither deployed nor mainline authority.
- `UNKNOWN`: no verified primary authority currently exists.
