# V7 Research Program and Current Development Backlog

> Current planning owner: isolated Window31 expanding development branch, linked by canonical
> Current State/Authority Map. Planning status does not grant research, execution,
> merge or production approval. Existing machine contracts win all conflicts.
> Historical per-window tasks are checkpoints, not parallel active queues.

## Window31 First Expanding CPU Development — COMPLETE / STOP

W30 fixed `b64ac353833a9248121f12ebe61e1333decb2dbf` 已由main e83dfe6本機ff-only整合；38pins/restore及6mock-only tests通過。970 daily＝940 defined＋30 None；2910 paired＝2820 defined＋90 None；原模型351/379 split nodes、預測非退化，未重Fit。見 [Integration](window31-integration.json)。
W31 branch `research/v7-window31-expanding`，base同固定SHA，worktree `.worktrees/v7-window31-expanding`；UNMERGED / NOT_DEPLOYED。首次Fit前 `ddcf5d835b43578ed9fc5567ee677d88e928e203` 固定 [Contract](window31-contract.json)／[Prepared](window31-prepared.json)。
三Origin2023-01-03／04-06／07-03，Forward54／59／63sessions。5d TRAIN14113/17559/21321，REFIT29229/32574/36276；10d TRAIN13711/17143/20890，REFIT28743/32046/35715。持有窗跨界/未成熟Label排除，額外purge/embargo0不是E530sessions。每Origin Validation Fit與Fresh Refit不同run/scaler/state；Validation診斷後refit，全部Forward評分延至三Origin完成，無Forward-feedback tuning。
Ridge12fits＋LGBM12fits已達本輪上限；L0±固定。453192完整Feature-only Predictions先保存後Label Join；7264daily/10896paired，含Validation明確None尾端不當0。Forward352個session×horizon全部可算，Daily N59–64，min40未降。
Forward Ridge5d O1/O2/O3 Mean .028606/.035161/.024427；10d −.012394/.008584/−.002463。LGBM5d .023592/.058135/.065496、10d −.021556/.026531/.000766。全部為EXPOSED DEVELOPMENT，不選winner/不宣稱獨立統計或收益。
10純數值/mock tests PASS；獨立程序Replay453192pred/7264daily/10896paired完全吻合，0追加Fit。72files91000410bytes/份同機Restore，310 prior W29/W30pins未變；CPU87.690630＋60reserve<1800，artifact+restore182002029<512MiB；BACKUP_PENDING。
初始導航authority_class缺漏使healthFAIL；prefit shell未set-e，未攔住O1。已修導航，保留原FAIL與修後WARNING證據，數值code/contract未變；不可宣稱所有preflight從未失敗。獨立Review GO Critical0／Required0，0輪review rework；見window31-review.json。無額外API/GPU/W19或V6操作。
[Run](window31-run.json)、[Results](window31-results.json)、[Evidence](window31-evidence.json)、[Report](../../reports/v7_window31_expanding_cpu.md)。0未授權整合；main保持b64ac35，W31未合併。
下一Human：正式Dataset/Universe與UNKNOWN事件政策、Decision-time/Label/maturity/purge、selection/refit/evaluation/statistics、Exposure/prospective評估、運算/獨立備份。無剩餘已授權READY；不得重跑已用完額度的O1/O2/O3。
恢復只讀入口：上述worktree → Current_State → 唯一Backlog → window31-evidence.json；W29/W30 raw/models/restore、W11環境依賴保留。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED。STOP，不開下一窗口。

## Window30 Fixed Real-data LightGBM Baseline

W29 fixed `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` 已由main1d9a2a7本機ff-only整合；340pins/restore、4tests、ReviewGO核對。
W30 branch `research/v7-window30-real-lightgbm`，base同固定SHA，worktree `.worktrees/v7-window30-real-lightgbm`；LOCAL_INTEGRATED_FIXEDb64ac35 / NOT_DEPLOYED。
任何Fit前60e7ca7固定唯一Recipe/版本/native-lib SHA/feature order/seed/TRAIN view/planned ModelSpec IDs；weights SHA僅Fit後產生。
W11環境LightGBM4.6.0/NumPy2.0.2/SciPy1.16.3/sklearn1.6.1不變；L2/64trees/7leaves/depth3/lr.05/minleaf100/L2=1/maxbin63/seed0/thread1，無eval_set/earlystop/search。
W29 Cohort/Data/兩Features/Labels/Fold/support完全沿用；TRAIN14113/13711，最新exit2021-12-30；identity preprocessing，bin mapper只學TRAIN。
2個fresh LGBM fits各horizon一次，皆64trees，fit約.02777/.02791秒，modeltext52369/54849bytes；Ridge/API/GPU0。
60726完整Feature-only Predictions先immutable保存後Label Join；同W29 support/AVERAGE/min40，970groups940defined N58–64，30tail/maturity保留拒絕。
LGBM Mean/Median：2022/5d .018814/.021627、2022/10d −.012617/−.016793、2023/5d .021182/.026036、2023/10d −.013374/−.013470。
5d正但四組Mean均低於既有Ridge，10d兩年負；沒有改Recipe重Fit、選winner或批准Mamba2。2022/2023 EXPOSED，UNKNOWN actions/asof/vintage與Cohort bias仍保留。
6 mock-only tests PASS；獨立Replay60726pred/970daily/2910paired PASS，0Fit；1主要Review GO Critical0 Required0／0 review rework。
Reviewer不能只由stdout獨立重證shell timeout wrapper；已保存實際tool command/session/start/complete reference，未重跑Fit；OS CPU guard及實測fit保留。
11files34874071bytes/份同機restore、W29原299files hash未變；artifact+restore69749181<256MiB，CPU30.723458＋60reserve<1200；BACKUP_PENDING。
[Recipe](window30-recipe.json)、[Prepared](window30-prepared.json)、[Run](window30-run.json)、
[Results](window30-results.json)、[Evidence](window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)、
[唯一Backlog](research-program.md#window30-fixed-real-data-lightgbm-baseline)。
下一Human：Expanding origins/model-ready cutoffs/train/validation/forward/maturity/purge/selection/refit/evaluation與data/proxy/universe/events/exposure/CPU/storage上限。
固定CPU E2E基礎已驗，但W29 masks為固定split，各新Origin需重新核對maturity/purge，不能直接宣稱Expanding實驗已批准或認證；無剩餘已授權READY，STOP。
StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；不mergeW30/push/deploy/nextwindow。

## Window29 Expanded Cross-sectional Ranking Development

W28 fixed `1d9a2a7a1ee57c455ba58250f63681efbbdd9050` 已由main39e2fed本機ff-only整合；35pins/restore、6tests、ReviewGO核對。
W29 branch `research/v7-window29-expanded-development`，base同固定SHA；worktree `.worktrees/v7-window29-expanded-development`；LOCAL_INTEGRATED_FIXEDe83dfe6 / NOT_DEPLOYED。
GET前746ca04固定68新Cohort（W24重疊0）：60pre2021 current issuers＋4periodlisting＋4delisted start/typeUNKNOWN；64檔進Prediction，仍HINDSIGHT/存活者偏差，非HistoricalUniverseVerified。
68真實FinMind price GET／metadata0／8387840rawbytes，全部HTTP200；單並發、實際起始間隔min5.001297秒，known-local100/hour（外部匿名用量UNKNOWN），unused預算CLOSED。
一次本機mutable counter修復：seq1保留不重送、seq2由Trace證明尚未GET；E1atomic_json只修counter，失敗CPU保守扣600秒未reset；不清除providerSTOP、不擴預算。
最小feature-only介面分離原W24未來Label gate；只讀當前/前一session和當下membership，Predictions先保存後Label Join；未來Label刪除/eligibility變動不能改PredictionUniverse。
48061raw rows／44899feature-only rows；940核心groups N58–64，min40未降；30tail/maturity groups/model N0保留拒絕證據，尾端Predictions仍保存。
原兩Features，L0正／反（反向post-W25 control）、Ridgealpha1兩次freshfit、TRAIN-only scaling，TRAIN N14113/13711。LGBM於GET前略過既有synthetic≤64/toy guard，沒有新Recipe或繞過。
5d Ridge Mean/Median：2022 .037333/.033014、2023 .047138/.050515，正向延續；10d2022 .012073/−.004079、2023 −.008580/−.002417，一致正向未延續。
不能宣稱跨Cohort泛化/優勢/正式統計成立：來源/Universe/support/protocol不同，2022/2023皆EXPOSED；UNKNOWN actions/suspension/asof/vintage保留。
182178predictions，3模型相同可評估Support；18targetedtests，獨立Replay Raw-derived feature/masks＋2910daily/2910paired PASS，0額外Fit。獨立Review GO，Critical0／Required0，1主要輪／0 review rework。
299files／74612661bytes同機restore；總artifact+restore149226490bytes<1GiB，CPU計帳647.740401秒（含failed600）＋60reserve<3600；BACKUP_PENDING。
[Plan](window29-plan.json)、[Cohort](window29-cohort.json)、[Coverage](window29-coverage.json)、
[Run](window29-run.json)、[Results](window29-results.json)、[Evidence](window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)、
[唯一Backlog](research-program.md#window29-expanded-cross-sectional-ranking-development)。
下一Human：補強ordinary/historical membership與事件未知政策，事先固定新資料/新時間或prospective的5d primary/10dsecondary＋L0±/Ridge研究與精確資料/CPU/storage/backup上限；不因本結果調參或選勝者。
本輪無剩餘已授權READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED/W19完全隔離。

## Window28 Clean Training Refit and Temporal Causality

W27 fixed `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8` 已本機 ff-only 整合（main前367005d）；35pins＋原計算source/restore、14tests、ReviewGO核對。
W28 branch `research/v7-window28-clean-refit`，base同固定SHA，worktree `.worktrees/v7-window28-clean-refit`；LOCAL_INTEGRATED_FIXED1d9a2a7 / NOT_DEPLOYED。
Plan/code/prefit先commit `ec0b072` 再執行；原W24兩Features／Ridgealpha1／W27Policy與Mask不變，2021各horizon只fresh fit一次，總2fits，0resume/search。
5d TRAIN 7338→7164、10d 7188→6841；mean/std僅在clean TRAIN重估並逐值驗證。2022/2023完全不參與Fit/Scaling。
同一W27 Eval support、940日期×horizon、N>=20未降：Ridge舊→新 2022/5d .033873→.034682、2022/10d .045471→.044561、
2023/5d .080008→.080407、2023/10d .053166→.054248；2022/2023皆EXPOSED Development，非模型選擇或獨立效能證明。
18個future有效OHLCV數值擾動案例，Features/固定模型Predictions exact-equal、當日positive control有效，Raw不改。
只驗數值因果性：既有feature wrapper也有未來Label資格gate，不宣稱row eligibility為as-of線上接口，亦非歷史publication/vintage認證。
原July1之2321不在W24資料；任何Fit前依原Mask第一共同合格日固定2022-07-18，不改Mask或樣本池。
6targetedtests；獨立Replay1880daily/940paired/29143newRidge scores/18cases PASS，Replay0Fit。獨立Review GO，Critical0／Required0，1輪主要審查／0 review rework。
227 prior private source identities核對；W28 11files／28170524bytes同機restore，BACKUP_PENDING，原W24/W27 models/predictions/metrics均保留。
[Plan](window28-plan.json)、[Run](window28-run.json)、[Results](window28-results.json)、
[Causality](window28-causality.json)、[Evidence](window28-evidence.json)、[Report](../../reports/v7_window28_clean_refit.md)、
[唯一Backlog](research-program.md#window28-clean-training-refit-and-temporal-causality)。
UNKNOWN actions/suspension/asof/vintage、hindsight/universe/已曝光仍保留；不宣稱完整污染已去除或投資收益。
下一Human：限定新資料/universe/事件未知政策及Label用途，凍結Exposure-aware Study/Fold/Recipe/evaluation與資料/CPU/儲存/備份上限；無剩餘已授權READY，STOP。
0 API/GPU/Colab/Bootstrap/LOSO/V6/W19操作。StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED。

## Window27 Pragmatic Label Safety and Frozen-Prediction Sensitivity

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已本機 ff-only 整合（main 前27055f7）；38 pins／3 tests／獨立 Closure GO。
W27 branch `research/v7-window27-label-safety`，base367005d，worktree `.worktrees/v7-window27-label-safety`；LOCAL_INTEGRATED_FIXED39e2fed / NOT_DEPLOYED。
Policy `W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1`／Predicate／Mask 先在 `e77f32d` commit 固定，再重評既有 W24 預測。
原44266→新43148樣本；排除1118（Training521僅診斷，evaluation597），disjoint primary：confirmed event3／missing33／zero1082。
UNKNOWN event/suspension/asof 對所有樣本保留，不升格 CLEAN；新 Mask 不修改原 W24 mask，沒有 refit 或消除原模型 Training 風險。
四組原940日期×horizon仍全可計算；N>=20未降，Ridge均值：2022/5d .037016→.033873、2022/10d .043930→.045471、
2023/5d .080310→.080008、2023/10d .054056→.053166。這是 post-hoc development sensitivity，非正式模型選擇／獨立驗證。
2023/5d排除68筆持有路徑零價，沒有本輪已確認減資跨界；不得用三筆10d事件解釋5d。無 event-only 追加評分。
14 targeted tests PASS；獨立 Replay1880 daily／940 paired，原44266 Mask 由 raw 重建吻合；W24/W25/W26 218 prior byte pins 未變。
一次 import namespace 修正在任何 metric 前；Replay int-key JSON 比較問題以 canonical hash 修復，原計算source留存、calculate AST相同，metrics不覆寫。
獨立 Review GO，Critical0／Required0，1輪主要審查／0輪 review rework；新成果留候選。0 API／fit/refit／new prediction／Bootstrap／LOSO／GPU／V6或W19修改。
Private `.artifacts/window27-label-safety/` 與同機 `.artifacts/window27-restore/`；BACKUP_PENDING，不宣稱獨立備份。
[Policy](window27-policy.json)、[Mask](window27-mask-manifest.json)、[Results](window27-results.json)、
[Report](../../reports/v7_window27_label_safety.md)、[Evidence](window27-evidence.json)、[唯一Backlog](research-program.md#window27-pragmatic-label-safety-and-frozen-prediction-sensitivity)。
下一 Human：更大研究 Universe/事件覆蓋/未知政策/Label路線、Fold/maturity/selection/refit/evaluation/exposure及精確資料/CPU/保存上限；無剩餘已授權 READY，STOP。
StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED；W19完全隔離。

## Window26 Label Integrity and Next Study Contract

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327` 本機 ff-only 整合；
4 focused tests、12 private artifacts／21 deliverable pins 與 Review GO 已核對。W26 base 等於該固定 SHA。
W26 branch `research/v7-window26-label-contract`，worktree `.worktrees/v7-window26-label-contract`；LOCAL_INTEGRATED_FIXED367005d / NOT_DEPLOYED。
2321 已確認 68% 虧損減資；2023-02-17 市場參考價單位變更與 2022-12-08 會計減資基準日分離。
三筆既有 10d Label 算式／市場索引吻合，但跨股數單位變更，不可作未調整持有期報酬。
六個缺行情日期是市場交易日，非市場休市；原始官方停牌公告尚未核實，精確停牌區間保留 UNKNOWN。
其他公司行動覆蓋／historical availability／vintage 未認證；不重寫 W24／W25 歷史結果，不重新評分。
兩路 Label Policy、新 Study、L0 正反向對照與所有日期／閾值／資源額度均為 PROPOSED，待 Human 決定。
曝光帳本核對 E5／E1 machine contracts；2022／2023 已曝光，未列期間為 EXPOSURE_UNKNOWN，不自動成為獨立測試。
FinMind 非價格 1/6、行情 0；官方文件 loads/attempts 4/4；全部預算 CLOSED。0 fit/refit/new prediction/RankIC/GPU。
W26 private root `.artifacts/window26-label-integrity/`；15 files／4,785,564 bytes 獨立同機 Restore PASS；
W24 191＋W25 12 files 原 Hash 保持。BACKUP_PENDING，不宣稱正式備份或完整重現。
[Report](../../reports/v7_window26_label_integrity.md)、[Evidence](window26-evidence.json)、
[事件矩陣](window26-event-matrix.json)、[Study Candidate](window26-next-study-candidate.json)、
[Exposure Ledger](window26-exposure-ledger.json)、[唯一 Backlog](research-program.md#window26-label-integrity-and-next-study-contract)。
Review／tests 狀態以 window26-evidence.json／window26-review.json 為準；本窗口無剩餘已授權工程，STOP。
下一 Human：Label 路線及未知／停牌／除息／下市政策、資料與 Universe 資格、Development-only 或 prospective、
Fold／Selection／Refit／評估／資源上限；本次未批准其執行。Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED，W19 不操作。

## Window25 Ranking Signal Robustness and Leakage Audit

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
[Report](../../reports/v7_window25_robustness_audit.md)、[matrix](window25-leakage-matrix.json)、
[evidence](window25-evidence.json)、[唯一Backlog](research-program.md#window25-ranking-signal-robustness-and-leakage-audit)。
10focused/26mainpost/4independent新testsPASS；GO0CriticalRequired／0rework，無本輪後續READY。
下一Human：公司行動/停牌/adjustment與2321有界補證、universe/PIT或proxy政策、matchedbaseline方向與新未曝光資料/正式統計設計。
本輪無剩餘READY；STOP。StrictSTOP/FormalAdmissionNOT_APPROVED/E1BLOCKED/H2NOT_APPROVED/W12HALTED，W19完全不操作。

## Window24 Expanded Retrospective Ranking Pilot

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
[Report](../../reports/v7_window24_expanded_pilot.md)、[evidence](window24-evidence.json)、
[唯一Backlog](research-program.md#window24-expanded-retrospective-ranking-pilot)。
50focused/26mainpost/44independenttestsPASS；GO0CriticalRequired／0rework；無更多授權READY。
下一Human：正式raw/adjusted/actions政策與historicaluniverse/calendar/Label/eval/compute批准；本輪無剩餘READY，STOP。
W19工作樹/task/artifacts/budget/runtime不操作不干擾，不新增runtime觀測；不mergeW06/W12/W19或W24，不push/deploy。

## Window23 Historical Calendar and CPU Smoke

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
[Report](../../reports/v7_window23_calendar_smoke.md)、[evidence](window23-evidence.json)、
[唯一Backlog](research-program.md#window23-historical-calendar-and-cpu-smoke)。
正式study仍PROPOSED/DataAdmissionNOT_APPROVED；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED；W19完全不操作。
58focused/23mainpost/26independenttestsPASS；GO0CriticalRequired；1rework/no-refit。
下一Human集中決定正式historicalcalendar/universe/action/label/availability/evaluation/compute；無本輪剩餘工程READY，STOP。

## Window22 Continuous Historical Panel

W21兩Required獨立複核CLOSED，15testsPASS/Critical0Required0；133localfiles驗Hash。
main由6a34520 ff-only至 `428a5fa5cb7b771db39a053f88c7046501399f7e`；沒有提前main文件commit。
W22 `.worktrees/v7-window22-historical-panel` / `research/v7-window22-historical-panel`，base428a5fa；LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED。
10FinMindpriceGET、0metadata/diagnostic，均HTTP200，unused20CLOSED；10檔9864rows/1762931rawbytes，2019–2023。
6檔同1216reference日期（不是合格officialcalendar），9725structuralVALID/131UNVERIFIED/8zero，不fill或算return/RankIC。
官方18419/25036/11543CSV各1GET；TPExdelisted403立即停，共4resourceattempts；cataloguedocs另3GET。
6515TWSE上市20210120、8272TPEx上櫃20241023、2499終止上市20201110有officialCSV證據；逐日suspension/membership未知。
實際5d/10d eligible0：calendar/actions/availability/labelmaturity仍UNKNOWN，historicalstudyPROPOSED，不作admission。
Root.artifacts/window22-historical-panel與同機restore保存raw/panel/mask；15產物獨立程序restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window22_historical_panel.md)、[coverage](window22-panel-summary.json)、
[唯一Backlog](research-program.md#window22-continuous-historical-panel)。
23focusedtestsPASS／獨立GO0CriticalRequired／0rework；研究執行無READY。
下一步只建議精準calendar/lifecycle補證與Human統一data/label/eval/CPU授權；本輪完成後STOP。
W19工作樹/排程/Artifacts/budget不讀寫不觸發不整合；W12SAFETY_HALTED/StrictSTOP/E1BLOCKED/H2NOT_APPROVED不變。

## Window21 FinMind Historical Data Proof

COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED；research/v7-window21-finmind-proof，base/main6a34520。
U1：W20verifiedff＋有限匿名實測18/22（14price/3meta/1repeat），六檔116rows/48dates2010..2025；
U2：source/schema/QC/raw/provenance/18independentRestore＋一次review/單輪修復；15focusedPASS；W22已針對R1/R2取得獨立複核GO並整合固定428a5fa。
根目錄.artifacts/window21-finmind-proof/restore保存actualbytes；CLOSED，4unused不可重跑；備份BACKUP_PENDING。
來源CONDITIONAL_CANDIDATE，非Dataset/研究准入；3zeroPrice不補、2marketexception/11unknown保留。
BLOCKED：actualCalendar（vendor含future/20261023休市）、historicalUniverse/lifecycle、missing/actions/Labelmaturity與eval/compute批准。
下市2499earlyprice可取但near/post空不證實missing；上市/轉板確切日期未知。不得以六檔推全市場完整性。
NextHuman：先批准獨立少特徵retrospective dataset範圍／合格日曆生命週期，再Label/eval/有限CPUStudy與取得預算；本輪不開研究。
W19工作樹/Task/artifact/預算不操作，無runtime新觀測；不merge其他candidate。
[Report](../../reports/v7_window21_finmind_proof.md)、[summary](window21-proof-summary.json)、[evidence](window21-evidence.json)。

## Window20 Historical Data Source Decision Sprint

COMPLETE/STOP/LOCAL_INTEGRATED_BY_W21/NOT_DEPLOYED；research/v7-window20-historical-sources，base87695e4。
U1：三類來源documentmatrix完成；U2：主FinMind／備富邦條件建議、次輪22request資格Pilot方案完成。
RECOMMENDATION_PROPOSED：非商業Free/anonymous優先，不是資料准入；多年度Coverage／下市／市場轉換／adjustment／bytesRestore未實測。
BLOCKED：W14兩歷史route下載重用資格、富邦帳號及使用權、永久保存／完整歷史Universe與Label/evaluationpolicy。
下一Human：是否批准6檔×3小窗口18price+3metadata+1重取=22匿名FinMind請求；匿名拒絕就STOP，不註冊/Token/自動備援。
無本輪剩餘READY，不自行開始研究。W19cf1a84f獨立限期Collector不修改／停用／觸發／整合；本輪無新runtime觀測。
[Brief](../../reports/v7_window20_source_decision.md)、[Matrix](window20-source-matrix.json)、[Evidence](window20-evidence.json)。

## Window18 Limited Real Forward Collector Pilot

WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED; branch development/v7-window18-manual-collector,
base/main96e8c0c。W17已ff-only整合，W06/W12未整合。兩GET額度用完，不可重跑。

| ID | Human Objective | Completion / remaining |
|---|---|---|
| W18-U1 | MinimalmanualCLI＋exactlicensedCSV1GETeach | COMPLETE；2HTTP200／279436bytes；原碼與第一次immutable記錄保留；無retry/redirect/fallback |
| W18-U2 | QC／immutable保存／independentRestore／faulttest | COMPLETE；25targetedPASS；一次review3Required→唯一修復輪RED/GREEN/實際restore自行確認，無第二獨立GO；BACKUP_PENDING |

原TPExQC_FAILED(27INVALID)保留；離線修後QC_EXCEPTIONS(27MISSING)，不重取、不重寫舊結果。
資料皆10/22單session，publicationUNKNOWN，Capture時間未qualified；Raw成功不等於研究／股票Universe准入。
下一步 POLICY_REQUIRED：daily19:00候選、執行環境／有界retry／QC缺漏／backup監測通知／停止撤銷；
DATA_EVIDENCE_BLOCKED：TPEx有效日曆、來源更新時刻；不能以牆上日期判stale。
ENGINEERING：每日啟用前將既有manual入口接有限授權run範圍（目前Window18一次性ledger已耗盡），
並驗指定獨立backup/monitoring/恢復，不另建platform。
HistoricalStudy PROPOSED，沿用W17限定官方授權/coverage文件調查建議；本輪無歷史payload/RankIC。
[Brief](../../reports/v7_window18_manual_pilot.md)、[evidence](window18-evidence.json)、[dailycandidate](window18-daily-activation-candidate.json)。
No remainingREADY work in completedgrant；STOP，下一窗口需新授權。

## Window17 Forward Vintage Operational Readiness

WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_W18/NOT_DEPLOYED; branch research/v7-window17-forward-readiness,
base/main000b0b9. W16ff-only整合，W06/W12未整合。

| ID | Objective / Authority | Result / dependency |
|---|---|---|
| W17-U1 | HumanW17 W16資產保護與整合 | COMPLETE；14deliverable／9localhash，279436rawbytes＋manifests／request獨立於worktree持久保存並重驗；BACKUP_PENDING；原129protectedfiles不動 |
| W17-U2 | HumanW17 offline操作準備 | COMPLETE；參數化daily-calendar/asof修訂及storageguard＋既有Synthetic vintage29tests，6日面板；不啟用collector；exactCSV定期服務條件及TPEx直接calendar仍BLOCKED |

下一步 POLICY_REQUIRED：runtime／明確時間頻率與有限重試／QC與missing／storage及獨立backup／通知／backfill／停止恢復撤銷。
LICENSE_SERVICE_UNRESOLVED：每日更新不是numericquota／自動化permission／SLA；不可探測或換API。
DATA_EVIDENCE_BLOCKED：TPEx直接日曆及10/27實際交易證據；已知休市不計漏抓。
HistoricalStudy PROPOSED：下次限定查W14兩歷史route授權及coverage官方文件，或一項Human指定合法歷史產品；不取payload／不全面PIT搜尋／不fit／不RankIC。
No READY execution in this completedgrant；下一窗口須單獨授權。
[Brief](../../reports/v7_window17_activation_brief.md)、[Activation](window17-activation-package.json)、[evidence](window17-evidence.json)。

## Window16 Real Resource Coverage and Capture Proof

WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_W17/NOT_DEPLOYED;research/v7-window16-resource-proof,
base/main27dcb36. W15fixedff integrated, W14/W15rawbytes remainoriginaltrees.

| Unit | HumanWindow16 objective | Result |
|---|---|---|
| W16-U1 | Exactlicensedresource coverage,1GETeach | BothHTTP200UTF8,2026-10-22 only;TWSE1383rows137956bytes/TPEx1017rows141480bytes;nohistory/fallback expansion |
| W16-U2 | Immutablebytes+independentrestore/minimalresilience | 279436rawbytes/hash+rows restored;3targetedSnapshotPASS;fallbackPROPOSED/whitelistempty;no collector |

[Brief](../../reports/v7_window16_resource_proof.md),[evidence](window16-evidence.json).
SingleindependentreviewGO0Critical/Required,0repair;28focusedtests, no fullsuite repeat.
HistoricalTrainingCoverageBLOCKED:two snapshot resources are Forwardraw CANDIDATES
only. No newfit/RankIC/admission/collector/GPU. Human nextchoice: separatelyapprove
legalhistoricalsource coverage qualification, OR limitedForwardoperations preparation
(source/frequency/archive/backup/clock/QC/calendar/missing/notice). No READY execution
within this finishedgrant. W12HALTED,StrictSTOP/E1BLOCKED/H2NOT_APPROVED/
ExploratoryPROPOSED/CollectorNOT_APPROVED unchanged. No nextwindow/fallback access.

## Window15 Licensed Dual-track Foundations

WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;research/v7-window15-dual-track,
base/main4da1bcf. W14ff integrated, originalrawstillW14tree; noW06/W12merge.

| Unit | HumanWindow15 objective | Status / evidence |
|---|---|---|
| W15-U1 | Exactsource/license verification | COMPLETE_REVIEWED;2exactgovCSV OGDL1.0scoped,legacyhistoricalAPIunclear;3HTML1547650bytes archived;0marketpayload;0rework/0CriticalRequired |
| W15-U2 | Separate exploratory study/dataset/features | COMPLETE_REVIEWED;PROPOSED;7synthetictests;DATA_READY/execution/admission=false;lag/paramsnull;0rework/0CriticalRequired |
| W15-U3 | Forwardvintage syntheticdurability | COMPLETE_REVIEWED;18tests/8persistentevents/3vintages;W14+Snapshot+W09reuse,2repairs/1fixturecorrection,0CriticalRequired;no collector |

[Brief](../../reports/v7_window15_dual_track_brief.md),[evidence](window15-evidence.json),
[license](window15-license-matrix.json),[study](window15-exploratory-study.json),
[activation](window15-forward-activation.json). Final175CPUtestsPASS14.625s.
No READY execution in completedgrant. Minimum nextHuman decisions:
1. Approve bounded exactlicensedresource/coverage/bytes/restore qualification, plus
   necessary historicalcalendar/universe/label source qualification; not realfit yet.
2. Freeze independentStudy evidenceclass/feature/timing/maturity/fold/purge/selection/
   refit/eval/final/CPUbudget and data/researchadmission once evidenceavailable.
3. Separately approve TrackB exactsource/frequency/service/archive/backup/capacity/
   clock/QC/missing-backfill/monitoring/notice operation scope. None activated here.
Oldhistoricalroute permission unclear, originalvintage missing, no conservative lag
assumed. Actualdataset DATA_READY=false; licensing2currentresources is one cleared
scopeddependency only. Existing adapters stillsynthetic-only; scopeextension requires
review/grant, no statusflipshortcut. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/
PreNextOpenPROPOSED/GPU_SAFETY_HALTED/RecoveryDEFERRED unchanged; no nextwindow.

## Window14 Evidence-first Admission Pilot

WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED,branch research/v7-window14-data-admission,
base/mainc78098f. W13ff integrated; W06/W12/other candidates remainunmerged.

| Unit | HumanWindow14 objective | Status / evidence |
|---|---|---|
| W14-U1 | Bounded official daily-data pilot/classification/brief | COMPLETE_REVIEWED;2responses337221bytes/4stocks/2dates;RETROSPECTIVE_ONLY candidate,0asof/0proxy;rights separateUNKNOWNgate;1docrepair,0Critical/Required |
| W14-U2 | ForwardVintage proposal/synthetic readonly checks | COMPLETE_REVIEWED;Snapshot/W07/W09reuse;10synthetictests/150jointPASS;1containerrepair,0Critical/Required;no collector/admission |

[Decision Brief](../../reports/v7_window14_decision_brief.md),[evidence](window14-evidence.json),
[classes](window14-evidence-classes.json),[forward proposal](window14-forward-contract.json).
No READY execution remains in grant. Minimum nextHuman decisions:
1. Confirm permittedsource/API/open-data/contract and research/storage/redistribution use.
2. Qualify pinnedbytes/calendar/universe/labels/restore and historicalvintage evidence;
   or separatelyapprove distinct conservative-lag/revision proxy/newfeatureidentity;
   missingoldvintages may need licensedarchive evidence or futureforward accumulation.
3. Freeze actualtiming/missing/dates/maturity/purge/eval/selection-refit/statistics/final/
   CPUcompute before anyrealbaselinefit. Fourrows cannot supply rollingfeatures/labels.

Prototype/callerpins/nowdownloads/QC cannot clearhistoricalPIT orrights. No further
collection, realRankIC, fit, GPUretst or newwindow. Raw4files localretained, notGit.
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED. W11bytes/W12HALTED retained unchanged.

## Window13 Thermal and Admission Preparation

WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Branch research/v7-window13-admission-preparation,
base/main8db unchanged. U1sourcef26dbf5; finalHEAD containingU2closurecommit.

| Unit | Authority/objective | Status / evidence |
|---|---|---|
| W13-U1 | HumanWindow13 preserve thermal direction | COMPLETE_REVIEWED;87C reference/futurestop,83C warningPROPOSED,GPU_RETEST_DEFERRED;oldHALTEDbudget retained;0Critical/Required/0rework |
| W13-U2 | HumanWindow13 minimal data/decision-time/evaluation brief | COMPLETE_REVIEWED;DT1–4/EV1–6 PROPOSED;140main+13fixedW06CPUtestsPASS;24sourceidentities/2raw176096bytes checked;no qualified full48panel in named scope;0Critical/Required/0rework |

[Decision Brief](../../reports/v7_window13_decision_brief.md),
[Decision Record](../../knowledge/02_History/Decisions/20261024-window13-thermal-admission-preparation.json),
[evidence](window13-evidence.json),[bounded verification](window13-verification.json).
No READY execution remains in completedgrant. Next priorities (HUMAN_DECISION_REQUIRED):
1. DT1–4 actualfreeze/deadline/source/missing/stale/revision/evidenceclass/storage.
2. Specify authorized immutable smallpanel/calendar/universe/label/restorebytes;
   DATA_EVIDENCE_BLOCKED if unavailable, or separatelyapprove forwardvintage collection.
3. EV1–6 actualfolds/maturity/purge/evaluator/selection-refit/statistics/final/compute;
   then consider separatelyauthorized smallCPUbaseline research. GPU SAFETY_HALTED
   is independent; no retest/reset/driver/thermal change here.

ENGINEERING_READY limited existingW07–11 synthetictools/CPUevidence. W06/Protocol/
Recovery/W12 remain fixedSHA references only, not merged/adopted; formalresearch/newdata
admission unapproved. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/
PreNextOpenPROPOSED/GPU_SAFETY_HALTED/RecoveryDEFERRED. No nextwindow execution.

## Window11 L2 and CPU Resume

WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Branch development/v7-window11-l2-cpu-resume,
base/main f251816. Sole queue below; older window entries historical.

| Unit | Objective/authority | Acceptance/status |
|---|---|---|
| W11-U1 | Human actual LightGBM completion | COMPLETE_REVIEWED:LightGBM4.6.0 actual fit/refit/commonSupport/replay;5officialwheels pinned;setupcorrection1,0reviewrework,0Critical/Required |
| W11-U2 | Human real PyTorchCPU interrupt/resume via E1 | COMPLETE_REVIEWED:9CPUtests,2bounded reworks;actual0/75/0processes,12steps/4epochs,23fieldsx3roles exact,0Critical/Required |
| Closure/source handoff | Required evidence consistency | COMPLETE:211jointtestsPASS;new source-bound L2run uses U2commit, initial artifacts retained;source/wheel/data hashes checked |

Source U1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b. Final checkpoint is containing
closure commit. No READY work remains in this exhausted grant. Exact proof only
CPU FP32/1thread/deterministic daily cursor/0workers/fulloptimizerstep. No GPU/AMP/
shuffled/multiworker/midbatch/full reproduction certification. CPU checkpoint bytes
remain local .artifacts inventory; missingbytes means NOT_RESTORABLE, not hash recovery.

Next PROPOSED_NEXT_GRANT: define a bounded LocalGPU synthetic compatibility/resume
acceptance window (exact model/kernel/environment/precision/state/resource limits),
then obtain Human approval. No execution here. Real-data training additionally needs
qualified dataset/PIT/evaluation/selection/decision-time and research admission.
Permanentstorage/retention/GC and GPU/Colab resource conditions remain Human decisions.
E1matrixBLOCKED/H2unapproved/StrictSTOP/FairEvaluation&PreNextOpenPROPOSED unchanged.
[Evidence](window11-evidence.json),[report](../../reports/v7_window11_l2_cpu_resume.md).

## Window10 Classical Synthetic Foundation

WINDOW_COMPLETE/STOP. Sole queue: Human Window10, exactly two CPU/SYNTHETIC units. Main91fa82c,
branch `development/v7-window10-classical-e2e`, UNMERGED/NOT_DEPLOYED. Older queues
historical. No real research/model training admission implied by tiny toy fits.

| Unit | Objective/authority | Acceptance/status |
|---|---|---|
| W10-U1 | Human Classical Baseline Adapters, W07 guards | COMPLETE_REVIEWED:13tests/independent13PASS,1rework,0Critical/Required; L0/Ridge tinyfits; L2 code present but dependency BLOCKED |
| W10-U2 | Human synthetic end-to-end lifecycle | COMPLETE_REVIEWED:13tests; independent53 incl40oldW09PASS;1rework/0Critical/Required; val+selection/forward replay and v2 fit lineage verified |
| L2 real synthetic fit verification | Human allows only installed qualified dependency | BLOCKED: lightgbm absent; candidate requirement >=4,<5, exact lock/test still needed; no network/global env change |

U1 4ec22a4330ed6db7df0b3dafb4106cea820516e2; U2/final checkpoint is containing closure commit.
167jointtestsPASS. No READY work remains under this completed grant. Next high-value
CPU candidate PROPOSED_NEXT_GRANT: exact isolated LightGBM dependency lock + tiny
fit/scoring/e2e verification; do not claim existing untested L2 branch complete.
Existing E1 resume/trainer is not rebuilt. Actual runtime-state/bundle binding, full
source/runtime closure and real data eligibility need separate admission/authority.
Permanent artifact storage/retention, qualified real data, evaluation policy,
actual CPU/GPU runtime acceptance require Human decisions. No Window11 authority.
[Evidence](window10-evidence.json),[report](../../reports/v7_window10_classical_e2e.md).

## Window09 Durable Artifacts and Offline Replay

WINDOW_COMPLETE/STOP. Current sole queue: Human Window09, two SYNTHETIC/CPU units only. Branch
`development/v7-window09-artifact-replay`, base/main bdea30d. UNMERGED/NOT_DEPLOYED.
Window08 integrated after pre/post101tests/independentGO; older queues historical.

| Unit | Objective / authority | Acceptance / status |
|---|---|---|
| W09-U1 | Human Durable Artifact Completion; reuse E1 filesystem/hash primitives | COMPLETE_REVIEWED:20tests; stage/verify/atomicpublish/idempotence/immutable/conflict/failures/concurrency;2rework,0remainingCritical/Required |
| W09-U2 | Human Offline Replay/Restore | COMPLETE_REVIEWED:20tests including separate CPU CLI; exact W08 daily/pair/rejection replay, safe JSON-only checkpoint/dataset eligibility;1rework,0Critical/Required |
| Future Window10 | Further development | NOT_AUTHORIZED; stop after this bounded window |

U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final U1 container fix/U2 in containing closure commit.
141joint tests/independent40PASS;2schemas/3outputsPASS. No READY task remains within
this exhausted window. Next high-value candidate (PROPOSED_NEXT_GRANT): bind actual
future runs to immutable data/runtime states and full dependency/bundle identities
using existing E1 pipeline, after data/research permission; no duplicate trainer.
Permanent storage location, retention/GC, formal data/admission and GPU runtime
acceptance remain HUMAN_DECISION_REQUIRED. No real Frozen Matrix bytes recovered.
[Evidence](window09-evidence.json),[report](../../reports/v7_window09_artifact_replay.md).

## Window08 Common Ranking Evaluation

Human Window08: Window07 fixed1631020 integrated local main; Window08 base1631020,
branch `development/v7-window08-ranking-evaluation`, UNMERGED/NOT_DEPLOYED.
WINDOW_COMPLETE/STOP:U1 826b8ddf798e752584d180181d24cc1db967e019; U2 containing closure commit.
101joint CPUtests,24expectedcases,2schemas/3outputsPASS. Two units only; no real
labels/training/policy adoption. Older queues
are historical; this section is the sole current queue.

| Priority / unit | Objective / authority | Acceptance / state |
|---|---|---|
| 1 W08-U1 | Human StageB common ranking + W07 isolation | COMPLETE_REVIEWED:19synthetic+independent19 tests;1rework/0remainingRequired; explicit tie/minN/weight policy, metadata-first maturity, keyed traceable support/daily rejections |
| 2 W08-U2 | Human StageC fair paired comparison | COMPLETE_REVIEWED:15synthetic tests; common valid support/daily deltas, exclusions, actual-session periods;0rework,0Critical/Required |
| Next Window09 | Artifact/experiment tracking | NOT_AUTHORIZED: consume stable output schema only after separate bounded grant; no registry built |

No remaining READY implementation under the exhausted Window08 grant. Next high-value
CPU candidate (PROPOSED_NEXT_GRANT): consume these interfaces to retain/replay full
prediction+eligibility inputs and manifests; reuse current hashes/manifests, no platform.
Dependencies remaining: qualified actual data/source/vintage/time evidence and Human
research decisions (minN/ties/aggregation/statistical policies/admission). E1matrix
BLOCKED, StrictSTOP, H2unapproved; FairEvaluation/PreNextOpenPROPOSED, RecoveryDEFERRED.
No FrozenE5/E1 edits. [Report](../../reports/v7_window08_ranking_evaluation.md),
[evidence](window08-evidence.json). Resume named branch/maps/Charter + this queue.

## Window07 Temporal Fold & Research Isolation

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED; branch
`development/v7-window07-temporal-isolation`, base/main465dca2. Current grant only
U1/U2; newstudy parameters not approved. No duplicate trainer/orchestrator/registry.
U1 source5d736e9; containing Unit2/closure commit is final source identity.67related
CPUtests and16explicitexpectedcasesPASS. STOP; no Window08 or mainintegration.
Resume by verifyingHEAD/clean in this tree, Current State/Authority/Charter and
this section. A new bounded grant and any necessary policy choices are required.

| Unit | Objective/authority | Acceptance/state |
|---|---|---|
| W07-U1 | HumanUnit1 temporalfold+label eligibility, existing temporal/hash primitives | COMPLETED_REVIEWED:13synthetic+independent13PASS;calendar nextsession/horizon, owner/source/vintage/pins/clocks/maturity/overlap/gap checks;1labelbinding rework,0remainingRequired |
| W07-U2 | HumanUnit2 data/read isolation, fresh refit identity, forward exposure | COMPLETED_REVIEWED:11focused tests;only eligible views,noForwardlabel reads,freshRefit/deadline/exposure;1source-hash rework,0remainingRequired |
| Future Common Evaluator | Prior auditRP-C2 | PROPOSED_NEXT_GRANT: can consume accepted identity/support views after this foundation; actual tie/minN/missing/statistical/admission rules not frozen |

[Report](../../reports/v7_window07_temporal_isolation.md),[evidence](window07-evidence.json).
Reuse mainnormalize_calendar/canonical_hash; W06validator96c6056 reference only,
not merged/admitted. All calendar data synthetic, originalE1matrix stillBLOCKED;
no data reacquisition/realRankIC/training/GPU. H2unapproved/StrictSTOP/newFairEval
andPreNextOpenPROPOSED/InferenceRecoveryDEFERRED. Max2reviewrework/unit.

## Window06 — Window05 integrated; decision-time preparation authorized

Fixed `65462ec185783eedb2ba81b66cfa04783f86c6e2` integrated ff-only from main
`2ee4ca83e709af64be4dd63032ba4fb47903d769`; pre/post87 tests PASS,13source/output hashes
verified, independent13tests/0Critical/Required. LOCAL_INTEGRATED / NOT_DEPLOYED.
This containing main commit is integration closure documentation only.
[Evidence](window06-integration.json).

Same Human grant authorizes isolated `research/v7-window06-decision-time` at
`.worktrees/v7-window06-decision-time` from this closure: VERSIONED PROPOSAL,
<=3source families/~10sessions diagnostic panel and synthetic eligibility tests.
New pre-next-open proposal does not replace frozen E5/E1. No formal cutoff/time,
feature/label/admission or execution policy selected. StrictSTOP/E1BLOCKED/H2unapproved/
FairEvaluationPROPOSED remain. No F2 repeat search/training/production changes.
Read local Current State, Authority Map, Charter and unique Backlog on that branch
for subsequent progress; all pending documents stay isolated. Older STOP entries
are historical for their windows; no indefinite authorization.

## Window05 Feature Integrity and Cost-Effective Data Research

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Human grant covered only F0/F1/F2,
CPU/read-only/synthetic preparation. Base main2ee4ca8; branch
`research/v7-window05-feature-integrity` at `.worktrees/v7-window05-feature-integrity`.
Prior queue rows below are historical checkpoints; this section is the single active queue.

| Unit | Objective/authority | Status/acceptance/dependency |
|---|---|---|
| F0 | Human Stage A + frozen48manifest/policy; audit not selection | COMPLETED source-level audit48ordered,4tests+independent4,0Critical/Required;21conditional27unknown; no payloadPIT promotion. Source syntax typo corrected once before firstGREEN, no review rework |
| F1 | Human Stage B exploratory low-cost candidates | COMPLETED_SYNTHETIC_ONLY: timestamp-gated formulas9tests/independent9PASS,0Required,1timezone rework; real diagnostics BLOCKED; no training/input-dim changes |
| F2 | Human Stage C official/repo PIT feasibility | COMPLETED_NOT_CURRENTLY_FEASIBLE:4original metadata blobs verified/independent0Required;0newqualified rows,0rework; no new collector or policy |

Priority: resolve t-close versus later decision semantics and availability evidence
before new real-data feature diagnostics. F0 risk prohibits dependent empirical
analysis; independent F1synthetic/F2feasibility were permitted within this now-closed
window and both completed. Historical E5/E1
results unchanged. H2 NOT_APPROVED/StrictSTOP/E1BLOCKED/FairEvaluationPROPOSED.
All engineering is additive research helpers, no production/contract edits.

Final checkpoint: this containing F2/closure commit; F0 source9fbf1bd, F1source1a6a452.
87 related tests PASS, F0 independent4/F1independent9, all3reviews0Critical/Required.
F0 initial syntax correction1/reviewrework0; F1reviewrework1; F2rework0. STOP.
Main remains2ee4ca8; no integration/push/deploy. Read local Current State/Authority,
Charter and this section after checking branch/HEAD/clean. No full Roadmap Recovery
or frozen-matrix rescan; next execution needs a new bounded Human grant.

Highest-value next preparation: Human-frozen decision/availability semantics, then
identity-qualified small unscaled panel + publication/vintage/missingness sidecar
and matched historical-universe quality audit. Empirical F1diagnostics stay BLOCKED;
synthetic formula completion does not admit any feature. F2forward-vintage pilot
is only PROPOSED and needs source/terms/coverage/revision/cadence approval before
collection. Genuine revenue surprise lacks ex-ante expectations. No GPU/Colab.
Long-term Mamba-2, expanding baseline, quarterly refresh and ranking are planning
directions, not amendments to existing frozen E5/E1 or current execution permission.

[Evidence](window05-evidence.json), [F0 report](../../reports/v7_window05_feature_integrity.md),
[F0 matrix](window05-feature-integrity.json). Maximum2evidenced review-rework rounds/unit.

## Window04 W04B-LOG integration closure — STOP

Human-approved fixed `97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9` integrated by local ff-only
from `31f7accc475b5b735684d5a06725a237d85d5693`. Engineering source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`;
no new merge commit. This containing main commit adds closure documentation only.
LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED / WINDOW_COMPLETE / STOP.
Pre/post274 tests each PASS; independent runtime6 PASS,0 Critical/Required.
Both source hashes match;23 existing worktrees clean before integration and preserved.
[Closure evidence](window04-log-integration-closure.json).

Current continuation entry is WSL main: verify HEAD/clean, read this section,
Authority Map, `knowledge/GOVERNANCE.md` and the local unique
`research/v7/research-program.md`. Older branch-local/unmerged pointers below are
historical, superseded only for W04B-LOG local integration. No new work authorized.
No additional high-value READY item was established by the prior bounded discovery;
future concrete CPU defects can be considered under a new bounded grant.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; research options PROPOSED.

### Next development readiness (assessment only)

| Priority / item | State / prerequisite |
|---|---|
| Fair evaluation preparation continuation | HUMAN_DECISION_REQUIRED: choose/freeze from existing PROPOSED package; only then implement split/selection/config fixtures under new bounded grant. No new design here |
| MOPS acceptance completeness | HUMAN_DECISION_REQUIRED: explicit monthly completeness policy; existing parser semantics unchanged |
| Original E1 payload identity | BLOCKED: original Frozen Matrix/new location evidence required; no repeated search or substitute generation |
| Portfolio/ranking/economic extension | HUMAN_DECISION_REQUIRED: H2 and financial definitions; H1 fixture slice already completed |
| Independent CPU reliability defects | No additional concrete high-value READY item established; future evidence may justify new bounded engineering independent of research gates |

## Window04 Stage B — CPU continuation

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED; Human approved same-window continuation after
successful fixed98f270d integration. Base is main closure31f7acc, branch
`development/v7-window04-cpu-continuation`. Maximum4 units, max2 rework rounds/unit.

| ID / priority | Objective / authority | Acceptance / resources / duplicate check | State |
|---|---|---|---|
| W04B-LOG / P1 | System reliability + experiment preparation: existing durable-log/explicit-failure contract, Training domain invariants and admitted runtime a7992f0 | Real CPU child + injected log write/decode errors must raise with cause/path, stop child, never return success after reader failure; legal success/nonzero/heartbeat unchanged. No GPU/data. All-branch path history only a7992f0/044bfa6/0b7c98e: no equivalent fix. Prevent hidden evidence loss and wasted future compute; thread cleanup risk; bounded S effort | COMPLETED_VERIFIED: RED6tests/3failures → GREEN6; independent6PASS/0Required; one initial fix,0 rework |
| W04B-DATA / P2 | Existing TWSE/TPEx/MOPS/PIT scope | Existing parser and snapshot regressions passed; known deterministic date/header issues already integrated; further monthly completeness policy needs Human. No production-path changes or new acquisition | NO_NEW_READY_UNIT_ESTABLISHED |
| W04B-PORT / P2 | H1 fixture journal/replay/history | Existing67tests PASS and completed PORT01–04; real-data adapter/economic metrics depend on H2/frozen financial definitions | HUMAN_DECISION_REQUIRED for extension |
| W04B-EVAL / P2 | Existing E1 preparation | Metadata audit already integrated and verified; payload checking needs original matrix; new split/selection needs Human. Do not duplicate decision package | BLOCKED / HUMAN_DECISION_REQUIRED |

W04B-LOG source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`; final274 joint tests PASS, independent6 PASS,
0 Critical/Required; initial fix1/review rework0. One of maximum4 units completed.
Final re-evaluation found no additional demonstrated independent high-value READY
unit; no full roadmap recovery, repeated fairness proposal or E1 location rescan.
STOP: no background continuation. A new bounded grant is required. Main remains
31f7acc; this containing docs-only closure commit is the final branch checkpoint.
Resume in `.worktrees/v7-window04-cpu-continuation`, verify HEAD/clean, read
Current_State, Authority_Map, `knowledge/GOVERNANCE.md` and this section.
[Evidence](window04-cpu-continuation-evidence.json). Future decision: Human review
of this fixed engineering candidate; H2/MOPS/research choices are separate and
not prerequisites for unrelated future CPU engineering if a concrete defect emerges.

## Window04 integrated; bounded CPU continuation authorized

Human-approved fixed `98f270dd63f05831b2a42695d8005638fda2bbf5` was ff-only integrated
from main `c733a53f81ff7bd572739b5d75f8baabf1189976`. LOCAL_DEVELOPMENT_INTEGRATED /
NOT_DEPLOYED. Pre/post272 tests each PASS; independent metadata5 PASS,0 Critical/Required.
This containing main commit is integration closure documentation only.
[Evidence](window04-main-integration.json).

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
conflicts. [Integration evidence](window04-reconciliation-evidence.json). The containing docs-only
closure commit is the final candidate SHA; main and original candidate are ancestors.
Resume by verifying branch/HEAD/clean here, reading this section and the evidence,
then await Human approval of that fixed SHA. Do not start Stage B.
Strict Phase 0 STOP; E1 Frozen Matrix BLOCKED; H2 NOT APPROVED; comparison options
remain PROPOSED. Read the local unique `research/v7/research-program.md` first
for reconciliation checkpoint; original preparation artifacts are unchanged.

### Reconciliation completion and next decision

- W04-GIT: COMPLETED_VERIFIED / GO_CANDIDATE_ONLY, current Human objective;
  two-parent history preserved, 272 joint tests and independent review passed,
  zero rework rounds. Acceptance is engineering integration only.
- W04-MAIN: HUMAN_DECISION_REQUIRED for the containing final candidate SHA;
  main stays c733a53. Stage B NOT_STARTED and not authorized by this grant.
- Process suggestion only: keep pending-integration closure documents on candidate
  until approval/merge. Avoid main-only navigation commits while a SHA is pending;
  if main advances, reconcile and review a new SHA before requesting approval.
  This recommendation does not modify global governance.

## Current development program — 2026-10-21

### Fair evaluation preparation window 04

## Fair evaluation preparation complete — fixed continuation pointer

Window02/03 source is integrated in local main at approvedeba6b649; this main
commit adds navigation only. New Window04 work remains UNMERGED/NOT_DEPLOYED on
`development/v7-fair-evaluation-decision@489a127cf951f93f7bfc873e9333f05113f5b8e9`, base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`.
Worktree `/home/frank/projects/MarketMamba/.worktrees/v7-fair-evaluation-decision`.
WINDOW_COMPLETE / STOP. Readiness tool2a9de32 and decision package489a127 completed;
64 related non-training tests PASS,15 source hashes/cost arithmetic verified,
independent reviews0 Critical/Required. New comparison options remain PROPOSED;
no dates/thresholds/training/H2 approved. Original matrix BLOCKED, Strict Phase0 STOP.

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED for Window04; Window02/03 now integrated in local main at
exacteba6b649, with pre/post267 PASS. This containing docs commit records closure.
New work uses `development/v7-fair-evaluation-decision` from main closuref31fcbb;
no new comparison or training approved. Readiness checkpoint in containing commit;
[window evidence](fair-evaluation-preparation-evidence.json).

| Unit | Objective/authority | Acceptance/status |
|---|---|---|
| W04-INT | Human fixed integration | COMPLETED;267pre/post, hashes, independent GO; no deploy |
| W04-DECISION | Human Stage2/frozen E5/E1 evidence | COMPLETED_VERIFIED15sources/independent0 Required: extend existing review into data-exposure ledger, interval requirements, symbolic fit counts/cost and Human options; all PROPOSED |
| W04-READINESS | Human Stage3/existing contracts | COMPLETED_VERIFIED5tests/independent5: reproducible read-only integrity audit joining existing contract/config/result/checkpoint metadata checks; fail closed on drift; no evaluation or payload loading |

Window04 closed: readiness2a9de32 (1 repair), decision package/closure containing
commit (0).64 relevant non-training tests PASS; independent reviews0 Required,
15 source identities/cost arithmetic PASS. [Decision package](../../reports/v7_fair_evaluation_decision_package.md),
[evidence](fair-evaluation-preparation-evidence.json). All new comparisons PROPOSED;
no dates/thresholds or training approved. Main integration remains separate fact.

Resume: cd `.worktrees/v7-fair-evaluation-decision`, verify HEAD/clean, read Current
State/Authority Map and this queue; main pointer pins closure SHA. No historical
rediscovery or matrix search. Next valuable CPU work is implementing a Human-frozen
session split/selection/refit/support design, or payload identity checks when original
matrix becomes available. Both currently dependency-bound, not unqualified READY.
New bounded execution grant is required; no background continuation.

Latest unique backlog is that pinned commit's `research/v7/research-program.md`
Window04 section; report `reports/v7_fair_evaluation_decision_package.md`, evidence
`research/v7/fair-evaluation-preparation-evidence.json`. New audit is not merged
into main. To resume, cd named worktree, verify HEAD/clean, read Current State,
Authority Map and that queue. No full historical rediscovery or repeated matrix
search. New bounded execution needs Human grant; no background continuation.
This pointer supersedes earlier Window04 authorized/READY text below.

### Data evaluation window 03

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Human approved deterministic parser engineering and research
preparation, not main merge, training or policy changes. Candidate branch
`development/v7-data-evaluation-preparation` from main9956446 plus Window02f34f9f3.
This queue supersedes Window02's next-decision status for the newly authorized scope.

| Unit | Priority / authority | Acceptance / resources / status |
|---|---|---|
| W03-INT | P0 Human Stage1 | Both fixed parents, four source units unchanged,259 joint non-training regressions,source hashes/governance/independent review; GO_ISOLATED_ONLY, containing merge commit |
| W03-PARSER | P1 Human Stage2/existing fetcher contracts | RED then legal input compatibility and wrong date/header fail-closed; mock only; COMPLETED8PASS/independent8,1 repair. All existing worktrees examined, W02 diagnostic only, no duplicate fix |
| W03-FAIR | P1 Human Stage3/frozen E5/E1 contracts | Trace split/selection/final support, evidence limits, compare temporal holdout/walk-forward/nested designs; proposals only; COMPLETED_VERIFIED independent0 Required |
| W03-PREP | P2 Human Stage4 | COMPLETED_CPU_METADATA_CHECKS: reuse existing validators,12 source identities/7 result hashes/2seed chains; NOT_RUN_READY without original matrix |
| W03-MOPS | P1 existing monthly/quarterly parser contract | COMPLETED existing quarterly/monthly regression; monthly all-market acceptance policy remains HUMAN_DECISION_REQUIRED, no invented rule |

Closed after integration, parser and research/readiness preparation. Integration9bb4247
(0 repair), parser74ccb2d (1), research/preparation in containing commit (0).
Independent reviews0 Critical/Required;267 tests PASS. See
[report](../../reports/v7_data_evaluation_window03.md) and
[evidence](data-evaluation-window03-evidence.json). New execution needs a bounded
Human grant; no additional unblocked high-value READY item demonstrated.

At most two evidence-backed repair rounds per engineering unit. Data fixtures only;
no formal data, source fetch, historical matrix regeneration or training. Record
unit evidence in data-evaluation-window03-evidence.json; this window is now CLOSED.
Resume at `.worktrees/v7-data-evaluation-preparation`, verify branch/HEAD/clean,
read Current_State/Authority_Map and this queue. Main9956446 remains unchanged.
Human decisions: candidate integration, MOPS monthly policy, exact new comparison
design and compute grant. E1 matrix BLOCKED; no repeat search without new evidence.
Next valuable CPU work depends on these: approved session-split/acceptance fixtures,
or verification of original matrix when located. Do not invent policy or start now.

### Cross-domain window 02

Status: WINDOW_COMPLETE / STOP_AFTER_WINDOW / UNMERGED / NOT_DEPLOYED. StageA main6d74d738→f7af348 ff-only,
then docs closure `c1179fe`; pre/post254 PASS, independent0 Required/Critical.
[Decision](../../knowledge/02_History/Decisions/20261021-charter-integration-window02.json),
[integration evidence](charter-integration-20261021.json). New isolated branch
`development/v7-cross-domain-window-02` from exactc1179fe. UNMERGED / NOT_DEPLOYED.

| ID / priority | Objective / authority | State / resource & Human dependencies | Acceptance / value / risk / duplicate check |
|---|---|---|---|
| W02-DATA / P1 | TWSE/TPEx/MOPS reliability diagnosis; Data provenance boundary + current Human direction | COMPLETED_VERIFIED diagnostic only; offline provider fixtures, CPU; production fixes HUMAN_DECISION_REQUIRED | Pin actual parser AST/source, contrast valid/mismatched dates/schema/missing-market cases; report exact expected vs observed; no network/production import side effects. Makes source risk actionable; no certification or new pipeline. Existing validation scripts require operational data/network; reuse code semantics without invoking them |
| W02-PREP / P1 | E1 frozen contract/matrix metadata integrity before training; existing frozen experiment+Training identity contract | COMPLETED_VERIFIED; 34 non-training tests PASS, independent14 PASS; metadata only | Reject hash/semantic/feature-order/file-set drift even if self-rehashed; valid frozen contract/manifest unchanged. Prevent expensive run against wrong identities; no new experiment. All-branch history inspected: no equivalent fix |
| W02-EVAL / P1 | Frozen two-seed decision completeness; SEEDS/ACCEPTANCE contract | COMPLETED_VERIFIED; 6 tests PASS + independent6; synthetic summaries only | Missing/wrong seeds or nonfinite gate inputs cannot PASS; preserve legal decision vocabulary/thresholds. Prevent vacuous apparent research success; no historical recalculation. Existing evaluator lacks seed-set validation |
| W02-MODEL / P2 | E5/E1 architecture/ablation/evaluation fairness audit; research objective + Human prep grant | COMPLETED_VERIFIED research preparation; 11 pinned sources reviewed | Distinguish trained ablation vs occlusion vs proposal; trace selection/evaluation support and test reuse; explicit PROPOSED choices, source hashes, no inferred research approval; adds code-level trace beyond Window01 decision summary |
| W02-DATA-FIX / P1 | TWSE date binding / TPEx header-index drift | HUMAN_DECISION_REQUIRED for production edits | Proven7-case probe; no historical data correction authorized; pin source and fix original parser only after scoped production approval |
| W02-MOPS-POLICY / P2 | Monthly market completeness and publication provenance | HUMAN_DECISION_REQUIRED | Existing quarterly all-market guard is not automatically monthly policy; do not usecache Date as PIT |
| W02-PORT / P3 | Portfolio/Operations reliability | DEFERRED; existing H0/PhaseA complete | No demonstrated higher-value gap than first3 domains; avoid repeating UX/low-impact bug work |
| DATA-01 | Frozen E1 matrix | BLOCKED | No new location; no repeated search or generated substitute |
| RANK-01/METRIC-01/RES-02/TRAIN-01 | Ranking/economic/new research | HUMAN_DECISION_REQUIRED | H2, strict correctness, frozen new design/compute; preparation does not admit execution |

Limits: four selected units maximum for this window;2 evidence-backed repair
rounds each. No model training, including tiny smoke; no GPU probes, formal data
or weights loaded. NETWORK=0. Existing recovery venv Python3.12.3, CUDA hidden,
MARKETMAMBA_DATA_ROOT unset; candidate Data absent. Main data symlink to Windows
remains untouched. Only temporary synthetic fixtures. Commit evidence and update
this single queue per unit. STOP after these units or unresolved authorization.

Resume: this window is CLOSED. Verify branch/base/status and read this section,
Charter and existing window02 evidence. No unit may continue under this closed grant;
a new bounded Human grant is required. See final checkpoint below for commits
371d55d/e17fcf0/164dbc1 and MODEL containing closure commit, pinned by main navigation.
Other17 checkouts retain their purposes; fixed Charter staysf7af348. No main merge
of window02 implementation.

### Autonomous charter window 01

Status: `WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW`; base `6d74d738`; isolated
`development/v7-autonomous-charter`, UNMERGED / NOT_DEPLOYED.
[Charter](../../knowledge/GOVERNANCE.md#autonomous-development-charter--2026-10-21)
and [Human grant](../../knowledge/02_History/Decisions/20261021-autonomous-charter.json)
supersede the old no-next-program restriction below for this window only.
The previous ranked backlog and S1–S15 remain historical objective/dependency
sources; the following table owns current execution state. No duplicate queue.

| ID / priority | Objective / authority | State; dependencies/resources | Acceptance / value / risk / duplication |
|---|---|---|---|
| AUTO-GOV / P0 | Reusable bounded autonomy; current Human grant | COMPLETED_VERIFIED; CPU/stdlib, no data/GPU | Charter, one backlog, map/register, independent review, governance checks; avoids repeated supervision; risk: accidental permission expansion; no existing window owns it |
| AUTO-DATA / P1 | E1 split completeness; frozen E1 split + validate_split_rows | COMPLETED_VERIFIED; synthetic dates only, no GPU/new data | Reject unmatched row/split lengths rather than silently zip-truncate; preserve legal splits; protects preparation integrity; RED confirmed before implementation; git history has no prior fix |
| AUTO-RES / P2 | Prepare next decision from admitted diagnostics S7–S10 | COMPLETED_VERIFIED; git-pinned JSON/code only; no matrix/checkpoints loaded | Evidence-linked options, leakage/fairness/dependency checklist; explicit proposal vs approval; reduce Human research decision effort; no new run/results; existing diagnosis retained, not recomputed |
| AUTO-CKPT / P1 | E1 malformed pointer fallback; Training latest/previous/best contract | COMPLETED_VERIFIED; synthetic local CPU only | Reject malformed entries before body access, use valid previous after restart, preserve best and identity, SIGKILL/orphan regression; avoids losing recoverable progress; no historical loader certification; reviewer reproduced TypeError/KeyError, no prior fix in all-branch history |
| AUTO-SNAP / P1 | Immutable snapshot writer/reader consistency; existing v1 schema | COMPLETED_VERIFIED; offline RequestSpec fixtures only | Reject reader-invalid input before any raw/manifest publication; valid HTTP-error roundtrip and concurrency/restart intact; prevents immutable unusable artifacts; status0/600 reproduced, no prior all-branch fix |
| AUTO-PORT / P2 | Portfolio H1 inspector follow-on | DEFERRED_NO_CONFIRMED_GAP; existing fixtures available | Engine/export/API/UI already completed H0; avoid repeated UX/test work; real signals/metrics need H2; no new policy authorized |
| DATA-01 / P1 blocker | Original frozen matrix | BLOCKED; new exact location required | No new lead; do not repeat previous search, regenerate or deserialize |
| RANK-01 / METRIC-01 / RES-02 | End-to-end ranking/economic research | HUMAN_DECISION_REQUIRED; H2 and strict gates | High product/research value; no inferred policy, admission or Phase1–5 execution |
| OPS-02 / P3 | Provider certification B–E | PROPOSED; per-domain policy/architecture decisions | Existing Phase A complete; broad migration not implicitly approved; reliability fixes to existing contracts may be separately decomposed |
| TRAIN-01 | Next discriminating training | HUMAN_DECISION_REQUIRED; hypothesis and GPU grant | CPU preparation does not approve training |

Runtime: Python 3.12.3 via existing recovery venv, CUDA hidden; system python3 for
governance. MARKETMAMBA_DATA_ROOT unset; main Data/processed_v6 symlink targets
`/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; not read.
New checkout has no Data directory. Main started clean; no other tree modified.
Resume checkpoint: this first window is CLOSED. Verify `development/v7-autonomous-charter`
HEAD, ancestry from6d74d738 and clean status; read Charter and this table, then
[window evidence](autonomous-window-01-evidence.json). Do not restart completed
units or old E1 search. Another execution window needs a new bounded Human grant;
merge/push/deploy remain separately unauthorized. No READY item remains in this
window's confirmed selection; this is not an exhaustive no-bugs claim.

| Completed unit | Local commit / verification / review |
|---|---|
| AUTO-GOV | `be7998d`; governance23 PASS; independent authority review0 Critical/Required |
| AUTO-DATA | `3f07f80`; focused14, final E1 suite33 PASS; independent14 PASS; one repair round |
| AUTO-RES | `5847afc`;4/4 pinned source identities and claims independently checked; zero repairs; proposal NOT_APPROVED |
| AUTO-CKPT | `2c60415`; checkpoint/run/runtime19 PASS, independent19 PASS; two rounds, Required closed |
| AUTO-SNAP | Containing snapshot/window closure commit; snapshot19, final PIT49 PASS, independent19 PASS; one round;6 baseline-byte-compatible cases |

Final shared verification: E1 33 + PIT49 + governance23 =105 tests PASS, no skips.
Syntax/JSON/source and four pinned artifact hashes checked; diff check PASS;
knowledge health WARNING (governance/authority/legacy), zero structural FAIL.
Local CPU only; existing suite contains a one-step synthetic smoke, no new
research/model training. No formal datasets/weights loaded or network requests.

Priorities after this window: P1 DATA-01 only with a new original-matrix location;
P2 RANK-01/METRIC-01 and next-study decisions via the
[research preparation package](../../reports/v7_autonomous_research_preparation.md);
OPS-02 remains PROPOSED pending domain/architecture scope. No unexecuted candidate
has been mislabeled READY. GPU/new training requires hypothesis+compute approval.
Portfolio fixtures already meet H0; no repeated UX work. The two reviewer-found
reliability gaps were evaluated and completed in this window, not left for rediscovery.

Efficiency: no duplicate implementation, bulk data search or padding units. Initial
broad reads were truncated, and a reviewer test-module typo required a corrected
run; these were avoidable token/tool overhead. Same review agent resumed after
pending initialization; no extra parallel memory/MAS system. Available native
usage snapshot is in evidence and is unaudited, not a hard cost limit.

### Historical H0 checkpoint (superseded only as to next-window authorization)

Status: `PORTFOLIO_SLICE_LOCAL_DEVELOPMENT_INTEGRATED / NEXT_PROGRAM_NOT_AUTHORIZED`.

Latest H0 checkpoint: Human accepted the limited fixture product and approved exact
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; main fast-forwarded from83a6a2e.
CPU01/02 + PORT01–04 + UX01–04 + final closure/font fix are now in local main.
Pre/post CPU and actual Chrome verification passed; no Required/Critical remains.
[Human approval](../../knowledge/02_History/Decisions/20261021-portfolio-final-h0-approval.json),
[closure](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](portfolio-h0-integration-closure.json).
No further Portfolio UX work or new Program is authorized. STOP; future scoped CPU
work may use this main baseline. H2/StrictSTOP/E1BLOCKED remain unchanged.
The older planning/window statements below retain their observation-time context;
this checkpoint supersedes their H0-pending/UNMERGED status, not research authority.

Historical roadmap recovery source record:
Source baseline: `4c388ded57062194114ea3e221d00c177eb96892`; integration candidate
`integration/v7-cpu-roadmap`, UNMERGED / NOT_DEPLOYED; main `83a6a2e` unchanged.
Authority for this update: Human “Strategic Roadmap Recovery & Autonomous
Development Planning” request, authorizing this inventory/proposal, not future
large implementation. [Evidence report and source IDs S1–S15](../../reports/v7_strategic_roadmap_recovery.md)
and [candidate test/hash index](cpu-integration-candidate-20261021.json).

### Recovered objective and interpretation

Human-confirmed product intent (S1/S2): trustworthy full-market Taiwan-equity
research, daily relative rankings, weight-based simulated portfolio and traceable
performance. No personal account/capital/broker orders. Current machine baseline
(S5–S8) is 48 features, no explicit graph/industry neutralization, E5 64/32 and
formal seeds 17/29. Neither old three-seed/59-feature plans nor old optimal V6.2
portfolio parameters define current V7. Strict Phase 0 STOP and E1
portfolio_admission=false remain binding. Research portfolio metrics, risk limits,
benchmark/slippage and next experiments need separate decisions.

Single queue rule: edit work items here, not copies in task handoffs or MAS DB.
Current State projects active IDs and checkpoint only. Reports are dated evidence.
`implementation_state` describes existing code/evidence, `execution_state` describes
what can be attempted, `approval_state` names permission. READY requires a precise
contract, available inputs/runtime, independent acceptance and current authorization.
A proposal awaiting blanket scope approval is **not READY**. No GPU dependency does
not imply no Human/data dependency. H1 was explicitly approved by Human on 2026-10-21; only PORT-01–04 fixture engineering may execute. H0/H2 remain absent.

### Active UX refinement — Human approved 2026-10-21

[Human product direction](../../knowledge/02_History/Decisions/20261021-portfolio-ux-human-decision.json)
authorizes a concise default dashboard and advanced research mode. Sole active
queue remains this file; old PORT closure is historical completion, not a blocker
for this new explicitly authorized presentation window. Base `5a2bc6f`; isolated
`development/v7-portfolio-ux`; no GPU/data/research gate required for these fixtures.

| ID | State | Dependencies / acceptance | Value / risk |
|---|---|---|---|
| UX-01 | COMPLETED_VERIFIED | Same validated API object; accessible default/general vs advanced; preserve event on toggle | Simple overview with research detail; no parallel model |
| UX-02 | COMPLETED_VERIFIED | Presentation decimal formatting only; exact raw in advanced; zero vs unknown; immutable tests | Readability without accounting change |
| UX-03 | COMPLETED_VERIFIED | Dated ordered Chinese event timeline, chart selection sync, observable state changes only | Explain actual state, never infer causes |
| UX-04 | COMPLETED_VERIFIED | Real Chrome desktop/mobile all7 events/modes, exact values, errors, keyboard; regressions/review/evidence | Verifiable product review, not H0/H2 |

Completed as one coherent UX-01–03 page/helper commit `764dfbf` plus UX-04 acceptance
commit. Real Chrome desktop/mobile, both modes/all7events, exact values and errors
verified. Full lint remains identical baseline failure; scoped new code PASS.
No UX unit remains READY: stop for Human Product Slice Review, no H0/H2 inferred.
Implementation shares the same page/helper to avoid artificial subsystem splits.
Each logical unit has at most two evidence-supported repair rounds. Completion
records in [UX report](../../reports/v7_portfolio_ux_acceptance.md); no extension
beyond UX-04 without new scope. Original PORT evidence will not be rewritten.

### Final UX closure — Human authorized 2026-10-21

[Decision](../../knowledge/02_History/Decisions/20261021-portfolio-ux-closure-decision.json):
dual-mode direction accepted; only the following finishing scope was authorized.
Same branch `development/v7-portfolio-ux`, fixed base `1553a59`; containing closure
commit owns final identity. CPU/fixture only, no GPU or new Human research decision.

| ID | State | Dependencies / acceptance | Value / risk |
|---|---|---|---|
| UX-FINAL-01 | COMPLETED_VERIFIED | Existing UX1–4; all7 selections including initial7 visible inside timeline; chart/summary/state sync; pageY/focus stable; keyboard | Closes observed selection defect; one bounded scroll-anchoring repair |
| UX-FINAL-02 | COMPLETED_VERIFIED | Existing validated API states; deterministic exact-delta Chinese narratives; no unsupported causes or action types; 9 unit tests | Readable explanations without financial-policy change; no repair needed |

[Final report](../../reports/v7_portfolio_ux_closure.md) and
[evidence index](portfolio-ux-closure-evidence.json) retain baseline RED, repaired
browser result, screenshots, exact identity and independent review. No further UX
work is READY or authorized. Stop for Human closure review. H0 local main integration
requires explicit approval of the exact full candidate ancestry; H2 remains separate.
Mobile advanced visual optimization is expressly outside this program; basic access
is sufficient. Existing API/JSON is the preferred future AI research interface.

### Current font blocker closure — Human authorized 2026-10-21

The prior limited Product Slice UX Closure was Human accepted. Old H0 preflight at
6c894794 stopped on late-font clipping; audit7ecb920 remains immutable evidence, not
candidate ancestry. This explicit correction scope supersedes the older no-READY-UX
checkpoint solely for the one confirmed defect.

| ID | State | Acceptance / dependency |
|---|---|---|
| UX-FONT-VISIBILITY | COMPLETED_VERIFIED | True delayed-font RED→GREEN, desktop/mobile last selection fully visible; all7 selections, page/focus/precision/source unchanged; independent review |
| H0-NEW-CANDIDATE | COMPLETED_VERIFIED | Human-approved exact4a66928 ff-only from83a6a2e; pre/post regressions and Chrome PASS; LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED |

[Decision](../../knowledge/02_History/Decisions/20261021-portfolio-font-blocker-authorization.json),
[report](../../reports/v7_portfolio_font_visibility.md),
[evidence](portfolio-font-visibility-evidence.json).
No Required/Critical remains in tested scope. No further UX work READY/authorized.
This gate is now closed by explicit final4a66928 Human approval and local integration above. Stop; no next Program starts.
Strict Phase0 STOP/E1 BLOCKED/H2 unapproved unchanged. Never merge audit branch.

### Ranked backlog

Sizes are bounded engineering estimates, not delivery promises. Priority is value
and dependencies; completed component testing is not recycled to fill weeks.
All current implementation observations are pinned to `4c388de` plus the report's
branch-qualified evidence, and new integration verification is in the candidate
index. `COMPLETED_VERIFIED` means the named offline scope only.

| ID / priority / objective | Source; implementation / execution / approval | Dependencies; GPU/Colab; Human decision | Acceptance and value | Complexity / main risk |
|---|---|---|---|---|
| INT-01 / P0 / five-fix integration candidate | S3, window01/02; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / current request covers candidate only | 83a6a2e→4c388de chain; GPU no; merge decision separate | Five unchanged source/test hashes; 60 portfolio,47 PIT,81 ops,3 fetch,23 governance; old journal compatibility; independent GO. Consolidates trustworthy core. | S; accidental promotion/inherited evidence omitted |
| PLAN-01 / P0 / recover full objective and dependency map | S1–S15; IMPLEMENTED_VERIFIED planning artifact / COMPLETED_VERIFIED / current request | Canonical map/contracts/history; GPU no; no research decision made | Each formal objective traces source; proposal vs approval explicit; one queue, recoverable checkpoint, independent authority review. Prevents random bug work. | M; stale prose mistaken for authority |
| INT-02 / P1 / local main integration | INT-01/report §5; NOT_STARTED / HUMAN_DECISION_REQUIRED / exact candidate approval absent | Clean fixed main+candidate, reviewed entire ancestry; GPU no; Human merge approval | Exact approved SHA ff-only; post identity/tests; no push/deploy. Makes all five fixes available on development main. | S; main movement or partial cherry-pick |
| PORT-01 / P1 / deterministic offline journal history exporter | S1,S3 + H1 contract below; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | Five-fix source, H1 approved schema; GPU no; no research decision, engineering scope approval | One locked prefix; per-event states exactly reproduce engine; source digest; deterministic bytes; invalid input leaves output unchanged. Turns engine into inspectable workflow. | M, 2–3 focused days; read/hash race, accidental repair |
| PORT-02 / P1 / read-only artifact API | S1/S2 product intent, S12 patterns + H1 below; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01 schema; GPU no; H1 | Missing/corrupt/unsupported explicit, public field projection, no arbitrary path/query/write; existing routes unchanged. Exposes verified simulation artifact safely. | S–M, 1–2 days; stale fallback or private path leak |
| PORT-03 / P1 / offline portfolio history view | S1/S2 + H1 view contract; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01 frozen schema; mocks in parallel with PORT-02, real integration after both; GPU no; H1 | NAV/cash/positions/pending/cost/event timeline, provenance, explicit fixture label; API agreement, browser desktop/mobile, missing/error/zero. Completes useful product slice. | M, 2–3 days; simulation mistaken for live/research result |
| PORT-04 / P1 / workflow acceptance and restart handoff | S3 + H1 fixture program; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01/02/03; GPU no; H1 | Fixed scenarios include no-trade, partial/blocked fills, action, retry, corrupt/truncated rejection; fresh process replay→export→API matches; build/browser; independent review/checkpoint. Closes end-to-end engineering evidence. | M, 1–2 days; synthetic success mistaken for market validation |
| DATA-01 / P1 / original E1 matrix recovery | S8/S9/S10; PARTIAL / BLOCKED / bounded new-location inspection may be requested | Exact original a590b280…947f62 matrix+manifest location; GPU no; Human supplies location/access as needed | Per-file size/hash, aggregate identity, contract/split match; never regenerate replacement. Unlocks original replay. Previous unchanged search not repeated. | Unknown I/O; unique evidence unavailable |
| RES-01 / P2 / safe retained checkpoint replay | S7/S8/S9/S10; PARTIAL / BLOCKED / exact replay scope/runtime must be approved | DATA-01, safe deserialization validation, retained checkpoint identities; no new training; CPU feasibility UNKNOWN | Validate embedded contract/matrix/seed, paired support, existing BEST rule; preserve output namespace. Recover final-support evidence without altering historical result. | M–L; unsafe loading, missing epochs2–4, hardware feasibility |
| DATA-02 / P1 research gate / strict executable evidence | S5/S6; PARTIAL / BLOCKED / source admission and any acquisition permission separate | Direct dated tradability/open/PIT evidence; GPU no; Human source/policy questions if unresolved | Existing zero-tolerance/no-roll/unknown rules unchanged; qualified evidence inventory, no simulated→verified relabel. Unlocks strict capability only when evidence qualifies. | L/unknown; historical coverage/license/provenance |
| RANK-01 / P2 / corrected prediction-to-ranking adapter | S1/S2/S7/S8, existing stability contract; PARTIAL utilities / HUMAN_DECISION_REQUIRED / H2 missing | Precise corrected two-seed schema/admission; existing predictions/identities; GPU no for artifact display | Reuse validator/ranks where semantics match; freeze mismatch/maturity/missing/tie/support behavior before code; never use historical three-seed output as corrected evidence. Delivers actual V7 ranking path. | M; hidden contract migration and portfolio admission |
| METRIC-01 / P2 / economic and risk report | S1/S2/S4; PARTIAL (NAV,cost,RankIC utilities) / HUMAN_DECISION_REQUIRED / H2 + strict gate | Benchmark/turnover/drawdown-calendar/annualization/slippage/exposure definitions and admissible data; GPU no | Frozen definitions, hand-calculated cases, maturity/support and matched comparison; research run only after its gates. Needed for credible portfolio conclusions. | M–L; unverifiable defaults/selection bias |
| RES-02 / P2 / signal anatomy, portfolio-response, failure-slicing | S4/S6/S8; NOT_STARTED under admitted current scope / BLOCKED / research gate absent | Strict correctness PASS + explicit admission/frozen comparisons, METRIC-01; generally existing artifacts, no new training required | Exact approved Phase1–5 acceptance, equal support/costs, evidence classes; no renamed bypass. Core research validation. | L; false readiness/overfit grid |
| OPS-02 / P3 / provider certification and resumable publication B–E | S11; PARTIAL primitives / PROPOSED / separate production scope required | One domain policy, provider licensing/authority, stage contracts; GPU not for framework, inference later artifact-dependent | Per-domain identity/fallback/certification; kill/restart/pointer/rollback per approved contracts. Complete durable operation incrementally, not broad rebuild. | L; production authority/large subsystem |
| PAPER-01 / later / daily forward simulation | S1/S11 Phase F; NOT_STARTED / BLOCKED / explicit research+paper approval | Qualified inputs, model/strategy/risk freeze, publication and calendar time; existing model inference may need compute | Save signals before maturity; replayable simulation; no stale-as-new; measured forward window. Not broker trading. | L; code completion mistaken for forward evidence |
| TRAIN-01 / later / next discriminating experiment | S7/S9/S13; NOT_STARTED / HUMAN_DECISION_REQUIRED / H2 + compute approval | Resolve diagnostic gaps and predeclare question; GPU/Colab yes for new training | One frozen hypothesis, splits/seeds/selection/cost/stop rule; durable evidence. Human retains promotion. | L/compute; unsupported new direction |
| MAS-01 / separate maintenance / Core Live certification | S12 + historical MAS handoff; PARTIAL / BLOCKED / outside V7 implementation grant | MAS owner exact-version certification, valid occurrence approvals/routing; GPU no | Actual Builder+Verifier lifecycle and restart evidence, not a Codex takeover. Never make this a prerequisite for independent Codex CPU work. | Separate project; authority/state corruption |

**H1 execution checkpoint — 2026-10-21:** PORT-01 completed and independently
reviewed (7 focused / 67 combined tests); PORT-02 completed (8 API tests, independent live review); PORT-03 completed (real browser + unit/build); PORT-04 COMPLETED_VERIFIED
(actual HTTP/restart and browser missing/corrupt/restore; 47 PIT,81 ops,3 fetch,23 governance).
All four authorized units are complete; stop for Human Product Slice Review.
No remaining READY work is authorized by H1. Further CPU opportunities are
H0 integration or separately bounded fixture extensions/source recovery; these
need their existing gates, not new training by default. See [unit evidence](portfolio-history-evidence.json).
Branch `development/v7-portfolio-history` starts at `bebf19ee937a2237641e09701d0564590ed30fc6`;
UNMERGED / NOT_DEPLOYED. No other backlog permission is implied.

### H1 approved engineering contract

[Explicit Human H1 decision](../../knowledge/02_History/Decisions/20261021-portfolio-history-h1.json)
approves this local fixture-only offline history slice. No real predictions,
market data, strategy selection, new economic metrics or background jobs.
This exposes the existing deterministic engine; it is not S4 Phase1–5 research.
Engineering binding: source additionally carries `fixture_id` and
`fixture_manifest_sha256`; operator declaration `v7-portfolio-fixture-v1` binds
exact journal bytes. This declaration is not an authenticity certificate or
admission. API requires an operator-pinned artifact SHA-256 and never accepts
client paths. Output is confined to checkout `.artifacts` or OS temporary paths.

1. **Input:** explicit local synthetic journal generated from existing S3 test
   fixtures; operator-selected source path, never a browser-supplied path. Fixtures
   always labeled `OFFLINE_SYNTHETIC_ENGINEERING`. Existing historical journals
   remain read-only compatibility inputs in tests, not real research admission.
2. **Output (proposed `v7-portfolio-history-v1`):** `schema`, `evidence_class`,
   `strict_phase0: STOP`, `research_admission: false`, `source` containing full
   source commit, engine version, original byte SHA-256, final record_hash and
   event_count; `spec`; ordered `events` with seq/event_id/kind/occurred_at and
   exact post-event `state`; `final_state`. State copies existing `snapshot()`
   keys/Decimal text, including pending null versus populated and true zeros.
   No new return/risk/ranking fields, raw private filesystem paths or wall-clock
   generated_at in deterministic payload. Actual run time belongs in evidence receipt.
3. **Consistency:** obtain bytes/hash/validated records from one shared-lock
   prefix, then reduce exactly that prefix in order, not a second unlocked read.
   Preserve the existing journal API. No journal migration/repair/writes during
   export. Export every event including genesis; final_state equals replay of that
   prefix. No repeated full-prefix replay that makes export quadratic by design.
4. **Artifact handling:** explicit distinct output path; no alias of source
   (resolve/samefile checks), models or production. Validate complete payload
   before atomic local temp/fsync/replace; failure leaves existing output bytes
   unchanged, identical replay creates identical content. No shared default output.
   This certifies local cooperative filesystem scope, not arbitrary host power loss.
5. **API:** additive local-only GET `/api/v7/portfolio-history`, separate router
   `app/backend/routers/v7_portfolio.py`, configured artifact path via a dedicated
   opt-in setting; no writes/auth changes/external network. Missing/disabled returns
   HTTP 200 `not_ready` with no states; malformed, unsupported or inconsistent
   payload returns 503 `error`; valid returns 200 `ready` plus projected public
   contract. `ready` means artifact readable, never tradable. No stale fallback.
   Validate event count/order/final equality and required types/evidence labels;
   digest is integrity metadata, not authenticity of arbitrary market facts.
   Consumer validation checks schema/internal consistency of a trusted local
   exporter artifact; it does not independently prove the accounting of an
   arbitrary edited timeline. Exact accounting is verified at exporter tests.
6. **View:** separate opt-in local route `/v7/portfolio-history` using existing
   app components. Show event time, NAV/cash/quantity/last-price/cost/pending and
   source identity; fixture and research-not-admitted banner always visible.
   No holdings weights until a separate display formula/schema is approved;
   no implied target allocation from pending when unknown. Avoid new policy
   calculations; empty/unknown displayed explicitly, true zero retained.
7. **Tests:** unittest temp fixtures for deterministic complete replay and failures;
   corrupt/truncated/hash mismatch/duplicates fail closed; valid old writer format
   retained; one real-process append/read boundary test proves snapshot coherence.
   API uses existing httpx ASGI tests; UI uses Node contract tests plus production
   build and real browser desktop/mobile/keyboard checks. No new paid dependencies.
8. **Files:** reuse `V6/experimental/v7_portfolio_{contract,engine,journal}.py`;
   proposed exporter/test siblings `v7_portfolio_history.py`, `..._test.py`;
   additive router/test and frontend page/contract/test in existing app trees.
   Each milestone narrows exact paths before edits. A small journal read helper
   extraction may be needed for lock-consistent export, with regression tests;
   no broad refactor. Prefix public state keys with no invented strategy semantics.

Initial program command sketch (see PORT-04 acceptance report for exact runnable commands):
`python3 -B -m unittest V6.experimental.v7_portfolio_history_test app.backend.test_v7_portfolio`;
`python3 -B -m V6.experimental.v7_portfolio_history --journal <fixture> --output <isolated-artifact>`.
Existing portfolio/PIT/governance commands remain in candidate index. Frontend:
`node --test <new-contract-test.mjs>` and `npm --prefix app/frontend run build`;
use existing pinned runtime, inspect browser with available tools before claiming
UI verified. If tools/runtime unavailable, mark that milestone BLOCKED, never waive.
Python style follows existing type hints, Decimal text, `JournalIntegrityError`
and unittest conventions; don't duplicate accounting formulas in API or JS.

### Original H1 sequencing plan (all four units now complete)

| Window estimate | Milestone | Gate / parallelism / stopping point |
|---|---|---|
| Days 1–3 | PORT-01 exporter/schema and hand-fixture timeline | H1 first. Freeze engineering schema as approved v1 before implementation; exact replay/hash/atomicity tests and independent review. |
| Days 4–5 | PORT-02 artifact API; PORT-03 UI contract/mock concurrently | Only after schema stable; distinct owners/files, no shared writer. API and mock UI can proceed in parallel. No external deployment. |
| Days 6–8 | PORT-03 real API view and PORT-04 end-to-end/restart | Needs both outputs; browser/build/fixture checks, immutable evidence and one acceptance report. |
| Days 9–10 if needed | Evidence-backed bounded rework and closure | Max two fix rounds per unit; not extra features. Finish early when accepted, stop if no authorized READY work. |

H0 (optional): approve exact local main integration candidate; this is independent
of H1 and not necessary if development continues on verified candidate ancestry.
H1 (APPROVED 2026-10-21): Human approved PORT-01–04 and the engineering contract
above as a single fixture-only program. This is engineering scope approval, not
research correctness. Routine choices within it need no repeated micro-approval.
H2 (defer): real prediction admission, economic/risk definitions and next research
question. Do not demand these just to start H1. Supplying an original E1 matrix
location is a separate data action; it does not authorize replay/training by itself.

### Codex + MAS sustainable work protocol

- At session start read Current State → Authority Map → this queue → selected
  spec/manifest/tests/domain. Verify repository/worktree/head/dirty state, runtime,
  data realpath and goal objective. Never resume from a title or old MAS task id.
- Codex owns implementation in the selected product worktree, executable tests,
  independent review, coherent local commits and memory. One writer per path.
- Native Goal maintains the thread's bounded outcome/continue/check/complete state;
  it does not replace repository memory, authorize new scope or certify research.
  Verify get_goal each session; if unrelated, stop that continuation using normal
  supported lifecycle controls, never internal DB/global edits. Fresh thread restores
  queue/checkpoint first and creates a matching Goal only under user authorization.
- MAS Advisory can propose ordering and challenge dependency/contract gaps;
  Verifier reviews evidence; Archivist checks identities/records. These are roles
  under existing research-program boundaries, not proof an available MAS service
  has been executed. Core Live remains unproven; use ordinary independent Codex
  review now. Historical occurrence approvals never transfer to new tasks.
- Select highest-value authorized READY item; record IN_PROGRESS, source commit,
  exact allowlist, acceptance, input identities and budget/stop envelope. If one
  item blocks, preserve reason and select independent READY work; no polling for
  Colab or repeated matrix search without new evidence.
- Implement/test/review ≤2 evidence-supported fix rounds per unit. On unresolved
  findings mark BLOCKED with exact reproduction; don't relax gate. Defects are
  handled when they block selected functional acceptance or have high-impact
  evidence, not because unlimited easy tests can be written.
- Checkpoint includes commit and source/input/output hashes, commands/counts,
  reviewer findings, rounds, known limits, elapsed observation, telemetry class,
  Human interventions and next ID. Goal token/time counters are tool observations,
  not independently verified cross-agent aggregate or a hard cap unless enforced.
- Update this queue and canonical projections in the same commit; normal tests →
  impact checker → knowledge update/explicit no-change reason → health → clean.
  Completed units are not rerun except new changes, runtime drift or integration
  need. At no READY work, stop with completed/blocked/decision summary.
- Once approved, H1 can preauthorize CPU fixture coding/tests, local commits,
  independent review and bounded repairs. It cannot preauthorize label/PIT changes,
  hypothesis selection, result promotion, GPU/paid services, production changes,
  main merge/push/deploy or MAS Core authority. Those remain explicit Human gates.

Official Goal design reference, retrieved 2026-10-21:
[OpenAI — Using Goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).
It describes thread-scoped persistent objectives and explicit stopping conditions;
our live API confirms this thread's new V7 objective, not weeks-long unattended
reliability or cross-thread/global project memory. No scheduler is enabled here.

### H1 Goal (now authorized; stop after PORT-04)

> Complete PORT-01–04 in research/v7/research-program.md: a deterministic,
> fixture-only V7 journal history exporter, additive read-only API and local
> portfolio history view, verified by exact replay/identity/atomicity tests,
> old-format compatibility, browser/build checks and independent review. Preserve
> Strict Phase 0 STOP, no real predictions/market data/research admission, no GPU,
> production changes, merge/push/deploy or MAS Core execution. Select only the next
> dependency-ready approved item; at most two evidence-backed repair rounds each;
> checkpoint/update the single backlog and local commit after every milestone.
> Stop once all four are verified or no authorized READY item remains.

## Historical research-program contract (preserved)

The following is the original umbrella research/role record, retained for
provenance. Its “current environment”, “tomorrow” and unexecuted experiment language
are historical observations, not present runtime facts or permission to activate
MAS/train. The current dated planning section above owns work ordering only;
frozen experiment contracts still own research semantics.

# V7 MAS Research Program Contract

## Objective

Determine which literature-derived hypotheses materially improve MarketMamba over the frozen current baseline under realistic Taiwan-equity cross-sectional forecasting constraints, using the smallest dependency-aware sequence of experiments that preserves causal attribution.

## Current Environment Facts

- WSL canonical repo exists and is the repository for V7 preparation.
- Data stays on `D:` through `MARKETMAMBA_DATA_ROOT`.
- Representative parquet/data checks passed during migration.
- Lightweight tests passed during migration.
- GPU model runtime is not fully ready.
- `torch-scatter`, `torch-sparse`, CUDA-specific `mamba_ssm`, and `causal_conv1d` still need later resolution.
- This runtime dependency state must not be mistaken for a model or research failure.

## Non-Negotiable Rules

- Freeze baseline before architecture changes.
- One causal question at a time.
- Cheapest discriminating experiment first.
- Negative results are first-class results.
- No random time-series train/test split.
- Record predictive, economic, robustness, and systems/compute metrics.
- Papers are priors, not local evidence.
- Human retains final architecture authority.

## Required Metrics

Record these where applicable:

- IC
- RankIC
- ICIR
- RankICIR
- Top-K / Bottom-K
- existing portfolio metrics
- transaction-cost-adjusted metrics
- turnover
- drawdown if available
- parameter count
- training time
- peak VRAM
- inference latency/throughput
- seed/regime stability

Experiment verdict must be exactly one of:

- `SUPPORTED`
- `INCONCLUSIVE`
- `REJECTED`
- `BLOCKED_BY_COST`

## MAS Role Boundaries

### Advisor

May audit, prioritize, design experiment families, and select the next high-information experiment.

May not declare final V7 architecture.

### Builder

May implement approved scoped experiments.

Must preserve provenance, baseline comparability, and data integrity.

### Verifier

Independently checks leakage, experiment isolation, metric correctness, baseline fairness, claimed gain, and compute cost.

### Archivist

Preserves evidence, rejected/inconclusive results, rationale, and references.

### Human

Approves baseline freeze, compute escalation, major scope changes, architecture promotion, and final V7 freeze.

## First MAS Activation

The first MAS activation tomorrow must be planning/audit only:

Audit the WSL MarketMamba repository, GPU/runtime readiness, current V6.x baseline, literature corpus, and hypothesis backlog. Produce a minimal dependency-aware experiment plan and identify the exact first experiment. Do not begin expensive training until Human approves it.

Expected first MAS output:

- baseline/project audit
- GPU/runtime dependency assessment
- hypothesis dependency graph
- proposed first 3-5 experiments
- compute/cost/risk estimate
- missing evidence/questions
- Human approval request

### AUTO-DATA completion checkpoint — 2026-10-21

Authority: frozen E1 split and existing per-row validation, Charter engineering.
Before implementation: READY after RED confirmed silent omitted tails. One repair
round (`zip(strict=True)`), no new split/label/PIT rule. RED: 6 tests, four failing
subcases; GREEN focused14, E1 regression29 (includes existing one-step synthetic
CPU smoke, not new research training). Independent reviewer reran14 PASS; zero
Critical/Required. [Evidence](autonomous-window-01-evidence.json) retains raw output.
Source/test identity: containing AUTO-DATA commit. Data note and both maps updated;
frozen contracts/historical artifacts untouched. Remaining: original matrix BLOCKED.

### AUTO-RES completion checkpoint — 2026-10-21

[Decision preparation](../../reports/v7_autonomous_research_preparation.md) completed:
4 pinned sources/3 JSON documents checked against recorded SHA256 and values;
independent review confirms4/4 identity and no admission/approval inflation.
Zero repair rounds. Proposal stays NOT_APPROVED. This is preparation, not new
research analysis execution. Human still owns next hypothesis/design/compute.
Identity: containing research-preparation commit; no external data/GPU needed.

### AUTO-CKPT completion checkpoint — 2026-10-21

Two repair rounds: entry rejection/fallback, then independent-review Required fix
for valid previous retention through save/orphan recovery. Final19 CPU tests PASS,
independent15+4 PASS; Critical/Required0. Real spawned SIGKILL before pointer,
restart/orphan, repeated latest loss, seed/contract/matrix mismatch covered.
No historical checkpoint loaded. Existing unsafe-for-untrusted-pickle loader is
not certified. [Evidence](autonomous-window-01-evidence.json); identity: containing
checkpoint commit. Training note/maps updated. AUTO-SNAP subsequently completed below.

### AUTO-SNAP completion checkpoint — 2026-10-21

Writer validates serialized payload with existing reader rules before raw or
manifest publication. RED19 failures7/errors6; GREEN19, full PIT49; independent19
PASS, no Critical/Required. One repair round; valid six-case payload bytes match
base6d74d738. Frozen data/PIT authority unchanged. All five units complete; STOP.
Identity: containing snapshot/window closure commit, preserved in branch history.

### W02-DATA checkpoint

Completed7-case synthetic provider diagnosis, independent7-case rerun/source/probe
hash PASS,0 Required/Critical,0 repair rounds. Existing fetcher unchanged. Source,
numerical observations and limitations in [window evidence](cross-domain-window02-evidence.json)
and [report](../../reports/v7_cross_domain_window02.md). Identity: containing data
unit commit. Production date/schema corrections and monthly completeness remain
Human decisions; no network/official-data inference. Continue W02-PREP/EVAL/MODEL.

### W02-PREP checkpoint

Containing commit pins the already approved contract digest and validates matrix
metadata field identities/types, including self-rehashed drift. RED13 tests with25
failures → GREEN34 non-training tests; independent14 PASS,0 Required;1 repair round.
No actual arrays inspected or rebuilt; frozen matrix remains BLOCKED. Contract and
ACCEPTANCE unchanged. Recovery evaluator byte/provenance checks at de2b62e are
separate branch-local work, not duplicated or silently adopted here.
Continue W02-EVAL then W02-MODEL; see window02 evidence/report.

### W02-EVAL checkpoint

Containing commit rejects incomplete/noncanonical seed sets and missing/nonfinite/
boolean aggregate acceptance values. RED6 tests:29 failures/13 errors; GREEN6 PASS;
independent6 PASS,0 Required;1 repair round. Thresholds/bootstrap/outcomes unchanged.
No historical result recalculation; recovery branch de2b62e has the same original
`decide` gap but separate stronger support/provenance implementation. Continue only
W02-MODEL and final bounded-window validation; do not reopen original matrix search.

### Window02 final checkpoint — STOP

Four units completed: DATA371d55d (0 repair rounds), PREPe17fcf0 (1),
EVAL164dbc1 (1), MODEL in containing closure commit (1 documentation-only closure repair). Independent reviews:
0 Critical/Required each. Final259 tests PASS: PIT49/E1 non-training36/
Portfolio67/Operations81/fetch+governance26. Final model review verified11/11
source identities and claim trace. No training, matrix search, production edit,
network access, or research approval. New proposals stay PROPOSED.

No additional demonstrated high-value READY work remains from this cross-domain
scan under the current boundaries. P1 W02-DATA-FIX and research comparison design
need Human scope/specification; DATA-01 stays BLOCKED without original matrix;
P2 MOPS policy/new ablation/walk-forward require Human decisions. Do not invent
work to fill a fifth slot. Next execution window needs a new bounded Human grant.

Minimal recovery: cd `.worktrees/v7-cross-domain-window-02` from WSL root; verify
`git status --short --branch`, HEAD and worktrees; read Current_State, Authority_Map,
this dated queue and cross-domain-window02-evidence.json. Expected branch is
`development/v7-cross-domain-window-02`, basec1179fe, four completed units,
STOP/UNMERGED/NOT_DEPLOYED. Read model report only for selected follow-up design.
Do not rerun historical discovery or E1 matrix search. Main navigation will pin
this closure commit; no window02 implementation merge is authorized here.
