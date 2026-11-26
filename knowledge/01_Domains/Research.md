## 第三輪 forward F1–F4 有界修補 — 2026-11-26

`builder_fwd_fix_r3@5b9f352`，no-commit、SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。[WP2報告](../../reports/forward_wp2_20261126.md)新增14日×三seed×五擾動的210組oracle，future feature／label missingness雙向切換且新missingness mutant KILLED；這修補驗收盲點，不證明真實PIT或現行計算曾洩漏。[WP5報告](../../reports/forward_wp5_20261126.md)記錄事故journal而非manifest摘要控制開盲、兩fresh程序receipt與store／C0+配對binding、排除calendar metadata的唯一開盲identity。完整forward428 passed／172 NaN warnings／386.13秒，測試等待放寬不改產品timeout。

同權限commitment仍非OS隔離／簽章；P2規格、selection與研究門檻不變。原seed17真實bitwise／WP6／calendar／lifecycle／lagus STOP未解，整包NO_GO待獨立複審與Human acceptance。證據僅本worktree`.artifacts/forward/fix-r3-20261126-5b9f352/`，沒有真資料fits、blind讀取或外部runtime observation更新。

## 前瞻 F1／F4 審查修補候選 — 2026-11-26

`job-run-mwxcypdo-30712f/builder_fwd_fix12`，既有detached base `d8d76c7cd4f1c8533c3c7c16fc661efb03d76c88`，Human核准有界no-commit，**CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。回應主checkout私有 `fwd-wp1-4-review.md` 的F1／F4；[WP2測試](../../tests/test_forward_wp2.py)改在origin與+1／+3／+8 sessions逐次改未來labels／features，另按+5／+10收益端點改未成熟labels；只改列日期>t不足以抓直接使用當列eval label。三seed共36 cases核全部≤t scores／ordered keys及實際LightGBM fit target／features與歷史predict features雜湊。審查eval-label及未來feature source mutants在四個時點共8／8被攔，正式oracle對兩mutants各exit1，原實作GREEN；未發現現行計算洩漏，未改fit/predict邏輯。[WP1測試](../../tests/test_forward_wp1.py)鎖定helper與WP2 train≤origin−21、origin−20排除；兩模組僅註解統一。

標準PYTHONPATH=V6:V6/research_lab/tests:.、無--basetemp的focused61 passed／12.81秒，rolling split／runner29 passed／4.29秒；本輪未跑全套或真資料fit。來源PIT／日曆／lifecycle、WP2 seed17 bitwise待Human、WP3同版本mask／reference與WP4環境還原lock仍STOP，**WP1–WP4整包NO_GO**不解除；修補待獨立複審與Human acceptance。詳[WP1回應](../../reports/forward_wp1_20261126.md#f1f4-獨立審查回應--2026-11-26)與[WP2回應](../../reports/forward_wp2_20261126.md#f1f4-獨立審查回應--2026-11-26)。私有GENERATED evidence只在本worktree`.artifacts/forward/fix12-20261126/`，未改規格／acceptance門檻／machine或production authority；無外部observed_at更新，無網路／GPU／agents／正式Data／V6/models／P4d／第三步／排程／blind操作，無commit／push／merge／deploy。

## P2 WP2 任意 origin／重播邊界 — 2026-11-26

[薄 adapter](../../tools/forward_wp2.py)只使用 C0+ fit/predict，不執行舊 `c0plus.run` 的 day-t fit 或 eval IC／收益診斷；完整train截面rank後三seed抽樣，W60只限推論，不用y10／eval label篩選。origin−21含端點承接P1／P2權威；WP1原排他註解已由本次F4修正，跨WP1／WP2邊界測試鎖定含端點，修補仍待複審。全量／抽樣row索引及date/code keys保存於私有forward namespace且可重播，損壞或seed不符STOP。舊fold_2026 bitwise出現差異即維持STOP，即使ranks一致也不放寬。合成future-label／feature mutation不變性不等於真實PIT證明；完整結果、環境／hash及限制見[報告](../../reports/forward_wp2_20261126.md)。CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；不是WP4持久化／交易／前瞻准入。

## Step3 延後評估入口 — 2026-11-26

[新驅動程式](../../tools/run_step3_evaluation.py) 為 no-commit、UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED 候選；僅合成工程測試，不讀實際 fits 或執行真資料評估。[公開模板](../../reports/step3_results_20261126.md) 結果仍 pending。72 COMPLETE／bundle／Human 決定／request／鍵／標籤／SHA 全驗後才准評估，沿用既有統計模組，保留完整日曆缺日；已曝光歷史只報兩段量級與區間，不宣稱正式成功。詳 project map 新節；來源 gate 的實際 evidence、獨立審查與後續執行由 Claude 核對，strict 前瞻 STOP 不變。

## P4d 進度與本機量測證據邊界 — 2026-11-25

`job-run-mwvqcr5w-e515cb/builder_p4d_fix1@d36e709` 的未提交 bounded patch，**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**。[工程報告](../../reports/v7_p4d_engineering.md#排練-heartbeat-required-與最大-fit-量測入口--2026-11-25)記錄排練前啟動共用Progress、REHEARSAL／FORMAL模式與stage、60秒heartbeat替身、寫入失敗繼續及2026全列C0+資源量測入口；不改split／features／selection／queue／threshold或P4a來源policy。僅36列合成特徵＋擬合，未實際量測2026全列RSS，不恢復抽樣cap；CPU回歸不是現場Drive／T4排練證據。原獨立NO_GO仍待複審，完成不等於Human acceptance；沒有研究結果／canonical artifact supersession／production authority變更，不更新authority-map.json或外部observed_at。Next：獨立審查與另案本機最大fit量測。

## P4d stage1b 接續工程 — 2026-11-25

沿用WIP `b1871a9`的tools獨立bundle邊界；初始完整CPU回歸1265 passed／4 skipped／0 failed，未改P4a來源policy。排練產物持久化、marker重驗與空證據STOP為工程候選，不改第一段split／recipe／selection／threshold。prepared完整anchor及2021outer support已核對，65股縮小支援六fit／五項mutation／notebook compile與解包驗證僅CPU工程證據，非完整研究或現場排練。[工程報告](../../reports/v7_p4d_engineering.md#第一段接續-stage1b--2026-11-25)列最終回歸與identity。CUDA／全列C0+RAM／Drive／長排練progress／獨立review仍未驗；PIT／universe／tradability仍UNKNOWN／STOP，未新增績效結論。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；無GPU／Colab／Drive／研究fit或外部runtime observation變更。

## P4b 對話凍結／CPU 工程候選 — 2026-11-24

Human核准 [P4b](../../tasks/spec-p4b-blend-20261124.md) 採原SHA `8d4f194730c2a147b42afe69750f722fccab1d0111e721d2a1f279df369f53c2` 預設值凍結（對話、非數位簽章）；前瞻／P4d仍DRAFT。指派detached bc549b8／builder_p4b的未提交adapter與tests為CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，詳[工程報告](../../reports/v7_p4b_engineering.md)。獨立36-fit准入沿用resume_fit、原64-fit gate不變；原始seed共同S後重rank、固定B1/B2/B3、95%／50、D及日期聯集、NaN與Holm3/centered雙邊契約不變。

實讀本機36-fit與P1三seed、ledger769 generations，2021共同378,547列／244 dates，976個gate全部通過、最小保留率1.0。只混合計算，未執行真實portfolio／significance／nomination、不重訓；先前29 focused、9 mutation證據詳報告。C0+調參三seed vs 神經固定兩seed不對稱，已曝光、僅描述性，raw/PIT/universe/下市限制未解除。下一步為獨立複審及事前凍結price/market_value/volume snapshot，Human另行跑全量與判讀；現場全六折評估及acceptance仍未驗證。無外部runtime observation或production authority變化。

最終補錄：adapter `233f08fb7462bf3c5375772b34fa0f7f20bc85a4c44edb99a93f1424bdd5c399`；工程identity `21b33e7a4a7bca49ca375e07bba8f570f7f71d822dfbf819e4956a85c5c931d3`；最終2021排練manifest `0dcd0477c5318edfb764bf646585d08fd132d5f3616231393e045cc54ace35e4`（`.artifacts/p4b/rehearsal-2021-verified/`），8份分數hash與前次相同。focused **30 passed／9 mutants killed**；全套 **1153 passed／28 failed／6 skipped**（384.86s），28項均P4a source-compat gate拒絕未獨立審查的新增P4b模組，已單測重現。**整合驗收STOP**；不放寬gate或偽稱全套通過。需獨立審查數值中立／來源打包範圍，及price-volume snapshot／Human另行全量評估；未變production／64-fit authority。

## 三份DRAFT coverage／凍結索引修補 — 2026-11-24

`job-run-mwuu88xi-da927f/spec_fixer2@abc12f5`的未提交文件候選，DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。[前瞻P2](../../tasks/spec-forward-p2-primary-20261124.md)公開觀測比例以全宇宙含停牌為分母；必要資料gate逐欄適用集合100%，非停牌必要OHLCV／features不得豁免，合法停牌估值／lifecycle證據對持有與未持有都須完整，不冒充當日OHLCV。成對S/M合成fixture定義公開來源投影／分類／節奏不受持股影響；runner／推導與時序驗收尚未實作。入口名稱核對為`portfolio.run_backtest`；P4d B表格只描述訓練隨機性與集成。P4a空標題為原正文錯置，已移回Current State／Authority Map原標題，未重標結果或新造驗收狀態。

沿用Current State承載[跨案Human凍結彙整表](../00_Project_Map/Current_State.md#三份draft-human凍結決定彙整)，三份規格決定清單均指向該表，完整語義以DRAFT正文為準。尚待獨立複審／Human凍結及另案工程、資源、來源／日曆驗收；不改machine authority／production／runtime observed_at，不接觸Data、blind、GPU、Colab或Drive。

## 三份DRAFT審查修訂候選 — 2026-11-24

`job-run-mwutomlv-e3974c/spec_fixer`，既有detached HEAD/base `3ed6876`，起始clean、6份Markdown未提交patch；**DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[前瞻P2](../../tasks/spec-forward-p2-primary-20261124.md)補F1完整252收益契約：建議T1純現金→次日open首建含費、s1…s252 close估值、無終止清倉費、停牌／未知下市fail closed及同基準時鐘；與舊engine末entry open清倉不同，WP6只驗已選語義。F2封閉公開來源／全宇宙摘要schema、分母、持股診斷禁公開、私有256-bit nonce承諾、固定錯誤處置及與公開市場資訊結合推導驗收。[P4b](../../tasks/spec-p4b-blend-20261124.md)補B1事前日期域與聯集核對、整日／整折消失不得被交集隱藏。[P4d](../../tasks/spec-p4d-equal-tuning-20261125.md)補D1明確36…47欄macro schema／fit support cache與未來特徵mutation；D2雙邊centered公式＋Holm2；D3改稱訓練隨機性與集成的描述性反應，anchor選更新數有選擇偏向。

唯讀CPU實證：prepared及pilot09六折共12組，X實算SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`；整矩陣60日、train60日、2021前60日、train逐日及2021前逐日的mask全同36…47。P1／P3六折X／split identity均吻合，P4a限本機prepared；**已完成結果不受此問題影響（僅限該mask及所列identity）**，不升格P3完整性／P4a逐fit驗收或PIT。輸入hash、方法與逐折數值在P4d附註，不依賴臨時輸出。

Next：獨立再審後Human選定端點A（收盤估值；替代開盤／收盤清倉）、摘要／nonce／處置表及macro schema（替代僅fit估計後凍結），並決定其餘既有DRAFT選項；文件修訂不解除原NO_GO或代替Human凍結。正式calendar／來源／事件、實作、future-mutation／推導測試及現場驗收仍blockers。只文件與指定唯讀實證，無正式Data／blind／models／GPU／Colab／Drive／production／scheduler／排程／額外agents／commit／push／merge；不更新外部runtime observed_at，不改machine authority或歷史結果。遵照no-commit保留6份unstaged，非clean-worktree完成宣告。

## P4d 對等調參／資料、容量、種子曲線候選 — 2026-11-24

[P4d規格](../../tasks/spec-p4d-equal-tuning-20261125.md)仍DRAFT／E2 RESEARCH_ONLY／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED、全部已曝光。核心N12三臂對等調參不變；新增Human同意的A六格容量（12格替代）、B固定seed前綴1/2/4/8排名平均，D為可選25/50/100%最近完整交易年等值。三模組統一固定P4a配方及2021前baseline BEST更新長度，主看驗證IC5、train–val gap診斷，outer IC及組合層僅描述；固定headdim8／state16／expand2，不因新結果換設計。成本及同預算「更多seed vs 更大單模型」、27種三軸方向解讀與階段資源決策見規格。

Human `p4a-results (3).zip` 的ledger／metrics派生摘要已取得，本輪核主checkout摘要64 fits；W60 GRU均值9.05／max10.5、Mamba2均值27.5／max29.0min，原始hash-chain／逐fit環境未重驗。P4a逐折早停／80k validation cap與本案不同，預設不重用。三模組共享baseline共308神經fits／93.81h基準等效，加N12為115.74h；大容量時間／顯存未知，並非預算上界，T4估計1.5–2.5 CU/h非實扣。先GRU B/A再Mamba B，昂貴容量需排練後按資源停／續，不能據勝負加格；全列CPU/RAM、split／hash、machine實作／獨立審查／Human凍結與現場仍待辦。Project map／machine authority不變，不授執行／費用或production權限，不重標舊證據。

## P4a 固定佇列候選 — 2026-11-22

[規格](../../tasks/spec-p4a-colab-20261121.md)／[機器契約](../../tasks/p4a-colab-contract-20261121.json)依Human核准在任何P4a結果前FROZEN：36主格＋24回看＋4學習率，再做排列診斷；環境使用既有OBSERVED cp313契約。工程候選 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，只完成CPU合成驗證，現場Colab仍UNVERIFIED。歷史mamba2/crossstock是P3已曝光replication，A/B/C Holm3/2/4與共同列限制不提供獨立確認；唯一獨立路徑仍是未曝光P2前瞻資料。C0+調參不對稱保留，本機CPU P4d另案OPEN。現況與authority見project map，無新研究結論。

# Research

## P3 修訂3 預算完整性 — 2026-11-20

[凍結修訂3](../../tasks/spec-mamba-portfolio-p3-20261120.md) 的[候選 adapter](../../V6/research_lab/rolling_seq_scores.py) 使用獨立 durable anchor 偵測全域 ledger 清空／合法前綴回退，與所有 sidecar snapshots 交叉核對，不依 out-dir。sidecar 在 END 後記錄 sequence／head／spent，manifest 與 evaluator 重驗。缺 state 時僅明確 `--rebuild-budget --reason` 能恢復，以全 namespace sidecar spent 最大值及存活可驗 state 保守扣帳，保留原檔與 REBUILT，STOP 待檢查。缺 spent／orphan 不推定0；重建不提供舊研究結果的新准入。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，CPU synthetic evidence only；同時刻意刪除 ledger／anchor／全部 sidecar 不在防護範圍。[報告與限制](../../reports/v7_p3_portfolio_engineering.md)。

## P3 修訂2工程候選 — 2026-11-20

[P3 修訂2](../../tasks/spec-mamba-portfolio-p3-20261120.md) 採 executor CUDA-only smoke v2 → 獨立審查 job → record-review → 正式24fits；審查 metadata 與 standing authorization 不代表逐項 Human acceptance。evaluator 重驗 evidence/source/environment/frozen cfg/mixer defaults、全域帳本 SHA/14h/STOP 與24fit sidecars，不信任 RESOLVED 字串。全域帳本含 hash chain、目錄 inode flock；STOP 先 fsync 再 END，必須帶 reason 的 CLI 留 CLEARED，刪檔不清除 latch。硬中斷仍保守扣帳，舊 per-namespace 證據不自動遷移。凍結隱含設定與 source constants hash gate 不改研究超參數。UNRESOLVED／GPU_SMOKE_NOT_RUN／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED；CPU替身及metadata工程驗證不升格研究結論。程序、實際數字與限制見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。

## P2 evaluator candidate — 2026-11-19

[Registered P2](../../tasks/spec-lowturnover-p2-20261119.md) 由現有 [evaluator](../../V6/research_lab/portfolio_eval.py) 的 `--mode p2` 執行；P1 為預設。重用唯讀 P1 rank-average scores，修訂 1 的 OFAT＋三指定組合（去重）14 descriptive cells／fold×system、六折兩系統共 168，另 12 個 P1 references；主要格與同分數 P1 reference 用共同日期比較，tail exclusions 明列。H-P2a/b 為 20-session／1000-sample circular bootstrap、one-sided centered-bootstrap p values、Holm family=2；缺 2021–2023 完整主要支援、非正式 seeds/train support 或 primary-only smoke 不作確認性判定。--fold／--system 子集的 grid_complete 必為 false，確認性維持 incomplete。P2 schema／run_hash 與 P1 不同，舊 P1 parts 不能續跑；位元一致只指預設路徑逐日策略與基準報酬。原子不覆寫 parts、input/spec/implementation SHA-256 與 source commit 控制續跑，ERROR 重試沿用 P1 immutable chain。最低費用為未驗證常見假設，比例另拆 full floor-binding commission 與 uplift。UNMERGED / NOT_DEPLOYED，E2 RESEARCH_ONLY；[工程報告](../../reports/v7_p2_portfolio_engineering.md) 不代表完整研究或 Human acceptance。

## P1 review-repair candidate — 2026-11-19

[Evaluation](../../V6/research_lab/portfolio_eval.py) now checkpoints each fold×system under `.artifacts/p1/<run>/part-*.json`; a file is atomically published without replacement after serialization/fsync. Source/input/config identity plus payload hash control resume; corrupt/mismatched parts stop rather than being overwritten. Individual cell exceptions and nonfinite output are ERROR with type/message; valid cells still finish and aggregate. No power-loss or network-filesystem guarantee is inferred from the process-kill smoke. Early-price-termination reports count held session marks and blocked scheduled-open marks in initial-NAV units; the sums are exposure diagnostics, not liquidation proceeds. Every computed cell reports actual start/end and return count. Candidate confirmation, separated clean/descriptive paired inference, tail truncation, net benchmark costs and cross-fold block caveats are explicit. Actual raw Volume magnitude for 2330 corroborates shares; both liquidity and capacity use shares after conversion, with UNKNOWN rejected. Full study, delisted coverage, PIT and trading readiness remain unverified. See [current state](../00_Project_Map/Current_State.md#p1-review-repairs--2026-11-19) for this bounded repair.


## P1 portfolio study candidate — 2026-11-19

[Registered P1](../../tasks/spec-portfolio-eval-20261119.md) now has uncommitted [score-export](../../V6/research_lab/rolling_scores.py) and [evaluation](../../V6/research_lab/portfolio_eval.py) candidates with [causal/accounting regressions](../../V6/research_lab/tests/test_p1.py), UNMERGED / NOT_DEPLOYED. Reuses existing rolling/C0+/portfolio semantics; full study is not executed by the bounded worker smoke. Previous-day market-cap strata remain NON_PIT due to snapshot shares. Read-only raw prices lack confirmed delisted coverage and adjustment data; primary-only smoke is E2 RESEARCH_ONLY engineering evidence, never H-P1 confirmation. Current navigation/boundaries are in the project map.

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: experiment lifecycle, evidence classes, current research decisions
- Update triggers: experiment completion, decision report, baseline, or evidence-policy change

## Purpose

Provide an index for research state without turning reports, diagnostics, or historical artifacts into machine contracts.

## Current Boundary

Corrected V7 research proceeds under explicit dataset, feature, label, model, split, seed, and acceptance contracts. Strict Phase 0 is `STOP`; historical-simulation readiness is separately `PASS`. The latest diagnostic decision is `STOP_FOR_TRADING_CONTINUE_RESEARCH`.

## Authority

Machine contracts and completed artifacts on `fix/v7-historical-pit-reconstruction@e3bce91` and `feature/v7-e1-rolling-origin-refresh@044eef7` outrank prose reports. History explains decisions but does not override current contracts.

## Inputs

Versioned data/feature/label/model contracts, frozen hypotheses, seeds, splits, acceptance criteria, and compute environment.

## Outputs

Immutable run artifacts, diagnostics, comparison tables, decisions, and explicit next-experiment contracts.

## Current Implementation

The corrected E5 replication formally supports seeds 17 and 29. Seed 43 is incomplete and excluded from the formal ensemble. Diagnostics found signal degradation and near-redundant heads, leading to E1: a rolling-origin refresh under unchanged architecture/training. E1 is now admitted as a completed two-seed research experiment with `PASS_REFRESH_HYPOTHESIS`; it remains historical-simulation proxy evidence under strict Phase 0 `STOP`. Post-run month analysis finds July-sensitive 10d absolute IC, while final-support epoch degradation and ex-ante regime dependence remain inconclusive.

## Evidence Classes

- Completed contract-matched run: experiment evidence.
- Diagnostic: explanatory evidence, not promotion.
- Partial checkpoint: forensic/convergence evidence.
- Spec/package without run: planned capability.
- Legacy A/B/C: historical evidence under their original contracts only.

## Current vs Legacy / Research

Legacy A/B/C artifacts remain unchanged and are not tradable evidence. Corrected replication, E1, and Phase A answer different questions and must not be merged into a single readiness claim.

## Invariants

- Freeze the question and acceptance criteria before execution.
- Record all exclusions and partial seeds.
- Compare only identity-compatible artifacts or label the comparison limitation.
- No research result bypasses correctness gates.
- Package success is not run success.

## Known Limitations

2026-10-21 bounded recovery: six E1 checkpoint hashes and both prediction hashes
match admitted evidence; seven postrun files match fixed Git. The frozen E1
matrix remains unlocated in searched local paths/archives; remote Drive was not
inspected. Replay is STOP, not a new research conclusion.
See [recovery report](../../reports/v7_e1_evidence_recovery.md).

E1 result artifacts are versioned, but post-run epoch replay lacks the frozen full matrix and safe checkpoint deserialization; causal regime decomposition lacks ex-ante state variables. Runtime telemetry for the earlier corrected replication is incomplete.

## Do Not Use / Do Not Mix

- Do not continue Phase 1–5 while strict prerequisites say `STOP`.
- Do not promote on simulated labels without stating the evidence class.
- Do not rewrite legacy artifacts after learning corrected semantics.

## Important Paths

- Correctness baseline: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/`
- Diagnostic v2: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/`
- E1 contract: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json`
- E1 result: `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`
- E1 post-run diagnostic: `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`

## Related Domains

[Data](Data.md), [Features](Features.md), [Models](Models.md), [Training](Training.md), [Labels](Labels.md), [Production](Production.md)

## Update Triggers

Update after a completed/rejected experiment, baseline promotion, acceptance-policy change, or new authoritative diagnostic decision.

**2026-09-22 source integration:** Corrected E5/E1 reusable research modules and selected frozen contract copies are now in development main; see `research/v7/source-adoption-register-20260922.json`. Run results, bundles, postrun diagnostics and operational/deployment authority remain separate. No model training or production promotion occurred in Phase 3B-2.

## Strategic recovery planning pointer — 2026-10-21

Current objective/implementation gaps are recorded in the
[dated recovery report](../../reports/v7_strategic_roadmap_recovery.md).
The sole active cross-domain backlog is the
[existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
This pointer does not replace this domain's machine contracts or promote evidence.
Five CPU fixes have a verified integration candidate from `4c388de`, UNMERGED /
NOT_DEPLOYED. PORT-01–04 H1 fixture engineering is now COMPLETED_VERIFIED;
Human Product Slice Review remains pending. See the
[acceptance report](../../reports/v7_portfolio_history_acceptance.md). Strict Phase 0
STOP, E1 portfolio admission=false and production boundaries remain unchanged.

## Autonomous research preparation — 2026-10-21

[Decision preparation](../../reports/v7_autonomous_research_preparation.md) maps
fixed E1 result/diagnostic/recovery evidence to options and fairness checks.
PROPOSED / NOT_APPROVED: no E2 selection, new experiment or admission. Sources and
checks are in [window evidence](../../research/v7/autonomous-window-01-evidence.json).
E1 matrix stays BLOCKED; no repeated discovery without a new exact location.
The Charter permits preparation, while Human retains formal research decisions.

## Charter integration — 2026-10-21

Window01 source/governance is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED
at exactf7af348 (supersedes its branch-local status only). Pre/post CPU regression
and authority checks are in [integration evidence](../../research/v7/charter-integration-20261021.json).
A separate Human-authorized cross-domain CPU window may proceed in isolation;
research contracts, formal data, production and StrictSTOP boundaries are unchanged.

## Window02 research preparation — 2026-10-21

Source-level audit confirms shared trainer reuses split_code2 evaluation for
checkpoint selection and final evaluation; this is research-used historical
support, not untouched holdout certification. This does not revoke admitted
results or quantify historical selection bias. Occlusion is not retrained ablation.
[Window02 report](../../reports/v7_cross_domain_window02.md) records11 pinned source
identities and PROPOSED comparison-design options; Human approval remains required.
[Unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
is WINDOW_COMPLETE/STOP. Strict Phase0 STOP; E1 matrix BLOCKED; no new search.

## Window03 fairness and readiness preparation — 2026-10-21

Completed source-level E5/E1 split/selection/final-evaluation review and PROPOSED
comparison designs, without changing existing result classifications or policy.
Existing validators/configs match historical E1 contract/config identities;7 copied
result hashes and2 seed best-pointer metadata chains checked. No matrix/weight
payload validation, training or Colab launch. NOT_RUN_READY; original matrix BLOCKED.
[Report](../../reports/v7_data_evaluation_window03.md),
[unique completed queue](../../research/v7/research-program.md#data-evaluation-window-03).
Main9956446 unchanged; branch `development/v7-data-evaluation-preparation` is
UNMERGED/NOT_DEPLOYED. New bounded execution and formal research decisions need Human.

## Window04 Research Decision Package

[Evidence-grounded package](../../reports/v7_fair_evaluation_decision_package.md)
adds exposure ledger, symbolic sample/fit-count requirements and telemetry cost
limits to Window03 review. All options PROPOSED; no final dates/thresholds or new
research classification. Historical logged epoch wall time is not a GPU cost forecast.
Original matrix and unretained epochs independently limit replay; STOP/H2 unchanged.
Current unique backlog records WINDOW_COMPLETE; next execution needs Human grant.

## Window07 temporal foundation — 2026-10-23

Branch-local engineering guard checks explicit calendar/fold/clock/required-horizon/
owner/pin/maturity/interval/gap constraints. ELIGIBLE means consistency with supplied
engineering inputs, not source truth, frozen study or admission. Calendar/maturity/
cutoff/gap policy remains a Human dependency before real research. Existing E5/E1
semantics/results unchanged. [Report](../../reports/v7_window07_temporal_isolation.md).

Window07 U2 adds run-lineage/read-scope validation and explicit later-fold forward
exposure records. Forward views never read label_store; fresh-model/optimizer/
schedule/RNG and readiness are mock declarations, not actual training/approval.
Source/ledger pins and all-or-nothing views support a future synthetic comparator;
no Common Evaluator policy, Window08 or real-data execution is authorized.

## Window08 synthetic evaluator foundation

Common ranking adapter reuses frozen E5 average-tie RankIC without changing historical
results. Named synthetic policy owns minN/weighting; metadata-first maturity/role
guards precede label payload reads. [Evidence](../../research/v7/window08-evidence.json)
and sole Backlog own branch-local status; real policy/data admission remain absent.

## Window09 durable synthetic evidence

[Bundle/replay evidence](../../research/v7/window09-evidence.json) tracks immutable
filesystem publication and safe offline validation. Artifacts verified != training
complete != research admitted. Hash without actual bytes is NOT_RESTORABLE. No old
E1 matrix reconstruction or legacy unsafe checkpoint load. Main W08 integrated;
new Window09 source and memory stay isolated; sole Backlog owns current status.

Window09 complete: 141CPUtests, independent40,0Critical/Required. Replay is byte-/
metadata-pinned W08 recomputation, not model retraining. Four proof flags remain
independent: metadata/evaluation true; training/full reproduction false. Safe JSON
checkpoint roles and latest->previous fallback do not validate actual PyTorch
optimizer/RNG/AMP/sampler execution. New bounded authorization required; STOP.

## Window10 classical synthetic adapters

Human authorizes tiny CPU toy fitting only. W07 owns eligibility/roles; new
L0/Ridge adapters consume guarded views, no new split or ranking implementation.
L2 dependency absent, code-present != fit-verified. [Evidence](../../research/v7/window10-evidence.json)
and sole Backlog own current status. Actual fit status separate from immutable W07
initial authorization declarations; no training resume or research admission.

Window10 complete:167jointtests; independent final U2+oldW09=53PASS,0Critical/Required.
L0/Ridge tiny synthetic validation/selection/freshrefit/forward and W09 independent
validation+forward replay proven. V1 no-fit semantics retained; v2 classical-fit
profile requires actual fitted producer ledger/lineage/config pins. L2 blocked by
missing dependency; no real financial evidence, training resume or GPU acceptance.
No remaining READY work under this grant; STOP/new bounded authorization required.

## Window11 L2 and actual CPU resume

[L2 evidence](../../research/v7/window11-l2-evidence.json): dedicated fixed official
overlay actual fitting validated, no real research admission. E1 reused for actual
CPU resume proof, status owned by [Window11 evidence](../../research/v7/window11-evidence.json)
and sole Backlog. No unknown pickle, GPU or missing E1matrix reconstruction.

Window11 actualCPU proof: 12steps/4epochs, interrupted atstep4 afterwarmup, new
process exact state/trace match; LATEST/PREVIOUS/BEST 23fields each. Granularity is
deterministic daily cursor at fulloptimizerstep,0workers/FP32/singleCPUthread only.
No shuffled/multiworker/AMP/GPU or separateepochinterrupt claim. New safe codec
weights_only=True with primitive NumPy RNG, original E1 default behavior unchanged.
Checkpoint bytes retained locally, not derivable from hashes. U2 source complete;
source-bound L2 republishing and final memory closure still pending.

Window11 complete/STOP:211jointtests, actual cross-process CPU full-step/mid-epoch
resume proven for fixed daily-order/0workers/FP32/1CPUthread only. Separateepoch
interrupt/GPU/AMP/shuffledloader/full reproduction unverified. Code/source witnesses
map actual bytes to U2 sourcecommit; originalL2snapshot retained, latest sourcebound
L2run is current. CPU actual checkpointbytes stay local/preserved, not hash-only
restoration. Source hashes/authenticity/research admission remain distinct.

## Window25 bounded robustness audit

Fixedpredictions showconditional exploratorypositivity, not formalrelativeperformance:2022CBBblock-sensitive,2023exposed. LOSOonlyevaluation,nottraininginfluence; matchedbaseline/statisticaldesign requiresHuman.
See [report](../../reports/v7_window25_robustness_audit.md) and solebacklog. 0datarequests/fit;W19protected;FormalAdmissionNOT_APPROVED.

## Window26 bounded label evidence — branch-local / not admitted

2321 的 68% 虧損減資與市場股數單位轉換已有來源證據；三筆既有 10d raw labels 算式正確，但不具同股數未調整報酬語意。
原始停牌公告與其他事件覆蓋仍 UNKNOWN；W24/W25 結果不改寫／不重算。兩路 Label Policy 與 next-study 全為 PROPOSED。
參見 [W26 evidence](../../research/v7/window26-evidence.json) 與 [事件矩陣](../../research/v7/window26-event-matrix.json)。

## Window28 limited clean Ridge / causal check

Human限定批准W27安全Mask下的2021 Ridge fresh fits各horizon一次（2/2已用），原alpha1與兩Features不變；Scaling只取clean TRAIN。
2022/2023同W27 support為已曝光Development；18個有限future OHLCV數值擾動不改Features/固定模型Predictions。
此pass不驗未來Label eligibility gate為as-of，也不認證完整公司行動/vintage。
[Run](../../research/v7/window28-run.json)、[Report](../../reports/v7_window28_clean_refit.md)。

## Window29 disjoint development / feature-label isolation

68新TWSE候選（W24重疊0）、64有conditional Prediction身分；current survivor/hindsight與4下市起日/普通股UNKNOWN不升格歷史Universe。
最小research feature-only介面先保存Predictions，再做Label Join；future label刪除只改evaluation coverage，18回歸tests與獨立Raw-derivedReplay驗證。
5d正向Mean/Median在新樣本延續、10d不一致；未知actions/asof/vintage與2022/2023曝光保留，不作模型選擇/投資/Strict PIT claim。
[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)。

## Window30 fixed real LightGBM development baseline

Human明確批准一組小型real-data recipe，60e7ca7在Fit前固定；W11環境4.6.0不變，2fits/64trees、TRAIN-only bins，無validation earlystop或search。
60726 feature-only predictions /same W29support與獨立Replay通過；5d正、10d負，4組Mean均低於舊Ridge，但不宣告winner/研究准入。
Expanding各Origin仍需另凍結model-ready/maturity/purge/refit/selection與資源；本次不自動批准。
[Evidence](../../research/v7/window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)。

## P3 bounded engineering candidate — 2026-11-20

[P3 spec](../../tasks/spec-mamba-portfolio-p3-20261120.md) 對官方Mamba2/no-conv有未解語義矛盾；[adapter](../../V6/research_lab/rolling_seq_scores.py) 與 [P3 portfolio mode](../../V6/research_lab/portfolio_eval.py) 為UNMERGED / NOT_DEPLOYED工程候選，正式訓練fail closed。只驗證injected CPU與synthetic三系統流程、P1相同列集合、maturity/purge、種子rank PIT、common dates、Holm及溫控；不升格IC／portfolio／GPU或trading evidence。不得用既有pilot09歷史結果取代此次研究。493passed/1skipped、五反證及完整限制見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。未完成fit跨程序GPU預算尚非可靠會計；需先解決才能正式解鎖。

## P2 前瞻與 P4b 文件候選 — 2026-11-24

[P2前瞻草案](../../tasks/spec-forward-p2-primary-20261124.md)／[P4b草案](../../tasks/spec-p4b-blend-20261124.md)仍DRAFT待Human凍結，E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，無新結果或執行授權。2026-11-24最新兩階段文件修訂：`job-run-mwuoasa7-93afef/spec_writer_phases`／detached base `0db633bcedc97484bff6c316d7cc9d19334137d2`，初始clean，4份文件未提交patch；DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Human新決定取代Q1／Q3：嚴格T0（候選2026-11-24之後）僅試運行整日流程，績效封盲且永不調規；年報後重訓T1才正式起算252收益sessions、唯一H-F1／Holm m=1／雙邊保守p。一般年度3/31＋Q2 8/14作整體更新時鐘，75日、金融半年8/31、順延／延期／非曆年等按公司期別核實，不能當可得日。DRAFT提案年報N_A=10／Q2 N_H=8＋必要資料實際公開／入管線2-session緩衝，保留18:00與最多2-session重試／超時STOP；N是作者提案、非已凍結Human數值。試運行預設直到正式端點才描述性揭曉，另列T1簽署且啟動後揭曉／永不揭曉替代風險；正式規則／日曆必在任何階段0績效曝光前簽署。

無假日週日曆示例為T1=2027-04-14、Q2更新8/25、252末日2028-03-31、次年年報更新4/14；正式交易所calendar未驗，真實端點會受休市影響，第二次更新可能入窗，正式次數UNKNOWN，不按績效選日。工程按兩次容量準備，逐版計算訊號與20-session再平衡暴露。WP1～WP6拆T0前最低安全日鏈與T1前完整驗收，建議T1前20／5 sessions截止；20日穩定流程及跨Q2仍未啟動就STOP是待凍結提案。試運行不穩只按無績效gate延後T1，重新as-of重訓、不回填；正式開始後不得重設T1／延長252日。所有工程仍另案批准／獨立驗收，文件完成不等於Human acceptance。

期限與例外來源見[期限研究](../../tasks/research-financial-deadlines-20261118.md)，僅引用其既有已驗範圍。當初是否量測被移除欄位的洩漏幅度未驗證；加回基本面需另案逐公司公告時間／修訂歷史與PIT驗收。Source核對V6 builder有revenue月末+11日、financials預設期末+45日，fundamentals_v2才年度+90日，沒有由此證明實際publication／revision PIT。P1 corrected 48欄manifest已排除財報／營收等11欄，C0+185欄未加回，季報後重訓只是更新時鐘，不宣稱納入新財報。Human兼任封盲保管及重訓owner，私有區 `D:\Desktop\work\ProjectForMe\MarketMamba-blind\`／`/mnt/d/Desktop/work/ProjectForMe/MarketMamba-blind/`，Claude／Codex禁止讀寫；可見區僅淨化hash chain／行數／覆蓋率／錯誤計數／缺日。Claude拒絕設定只約束Claude工具，Codex同WSL權限，屬防疏忽紀律、非防竄改；未驗OS隔離。Human手動一次開盲，Claude開盲後才取結果，偷看降描述性。WP1～WP6前置工程另案批准／驗收、資源僅估計，未實作未啟動；實際T0／T1／calendar、PIT／lifecycle／coverage、矩陣／模型／252日adapter與封盲演練仍Blockers。P4b原固定歷史C0+、三混合Holm3、36-fit subset與64-fit gate及文件bytes不變；既有strict STOP／production authority不變。詳見[Current State](../00_Project_Map/Current_State.md)，無runtime observed_at更新。

## P4d 第一段候選 — 2026-11-25

Human對話分段凍結baseline、GRU B及A六格；原全文SHA見[Current State](../00_Project_Map/Current_State.md)，其餘Mamba2／D／N12仍DRAFT。[工程報告](../../reports/v7_p4d_engineering.md)與[契約](../../tasks/p4d-stage1-contract-20261125.json)為UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED候選；七origins共140 fits、不重用P4a，新帳本獨立，outer不選epoch。CPU合成測試及prepared唯讀split不是新績效、現場GPU／Drive驗收或PIT證據；Holm2家族不縮減，第一段只描述。下一步獨立審查，沒有額外訓練或交付授權。
