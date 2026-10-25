# Window23 — Calendar Closure and Retrospective Engineering Smoke

Status COMPLETE_STOP_REVIEWED_GO / UNMERGED / NOT_DEPLOYED. Human Window23 limited engineering authority,
not Formal Data Admission or Exploratory Study Admission. Main ff-only7027f03 from428a5fa,
W22 GO/23tests/78 localfiles+23deliverablepins reverified; W22 original data/mask and W19 untouched.
Branch research/v7-window23-calendar-smoke; base7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470.

## Calendar and lifecycle evidence

Ten official-document loads:8 direct GETs preserved with raw/snapshot/receipt,2 web document opens
(TWSE landing page and existing exact closurePDF for text verification). Search-index metadata queries
were separate, not quotation API downloads. FinMind requests0. Budget CLOSED; no further requests.
TWSE holidaySchedule2019/2020 returnedHTTP200 but emptydata, not evidence those years had no sessions.
2021/2022 schedules plus weekday geometry exactly match244/246 W22 observed dates.
2023 schedule with official20230802 announcement closing20230803 exactly matches239 observed dates.
Do not assume annual schedules contain all temporary closures; date equality and archived temporary
announcement support only the declared retrospective engineering calendar. Not historical-asof admission.
20230117 is final trading day despite its note mentioning Jan18/19 nontrading; parser tests this distinction.
TPEx2023 officialPDF and2020 notice archived; full TPEx2019–2023 machine-qualified calendar not established.
No replacement calendar source or quotation API was queried. Raw docs live in root.artifacts/window23-calendar-smoke/official-docs.

Private date matrix:1,826 days =9 OFFICIAL_SESSION_VERIFIED,1,207 OBSERVED_MARKET_SESSION,
64 OFFICIAL_CLOSURE,1 TEMPORARY_CLOSURE,545 UNKNOWN. Explicit opens differ from inferred scheduled weekdays;
FinMind alone is never promoted.2019/2020 excluded from smoke, though their retrospective prices remain stored.
TPEx source policy not silently inherited from TWSE; OTC stocks excluded from this smoke.

Ten-stock date membership matrix:12,160 rows; TWSE_LISTED5,581 / TPEX_LISTED3,648 / DELISTED767 /
SUSPENDED26 / EMERGING2 / UNKNOWN2,136. Listing interval evidence uses W22 official source boundaries;
it is not daily executable-tradability certification.6515 TWSE interval begins2021-01-20;
8272 TPEx interval begins2024-10-23, so its2019–2023 prices never become TPEx-qualified here.
EMERGING only the two exact documented W21 dates.2499 known-suspended-by20200930 until delisting20201110;
earlier missing-price period remains UNKNOWN, not inferred suspension. All daily tradability fields UNKNOWN.

## Original label failure decomposition

W22 Strict mask unchanged:19,728 entries,0 stricteligible. Overlapping reasons:
calendar/membership/corporate-actions/feature-availability/label-maturity each19,728;
entry-outside18,exit-outside135,invalidentry16,invalidexit16,missingentry4,missingexit27.
Historical PIT evidence is insufficient for all19,728; this is not a claim all numeric data is invalid.
Exclusive operational classes after independent engineering-mask/split rules:
A195 W22 price/endpoint geometry not computable;
B7,009 usable only for retrospective engineering diagnostics, not strictPIT;
C12,524 blocked by remaining data/calendar/market/feature/split requirements.
Classes are not research admission. Known corporate-action crossings would be excluded, but no complete
qualified action feed exists: every retained raw-price diagnostic carries this limitation.
No raw price adjustment or forward/zero fill. No rawreturn is executable or total-return evidence.
[Detailed diagnostics and private mask references](../research/v7/window23-diagnostics-summary.json).

## Fixed smoke, actual run and replay

[Recipe fixed before fitting](../research/v7/window23-smoke-contract.json):
entry open[t+1], exit close[t+h] on shared session index(h5/10); label close[t+h]/open[t+1]-1.
Two causal raw features: close[t]/close[t-1]-1 and(high[t]-low[t])/close[t].
Historical publication/revision/PIT remainUNKNOWN; featuredecision is a logical t-close diagnostic timestamp,
not a claim these snapshots were actually generated then. Known action windows excluded if evidence exists.
Train signals2019–2021 masked to2021,Validation2022,Forward-style2023. Labels cannot cross split boundary.
This is not untouched final evaluation. Hindsight-selected sample never represents historical full market.
Eligible subset2303/2317/2330/6515/9929, chosen by calendar/market evidence rather than outcome.
Two Ridge fits only(alpha1,training-onlystandardization,seed0), L0 uses named momentumfeature,+direction.
No selection, hyperparametersearch, fresh refit or model contest. Forward/validation labels never enter fit.
Controlled process timeout300s, CPU thread1 / CUDA_VISIBLE_DEVICES empty; no GPU/Colab.
Actual sample counts:Train5d1171/10d1146;Validation5d1201/10d1176;Forward5d1170/10d1145.
W08 shared daily metric kernel produced1,880 model/session/horizon rows:1,864 defined,16 insufficient support.
Engineering-only average ties/minN5/equal-session policy was fixed before fit, not frozen formal policy.
Both models use identical per-session support. Per-dayN/stockkeys/reasons are preserved, never fill RankIC0.
No model superiority or investing interpretation is permitted.

W10 existing Ridge numerical solve/prediction extracted into shared functions without altering synthetic
recipe/admission guards. W07 calendar normalization and W08 daily_metric reused directly;
full synthetic-only evaluate/fit_baseline interfaces were not bypassed by falsely labeling data synthetic.
W09 publisher extended with explicit RETROSPECTIVE_ENGINEERING_SMOKE profile and HumanWindow23 scope;
synthetic profiles and research_admitted=False guard retained. No new trainer/evaluator/registry/platform.
Full W09 durable artifact verification is used; W09 synthetic-only replay_bundle is not claimed to support
this new profile. Thin offline replay verifies bytes/identity, uses the same W08 kernel and compares hashes.

Initial replay FAILED list-order comparison; all keyed metric values were independently shown identical.
Single evidenced repair normalizes model/session/horizon identity, rejects duplicates, and preserves original
bundle/metrics/fits/run-code bytes. Replay adapter has a separate code hash; original run code pins remain verified.
No repeated fitting. Independent process replay matched all1,880 daily metrics and checked saved diagnostic label identities,
code hashes and W22 raw inventory. Bundle includes config/environment/data/split/mask/models/logs/predictions/
daily metrics/completion. Actual publication timestamp2026 is kept separate from logical2023 origin.
Private bundle root.artifacts/window23-calendar-smoke/bundles; no vendorpayloads/predictions/labelvalues inGit.
[Run identity](../research/v7/window23-run-summary.json),[Replay](../research/v7/window23-replay.json).
Same-device restore copies are not independent backup: BACKUP_PENDING. No artifact cleanup authorized.

## Remaining minimum Human decisions

Formal study still needs qualified/scoped historical calendar and universe/action/missing policies;
independent retrospective feature/label/availability contract and dataset-use admission;
actual fold/purge/selection/refit/evaluation/compute approval. The smoke freezes none of those policies.
No further data purchase or broad download is justified by this pipeline smoke alone.
StrictPhase0STOP/E1BLOCKED/H2NOT_APPROVED/W12SAFETY_HALTED remain. W19task/worktree/artifacts/budget/runtime not operated.
After localcommit STOP. No W23mainmerge/push/deploy or nextwindow.

## Final verification and recovery

58 focused testsPASS;23main-postintegrationtestsPASS;independent26testsPASS(3.046s),GO/Critical0Required0.
One bounded replay-order repair;no real-data refit and no extra requests.
Reviewer verifiedW22 78/W23 55artifactpins,bundle/code/data/split/predictionidentities anddailyreplay.
Independent same-hostrestore56filesPASS,includingpublishedbundleandarchivedofficialsourcebytes.
Governance/AuthorityDrift/LegacyDocs retainedWARNING;structuralchecksPASS.
[Review](../research/v7/window23-review.json),[restore](../research/v7/window23-independent-restore.json),
[finalevidence](../research/v7/window23-evidence.json).

Recovery:`cd /home/frank/projects/MarketMamba/.worktrees/v7-window23-calendar-smoke`;
readCurrent_State→Authority_Map→solebacklogWindow23→report/evidence,verifyHEAD/cleanandlocalinventory.
RUN_CLOSED blocksrepeatfitting;officialdocumentbudgetCLOSED. Onlyauthorizedoffline replay may be repeated.
Original source-at-runpins remain inprivate rootandrestore;currentreplayadapter is recordedseparately.
Do not deleteworktrees/raw,mergeW23,pushorstartanotherwindow.
