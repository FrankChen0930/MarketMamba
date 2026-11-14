# Window25 — Ranking Signal Robustness and Leakage Audit

Status COMPLETE_STOP_REVIEWED_GO / UNMERGED / NOT_DEPLOYED. Scope: HumanWindow25 evidence audit only.
W24 ff-only integration87af76b→90b1327560f717727ad3fd53dfe86cb421c70f5b afterGO/26tests,
191sourceartifacts/33deliverablehashes/rawrestore verified. NoW06/W12/W19 integration.
W25branch research/v7-window25-robustness-audit,base90b1327,worktree .worktrees/v7-window25-robustness-audit.
Zero requests/fit/refit/newpredictions/GPU; originalW24data/scores/policy immutable. Runtime guards deny
urllibopen,socketconnect,np.linalg.solve andbaselinefit. LocalCPU/W11env/threads1,process timeout300s.
[Fixed pre-analysis plan](../research/v7/window25-analysis-plan.json);not a new formal statistical policy.

## C — Leakage matrix

[Machine matrix](../research/v7/window25-leakage-matrix.json):44,266source-reconstructed samples checked;
zero confirmed logical feature/entry/exit/split mismatch. Featuresuseonlysignal/priorOHLC;
no currentcapital feature. BothTrainlabels end no later2021-12-30, beforedeclaredlogical2022-01-01cutoff.
5d training signals end2021-12-23;10d end2021-12-16. Horizon-aware boundary exclusion matchesW24;
no extraE5purge/embargo adopted. Val2022/Forward-style2023 labels remain outsidefitting.
7,338/7,188training hashes andTRAIN-onlymean/std match savedmodels; no imputation added.
Verification uses saved code/view/state, not a refit or proof of any unrecorded experimentation.

Actualhistoricalpublication/availability/labelmaturity/revisionvintage remainUNKNOWN.
W24generated2026-10-25, not itslogical2023origin or2021trainingcutoff. Cohort/currentcapitalfrom2026
is knownhindsightselectionbias, not historical universe evidence.3retirees notcompared dueunknownstart.
Recorded L0/Ridge/features/seed matchW23; no recorded selection/refit/search or adaptiveparameterchange.
2022/2023 were already exposed; no claim about unrecorded subjectivechoices or untouched final evaluation.
Conclusion: no confirmed numerical cross-period leakage in storedpipeline; NOT full no-leakage/PIT certification.

## D — Original fixed common-support diagnostics

Original W24AVERAGEties/minN20/equal-sessionpolicy reused.33eligible securities;N27–33.
940coredate×horizon groups;1,880daily metrics replay identical.0missingcore/undefinedoriginalsessions;
5or10tail/split dates peryear excluded explicitly, not fabricatedas0RankIC.
Allstats use dailycross-sections, not individualsecurityrows as independent observations.
Full L0/Ridge/delta medians,ddof1dispersion,quantiles,pos/neg/zerofractions,Nhistograms,ties,lagcorrelations in
[robustness JSON](../research/v7/window25-robustness.json).

| Group | Days | L0 mean | Ridge mean / median | Ridge daily SD | Ridge positive | Paired mean delta | Delta positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022/5d | 241 | -0.01356 | 0.03702 / 0.04412 | 0.20363 | 58.9% | 0.05058 | 54.4% |
| 2022/10d | 236 | -0.00589 | 0.04393 / 0.05656 | 0.19623 | 61.0% | 0.04982 | 56.8% |
| 2023 exposed/5d | 234 | -0.04995 | 0.08031 / 0.07899 | 0.22802 | 62.4% | 0.13026 | 65.8% |
| 2023 exposed/10d | 229 | -0.04599 | 0.05406 / 0.06764 | 0.21907 | 59.8% | 0.10005 | 59.8% |

Ridge positive on58.9–62.4%ofdates, not everydate. Means/medianspositive;dailySD~0.20–0.23 greatlyexceedsmeans.
Adjacent5d/10dlabels shareexitwindows and marketconditions;940groups are NOT940independent experiments.

### Fixed sequence-dependence sensitivity

Both10/20blocks,2,000resamples each,seedbase2501+groupindex*100+blocklength fixed beforeanalysis.
ReuseexistingE1 CIRCULARmoving-block lower helper,upper=-lower(-values); notE1acceptance/RankICcode.
Same draws apply pairedvectors. Finite contiguous marketseries required; no NaN/gap compression.
Bounds are nominal2.5/97.5empiricalpercentiles, EXPLORATORY ONLY, not formalCI/significance.
Assumptions: approximatewithinperiodstationarity; artificialend/startwrap; longerregimesuncovered;
no multiplecomparisons correction orformalcoverage certification. Allsettingsreported,nonechosenasbest.

| Group | Delta block10 bounds | Delta block20 bounds | Ridge block20 bounds |
|---|---|---|---|
| 2022/5d | [-0.0060, 0.1101] | [-0.0051, 0.1048] | [0.0049, 0.0671] |
| 2022/10d | [-0.0018, 0.1035] | [0.0011, 0.1007] | [-0.0038, 0.0901] |
| 2023 exposed/5d | [0.0705, 0.1844] | [0.0787, 0.1840] | [0.0433, 0.1174] |
| 2023 exposed/10d | [0.0359, 0.1652] | [0.0410, 0.1646] | [0.0094, 0.0998] |

2022delta comparisons are block-sensitive:5d crosses0underboth;10d crosses0with10andnot20,
whileRidge10dblock20itselfcrosses0. Do not selectthefavorableblock orclaimformalcomparison established.
2023bounds directionallyconsistent butalreadyexposed,not independentconfirmation.

## E — Security/date influence and raw-price risk

LOSO33securities×4groups=132cases/31,020paired-day recomputations; predictions/models unchanged.
W08average-tieRankICrecomputed onexactsamepost-removalstock/labelsupport,minN20unchanged;
N26–32 andzero undefinedLOSOdays. Everyoriginaldate remains accountedfor.

| Group | Ridge LOSO mean range | Paired delta LOSO mean range | Ridge mean without5largestpositive days |
|---|---|---|---|
| 2022/5d | 0.03036–0.04072 | 0.04032–0.05739 | 0.02513 |
| 2022/10d | 0.03188–0.05152 | 0.04139–0.05766 | 0.03405 |
| 2023 exposed/5d | 0.07501–0.08660 | 0.12062–0.13859 | 0.06816 |
| 2023 exposed/10d | 0.04569–0.06638 | 0.09252–0.11271 | 0.04403 |

No single-security removal orpre-fixedtop5positive-date diagnostic flipsRidge meannegative.
ThisdoesNOT exclude multi-security/sector/regime dependence; evaluationLOSOdoesnotremovethatstockfromtraining.
Theseare sensitivities,not permissiontodelete stocks/dates,rebuildcohort orrestateperformance.

No qualified datedcorporate-actionfeed is saved, so actualevent-exclusion sensitivity UNKNOWN/NOT_ESTIMABLE,
not evidenceofzeroeffect. Rawlabels include2023-10dmaximum~+224.37% for2321; predefinedtop10caseinspection
tracedthreeoverlapping2321labels acrosssixmissingmarketrecords(2023-02-09..02-16).
CauseUNKNOWN:cannot infercorporateaction,suspension,units orsourceomission fromjumpalone; no newdataqueries.
Post-auditcase trace is explicitly investigative,not pre-registered filtering/threshold. W24endpointpolicy
allowedintermediategaps:3TRAIN/26VAL/7FORWARDlabels crossmissing records;
518TRAIN/348VAL/222FORWARDlabels containintermediatezeroOHLC records(countshorizons/samples,notuniqueevents).
Endpointpriceswerevalidated,butthisdoesnotestablisha continuous executableorcorporate-action-comparablepath.
No labels/metrics were removed orcorrected. ExistingLOSO2321stillleaves2023-10dRidge~0.04837;
thiscannotundoanypossiblecontaminatedtraininginfluence. Rawreturn≠totalreturn/executablereturn.
Privatecase/rootLOSOdetails preserveidentities/hashes withoutpublishingvendorpayloads.

## F — Baseline meaning and evidence class

L0fixed +momentum1;Ridgepredictssamehigherrawreturnusing2supervisedfeatures.
Savedstandardizedcoefficients for(momentum,range):5d(-0.0017033,-0.0015824),10d(-0.0008058,-0.0017188).
Directionsarelegitimatelearned2021weights, not an architectureadvantageovera matchednegative-momentumfactor.
No L0directionflip ornewfactor/baselinewascomputed/selected; comparisoncontainsdirectionandfeature-setdifferences.

Overallconditionalclassification **ENGINEERING_VALID_WITH_EXPLORATORY_SIGNAL**, scoped ONLYtoW24rawproxy.
2022relativecomparison remains SIGNAL_FRAGILE / block-sensitive; actualactions/asofquality INSUFFICIENT_EVIDENCE.
Thisis agent-observeddiagnosticclassification,notHumanresearchcorrectness/admissiongate orupgradeofW24status.
[Classification](../research/v7/window25-classification.json). No confirmedcode-levelcrossperiodcontamination;
unknownPIT/actions maystillcontaminateeconomicinterpretation. FormalResearchAdmission remainsNOT_APPROVED.

## G — Minimum next-study requirements

Targeteddatedcorporateaction/suspension/priceunit evidence,especially2321gap, andexplicitraw/adjustedlabelpolicy;
historicaluniverse/calendar/availabilityorclearlyapprovedproxy/bias scope; predeclared direction-matchedbaselines,
newunexposedcomparisonwindow/selectionbudget; Humanstatisticaldependence/multiplicity/refit/compute approval.
Do not fixsignalbyrecohorting,reverseL0,retuneblocklengthornewmodelsearch. NoV7roadmaprecoveryneeded.

W24original191files verifiedbefore/after; nooriginalartifactoverwrite. W25outputsprivate root
/home/frank/projects/MarketMamba/.artifacts/window25-robustness-audit/;same-hostrestoreisnotindependentbackup.
W19task/worktree/artifact/budget/runtime notoperated;StrictPhase0STOP/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED kept.
AfterlocalcommitSTOP;noW25merge/push/deploy/WindowsScheduler/GPU/nextwindow.

## H/I — Verification and recovery

10focusedtestsPASS;26main-postintegrationtestsPASS (no fit);independent4newtestsPASS(.010s).
One targeted independentreviewGO/Critical0Required0;0rework. Reviewer checked191W24inputpins,12W25pins/restorebytes,
independentlyrecomputed16bootstrapbounds,132LOSOsummarychecks/264dailycasechecks and44266samplecasetraces.
[Review](../research/v7/window25-review.json),[Restore](../research/v7/window25-independent-restore.json),
[Final evidence](../research/v7/window25-evidence.json). Source191W24filesunchangedafteranalysisandclosure.
Syntax/JSON/diff/hashchecksPASS;Governance/AuthorityDrift/LegacyDocsexistingWARNINGonly,structuralPASS.

Recovery:`cd /home/frank/projects/MarketMamba/.worktrees/v7-window25-robustness-audit`;readCurrent_State→
Authority_Map→solebacklogWindow25→report/matrix/classification/final-evidence;verifyHEAD/clean/inputpins.
AllanalysisoutputsreferencefixedW24bundleandseparateanalysisplanhash; nooriginalpolicyorperformancereplacement.
RuntimeW11CPU Python3.12.3/NumPy2.0.2/torch2.10.0cpu,CUDAempty,threads1;operationalpayloadnotaccessed.
Same-hostrestoreBACKUP_PENDING; newrequests/fit/nextwindowstillnotauthorized. STOPafterlocalcommit.
