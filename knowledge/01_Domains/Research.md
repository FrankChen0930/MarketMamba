# Research

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
