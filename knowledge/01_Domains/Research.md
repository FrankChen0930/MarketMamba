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
