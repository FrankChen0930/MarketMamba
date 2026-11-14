# V7 Strategic Roadmap Recovery — 2026-10-21

Status: REVIEWED_CANDIDATE / UNMERGED / NOT_DEPLOYED. This is a dated evidence
report, not a second active roadmap. The single current planning/backlog owner is
[research-program.md](../research/v7/research-program.md#current-development-program--2026-10-21).
Frozen machine contracts and Human decisions retain all existing authority.

## 1. Scope, Git and integration recommendation

Human request: “Strategic Roadmap Recovery & Autonomous Development Planning”.
Authorized inventory, isolated candidate, non-training validation, bounded fixes,
independent review, local commits and canonical memory updates only. Future program
implementation requires a new one-time scope approval. No training, main merge,
push, deploy, production changes or MAS Core live execution occurred.

Observed main `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` and CPU branch
`4c388ded57062194114ea3e221d00c177eb96892`, both clean at entry. Merge-base equals
main; ancestry is a single continuous sequence:

```text
83a6a2e main
  → 11ae71d E1 recovery inventory/report (documentation/evidence dependency)
  → 585ad65 non-due signal preserves completed portfolio state
  → f42aef0 semantic transition validation before durable journal append
  → 7eb9cd9 shared-lock readers and real-process recovery tests
  → 735ae6f normalized quote collision rejection
  → cd5c83a duplicate JSON member rejection
  → 4c388de window closure/verification limits
```

No cherry-picks or synthetic merge needed. New worktree
`.worktrees/v7-cpu-roadmap`, branch `integration/v7-cpu-roadmap`, starts exactly at
`4c388de`. Existing CPU tree is a completed-window checkpoint; Phase A trees serve
separate historical approvals. The new candidate preserves those identities while
isolating the requested integration/planning closure. It creates no new data tree.
All five source/test files remain byte-identical to `4c388de`; new commits only
contain planning, knowledge and evidence. Integrating the full ancestry also
imports `11ae71d` recovery evidence and window documentation, not only five fixes.

**Agent recommendation: GO for the local development candidate**, subject to
separate Human approval of the final exact candidate SHA. Critical/Required: 0.
No engineering repair needed. Phase 0 STOP and E1 BLOCKED remain unchanged.

| Joint verification | Newly observed result |
|---|---|
| Portfolio contract/engine/journal/process + health/audit | 60 PASS, 2.468 s |
| PIT/snapshot/provenance/universe/labels | 47 PASS, 1.172 s |
| Operations Phase A | 81 PASS, 16.746 s |
| Mocked fetch-only + governance | 26 PASS = 3 + 23, 0.075 s |
| Historical output-contract/diagnostic fixtures | 6 PASS, 0.003 s; no new research admission |
| Independent reviewer | 60 + 47 + 23 PASS; no Critical/Required |
| Old main writer → candidate replay/retry | Three records; identical state/bytes; SHA `f2b40d1b9bcb49d0463bb35e80fef93a595c53f5b81b1696895986e8026636fd` |
| Syntax/source identity | Five source/test ASTs and fixed-source bytes checked |

Evidence and commands: [candidate index](../research/v7/cpu-integration-candidate-20261021.json).
Initial Operations discovery missed `PYTHONPATH=V6` (7 import errors); historical
stability tests missed `PYTHONPATH=V6/experimental` (2 import errors). Corrected
invocations passed; source was not changed to compensate. Raw attempts are retained.
No GPU, model deserialization/training or formal data was used. Python 3.12.3,
existing recovery venv for PIT. MARKETMAMBA_DATA_ROOT unset. Candidate Data is absent;
main Data/processed_v6 points to D: operational data and was not read or changed.

Full-range diff check retains two known raw RED-log trailing-space warnings from
window 02; excluding just those two immutable evidence files passes. Current
planning changes must pass ordinary diff check. No power-loss, hostile filesystem,
network filesystem or full-model-suite guarantee. Remote tracking not refreshed.

## 2. Objective sources and authority distinctions

Source IDs below are used by the current backlog. `4c388de:path` pins the recovered
pre-planning source; new planning text does not rewrite these historical bytes.
Human intent recorded in older prose is evidence of product scope, not a fresh
execution grant or cryptographically signed transcript.

| ID | Precise source | What it can establish |
|---|---|---|
| S1 | `4c388de:research/v7/product-architecture-alignment.md`, §1/7 | Recorded Human-confirmed full-market research, daily 5d/10d rankings, weight-based simulated portfolio and performance; no personal account/capital/broker orders |
| S2 | `4c388de:tasks/plan.md`, P0/P2/P3/P4/P5 | Historical functional decomposition; cannot restart old work automatically |
| S3 | `4c388de:docs/superpowers/specs/2026-09-16-v7-replayable-portfolio-engine-design.md` | Approved deterministic engine/journal, costs, explicit execution and recovery contract |
| S4 | `4c388de:docs/superpowers/specs/2026-09-16-v7-signal-to-portfolio-validation-design.md` and corresponding tasks | Approved correctness audit, conditional anatomy/portfolio/slicing/report; STOP forbids implementing/running gated Phase 1–5 |
| S5 | `e3bce91:research/v7/corrected-baseline-v1/{policy,feature-manifest,label-manifest,universe-manifest}.json` | Corrected data/PIT/48-feature/next-open/no-roll/universe semantics |
| S6 | `e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` | Simulation PASS versus strict STOP, zero verified labels, no Phase 1–5 start |
| S7 | `044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json` and E1 `experiment-contract.json` | Corrected 64/32 architecture, no graph/industry neutralization, fixed training and E1 comparison |
| S8 | `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` | Completed two-seed E1 research; portfolio_admission=false |
| S9 | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json` | Month sensitivity, unresolved final-support epoch/regime inference; no new training yet |
| S10 | `11ae71d:reports/v7_e1_evidence_recovery.md` | Revalidated retained artifact identities, matrix absent locally; no new matrix substitution |
| S11 | `4c388de:research/production-system/implementation-roadmap.md`; `89f9496:research/production-system/phase-a/` | Phase A–F architecture; only Phase A implemented/integrated; later phases require scope/production decisions |
| S12 | `4c388de:research/v7/system-observability-foundation-review-20260916.md`; `app/backend/routers/v7.py`, `app/frontend/src/pages/V7Status.jsx` | Status UI/source and historical browser verification; MAS Builder/Verifier NOT completed; deployment not proven |
| S13 | `4c388de:research/v7/research-program.md`; `experiment-hypotheses.md` | Causal experiment discipline, Advisor/Builder/Verifier/Archivist roles and Human architecture/compute gates; hypothesis catalog is not approved execution |
| S14 | `4c388de:research/v7/historical-notes-register-20260922.md` | Explicit historical status of tasks/plan/todo, output contract and product prose; branch variants preserved |
| S15 | `83a6a2e:knowledge/02_History/Decisions/20261021-task4-human-approval.json` and Task5 approval JSON | Scoped Human offline engineering approvals, not research/production promotion |

V7's recovered objective is a traceable Taiwan-equity research and simulated
portfolio system: trustworthy time-aware inputs → reproducible ranked model
signals → explicit portfolio rules/execution assumptions → auditable state and
performance → useful read-only presentation and eventual separately admitted
forward operation. Current product intent is broader than E1 matrix recovery.
It does not approve every desirable risk policy, dashboard or model experiment.

## 3. Implementation versus end-to-end completion

`IMPLEMENTED_VERIFIED` below is scoped to listed component tests or admitted
historical artifacts; it never means live deployment or all conceivable inputs.

| Domain / objective | Implementation status | Evidence and real gap |
|---|---|---|
| Data snapshots/PIT/provenance | IMPLEMENTED_VERIFIED components; PARTIAL end-to-end | Six adopted primitives and 47 tests. Mutable operational cache is not PIT. Provider-wide certified acquisition not delivered (S5/S6/S11). |
| Universe/labels/execution | PARTIAL; strict BLOCKED | Lifecycle and no-roll materializers exist. 1,975-stock v1 universe; simulated v1 labels 6,889,229/6,790,554; strict 5d/10d=0. Membership never proves tradability. |
| Features/matrix | IMPLEMENTED_VERIFIED contract; replay BLOCKED | Ordered 48 features; causal scaling; no graph/industry backfill. Some older manifest status text predates completed E1; don't read it as no matrix ever existed. Exact E1 frozen matrix currently unavailable. |
| Model/training | IMPLEMENTED_VERIFIED for fixed research lineage; PARTIAL ensemble | Formal corrected seeds 17/29, seed43 excluded; E1 completed, PASS_REFRESH_HYPOTHESIS. Not production model selection or final V7 architecture. No new training run this audit. |
| Prediction row/rank/diagnostic utilities | IMPLEMENTED_VERIFIED bounded fixtures; PARTIAL corrected integration | `v7_stability_contract.py`, `v7_stability_diagnostics.py` plus 6 tests; E1 evaluator separately exists. Three-seed historical schema and corrected two-seed outputs are not automatically interchangeable. |
| Real daily V7 ranking delivery | NOT_STARTED as admitted end-to-end slice | Historical UI route names and V6 pages do not establish V7 ranking API/admission; model/data/publication adapter spec missing. |
| Portfolio construction/accounting | IMPLEMENTED_VERIFIED components | S3 spec, 60 combined tests including five fixes. Equal weights/fractional quantities, deterministic ties, buffer, cadence, pending/partial fills, explicit actions, cash/NAV/cost. Real-signal strategy not selected. |
| Replay/reader/writer | IMPLEMENTED_VERIFIED local POSIX scope | Locked append/replay, duplicates/conflicts, SIGKILL retry/refusal; old valid format verified. No automatic corrupt-journal repair. |
| Simulation/backtest workflow | PARTIAL | Engine core exists; fixed historical research scan, event adapters, batch workflow/report provenance and approved real-signal admission not complete. A fixture inspector is a separate proposed engineering slice, not Phase 1–5. |
| Trading costs/constraints | IMPLEMENTED_VERIFIED components; PARTIAL research realism | Comparison fees 0.0015/0.0045, explicit status/fill ratios, cash limits. No calibrated slippage, capacity, lot sizing or broker model. Fees are assumptions, not legal/current broker claims. |
| Risk management/performance | PARTIAL; HUMAN_DECISION_REQUIRED for policies | NAV and total_cost exist; old RankIC/time grouping exists. Benchmark, turnover convention, drawdown duration/calendar, annualization, exposure limits, cash policy and liquidity stress are not frozen current V7 contracts. |
| Reproducibility/recovery | IMPLEMENTED_VERIFIED reusable primitives; PARTIAL artifact availability | Adoption register, lifecycle/checkpoint/bundle tests and admitted runs exist. Safe E1 payload validation/matrix retrieval still blocked; packaging alone is not formal completion. |
| Signal anatomy / portfolio grid / failure slicing | BLOCKED | S4 strict prerequisites STOP; historical E1 diagnostics don't supply portfolio admission. Do not start them under a renamed fixture task. |
| Operations | IMPLEMENTED_VERIFIED Phase A; NOT_STARTED B–F as complete systems | Main Phase A 81 tests + 3 fetch-only. Snapshot primitives cover part of B but registry/certifier/migration are not complete. No new scheduler/deploy. |
| UI/API | IMPLEMENTED_VERIFIED status slice historically; IMPLEMENTED_UNVERIFIED live state | V7 status only; historical local build/browser evidence exists. This audit did not rerun browser/deployment. Ranking/portfolio/performance screens NOT_STARTED for admitted V7 products. |
| MAS | PARTIAL Advisory history; BLOCKED Core Live | Historical Advisor output, failed linked-worktree Builder path and Codex takeover. No current live certification or new routing/state query. |

Conflicts resolved: old 59-feature/graph/three-seed prose cannot supersede S5–S8;
old next-experiment `SPECIFIED_NOT_EXECUTED` is pre-run history, superseded by S8
for completion only; old Phase A UNMERGED/Task5 NOT_STARTED paragraphs are historical
and superseded by explicit main closure/S15. Neither a source file nor a green
engineering suite implies research correctness or deployment.

## 4. Dependency analysis

- **GPU/new training:** new frozen hypothesis, architecture/loss/recipe/target
  experiment or seed43 formal rerun. Requires Human research+compute authorization.
  The old H0–H12 catalog remains hypotheses, not a scheduled training queue.
- **Existing artifacts, no new training:** E1 retained-checkpoint inspection/replay
  and fixed prediction comparison. Need original matrix `a590b280...947f62`, safe
  loading, identity matching and support rules. CPU inference feasibility must be
  measured only after admission; do not promise GPU-free full Mamba replay.
- **Pure CPU/fixtures:** existing component validation, replay/accounting, proposed
  journal exporter/report/read-only view. No model artifacts needed. Snapshot/PIT
  synthetic tests do not create historical evidence.
- **Data/evidence:** direct executable-open/tradability/publication/lifecycle gaps.
  GPU cannot create authentic missing history; do not buy/fetch substitute data
  or rebuild a differently identified matrix to make recovery pass.
- **Human research decisions:** corrected real-prediction adapter/admission,
  ex-ante regime definitions, benchmark/slippage/risk/portfolio comparison rules,
  next experiment and promotion. Choosing best N/head from results is not engineering.
- **Operations authority:** live data migration, provider licensing, notifications,
  scheduler, deployment, main merge and forward paper require their separate gates.

## 5. Guarded integration procedure — not executed

After Human approves an exact full candidate SHA (the final checkpoint in this
handoff), record it as `approved_sha` rather than trusting a moving branch name.
Confirm main is still `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e`, main/candidate
clean, candidate contains `4c388de`, source/log hashes match this index, and no
production/research contract diff appeared. If main moved, STOP and re-review
against the new base; do not cherry-pick or resolve silently.

Rerun the index's bounded commands on the exact approved candidate when approval
arrives after new changes/runtime drift. Human authorization must explicitly cover
local main integration of the inherited E1 evidence and planning documents too.
Only then, from main: `git merge --ff-only <approved_full_sha>`; post-check exact
HEAD, hashes, governance, relevant tests and clean status. Retain all source
branches/worktrees; no automatic push/deploy or worktree deletion. A GO
recommendation is not permission to execute this command.

## 6. Program conclusion

Existing functionality is sufficient to build a useful offline portfolio
inspection workflow without training. It is insufficient to declare a fully
specified READY real-data backtesting/ranking product. The next program should
first receive one bounded approval for the proposed fixture-only interface in the
single backlog. Broader risk/performance/research admission remains gated.

Completed this audit: candidate validation, full objective/status/dependency
recovery, persistent backlog and reviewable program proposal. Changed: documents
and evidence only after candidate creation. Remaining: Human program/integration
decisions. Blockers: data/research gates above. Project memory/Authority Map updated
in same change; no merge/push/deploy. Final Git identity is the containing closure
commit; original five-fix source identity remains `4c388de`.

Final verification after memory updates: governance 23 PASS (0.070s); knowledge
health WARNING (Governance/Authority Drift/Legacy Docs), no structural FAIL.
37 source-reference hashes, 5 source hashes/ASTs, 19 historical unit hashes and
3 protected main artifacts PASS. Original research-program text retained verbatim
as historical suffix. Final navigation reviewer: Critical 0 / Required 0; one
active queue across Markdown/JSON map and domain pointers. New changes diff check
PASS; inherited raw RED whitespace limit remains explicit. Additional Human
interventions: 0. Goal API observed the correct V7 objective; counters are recorded
in index as telemetry, not exact cross-agent aggregate. No future autonomous
continuation reliability certification. Stop after local commit, awaiting H1.
