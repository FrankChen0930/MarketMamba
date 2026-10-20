# V7 CPU-only development window 01

Status: ACTIVE, UNMERGED / NOT_DEPLOYED. Human provenance: current conversation
“Autonomous CPU-Only Development Program”, authorizing bounded engineering,
not new research contracts. Main remains `83a6a2e`; this isolated branch
`development/v7-cpu-window-01` starts at `11ae71d` (E1 evidence-only successor).
The runtime clock reported `2026-10-20T17:03:59Z` (Taipei 2026-10-21).

## Objective / roadmap / ranked backlog

Human long-term objective: credible Taiwan-equity research, correct data,
prediction and simulated portfolios. Current accepted research is corrected
E5/E1, 48 features, no graph, explicit historical-simulation evidence.
Strict Phase 0 remains STOP. Sources: canonical Research, Labels, Execution,
Portfolio notes and Authority Map; fixed E1 contract `044eef7`, result `eec63c5`.
The older integrated candidate/59-feature draft is historical, not roadmap.

| Priority / item | Initial status | Basis and decision |
|---|---|---|
| E1 historical frozen matrix recovery/replay | BLOCKED | `11ae71d:reports/v7_e1_evidence_recovery.md`; missing identity-matched matrix; no new location supplied. Do not repeat unchanged search or generate a replacement. |
| Strict executable label evidence | BLOCKED | `e3bce91` readiness: zero strict labels, direct daily tradability/open evidence missing. No proxy promotion. |
| U1: completed-rebalance early signal | READY to reproduce | Approved replayable engine spec/plan; `apply_signal` assumes pending exists when not due, although completed rebalance clears it. Preserve scheduling and state. |
| U2: reject invalid journal transitions before persistence | READY after U1 | Approved journal fail-closed/replay contract; append currently persists before engine validation. Preserve hash chain and bytes on semantic refusal. |
| U3: process concurrency and interrupted retry evidence | READY after U2 | Existing locked journal and idempotence contract; existing tests are sequential. Real processes, deterministic synchronization, no timing sleeps. |
| Signal anatomy / portfolio response / failure slicing | HUMAN_DECISION_REQUIRED (gate blocked) | Existing signal-to-portfolio design explicitly gates Phase 1–5 on correctness PASS; no automatic admission of E1 signals. |
| New E2 hypothesis / ex-ante regime variables | HUMAN_DECISION_REQUIRED | Agent candidate only; no frozen next-experiment contract. Do not implement a research direction. |
| New training or Colab rerun | BLOCKED | Requires explicit Human authorization and runtime; not needed for U1–U3. |
| Production Phase B–F rollout | HUMAN_DECISION_REQUIRED | Existing production roadmap, not authorized by this CPU window. No operational edits. |

Selection prioritizes repeatable state/cost integrity and recovery over cosmetic
work. Portfolio infrastructure tests use explicit synthetic hand fixtures only;
they do not establish returns, tradability, research correctness or forward paper
readiness. Existing approved spec:
`docs/superpowers/specs/2026-09-16-v7-replayable-portfolio-engine-design.md`.
Existing implementation plan is completed history, not permission to redo it.

## Scope and acceptance

Write allowlist: `V6/experimental/v7_portfolio_engine.py`,
`v7_portfolio_engine_test.py`, `v7_portfolio_journal.py`,
`v7_portfolio_journal_test.py`, optional focused process test in the same
directory; this task file; affected `knowledge/00_Project_Map/` and
`knowledge/01_Domains/Portfolio.md`; bounded raw test evidence under existing
`research/v7/`. No production, data, models, dependencies or research contracts.

U1: early signals after completed fill are a non-mutating not-due result;
pending signals and exact frequency boundary retain original semantics.
U2: head mismatch/invalid transition is rejected before append; valid later
events still replay; result reflects the committed prefix, with no second
unlocked read. No automatic tail repair or overwrite.
U3: competing same-ID writers charge once, conflicting IDs/content refuse,
readers observe committed prefixes, durable append + killed caller + fresh
retry does not duplicate events/costs. Fault injection is labeled as such.

Each unit: failing reproduction where behavior changes, minimal fix, focused
and portfolio regression, independent review, at most two repair rounds,
memory checkpoint and local commit. Stop after these three units; reassess
remaining work without expanding this window. No merge/push/deploy/training.

## Runtime / accounting

Python 3.12.3 stdlib for portfolio tests. Development `Data/processed_v6`
resolves to Windows operational data and is NOT used. MARKETMAMBA_DATA_ROOT
unset at preflight. All ten original worktrees were clean.
New checkout needed to preserve the completed E1 evidence branch and all other
subsystem checkouts. No MAS Core live execution is used.

Goal API still holds a blocked historical MAS objective; create_goal refused
an unfinished goal. Do not mark that unrelated objective complete. Current
aggregate token usage is unavailable; no numerical hard-cap claim. Record
actual test counts/durations and bounded repair counts instead.

## Checkpoints

- Plan: three units selected; no research decision changed.
- U1 RED: focused test exits 1 with AssertionError at `apply_signal`'s
  `assert self.pending is not None` after a successful fill cleared pending.
  GREEN: portfolio contract/engine/journal suite 35 tests, 0.297s, exit 0.
  One implementation repair; no policy or frequency change. Independent review:
  no Required/Critical findings, independently 35 PASS (0.309s).
  Test uses frequency 2, checks both early sessions and exact due boundary.
  Knowledge health WARNING/no FAIL; Operations and Governance reviewed, no
  policy/runtime change requiring updates there. Next U2 remains READY:
  disposable real-journal probe confirms head mismatch is rejected only AFTER
  bytes change, and the resulting journal cannot replay.
