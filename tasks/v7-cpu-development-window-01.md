# V7 CPU-only development window 01

Status: WINDOW_COMPLETE, UNMERGED / NOT_DEPLOYED. Human provenance: current conversation
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
| Next window: quote normalization collision refusal | READY, reproduced, NOT FIXED | Journal maps `A` and ` A ` to one key and accepts the last quote; engine already rejects normalized duplicates. Existing contract suffices. |
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
- U2 RED: 6 boundary tests, 2 failures (new semantic refusal and extension of
  an existing semantically invalid chain both changed bytes), exit 1.
  GREEN: all 38 portfolio tests PASS, 0.305s. Shared prefix reducer now validates
  the proposed chain under the append lock before writing, and returns that
  prefix without a second unlocked read. Historical corrupt journals remain
  refused and unchanged, never auto-repaired. One implementation repair;
  independent review passed: no Required/Critical; independent 38 tests PASS
  (0.294s), and both RED regressions independently reproduced on `585ad65`.
  U3 remains READY; shared reader locking is not yet certified.
- U3 COMPLETE: RED reader returned “truncated final line” during a writer's
  locked partial append. Shared-lock read fixes this. Five real spawn-process
  cases PASS (0.752s): duplicate/conflict, reader/writer, fsync→SIGKILL→new-process
  retry, SIGKILL partial tail refusal. Independent combined 43 PASS (1.062s),
  no Required/Critical findings. One initial repair, zero review rework.
- Final bounded regression: 53 portfolio/process/health/audit, 47 PIT and
  23 governance PASS. [Evidence index](../research/v7/cpu-window-01-evidence.json)
  binds raw logs and changed source hashes. No full repository/training claim.
  Protected Phase 0 evidence/report and E1 contract remain byte-identical to
  main; read-only preserved legacy audit still STOP with six blockers.
- Review skill influenced delivery: different-model independent testing and
  RED confirmation against prior commits. No additional Human interruption.
- Known remaining defect: a disposable journal accepts quotes keyed `A` at
  price10 and ` A ` at price20 with the same ticker; stores only A at price20.
  Next window should add rejection/no-write RED coverage and match the engine's
  duplicate-refusal contract. No new research decision needed. No blanket
  malformed-input or production certification is claimed.

## Resume / next window

Resume this branch at the commit containing the evidence index. Verify clean
worktree, main still `83a6a2e`, source/log digests and STOP first.
U1 `585ad65`, U2 `f42aef0`, U3 containing this checkpoint are local only.
Reassess the reproduced READY quote collision first. Do not repeat unchanged
E1 discovery without a new location. E1 replay, strict executable labels and
new research hypotheses remain separately blocked/gated. E1 recovery `11ae71d`
is preserved as ancestry, not merged into main. Stop after three units.
