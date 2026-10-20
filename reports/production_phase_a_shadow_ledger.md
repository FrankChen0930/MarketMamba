# Phase A Shadow Ledger — Task 5 correctness review package

Date: 2026-10-21. Candidate: `integration/phase-a-task5`.
Status: `OFFLINE_EVIDENCE_READY_HUMAN_REVIEW_PENDING`.
Strict Phase 0 remains **STOP**. Task 5 Human correctness is **NOT APPROVED**.

## Authority and integration

Human approved only Task 4 limited engineering correctness at
`1e119ef855f8fd319f774162beca0bd729d4b921`.
Before recording approval, the fixed Git index SHA-256
`ed254b828683a83035fe791117266d9ea9ed1a56ac477e9a4125dcbf9ff380ec`,
all 12 raw isolation records and 36 generation digests were revalidated.
The explicit Human statement and exclusions are retained in
`knowledge/02_History/Decisions/20261021-task4-human-approval.json`.
Agent transcription is not an independent Human signature.

The new worktree starts at Snapshot `63a040d306e402dbc22e8592f57f034fd4a9a863`.
Common ancestor with Task 4: `7bdf2e4e41a24d9eee30821162b9524227d2b859`.
Seventeen Phase A-only commits were cherry-picked with source provenance:
`9efed7b,014b368,38271ea,b2e126f,72087e7,59c1e46,edf9aff,8de4258,
be51427,f9e1559,a6d9fca,4422cba,05c2b90,e47ea59,f5b0ff3,46df7b3,1e119ef`.
Each destination commit retains its full source commit in Git history.
No unrelated research history or dirty checkout was imported.
Operations implementation matched Task 4; Snapshot implementation matched its
source. Joint regression before Task 5: 75 Operations, 47 PIT, 3 fetch-only,
18 governance tests passed. This is engineering integration evidence, not an
extension of the Human Task 4 approval.

## Architecture and evidence

Existing fetch callback → best-effort shadow adapter → transactional SQLite
ledger (logical run / attempts / stages / artifacts / checks / lease fencing)
→ atomic read-only projection. Legacy fetch result remains success authority.
No new orchestration or memory system was added.

`research/production-system/phase-a/schema.json` captures actual SQLite DDL and
migration checksums. `shadow-replay-results.json` contains raw records, a real
projection example, process exits and timing arrays. `task5-evidence-index.json`
binds these and the executed test source by SHA-256. `acceptance.json` is an
engineering assessment, explicitly not Human approval.

## Replay, interruption and concurrency

Ten fixed observed-versus-shadow cases cover clean success, missing source,
freshness failure, nonzero-result mapping, transient exception then success,
same-logical-run retry, hygiene warning, source-count mismatch, projection
failure, and missing artifact. Callback objects/exceptions are preserved;
SQLite integrity, artifact count, identity and projection are checked.
These are synthetic callback fixtures with real storage, not live production
or research data. Temporary files named parquet are stat-only fixtures.

Four additional records cover:

- Three real concurrent lease owners: acquired → conflict → released →
  successor with increasing fence. Pipe synchronization, no sleep-based timing.
- SIGKILL after durable RUNNING, exit `-9`, then a fresh process with explicit
  expired-lease clock. Same logical identity, new attempt, increasing fence,
  successful projection; old RUNNING stays historical rather than falsely
  becoming complete. Pre-existing unregistered artifact bytes remain identical.
- Real uncaught child failure: actual exit `1`, FAILED ledger and matching
  projection. This supplements, rather than disguises, mapped replay exits.
- Stale-fence rejection and active shadow-owner conflict: callback still returns
  its original result; the conflicting observer creates no shadow attempt.

The durable ledger is the checkpoint here. No Phase D automatic pipeline
resume is claimed. The orphan case does not prove a worker publication crash
window. File-stat identity is not content-digest certification. A fixture
hygiene log is not Phase C incident/diagnostic implementation.

## Verification and overhead

- Task 5: 6 tests passed, 4.139 s, including actual subprocesses.
- Full Operations (including Sanitizer, Aggregation and isolation): 81 passed,
  18.303 s.
- Joint PIT/Snapshot: 47 passed; fetch-only: 3 passed.
- Governance: 19 passed.
- Additional non-training portfolio/engine/journal/health regression: 38 passed.
- Full experimental suite intentionally not run: it contains training and
  backward/epoch execution forbidden by this authorization.

Ten paired callback/shadow timings are retained, including first-run migration
and subsequent same-ledger growth. Median added latency: **98.475144 ms**;
maximum shadow duration: **136.371430 ms**. The evidence run overlapped the
Operations regression on a shared host. This is reproducible methodology and
raw measurement, not an isolated-host SLA or production performance approval.
No performance threshold was invented. Provider/network and large-history
scalability are excluded.

## Review and remaining gates

Independent initial review found no Required/Critical findings. Its artifact
identity/count suggestion was implemented, and actual nonzero child evidence
was added. Work stayed within two implementation/review rounds; no production
correctness bug was found by this Task 5 work.
Final independent review approved delivery as offline evidence only: 6/6 replay
tests (3.978 s), 19/19 governance, project-map validation and diff checks passed;
all four indexed hashes and benchmark arithmetic were independently verified.
The reviewer did not repeat the parent's full regression suites.
One generated history-index synchronization omission was repaired during
Stage 1; final knowledge health has no FAIL (existing governance, authority
drift and legacy-documentation warnings remain). Knowledge-impact review
required the updated Operations/current-state/authority/register records;
production semantics and the governance hierarchy did not change.
No additional Human intervention was requested after the initiating decision.

Task 5 offline engineering evidence is ready for Human review. It cannot alone
prove research semantics, live provider behavior, production correctness,
whole-pipeline recovery or operational promotion. Phase B provider integration,
Phase C diagnostics and Phase D resumable orchestration remain separate work.
Main merge, deployment, training and research changes remain unauthorized.
Stop after this package; Human decides Task 5 correctness and later integration.
