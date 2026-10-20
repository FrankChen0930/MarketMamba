# Operations

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-10-20
- Authority topics: shadow ledger, resumable orchestration, incident/read-model boundaries
- Update triggers: Phase A review, merge, deployment, schema, or operational-policy change

## Purpose

Describe the target operational control plane and its current implementation/review boundary.
The 2026-10-20 review is bounded engineering evidence recovery, not Human approval.

## Current Boundary

Phase A builds a shadow operational ledger around existing workflows. It records runs, attempts, stages, artifacts, and checks with leases/fencing and transactional behavior. It does not own or redefine data-fetch success, model correctness, broker execution, or publication.

## Authority

Implementation lives on unmerged branch `feature/production-phase-a-shadow-ledger@f5b0ff3`, primarily under `V6/marketmamba/ops/`. The production architecture audit at `docs/production-system-architecture-audit@9efed7b` defines the roadmap.

## Inputs

Observed workflow events, stage outcomes, artifact identities, validation checks, lease ownership, and timestamps.

## Outputs

Append/transaction-backed operational records and read-only projections/CLI views.

## Current Implementation

Current update (2026-10-21): Human approved Task 4 limited engineering correctness
at `1e119ef`, bound to the verified historical evidence index and 12 raw records.
See `knowledge/02_History/Decisions/20261021-task4-human-approval.json`.
Task 5 is authorized for offline work in `integration/phase-a-task5`, not approved.
The following earlier observations remain historical; no deployment is implied.

Tasks 1–3 are completed and reviewed. Task 4 has implementation and sanitizer fixes, but correctness approval is not established. Task 5 has not started. The branch is unmerged, not deployed, and requires human review.

The isolated follow-up `fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
fixes Task 4 artifact aggregation when multiple stages reference one identity.
Original `f5b0ff3` remains unchanged. Offline ops 70/70 and mocked fetch-only 3/3
pass; independent engineering review has no Required finding. The candidate's
`research/production-system/phase-a/task4-evidence-recovery.json` indexes original
spec/report hashes, RED/GREEN evidence and remaining gates. Not merged or deployed.

## Evidence Classes

- Ledger transaction and invariant tests: component evidence.
- Review approval: merge-readiness evidence.
- Deployment/runtime observation: operational evidence, currently absent for Phase A.

## Current vs Legacy / Research

Existing scripts own their actual work. Phase A observes and records rather than replacing them. Notification policy, incident workflow, and atomic publication remain later roadmap concerns.

## Invariants

- Run/attempt/stage identities are stable and idempotent.
- Lease and fencing prevent stale writers from claiming ownership.
- Artifact/check provenance is recorded transactionally.
- Projections are read-only derivations.
- Shadow status never implies production deployment.

## Known Limitations

Task 4 lacks established correctness approval; Task 5, production notifications/incidents, and atomic publication are incomplete.

At the earlier `46df7b3` checkpoint, the sequential rebuild test did not test
an uncommitted concurrent writer or a commit midway through reading a projection.
SQLite BEGIN supports snapshot consistency in the implementation, but that is
not a direct test result in that original test. Sanitizer cases pass; Human correctness review remains
pending. Task 5 must supply process crash/restart/replay and overhead evidence;
thread ordering and injected replace failure are not substitutes. The projection
scans historical artifact identities and watermark tables; long-ledger cost is
not yet benchmarked. No Task 5 work is authorized by this note.

Follow-up `1e119ef855f8fd319f774162beca0bd729d4b921` on the same isolated
Task 4 branch closes those two evidence gaps: five real-process tests, 12
scenarios including an explicit no-BEGIN negative control. Full ops 75/75 and
mocked fetch regression 3/3 pass; independent isolation review/test run passed,
no Required/Critical finding. Production code unchanged. Raw before/during/after
states, identities, counts, process exits and digests are indexed in the existing
Task 4 evidence package. Human correctness decision is now ready for review,
not granted. SIGKILL evidence is limited to an uncommitted ledger writer, not
publication recovery or the Task 5 full replay program.

## Do Not Use / Do Not Mix

- Do not treat ledger success as proof that an upstream fetch/model/trade is correct.
- Do not deploy or merge the Phase A branch based on this note.
- Do not let a projection mutate ledger state.

## Important Paths

- `feature/production-phase-a-shadow-ledger@f5b0ff3:V6/marketmamba/ops/`
- `docs/production-system-architecture-audit@9efed7b`
- [Current State](../00_Project_Map/Current_State.md)

## Related Domains

[Production](Production.md), [Training](Training.md), [Portfolio](Portfolio.md), [UI](UI.md)

## Update Triggers

Update after review approval, branch merge, schema or lease-policy change, Task 5 implementation, or any deployment.
