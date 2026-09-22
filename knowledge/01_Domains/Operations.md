# Operations

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: shadow ledger, resumable orchestration, incident/read-model boundaries
- Update triggers: Phase A review, merge, deployment, schema, or operational-policy change

## Purpose

Describe the target operational control plane and its current implementation/review boundary.

## Current Boundary

Phase A builds a shadow operational ledger around existing workflows. It records runs, attempts, stages, artifacts, and checks with leases/fencing and transactional behavior. It does not own or redefine data-fetch success, model correctness, broker execution, or publication.

## Authority

Implementation lives on unmerged branch `feature/production-phase-a-shadow-ledger@f5b0ff3`, primarily under `V6/marketmamba/ops/`. The production architecture audit at `docs/production-system-architecture-audit@9efed7b` defines the roadmap.

## Inputs

Observed workflow events, stage outcomes, artifact identities, validation checks, lease ownership, and timestamps.

## Outputs

Append/transaction-backed operational records and read-only projections/CLI views.

## Current Implementation

Tasks 1–3 are completed and reviewed. Task 4 has implementation and sanitizer fixes, but correctness approval is not established. Task 5 has not started. The branch is unmerged, not deployed, and requires human review.

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
