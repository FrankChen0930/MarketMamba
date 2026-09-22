# MarketMamba K1 Domain Knowledge Bootstrap

## Outcome

K1 now implements the agreed five-layer knowledge model:

1. `AGENTS.md` / `CLAUDE.md`: stable working policy.
2. `knowledge/00_Project_Map/`: current state and authority map.
3. `knowledge/01_Domains/`: subsystem semantics and navigation.
4. Machine contracts, tests, code, and artifacts: binding technical truth.
5. `knowledge/02_History/` / Obsidian: historical rationale.

This work changes documentation, navigation, and validation only. It does not change data, features, labels, models, training behavior, portfolio behavior, runtime scheduling, APIs, frontend behavior, or production authority.

## Delivered Domains

Tier A:

- Data
- Features
- Models
- Training
- Labels
- Production

Tier B:

- Universe
- Execution
- Portfolio
- Operations
- UI

Tier C navigation:

- Research

Each note records status, ownership, review date, purpose, boundary, authority, inputs/outputs, implementation state, evidence classes, legacy/current separation, invariants, limitations, prohibited mixing, important paths, dependencies, and update triggers.

## Protected Conclusions Preserved

- Corrected baseline: `E5-PIT-Clean-v1`, 48 ordered features, no explicit graph, no industry neutralization.
- Historical-simulation readiness: `PASS`.
- Strict Phase 0: `STOP`.
- Strict verified executable labels: 5d=`0`, 10d=`0`.
- Historical simulated labels v1: 5d=`6,889,229`, 10d=`6,790,554`.
- Corrected E5 formal seeds: 17 and 29; seed 43 remains incomplete and excluded.
- Diagnostic decision: `STOP_FOR_TRADING_CONTINUE_RESEARCH`; 2026 head correlation about `0.9983` is diagnostic only.
- E1: specified/packaged, not verified complete.
- Production observation: fetch-only active at the recorded timestamp; full V6.2 schedule disabled; API/UI deployment unknown.
- Phase A: unmerged/not deployed; Tasks 1–3 reviewed, Task 4 approval not established, Task 5 not started.
- Legacy A/B/C artifacts remain historical and unchanged.

## Generated Navigation and Validation

- Tier A index, dependency map, important-path inventory, coverage result, and new-knowledge findings.
- Deterministic domain validator for required schema, local links, branch-qualified evidence, coverage, and protected semantics.
- Drift checker for pinned branch tips and seven-day external-observation freshness.
- Current dated drift output is `PASS`.
- A post-expiry test returns `WARNING` with `STALE_OBSERVATION`, while structural or branch-pin drift fails.

## New Findings

Three existing uncertainties are now explicit navigation findings rather than implied facts:

1. Corrected E5 runtime telemetry is incomplete (matrix wall time, epoch timing, GPU utilization, and durable-storage I/O).
2. A historical universe listing-start case remains unresolved and must stay fail-closed.
3. API/frontend deployment and service ownership remain unverified.

These findings do not change a gate or promote any artifact.

## Validation

The final verification runs:

- project-map validator
- agent-guide validator
- domain validator
- current-date drift check
- expired-observation warning check
- Python syntax compilation
- whitespace/diff check
- scope audit against the K1 base commit

No push, merge, deploy, GPU execution, E1 execution, or production mutation was performed.

## Commits

- `b13698e` — K1 domain bootstrap plan and checklist.
- `ac65660` — Tier A domain knowledge checkpoint.
- `935eccd` — Tier B/C, navigation, domain validator, and drift checker.
- Final report/checklist commit follows this report in branch history.
