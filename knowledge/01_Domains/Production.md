# Production

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: observed scheduling, production boundary, deployment status
- Update triggers: scheduler observation, deployment, runtime ownership, or promotion change

## Purpose

State what is actually operating today and keep that separate from available code and target architecture.

## Current Boundary

The latest recorded observation (2026-09-20 01:30 Asia/Taipei) found the Windows task `MarketMamba_DataFetch` active and invoking `V6/scripts/v62_daily.bat --fetch-only`. The full V6.2 daily flow exists for manual use but its full schedule is disabled. V7 remains research/shadow work, not a promoted production trading system.

## Authority

Current state and runtime observations are indexed by [Current State](../00_Project_Map/Current_State.md) and [Authority Map](../00_Project_Map/Authority_Map.md). The production audit at `docs/production-system-architecture-audit@9efed7b` defines target direction, not deployed fact.

## Inputs

Scheduled fetch configuration, provider data, runtime environment, and explicit promotion/deployment records.

## Outputs

Fetched data and, only in manually invoked legacy flows, downstream V6.2 processing. No V7 production publication is established.

## Current Implementation

- Observed active: fetch-only task.
- Available but not actively scheduled: full V6.2 daily pipeline in `V6/run_v62_daily.py`.
- Research/shadow only: Phase A ledger work and corrected V7 experiments.
- API/UI deployment: `UNKNOWN`.

Runtime observations have a seven-day TTL and expire on 2026-09-27 01:30 Asia/Taipei. After that, readers must emit `STALE_OBSERVATION` rather than assume the scheduler is unchanged.

## Evidence Classes

- Timestamped host observation within TTL: current operational evidence.
- Code/config present: capability, not proof it is running.
- Audit/roadmap: target state, not deployment.

## Current vs Legacy / Research

V6.2 is the legacy operational code path. Corrected V7, E1, and Phase A are not deployed. Phase A work is unmerged and explicitly shadow-only.

## Invariants

- Never infer running/deployed state from source code alone.
- Preserve fetch-only versus full-pipeline distinction.
- Promotion requires explicit evidence and review.
- Expired observations become warnings, not silently refreshed claims.

## Known Limitations

Live API/UI status, service ownership, and external scheduler state are unknown beyond recorded observations.

## Do Not Use / Do Not Mix

- Do not describe V7 research artifacts as production signals.
- Do not treat Phase A ledger events as broker execution.
- Do not infer active full-pipeline scheduling from the batch file’s existence.

## Important Paths

- [Current State](../00_Project_Map/Current_State.md)
- Legacy runner: `V6/run_v62_daily.py`
- Fetch wrapper: `V6/scripts/v62_daily.bat`
- Target audit: `docs/production-system-architecture-audit@9efed7b`
- Shadow implementation: `feature/production-phase-a-shadow-ledger@f5b0ff3:V6/marketmamba/ops/`

## Related Domains

[Operations](Operations.md), [Portfolio](Portfolio.md), [UI](UI.md), [Research](Research.md)

## Update Triggers

Update after a fresh host observation, scheduler change, deployment, model promotion, or change in API/UI service ownership.
