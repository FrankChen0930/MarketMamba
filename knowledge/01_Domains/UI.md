# UI

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: API/UI code boundary, deployment uncertainty, publication safety
- Update triggers: API/frontend contract, deployment observation, or publication-path change

## Purpose

Map the user-facing code and prevent source presence or local candidates from being mistaken for deployed product behavior.

## Current Boundary

The repository contains a FastAPI backend at `app/backend/main.py` and a React frontend rooted at `app/frontend/src/App.jsx`. Their live deployment and current service ownership are `UNKNOWN`.

## Authority

Checked-in code establishes capability. Deployment requires timestamped runtime or platform evidence. Untracked local V7 UI candidates are `UNMERGED_CANDIDATE`, not project truth.

## Inputs

Published model/portfolio artifacts, backend configuration, API contracts, and deployment configuration.

## Outputs

API responses and user-facing visualizations only when a verified publication/deployment path exists.

## Current Implementation

The current tree exposes legacy application entry points. The production audit proposes atomic publication and a V7 paper/shadow surface, but no deployed V7 UI is established.

## Evidence Classes

- Checked-in route/component: implementation capability.
- Contract/integration test: interface evidence.
- Timestamped endpoint/platform observation: deployment evidence.
- Untracked local file: candidate only.

## Current vs Legacy / Research

Existing app code must not automatically display corrected V7 research output. A future V7 paper UI needs explicit artifact admission, atomic publication, freshness, and failure-state semantics.

## Invariants

- Displayed evidence class and freshness must be explicit.
- Partial publication must not appear complete.
- UI state must not silently promote research artifacts.
- Deployment claims require runtime evidence.

## Known Limitations

API/UI live status, hosting, authentication, freshness behavior, and V7 integration are unknown from current authoritative evidence.

## Do Not Use / Do Not Mix

- Do not cite untracked local candidates as merged functionality.
- Do not expose simulated labels as verified/tradable signals.
- Do not infer deployment from local dev configuration.

## Important Paths

- Backend: `app/backend/main.py`
- Frontend: `app/frontend/src/App.jsx`
- Target architecture: `docs/production-system-architecture-audit@9efed7b`

## Related Domains

[Production](Production.md), [Operations](Operations.md), [Portfolio](Portfolio.md), [Research](Research.md)

## Update Triggers

Update after route/schema changes, merged V7 UI work, deployment verification, or publication/freshness contract changes.
