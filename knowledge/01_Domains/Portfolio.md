# Portfolio

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: portfolio simulation boundary, deterministic accounting, legacy versus V7 paths
- Update triggers: portfolio contract, cost model, broker integration, or promotion change

## Purpose

Describe portfolio construction/accounting capabilities while preserving the distinction between legacy operation, V7 research simulation, and live execution.

## Current Boundary

`V6/v62_portfolio.py` belongs to the legacy V6.2 operational path. The V7 portfolio contract, engine, and journal are deterministic research components supporting targets, pending orders, fills, partial fills, costs, actions, NAV, hash-chain integrity, and replay. They are not a live broker.

## Authority

Current checked-in V7 portfolio code and tests define research behavior. Promotion or production ownership requires a separate explicit decision.

## Inputs

Scores, eligible universe, portfolio policy, prices/execution events, corporate actions, costs, and prior journal state.

## Outputs

Deterministic target positions, orders/fills, holdings/cash/NAV transitions, and auditable journal records.

## Current Implementation

The research engine models partial execution and deterministic state transitions with replayable hash-linked records. No active scheduler, broker adapter, or verified live-paper loop is established for V7.

## Evidence Classes

- Passing deterministic tests: component correctness evidence.
- Replay/hash-chain match: state-integrity evidence.
- Backtest result: research evidence only.
- Broker acknowledgement: would be external execution evidence; none is currently authoritative here.

## Current vs Legacy / Research

V6.2 portfolio behavior is legacy and may use different data/signal contracts. V7 portfolio components are research infrastructure and must not inherit production status from V6.2.

## Invariants

- Every state transition must be reproducible from prior state and events.
- Pending, partial, rejected, and filled quantities remain distinct.
- Costs and corporate actions are explicit inputs.
- Research portfolio output is not an execution confirmation.

## Known Limitations

No authoritative V7 broker integration, active scheduler, or live deployment has been established.

## Do Not Use / Do Not Mix

- Do not combine V6.2 portfolio state with V7 journals without migration.
- Do not call simulated fills live fills.
- Do not publish a V7 portfolio from strict labels while Phase 0 remains `STOP`.

## Important Paths

- Legacy portfolio: `V6/v62_portfolio.py`
- V7 contract: `V6/experimental/v7_portfolio_contract.py`
- V7 engine: `V6/experimental/v7_portfolio_engine.py`
- V7 journal: `V6/experimental/v7_portfolio_journal.py`

## Related Domains

[Models](Models.md), [Labels](Labels.md), [Execution](Execution.md), [Operations](Operations.md), [Production](Production.md)

## Update Triggers

Update when portfolio policy, cost/action semantics, journal schema, orchestration ownership, broker connection, or promotion status changes.
