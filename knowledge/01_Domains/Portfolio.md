# Portfolio

## P2 research accounting candidate — 2026-11-19

[Research engine](../../V6/research_lab/portfolio.py) 依 [P2 spec](../../tasks/spec-lowturnover-p2-20261119.md) 新增 hold_buffer_rank、min_fee、initial_capital。NAV、cash、order notional 都以初始 NAV=1 表示；notional × initial_capital 才是 TWD，不能再重複乘 NAV。緩衝只讀 signal-day eligible ranks，ties 以 code 排序；保留既有持股後按排名補缺。最低佣金逐策略委託收取，blocked／零委託不收；修訂 1 的 P2 基準只收比例成本，另列也收最低費用的 descriptive sensitivity，不能進確認性。P2全格明確啟用 preserve_frozen_overflow（含 buffer=top_k），P1預設關閉。凍結加選股數超過 top_k 時保留所有既有股數，不新增、不強制賣出可成交持股；terminal liquidation 不變。預設仍走原始比例成本路徑，另以 pre-P2 ace2eef 引擎的固定合成逐日報酬 hash 回歸。無整股／零股、券商或實際 execution 證據。UNMERGED / NOT_DEPLOYED；[tests](../../V6/research_lab/tests/test_p2.py) 與 [工程報告](../../reports/v7_p2_portfolio_engineering.md) 是 bounded 工程證據。

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

## H1 fixture history slice

Exporter `V6/experimental/v7_portfolio_history.py` captures raw bytes and records
from a single shared lock, replays each event once and exports exact snapshots.
Operator fixture manifest and hashes are provenance declarations, not research
authenticity/admission. Output is atomic, limited to temporary/task artifacts.
See [H1 decision](../02_History/Decisions/20261021-portfolio-history-h1.json) and
[unit evidence](../../research/v7/portfolio-history-evidence.json). Branch-local
UNMERGED / NOT_DEPLOYED; existing accounting and Strict STOP remain unchanged.

Read-only API: `app/backend/routers/v7_portfolio.py`; opt-in main registration,
loopback clients, explicit pinned SHA-256, no stale cache or write methods. Local
launcher `app/backend/portfolio_history_local.py` reuses this router without legacy
startup. [Launch/evidence report](../../reports/v7_portfolio_history_acceptance.md).

## Authority

Current checked-in V7 portfolio code and tests define research behavior. Promotion or production ownership requires a separate explicit decision.

## Inputs

Scores, eligible universe, portfolio policy, prices/execution events, corporate actions, costs, and prior journal state.

## Outputs

Deterministic target positions, orders/fills, holdings/cash/NAV transitions, and auditable journal records.

## Current Implementation

CPU window 01, isolated `development/v7-cpu-window-01` (UNMERGED /
NOT_DEPLOYED): reproduced an AssertionError for a valid early signal after
a completed rebalance. The candidate returns a non-mutating not-due result
with deterministically ordered current holdings when no pending target exists.
The pending-target behavior and exact frequency boundary are preserved.
See [bounded plan/evidence](../../tasks/v7-cpu-development-window-01.md).
This is synthetic component engineering, not signal/portfolio research admission.
The same candidate also rejects invalid semantic journal transitions before
persistence, preserving existing bytes and replayability. Append/duplicate
responses derive from their locked prefix, not a later unlocked read.
Shared-lock reads now exclude in-progress append bytes. Five real-process
synthetic/fault-injection cases cover duplicate/conflict, concurrent reading,
durable-before-response SIGKILL retry and partial-tail refusal. No power-loss,
network-filesystem or uncooperative-writer guarantee is inferred.
Window 02 now closes that normalized quote collision with the same rejection
as the engine; unique whitespace keys and empty quotes remain legal. Local
60-test portfolio/process/health/audit regression passes, independent review
passes with no Critical/Required. Existing persisted
ambiguous/corrupt journals remain refused without repair. See the
[current checkpoint](../../tasks/v7-cpu-development-window-02.md). Duplicate
raw JSON members are also now rejected at all object depths (including escaped
equivalent names), before hash validation can discard ambiguity. Legal JSON
key order/whitespace/Unicode formatting remain readable; hashes are semantic
JSON integrity, not byte-level authentication. Two-unit window complete; no
research/production promotion.

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

## Strategic recovery planning pointer — 2026-10-21

Current objective/implementation gaps are recorded in the
[dated recovery report](../../reports/v7_strategic_roadmap_recovery.md).
The sole active cross-domain backlog is the
[existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
This pointer does not replace this domain's machine contracts or promote evidence.
Five CPU fixes have a verified integration candidate from `4c388de`, UNMERGED /
NOT_DEPLOYED. PORT-01–04 H1 fixture engineering is now COMPLETED_VERIFIED;
Human Product Slice Review remains pending. See the
[acceptance report](../../reports/v7_portfolio_history_acceptance.md). Strict Phase 0
STOP, E1 portfolio admission=false and production boundaries remain unchanged.

## Local H0 integration closure — 2026-10-21

The complete limited fixture History slice, five CPU fixes and UX/font correction
are LOCAL_DEVELOPMENT_INTEGRATED at exact4a66928, NOT_DEPLOYED. This supersedes
pre-H0 branch-local/pending-review status above; original implementation/acceptance
semantics remain unchanged. [Closure](../../reports/v7_portfolio_h0_integration_closure.md)
and [Human approval](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
No real data/model admission, formal performance policy or research approval.
Static UI UNMERGED badge is inherited candidate wording, not live Git status;
canonical state records local integration. No UX changes made during integration.
Strict Phase0 STOP/E1 BLOCKED/H2 unapproved. Stop; next Program needs separate scope.
