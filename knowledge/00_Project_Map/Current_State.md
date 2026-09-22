# MarketMamba Current State

> Status: `CANONICAL_CURRENT_PROJECT_MAP`

## Observation Metadata

| Field | Value |
|---|---|
| `observed_at` | `2026-09-20T02:45:00+08:00` |
| `owner` | Human + project agents |
| Repository context | `main` at governance consolidation; K0/K1/K2 lineage adopted from `integration/project-knowledge-k1`, E1 admission from `research/v7-e1-postrun-diagnostics` |
| State updated | `2026-09-22`; this is a documentation review date, **not** a new scheduler observation |
| Source basis | Reviewed K0 discovery, versioned machine contracts, named branch implementations, and timestamped scheduler observation |
| Human review | Accepted `2026-09-20`; seed43 forensic and Phase A task wording reconciled; K1 domain navigation added without changing technical authority |
| Runtime freshness | External scheduler/deployment observations expire after 7 days and become `STALE_OBSERVATION`; they are retained until re-observed |
| Repository freshness | Contracts do not expire by time; commit identity or explicit supersession changes them |

Semantic change owners must update this file in the same change. K0 remains discovery evidence. `CANONICAL_CURRENT_PROJECT_MAP` means current accepted navigation/state authority, subject to semantic updates and explicit supersession; it does not mean immutable or permanently final.

## Project Summary

**Active objectives:** recover the identity-matched E1 matrix and safely replay retained checkpoints before deciding further research; review Phase A Task 4 evidence before any merge/deploy. Governance consolidation supplies navigation, not a new research or production gate.

MarketMamba contains separate production-data, legacy model runtime, V7 correctness, research-training, historical-simulation, portfolio, delivery, and operations-control-plane subsystems. Production acquisition uses mutable provider data; V7 research authority uses versioned PIT/provenance evidence. The current corrected research baseline is `E5-PIT-Clean-v1`, with 48 features, no graph and no industry neutralization. Historical simulation is `PASS`, while strict trading correctness remains `STOP`. The observed production scheduler runs fetch-only; full V6.2 inference/portfolio/publication is `DISABLED`. E1 is a completed research experiment with `PASS_REFRESH_HYPOTHESIS`, not trading evidence. Phase A operations work is `UNMERGED` and `NOT_DEPLOYED`.

## Knowledge Layers

Canonical development/research checkout is WSL-native `/home/frank/projects/MarketMamba`; existing Windows scheduler/data boundary is `D:\Desktop\work\ProjectForMe\MarketMamba`. `MARKETMAMBA_DATA_ROOT` and `Data/` symlink realpaths must be checked per runtime. Colab is a separate locked training environment. The live Windows task/action has **not** been reconciled on 2026-09-22; the 2026-09-20 observation remains the most recent recorded one.

- **Layer 1 — Current State:** this short file and its JSON projection hold active objectives, blockers, next actions, current runtime and branch/worktree context. [Authority Map](Authority_Map.md) determines which evidence wins; [Worktree Register](Worktree_Register.md) records checkout responsibility.
- **Layer 2 — Project Knowledge:** [Domain Notes](../01_Domains/), [GOVERNANCE.md](../GOVERNANCE.md), and stable architecture/decision contracts describe boundaries and durable rules.
- **Layer 3 — History / Evidence:** [History](../02_History/), research contracts/manifests/results, dated runtime observations, reports, Obsidian and Git history preserve completed or superseded work. Machine contracts, hashes, source and executable tests bind technical claims; historical prose does not override them.
- `AGENTS.md` and `CLAUDE.md` are the agent operating contract outside the three memory layers.

## Knowledge Governance State

| Topic | State | Authority / notes |
|---|---|---|
| Domain layer | `COMPLETE_CURRENT_SCOPE` | Twelve reviewed domain notes; machine contracts still bind technical truth. |
| Governance | `ACTIVE` | [`knowledge/GOVERNANCE.md`](../GOVERNANCE.md); impact, supersession, runtime-freshness and history protocols are defined. |
| Unified health | `AVAILABLE` | `tools/project_knowledge/knowledge_health.py`; PASS/WARNING are non-blocking, structural FAIL is blocking. |
| Runtime observation store | `ACTIVE_WITH_STALE_SEMANTICS` | `knowledge/runtime-observations/`; timestamps change only after an actual observation is recorded. |
| Generated navigation | `DETERMINISTIC` | `generate_indexes.py`; declared generated files are not hand-edited. |

## Production State

| Topic | State | Authority / observation |
|---|---|---|
| Active scheduler path | `ACTIVE` | `MarketMamba_DataFetch → V6/scripts/v62_daily.bat --fetch-only`; observed `2026-09-20T01:30:00+08:00`, may expire after 7 days. See [K0 runtime map](../../research/project-knowledge/k0/runtime-map.json). |
| Fetch-only behavior | `ACTIVE` | [`V6/scripts/v62_daily.bat`](../../V6/scripts/v62_daily.bat) and [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) stop before matrix, inference, portfolio and publication. |
| Full V6.2 schedule | `DISABLED` | `MarketMamba_V62` scheduler observation; code remains manually runnable. |
| Result publication | `DISABLED_AS_SCHEDULED`; manual state `UNKNOWN` | Full pipeline contains publication logic, but no active scheduled publication was observed. |
| API deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |
| Frontend deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |

Runtime observations are not permanent facts. After `2026-09-27T01:30:00+08:00`, scheduler statements above must be treated as `STALE_OBSERVATION` until checked again.

## Data / Correctness State

| Topic | State | Authority |
|---|---|---|
| Historical-simulation readiness | `PASS` | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` |
| Strict Phase 0 | `STOP` | Same readiness evidence and corrected Phase 0 audit; it is intentionally separate from historical-simulation readiness. |
| Verified executable labels | `5d=0`, `10d=0` | Corrected E5/E1 machine contracts. |
| Corrected feature contract | `48 FEATURES`, ordered and manifest-pinned | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` |
| Graph | `DISABLED`, zero edges, fusion branch absent | Completed corrected E5 contract. |
| Industry neutralization | `DISABLED` | Corrected baseline and E1 contracts. |
| Historical labels | `RESEARCH_ONLY`, evidence class `HISTORICAL_SIMULATION_PROXY`, policy `P0` | Never call these `VERIFIED_EXECUTABLE` or tradability evidence. |

## Research State

| Topic | State | Authority / notes |
|---|---|---|
| Corrected baseline | `CURRENT_REFERENCE`: `E5-PIT-Clean-v1` | Corrected baseline manifests and completed incumbent contract. |
| Corrected E5 | `COMPLETED_PARTIAL_ENSEMBLE` | Seeds 17 and 29 completed. Architecture authority is `d_model=64`, `d_state=32`; the old 32/8 draft is superseded. |
| Formal seeds | `17`, `29` | Used by diagnostic conclusions. |
| Seed 43 | `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Pointer metadata referenced `best_epoch=8`, but pointer-named best/latest/previous generations are missing. The surviving orphan checkpoint is epoch 15, step 39,915, `terminal=false`, contract/matrix matched, and is convergence evidence only. It is not a formal seed and must not be promoted or directly resumed. |
| Extended diagnostic | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Broad 2026 degradation and near-redundant heads; authority is diagnostic v2. |
| E1 rolling-origin refresh | `COMPLETED_RESEARCH_EXPERIMENT`; `PASS_REFRESH_HYPOTHESIS` | Formal seeds 17/29; `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`. The specification retains its pre-run wording as history. Historical-simulation proxy only; strict Phase 0 `STOP`. |
| E1 post-run diagnostic | `MONTH_SENSITIVITY_FOUND`; `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`. Logged validation declines after epoch 1; final-support replay and causal regime attribution remain unavailable. |

## Production-System State

| Topic | State | Authority / notes |
|---|---|---|
| Architecture audit | `COMPLETE_CURRENT_REFERENCE` | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and its production-system roadmap. |
| Phase A implementation | `UNMERGED`, `NOT_DEPLOYED` | `feature/production-phase-a-shadow-ledger@f5b0ff3`; SQLite WAL ledger, fencing, shadow fetch observer and atomic current-state projection exist on that branch. |
| Phase A task state | Task 1 contracts/schema migrations: `COMPLETED_REVIEWED`; Task 2 transactional store/lease/fencing: `COMPLETED_REVIEWED`; Task 3 fetch-only shadow adapter: `COMPLETED_REVIEWED`; Task 4 atomic projection/read-only CLI: `IMPLEMENTED_WITH_SANITIZER_FOLLOW_UP_FIXES`, correctness approval `NOT_ESTABLISHED`; Task 5 replay/formal acceptance report: `NOT_STARTED` | Branch-local implementation is not production authority. |
| Current blocker | `HUMAN_REVIEW_REQUIRED` | Task 4 correctness approval remains pending; sanitizer/path-redaction fixes do not constitute acceptance by themselves. |
| Production success authority | Legacy fetch result remains authoritative | Shadow-ledger failures must not change fetch-only success semantics. |

## Delivery/UI State

- Tracked FastAPI and React legacy delivery surfaces: `PARTIALLY_ACTIVE`; deployment is `UNKNOWN`.
- Local V7 status API/UI additions in the main worktree: `UNMERGED_CANDIDATE`, not authority.
- V7 status design is read-only and fail-closed, but live behavior is `UNKNOWN`.

## Active Blockers

1. `STOP`: strict Phase 0 has no verified executable 5d/10d labels.
2. `HUMAN_REVIEW_REQUIRED`: Phase A Task 4 correctness approval, sanitizer/path-redaction evidence, and merge decision.
3. `UNKNOWN`: E1 final-support epoch dynamics and causal regime dependence require the frozen matrix and ex-ante market-state evidence; completed result is admitted.
4. `UNKNOWN`: live API/frontend deployment and publication state.

## Current Decisions / Next Actions

1. Recover the identity-matched E1 matrix and safe checkpoint replay, then predeclare causal regime series. Do not launch another training experiment until diagnostic limitations are resolved.
2. Review Phase A Task 4 sanitizer evidence before any merge/deployment decision.
3. Maintain K1 domain notes and drift checks in the same change when domain semantics, authority, or boundaries change.
4. Use the consolidated K2 Current State, Authority Map and knowledge-impact checks. Keep the K0/K1/K2 worktrees until their artifact/merge dependencies and safe-removal conditions are reviewed; do not begin K3 automatically.

## Active worktree boundary

Eight linked worktrees were observed on 2026-09-22; their branch, purpose, authority and retirement preconditions are recorded in [Worktree Register](Worktree_Register.md). Branch-local research/ops results remain branch-qualified even after governance adoption. The main worktree retains unrelated dirty and untracked user changes.

## Authority Links

- [Authority Map](Authority_Map.md)
- [Domain knowledge index](../README.md)
- [K0 discovery report](../../reports/marketmamba_k0_project_knowledge_discovery.md)
- [K0 machine authority map](../../research/project-knowledge/k0/authority-map.json)
- [K0 boundary map](../../research/project-knowledge/k0/boundary-map.json)
- Corrected/PIT authorities are branch-qualified in [Authority Map](Authority_Map.md).

## Knowledge-impact Rule

Update canonical knowledge in the same change for: semantic changes, authority changes, subsystem-boundary changes, active runtime-state changes, correctness-gate changes, new canonical artifacts, deprecations, or major blocker/next-decision changes.

No knowledge update is normally required for formatting, comments, pure refactors, tests-only changes, or internal renames that do not change public semantics or authority.
