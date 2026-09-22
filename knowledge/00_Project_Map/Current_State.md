# MarketMamba Current State

> Status: `CANONICAL_CURRENT_PROJECT_MAP`

## Observation Metadata

| Field | Value |
|---|---|
| `observed_at` | `2026-09-22T13:25:21+08:00` for this runtime reconciliation; individual observations retain their own timestamps |
| `owner` | Human + project agents |
| Repository context | WSL development `main@3b0df0f` at observation; Windows operational `main@8b82a68`; research authority remains branch-qualified |
| State updated | `2026-09-22`; new Scheduler observation at `2026-09-22T13:21:35+08:00` |
| Source basis | Reviewed K0/contracts plus [new Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) and [operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) |
| Human review | Prior research/Phase A scope accepted `2026-09-20`; the new 2026-09-22 runtime observations are agent-observed and have not received separate human review |
| Runtime freshness | External scheduler/deployment observations expire after 7 days and become `STALE_OBSERVATION`; they are retained until re-observed |
| Repository freshness | Contracts do not expire by time; commit identity or explicit supersession changes them |

Semantic change owners must update this file in the same change. K0 remains discovery evidence. `CANONICAL_CURRENT_PROJECT_MAP` means current accepted navigation/state authority, subject to semantic updates and explicit supersession; it does not mean immutable or permanently final.

## Project Summary

**Active objectives:** recover the identity-matched E1 matrix and safely replay retained checkpoints before deciding further research; review Phase A Task 4 evidence before any merge/deploy. Governance consolidation supplies navigation, not a new research or production gate.

MarketMamba contains separate production-data, legacy model runtime, V7 correctness, research-training, historical-simulation, portfolio, delivery, and operations-control-plane subsystems. Production acquisition uses mutable provider data; V7 research authority uses versioned PIT/provenance evidence. The current corrected research baseline is `E5-PIT-Clean-v1`, with 48 features, no graph and no industry neutralization. Historical simulation is `PASS`, while strict trading correctness remains `STOP`. The observed production scheduler runs fetch-only; full V6.2 inference/portfolio/publication is `DISABLED`. E1 is a completed research experiment with `PASS_REFRESH_HYPOTHESIS`, not trading evidence. Phase A operations work is `UNMERGED` and `NOT_DEPLOYED`.

The `environments/v7-colab-2026.04/` identity files are preserved in development main. Its exact lock matches the E1 branch copy; the E1 frozen contract and run manifests still own experiment semantics and result provenance. The dated local verification record is not proof of a current Colab GPU runtime.

## Knowledge Layers

Canonical development/research checkout is WSL-native `/home/frank/projects/MarketMamba` in the distro named `Ubuntu-24.04` (`3b0df0f` at observation). Windows Scheduler invokes the separate D: checkout `D:\Desktop\work\ProjectForMe\MarketMamba` (`8b82a68`, clean) through the distro named `Ubuntu`, not the development checkout. Both distros report Ubuntu 24.04, but they are distinct registrations. The batch activates `mamba_env` with `/home/frank/miniconda3/etc/profile.d/conda.sh` and uses Python 3.11.15 at `/home/frank/miniconda3/envs/mamba_env/bin/python`. Windows operational `Data/processed_v6` is the D: physical directory; the development checkout symlink resolves there. The read-only interactive query shell in `Ubuntu` had `MARKETMAMBA_DATA_ROOT` unset; the scheduled process environment was not captured; the Windows checkout V6 config falls back to its own `Data/`. Colab remains separate. No path or runtime was changed.

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
| Active scheduler path | `ACTIVE`, observed 2026-09-22 | `MarketMamba_DataFetch` (Ready/enabled) → `wscript.exe` → `run_hidden.vbs` → D: `v62_daily.bat --fetch-only` → `wsl -d Ubuntu` → `mamba_env` → D: checkout `V6/run_v62_daily.py`. [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json). Last run 2026-09-21 22:30 +08:00, Task result 0; log exit 0 at 22:39:52. |
| Fetch-only behavior | `ACTIVE` | [`V6/scripts/v62_daily.bat`](../../V6/scripts/v62_daily.bat) and [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) stop before matrix, inference, portfolio and publication. |
| Operational data freshness | `10/10 MATCH_REFERENCE_DATE` | Ten `DAILY_SOURCES` have Parquet metadata max date 2026-09-21, matching `prices_raw` and last fetch-only log. This is date completeness, not data-quality certification; log recorded a stock-universe delta warning. [Operational observation](../runtime-observations/operational-path-20260922T132521+0800.json). |
| Full V6.2 schedule | `DISABLED` | `MarketMamba_V62` is disabled; last run 2026-08-25 22:15, last result 255. Code remains manually runnable. |
| Result publication | `DISABLED_AS_SCHEDULED`; manual state `UNKNOWN` | Full pipeline contains publication logic, but no active scheduled publication was observed. |
| API deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |
| Frontend deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |

Runtime observations are not permanent facts. The new Scheduler observation expires after `2026-09-29T13:21:35+08:00`; the prior 2026-09-20 observation remains preserved. A successful fetch-only run does not prove the next run, deployment, trading, or complete data validity.

## Data / Correctness State

| Topic | State | Authority |
|---|---|---|
| Reusable PIT/provenance source | `IN_MAIN_RESEARCH_ONLY` | Six fail-closed primitives and their focused tests were adopted from `fix/v7-historical-pit-reconstruction@e3bce91`; source and policy bytes match branch blobs, with one test's extra EOF blank line normalized. Blob map: `research/v7/source-adoption-register-20260922.json`. Frozen experiment evidence/readiness remains branch-qualified. |
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

- Tracked FastAPI and React legacy delivery surfaces exist; live deployment and deployed revision remain `UNKNOWN` because public endpoint queries were inaccessible and provider dashboards were not accessed.
- V7 status API/UI source is `COMMITTED_NOT_DEPLOYED` in the WSL development main checkout. The public API projects only the summary schema/state/decision/data ID/time/counts; original health artifacts remain local research evidence. This source commit does not change Windows operational checkout or live deployment authority.
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
4. Use the consolidated K2 Current State, Authority Map and knowledge-impact checks. K0/K1/K2 branch history is retained without active checkouts; do not begin K3 automatically.

## Active worktree boundary

Four research/operations linked worktrees remain after the authorized K0/K1/K2 checkout retirement on 2026-09-22. Their branch, purpose and authority are recorded in [Worktree Register](Worktree_Register.md). The four K branches remain as historical lineage without active checkouts. Branch-local research/ops results remain branch-qualified; development main was clean after the retirement commit.

## Repository Rationalization Inventory

Phase 3A read-only inventory completed on 2026-09-22; report: `D:\Desktop\work\project-audit\market-mamba\repository-rationalization\MANIFEST.md`. It classified the three tracked dirty files, 69 collapsed untracked status entries (73 files), eight worktrees, local branches, large artifacts and duplicate candidates. That phase made no source/data/worktree/branch cleanup, Windows checkout sync, push or deployment. Its evidence does not alter research, scheduler or deployment authority.

Phase 3B-1 source canonicalization completed on 2026-09-22 in WSL development main; `D:\Desktop\work\project-audit\market-mamba\phase3b-source-cleanup\MANIFEST.md` is its external audit artifact. The V7 status UI/API is `COMMITTED_NOT_DEPLOYED`, E1-matching environment identity is in Git, early V7 source has a historical SHA register, and local research/task notes are preserved and explicitly dated. `research/v7/current-state.md` defers to this canonical state file. The development main source working tree was clean at completion. Corrected E5/E1/recovery/postrun and Phase A implementation/result authority remain branch-qualified for Phase 3B-2 selective review. No worktree, branch, large artifact or Windows operational checkout was changed, and nothing was pushed or deployed.

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

## Phase 3B-2 selective source integration (2026-09-22)

Development main now contains the reusable corrected PIT primitives, corrected E5 research core, formal E1 research core and frozen contract copies, plus the E1 telemetry correction and reusable Colab log wrapper. The [source adoption register](../../research/v7/source-adoption-register-20260922.json) records source commits, blobs, destinations and test evidence. This is research source authority in main, **not** result, portfolio, operational, Colab run or deployment promotion. The old 32/8 baseline draft remains branch-qualified; the accepted E5/E1 architecture is 64/32. E1 historical-simulation `PASS_REFRESH_HYPOTHESIS` and strict Phase 0 `STOP` remain unchanged. E1 postrun diagnostics remain active; Phase A Task 4 approval and Task 5 acceptance remain open. Eight worktrees were inspected and retained; no branch/worktree/artifact was removed. Windows operational checkout, Scheduler and deployment were not changed; no push.

## Phase 3B-3A governance checkout retirement (2026-09-22)

The four authorized K0/K1/K2 governance checkouts were removed using normal Git worktree removal after clean-state, ignored-file, branch-history and path-dependency checks. Their branch refs and commits remain as `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT`; the four E1/Phase A worktrees remain present. Details and local bytecode disposition are in [Worktree Register](Worktree_Register.md). No research artifact, branch, Windows operational checkout, deployment or scheduler was changed; no push.

## Phase 3C storage audit (2026-09-22)

Read-only reachability/rebuildability audit completed; [external manifest](/mnt/d/Desktop/work/project-audit/market-mamba/storage-audit/MANIFEST.md). No storage item was deleted. D: scheduled fetch-only inputs/outputs, historical baseline caches, dated ZIPs, E1 formal bundle/results, V7 provenance/label snapshots and active recovery `.venv` remain protected. A clean offline frontend install/build demonstrated that local `node_modules` and `dist` are future low-risk generated cleanup candidates (~187 MB), subject to separate cleanup authorization. Large-cache/archive deletion has no approved exact survivor or rebuild path. `DATA_QUALITY_DIAGNOSTIC` remains separate.

## Repository/governance rationalization closure (2026-09-22)

The **repository/governance rationalization program** is complete: canonical governance and authority navigation are established, runtime boundaries were reconciled, development main source is clean, reusable PIT/E5/E1 research source was selectively integrated, historical source was preserved, the four K governance checkouts were retired with branch history retained, and storage received a retention decision without deletion. The E5/E1 refresh, E1 recovery, E1 postrun, and Phase A worktrees remain intentionally present. Development main is ready for remote publication; the finalization audit records the actual Git push and remote-HEAD verification separately.

This closure does **not** complete V7 research, strict Phase 0 (`STOP`), E1 postrun diagnostics, Phase A acceptance, `DATA_QUALITY_DIAGNOSTIC`, Windows operational promotion, backend/frontend deployment verification or promotion, or large-storage cleanup. The Windows operational checkout and Scheduler remain separate and unchanged. Research result authority and branch-local operations authority remain as stated above and in the Authority Map.
