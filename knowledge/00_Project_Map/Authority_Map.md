# MarketMamba Authority Map

> Status: `CANONICAL_CURRENT_PROJECT_MAP`
> Owner: `Human + project agents`
> Semantic change owner must update this map in the same change.
> Prior research/Phase A map accepted `2026-09-20`; 2026-09-22 runtime additions are agent-observed, pending separate human review. Canonical means current navigation authority, not immutable final state.

## Source-of-truth Hierarchy

1. Versioned machine contracts, manifests, hashes, and executable tests.
2. Completed run-local final artifacts and validated checkpoint manifests.
3. Implementation on an explicitly named branch and commit.
4. Timestamped runtime observations.
5. Human reports and guides.
6. Historical notes and legacy artifacts.

An authority class describes evidence strength, not deployment status. Branch-local code and untracked candidates are never promoted to `CANONICAL` merely because they exist.

## Authority by environment and evidence class

- **Operational authority:** implementation plus a timestamped external action/observation. `AGENTS.md` and a runnable script alone do not prove a live schedule. A stale observation becomes `STALE_OBSERVATION`, never a fresh fact by documentation update.
- **Research state:** Current State summarizes admitted branch-qualified results. A main file can lag an accepted worktree result; the frozen experiment contract controls the question, completed run manifest/hashes control the result, and neither promotes production.
- **Artifacts:** a result filename or copied ZIP is insufficient without source commit, contract, input/matrix identity, environment, split/support, model/checkpoint identity, outputs and evidence class. Mutable Parquet caches need separate provenance snapshots. Untracked local files are candidates, not canonical authority.
- **Runtime:** WSL development, Windows Scheduler checkout, Colab Drive and deployed API/UI are distinct. Record `observed_at` for Windows/service facts; use realpath and source commit for cross-environment identity. No inferred equivalence.
- **Branch/worktree:** a versioned `commit:path` remains valid historical/research authority without being merged or deployed. A branch label aids navigation but can move. Worktree checkout name is not evidence of its current branch.

## Freshness Rules

- `RUNTIME_OBSERVATION`: requires `observed_at`; after 7 days it becomes `STALE_OBSERVATION` until rechecked. It is not deleted automatically.
- Repository contracts: no time expiry. They change only through commit identity or explicit supersession.
- Completed-run artifacts: remain evidence for that run; they do not automatically describe a newer run or current production.
- `UNKNOWN`: has no primary authority. Secondary evidence may explain what is known without filling the gap.

## Topic Map

| Topic | Primary Authority | Authority Class | Status | Supersedes | Secondary Evidence | Conflicts / Notes |
|---|---|---|---|---|---|---|
| Worktree responsibility | [Worktree Register](Worktree_Register.md) | `CURRENT_REFERENCE` | `EIGHT_OBSERVED_2026_09_22` | Ad-hoc worktree cleanup assumptions | Git worktree list, branch HEAD and status | Registration does not authorize removal; ownership/creation dates may be UNKNOWN. |
| Development runtime contract | [Current State](Current_State.md) and [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | `WSL_NATIVE_DEVELOPMENT` | Old assumption that `/mnt/d` checkout is canonical development | `docs/operations/wsl-migration-2026-09-07.md` | Operational Windows/WSL bridge is separate; no runtime migration this change. |
| Mutable data cache | [`V6/marketmamba/config.py`](../../V6/marketmamba/config.py) and source manifests | `CURRENT_REFERENCE` | `MUTABLE_NON_PIT_CACHE` | Treating cache as frozen research evidence | `MARKETMAMBA_DATA_ROOT`, symlink realpath, V7 source registry | Raw caches can change; authority requires snapshot/identity. |
| Colab runtime | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CURRENT_REFERENCE` | `SEPARATE_LOCKED_TRAINING_RUNTIME` | Local Python environment inferred as formal Colab | E1 bundle and exact requirements lock | Formal result needs environment/bundle/run identity. |
| Local untracked candidates | None | `UNKNOWN` | `NOT_ADMITTED` | File-existence/current-version inference | Main untracked research files | No branch commit/contract or deployment authority from untracked source. |
| Current project state | [Current_State.md](Current_State.md) | `CANONICAL` | `CANONICAL_CURRENT_PROJECT_MAP` | README/OVERVIEW current-state prose and old handoffs | [K0 report](../../reports/marketmamba_k0_project_knowledge_discovery.md) | Accepted 2026-09-20; update or supersede when semantics change. |
| Domain semantics/navigation | [Domain notes](../01_Domains/) | `CURRENT_REFERENCE` | `CANONICAL_DOMAIN_NOTE` / `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN` | Scattered README/guide summaries | [Knowledge index](../README.md), machine contracts linked per note | Notes summarize boundaries and point to authority; they never override machine contracts. |
| Knowledge governance | [GOVERNANCE.md](../GOVERNANCE.md) | `CANONICAL` | `ACTIVE_GOVERNANCE_V1` | K0 governance proposals and manual-only K1 maintenance | `knowledge/knowledge-impact-map.json`, governance tools/tests | Governance detects and records; it never rewrites authority or business semantics automatically. |
| Runtime observation store | [`runtime-observations/`](../runtime-observations/) | `CURRENT_REFERENCE` | `TIMESTAMPED_EVIDENCE_WITH_TTL` | Runtime facts copied only into prose | Runtime status tool and K0 observation evidence | Recording requires an actual observation; stale evidence remains stale. |
| Production runtime | [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) in Windows `main@8b82a68` plus [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `CURRENT_REFERENCE` | `FETCH_ONLY_ACTIVE`; full path `DISABLED` | README/OVERVIEW implication of full scheduled daily inference | D: batch and [operational path](../runtime-observations/operational-path-20260922T132521+0800.json) | Scheduler selects Windows checkout, not WSL development HEAD. |
| Scheduler state | [2026-09-22 Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `RUNTIME_OBSERVATION` | Active fetch-only / disabled full V6.2; last run 2026-09-21 result 0 | 2026-09-20 observation and older snapshots | [K0 runtime map](../../research/project-knowledge/k0/runtime-map.json), local logs | Becomes `STALE_OBSERVATION` after 2026-09-29 13:21:35 +08:00. |
| Operational checkout and WSL environment | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | Windows `main@8b82a68` → distro `Ubuntu` → `mamba_env` Python 3.11.15 | Assumed WSL-native development checkout is scheduled | Batch content, `wsl.exe --list --verbose`, Git | Development `Ubuntu-24.04` checkout is distinct `main@3b0df0f` at observation. |
| Operational data root and freshness | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | D: physical `Data/processed_v6`; ten daily sources max 2026-09-21 | Mtime-only freshness or guessed path equivalence | Parquet metadata, `v62_daily.log` | Date check 10/10 does not certify source validity; universe delta warning remains. |
| Data provenance policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/source-reliability.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Mixed-source OHLCV admission | Provenance lineage, source index and mismatch inventory in the same directory | Source classification must precede proxy classification. |
| PIT policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Current-world/backfilled data assumptions | Executable builders and Phase 0 tests | Missing lifecycle/publication/provenance evidence fails closed. |
| Feature contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` | `CANONICAL` | `48_FEATURES` | Legacy V6.1 56-feature, V6.2 59-feature and baseline 66-feature contracts | Corrected baseline builder/tests | Counts belong to separate contracts; they are not interchangeable. |
| Feature list | Same versioned feature manifest and its order hash | `CANONICAL` | `ORDER_PINNED` | Legacy feature orders and removed fundamentals | [K0 feature map](../../research/project-knowledge/k0/feature-map.json) | No current-industry backfill, graph features, or industry neutralization. |
| Label contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/label-manifest.json` | `CANONICAL` | `NEXT_OPEN_NO_ROLL` | Legacy close-to-close semantics | Policy plus historical label materializer tests | Entry is next frozen session open; exit is h-th holding-session close. |
| Universe contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/universe-manifest.json` | `CANONICAL` | `VERIFIED_LIFECYCLE` | Current constituent backfill | Historical security-master builder/tests | Membership and tradability remain separate concepts. |
| Historical simulation proxy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json` | `CANONICAL` | `P0_FROZEN`; simulation readiness `PASS` | Ad-hoc execution heuristics | Proxy validation and provenance-remediation readiness | Must never be called `VERIFIED_EXECUTABLE`. |
| Strict executable evidence | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` | `CANONICAL` | `STOP`; verified labels 5d=0, 10d=0 | Any claim that proxy PASS implies trading readiness | Corrected Phase 0 audit | Strict STOP and simulation PASS are intentionally simultaneous. |
| Corrected E5 architecture | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json` | `CANONICAL` | `d_model=64`, `d_state=32`, 48 features, no graph | `architecture-config.json` draft values 32/8 | Completed E5 result identity and final runner | Completed result authority outranks the earlier draft. |
| Training recipe | Same incumbent contract; E1 freezes it in `experiment-contract.json` | `CANONICAL` | `FP32_ADAMW_20_EPOCH_MAX` | Earlier notebook-only assumptions | Exact environment lock and final runner source | E1 changes split, not recipe. |
| Corrected E5 checkpoints | Validated run-local checkpoint manifests for each seed | `RESEARCH_EVIDENCE` | Seeds 17/29 complete; seed43 `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Filename/pointer-only inference | Diagnostic resume forensics and telemetry | Seed43 pointer metadata named missing generations and recorded `best_epoch=8`; the surviving orphan is epoch 15/step 39,915 and is convergence evidence only, not promotable or directly resumable. |
| Diagnostic v1 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v1/summary.json` | `RESEARCH_EVIDENCE` | `SUPERSEDED_FOR_CURRENT_CONCLUSION` | Raw partial-run impressions | v1 component analyses | Extended diagnostic v2 is the current conclusion. |
| Extended diagnostic v2 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Diagnostic v1 as final conclusion | v2 head, drift, regime and equal-window analyses | Research evidence only; not production/trading authority. |
| E1 experiment specification | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CANONICAL` | `FROZEN_EXPERIMENT_SPECIFICATION` | Informal experiment proposals | Contract diff, runner, checkpoint store, generated notebook | The pre-run decision field remains historical specification, not completed-result authority. |
| E1 completed result | `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` | `RESEARCH_EVIDENCE` | `PASS_REFRESH_HYPOTHESIS` | External-only/unverified result state | Final evaluation, run summary, seed manifests and hashes in the same directory | Historical-simulation proxy only; strict Phase 0 `STOP`; no portfolio admission. |
| E1 post-run diagnostic | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | Undiagnosed post-run state | Month and leave-one-month-out analyses, checkpoint inventory, logged validation history | July-sensitive 10d absolute level; no final-support epoch replay or causal regime series. |
| Portfolio semantics | [`V6/experimental/v7_portfolio_contract.py`](../../V6/experimental/v7_portfolio_contract.py) plus engine/journal tests | `CURRENT_REFERENCE` | `RESEARCH_ONLY` | V6.2 JSONL as V7 semantic authority | `v7_portfolio_engine.py`, `v7_portfolio_journal.py` | No live broker or scheduler integration was discovered. |
| Production-system architecture | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and roadmap | `CURRENT_REFERENCE` | Audit complete; roadmap Phase A–F | Earlier production prose | Production-system machine contracts | Branch implementation may be newer than prose. |
| Phase A implementation | `feature/production-phase-a-shadow-ledger@f5b0ff3:V6/marketmamba/ops/` | `UNMERGED_IMPLEMENTATION` | Tasks 1–3 completed/reviewed; Task 4 implemented with sanitizer fixes but correctness approval pending; Task 5 not started; `UNMERGED`, `NOT_DEPLOYED` | Roadmap-only description of Phase A | Branch tests and commit history through sanitizer fixes | `HUMAN_REVIEW_REQUIRED`; implementation/fixes do not imply correctness acceptance. Shadow evidence does not own legacy fetch success. |
| V7 status API/UI source | [`app/backend/routers/v7.py`](../../app/backend/routers/v7.py), [`app/frontend/src/pages/V7Status.jsx`](../../app/frontend/src/pages/V7Status.jsx), focused tests | `CURRENT_REFERENCE` | `COMMITTED_NOT_DEPLOYED` in WSL development main | Local untracked V7 status candidate | Health summary builder and frontend contract tests | Additive status route only; no model inference, portfolio or scheduler authority. |
| API/UI live deployment | None for live deployment state | `UNKNOWN` | `UNKNOWN` | Claims inferred from source presence | Source paths above | Source capability does not prove deployment; provider revision remains unverified. |
| Agent instructions | [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | Navigation/policy only | Stale duplicated assumptions in `CLAUDE.md` | K0 agents/CLAUDE audits | Machine contracts and this map outrank project-state prose in agent guides. |

## Authority Classes

- `CANONICAL`: accepted contract or project-map authority for its explicitly stated scope.
- `CURRENT_REFERENCE`: best current implementation/report, but not necessarily immutable or deployed.
- `RESEARCH_EVIDENCE`: valid research conclusion for its run/evidence class.
- `HISTORICAL_ONLY`: retained context that must not drive current decisions alone.
- `RUNTIME_OBSERVATION`: external state valid only at its observation time and freshness window.
- `UNMERGED_IMPLEMENTATION`: branch-local implementation, neither deployed nor mainline authority.
- `UNKNOWN`: no verified primary authority currently exists.
