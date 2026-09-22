# MarketMamba K0 Project Knowledge Discovery

> GENERATED DISCOVERY, observed 2026-09-20. This report is evidence-indexed but is not itself canonical project truth.

## Executive finding

MarketMamba is not one linear “model project.” It currently contains nine trust-separated subsystems: mutable market-data acquisition, legacy V6.2 scoring/publication, immutable V7 correctness evidence, corrected feature/label contracts, Colab research training, historical simulation, a research portfolio state machine, delivery API/UI, and an unmerged shadow operations control plane. Most dangerous documentation errors come from crossing these boundaries.

The operationally observed scheduler runs acquisition only. The full V6.2 pipeline remains executable but disabled. V7 historical simulation is admitted under a frozen proxy and provenance gate, while strict verified-executable readiness remains STOP with 0/0 strict labels. E1 is specified and packaged; repository evidence does not prove a completed formal run.

## Repository and Git snapshot

- Main: `c0babe0`, ahead of `origin/main` by 16, dirty with pre-existing V7/UI candidates.
- K0 branch/worktree: `docs/project-knowledge-k0-discovery`, started at `9401ddf` after the plan commit.
- E1 branch: `feature/v7-e1-rolling-origin-refresh` at `044eef7`, clean worktree.
- Phase A branch: `feature/production-phase-a-shadow-ledger` at `f5b0ff3`, clean worktree.
- Important local history also includes corrected E5 diagnostic v1/v2, PIT reconstruction, provenance remediation and production architecture audit branches.
- No push, merge, rebase, production change, data modification, GPU run or E1 execution occurred in K0.

The main worktree’s local candidate files are evidence only. They are not merged authority.

## Current subsystem shape

### 1. Production acquisition

`MarketMamba_DataFetch` is enabled and invokes `V6/scripts/v62_daily.bat --fetch-only`. That path terminates before matrix construction, inference, portfolio and publication. It writes mutable provider data under `Data/processed_v6` and must not be used as PIT authority.

### 2. Legacy V6.2 runtime

`V6/run_v62_daily.py` still implements fetch → matrix → Mamba/baseline inference → portfolio → performance → Git publication. The corresponding full scheduler is disabled. V6.2 is therefore partially active: ingestion code is operational, while the model/publication path is legacy and manually runnable.

### 3. V7 correctness/PIT layer

Official source snapshots, verified lifecycle intervals, publication records and OHLCV provenance form an artifact DAG rather than a single pipeline. Missing provenance, publication time or lifecycle evidence fails closed. Machine-readable manifests and tests are the authority.

### 4. Corrected V7 feature/label layer

`E5-PIT-Clean-v1` freezes 48 ordered features, no explicit graph, no current-industry backfill and no industry neutralization. Corrected labels use next frozen-session open, never roll, and the h-th holding-session close. Strict executable and historical proxy labels are different classes.

### 5. Corrected E5/E1 research

Completed corrected E5 authority resolves to `d_model=64`, `d_state=32`, seeds 17/29/43, superseding a draft 32/8 architecture file. Formal diagnostics use complete seeds 17/29; seed43 remains partial convergence evidence. Extended diagnostic v2 concludes `STOP_FOR_TRADING_CONTINUE_RESEARCH`. E1 changes the temporal split only and freezes architecture, features, labels and acceptance.

### 6. Portfolio semantics

V6.2 has a legacy operational portfolio path and append-only JSONL. V7 has a stronger research-only engine with explicit target/fill/cost/action semantics and hash-chained replay. No discovered path connects it to a broker or active scheduler.

### 7. Delivery UI/API

FastAPI and React expose legacy result artifacts. Local V7 status additions are fail-closed but untracked on main, so they are candidate implementation rather than merged authority. Live hosting and traffic were not inspected.

### 8. Operations control plane

Phase A implements an SQLite WAL shadow ledger, fencing, transactional stage commit, sanitization and atomic current-state projection around fetch-only. It is isolated on a feature branch, not deployed, and deliberately does not override the legacy fetch success decision.

### 9. Knowledge/documentation

AGENTS and Obsidian are useful navigation memory. README, PROJECT, OVERVIEW, handoffs and old results contain history but frequently lag current runtime or correctness semantics. Machine contracts/evidence/tests outrank prose.

## Authority order

1. Versioned machine contracts, content hashes, manifests and executable policy tests.
2. Completed run-local final results and validated checkpoint manifests.
3. Current implementation on the explicitly named branch/worktree.
4. Time-stamped external runtime observations such as scheduler state.
5. Human reports and agent guides as interpretation/navigation.
6. Old handoffs, filenames such as “final,” and legacy A/B/C artifacts as historical evidence only.

No single document currently owns all “current state.” This absence is the primary K1 problem.

## Data and label safety conclusions

- Mutable V6 data is availability-oriented, not immutable PIT evidence.
- Source provenance eligibility must run before execution-proxy admission.
- P0 historical-simulation readiness is PASS with zero critical false executable cases.
- Strict Phase 0 remains STOP and strict verified executable labels remain 0/0.
- Historical simulated v1 labels are 6,889,229 (5d) and 6,790,554 (10d); v2 has 7,023,920 and 6,923,464. These figures are not trading evidence.
- Mixed-source, synthetic/backfilled and current-world data must be explicitly classed and cannot silently enter corrected authority.

## Research lineage conclusion

V6.1/V6.2 and old capacity/confirmation experiments explain architecture history but their IC values are not current evidence. PIT reconstruction and provenance remediation changed the admissible data universe. Corrected E5 then exposed broad 2026 degradation and nearly redundant heads. E1 is the next controlled experiment because it tests refresh timing without searching a new architecture. It must retain strict STOP regardless of model outcome.

## Test posture

Coverage is strongest around V7 correctness policies, corrected E5/E1 contracts/checkpoint recovery, V7 portfolio replay, and Phase A ledger behavior on its branch. Coverage is weakest at live provider boundaries, external scheduling/deployment, the complete V6.2 publication path, browser-level UI behavior and knowledge drift. Test counts alone would conceal these gaps.

## Documentation drift and ambiguity

Top drift:

1. README/OVERVIEW can imply full daily model production while scheduler reality is fetch-only.
2. Feature counts 56/59/66/48 appear without contract qualification.
3. Draft corrected architecture 32/8 conflicts with completed 64/32 authority.
4. `CLAUDE.md` retains a large stale duplicate of project knowledge.
5. Roadmap prose and main documentation lag Phase A branch and local V7 UI candidates.

Top ambiguity:

1. Live API/frontend deployment and traffic.
2. Whether external Drive contains a newer completed E1 run than repository state.
3. The scheduler’s stale hardcoded D-drive/WSL path risk.
4. Which ignored/local artifacts are durable authorities versus disposable caches.
5. Ownership and update cadence for a future current-state authority.

## K1 recommendation

Use a hybrid structure. Put canonical current state, authority mapping and stable domain knowledge in the repository so branches, reviews and agents see the same truth. Keep external Obsidian for private hypotheses, reading notes and navigation, linking to repo authorities rather than duplicating them.

The first K1 slice should create only `knowledge/00_Project_Map/Current_State.md` and `Authority_Map.md`, seed them from reviewed K0 findings, and define ownership/observation expiry. Domain notes, thin AGENTS/CLAUDE wrappers and automated drift checks should follow incrementally.

## Validation scope

K0 deliverables are under `research/project-knowledge/k0/` and this report. JSON parsing, required-file presence, path-reference sampling, duplicate active-canonical review, diff checks and final Git cleanliness are recorded at completion. Missing external runtime evidence is represented as unknown rather than inferred.
