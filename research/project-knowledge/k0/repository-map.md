# MarketMamba Repository Map

> GENERATED / DISCOVERY — snapshot 2026-09-20. This map is not a canonical human note.

## Top-level

```text
MarketMamba/
├── Data/                 mutable V6 inputs plus versioned V7 evidence/artifacts
├── V6/                   ingestion, models, pipelines, portfolio, experimental V7
├── app/                  FastAPI backend and React frontend
├── archive/              historical/superseded material
├── deliveries/           transfer bundles and user-returned outputs
├── docs/                 operations, plans and historical documentation
├── environments/         exact environment locks
├── obsidian_note/        local supporting notes; not empirical authority
├── reports/              human audit and analysis reports
├── research/             machine contracts, evidence and decisions
└── tasks/                plans and handoffs
```

Snapshot counts: 205 Python files under `V6`/`app`, 26 test-pattern files, 382 Markdown files and 23 notebooks, excluding `.git` and worktrees where noted.

## Important subtrees and entrypoints

| Area | Key path | Discovered role |
|---|---|---|
| Scheduled fetch | `V6/scripts/v62_daily.bat --fetch-only` | Active acquisition entry |
| Full V6.2 | `V6/run_v62_daily.py` | Manually runnable; full scheduled task disabled |
| Data pipeline | `V6/marketmamba/data/` | Mutable ingestion/merge/features |
| V7 correctness | `V6/experimental/v7_*` | Builder/audit artifact DAG |
| Corrected baseline | `research/v7/corrected-baseline-v1/` | Canonical V7 research contract |
| Readiness | `research/v7/ohlcv-provenance-remediation-v1/readiness.json` | Historical PASS / strict STOP |
| E1 | `V6/experimental/v7_e1_run.py` | Current rolling-origin experiment runner |
| V7 portfolio | `V6/experimental/v7_portfolio_{contract,engine,journal}.py` | Research-only deterministic simulator |
| Phase A ops | branch `feature/production-phase-a-shadow-ledger`, `V6/marketmamba/ops/` | Unmerged shadow ledger |
| API/UI | `app/backend/main.py`, `app/frontend/src/App.jsx` | Delivery surface; deployment unknown |

## Active runtime path

```text
Windows Task Scheduler
  -> V6/scripts/v62_daily.bat --fetch-only
  -> V6/run_v62_daily.py
  -> V6/marketmamba/data/fetcher.py
  -> Data/processed_v6/*.parquet
```

The full V6.2 path continues from features to inference, portfolio, performance and Git publication, but its scheduler task is disabled. Phase A ledger integration exists only on an isolated branch.

## Canonical artifacts

- Feature contract: `research/v7/corrected-baseline-v1/feature-manifest.json`.
- Label/correctness policy: corrected baseline policies/manifests plus executable tests.
- Provider provenance/readiness: `research/v7/ohlcv-provenance-remediation-v1/`.
- E1 experiment specification: `research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json`.
- Completed-run authority: final run-local manifests/results, not notebook names or prose.

## Legacy/deprecated paths

- V6.1 and full V6.2 model/portfolio publication are legacy operational paths, though code remains runnable.
- V6 graph/GAT paths do not belong to corrected V7.
- Legacy A/B/C and capacity IC results remain historical evidence and must not be relabeled as tradability evidence.
- `archive/` and old handoffs are context, never current authority by filename.
