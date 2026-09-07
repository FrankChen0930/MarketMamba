# WSL Migration Snapshot - 2026-09-07

## Working Copy

- Linux-native repository: `/home/frank/projects/MarketMamba`
- Canonical remote: `https://github.com/FrankChen0930/MarketMamba.git`
- Migrated HEAD: `8b82a681e43148ff5ee6389c8cb4f6394f5360e5`
- Original Windows working copy left untouched: `/mnt/d/Desktop/work/ProjectForMe/MarketMamba`

## Dataset Access

Large generated datasets remain on the Windows D: drive:

```text
/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data
```

The V6 config honors `MARKETMAMBA_DATA_ROOT`; the local ignored file
`V6/.env` in the WSL checkout sets:

```text
MARKETMAMBA_DATA_ROOT=/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data
```

Ignored compatibility symlinks exist for legacy relative-path scripts:

```text
Data/processed_v6 -> /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6
Data/cache_v6 -> /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/cache_v6
Data/raw_v6 -> /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/raw_v6
Data/raw_cache -> /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/raw_cache
```

No generated dataset directory was copied into WSL.

## Environment

Created `.venv` with system Python 3.12.3 and installed the V6 package editable.
Runtime/data dependencies from `V6/requirements.txt` were installed without
editing dependency constraints. `torch-scatter` and `torch-sparse` were not
installed: pip selected source builds for this Python/torch combination, and the
compile was stopped because it is not a lightweight migration smoke check.

Model import currently stops at the existing project boundary for CUDA-specific
Mamba wheels:

```text
mamba_ssm is required. Install it via the pre-built wheel:
  pip install mamba_ssm-*.whl causal_conv1d-*.whl
These must be compiled for your CUDA version.
```

## Validation

Completed checks:

- WSL clone HEAD matches Windows HEAD: `8b82a681e43148ff5ee6389c8cb4f6394f5360e5`
- `marketmamba.config` resolves `DATA_DIR` to `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data`
- `prices_raw.parquet` read through the WSL clone: 8,774,375 rows, 2005-01-03 to 2026-09-07
- Representative parquet reads succeeded for `stock_info`, `macro_raw`, and `institutional_raw`
- Imports succeeded for `marketmamba.config`, `marketmamba.data.merger`, `marketmamba.quant.market_data`, and `marketmamba.signals.scanner`
- `V6/scripts/test_raw_parquet_append_integrity.py`: 3 tests passed
- `V6/scripts/test_v61_legacy_config_isolation.py`: 2 tests passed
- `compileall` passed for active V6 package modules and the lightweight test scripts
- `V6/scripts/quick_data_check.py` completed against the D: dataset
- `pip check` reported no broken requirements among installed packages

## Remaining Windows-Specific Assumptions

The active data path now goes through `MARKETMAMBA_DATA_ROOT`, but the audit still
found hard-coded Windows/old WSL paths in these categories:

- Scheduler wrappers and operational scripts: `V6/scripts/*.bat`,
  `V6/scripts/run_inference.sh`, `V6/scripts/run_gru_v2.sh`,
  `V6/scripts/run_gbdt_wf.sh`, `V6/scripts/rerun_20260812.sh`,
  `V6/scripts/launch_rerun_20260812.sh`
- Historical docs and agent handoff notes: `README.md`, `AGENTS.md`, `CLAUDE.md`,
  `docs/operations/scheduler-mitigation-snapshot-2026-08-25.md`
- Experimental or one-off diagnostics: `V6/experimental/fix_prices_index_column.py`,
  `V6/experimental/score_mamba_local.py`, `V6/experimental/rescore_all.sh`,
  `V6/experimental/diagnostics/*.py`
- Archived V5 patch scripts and notebooks retain old Windows or Colab paths by design.

These were not rewritten during migration because changing scheduler commands,
archival notes, or experiment scratch scripts could alter operational behavior or
historical record beyond the portability fix.
