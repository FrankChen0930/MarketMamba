# Task 1 Report: Operational contracts and schema migrations

## Scope

Added only the new V6/marketmamba/ops package and its focused tests. No legacy runner, provider, scheduler, audit document, or V6/models file changed. No runtime SQLite database was created in the repository.

## RED

Command:

    PYTHONPATH=V6 python3 V6/marketmamba/ops/test_contracts_schema.py

Observed result:

    ModuleNotFoundError: No module named "marketmamba.ops.contracts"

This confirmed the new contract behavior was absent before implementation.

## GREEN

Command:

    PYTHONPATH=V6 python3 V6/marketmamba/ops/test_contracts_schema.py -v

Result: 7 tests passed.

Additional verification:

    PYTHONPATH=V6 python3 -m py_compile V6/marketmamba/ops/__init__.py V6/marketmamba/ops/contracts.py V6/marketmamba/ops/schema.py V6/marketmamba/ops/test_contracts_schema.py
    git diff --check

Both completed successfully. A credential-pattern scan found no keys or private credentials.

## Files

- V6/marketmamba/ops/__init__.py: package marker and minimal public enums.
- V6/marketmamba/ops/contracts.py: validated mode, authority, state, and deterministic identity dataclasses.
- V6/marketmamba/ops/schema.py: SQLite v1 migration, checksum validation, and safe connection setup.
- V6/marketmamba/ops/test_contracts_schema.py: focused unit tests.
- .superpowers/sdd/derived-plan/task-1-report.md: this report.

## Schema and identity decisions

- The fixed operational identifiers are FETCH_ONLY, LEGACY_V62, and default policy ops-policy-v1.
- A logical run is canonicalized as requested Asia Taipei date, mode, authority, and policy version. Its SHA-256-derived ID therefore has no timestamp, host, PID, or other volatile field.
- Attempts, stages, artifacts, and checks derive deterministic SHA-256 IDs from their respective stable keys.
- SQLite v1 creates only schema_migrations, logical_runs, attempts, stage_runs, artifacts, checks, and leases. Nullable incident references reserve a minimal link without adding incident lifecycle tables.
- Every connection enables WAL, foreign keys, and a 5000 ms busy timeout. Applied migrations record version, checksum, and UTC ISO timestamp.
- Reopening an initialized ledger is idempotent. Newer or checksum-mismatched schemas fail closed.

## Tests

The focused suite covers stable logical IDs, validation of invalid identities, terminal states, deterministic child identities, v1 core tables, WAL, foreign keys, busy timeout, repeated open, migration checksum persistence, future-version rejection, and checksum-mismatch rejection.

## Commit

Atomic commit message: feat: add operational ledger contracts and schema.

## Concerns

The package deliberately provides contracts and schema initialization only. Writing run records, artifacts, checks, incident lifecycle, and lease acquisition belongs to later phases.
