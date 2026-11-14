# Knowledge Drift Checker Proposal

Design only; no checker is implemented in K0.

The future checker should parse a small machine-readable index and report:

1. Missing/dead referenced paths and malformed links.
2. More than one active canonical authority for a topic.
3. `Current_State.md` older than its declared observation TTL.
4. Deprecated/historical components referenced as active.
5. Branch-only paths represented as merged/deployed.
6. Generated inventory count/hash drift.
7. Contract values duplicated inconsistently in prose.
8. Strict readiness and historical-simulation readiness collapsed into one field.

It should be read-only, deterministic, CI-safe and return nonzero only for declared blocking classes. External scheduler/deployment checks should produce `UNKNOWN` when unavailable, not reuse stale success. Reports belong under `knowledge/_generated/` with observation timestamps.
