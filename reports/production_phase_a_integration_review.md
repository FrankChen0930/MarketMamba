# Phase A — Final Engineering Integration Review

2026-10-21. Engineering verdict: **GO for the bounded development candidate**.
Final Phase A acceptance and development-main merge remain **Human decisions**.
No merge, push, deployment, schedule change, live fetch or training was executed.
Strict Phase 0 remains **STOP**.

## Fixed review identity and Human provenance

Reviewed engineering source: `1c5250e67461456b7ca0b361f8da3d9cb3f22514`,
tree `72284035f4a7f32904896e8cbeaf08c7c0e11ab2`, branch `integration/phase-a-task5`.
This closing commit adds only decision/governance/tests/documentation; it does
not change Snapshot, operational implementation, replay tests or raw evidence.
The final full commit is supplied in the handoff. Resolve this report's containing
commit with `git log -1 --format=%H -- reports/production_phase_a_integration_review.md`.

- Task 4: explicit Human limited engineering approval for
  `1e119ef855f8fd319f774162beca0bd729d4b921`; original decision record retained.
- Task 5: explicit Human limited **offline** engineering approval for fixed
  `1c5250e`; see `knowledge/02_History/Decisions/20261021-task5-human-approval.json`.
  This is transcription of the current Human message, not agent-generated
  acceptance or a cryptographic identity proof.
- Original Task 4/5 reports and acceptance artifacts retain their historical
  pre-decision status. Current State and separate decisions supersede only
  pending-approval navigation, not historical bytes or evidence limitations.

Recomputed fixed-file SHA-256:

| Artifact | SHA-256 |
|---|---|
| Task 4 evidence index | `ed254b828683a83035fe791117266d9ea9ed1a56ac477e9a4125dcbf9ff380ec` |
| Task 5 evidence index | `51e6b00c3dde35e583ed32c466286002d849a518aee24810b5216ae16f056d98` |
| Task 5 acceptance | `439dfcc3b1be1286a1ae986aef753285caa6193aa653681f96156465038ae7ca` |
| Task 5 review package | `bc1a61dc9055c43de7ce0d36679cb8165fa8f457381e506991428ed438dcbbab` |

All four Task 5 indexed file hashes match fixed Git and checkout bytes.
Task 4's 12 raw hashes and 36 embedded generation digests recompute correctly;
the raw records are unchanged between source and candidate.
Task 5 has 14 records (10 fixed replays plus four process scenarios).
Benchmark arrays recompute to 98.475144 ms median added latency and
136.371430 ms maximum shadow time. Original shared-host limitations remain.
Historical test totals agree between the fixed acceptance/report and prior
execution/review records in this conversation; these totals are not newly run
tests in this final review.

## Task 1–5 acceptance map

Task briefs were read from the preserved original Operations checkout's
`.superpowers/sdd/derived-plan/task-{1,2,3,4,5}-brief.md`. Tracked roadmap:
`research/production-system/implementation-roadmap.md`. The ignored briefs are
historical specification context, not new runtime/approval authority.

| Task | Existing requirement and evidence | Current limited status |
|---|---|---|
| 1 | Stable typed identities, migration idempotence/rollback/future-version refusal; `test_contracts_schema.py` | Implemented/reviewed; no newly inferred Human approval |
| 2 | WAL, atomic stage/artifact/check commit, retry history, leases/stale fencing; `test_store_lease*.py` | Implemented/reviewed |
| 3 | Original callback result/exception, retries and legacy success preserved; shadow failure/conflict cannot grant authority; `test_shadow.py`, fetch-only regression | Implemented/reviewed; live production not certified |
| 4 | Committed-only consistent projection, atomic output/read-only CLI, sanitizer and distinct-artifact aggregation; projection and real-process isolation tests | Human-approved limited engineering at `1e119ef` |
| 5 | Fixed replays, durable RUNNING/SIGKILL/new-attempt recovery, concurrent ownership, schema/results/acceptance/report/overhead | Human-approved limited offline engineering at `1c5250e` |

The original Task 5 brief also names an experimental suite. Prior work ran 38
non-training portfolio/engine/journal/health tests, not the full suite containing
training. Human's present decision explicitly excludes full experimental and
training certification; the narrow GO does not silently mark that broad suite
complete or redefine research acceptance.

## Ancestry, compatibility and remaining repairs

Main remains clean at `50c5b785e8c373913c3301a056497eb955f36681`, an ancestor of
the candidate. Snapshot base: `63a040d306e402dbc22e8592f57f034fd4a9a863`.
The 17 destination/source pairs in `git log 63a040d..7e3c212` each retain full
`-x` provenance and have equal stable patch-id; every source is in Task 4 ancestry.
The source sequence is preserved in the fixed Task 5 review package.
No uncommitted source or unrelated research lineage was bulk-imported.

Snapshot source/tests equal `63a040d` byte-for-byte. Operations implementation
and fetch-only entrypoint equal Task 4 `1e119ef`; only Task 5 tests were added.
Snapshot is a separate research primitive, not an operational-ledger dependency:
the joint gate checks coexistence/regressions, not a nonexistent direct pipeline.
Known sanitizer/aggregation/lease fixes are included. Other referenced branch
commits were checked; no necessary unmatched repair was identified. This is not
a claim about undiscovered defects.

Prior joint evidence: Operations 81, PIT/Snapshot 47, fetch-only 3, governance 19,
non-training portfolio 38 passed. The independent final engineering reviewer
checked code/test coverage, identities, history and arithmetic; no Required or
Critical finding and no missing test requiring production changes.
Those suites were not rerun solely to increase counts.

The only behavioral update here is the existing project-map validator accepting
the newly recorded **scoped** Task 5 state after fixed decision/evidence checks.
TDD: the new valid-decision test first failed with the old “no Task 5 approval”
guard. Final governance regression: **23/23 PASS**, 0.086 s, including missing
record, seven mutated fields, corrupt fixed evidence and generic APPROVED refusal.
This validator checks recorded consistency, not independent Human identity or
an execution grant. Existing authority hierarchy and production policy are unchanged.

Memory cleanup distinguishes current approvals from historical pending states,
updates current worktree count, and preserves old runtime observation timestamps.
No new memory architecture or database was created.
The implementation navigation now points to fixed `1c5250e` (still UNMERGED),
with `f5b0ff3` retained as historical source, rather than presenting the older
pre-aggregation-fix source as the latest candidate.
Final independent governance review: 23/23 PASS (0.073 s), project-map 38 topics
with zero broken links, and diff-check PASS; no Required/Critical finding.
Generated indexes were refreshed with the existing generator. Final knowledge
health: no FAIL; Governance, Authority Drift and Legacy Docs remain WARNING,
while Project Map, Agent Guides, Domains, Runtime Freshness and Generated Index
pass. Warning status does not refresh stale external runtime evidence.
Runtime: existing Python 3.12.3, CPU-only. Candidate `Data/processed_v6` is absent;
development main's data link resolves to the separate Windows operational data
root, which was not read or modified by this review.

## Guarded integration procedure — only after separate Human approval

Recommended: fast-forward the **fixed final candidate**, not cherry-pick the old
Operations branch again. Main is currently an ancestor, so no conflict resolution
or merge commit is necessary. Approval must name the full final commit from
the handoff; do not substitute a moving branch HEAD.

Before any merge:

1. Confirm candidate/main clean, original source identities unchanged, and the
   intended candidate descends from `1c5250e`.
2. Run the commands below from the candidate. Do not set evidence-output env
   variables: existing historical artifacts must not be overwritten.
3. Verify fixed evidence hashes and governance health; inspect the complete
   `git diff --stat 50c5b78..<approved-commit>` and preserve both decisions.
4. If main moved, any test fails, identity differs or a worktree is dirty, stop.
   Do not auto-rebase, reset, resolve semantic conflicts or weaken gates.

```bash
set -eu
cd /home/frank/projects/MarketMamba/.worktrees/phase-a-task5
# Set to the full final commit separately approved by Human, not branch HEAD.
candidate='<HUMAN_APPROVED_FULL_FINAL_COMMIT>'
test "$(git rev-parse HEAD)" = "$candidate"
test -z "$(git status --porcelain)"
test "$(git rev-parse main)" = 50c5b785e8c373913c3301a056497eb955f36681
git merge-base --is-ancestor 1c5250e67461456b7ca0b361f8da3d9cb3f22514 "$candidate"
git merge-base --is-ancestor main "$candidate"
git diff --check main.."$candidate"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6 python3 -B -W error::ResourceWarning -m unittest discover -s V6/marketmamba/ops -p 'test_*.py' -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6 python3 -B -m unittest V6.scripts.test_v62_fetch_only -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/bin/python -B -W error::ResourceWarning -m unittest V6.experimental.v7_source_snapshot_test V6.experimental.v7_temporal_contract_test V6.experimental.v7_executable_labels_test V6.experimental.v7_ohlcv_provenance_test V6.experimental.v7_historical_security_master_test V6.experimental.v7_historical_label_materialize_test -q
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest tools.project_knowledge.test_knowledge_governance -q
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/project_knowledge/knowledge_health.py --ci --as-of 2026-10-21
```

Use the actual review date for a later health run; do not refresh old external
observations. Verify the existing CPU interpreter still exists; no implicit
package installation or environment replacement is authorized.

Only after the above succeeds and Human grants merge:

```bash
cd /home/frank/projects/MarketMamba
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = 50c5b785e8c373913c3301a056497eb955f36681
test -z "$(git status --porcelain)"
git merge --ff-only "$candidate"
test "$(git rev-parse HEAD)" = "$candidate"
```

Afterward rerun the same bounded commands from main (checkout-location
acceptance, not training), verify hashes and cleanliness. Record actual main
integration in existing memory without changing historical Human scopes or
production status. If verification fails, preserve evidence and stop; no
automatic reset or rollback that could discard work. No push follows implicitly.
The commands above have **not** been executed to change main during this task.

## Remaining restrictions and next valuable V7 work

GO covers a consistent, reproducible **limited offline development candidate**.
No blocking correctness/provenance issue was found within that scope.
Final Phase A closure still requires Human judgment; production readiness is
NO-GO/unverified. Stat fingerprints are not content certification; no live
provider, host power-loss, long-ledger SLA, full pipeline restart, full training
suite or PIT/label truth was certified. Runtime observations remain stale.

After the integration decision, prefer the already identified **E1 frozen
matrix/checkpoint identity recovery and safe offline replay readiness** work,
not automatic Phase B expansion or new training. First inventory existing
manifests and surviving artifacts, verify exact hashes/contracts/split/support,
and stop on missing or untrusted checkpoint state. This addresses the recorded
inconclusive final-support epoch diagnosis without changing hypotheses or
test-set purpose. Loading/replaying checkpoints and any causal-regime research
design need their own bounded authorization; no such work starts here.
