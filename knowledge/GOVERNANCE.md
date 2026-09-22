# MarketMamba Knowledge Governance

> Status: `ACTIVE_KNOWLEDGE_GOVERNANCE_V1`

Knowledge governance keeps navigation aligned with repository evidence. It does not decide model, data, trading, deployment, or production questions.

## Completion sequence

For substantive changes:

1. Run the change's normal tests.
2. Run the knowledge-impact checker for the diff.
3. Review/update affected knowledge, or record `NO_KNOWLEDGE_CHANGE_REQUIRED` with a non-empty reason.
4. Run unified knowledge health.
5. Confirm the intended worktree is clean after commits.

Impact warnings never authorize automatic note rewrites. Acknowledgment records judgment; it does not suppress structural failures.

## Severity semantics

- `PASS`: canonical knowledge is structurally consistent.
- `WARNING`: knowledge remains usable, but an external observation is stale, an `UNKNOWN` remains, optional evidence is unavailable, legacy prose drifts, or a repin should be reviewed.
- `FAIL`: a canonical path/evidence identity is invalid, mandatory link is broken, active canonical ownership is duplicated, or protected current-state semantics contradict their authorities.

## Authority identity and lifecycle

Durable authority identity is `<commit>:<path>`. A branch label is optional navigation:

```text
feature/example@abc1234:path/to/contract.json
abc1234:path/to/contract.json
```

The commit/path must resolve. If a branch moves or is deleted while the commit remains valid, governance reports a warning rather than losing the authority. If identical evidence is reachable on `main`, governance reports `AUTHORITY_REPIN_CANDIDATE`; it never repins automatically.

### Supersession workflow

When one authority replaces another:

1. Establish the new evidence identity and validate it.
2. Update `knowledge/00_Project_Map/Authority_Map.md` and `authority-map.json` primary authority.
3. Preserve the old evidence; mark its role `SUPERSEDED`, `HISTORICAL_ONLY`, or `RESEARCH_EVIDENCE`.
4. Update Current State only if current state changed.
5. Update the affected domain note's semantic/navigation boundary.
6. Append an entry to `knowledge/_generated/authority-history.json`; never rewrite prior entries.
7. Regenerate indexes and run knowledge health.

The history starts with K2. Earlier changes are not reconstructed without verified evidence.

## Runtime observations

External state lives in `knowledge/runtime-observations/*.json`, with observation time, TTL, actor, method, and evidence reference. Markdown may project it but is not the only runtime database.

- `status` computes freshness without changing evidence.
- `record` requires an actual observation supplied by a human/agent and writes atomically.
- A stale record stays stale until a new observation is recorded. No tool refreshes a timestamp merely because it ran.

## Generated artifacts

`generate_indexes.py` owns declared generated navigation. Do not hand-edit files marked `GENERATED / DO NOT HAND EDIT`. Identical repository/evidence state must produce identical bytes. Volatile wall-clock time is excluded; commands accept explicit `--as-of` where freshness matters.

## History and Obsidian

History records why, not current truth. Use a decision record only for material project direction or contract changes. Research results remain authoritative through their machine artifacts. Significant incident notes retain durable lessons, not transient provider misses.

Obsidian is a private/free-form thinking layer. It may link to canonical repository paths, but personal vault paths never become repository invariants. See [Obsidian Bridge](OBSIDIAN_BRIDGE.md).

## CI readiness

```bash
python3 tools/project_knowledge/knowledge_health.py --ci --as-of YYYY-MM-DD
```

PASS and WARNING exit zero. FAIL exits nonzero. CI integration and local hooks are proposals until separately authorized.

## Same-change completion contract

For a substantial task, review whether current state, active objective, architecture decision, experiment conclusion, blocker, next action, canonical implementation, authority boundary, runtime assumption, or worktree/branch responsibility changed. If any changed, update Layer 1 and affected Layer 2 notes in the **same coherent commit** as source/config/contract changes. Promote a result only after checking its machine evidence; preserve older evidence with an explicit supersession link. `check_knowledge_impact.py` identifies candidates but does not make the decision.

If nothing changed, the handoff must say `knowledge_update: not_required` with a reason. This acknowledgment never suppresses a structural `FAIL`. Do not postpone memory maintenance to a later conversation. Keep Current State short: move completed phases and narrative into existing history/research evidence.

## Precreation and artifact lifecycle

Before creating a top-level directory, implementation, workflow, experiment/report tree, memory system, architecture, replacement version, backup, ZIP snapshot, or worktree, search existing canonical structures and Authority Map. Record why reuse cannot meet the contract. Names such as `final2`, `backup-new`, and `v2-new` do not establish an artifact version. New experiment outputs require a separate namespace and, where practical, a manifest with source commit, contract/config, input identity/hash, environment, split/support, model/checkpoint identity, output paths/hashes, and evidence class.

Shared `.gitignore` excludes checkout-local `.artifacts/`, `.runtimes/`, and `.worktrees/` containers; this prevents accidental staging across clones and does not authorize deletion. `environments/` remains source-controlled. `deliveries/` remains a local exclude in this checkout because it contains large, potentially unique research evidence; future shared rules require a manifest/source-boundary review before changing that status.

| Class | Meaning | Handling |
|---|---|---|
| `CANONICAL` | Reviewed active authority for a named scope | Commit source/contract; supersede explicitly. |
| `GENERATED` | Reproducible output derived from identified inputs | Record generator and identity; do not hand-edit generated indexes. |
| `TEMPORARY` | Partial, smoke or runtime material | Isolate namespace; never mistake existence for completion. |
| `ARCHIVE` | Preserved historical record | Keep provenance and supersession; no silent relabeling. |
| `EXTERNAL_LARGE_DATA` | Parquet, checkpoints, predictions, bundles or Drive data outside ordinary Git source | Keep manifests/hashes and location; no implicit commit, deduplication or deletion. |
| `UNKNOWN` | Identity or authority unverified | Fail closed until inspected. |

A file name, size, or matching row count is insufficient to deduplicate or promote. Storage cleanup is a separate authorized phase; this governance contract does not permit deletion.

## Git and worktree lifecycle

Git is required for source, configuration, experiment contracts, governance documents and coherent decisions. Stage explicit paths and commit each completed substantive unit. Never use `git add -A` by default. Keep large generated datasets/checkpoints/predictions out of governance commits. Preserve unrelated dirty/untracked files and report staged versus unstaged state. Push, merge and deploy are separate authorizations; completion alone grants none.

Before creating a worktree, check whether an existing checkout serves the bounded objective. Register purpose, branch, base commit, owner/task, created date (or `UNKNOWN`), current state, authority carried, merge dependency, artifact dependency, retired date and safe-removal preconditions in [Worktree Register](00_Project_Map/Worktree_Register.md). Update responsibility in the same change. A clean checkout may still carry unique authority; retirement requires retained branch/history, no unique required artifacts, resolved manifest/path dependencies and explicit review. The eight existing worktrees remain untouched by this consolidation.

## Runtime and handoff

Development/research uses the WSL-native checkout `/home/frank/projects/MarketMamba`. Existing Windows Scheduler/data operation references `D:\Desktop\work\ProjectForMe\MarketMamba` and invokes WSL; only timestamped observations establish the live action. Colab is a separate exact-lock training runtime. Record `MARKETMAMBA_DATA_ROOT`, symlink realpath, environment and source commit before asserting cross-runtime equivalence. No runtime or data move follows from this contract.

Every substantial task closes with: Completed, Changed, Decisions, Evidence, Remaining, Blockers, Next recommended action, Project memory updated, Authority map updated, Git status, Commit, Push, Worktree, Runtime/environment. Write `N/A` when inapplicable. The final response plus relevant canonical state update suffice; avoid a new handoff file for every task.
