# MarketMamba K2 Knowledge Governance

## Integration result

Git topology proved that `docs/project-knowledge-k1-domains@2c6c6f3` already contains the complete K0 discovery, K1 project map, K1 agent-guide integration, and K1 domain lineage. No repeated merges were needed.

Main is dirty with user-owned research and UI candidates. K2 therefore uses the clean successor branch `integration/project-knowledge-k1` in an isolated worktree. Local main was not modified or merged. No unrelated research, E1, Phase A, UI, or production branch was integrated.

## Governance architecture

`knowledge/GOVERNANCE.md` defines PASS/WARNING/FAIL, completion order, durable authority identity, supersession, runtime freshness, deterministic generation, history, Obsidian ownership, and CI-ready behavior.

Governance is detection and lifecycle infrastructure. It never rewrites canonical notes, changes business semantics, promotes research, or observes external systems by assumption.

## Impact checker

`check_knowledge_impact.py` accepts a Git diff range, staged changes, or explicit changed-file inputs. Rules in `knowledge/knowledge-impact-map.json` map semantic surfaces to:

- `REVIEW_REQUIRED`
- `UPDATE_IF_SEMANTIC_CHANGE`
- `NO_KNOWLEDGE_IMPACT`

`NO_KNOWLEDGE_CHANGE_REQUIRED` requires a non-empty reason. The acknowledgment is recorded in output and does not bypass structural validation or automatically update any note.

Focused tests cover feature manifests, the production runner, portfolio code, governance-tool changes, and acknowledgment behavior.

## Runtime observations

The accepted K0 Windows Scheduler inspection is stored as timestamped evidence in `knowledge/runtime-observations/scheduler.json`. `runtime_observations.py status` computes `FRESH` or `STALE_OBSERVATION`; `record` requires an explicit actor, method, evidence reference, observed timestamp, TTL, and non-empty value object, then writes atomically.

No timestamp is refreshed by running status. The accepted observation expires at `2026-09-27T01:30:00+08:00`; a 2026-09-28 test correctly produces WARNING.

## Authority supersession

Authority references support durable `commit:path` identities plus optional branch labels. Invalid commit/path is FAIL. A moved/deleted branch label is WARNING when the durable commit still resolves. An identical main artifact becomes `AUTHORITY_REPIN_CANDIDATE`, never an automatic rewrite.

The append-oriented authority history begins at K2. It contains one verified event establishing governance authority; no earlier history was fabricated.

## Generated navigation

`generate_indexes.py` deterministically owns:

- Tier A index
- domain coverage
- important paths/authority references
- domain dependencies
- authority-history summary
- legacy-documentation drift inventory

Two consecutive generations produced byte-identical SHA-256 values. Generated artifacts are marked `GENERATED / DO NOT HAND EDIT`. Volatile wall-clock timestamps are excluded from deterministic navigation.

## History protocol

The history layer now has minimal `Decisions`, `Experiments`, `Incidents`, and `Milestones` directories plus decision, experiment, and incident templates. Records are reserved for material direction, durable research navigation, and significant root-cause lessons—not every commit or transient provider miss.

## Obsidian bridge

`OBSIDIAN_BRIDGE.md` defines the repository as owner of current truth/authority/domain boundaries and Obsidian as owner of private notes, papers, hypotheses, backlinks, and daily thinking. Canonical links use repository-relative paths; personal Windows/WSL/vault paths are not invariants.

The optional exporter is read-only. No vault was explicitly supplied in this worktree, so the generated result is truthfully `SKIPPED`; no vault content was modified or imported.

## Drift status

At the fixed acceptance date `2026-09-20`:

- Project Map: PASS
- Agent Guides: PASS
- Domains: PASS
- Governance: PASS
- Authority Drift: PASS
- Runtime Freshness: PASS
- Generated Index: PASS
- Legacy Docs: WARNING
- Overall: WARNING

The warning is expected: README, OVERVIEW, and PROJECT still contain legacy full-daily, feature-count, and architecture claims. K2 records exact line-level findings and does not rewrite those documents.

Explicit failure-injection tests prove that health fails when:

- feature authority changes but `Features.md` retains the old primary authority;
- Current State and Authority Map disagree about E1;
- Phase A is represented as deployed while current authority says not deployed.

The focused K2 suite contains 13 passing tests in total.

## Remaining unknowns

- Live API/frontend deployment and service ownership remain `UNKNOWN`.
- E1 remains specified/packaged but not verified complete.
- Strict Phase 0 remains `STOP`; verified executable labels remain 0/0.
- Phase A remains unmerged/not deployed and Task 4 correctness approval is not established.
- The scheduler observation will become stale unless actually re-observed.
- Obsidian candidate links remain `SKIPPED` until a readable vault is explicitly supplied.

## Recommended next step

Stop at K2 governance review. Review the integration branch and the declared legacy-documentation warnings. If accepted, decide separately whether and how to integrate the knowledge-only lineage into local main without overwriting its dirty user work. Do not begin K3 or business work automatically.

## Scope confirmation

No E1 execution, model training, GPU use, data modification, inference change, Phase A implementation, scheduler change, API/frontend behavior change, deployment, portfolio behavior change, notification, push, or local-main merge occurred.
