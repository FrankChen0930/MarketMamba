# MarketMamba K1 Project Map Bootstrap

## Scope

This slice creates only the first canonical-facing project-map layer:

- `knowledge/00_Project_Map/Current_State.md`
- `knowledge/00_Project_Map/Authority_Map.md`
- `knowledge/00_Project_Map/current-state.json`
- `knowledge/00_Project_Map/authority-map.json`

All four are `PROVISIONAL_CANONICAL_PENDING_HUMAN_REVIEW`. K0 artifacts remain preserved as discovery evidence. No domain encyclopedia, Obsidian migration, agent-guide rewrite, business/runtime/model/data change, merge, push, deployment, or GPU execution is included.

## Evidence Basis

The project map was derived from the K0 report, structure proposal, AGENTS/CLAUDE audits, authority/boundary/contradiction/runtime/research-lineage maps, and direct verification of named correctness, E1 and Phase A branch artifacts. Machine contracts and explicit commit-qualified paths outrank prose.

## Unresolved Unknowns

- Live API and frontend deployment/traffic.
- Manual publication activity outside the disabled full V6.2 schedule.
- Whether external Colab/Drive contains a newer formal E1 result.
- Scheduler state after its seven-day observation TTL.

## Contradictions Intentionally Preserved

- Fetch-only scheduler reality versus README/OVERVIEW full-runtime implications.
- Corrected E5 draft 32/8 architecture versus completed 64/32 authority.
- Historical-simulation `PASS` versus strict Phase 0 `STOP`.
- Source-present API/UI versus unknown live deployment.
- Phase A implemented on a feature branch versus not merged/not deployed.
- Seed43 loadable checkpoint versus incomplete formal result.

## Human Review Points

1. Confirm the seven-day TTL for scheduler/deployment observations.
2. Accept or amend the Phase A state and `HUMAN_REVIEW_REQUIRED` blocker wording.
3. Confirm that E1 remains `SPECIFIED / PACKAGED / NOT VERIFIED COMPLETE` until imported final artifacts pass validation.
4. Confirm that `E5-PIT-Clean-v1` and the commit-qualified authorities are the intended current research navigation layer.
5. Approve promotion from provisional to final canonical status.

## Validation

`tools/project_knowledge/validate_project_map.py` checks deliverable presence, JSON parsing, relative Markdown links, branch/commit/path references, authority classes, unique topics, `UNKNOWN` null-authority behavior, Markdown/JSON status alignment, required cross-reference, and key state invariants.

## Next Recommended K1 Slice

After Human approval, update `AGENTS.md` and reduce `CLAUDE.md` to thin navigation wrappers pointing to this project map. Do not begin full domain-note migration in that slice.
