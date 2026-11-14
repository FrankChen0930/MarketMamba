# Claude Instructions

Canonical project-agent policy:

1. Read [`AGENTS.md`](AGENTS.md).
2. Before substantial work, read [`knowledge/00_Project_Map/Current_State.md`](knowledge/00_Project_Map/Current_State.md).
3. Then read [`knowledge/00_Project_Map/Authority_Map.md`](knowledge/00_Project_Map/Authority_Map.md).
4. Follow the relevant machine contract, artifacts, executable tests, and available domain note.

Follow `AGENTS.md` for safety, authorization, testing, Git/worktree discipline, evidence classes, authority conflicts, knowledge-impact rules, and validation before completion. Do not maintain a second current-state database in this file.

Before creating any new directory, implementation, workflow, experiment/report tree, memory system, architecture, replacement version, backup, ZIP snapshot, or worktree, perform the `AGENTS.md` reconnaissance-before-creation check and record why existing canonical structure cannot serve it. For substantial work, update Project Memory in the same coherent change and use the fixed handoff fields in `AGENTS.md`.

## Claude-specific Notes

- Optional Claude role definitions live in `.claude/agents/{builder,verifier,archivist}.md`.
- The optional main-session coordination protocol lives in `.claude/skills/advisor/SKILL.md`; it is not a fourth agent role.
- Use these assets only when the active Claude environment supports them. They never override `AGENTS.md`, the canonical project map, user instructions, or machine evidence.
- Keep builder and verifier roles independent when using that workflow; share repository evidence, not persuasive cross-agent discussion.

The former long Claude guide is preserved at [`docs/archive/legacy-claude-project-guide.md`](docs/archive/legacy-claude-project-guide.md) as `HISTORICAL_ONLY / NON_AUTHORITATIVE`.
