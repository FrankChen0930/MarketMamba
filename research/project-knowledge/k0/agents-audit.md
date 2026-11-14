# AGENTS.md Audit

## Finding

The current condensed `AGENTS.md` is a useful operating guide and is materially fresher than `CLAUDE.md`. It should remain navigation/policy, not a duplicated database of empirical project state.

## Strengths

- Separates production, correctness and research semantics.
- Records Colab build-once matrix, resume, checkpoint and durable-output expectations.
- States preservation/destructive-operation habits and points to evidence.
- Captures strict STOP versus historical-simulation PASS and 48-feature/no-graph constraints.

## Missing or weak rules

- No single canonical `Current_State.md` or human authority index to link.
- External scheduler observations need an `observed_at` timestamp and expiry rule.
- Branch-local implementation must always be labeled unmerged/not deployed.
- Knowledge changes lack an explicit impact checklist and drift validation command.
- Tool-specific assumptions should be separated from project invariants.

## Duplication and ambiguity

- Some current-state facts overlap Obsidian and research prose.
- Exact counts/branch names will age quickly if kept inline.
- “Current” must distinguish repository contract, external runtime and local uncommitted candidate.

## K1 direction

After K1 creates canonical repo knowledge, reduce `AGENTS.md` to: safety invariants, development habits, required validation, and links to `Current_State.md`, `Authority_Map.md`, domain notes and generated drift reports. Do not change it during K0.
