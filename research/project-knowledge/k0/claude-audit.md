# CLAUDE.md Audit

`CLAUDE.md` is a roughly 66KB legacy near-duplicate of the former agent guide. It contains useful historical detail but is stale relative to the condensed `AGENTS.md`, corrected V7 contracts, fetch-only scheduler state and Phase A branch progress.

## Risks

- Duplicated rules can diverge silently.
- Historical model/data assumptions can be mistaken for current authority.
- Long tool-specific instructions bury project invariants.
- A new agent may prefer the longer file even when its state is older.

## Recommendation

In K1, convert it to a thin wrapper that says to read `AGENTS.md` and the canonical knowledge index, preserving only genuinely Claude-specific interaction notes. Archive unique historical content before shrinking it. Machine contracts and tests must continue to outrank both files.
