# K1 Knowledge Structure Proposal

Recommendation: hybrid. Canonical project knowledge lives in Git; external Obsidian remains a personal thinking and navigation layer that links back to repo authorities.

```text
knowledge/
├── 00_Project_Map/
│   ├── Current_State.md
│   └── Authority_Map.md
├── 01_Domains/
│   ├── Data.md
│   ├── Features.md
│   ├── Models.md
│   ├── Training.md
│   ├── Labels.md
│   ├── Universe.md
│   ├── Execution.md
│   ├── Portfolio.md
│   ├── Production.md
│   ├── Operations.md
│   └── UI.md
├── 02_History/
└── _generated/
```

`Current_State.md` should contain observation time, production path, research baseline, strict/historical gate statuses, active experiment and branches, blockers, next decision, and authority links. `Authority_Map.md` should use `Topic → canonical path → status → supersedes → secondary evidence → conflicts → notes`.

Each domain note answers what/why, authority, implementation, inputs/outputs, boundary, current status, known issues and do-not-use. Semantic owners update notes in the same change that alters a contract or boundary. Generated files are never hand-edited.

Obsidian may retain private hypotheses, reading notes and rich backlinks. It must not contain a second independently maintained “current truth.”
