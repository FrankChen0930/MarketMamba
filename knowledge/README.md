# MarketMamba Knowledge

This is the existing K1/K2 project memory system. Layer 1 is Current State and Authority Map; Layer 2 is stable domain/governance knowledge; Layer 3 is history and machine evidence. It navigates to technical authority; it does not replace contracts, tests, code, or artifacts. `AGENTS.md` and `CLAUDE.md` are the operating contract outside these layers.

## Reading Order

1. [Current State](00_Project_Map/Current_State.md) — what the project is now.
2. [Authority Map](00_Project_Map/Authority_Map.md) — which evidence wins for each topic.
3. [Domain Notes](01_Domains/) — subsystem meaning, boundaries, invariants, and important paths.
4. Machine contracts, tests, code, and artifacts linked from those notes.
5. [History](02_History/) — why the project arrived here; non-authoritative unless explicitly promoted.

Governance lifecycle, severity, supersession, runtime freshness, and completion workflow are defined in [Knowledge Governance](GOVERNANCE.md).

## Domain Index

- Core correctness/research: [Data](01_Domains/Data.md), [Features](01_Domains/Features.md), [Universe](01_Domains/Universe.md), [Execution](01_Domains/Execution.md), [Labels](01_Domains/Labels.md), [Models](01_Domains/Models.md), [Training](01_Domains/Training.md), [Research](01_Domains/Research.md)
- System delivery: [Portfolio](01_Domains/Portfolio.md), [Operations](01_Domains/Operations.md), [Production](01_Domains/Production.md), [UI](01_Domains/UI.md)
- Generated navigation: [Domain Dependencies](_generated/domain-dependencies.md), `important-paths.json`, `coverage.json`, and drift reports.

## Maintenance

Update a domain note in the same change when its semantics, authority, invariant, boundary, blocker, or important path changes. Run:

```bash
python3 tools/project_knowledge/validate_project_map.py
python3 tools/project_knowledge/validate_agent_guides.py
python3 tools/project_knowledge/validate_domains.py
python3 tools/project_knowledge/check_drift.py --as-of YYYY-MM-DD
python3 tools/project_knowledge/check_knowledge_impact.py HEAD~1..HEAD
python3 tools/project_knowledge/knowledge_health.py --ci --as-of YYYY-MM-DD
```

See [Obsidian Boundary](OBSIDIAN_BOUNDARY.md) and [Obsidian Bridge](OBSIDIAN_BRIDGE.md) before linking personal notes.
