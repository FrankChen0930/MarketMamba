# Obsidian Bridge Contract

## Repository owns

- Current project truth and readiness language.
- Authority ownership and supersession state.
- Domain boundaries and stable invariants.
- Reviewed decisions that change project direction or contracts.
- Machine-readable governance inputs and health outputs.

## Obsidian owns

- Private notes, papers, excerpts, backlinks, brainstorming, hypotheses, and daily thinking.
- Unreviewed discussions and alternative interpretations.
- Personal navigation that is useful to the owner but not a repository invariant.

## Canonical link convention

An Obsidian note may point to repository truth using a repository-relative reference:

```text
Canonical: MarketMamba repository → knowledge/01_Domains/Models.md
Authority: MarketMamba repository → knowledge/00_Project_Map/Authority_Map.md
```

Do not encode a Windows drive, WSL distribution, username, vault root, or `file://` URL as the canonical identity. The repository path is portable; the local vault path is not.

## Promotion protocol

Moving an idea into current truth requires a reviewed repository change to the project map/domain note and, when applicable, a machine contract. Copying text into Obsidian or adding a backlink does not promote it.

The optional exporter is read-only. It may list candidate note-to-canonical links, but never modifies, moves, or imports vault content.
