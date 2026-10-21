# Portfolio History UX refinement — H1 presentation scope

Status: IN_PROGRESS / UNMERGED / NOT_DEPLOYED. Base `5a2bc6fcee4c0bf78190a6ac74d11f53f8becbd4`.
User explicitly approved UX-01–04: default concise dashboard and advanced research
view, presentation-only formatting and conservative event descriptions. No H0/H2.
Original PORT artifacts/report remain historical and byte-preserved.

## Skills preflight / design decision

Project-local `.agents/skills` is the conventional skill discovery location; no
existing equivalent was found. No global settings changed. Upstream instructions
are frozen, subordinate guidance, not financial or project authority.
[Fixed-source review/hash inventory](../research/v7/portfolio-ux-skill-preflight.json).
Anthropic frontend-design `dbd4588f9e1033efb41dad4bef2f7947c8993d44`;
Vercel vercel-react-best-practices v1.0.0 `063bee94c3f4df8453406c830b0a7df0f2860278`.
Only documentation installed; no upstream commands/hooks executed or credentials used.

Design retains tokens: base #0d1117, panel #161b22, primary #e6edf3,
secondary #8b949e, cyan #00d4ff, amber #ffa500. Existing sans/mono fonts.
Left-aligned compact header + mode buttons → selected-event summary → NAV chart
and dated timeline → positions/pending. Advanced view adds exact table, raw selected
event and source/spec. The timeline encodes actual sequence; decoration and motion
are restrained. Mobile stacks sections; exact tables scroll within their panel.
Review against brief: avoid a new visual brand or marketing hero. Retain dark app
shell, prioritize event context and explicit synthetic status rather than giant
numbers or ornamental cards. No new library, font download or financial metric.

## Verification / closure

Pending implementation, independent review and real Chrome desktop/mobile acceptance.
