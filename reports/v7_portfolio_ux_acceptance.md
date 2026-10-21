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

UX-01–03 share a single page/helper change, avoiding parallel data models.
Independent review found no Critical/Required. Seven presentation tests and eight
API client contract tests PASS; existing frontend8, Portfolio67, Python API8 +
real HTTP/restart1 and governance23 PASS. Build PASS (inherited large-chunk warning).
The same schema/API/engine/exporter/fixture bytes and original PORT evidence remain
unchanged: [identity and test index](../research/v7/portfolio-ux-evidence.json).

Formatting is presentation-only, four decimal places (quantity six), half away
from zero using decimal text/BigInt; no float precision loss before rounding.
Tiny nonzero displayed as approximate zero. Original exact strings remain in
advanced mode/raw selectable JSON. Comparisons use full precision, not formatted
values. Unknown never becomes numeric zero. Event labels are Chinese translations;
changes describe observed adjacent states only, never reasons, returns or fills.

General: compact selected-event summary, dated interactive chart/timeline,
positions/pending, visible fixture/source class and STOP/not-admitted notice.
Advanced: same selected event, exact decimals, complete event table, raw selected
API event/state, schema/spec, engine version, journal/manifest/final-record hashes
and source commit. Does not invent raw Journal payload absent from API.
Mode switching uses the same validated result, preserves selection and does not
request another API result. Native buttons/select and SVG Enter/Space support keyboard.

Focused changed-source lint PASS. **Full frontend lint remains FAIL:58 errors,
3warnings, byte-identical output after path normalization to fixed5a2bc6f.** This
is an inherited repository gate, not claimed PASS and not expanded into this UX
scope. The project config excludes mjs rules; new helpers/driver get explicit
no-unused-vars/no-undef with Node globals. Vendored upstream markdown hard-break
whitespace is retained byte-for-byte; diff check excludes that pinned third-party
content, and does not silently normalize it.

UX-04 real browser evidence and final checkpoint are being consolidated.
