# Portfolio History UX refinement — H1 presentation scope

Status: COMPLETED_VERIFIED (fixture UI scope) / UNMERGED / NOT_DEPLOYED. Base `5a2bc6fcee4c0bf78190a6ac74d11f53f8becbd4`.
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

UX-04 is verified. Chrome155 actual desktop1440×1000 and mobile390×844 viewport
emulation: all7events via select/timeline/real pointer chart selection; both modes,
exact advanced values and raw JSON; mode selection retained with zero additional
API GETs; Enter/Space/Home; no whole-page horizontal overflow or runtime exceptions.
The same driver is RED on fixed original PORT default-mode assertion (no dual mode).
Missing/corrupt/network UI checks are **CDP-injected**, not falsely described as
actual file corruption; actual HTTP corrupt/missing/restart rejection is covered
by the separately rerun Python E2E. Browser error checks use mobile viewport.
Only Chrome tested; no physical-phone or general accessibility certification.

Independent gpt-6-astra code review and independent Chrome execution/visual review
found no Critical/Required product issues. Product repair rounds UX01/02/03 =0/0/0;
UX04 has2 bounded **test-driver-only** corrections (CDP Enter text and scroll race),
zero product fixes. No additional Human decision/intervention after this grant.

## Screenshots and exact evidence

- [Desktop general](../research/v7/portfolio-ux-desktop-general.png)
- [Desktop advanced](../research/v7/portfolio-ux-desktop-advanced.png)
- [Mobile general](../research/v7/portfolio-ux-mobile-general.png)
- [Mobile advanced](../research/v7/portfolio-ux-mobile-advanced.png)

[Evidence index](../research/v7/portfolio-ux-evidence.json) preserves raw tests,
RED/GREEN, browser receipts, screenshot/source hashes, fixed-baseline lint
comparison and14 byte-identical original engine/API/fixture/evidence identities.
All78 installed skill files match pinned source SHA256 inventories. No original
PORT report, screenshot, artifact, API or accounting source was overwritten.

## Local viewing / reproducibility

From this UX worktree root, use an environment with the existing FastAPI/uvicorn
dependencies (verification reused `/tmp/marketmamba-port-history-venv`). No install
or dependency change is necessary in the existing environment:

```bash
cd /home/frank/projects/MarketMamba/.worktrees/v7-portfolio-ux
export V7_PORTFOLIO_HISTORY_ENABLED=1
export V7_PORTFOLIO_HISTORY_PATH="$PWD/research/v7/portfolio-history-artifact.json"
export V7_PORTFOLIO_HISTORY_SHA256=d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6
/tmp/marketmamba-port-history-venv/bin/python -B -m uvicorn \
  app.backend.portfolio_history_local:app --host 127.0.0.1 --port 8000 --no-proxy-headers
# Second terminal, UX worktree:
cd app/frontend
VITE_V7_PORTFOLIO_HISTORY=1 npm run dev -- --host 127.0.0.1 --port 5178 --strictPort
```

Open `http://127.0.0.1:5178/v7/portfolio-history`. Refresh starts general mode;
mode switches retain the current event. API pins the original fixture bytes.
For browser reproduction, launch existing Chrome with a dedicated TEMP user-data-dir
and `--headless=new --remote-debugging-port=9229`; do not reuse a personal profile.
On the same host as Chrome run Node24 with the appropriate local/WSL UNC paths:

```bash
node app/frontend/src/api/portfolioHistoryUx.browser.mjs \
  http://127.0.0.1:5178 research/v7/portfolio-history-artifact.json /tmp/ux-browser \
  http://127.0.0.1:9229
node app/frontend/src/components/portfolioHistoryPresentation.test.mjs
node app/frontend/src/api/portfolioHistory.test.mjs
```

Browser script writes only screenshots/evidence; no source data mutation.
The original PORT browser harness is preserved as historical acceptance; the new
harness understands the approved dual-view presentation. Runtime exceptions=0,
but legacy shell market routes404 on the isolated API and inherited build chunk
warning remain; this is not complete legacy-dashboard or production acceptance.

## Handoff / stop

Local commits: preflight `bc0526b`, shared UX01–03 `764dfbf`, UX04 is the containing
acceptance commit. Branch `development/v7-portfolio-ux`; base5a2bc6f; main83a6a2e
unchanged. Canonical Project Memory/Authority Map and sole research-program backlog
updated; no parallel management system. API8000, both Vites and dedicated Chrome
stopped. No merge/push/deploy/GPU/data admission; Strict Phase0 STOP persists.

Ready for a new Human Product Slice Review; not approved by Agent on Human's
behalf. H0 integration and H2 real data/financial policy remain separate decisions.
No new financial definition is required for this presentation slice; weights,
returns/risk and currency policy remain undefined. Full-repo lint cleanup is an
inherited separate engineering issue, not silently repaired or waived here.
Stop now; no authorized UX READY work remains.
