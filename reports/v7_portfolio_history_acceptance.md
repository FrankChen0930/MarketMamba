# H1 Portfolio History product slice — acceptance and local handoff

Scope: fixture-only, CPU-only, read-only. H1 engineering authorization is in
`knowledge/02_History/Decisions/20261021-portfolio-history-h1.json`.
UNMERGED / NOT_DEPLOYED. Strict Phase 0 STOP, research_admission=false.
No market/model data, trading, H2 policy or research correctness approval.

## Local launch contract

Use the **repository root of this isolated worktree**, not the legacy backend cwd.
The dedicated local entry point loads the same additive router without legacy
startup imports, schedulers or data clients. A Python environment with FastAPI,
httpx (tests) and uvicorn is required; verification used 0.115.12 / 0.28.1 / 0.34.3.
Node 24.19.0 and the existing frontend lock/dependencies were used.

```bash
cd /home/frank/projects/MarketMamba/.worktrees/v7-portfolio-history
python -B -m V6.experimental.v7_portfolio_fixture \
  --directory .artifacts/my-fixture-run --source-commit "$(git rev-parse HEAD)"
# The explicit directory must be new. The command prints the artifact and SHA-256.
export V7_PORTFOLIO_HISTORY_ENABLED=1
export V7_PORTFOLIO_HISTORY_PATH="$PWD/.artifacts/my-fixture-run/history.json"
export V7_PORTFOLIO_HISTORY_SHA256="$(sha256sum "$V7_PORTFOLIO_HISTORY_PATH" | cut -d ' ' -f 1)"
python -B -m uvicorn app.backend.portfolio_history_local:app \
  --host 127.0.0.1 --port 8000 --no-proxy-headers
# In a second terminal, same worktree:
cd app/frontend
VITE_V7_PORTFOLIO_HISTORY=1 npm run dev -- --host 127.0.0.1
```

Open `http://localhost:5173/v7/portfolio-history` (use Vite's printed port if busy).
The same-origin `/api` proxy targets local port 8000. Do not set a remote API URL.
Do not expose the service through an external reverse proxy; do not deploy it.
For the optional existing backend entrypoint, launch from the repository root
with `PYTHONPATH="$PWD:$PWD/app/backend"` before `uvicorn main:app`; normal legacy
backend dependencies are additionally required. The isolated entrypoint above is
the tested H1 path. No new request accepts filesystem paths or write operations.

Fixture declarations and SHA pins provide explicit local identity, not proof of
market authenticity. The builder has no data-source parameter and only constructs
fixed synthetic events. API validates declared fixture flags, shape, event/final
consistency and pinned artifact bytes; exporter verifies accounting by replay.
An operator who deliberately rewrites declarations and hashes is outside this
cooperative local trust boundary. No general untrusted-input authenticity claim.

## Unit checkpoint

PORT-01: 7 focused / 67 combined PASS; one review repair restored legacy CR-only
and mixed-newline parsing while preserving original byte identity. Independent
review closed all Required findings. Commit `ddd75ff`.

PORT-02: four new and four existing API tests PASS. Independent review identified
launch-path documentation; the repository-root local launcher above resolves it.
Live server startup and actual browser/API evidence are tracked in the unit index.

PORT-03: 8 focused + 8 existing frontend cases, build/lint PASS. Actual Chrome
155.0.8059.85 desktop 1440 and mobile 390 verify all 7 event values, positions,
pending and provenance, keyboard navigation, no page overflow/runtime exception.
One CSS repair improved long exact decimal readability on mobile; no value changes.
Independent same-model and separate-model review: no Critical/Required.

PORT-04: COMPLETED_VERIFIED. Real process test runs fixture → every-prefix replay →
fresh exporter process → actual uvicorn HTTP → corrupt/missing/restore → stopped and
restarted API. One E2E test covers these sequential assertions. Actual Chrome then
verified missing and corrupt files on desktop/mobile, restored original bytes and
restarted API (PID326288 → PID327398), and reverified all seven events.

Full relevant regression: 67 portfolio/history/health/audit; 8 API; 8 focused and
8 existing frontend; 47 PIT; 81 operations; 3 fetch-only; 23 governance PASS.
Governance health has inherited WARNING categories, no structural FAIL.
Independent reviewer plus separate-model review: no remaining Critical/Required.
Repair rounds PORT-01/02/03/04 = 1/1/1/1 (PORT-04 is documentation-only:
stale H1 proposal pointers synchronized across domain notes). Extra Human interventions after H1 = 0.
One browser harness focus assumption was corrected separately (Recharts SVG is a
valid tab stop before SELECT); this is not an accounting/UI behavior change.

## Evidence and reproduction

[Machine evidence index](../research/v7/portfolio-history-evidence.json) preserves
raw test outputs, browser receipts, source hashes, fixture identity and runtime.
Fixed generated fixture inputs/output: `research/v7/portfolio-history-fixture.jsonl`,
`portfolio-history-fixture-manifest.json`, `portfolio-history-artifact.json`.
These are SYNTHETIC ENGINEERING fixtures, never a historical Frozen Matrix.
Original browser artifact SHA-256:
`d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6`.
It names committed exporter source `ddd75ff`; API `7ab3cb9`, UI `7592151` and
acceptance-source hashes are separately recorded. No circular/self-hash claim.

```bash
python -B -m unittest V6.experimental.v7_portfolio_history_test -v
python -B -m unittest app.backend.test_v7_portfolio app.backend.test_v7 -v
python -B -m unittest V6.experimental.v7_portfolio_history_e2e_test -v
node app/frontend/src/api/portfolioHistory.test.mjs
# Existing Chrome with a dedicated TEMP profile and CDP port 9228, local Vite/API:
node app/frontend/src/api/portfolioHistory.browser.mjs \
  http://127.0.0.1:5173 research/v7/portfolio-history-artifact.json /tmp/history-browser
# Missing/corrupt modes: replace artifact arg with '-' and append 'missing'/'corrupt'.
```

Browser receipts distinguish actual file/API cases from explicitly injected CDP
missing/corrupt/network transitions used to test same-page stale clearing.
Screenshots: [desktop](../research/v7/portfolio-history-desktop.png),
[mobile](../research/v7/portfolio-history-mobile.png). Exact decimal tables can scroll
horizontally inside their container. Browser runtime exceptions = 0; no whole-page
overflow. Recharts initial measurement emits a dimension warning; existing shell
market routes return 404 from the isolated fixture-only API. No claim of zero
console warnings or complete legacy dashboard acceptance. Build retains its
existing large-chunk warning. Only Chrome155 was actually tested.

## Closure / next authorization

Human Product Slice Review is ready, not yet accepted by Human. API/Vite/dedicated
Chrome were stopped; no server or scheduler intentionally remains running.
Main remains `83a6a2e`; entire branch including inherited CPU fixes is UNMERGED /
NOT_DEPLOYED. No push, merge, deployment, training, real-data admission or policy
change occurred. Strict Phase0 STOP; E1 exact Frozen Matrix BLOCKED; MAS Core Live
not certified. No weight/return/risk formula was invented.

Local commits: PORT-01 `ddd75ff`, PORT-02 `7ab3cb9`, PORT-03 `7592151`, PORT-04
is the containing acceptance commit (`git log -1`). Canonical memory/backlog and
Authority Map are updated in this closure. No second roadmap/orchestrator exists.
Next: Human product review; optionally exact-SHA H0 integration. CPU work such as
explicitly scoped fixture extensions or known-location E1 evidence recovery does
not inherently need GPU, but H1 is complete and grants no automatic extension.
H2 real inputs/admission/performance remains a separate Human research gate.
