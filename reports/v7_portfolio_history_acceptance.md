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

PORT-03/04: acceptance in progress. Build is not browser acceptance.
