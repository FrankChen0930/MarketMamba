# V7 Replayable Portfolio Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic normalized-value portfolio engine that executes after-close signals only at later explicitly tradeable sessions and resumes safely from a hash-chained journal.

**Architecture:** A strict Decimal-based contract feeds a pure state machine. An append-only, locked, hash-chained JSONL journal stores immutable input events; replay reconstructs the same state, while duplicate event IDs make crash retries idempotent.

**Tech Stack:** Python 3.10+ standard library, `unittest`, `decimal`, `dataclasses`, `fcntl`, JSONL, SHA-256.

**Spec:** `docs/superpowers/specs/2026-09-16-v7-replayable-portfolio-engine-design.md`

## Global Constraints

- Work only in `V6/experimental/` and `docs/superpowers/` on `feature/v7-portfolio-engine`.
- Do not modify V6.1, existing backtest engines, models, weights, Data, parquet, A/B/C results, app routes, schedules, or main worktree changes.
- Initial cash and net value are exactly Decimal `1`; no personal capital or broker integration.
- Buy cost is Decimal `0.0015`; sell cost is Decimal `0.0045`; these are frozen research comparison assumptions.
- Unknown or missing tradeability fails closed. Never infer tradability from adjusted returns or price movement.
- Signals never execute at the same timestamp. The first eligible session has `session.occurred_at > signal.as_of`.
- JSON persists Decimal values as strings and requires timezone-aware ISO-8601 timestamps.
- No third-party dependencies.

---

### Task 1: Versioned Contract and Validation

**Files:**
- Create: `V6/experimental/v7_portfolio_contract.py`
- Create: `V6/experimental/v7_portfolio_contract_test.py`

**Interfaces:**
- Produces: `PortfolioSpec`, `TradeStatus`, `MarketQuote`, `CorporateAction`, `ContractError`, `parse_decimal()`, `parse_timestamp()`.
- Consumes: Python standard library only.

- [x] **Step 1: Write failing contract tests**

Create table-driven tests with literal expectations:

```python
class PortfolioContractTest(unittest.TestCase):
    def test_default_spec_round_trips_with_decimal_strings(self):
        spec = PortfolioSpec(holdings_count=2, buffer_multiple="1.5",
                             rebalance_every_sessions=5, head="5d")
        self.assertEqual(spec.to_payload()["buy_cost_rate"], "0.0015")
        self.assertEqual(PortfolioSpec.from_payload(spec.to_payload()), spec)

    def test_unknown_is_the_explicit_safe_default(self):
        quote = MarketQuote.from_payload({
            "ticker": "2330", "price": "100",
            "buy_fill_ratio": "0", "sell_fill_ratio": "0",
        })
        self.assertEqual(quote.status, TradeStatus.UNKNOWN)

    def test_invalid_numbers_and_naive_timestamps_are_rejected(self):
        with self.assertRaises(ContractError):
            parse_decimal("NaN", field="price", minimum=Decimal("0"))
        with self.assertRaises(ContractError):
            parse_timestamp("2026-09-16T17:00:00")
```

- [x] **Step 2: Run RED**

Run: `python3 -m unittest V6.experimental.v7_portfolio_contract_test -v`

Expected: import failure because `v7_portfolio_contract` does not exist.

- [x] **Step 3: Implement the minimal validated contract**

Use frozen dataclasses and these exact enum values:

```python
ENGINE_VERSION = "v7-portfolio-v1"

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    BUY_BLOCKED = "BUY_BLOCKED"
    SELL_BLOCKED = "SELL_BLOCKED"
    HALTED = "HALTED"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class PortfolioSpec:
    holdings_count: int
    buffer_multiple: Decimal
    rebalance_every_sessions: int
    head: str
    buy_cost_rate: Decimal = Decimal("0.0015")
    sell_cost_rate: Decimal = Decimal("0.0045")
    engine_version: str = ENGINE_VERSION
```

`MarketQuote.from_payload()` requires ticker, positive price, and both fill ratios; status alone defaults to UNKNOWN. Validate ratios in `[0,1]`, rates in `[0,1)`, non-empty IDs/head/ticker, positive counts, and finite decimals. `CorporateAction` requires a positive quantity multiplier, non-negative cash per old share, positive post-action price, and aware timestamp.

- [x] **Step 4: Run GREEN and contract mutation checks**

Run: `python3 -m unittest V6.experimental.v7_portfolio_contract_test -v`

Expected: all contract tests pass. Confirm changing UNKNOWN to OPEN, accepting NaN, or accepting a naive timestamp would fail at least one test.

- [x] **Step 5: Commit only Task 1 files**

```bash
git add V6/experimental/v7_portfolio_contract.py V6/experimental/v7_portfolio_contract_test.py
git diff --cached --check
git commit -m "feat: define strict V7 portfolio event contract"
```

### Task 2: Signal Scheduling, Buffering, and Deterministic Targets

**Files:**
- Create: `V6/experimental/v7_portfolio_engine.py`
- Create: `V6/experimental/v7_portfolio_engine_test.py`

**Interfaces:**
- Consumes: Task 1 contract types.
- Produces: `PortfolioEngine(spec)`, `apply_signal(event_id, as_of, head, scores)`, `snapshot()`, and result dataclasses `SignalResult`, `SessionResult`, `TradeFill`.
- State fields used by later tasks: `cash`, `positions`, `pending`, `session_count`, `last_rebalance_session`, `last_session_id`, `total_cost`.

- [x] **Step 1: Write failing planner tests**

Test these independent mutations with literal targets:

```python
def test_signal_creates_pending_target_but_does_not_trade(self):
    engine = PortfolioEngine(spec(holdings_count=2))
    result = engine.apply_signal("sig-1", aware("2026-09-16T17:00:00+08:00"),
                                 "5d", {"B": "1", "A": "1", "C": "0"})
    self.assertEqual(result.target_tickers, ("A", "B"))
    self.assertEqual(engine.snapshot()["cash"], "1")
    self.assertEqual(engine.snapshot()["positions"], {})

def test_buffer_keeps_existing_name_and_frequency_skips_early_signal(self):
    # Seed a held C whose new rank is 3 and k*N is 3.
    # First due signal keeps C and adds highest unheld A.
    # After one session, a second signal is not due when frequency is 5.
    self.assertEqual(due.target_tickers, ("C", "A"))
    self.assertFalse(early.due)

def test_head_mismatch_is_rejected(self):
    with self.assertRaises(ContractError):
        engine.apply_signal("sig-x", aware("2026-09-16T17:00:00+08:00"), "10d", {"A": "1"})
```

- [x] **Step 2: Run RED**

Run: `python3 -m unittest V6.experimental.v7_portfolio_engine_test -v`

Expected: import failure because `v7_portfolio_engine` does not exist.

- [x] **Step 3: Implement minimal planner state**

Sort scores by `(-score, ticker)`. Rank starts at 1. Keep currently held tickers with rank `<= floor(buffer_multiple * holdings_count)`, preserving rank order, then fill from the sorted Top-N candidates. Candidate shortage yields fewer targets.

A first signal is due. Later signals are due only when no successful rebalance exists or `session_count - last_rebalance_session >= rebalance_every_sessions`. A due signal replaces an older pending target and records its signal ID; a non-due signal cannot mutate pending state.

- [x] **Step 4: Run GREEN**

Run: `python3 -m unittest V6.experimental.v7_portfolio_contract_test V6.experimental.v7_portfolio_engine_test -v`

Expected: contract and planner tests pass.

- [x] **Step 5: Commit only Task 2 files**

```bash
git add V6/experimental/v7_portfolio_engine.py V6/experimental/v7_portfolio_engine_test.py
git diff --cached --check
git commit -m "feat: plan deterministic V7 portfolio targets"
```

### Task 3: Execution, Costs, Partial Fills, and Corporate Actions

**Files:**
- Modify: `V6/experimental/v7_portfolio_engine.py`
- Modify: `V6/experimental/v7_portfolio_engine_test.py`

**Interfaces:**
- Produces: `apply_market_session(event_id, occurred_at, quotes)`, `apply_corporate_action(event_id, action)`, `net_value()`.
- `SessionResult.fills` contains side, ticker, requested quantity, filled quantity, gross notional, fee, and status.
- `PortfolioEngine.apply_event(kind, event_id, occurred_at, payload)` becomes the journal replay boundary.

- [x] **Step 1: Add RED tests for time and conservative blocking**

```python
def test_same_timestamp_does_not_execute_and_unknown_stays_pending(self):
    engine.apply_signal("sig-1", aware(T0), "5d", {"A": "1"})
    same = engine.apply_market_session("m0", aware(T0), {"A": open_quote("A", "10")})
    self.assertEqual(same.fills, ())
    blocked = engine.apply_market_session("m1", aware(T1), {
        "A": quote("A", "10", status="UNKNOWN", buy_ratio="0", sell_ratio="0")
    })
    self.assertEqual(blocked.fills[0].status, "BLOCKED")
    self.assertEqual(engine.cash, Decimal("1"))
    self.assertIsNotNone(engine.pending)
```

Run: `python3 -m unittest V6.experimental.v7_portfolio_engine_test.PortfolioExecutionTest.test_same_timestamp_does_not_execute_and_unknown_stays_pending -v`

Expected: fail because `apply_market_session` is missing.

- [x] **Step 2: Implement quote marking and side permissions**

Increment `session_count` for each unique session call. Update held last prices from positive quotes regardless of trade status. Permit buys for OPEN/SELL_BLOCKED and sells for OPEN/BUY_BLOCKED; HALTED/UNKNOWN block both. Missing quote creates a BLOCKED fill with reason `MISSING_QUOTE`.

- [x] **Step 3: Add RED hand-calculation tests for costs and partial fills**

Use one-stock literals:

```python
def test_full_buy_charges_buy_cost_once_and_conserves_value(self):
    # target gross = 1 / 1.0015; filled quantity = gross / price
    expected_gross = Decimal("1") / Decimal("1.0015")
    self.assertEqual(fill.fee, expected_gross * Decimal("0.0015"))
    self.assertEqual(engine.cash + engine.position_value(), engine.net_value())
    self.assertEqual(engine.total_cost, fill.fee)

def test_fill_ratio_and_cash_limit_leave_pending(self):
    # buy_fill_ratio=.5 fills exactly half the requested quantity.
    self.assertEqual(fill.filled_quantity, fill.requested_quantity / 2)
    self.assertEqual(fill.status, "PARTIAL")
    self.assertIsNotNone(engine.pending)
```

Run the two named tests; expected failure is missing execution logic, not fixture errors.

- [x] **Step 4: Implement sells-first then buys**

At an eligible session:

1. Compute marked pre-trade NAV.
2. Use a conservative provisional per-name gross target `NAV / (N * (1 + buy_cost_rate))`.
3. Sell non-target and overweight positions first, bounded by sell fill ratio.
4. Recompute NAV after sell fees and set final per-name target to `post_sell_NAV / (N * (1 + buy_cost_rate))`.
5. Buy underweight target names in deterministic target order, bounded by buy fill ratio and `cash / (price * (1 + buy_cost_rate))`.
6. Fees equal filled gross times the side rate and update `total_cost` once.
7. Keep pending while any target delta, blocked old position, or partial fill remains; otherwise set `last_rebalance_session`.

Never let cash fall below zero; use Decimal arithmetic without quantizing.

- [x] **Step 5: Add RED tests for sell-side block and explicit corporate actions**

```python
def test_sell_blocked_position_prevents_rebalance_completion(self):
    # A is no longer targeted, but SELL_BLOCKED leaves it held.
    self.assertIn("A", engine.positions)
    self.assertIsNotNone(engine.pending)

def test_split_preserves_position_value_and_dividend_adds_cash_once(self):
    before = engine.net_value()
    engine.apply_corporate_action("ca-1", CorporateAction(
        ticker="A", occurred_at=aware(T2), quantity_multiplier="2",
        cash_per_old_share="0", post_action_price="5"))
    self.assertEqual(engine.net_value(), before)
    old_cash = engine.cash
    old_qty = engine.positions["A"].quantity
    engine.apply_corporate_action("ca-2", CorporateAction(
        ticker="A", occurred_at=aware(T3), quantity_multiplier="1",
        cash_per_old_share="0.1", post_action_price="4.9"))
    self.assertEqual(engine.cash, old_cash + old_qty * Decimal("0.1"))
```

Run named tests; expected failure is missing sell blocking or corporate action behavior.

- [x] **Step 6: Implement company action and event dispatch**

Apply cash using pre-action quantity, then multiply quantity and replace last price. No position is a visible NO_POSITION outcome. `apply_event` accepts only `SIGNAL`, `MARKET_SESSION`, and `CORPORATE_ACTION`, parses contract payloads, and returns the corresponding result.

- [x] **Step 7: Run GREEN and mutation review**

Run: `python3 -m unittest V6.experimental.v7_portfolio_contract_test V6.experimental.v7_portfolio_engine_test -v`

Expected: all pass. Mentally verify tests fail if costs are applied twice, same-time execution is allowed, UNKNOWN becomes tradable, sell runs after buy, fill ratio is ignored, or dividend uses post-split quantity incorrectly.

- [x] **Step 8: Commit Task 3 modifications**

```bash
git add V6/experimental/v7_portfolio_engine.py V6/experimental/v7_portfolio_engine_test.py
git diff --cached --check
git commit -m "feat: execute V7 portfolio fills conservatively"
```

### Task 4: Hash-Chained Journal, Idempotent Resume, and Read-Only CLI

**Files:**
- Create: `V6/experimental/v7_portfolio_journal.py`
- Create: `V6/experimental/v7_portfolio_journal_test.py`

**Interfaces:**
- Consumes: `PortfolioSpec.from_payload()`, `PortfolioEngine.apply_event()`.
- Produces: `PortfolioJournal.create(path, spec, event_id, occurred_at)`, `append(event_id, kind, occurred_at, payload)`, `records()`, `replay()`, `JournalIntegrityError`, `EventConflictError`.
- CLI: `python3 -m V6.experimental.v7_portfolio_journal --journal PATH`.

- [x] **Step 1: Write RED tests for create, replay, and duplicate retry**

```python
def test_duplicate_retry_is_noop_and_replay_matches_live_state(self):
    journal = PortfolioJournal.create(path, spec, "genesis-1", aware(T0))
    journal.append("sig-1", "SIGNAL", aware(T1), signal_payload)
    first = journal.append("market-1", "MARKET_SESSION", aware(T2), market_payload)
    before = path.read_bytes()
    second = journal.append("market-1", "MARKET_SESSION", aware(T2), market_payload)
    self.assertFalse(second.appended)
    self.assertEqual(path.read_bytes(), before)
    self.assertEqual(first.state, second.state)
    self.assertEqual(journal.replay().state, first.state)
```

Run: `python3 -m unittest V6.experimental.v7_portfolio_journal_test -v`

Expected: import failure because the journal module does not exist.

- [x] **Step 2: Implement canonical records and locked append**

Canonical JSON uses `ensure_ascii=False`, `sort_keys=True`, and separators `(",", ":")`. Record hash is SHA-256 of the canonical record without `record_hash`. Genesis has `seq=1`, `prev_hash` equal to 64 zeroes, kind `GENESIS`, and payload containing the entire frozen spec.

Open the journal with `a+`, acquire `fcntl.LOCK_EX`, read and validate all records, check event ID, append exactly one newline-terminated record, flush, and fsync before unlock.

- [x] **Step 3: Add RED conflict and corruption tests**

```python
def test_same_id_with_different_payload_is_rejected(self):
    journal.append("sig-1", "SIGNAL", aware(T1), signal_payload)
    with self.assertRaises(EventConflictError):
        journal.append("sig-1", "SIGNAL", aware(T1), other_signal_payload)

def test_tamper_and_truncated_tail_fail_closed(self):
    # Rewrite one payload byte without updating hash, then assert records() raises.
    with self.assertRaises(JournalIntegrityError):
        PortfolioJournal(tampered).records()
    truncated.write_bytes(valid_bytes + b'{"seq":')
    with self.assertRaises(JournalIntegrityError):
        PortfolioJournal(truncated).records()
```

Run named tests; expected failures are absent conflict and integrity checks.

- [x] **Step 4: Implement validation and replay**

Validate non-empty newline-terminated JSONL, contiguous seq, zero genesis predecessor, exact hash links, record hashes, unique event IDs, non-decreasing aware timestamps, genesis-first, and matching engine version. Replay instantiates from genesis spec and dispatches later records through `apply_event`.

- [x] **Step 5: Add RED CLI summary test**

Run the module in a subprocess on a fixture journal and assert literal labels and values:

```python
self.assertIn("事件數：3", output)
self.assertIn("淨值：", output)
self.assertIn("現金：", output)
self.assertIn("持股數：1", output)
self.assertIn("累計成本：", output)
self.assertIn("待成交：否", output)
```

Expected: fail because CLI output is missing.

- [x] **Step 6: Implement read-only CLI**

`main()` accepts only `--journal`, calls `records()` and `replay()`, prints exact numeric state, and exits non-zero with `日誌驗證失敗：<reason>` on integrity errors. It never writes during inspection.

- [x] **Step 7: Run the complete P3.1 suite**

Run:

```bash
python3 -m unittest   V6.experimental.v7_portfolio_contract_test   V6.experimental.v7_portfolio_engine_test   V6.experimental.v7_portfolio_journal_test -v
python3 -m compileall -q   V6/experimental/v7_portfolio_contract.py   V6/experimental/v7_portfolio_engine.py   V6/experimental/v7_portfolio_journal.py
git diff --check
```

Expected: all tests pass, compileall exits 0, and diff check prints nothing.

- [x] **Step 8: Commit Task 4 files**

```bash
git add V6/experimental/v7_portfolio_journal.py V6/experimental/v7_portfolio_journal_test.py
git diff --cached --check
git commit -m "feat: add idempotent V7 portfolio journal replay"
```

### Task 5: Final Acceptance and Scope Audit

**Files:**
- Modify: `docs/superpowers/plans/2026-09-16-v7-replayable-portfolio-engine.md` only to check completed steps.
- Do not update `AGENTS.md` or main `tasks/todo.md` until the user reviews and confirms completion.

**Interfaces:**
- Consumes all task outputs.
- Produces fresh verification evidence and a reviewable feature branch.

- [x] **Step 1: Run fresh complete verification**

Run the full Task 4 Step 7 command again after all plan checkbox edits. Record exact pass count, duration, branch status, and commit list.

- [x] **Step 2: Audit protected paths and dependencies**

Run:

```bash
git diff --name-only 6cde79c..HEAD
git status --short
git log --oneline 6cde79c..HEAD
```

Expected changed paths are only the spec, plan, three production modules, and three test modules. Confirm no dependency file, model, weight, Data, parquet, result, app, schedule, V6.1 engine, or main-worktree path changed.

- [x] **Step 3: Run an end-to-end temporary journal demonstration**

Create a temporary journal through the public API, append genesis, one signal, and one later OPEN market session, then invoke the CLI. Verify it reports 3 events, one holding, no pending target, non-negative cash, net value, and a positive one-time cost. Delete only the temporary directory created for this demonstration.

- [x] **Step 4: Preserve branch for human review**

Do not merge, push, deploy, schedule, or remove the worktree. Report the worktree path, branch, commits, tests, files changed, timing, assumptions, and known limitations.
