from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from V6.experimental.v7_portfolio_contract import PortfolioSpec
from V6.experimental.v7_portfolio_journal import (
    EventConflictError,
    JournalIntegrityError,
    PortfolioJournal,
)


T0 = "2026-09-16T08:00:00+08:00"
T1 = "2026-09-16T17:00:00+08:00"
T2 = "2026-09-17T09:00:00+08:00"


def portfolio_spec() -> PortfolioSpec:
    return PortfolioSpec(
        holdings_count=1,
        buffer_multiple="1.5",
        rebalance_every_sessions=5,
        head="5d",
    )


def signal_payload() -> dict:
    return {"head": "5d", "scores": {"A": "1"}}


def market_payload() -> dict:
    return {
        "quotes": {
            "A": {
                "ticker": "A",
                "price": "10",
                "status": "OPEN",
                "buy_fill_ratio": "1",
                "sell_fill_ratio": "1",
            }
        }
    }


class PortfolioJournalResumeTest(unittest.TestCase):
    def test_duplicate_retry_is_noop_and_replay_matches_live_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portfolio.jsonl"
            journal = PortfolioJournal.create(
                path, portfolio_spec(), "genesis-1", T0
            )
            journal.append("sig-1", "SIGNAL", T1, signal_payload())
            first = journal.append(
                "market-1", "MARKET_SESSION", T2, market_payload()
            )
            before = path.read_bytes()

            second = journal.append(
                "market-1", "MARKET_SESSION", T2, market_payload()
            )

            self.assertFalse(second.appended)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(first.state, second.state)
            self.assertEqual(journal.replay().state, first.state)
            self.assertEqual(len(journal.records()), 3)
            self.assertEqual(first.state["total_cost"], "0.0014977533699450823764353469795307039440838741887169")


class PortfolioJournalIntegrityTest(unittest.TestCase):
    def _journal(self, directory: str) -> PortfolioJournal:
        return PortfolioJournal.create(
            Path(directory) / "portfolio.jsonl",
            portfolio_spec(),
            "genesis-1",
            T0,
        )

    def test_same_id_with_different_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.append("sig-1", "SIGNAL", T1, signal_payload())

            with self.assertRaises(EventConflictError):
                journal.append(
                    "sig-1",
                    "SIGNAL",
                    T1,
                    {"head": "5d", "scores": {"A": "2"}},
                )

    def test_tamper_and_truncated_tail_fail_closed(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.append("sig-1", "SIGNAL", T1, signal_payload())
            lines = journal.path.read_text(encoding="utf-8").splitlines()
            changed = json.loads(lines[1])
            changed["payload"]["scores"]["A"] = "2"
            lines[1] = json.dumps(
                changed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            tampered = Path(tmp) / "tampered.jsonl"
            tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(JournalIntegrityError):
                PortfolioJournal(tampered).records()

            truncated = Path(tmp) / "truncated.jsonl"
            truncated.write_bytes(journal.path.read_bytes() + b'{"seq":')
            with self.assertRaises(JournalIntegrityError):
                PortfolioJournal(truncated).records()

    def test_time_regression_is_rejected_without_changing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.append("sig-1", "SIGNAL", T1, signal_payload())
            before = journal.path.read_bytes()

            with self.assertRaises(JournalIntegrityError):
                journal.append(
                    "market-old",
                    "MARKET_SESSION",
                    "2026-09-15T09:00:00+08:00",
                    market_payload(),
                )

            self.assertEqual(journal.path.read_bytes(), before)


class PortfolioJournalCliTest(unittest.TestCase):
    def test_cli_prints_human_readable_replay_without_writing(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portfolio.jsonl"
            journal = PortfolioJournal.create(
                path, portfolio_spec(), "genesis-1", T0
            )
            journal.append("sig-1", "SIGNAL", T1, signal_payload())
            journal.append(
                "market-1", "MARKET_SESSION", T2, market_payload()
            )
            before = path.read_bytes()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "V6.experimental.v7_portfolio_journal",
                    "--journal",
                    str(path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("事件數：3", completed.stdout)
            self.assertIn("最後 session：market-1", completed.stdout)
            self.assertIn("淨值：", completed.stdout)
            self.assertIn("現金：", completed.stdout)
            self.assertIn("持股數：1", completed.stdout)
            self.assertIn("累計成本：", completed.stdout)
            self.assertIn("待成交：否", completed.stdout)
            self.assertEqual(path.read_bytes(), before)

    def test_cli_fails_closed_on_corrupt_journal_without_writing(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"seq":', encoding="utf-8")
            before = path.read_bytes()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "V6.experimental.v7_portfolio_journal",
                    "--journal",
                    str(path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("日誌驗證失敗：", completed.stderr)
            self.assertEqual(path.read_bytes(), before)


class PortfolioJournalBoundaryTest(unittest.TestCase):
    def test_head_mismatch_refusal_preserves_bytes_and_allows_valid_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(
                Path(tmp) / "portfolio.jsonl", portfolio_spec(), "genesis", T0
            )
            before = journal.path.read_bytes()
            with self.assertRaises(JournalIntegrityError):
                journal.append(
                    "signal", "SIGNAL", T1, {"head": "10d", "scores": {"A": "1"}}
                )
            self.assertEqual(journal.path.read_bytes(), before)
            self.assertEqual(journal.replay().event_count, 1)
            accepted = journal.append("signal", "SIGNAL", T1, signal_payload())
            self.assertTrue(accepted.appended)
            self.assertEqual(accepted.event_count, 2)
            self.assertEqual(journal.replay().state, accepted.state)

    def test_existing_semantically_invalid_chain_cannot_be_extended(self):
        import json
        from V6.experimental.v7_portfolio_journal import _make_record

        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(
                Path(tmp) / "portfolio.jsonl", portfolio_spec(), "genesis", T0
            )
            invalid = _make_record(
                seq=2, event_id="bad-head", kind="SIGNAL", occurred_at=T1,
                payload={"head": "10d", "scores": {"A": "1"}},
                prev_hash=journal.records()[0]["record_hash"],
            )
            with journal.path.open("a") as handle:
                handle.write(json.dumps(invalid) + "\n")
            before = journal.path.read_bytes()
            with self.assertRaises(JournalIntegrityError):
                journal.append("market", "MARKET_SESSION", T2, market_payload())
            self.assertEqual(journal.path.read_bytes(), before)

    def test_valid_early_signal_after_fill_is_replayable_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(
                Path(tmp) / "portfolio.jsonl", portfolio_spec(), "genesis", T0
            )
            journal.append("signal", "SIGNAL", T1, signal_payload())
            filled = journal.append("market", "MARKET_SESSION", T2, market_payload())
            early = journal.append(
                "early", "SIGNAL", "2026-09-17T17:00:00+08:00",
                {"head": "5d", "scores": {"B": "2"}},
            )
            self.assertEqual(early.state, filled.state)
            self.assertEqual(early.event_count, 4)
            self.assertEqual(journal.replay().state, filled.state)

    def test_invalid_signal_is_rejected_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(
                Path(tmp) / "portfolio.jsonl",
                portfolio_spec(),
                "genesis-1",
                T0,
            )
            before = journal.path.read_bytes()

            with self.assertRaises(JournalIntegrityError):
                journal.append(
                    "sig-invalid",
                    "SIGNAL",
                    T1,
                    {"head": "", "scores": {"A": "1"}},
                )

            self.assertEqual(journal.path.read_bytes(), before)

    def test_non_string_ticker_is_rejected_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(
                Path(tmp) / "portfolio.jsonl",
                portfolio_spec(),
                "genesis-1",
                T0,
            )
            before = journal.path.read_bytes()

            with self.assertRaises(JournalIntegrityError):
                journal.append(
                    "sig-invalid",
                    "SIGNAL",
                    T1,
                    {"head": "5d", "scores": {1234: "1"}},
                )

            self.assertEqual(journal.path.read_bytes(), before)

    def test_non_utf8_journal_fails_with_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.jsonl"
            path.write_bytes(b"\xff\xfe\n")

            with self.assertRaises(JournalIntegrityError):
                PortfolioJournal(path).records()


class PortfolioJournalQuoteCollisionTest(unittest.TestCase):
    def test_collision_refuses_without_write_and_valid_retry_recovers(self):
        from V6.experimental.v7_portfolio_contract import MarketQuote

        for alias in (" A ", "\tA\t", "\u3000A\u3000"):
            for reverse in (False, True):
                for same_price in (False, True):
                    for as_object in (False, True):
                        with self.subTest(alias=alias, reverse=reverse,
                                          same_price=same_price, as_object=as_object):
                            with tempfile.TemporaryDirectory() as tmp:
                                journal = PortfolioJournal.create(
                                    Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0
                                )
                                journal.append("signal", "SIGNAL", T1, signal_payload())
                                before = journal.path.read_bytes()
                                state = journal.replay()
                                first = market_payload()["quotes"]["A"]
                                second = dict(first, price="10" if same_price else "20")
                                if as_object:
                                    first = MarketQuote.from_payload(first)
                                    second = MarketQuote.from_payload(second)
                                pairs = [("A", first), (alias, second)]
                                if reverse:
                                    pairs.reverse()
                                with self.assertRaisesRegex(JournalIntegrityError, "duplicate quote ticker"):
                                    journal.append("market", "MARKET_SESSION", T2,
                                                   {"quotes": dict(pairs)})
                                self.assertEqual(journal.path.read_bytes(), before)
                                self.assertEqual(journal.replay(), state)
                                reopened = PortfolioJournal(journal.path)
                                accepted = reopened.append("market", "MARKET_SESSION", T2, market_payload())
                                self.assertTrue(accepted.appended)
                                self.assertEqual(accepted.event_count, 3)
                                durable = journal.path.read_bytes()
                                retry = PortfolioJournal(journal.path).append(
                                    "market", "MARKET_SESSION", T2, market_payload())
                                self.assertFalse(retry.appended)
                                self.assertEqual(retry.state, accepted.state)
                                self.assertEqual(journal.path.read_bytes(), durable)
                                self.assertEqual(journal.replay().state, accepted.state)

    def test_collision_cannot_be_treated_as_duplicate_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0)
            journal.append("market", "MARKET_SESSION", T2, market_payload())
            before = journal.path.read_bytes()
            quote = market_payload()["quotes"]["A"]
            with self.assertRaisesRegex(JournalIntegrityError, "duplicate quote ticker"):
                journal.append("market", "MARKET_SESSION", T2,
                               {"quotes": {" A ": dict(quote, price="20"), "A": quote}})
            self.assertEqual(journal.path.read_bytes(), before)

    def test_unique_whitespace_key_and_empty_quotes_remain_legal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0)
            quote = market_payload()["quotes"]["A"]
            first = journal.append("market", "MARKET_SESSION", T2, {"quotes": {" A ": quote}})
            self.assertEqual(set(journal.records()[-1]["payload"]["quotes"]), {"A"})
            retry = journal.append("market", "MARKET_SESSION", T2, market_payload())
            self.assertFalse(retry.appended)
            self.assertEqual(first.state, retry.state)
            empty = journal.append("empty", "MARKET_SESSION", "2026-09-18T09:00:00+08:00", {"quotes": {}})
            self.assertEqual(journal.replay().state, empty.state)

    def test_hash_valid_persisted_collision_is_refused_without_repair(self):
        import json
        from V6.experimental.v7_portfolio_journal import _make_record

        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0)
            quote = market_payload()["quotes"]["A"]
            bad = _make_record(seq=2, event_id="bad", kind="MARKET_SESSION", occurred_at=T2,
                               payload={"quotes": {"A": quote, " A ": dict(quote, price="20")}},
                               prev_hash=journal.records()[0]["record_hash"])
            with journal.path.open("a") as handle:
                handle.write(json.dumps(bad) + "\n")
            before = journal.path.read_bytes()
            for operation in (journal.records, journal.replay,
                              lambda: journal.append("later", "MARKET_SESSION", T2, market_payload())):
                with self.assertRaises(JournalIntegrityError):
                    operation()
                self.assertEqual(journal.path.read_bytes(), before)


class PortfolioJournalJsonMemberTest(unittest.TestCase):
    def test_duplicate_members_at_each_depth_refuse_read_replay_and_append(self):
        # The final value and existing record hash stay unchanged. A permissive
        # decoder would discard the first member before integrity validation.
        replacements = (
            ('"seq":1', '"seq":999,"seq":1'),
            ('"event_id":"g"', '"event_id":"other","event_id":"g"'),
            ('"kind":"GENESIS"', '"kind":"SIGNAL","kind":"GENESIS"'),
            ('"payload":{', '"payload":{},"payload":{'),
            ('"holdings_count":1', '"holdings_count":2,"holdings_count":1'),
            ('"scores":{"A":"1"}', '"scores":{"A":"9","A":"1"}'),
            ('"quotes":{', '"quotes":{},"quotes":{'),
            ('"quotes":{"A":{', '"quotes":{"A":{},"A":{'),
            ('"price":"10"', '"price":"999","price":"10"'),
            ('"price":"10"', '"price":"10","price":"10"'),
            ('"price":"10"', '"price":"999","\\u0070rice":"10"'),
        )
        for needle, replacement in replacements:
            with self.subTest(replacement=replacement):
                with tempfile.TemporaryDirectory() as tmp:
                    journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0)
                    journal.append("signal", "SIGNAL", T1, signal_payload())
                    journal.append("market", "MARKET_SESSION", T2, market_payload())
                    original = journal.path.read_text()
                    self.assertIn(needle, original)
                    altered = original.replace(needle, replacement, 1)
                    self.assertNotEqual(original, altered)
                    journal.path.write_text(altered)
                    before = journal.path.read_bytes()
                    for name, operation in (
                        ("records", journal.records),
                        ("replay", journal.replay),
                        ("retry", lambda: journal.append("market", "MARKET_SESSION", T2, market_payload())),
                        ("append", lambda: journal.append("later", "MARKET_SESSION", T2, market_payload())),
                    ):
                        with self.subTest(operation=name):
                            with self.assertRaisesRegex(JournalIntegrityError, "duplicate JSON key"):
                                operation()
                            self.assertEqual(journal.path.read_bytes(), before)

    def test_duplicate_member_cli_refuses_without_repair(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "g", T0)
            journal.append("market", "MARKET_SESSION", T2, market_payload())
            journal.path.write_text(journal.path.read_text().replace(
                '"price":"10"', '"price":"999","price":"10"'))
            before = journal.path.read_bytes()
            result = subprocess.run(
                [sys.executable, "-B", "-m", "V6.experimental.v7_portfolio_journal", "--journal", str(journal.path)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("duplicate JSON key", result.stderr)
            self.assertEqual(journal.path.read_bytes(), before)

    def test_legal_key_order_whitespace_and_unicode_encoding_remain_readable(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            journal = PortfolioJournal.create(Path(tmp) / "journal.jsonl", portfolio_spec(), "創始", T0)
            journal.append("signal", "SIGNAL", T1, signal_payload())
            filled = journal.append("market", "MARKET_SESSION", T2, market_payload())
            lines = [json.dumps(dict(reversed(list(record.items()))), ensure_ascii=True)
                     for record in journal.records()]
            journal.path.write_text("\n".join(lines) + "\n")
            before = journal.path.read_bytes()
            self.assertEqual(journal.replay().state, filled.state)
            retry = journal.append("market", "MARKET_SESSION", T2, market_payload())
            self.assertFalse(retry.appended)
            self.assertEqual(retry.event_count, 3)
            self.assertEqual(journal.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
