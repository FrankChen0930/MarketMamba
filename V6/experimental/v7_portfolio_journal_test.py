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
            self.assertEqual(first.state["total_cost"], "0.00149775336994508237643534698")


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


if __name__ == "__main__":
    unittest.main()
