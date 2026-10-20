"""Real-process synthetic journal acceptance; no market data or training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import tempfile
import unittest
from pathlib import Path

from V6.experimental.v7_portfolio_journal import (
    JournalIntegrityError, PortfolioJournal, _canonical, _make_record,
)
from V6.experimental.v7_portfolio_journal_test import (
    T0, T1, T2, market_payload, portfolio_spec, signal_payload,
)


def _append_worker(path, connection, barrier, payload, pause_after_fsync=False):
    """Pause only after a real durable append, before the caller gets a result."""
    if pause_after_fsync:
        real_fsync = os.fsync

        def durable_then_pause(fd):
            real_fsync(fd)
            connection.send({"event": "durable_before_response"})
            connection.recv()

        os.fsync = durable_then_pause
    try:
        if barrier is not None:
            barrier.wait(timeout=10)
        result = PortfolioJournal(path).append("market", "MARKET_SESSION", T2, payload)
        connection.send({
            "event": "result", "appended": result.appended,
            "count": result.event_count, "state": result.state,
        })
    except Exception as exc:
        connection.send({"event": "error", "type": type(exc).__name__, "message": str(exc)})
    finally:
        connection.close()


def _reader_worker(path, connection):
    real_flock = fcntl.flock

    def observe_real_contention(fd, operation):
        try:
            real_flock(fd, operation | fcntl.LOCK_NB)
        except BlockingIOError:
            connection.send({"event": "lock_blocked", "operation": operation})
            real_flock(fd, operation)

    fcntl.flock = observe_real_contention
    try:
        result = PortfolioJournal(path).replay()
        connection.send({"event": "result", "count": result.event_count, "state": result.state})
    except Exception as exc:
        connection.send({"event": "error", "type": type(exc).__name__, "message": str(exc)})
    finally:
        connection.close()


def _partial_writer(path, connection):
    with open(path, "ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(b'{"seq":')
        handle.flush()
        os.fsync(handle.fileno())
        connection.send({"event": "partial_tail_durable"})
        connection.recv()


class PortfolioProcessTest(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("spawn")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "portfolio.jsonl"
        self.journal = PortfolioJournal.create(self.path, portfolio_spec(), "genesis", T0)
        self.journal.append("signal", "SIGNAL", T1, signal_payload())

    def start(self, target, *args):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=target, args=(str(self.path), child, *args))
        process.start()
        child.close()

        def cleanup():
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
            parent.close()

        self.addCleanup(cleanup)
        return process, parent

    def receive(self, connection):
        self.assertTrue(connection.poll(10), "worker failed to reach synchronized boundary")
        return connection.recv()

    def finished(self, process, expected=0):
        process.join(timeout=10)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, expected)

    def evidence(self, scenario, **facts):
        print(json.dumps({
            "evidence_class": "SYNTHETIC_REAL_PROCESS_FAULT_INJECTION",
            "scenario": scenario,
            "journal_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            **facts,
        }, sort_keys=True), flush=True)

    def test_concurrent_duplicate_writers_charge_once(self):
        barrier = self.context.Barrier(3)
        workers = [
            self.start(_append_worker, barrier, market_payload()) for _ in range(2)
        ]
        barrier.wait(timeout=10)
        results = [self.receive(connection) for _, connection in workers]
        for process, _ in workers:
            self.finished(process)
        self.assertTrue(all(result["event"] == "result" for result in results), results)
        self.assertEqual(sorted(result["appended"] for result in results), [False, True])
        self.assertEqual(results[0]["state"], results[1]["state"])
        self.assertEqual([result["count"] for result in results], [3, 3])
        self.assertEqual(
            results[0]["state"]["total_cost"],
            "0.0014977533699450823764353469795307039440838741887169",
        )
        self.assertEqual(self.journal.replay().state, results[0]["state"])
        self.evidence("concurrent_duplicate", results=results, exits=[0, 0])

    def test_concurrent_conflicting_writers_admit_only_one_payload(self):
        barrier = self.context.Barrier(3)
        different = market_payload()
        different["quotes"]["A"]["buy_fill_ratio"] = "0.5"
        workers = [
            self.start(_append_worker, barrier, payload)
            for payload in (market_payload(), different)
        ]
        barrier.wait(timeout=10)
        results = [self.receive(connection) for _, connection in workers]
        for process, _ in workers:
            self.finished(process)
        accepted = [item for item in results if item["event"] == "result"]
        refused = [item for item in results if item["event"] == "error"]
        self.assertEqual(len(accepted), 1, results)
        self.assertEqual(len(refused), 1, results)
        self.assertEqual(refused[0]["type"], "EventConflictError")
        self.assertEqual(self.journal.replay().event_count, 3)
        self.assertEqual(self.journal.replay().state, accepted[0]["state"])
        self.evidence("concurrent_conflict", results=results, exits=[0, 0])

    def test_reader_waits_for_writer_commit_instead_of_observing_partial_line(self):
        # Controlled split-write injection, with real filesystem and advisory locks.
        record = _make_record(
            seq=3, event_id="market", kind="MARKET_SESSION", occurred_at=T2,
            payload=market_payload(), prev_hash=self.journal.records()[-1]["record_hash"],
        )
        encoded = (_canonical(record) + "\n").encode()
        with self.path.open("ab") as writer:
            fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
            writer.write(encoded[:len(encoded) // 2])
            writer.flush()
            process, connection = self.start(_reader_worker)
            blocked = self.receive(connection)
            self.assertEqual(blocked, {"event": "lock_blocked", "operation": fcntl.LOCK_SH})
            writer.write(encoded[len(encoded) // 2:])
            writer.flush()
            os.fsync(writer.fileno())
        result = self.receive(connection)
        self.finished(process)
        self.assertEqual(result["event"], "result", result)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["state"], self.journal.replay().state)
        self.evidence("reader_during_split_write", blocked=blocked, result=result, exit=0)

    def test_sigkill_after_fsync_then_new_process_retry_does_not_charge_twice(self):
        process, connection = self.start(_append_worker, None, market_payload(), True)
        self.assertEqual(self.receive(connection), {"event": "durable_before_response"})
        process.kill()
        self.finished(process, -signal.SIGKILL)
        committed_bytes = self.path.read_bytes()
        retry, retry_connection = self.start(_append_worker, None, market_payload())
        result = self.receive(retry_connection)
        self.finished(retry)
        self.assertEqual(result["event"], "result", result)
        self.assertFalse(result["appended"])
        self.assertEqual(result["count"], 3)
        self.assertEqual(self.path.read_bytes(), committed_bytes)
        self.assertEqual(self.journal.replay().state, result["state"])
        self.assertEqual(
            result["state"]["total_cost"],
            "0.0014977533699450823764353469795307039440838741887169",
        )
        self.evidence("sigkill_durable_before_response", result=result, exits=[-9, 0])

    def test_sigkill_partial_tail_is_preserved_and_refused_not_repaired(self):
        process, connection = self.start(_partial_writer)
        self.assertEqual(self.receive(connection), {"event": "partial_tail_durable"})
        process.kill()
        self.finished(process, -signal.SIGKILL)
        damaged_bytes = self.path.read_bytes()
        with self.assertRaises(JournalIntegrityError):
            self.journal.replay()
        retry, retry_connection = self.start(_append_worker, None, market_payload())
        result = self.receive(retry_connection)
        self.finished(retry)
        self.assertEqual(result["event"], "error", result)
        self.assertEqual(result["type"], "JournalIntegrityError")
        self.assertEqual(self.path.read_bytes(), damaged_bytes)
        self.evidence("sigkill_partial_tail", refusal=result, exits=[-9, 0])


if __name__ == "__main__":
    unittest.main()
