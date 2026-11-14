"""Tests for the committed-ledger current-state projection and CLI."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import fcntl
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from marketmamba.ops.diagnostics import render_current_state, sanitize_output_text
from marketmamba.ops.projection import (
    build_current_state,
    write_current_state,
)
from marketmamba.ops.schema import open_ledger
from marketmamba.ops.shadow import observe_fetch_only
from marketmamba.ops.contracts import ArtifactIdentity, LogicalRunIdentity, RunState
from marketmamba.ops.lease import acquire_lease
from marketmamba.ops.store import ArtifactRef, LedgerStore


class CurrentStateProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.v6_dir = self.root / "V6"
        self.v6_dir.mkdir()
        self.data_root = self.root / "Data"
        self.processed = self.data_root / "processed_v6"
        self.processed.mkdir(parents=True)
        self.db = self.root / "ops" / "operations.sqlite"
        self.sources = ("prices_raw", "margin_raw")
        for source in self.sources:
            (self.processed / f"{source}.parquet").write_bytes(source.encode())

    def _observe(self, day: int, result) -> None:
        observe_fetch_only(
            lambda: result,
            daily_sources=self.sources,
            v6_dir=self.v6_dir,
            ledger_path=self.db,
            data_root=self.data_root,
            requested_at=datetime(2026, 9, day, 14, tzinfo=timezone.utc),
        )

    def test_artifact_summary_counts_identities_not_stage_references(self) -> None:
        """Shared artifacts count once; distinct equal-sized artifacts count twice."""
        now = datetime(2026, 9, 18, 14, tzinfo=timezone.utc)
        self.db.parent.mkdir(parents=True)
        connection = open_ledger(self.db)
        try:
            store = LedgerStore(connection)
            lease = acquire_lease(
                connection, "daily-fetch", "offline-fixture",
                now=now, ttl=timedelta(minutes=5),
            )
            logical = store.open_logical_run(
                LogicalRunIdentity(requested_date=date(2026, 9, 18)), now=now,
            )
            attempt = store.start_attempt(logical, lease=lease, host="fixture", now=now)
            complete = ArtifactRef(
                ArtifactIdentity("fixture", "a" * 64, "v1", "b" * 64),
                "relative/one", 42, True,
            )
            incomplete = ArtifactRef(
                ArtifactIdentity("fixture", "c" * 64, "v1", "b" * 64),
                "relative/two", 42, False,
            )
            for stage_name, artifacts in (
                ("fetch", [complete, incomplete]), ("inspect", [complete]),
            ):
                stage = store.start_stage(
                    attempt.attempt_id, stage_name, "inputs-v1", now=now,
                )
                store.commit_stage(
                    stage.stage_run_id, RunState.SUCCEEDED, lease=lease,
                    artifacts=artifacts, now=now,
                )
            store.finish_attempt(
                attempt.attempt_id, RunState.SUCCEEDED,
                lease=lease, exit_code=0, now=now,
            )
            before = tuple(connection.iterdump())
            summary = build_current_state(self.db)["artifact_summary"]
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["complete_count"], 1)
            self.assertEqual(summary["total_bytes"], 84)
            self.assertEqual(summary["by_type"], [{
                "artifact_type": "fixture", "fingerprint_kind": "content_sha256",
                "count": 2, "complete_count": 1, "total_bytes": 84,
            }])
            self.assertEqual(tuple(connection.iterdump()), before)
            # Latest attempt must not inherit the previous attempt's artifacts.
            store.start_attempt(logical, lease=lease, host="fixture", now=now)
            summary = build_current_state(self.db)["artifact_summary"]
            self.assertEqual(summary["count"], 0)
            self.assertEqual(summary["complete_count"], 0)
            self.assertEqual(summary["total_bytes"], 0)
        finally:
            connection.close()

    def test_projection_reports_latest_retry_without_overclaiming_scope(self) -> None:
        self._observe(
            18,
            (
                {"prices_raw": True, "margin_raw": True},
                ["legacy reports missing"],
            ),
        )
        self._observe(
            18,
            ({"prices_raw": True, "margin_raw": True}, []),
        )

        state = build_current_state(self.db)

        self.assertEqual(state["mode"], "FETCH_ONLY")
        self.assertEqual(state["latest_requested_date"], "2026-09-18")
        self.assertEqual(state["latest_trading_date"], "2026-09-18")
        self.assertEqual(state["date_semantics"], "REQUESTED_ASIA_TAIPEI_DATE_UNVERIFIED_CALENDAR")
        self.assertTrue(state["latest_logical_run_id"].startswith("logical_run_v1_"))
        self.assertEqual(state["attempt_count"], 2)
        self.assertEqual(state["latest_attempt"], 2)
        self.assertEqual(state["run_status"], "SUCCEEDED")
        self.assertIsNone(state["active_stage"])
        self.assertGreaterEqual(state["duration_seconds"], 0)
        self.assertEqual(state["last_successful_fetch_date"], "2026-09-18")
        self.assertEqual(state["freshness"]["status"], "PASS")
        self.assertEqual(state["freshness"]["observed_value"], "missing=0")
        self.assertEqual(state["calendar"]["status"], "SKIPPED")
        self.assertEqual(state["calendar"]["knowledge"], "UNKNOWN_NOT_EXPOSED_BY_LEGACY")
        self.assertEqual(state["artifact_summary"]["count"], 2)
        self.assertEqual(state["artifact_summary"]["complete_count"], 2)
        self.assertEqual(state["artifact_summary"]["fingerprint_kinds"], ["file_stat_v1"])
        self.assertEqual(state["warning_count"], 0)
        self.assertEqual(state["incident_status"], "NOT_IMPLEMENTED_PHASE_C")
        self.assertIsNone(state["open_incident_count"])
        self.assertEqual(
            state["scope"]["excluded"],
            ["INFERENCE", "PORTFOLIO", "PUBLICATION"],
        )
        self.assertRegex(state["source_generation"], r"^[0-9a-f]{64}$")
        self.assertEqual(state["as_of"], state["source_watermark"]["committed_at"])

    def test_last_successful_fetch_date_does_not_treat_newer_failure_as_success(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        for source in self.sources:
            (self.processed / f"{source}.parquet").write_bytes(
                f"{source}-next-day".encode()
            )
        self._observe(
            19,
            ({"prices_raw": True, "margin_raw": False}, ["margin_raw missing"]),
        )

        state = build_current_state(self.db)

        self.assertEqual(state["latest_requested_date"], "2026-09-19")
        self.assertEqual(state["run_status"], "FAILED")
        self.assertEqual(state["last_successful_fetch_date"], "2026-09-18")
        self.assertEqual(state["freshness"]["status"], "FAIL")

    def test_fetch_exception_projects_failure_and_reraises_original(self) -> None:
        error = RuntimeError("provider token=secret-value failed")

        with self.assertRaises(RuntimeError) as raised:
            observe_fetch_only(
                lambda: (_ for _ in ()).throw(error),
                daily_sources=self.sources,
                v6_dir=self.v6_dir,
                ledger_path=self.db,
                data_root=self.data_root,
                requested_at=datetime(2026, 9, 18, 14, tzinfo=timezone.utc),
                projection_callback=write_current_state,
            )

        self.assertIs(raised.exception, error)
        state = json.loads(
            (self.db.parent / "current-state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["run_status"], "FAILED")
        self.assertEqual(state["freshness"]["status"], "FAIL")
        self.assertNotIn("secret-value", json.dumps(state))

    def test_projection_failure_does_not_replace_fetch_exception(self) -> None:
        error = ValueError("legacy error")
        callback_error = OSError("projection error")
        with self.assertRaises(ValueError) as raised:
            observe_fetch_only(
                lambda: (_ for _ in ()).throw(error),
                daily_sources=self.sources, v6_dir=self.v6_dir,
                ledger_path=self.db, data_root=self.data_root,
                projection_callback=lambda _: (_ for _ in ()).throw(callback_error),
            )
        self.assertIs(raised.exception, error)

    def _hold_projection_lock(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.db.parent / ".current-state.json.lock"
        handle = lock_path.open("a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(handle.close)
        return handle

    def test_stuck_projection_lock_cannot_delay_legacy_result_or_damage_json(self) -> None:
        target = self.db.parent / "current-state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        old = {"generation": "old", "valid": True}
        target.write_text(json.dumps(old), encoding="utf-8")
        lock_handle = self._hold_projection_lock()
        result = ({name: True for name in self.sources}, [])
        callback = lambda database: write_current_state(
            database,
            lock_timeout_seconds=0.05,
            lock_poll_seconds=0.005,
        )

        started = time.monotonic()
        observed = observe_fetch_only(
            lambda: result,
            daily_sources=self.sources,
            v6_dir=self.v6_dir,
            ledger_path=self.db,
            data_root=self.data_root,
            requested_at=datetime(2026, 9, 18, 14, tzinfo=timezone.utc),
            projection_callback=callback,
        )
        elapsed = time.monotonic() - started

        self.assertIs(observed, result)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), old)
        self.assertEqual(list(target.parent.glob(".current-state.json.*.tmp")), [])
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def test_stuck_projection_lock_cannot_replace_legacy_exception(self) -> None:
        target = self.db.parent / "current-state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        old_text = '{"generation":"old","valid":true}\n'
        target.write_text(old_text, encoding="utf-8")
        lock_handle = self._hold_projection_lock()
        error = RuntimeError("legacy provider failure")
        callback = lambda database: write_current_state(
            database,
            lock_timeout_seconds=0.05,
            lock_poll_seconds=0.005,
        )

        started = time.monotonic()
        with self.assertRaises(RuntimeError) as raised:
            observe_fetch_only(
                lambda: (_ for _ in ()).throw(error),
                daily_sources=self.sources,
                v6_dir=self.v6_dir,
                ledger_path=self.db,
                data_root=self.data_root,
                requested_at=datetime(2026, 9, 18, 14, tzinfo=timezone.utc),
                projection_callback=callback,
            )
        elapsed = time.monotonic() - started

        self.assertIs(raised.exception, error)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(target.read_text(encoding="utf-8"), old_text)
        self.assertEqual(list(target.parent.glob(".current-state.json.*.tmp")), [])
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def test_committed_evidence_paths_and_credentials_are_opaque_in_all_output(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        observed = (
            r"posix=/home/frank/private/data.parquet "
            r"windows=C:\Users\Master\private\data.csv "
            r'''quoted="D:\Market Data\private.csv" token=secret-value'''
        )
        threshold = (
            r"unc=\\server\secret-share\private.csv "
            r"absolute=file:/home/frank/private/raw.parquet "
            r"normal=file:processed_v6/prices_raw.parquet https://example.com/status"
        )
        connection = sqlite3.connect(self.db)
        connection.execute(
            "UPDATE checks SET observed_value = ?, threshold_value = ? "
            "WHERE check_name = 'freshness'",
            (observed, threshold),
        )
        connection.commit()
        connection.close()

        state = build_current_state(self.db)
        machine = json.dumps(state)
        from marketmamba.ops.__main__ import main
        human_stream = io.StringIO()
        with patch("sys.stdout", human_stream):
            self.assertEqual(main(["current-state", "--db", str(self.db)]), 0)
        human = human_stream.getvalue()

        for sensitive in (
            "/home/frank",
            "Master",
            "Market Data",
            "server",
            "secret-share",
            "secret-value",
        ):
            self.assertNotIn(sensitive, machine)
            self.assertNotIn(sensitive, human)
        self.assertEqual(state["freshness"]["observed_value"], "[PATH]")
        self.assertEqual(state["freshness"]["threshold_value"], "[PATH]")
        self.assertGreaterEqual(machine.count("[PATH]"), 2)
        self.assertIn("[PATH]", human)
        self.assertEqual(
            sanitize_output_text(
                "2026-09-18 PASS file:processed_v6/raw.parquet "
                "https://example.com/status"
            ),
            "2026-09-18 PASS file:processed_v6/raw.parquet "
            "https://example.com/status",
        )

    def test_unquoted_space_paths_and_extended_credentials_leave_no_fragments(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        observed = (
            r"posix=/home/jane/Secret Folder/private file.csv "
            r"windows=C:\Users\Jane Doe\private file.csv "
            r"unc=\\server\secret share\private file.csv "
            r"safe=ordinary prose"
        )
        threshold = (
            "authorization=Basic dXNlcjpwYXNz "
            "access_token=access-value client_secret=client-value "
            "AWS_SECRET_ACCESS_KEY=aws-value "
            "ftp://ftp-user:ftp-password@example.test/data "
            "postgresql://pg-user:pg-password@db.test/market "
            "safe_url=https://example.com/status"
        )
        connection = sqlite3.connect(self.db)
        connection.execute(
            "UPDATE checks SET observed_value = ?, threshold_value = ? "
            "WHERE check_name = 'freshness'",
            (observed, threshold),
        )
        connection.commit()
        connection.close()

        state = build_current_state(self.db)
        machine = json.dumps(state)
        from marketmamba.ops.__main__ import main
        human_stream = io.StringIO()
        with patch("sys.stdout", human_stream):
            self.assertEqual(main(["current-state", "--db", str(self.db)]), 0)
        human = human_stream.getvalue()

        for sensitive in (
            "/home/jane",
            "Secret Folder",
            "private file.csv",
            "Jane Doe",
            "server",
            "secret share",
            "dXNlcjpwYXNz",
            "access-value",
            "client-value",
            "aws-value",
            "ftp-user",
            "ftp-password",
            "pg-user",
            "pg-password",
        ):
            self.assertNotIn(sensitive, machine)
            self.assertNotIn(sensitive, human)
        self.assertEqual(state["freshness"]["observed_value"], "[PATH]")
        self.assertEqual(machine.count("[PATH]"), 1)
        self.assertNotIn("safe=ordinary prose", machine)
        self.assertNotIn("safe=ordinary prose", human)
        self.assertIn("https://example.com/status", machine)

    def test_bounded_path_segments_and_quoted_credentials_are_fully_opaque(
        self,
    ) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        observed = (
            r"prefix /home/jane/Secret Folder/private file.csv suffix "
            r"absolute=prefix file:/home/jane/Secret Folder/raw file.csv suffix "
            r"windows=prefix C:\Users\Jane Doe\private file.csv suffix "
            r"unc=prefix \\server\secret share\private file.csv suffix "
            r"status=PASS relative=file:processed_v6/raw.parquet"
        )
        threshold = (
            r'''authorization="Basic auth-one auth-two" '''
            r'''access_token='REDACTED_EVALUATION' '''
            r'''client_secret="client-one \"client-two\" client-three" '''
            r'''AWS_SECRET_ACCESS_KEY='aws-one \'aws-two\' aws-three' '''
            "safe=https://example.com/status"
        )
        connection = sqlite3.connect(self.db)
        connection.execute(
            "UPDATE checks SET observed_value = ?, threshold_value = ? "
            "WHERE check_name = 'freshness'",
            (observed, threshold),
        )
        connection.commit()
        connection.close()

        state = build_current_state(self.db)
        machine = json.dumps(state)
        from marketmamba.ops.__main__ import main
        human_stream = io.StringIO()
        with patch("sys.stdout", human_stream):
            self.assertEqual(main(["current-state", "--db", str(self.db)]), 0)
        human = human_stream.getvalue()

        for sensitive_fragment in (
            "prefix",
            "suffix",
            "/home/jane",
            "Secret Folder",
            "Jane Doe",
            "secret share",
            "private file.csv",
            "auth-one",
            "auth-two",
            "access-one",
            "access-two",
            "client-one",
            "client-two",
            "client-three",
            "aws-one",
            "aws-two",
            "aws-three",
        ):
            self.assertNotIn(sensitive_fragment, machine)
            self.assertNotIn(sensitive_fragment, human)
        self.assertEqual(state["freshness"]["observed_value"], "[PATH]")
        for tainted_neighbor in (
            "status=PASS",
            "file:processed_v6/raw.parquet",
        ):
            self.assertNotIn(tainted_neighbor, machine)
            self.assertNotIn(tainted_neighbor, human)
        self.assertIn("https://example.com/status", machine)
        self.assertIn("https://example.com/status", human)
        self.assertEqual(machine.count("[PATH]"), 1)
        self.assertGreaterEqual(machine.count("[REDACTED]"), 4)

    def test_any_absolute_path_opaques_the_entire_committed_evidence(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        cases = (
            (
                "posix",
                r'''prefix "/home/jane/Secret=Folder/private=file.csv", '''
                "suffix status=PASS",
            ),
            (
                "file_uri",
                r"prefix file:/home/jane/Secret=Folder/private=file.csv?x=1; "
                "suffix status=PASS",
            ),
            (
                "windows",
                r"prefix C:\Users\Jane= Doe\private=file.csv; suffix status=PASS",
            ),
            (
                "unc",
                r"prefix \\server\secret=share\private=file.csv, suffix status=PASS",
            ),
        )

        from marketmamba.ops.__main__ import main
        for label, evidence in cases:
            with self.subTest(path_family=label):
                connection = sqlite3.connect(self.db)
                connection.execute(
                    "UPDATE checks SET observed_value = ? "
                    "WHERE check_name = 'freshness'",
                    (evidence,),
                )
                connection.commit()
                connection.close()

                state = build_current_state(self.db)
                machine = json.dumps(state)
                human_stream = io.StringIO()
                with patch("sys.stdout", human_stream):
                    self.assertEqual(
                        main(["current-state", "--db", str(self.db)]),
                        0,
                    )
                human = human_stream.getvalue()

                self.assertEqual(state["freshness"]["observed_value"], "[PATH]")
                self.assertIn('"observed_value": "[PATH]"', machine)
                self.assertIn("observed: [PATH]", human)
                for leaked in ("prefix", "suffix", "status=PASS", "private=file"):
                    self.assertNotIn(leaked, machine)
                    self.assertNotIn(leaked, human)

    def test_colon_adjacent_paths_and_roots_are_opaque_but_safe_slashes_remain(
        self,
    ) -> None:
        path_cases = (
            "location:/home/frank/private/data.csv status=PASS",
            r"location:\\server\secret\data.csv status=PASS",
            "root=/ status=PASS",
            "root=file:/ status=PASS",
            "root=C:\\ status=PASS",
        )
        for evidence in path_cases:
            with self.subTest(evidence=evidence):
                self.assertEqual(sanitize_output_text(evidence), "[PATH]")

        safe_cases = (
            "https://example.com/a/b?x=1",
            "2026/09/18 status=PASS",
            "division 10 / 2 remains prose",
            "file:processed_v6/raw.parquet",
            "dataset/prices/v1 status=PASS",
        )
        for evidence in safe_cases:
            with self.subTest(safe=evidence):
                self.assertEqual(sanitize_output_text(evidence), evidence)
        self.assertEqual(
            sanitize_output_text(
                "https://url-user:url-password@example.com/a/b?x=1"
            ),
            "https://[REDACTED]@example.com/a/b?x=1",
        )

    def test_colon_adjacent_paths_and_roots_are_opaque_end_to_end(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        cases = (
            "location:/home/frank/private/data.csv status=PASS",
            r"location:\\server\secret\data.csv status=PASS",
            "root=/ status=PASS",
            "root=file:/ status=PASS",
            "root=C:\\ status=PASS",
        )

        from marketmamba.ops.__main__ import main
        for evidence in cases:
            with self.subTest(evidence=evidence):
                connection = sqlite3.connect(self.db)
                connection.execute(
                    "UPDATE checks SET observed_value = ? "
                    "WHERE check_name = 'freshness'",
                    (evidence,),
                )
                connection.commit()
                connection.close()

                state = build_current_state(self.db)
                machine = json.dumps(state)
                human_stream = io.StringIO()
                with patch("sys.stdout", human_stream):
                    self.assertEqual(
                        main(["current-state", "--db", str(self.db)]),
                        0,
                    )
                human = human_stream.getvalue()

                self.assertEqual(state["freshness"]["observed_value"], "[PATH]")
                self.assertIn('"observed_value": "[PATH]"', machine)
                self.assertIn("observed: [PATH]", human)
                self.assertNotIn("status=PASS", machine)
                self.assertNotIn("status=PASS", human)

    def test_projection_is_rebuildable_from_committed_ledger_only(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))

        first = build_current_state(self.db)
        second = build_current_state(self.db)

        self.assertEqual(first, second)

    def test_atomic_replace_failure_preserves_previous_valid_projection(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        target = self.root / "published" / "current-state.json"
        target.parent.mkdir()
        old = {"generation": "old", "valid": True}
        target.write_text(json.dumps(old), encoding="utf-8")

        with patch("marketmamba.ops.projection.os.replace", side_effect=OSError("stop")):
            with self.assertRaises(OSError):
                write_current_state(self.db, target)

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), old)
        self.assertEqual(list(target.parent.glob(".current-state.json.*.tmp")), [])

    def test_serialized_writers_cannot_publish_older_snapshot_last(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))
        target = self.root / "current-state.json"
        first_entered = threading.Event()
        release_first = threading.Event()
        real_atomic_write = __import__(
            "marketmamba.ops.projection", fromlist=["_atomic_write_json"]
        )._atomic_write_json
        calls = 0
        calls_lock = threading.Lock()

        def delayed_first(path, state):
            nonlocal calls
            with calls_lock:
                calls += 1
                ordinal = calls
            if ordinal == 1:
                first_entered.set()
                self.assertTrue(release_first.wait(5))
            return real_atomic_write(path, state)

        errors = []
        with patch("marketmamba.ops.projection._atomic_write_json", side_effect=delayed_first):
            first = threading.Thread(
                target=lambda: self._capture_error(
                    errors, lambda: write_current_state(self.db, target)
                )
            )
            first.start()
            self.assertTrue(first_entered.wait(5))
            self._observe(19, ({name: True for name in self.sources}, []))
            second = threading.Thread(
                target=lambda: self._capture_error(
                    errors, lambda: write_current_state(self.db, target)
                )
            )
            second.start()
            release_first.set()
            first.join(5)
            second.join(5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        published = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(published["latest_requested_date"], "2026-09-19")

    @staticmethod
    def _capture_error(errors, operation) -> None:
        try:
            operation()
        except BaseException as error:
            errors.append(error)

    def test_default_projection_path_is_next_to_database(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))

        written = write_current_state(self.db)

        self.assertEqual(written, self.db.parent / "current-state.json")
        self.assertEqual(
            json.loads(written.read_text(encoding="utf-8"))["run_status"],
            "SUCCEEDED",
        )

    def test_human_diagnostics_are_explicitly_fetch_only(self) -> None:
        self._observe(18, ({name: True for name in self.sources}, []))

        rendered = render_current_state(build_current_state(self.db))

        self.assertIn("Mode: FETCH_ONLY", rendered)
        self.assertIn("no inference, portfolio, or publication", rendered)
        self.assertIn("Calendar: SKIPPED", rendered)
        self.assertIn("unknown", rendered.lower())
        self.assertIn("Attempts: 1 (latest: 1)", rendered)
        self.assertIn("Incidents: NOT_IMPLEMENTED_PHASE_C", rendered)
        self.assertNotIn(str(self.root), rendered)

    def test_reader_opens_ledger_without_running_migrations(self) -> None:
        self.db.parent.mkdir(parents=True, exist_ok=True)
        connection = open_ledger(self.db)
        connection.close()
        before = sqlite3.connect(self.db).execute(
            "SELECT version, checksum, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()

        build_current_state(self.db)

        readonly = sqlite3.connect(self.db)
        self.addCleanup(readonly.close)
        after = readonly.execute(
            "SELECT version, checksum, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
        self.assertEqual(after, before)


class CurrentStateCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "operations.sqlite"
        self.v6 = self.root / "V6"
        self.data = self.root / "Data"
        self.v6.mkdir()
        processed = self.data / "processed_v6"
        processed.mkdir(parents=True)
        (processed / "prices_raw.parquet").write_bytes(b"prices")
        observe_fetch_only(
            lambda: ({"prices_raw": True}, []),
            daily_sources=("prices_raw",),
            v6_dir=self.v6,
            ledger_path=self.db,
            data_root=self.data,
            requested_at=datetime(2026, 9, 18, 14, tzinfo=timezone.utc),
        )

    def test_cli_human_default_and_json_option(self) -> None:
        from marketmamba.ops.__main__ import main

        human = io.StringIO()
        with patch("sys.stdout", human):
            self.assertEqual(main(["current-state", "--db", str(self.db)]), 0)
        self.assertIn("Mode: FETCH_ONLY", human.getvalue())

        machine = io.StringIO()
        with patch("sys.stdout", machine):
            self.assertEqual(
                main(["current-state", "--db", str(self.db), "--json"]), 0
            )
        state = json.loads(machine.getvalue())
        self.assertEqual(state["mode"], "FETCH_ONLY")
        self.assertEqual(state["incident_status"], "NOT_IMPLEMENTED_PHASE_C")

    def test_cli_failure_redacts_path_and_error_detail(self) -> None:
        from marketmamba.ops.__main__ import main

        secret_path = self.root / "token=secret-value" / "missing.sqlite"
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            self.assertEqual(
                main(["current-state", "--db", str(secret_path)]), 2
            )
        self.assertNotIn("secret-value", stderr.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())
        self.assertIn("unable to read operational ledger", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
