"""Integration tests for the non-authoritative fetch-only observer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from marketmamba.ops.lease import acquire_lease
from marketmamba.ops.schema import open_ledger
from marketmamba.ops.shadow import (
    artifact_ref_from_stat,
    default_ledger_path,
    observe_fetch_only,
)


class ShadowFetchObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.v6_dir = self.root / "V6"
        self.v6_dir.mkdir()
        self.data_root = self.root / "Data"
        self.processed = self.data_root / "processed_v6"
        self.processed.mkdir(parents=True)
        self.db = self.root / "runtime" / "operations.sqlite"
        self.logger = Mock(spec=logging.Logger)
        self.sources = ("prices_raw", "margin_raw")

    def _observe(self, fetch, **kwargs):
        ledger_path = kwargs.pop("ledger_path", self.db)
        data_root = kwargs.pop("data_root", self.data_root)
        return observe_fetch_only(
            fetch,
            daily_sources=self.sources,
            v6_dir=self.v6_dir,
            ledger_path=ledger_path,
            data_root=data_root,
            requested_at=datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc),
            logger=self.logger,
            **kwargs,
        )

    def test_success_returns_exact_legacy_result_and_commits_checks(self) -> None:
        for source in self.sources:
            (self.processed / f"{source}.parquet").write_bytes(source.encode())
        result = ({"prices_raw": True, "margin_raw": True}, [])
        fetch = Mock(return_value=result)

        observed = self._observe(fetch)

        self.assertIs(observed, result)
        fetch.assert_called_once_with()
        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute("SELECT state, exit_code FROM attempts").fetchone(),
            ("SUCCEEDED", 0),
        )
        self.assertEqual(
            connection.execute(
                "SELECT stage_name, state FROM stage_runs ORDER BY stage_name"
            ).fetchall(),
            [("calendar", "SKIPPED"), ("ingest_raw", "SUCCEEDED")],
        )
        self.assertEqual(
            dict(connection.execute("SELECT check_name, status FROM checks")),
            {
                "calendar_observation": "SKIP",
                "exit_gate": "PASS",
                "freshness": "PASS",
                "source_count": "PASS",
            },
        )
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 2
        )

    def test_missing_result_remains_failure_and_is_recorded(self) -> None:
        result = ({"prices_raw": True, "margin_raw": False}, ["margin_raw missing"])

        self.assertIs(self._observe(Mock(return_value=result)), result)

        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute("SELECT state, exit_code FROM attempts").fetchone(),
            ("FAILED", 1),
        )
        self.assertEqual(
            dict(connection.execute(
                "SELECT check_name, status FROM checks "
                "WHERE check_name != 'calendar_observation'"
            )),
            {"exit_gate": "FAIL", "freshness": "FAIL", "source_count": "FAIL"},
        )

    def test_existing_fetch_exception_is_recorded_and_reraised_unchanged(self) -> None:
        error = RuntimeError("provider token=secret-value failed")
        fetch = Mock(side_effect=error)

        with self.assertRaises(RuntimeError) as raised:
            self._observe(fetch)

        self.assertIs(raised.exception, error)
        fetch.assert_called_once_with()
        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute("SELECT state, exit_code FROM attempts").fetchone(),
            ("FAILED", 1),
        )
        persisted = " ".join(
            str(value)
            for row in connection.execute(
                "SELECT error_fingerprint, incident_ref FROM stage_runs"
            )
            for value in row
            if value
        )
        self.assertNotIn("secret-value", persisted)

    def test_ledger_initialization_failure_does_not_change_legacy_result(self) -> None:
        blocker = self.root / "not-a-directory"
        blocker.write_text("blocked", encoding="utf-8")
        result = ({"prices_raw": True, "margin_raw": True}, [])
        fetch = Mock(return_value=result)

        observed = self._observe(fetch, ledger_path=blocker / "operations.sqlite")

        self.assertIs(observed, result)
        fetch.assert_called_once_with()
        self.logger.warning.assert_called()

    def test_active_lease_conflict_does_not_skip_or_duplicate_legacy_fetch(self) -> None:
        self.db.parent.mkdir(parents=True)
        connection = open_ledger(self.db)
        self.addCleanup(connection.close)
        acquire_lease(
            connection,
            "v62-fetch-only",
            "other-owner",
            ttl=timedelta(hours=1),
        )
        result = ({"prices_raw": True, "margin_raw": True}, [])
        fetch = Mock(return_value=result)

        observed = self._observe(fetch)

        self.assertIs(observed, result)
        fetch.assert_called_once_with()
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 0
        )
        self.logger.warning.assert_called()

    def test_two_terminal_runs_create_distinct_attempt_and_stage_history(self) -> None:
        result = ({"prices_raw": True, "margin_raw": True}, [])
        first_fetch = Mock(return_value=result)
        second_fetch = Mock(return_value=result)

        self.assertIs(self._observe(first_fetch), result)
        self.assertIs(self._observe(second_fetch), result)

        first_fetch.assert_called_once_with()
        second_fetch.assert_called_once_with()
        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute(
                "SELECT attempt_number, state, exit_code FROM attempts "
                "ORDER BY attempt_number"
            ).fetchall(),
            [(1, "SUCCEEDED", 0), (2, "SUCCEEDED", 0)],
        )
        self.assertEqual(
            connection.execute(
                "SELECT attempts.attempt_number, stage_runs.stage_name, "
                "stage_runs.state FROM stage_runs JOIN attempts USING (attempt_id) "
                "ORDER BY attempts.attempt_number, stage_runs.stage_name"
            ).fetchall(),
            [(1, "calendar", "SKIPPED"), (1, "ingest_raw", "SUCCEEDED"),
             (2, "calendar", "SKIPPED"), (2, "ingest_raw", "SUCCEEDED")],
        )

    def test_projection_callback_failure_is_non_authoritative(self) -> None:
        result = ({"prices_raw": True, "margin_raw": True}, [])
        projection = Mock(side_effect=RuntimeError("projection failed"))

        observed = self._observe(
            Mock(return_value=result), projection_callback=projection
        )

        self.assertIs(observed, result)
        projection.assert_called_once_with(self.db)
        self.logger.warning.assert_called()

    def test_projection_is_not_requested_after_ledger_commit_failure(self) -> None:
        result = ({"prices_raw": True, "margin_raw": True}, [])
        projection = Mock()

        with patch(
            "marketmamba.ops.shadow._record_result",
            side_effect=RuntimeError("commit failed"),
        ):
            observed = self._observe(
                Mock(return_value=result),
                projection_callback=projection,
            )

        self.assertIs(observed, result)
        projection.assert_not_called()

    def test_hostile_logger_and_config_path_do_not_block_legacy_fetch(self) -> None:
        class HostileEnvironment(dict):
            def get(self, key, default=None):
                raise RuntimeError(f"blocked config {key}")

        logger = Mock(spec=logging.Logger)
        logger.warning.side_effect = RuntimeError("logger failed")
        result = ({"prices_raw": True, "margin_raw": True}, [])
        fetch = Mock(return_value=result)

        observed = observe_fetch_only(
            fetch,
            daily_sources=self.sources,
            v6_dir=self.v6_dir,
            logger=logger,
            environ=HostileEnvironment(),
        )

        self.assertIs(observed, result)
        fetch.assert_called_once_with()

    def test_hostile_path_does_not_block_legacy_fetch(self) -> None:
        class HostilePath:
            def __fspath__(self):
                raise RuntimeError("path conversion failed")

        result = ({"prices_raw": True, "margin_raw": True}, [])
        fetch = Mock(return_value=result)

        observed = observe_fetch_only(
            fetch,
            daily_sources=self.sources,
            v6_dir=self.v6_dir,
            ledger_path=HostilePath(),
            logger=self.logger,
        )

        self.assertIs(observed, result)
        fetch.assert_called_once_with()
        self.logger.warning.assert_called()

    def test_release_and_logger_failures_cannot_replace_legacy_exception(self) -> None:
        error = ValueError("legacy failure")
        fetch = Mock(side_effect=error)
        self.logger.warning.side_effect = RuntimeError("logger failed")

        with (
            patch(
                "marketmamba.ops.shadow.release_lease",
                side_effect=RuntimeError("release failed"),
            ),
            self.assertRaises(ValueError) as raised,
        ):
            self._observe(fetch)

        self.assertIs(raised.exception, error)
        self.assertEqual(str(raised.exception), "legacy failure")

    def test_connection_close_failure_cannot_replace_legacy_result(self) -> None:
        self.db.parent.mkdir(parents=True)
        real_connection = open_ledger(self.db)

        class CloseFailureConnection:
            def __getattr__(self, name):
                return getattr(real_connection, name)

            def close(self):
                real_connection.close()
                raise RuntimeError("close failed")

        result = ({"prices_raw": True, "margin_raw": True}, [])
        with patch(
            "marketmamba.ops.shadow.open_ledger",
            return_value=CloseFailureConnection(),
        ):
            observed = self._observe(Mock(return_value=result))

        self.assertIs(observed, result)

    def test_source_count_is_independent_from_authoritative_missing_list(self) -> None:
        result = (
            {"prices_raw": True, "margin_raw": True},
            ["legacy reports missing"],
        )

        self.assertIs(self._observe(Mock(return_value=result)), result)

        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        self.assertEqual(
            dict(connection.execute(
                "SELECT check_name, status FROM checks "
                "WHERE check_name != 'calendar_observation'"
            )),
            {"exit_gate": "FAIL", "freshness": "FAIL", "source_count": "PASS"},
        )

    def test_stat_artifact_declares_metadata_fingerprint_semantics(self) -> None:
        path = self.processed / "prices_raw.parquet"
        path.write_bytes(b"not hashed as content")

        artifact = artifact_ref_from_stat(
            path, data_root=self.data_root, complete=True
        )

        self.assertEqual(
            artifact.identity.artifact_type, "raw_parquet_metadata_ref"
        )
        self.assertEqual(
            artifact.identity.schema_version, "stat-fingerprint-v1"
        )
        self.assertEqual(artifact.identity.fingerprint_kind, "file_stat_v1")
        self.assertIsNone(artifact.identity.content_sha256)
        self.assertEqual(len(artifact.identity.fingerprint_sha256 or ""), 64)
        self.assertEqual(artifact.byte_size, len(b"not hashed as content"))
        self.assertEqual(artifact.uri, "file:processed_v6/prices_raw.parquet")

    def test_default_ledger_path_is_under_ignored_data_directory(self) -> None:
        self.assertEqual(
            default_ledger_path(self.v6_dir, environ={}),
            self.root / "Data" / "ops" / "operations.sqlite",
        )
        self.assertEqual(
            default_ledger_path(
                self.v6_dir,
                environ={"MARKETMAMBA_OPS_ROOT": str(self.root / "ops")},
            ),
            self.root / "ops" / "operations.sqlite",
        )


if __name__ == "__main__":
    unittest.main()
