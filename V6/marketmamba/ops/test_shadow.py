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
            [("calendar", "SUCCEEDED"), ("ingest_raw", "SUCCEEDED")],
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

    def test_setup_failure_after_acquire_does_not_leave_active_lease(self) -> None:
        result = ({"prices_raw": True, "margin_raw": True}, [])
        self._observe(Mock(return_value=result))

        second_fetch = Mock(return_value=result)
        self.assertIs(self._observe(second_fetch), result)

        second_fetch.assert_called_once_with()
        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        heartbeat_at, expires_at = connection.execute(
            "SELECT heartbeat_at, expires_at FROM leases "
            "WHERE lease_name = 'v62-fetch-only'"
        ).fetchone()
        self.assertLessEqual(expires_at, heartbeat_at)
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            1,
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
