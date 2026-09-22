import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from app.backend.routers import v7


class V7StatusApiTest(unittest.TestCase):
    def response(self):
        app = FastAPI()
        app.include_router(v7.router, prefix="/api")
        self.assertIn("/api/v7/status", [route.path for route in app.routes])
        result = asyncio.run(v7.status())
        if isinstance(result, dict):
            return 200, result
        return result.status_code, json.loads(result.body)

    def test_missing_publication_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"V7_RESULTS_DIR": directory}
        ):
            status, body = self.response()
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "not_ready")
        self.assertFalse(body["publish_allowed"])

    def test_valid_publication_is_returned(self):
        publication = {
            "schema": "v7-health-summary-v1",
            "state": "degraded",
            "publish_allowed": True,
            "data_id": "smoke",
            "generated_at": "2026-09-16T00:00:00Z",
            "counts": {"entries": 3, "affected_dates": 2, "affected_stocks": 1},
            "source": {"path": "/private/home/data-health.json"},
            "affected_stocks": ["private-stock-id"],
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "health-summary.json").write_text(
                json.dumps(publication), encoding="utf-8"
            )
            with patch.dict(os.environ, {"V7_RESULTS_DIR": directory}):
                status, body = self.response()
        self.assertEqual(status, 200)
        self.assertEqual(body, {key: publication[key] for key in (
            "schema", "state", "publish_allowed", "data_id", "generated_at", "counts"
        )})
        self.assertNotIn("private", json.dumps(body))

    def test_malformed_and_wrong_schema_fail_without_stale_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "health-summary.json")
            for raw in ("{broken", json.dumps({"schema": "old", "state": "healthy"})):
                target.write_text(raw, encoding="utf-8")
                with patch.dict(os.environ, {"V7_RESULTS_DIR": directory}):
                    status, body = self.response()
                self.assertEqual(status, 503)
                self.assertEqual(body["state"], "error")
                self.assertFalse(body["publish_allowed"])

    def test_private_identifier_and_conflicting_decision_fail_closed(self):
        publication = {
            "schema": "v7-health-summary-v1",
            "state": "blocked",
            "publish_allowed": True,
            "data_id": "/private/path",
            "generated_at": "2026-09-16T00:00:00Z",
            "counts": {"entries": 0, "affected_dates": 0, "affected_stocks": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "health-summary.json")
            for data_id, allowed in (("/private/path", True), ("safe-id", True)):
                publication["data_id"] = data_id
                publication["publish_allowed"] = allowed
                target.write_text(json.dumps(publication), encoding="utf-8")
                with patch.dict(os.environ, {"V7_RESULTS_DIR": directory}):
                    status, body = self.response()
                self.assertEqual(status, 503)
                self.assertNotIn("private", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
