"""HTTP contract tests using only synthetic journals."""
import asyncio
import hashlib
import os
import unittest
from unittest.mock import patch
import httpx
from fastapi import FastAPI
from V6.experimental import v7_portfolio_history_test as fixtures
from app.backend.routers import v7_portfolio

class PortfolioApiTest(unittest.TestCase):
    def setUp(self):
        fixtures.HistoryTests.setUp(self)
        self.history = fixtures.HistoryTests.export(self)
        self.env = {"V7_PORTFOLIO_HISTORY_ENABLED": "1", "V7_PORTFOLIO_HISTORY_PATH": str(self.output), "V7_PORTFOLIO_HISTORY_SHA256": hashlib.sha256(self.output.read_bytes()).hexdigest()}
    def request(self, method="GET", client_host="127.0.0.1"):
        async def run():
            app = FastAPI(); app.include_router(v7_portfolio.router, prefix="/api")
            transport = httpx.ASGITransport(app=app, client=(client_host, 123))
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
                return await client.request(method, "/api/v7/portfolio-history")
        with patch.dict(os.environ, self.env, clear=True): return asyncio.run(run())
    def test_api_exact_artifact_and_schema(self):
        response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"state":"ready", "history":self.history})
        self.assertEqual(response.headers["cache-control"], "no-store")
    def test_api_disabled_missing_and_no_stale_fallback(self):
        self.assertEqual(self.request().json()["state"], "ready")
        self.output.unlink()
        self.assertEqual(self.request().json()["state"], "not_ready")
        self.env["V7_PORTFOLIO_HISTORY_ENABLED"] = "0"
        self.assertEqual(self.request().json()["state"], "not_ready")
    def test_api_corrupt_and_identity_missing_or_conflicting(self):
        for value in ("", "bad", "0" * 64):
            self.env["V7_PORTFOLIO_HISTORY_SHA256"] = value
            response = self.request()
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(str(self.root), response.text)
        self.output.write_text("{")
        self.env["V7_PORTFOLIO_HISTORY_SHA256"] = hashlib.sha256(self.output.read_bytes()).hexdigest()
        self.assertEqual(self.request().status_code, 503)
    def test_api_write_and_remote_refused(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            self.assertEqual(self.request(method).status_code, 405)
        self.assertEqual(self.request(client_host="203.0.113.1").status_code, 403)

if __name__ == "__main__": unittest.main()
