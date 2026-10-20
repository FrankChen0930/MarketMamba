"""Real local HTTP/restart acceptance. Requires FastAPI and uvicorn; no data/GPU."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from V6.experimental.v7_portfolio_fixture import create_fixture
from V6.experimental.v7_portfolio_journal import PortfolioJournal, _replay_records

ROOT = Path(__file__).resolve().parents[2]

class HistoryEndToEndTest(unittest.TestCase):
    def test_fixture_replay_export_http_restart_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            receipt = create_fixture(Path(temporary) / "fixture", source)
            artifact = Path(receipt["artifact"])
            raw = artifact.read_bytes()
            history = json.loads(raw)
            journal = PortfolioJournal(artifact.with_name("fixture.jsonl"))
            records = journal.records()
            self.assertEqual(len(records), 7)
            for seq, event in enumerate(history["events"], 1):
                self.assertEqual(event["state"], _replay_records(records[:seq]).state)
            self.assertEqual(history["events"][1]["state"]["total_cost"], "0")
            self.assertEqual(history["events"][3]["state"]["positions"], {})
            self.assertIsNotNone(history["events"][4]["state"]["pending"])
            self.assertIsNone(history["events"][5]["state"]["pending"])
            second = artifact.with_name("restarted-export.json")
            subprocess.run([sys.executable, "-B", "-m", "V6.experimental.v7_portfolio_history", "--journal", str(journal.path), "--fixture-manifest", str(artifact.with_name("fixture-manifest.json")), "--output", str(second), "--source-commit", source], cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(raw, second.read_bytes())
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "CUDA_VISIBLE_DEVICES":"", "V7_PORTFOLIO_HISTORY_ENABLED":"1", "V7_PORTFOLIO_HISTORY_PATH":str(artifact), "V7_PORTFOLIO_HISTORY_SHA256":receipt["sha256"]}
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0)); listener.listen(32)
                port = listener.getsockname()[1]
                url = f"http://127.0.0.1:{port}/api/v7/portfolio-history"
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                def get():
                    try:
                        with opener.open(url, timeout=1) as response:
                            return response.status, json.load(response), response.headers["Cache-Control"]
                    except urllib.error.HTTPError as response:
                        with response: return response.code, json.load(response), response.headers["Cache-Control"]
                def start():
                    process = subprocess.Popen([sys.executable, "-B", "-m", "uvicorn", "app.backend.portfolio_history_local:app", "--fd", str(listener.fileno()), "--no-proxy-headers", "--log-level", "error"], cwd=ROOT, env=env, pass_fds=(listener.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(40):
                        if process.poll() is not None: self.fail("local API exited")
                        try:
                            if get()[0] == 200: return process
                        except (OSError, urllib.error.URLError): time.sleep(.05)
                    process.terminate(); process.wait(timeout=5)
                    self.fail("local API startup timeout")
                def stop(process):
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                process = start()
                try:
                    status, body, cache = get()
                    self.assertEqual((status, cache), (200, "no-store"))
                    self.assertEqual(body, {"state":"ready", "history":history})
                    artifact.write_bytes(b"{")
                    self.assertEqual(get()[0:2], (503, {"state":"error", "message":"Fixture history corrupt or identity invalid"}))
                    artifact.unlink()
                    self.assertEqual(get()[1]["state"], "not_ready")
                    artifact.write_bytes(raw)
                    self.assertEqual(get()[1]["history"], history)
                finally: stop(process)
                restarted = start()
                try: self.assertEqual(get()[1]["history"], history)
                finally: stop(restarted)
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), receipt["sha256"])
            self.assertEqual(PortfolioJournal(journal.path).records(), records)
            print(json.dumps({"case":"fixture-replay-export-http-restart", "events":7, "sha256":receipt["sha256"], "source_commit":source, "strict_phase0":"STOP", "research_admission":False, "result":"PASS"}), flush=True)

if __name__ == "__main__": unittest.main()
