from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from V6.experimental.v7_source_snapshot import (
    RequestSpec,
    SnapshotContractError,
    snapshot_response,
    verify_snapshot,
)


class RequestSpecTest(unittest.TestCase):
    def test_request_identity_is_canonical_and_rejects_credentials(self):
        first = RequestSpec("get", "https://example.test/data", {"b": "2", "a": "1"})
        second = RequestSpec("GET", "https://example.test/data", {"a": "1", "b": "2"})
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(first.canonical_request()["method"], "GET")
        with self.assertRaises(SnapshotContractError):
            RequestSpec("GET", "https://example.test/data", headers={"Authorization": "secret"})


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_atomic_fingerprinted_and_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RequestSpec("POST", "https://example.test/query", {"offset": 0}, body=b"q=2330")
            manifest_path = snapshot_response(
                spec,
                b'{"ok":true}\n',
                root,
                retrieved_at="2026-09-17T10:00:00+08:00",
                parser_version="security-master-v1",
                content_type="application/json",
                status_code=200,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["response"]["sha256"], "e5f1eb4d806641698a35efe20e098efd20d7d57a9b90ee69079d5bb650920726")
            self.assertEqual(manifest["response"]["bytes"], 12)
            self.assertEqual(manifest["parser_version"], "security-master-v1")
            self.assertTrue(verify_snapshot(manifest_path))
            raw_path = root / manifest["response"]["path"]
            raw_path.write_bytes(b"tampered")
            self.assertFalse(verify_snapshot(manifest_path))

    def test_identical_response_reuses_content_addressed_raw_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RequestSpec("GET", "https://example.test/data")
            one = snapshot_response(spec, b"same", root, retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            two = snapshot_response(spec, b"same", root, retrieved_at="2026-09-17T11:00:00+08:00", parser_version="v1")
            left, right = json.loads(one.read_text()), json.loads(two.read_text())
            self.assertEqual(left["response"]["path"], right["response"]["path"])
            self.assertEqual(len(list((root / "raw").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
