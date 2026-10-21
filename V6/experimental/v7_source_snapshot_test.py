from __future__ import annotations

import json
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    def reader_fixture(self, directory):
        path = snapshot_response(
            RequestSpec("POST", "https://example.test/query", {"a": 1}, body=b"q=1",
                        headers={"Accept": "application/json"}),
            b'{"value":1}', directory, retrieved_at="2026-09-17T10:00:00+08:00",
            parser_version="v1", content_type="application/json")
        return path, json.loads(path.read_text())

    def test_writer_rejects_unreadable_manifest_before_any_publication(self):
        defaults = {"retrieved_at": "2026-10-21T00:00:00Z", "parser_version": "v1"}
        cases = [{"status_code": value} for value in (0, 600, -1, True, None, "bad", float("inf"))]
        cases += [{"parser_version": ""}, {"content_type": None}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "unpublished"
                with self.assertRaises(SnapshotContractError):
                    snapshot_response(RequestSpec("GET", "https://example.test/data"), b"{}",
                                      root, **{**defaults, **kwargs})
                self.assertFalse(root.exists())
        for params in ({"nested": [float("nan")]}, {"nested": [float("inf")]},
                       {"nested": {1: "first", "1": "second"}}, {"value": {1, 2}}):
            with self.subTest(params=params), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "unpublished"
                with self.assertRaises(SnapshotContractError):
                    snapshot_response(RequestSpec("GET", "https://example.test/data", params),
                                      b"{}", root, **defaults)
                self.assertFalse(root.exists())

    def test_invalid_writer_retry_preserves_all_committed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.reader_fixture(root)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with self.assertRaises(SnapshotContractError):
                snapshot_response(RequestSpec("GET", "https://example.test/invalid"), b"new body",
                                  root, retrieved_at="2026-10-21T00:00:00Z",
                                  parser_version="v1", status_code=600)
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue(verify_snapshot(path))

    def test_reader_rejects_invalid_schema_types_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = self.reader_fixture(directory)
            cases = [
                ((), []), ((), None), (("response",), []), (("request",), []),
                (("response", "bytes"), True), (("response", "bytes"), "11"),
                (("response", "bytes"), 11.0), (("response", "bytes"), -1),
                (("response", "status_code"), "200"), (("response", "status_code"), True),
                (("response", "content_type"), None), (("parser_version",), ""),
                (("parser_version",), 1), (("retrieved_at",), "not-a-date"),
                (("schema_version",), "future-v2"), (("request", "method"), "DELETE"),
                (("request", "params"), []), (("request", "headers"), []),
                (("request", "body_sha256"), "bad"), (("response", "sha256"), "bad"),
                (("request_id",), "0"*64),
            ]
            for keys, value in cases:
                changed = copy.deepcopy(original)
                if not keys:
                    changed = value
                else:
                    target = changed
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                with self.subTest(keys=keys, value=value):
                    path.write_text(json.dumps(changed))
                    self.assertFalse(verify_snapshot(path))
            for section in (None, "request", "response"):
                for key in original if section is None else original[section]:
                    changed = copy.deepcopy(original)
                    del (changed if section is None else changed[section])[key]
                    with self.subTest(missing=(section, key)):
                        path.write_text(json.dumps(changed))
                        self.assertFalse(verify_snapshot(path))

    def test_reader_rejects_escape_paths_even_with_matching_bytes_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"snapshot"
            path, original = self.reader_fixture(root)
            raw = root/original["response"]["path"]
            outside = Path(directory)/"outside.json"
            outside.write_bytes(raw.read_bytes())
            for value in (str(outside), "../../outside.json", "../raw/"+raw.name,
                          "raw/../raw/"+raw.name, "raw//"+raw.name, "raw\\"+raw.name):
                with self.subTest(path=value):
                    changed = copy.deepcopy(original)
                    changed["response"]["path"] = value
                    path.write_text(json.dumps(changed))
                    self.assertFalse(verify_snapshot(path))

    def test_reader_rejects_symlink_files_and_directories(self):
        for kind in ("manifest", "raw_file", "raw_dir", "manifest_dir"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/"snapshot"
                path, original = self.reader_fixture(root)
                selected = {"manifest":path, "raw_file":root/original["response"]["path"],
                            "raw_dir":root/"raw", "manifest_dir":root/"manifests"}[kind]
                outside = Path(directory)/"outside"
                selected.rename(outside)
                selected.symlink_to(outside, target_is_directory=outside.is_dir())
                self.assertFalse(verify_snapshot(path))

    def test_reader_rejects_corruption_duplicates_and_partial_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = self.reader_fixture(directory)
            for text in ('{', '[]', path.read_text()[:-2], '{"schema_version":"bad",'+path.read_text()[1:]):
                with self.subTest(text=text[:30]):
                    path.write_text(text)
                    self.assertFalse(verify_snapshot(path))
            path.write_text(json.dumps(original))
            partial = path.with_name("."+path.name+".tmp")
            partial.write_bytes(path.read_bytes())
            self.assertFalse(verify_snapshot(partial))
            raw = Path(directory)/original["response"]["path"]
            raw.unlink()
            self.assertFalse(verify_snapshot(path))
            self.assertFalse(verify_snapshot(path.with_name("missing.json")))

    def test_reader_rejects_rehashed_request_under_wrong_publication_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.reader_fixture(directory)
            manifest["request"]["params"]["a"] = 2
            payload = json.dumps(manifest["request"], ensure_ascii=True, sort_keys=True,
                                 separators=(",", ":")).encode("ascii")
            manifest["request_id"] = hashlib.sha256(payload).hexdigest()
            path.write_text(json.dumps(manifest))
            self.assertFalse(verify_snapshot(path))

    def test_reader_accepts_existing_writer_shapes_and_http_error_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, (stamp, content, status) in enumerate((
                ("2026-09-17T10:00:00+08:00", "application/json", 200),
                ("2026-09-17T10:00:00", "application/octet-stream", 404),
                ("2026-09-17T10:00:00Z", "text/plain", 503),
            )):
                path = snapshot_response(RequestSpec("GET", "https://example.test/data",
                    {"finite": [0.5, -1.25e10]}),
                    b"", Path(directory)/str(index), retrieved_at=stamp,
                    parser_version="v1", content_type=content, status_code=status)
                self.assertTrue(verify_snapshot(path))

    def test_reader_rejects_overflowing_json_number_even_with_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.reader_fixture(directory)
            manifest["request"]["params"] = {"nested": [float("inf")]}
            canonical = json.dumps(manifest["request"], ensure_ascii=True, sort_keys=True,
                                   separators=(",", ":")).encode("ascii")
            manifest["request_id"] = hashlib.sha256(canonical).hexdigest()
            renamed = path.with_name(manifest["request_id"]+"-"+
                                     path.name.split("-", 1)[1])
            renamed.write_text(json.dumps(manifest).replace("Infinity", "1e999"))
            self.assertFalse(verify_snapshot(renamed))

    def test_failed_publication_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = RequestSpec("GET", "https://example.test/data")
            options = dict(retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            path = snapshot_response(spec, b"original", directory, **options)
            original = path.read_bytes()
            with patch("V6.experimental.v7_source_snapshot.os.link", side_effect=OSError("injected storage failure")):
                with self.assertRaises(OSError):
                    snapshot_response(spec, b"new", directory, **options)
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(verify_snapshot(path))
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    def test_corrupt_raw_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = RequestSpec("GET", "https://example.test/data")
            options = dict(retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            path = snapshot_response(spec, b"original", directory, **options)
            raw = Path(directory)/json.loads(path.read_text())["response"]["path"]
            raw.write_bytes(b"corrupt")
            with self.assertRaises(SnapshotContractError):
                snapshot_response(spec, b"original", directory, **options)
            self.assertEqual(raw.read_bytes(), b"corrupt")
            self.assertFalse(verify_snapshot(path))

    def test_manifest_symlink_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = RequestSpec("GET", "https://example.test/data")
            options = dict(retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            path = snapshot_response(spec, b"same", directory, **options)
            target = Path(directory)/"retained.json"
            path.rename(target)
            path.symlink_to(target)
            original = target.read_bytes()
            with self.assertRaises(SnapshotContractError):
                snapshot_response(spec, b"same", directory, **options)
            self.assertTrue(path.is_symlink())
            self.assertEqual(target.read_bytes(), original)

    def test_competing_processes_commit_exactly_one_conflicting_observation(self):
        code = """
import sys
from V6.experimental.v7_source_snapshot import *
sys.stdin.readline()
try:
    p = snapshot_response(RequestSpec("GET", "https://example.test/data"),
        sys.argv[2].encode(), sys.argv[1],
        retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
except SnapshotContractError:
    sys.exit(3)
print(p)
"""
        with tempfile.TemporaryDirectory() as directory:
            children = [subprocess.Popen([sys.executable, "-B", "-c", code, directory, body],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True) for body in ("first", "second")]
            try:
                for child in children:
                    child.stdin.write("go\n")
                    child.stdin.flush()
                outputs = [child.communicate(timeout=10) for child in children]
                self.assertEqual(sorted(child.returncode for child in children), [0, 3])
                winner = next(out.strip() for child, (out, _) in zip(children, outputs)
                              if child.returncode == 0)
                self.assertTrue(verify_snapshot(winner))
                self.assertEqual(len(list((Path(directory)/"manifests").glob("*.json"))), 1)
                self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.communicate()

    @unittest.skipUnless(os.name == "posix", "SIGKILL lifecycle requires POSIX")
    def test_killed_publisher_recovers_before_and_after_manifest_commit(self):
        # Controlled fault injection, not evidence of a real market-data run.
        code = """
import sys
from pathlib import Path
from V6.experimental import v7_source_snapshot as s
link = s.os.link
def interrupt(src, dst):
    if Path(dst).parent.name == "manifests":
        if sys.argv[2] == "after":
            link(src, dst)
        print("ready", flush=True)
        sys.stdin.readline()
        raise RuntimeError("parent should kill child")
    return link(src, dst)
s.os.link = interrupt
s.snapshot_response(s.RequestSpec("GET", "https://example.test/data"), b"same",
    sys.argv[1], retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
"""
        for stage in ("before", "after"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                child = subprocess.Popen([sys.executable, "-B", "-c", code, directory, stage],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    import select
                    self.assertTrue(select.select([child.stdout], [], [], 10)[0])
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                    child.kill()
                    child.communicate(timeout=10)
                    self.assertEqual(child.returncode, -9)
                    manifests = list((Path(directory)/"manifests").glob("*.json"))
                    self.assertEqual(len(manifests), int(stage == "after"))
                    path = snapshot_response(RequestSpec("GET", "https://example.test/data"),
                        b"same", directory, retrieved_at="2026-09-17T10:00:00+08:00",
                        parser_version="v1")
                    self.assertTrue(verify_snapshot(path))
                    # Interrupted temporary files remain non-authoritative; never deleted by recovery.
                    self.assertTrue(list(Path(directory).rglob("*.tmp")))
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.communicate()

    def test_conflicting_observation_cannot_replace_committed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = RequestSpec("GET", "https://example.test/data")
            options = dict(retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            path = snapshot_response(spec, b"original", directory, **options)
            original = path.read_bytes()
            for body, parser in ((b"changed", "v1"), (b"original", "v2")):
                with self.subTest(body=body, parser=parser):
                    with self.assertRaises(SnapshotContractError):
                        snapshot_response(spec, body, directory,
                                          **{**options, "parser_version": parser})
                    self.assertEqual(path.read_bytes(), original)
                    self.assertTrue(verify_snapshot(path))

    def test_identical_observation_is_noop_including_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = RequestSpec("GET", "https://example.test/data")
            options = dict(retrieved_at="2026-09-17T10:00:00+08:00", parser_version="v1")
            path = snapshot_response(spec, b"same", directory, **options)
            before = path.stat()
            self.assertEqual(snapshot_response(spec, b"same", directory, **options), path)
            after = path.stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns),
                             (after.st_ino, after.st_mtime_ns))

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
