"""Immutable, replayable source snapshots for V7 historical PIT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse


class SnapshotContractError(ValueError):
    pass


_SENSITIVE_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie", "set-cookie"}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Publish once; identical retries reuse bytes, conflicting identities fail.

    A unique staged file is fsynced before no-clobber publication. This assumes
    a trusted snapshot directory and a filesystem supporting hard links/fsync;
    unsupported storage fails rather than falling back to overwrite.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise SnapshotContractError("immutable snapshot identity conflicts with existing bytes")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RequestSpec:
    method: str
    url: str
    params: Mapping[str, Any] = field(default_factory=dict)
    body: bytes | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = str(self.method).upper().strip()
        parsed = urlparse(str(self.url))
        if method not in {"GET", "POST"}:
            raise SnapshotContractError("only GET and POST source requests are supported")
        if parsed.scheme != "https" or not parsed.netloc:
            raise SnapshotContractError("source URL must be an absolute HTTPS URL")
        if any(str(name).lower() in _SENSITIVE_HEADERS for name in self.headers):
            raise SnapshotContractError("credentials and cookies must not enter replay manifests")
        if self.body is not None and not isinstance(self.body, bytes):
            raise SnapshotContractError("request body must be bytes")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "url", str(self.url))

    def canonical_request(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "params": dict(sorted((str(key), value) for key, value in self.params.items())),
            "body_sha256": _sha256(self.body or b""),
            "headers": dict(sorted((str(key).lower(), str(value)) for key, value in self.headers.items())),
        }

    @property
    def request_id(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_request()))


def snapshot_response(
    request: RequestSpec,
    response_body: bytes,
    root: str | Path,
    *,
    retrieved_at: str,
    parser_version: str,
    content_type: str = "application/octet-stream",
    status_code: int = 200,
) -> Path:
    # Validate the serialized v1 observation with the reader's own rules before
    # publishing even its raw blob. Normalized valid writer output stays stable.
    try:
        if not isinstance(response_body, bytes):
            raise SnapshotContractError("response body must be bytes")
        if not isinstance(content_type, str):
            raise SnapshotContractError("content_type must be text")
        timestamp = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
        base = Path(root)
        response_sha = _sha256(response_body)
        suffix = ".json" if "json" in content_type.lower() else ".bin"
        raw_relative = Path("raw") / f"{response_sha}{suffix}"
        manifest = {
            "schema_version": "v7-source-snapshot-v1",
            "request_id": request.request_id,
            "request": request.canonical_request(),
            "retrieved_at": timestamp.isoformat(),
            "parser_version": str(parser_version),
            "response": {
                "status_code": int(status_code),
                "content_type": content_type,
                "bytes": len(response_body),
                "sha256": response_sha,
                "path": raw_relative.as_posix(),
            },
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True,
                             allow_nan=False).encode("utf-8") + b"\n"
        decoded = json.loads(payload, object_pairs_hook=_unique_object,
                             parse_constant=_reject_json_constant,
                             parse_float=_finite_json_float)
        manifest_name, _ = _manifest_identity(decoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise SnapshotContractError(f"invalid snapshot observation: {error}") from error
    manifest_path = base / "manifests" / manifest_name
    _atomic_bytes(base / raw_relative, response_body)
    _atomic_bytes(manifest_path, payload)
    return manifest_path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotContractError("duplicate manifest key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SnapshotContractError(f"non-finite JSON constant: {value}")


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise SnapshotContractError("overflowing JSON number")
    return value


def _manifest_identity(manifest: Any) -> tuple[str, str]:
    """Validate v1 writer shape; return only canonical publication basenames."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "request_id", "request", "retrieved_at", "parser_version", "response"
    }:
        raise SnapshotContractError("invalid manifest fields")
    request, response = manifest["request"], manifest["response"]
    if (manifest["schema_version"] != "v7-source-snapshot-v1"
            or not isinstance(request, dict) or set(request) != {
                "method", "url", "params", "body_sha256", "headers"}
            or not isinstance(response, dict) or set(response) != {
                "status_code", "content_type", "bytes", "sha256", "path"}):
        raise SnapshotContractError("invalid v1 schema")
    for value in (manifest["request_id"], request["body_sha256"], response["sha256"]):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise SnapshotContractError("invalid digest")
    if (not isinstance(manifest["retrieved_at"], str)
            or not isinstance(manifest["parser_version"], str)
            or not manifest["parser_version"].strip()
            or type(response["bytes"]) is not int or response["bytes"] < 0
            or type(response["status_code"]) is not int
            or not 100 <= response["status_code"] <= 599
            or not isinstance(response["content_type"], str)
            or not isinstance(request["method"], str)
            or not isinstance(request["url"], str)
            or not isinstance(request["params"], dict)
            or not isinstance(request["headers"], dict)
            or any(not isinstance(v, str) for v in request["headers"].values())):
        raise SnapshotContractError("invalid field types")
    canonical = RequestSpec(request["method"], request["url"], request["params"],
                            headers=request["headers"]).canonical_request()
    canonical["body_sha256"] = request["body_sha256"]
    if canonical != request or _sha256(_canonical_bytes(request)) != manifest["request_id"]:
        raise SnapshotContractError("request identity mismatch")
    timestamp = datetime.fromisoformat(manifest["retrieved_at"])
    if timestamp.isoformat() != manifest["retrieved_at"]:
        raise SnapshotContractError("noncanonical retrieval timestamp")
    stamp = timestamp.isoformat().replace(":", "").replace("+", "p")
    suffix = ".json" if "json" in response["content_type"].lower() else ".bin"
    raw_name = response["sha256"] + suffix
    if response["path"] != "raw/" + raw_name:
        raise SnapshotContractError("noncanonical raw path")
    return f"{manifest['request_id']}-{stamp}.json", raw_name


def _read_regular_at(directory: int, name: str) -> bytes:
    # Nonblocking prevents a substituted FIFO from hanging validation.
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SnapshotContractError("snapshot must be a regular file")
        return stream.read()


def verify_snapshot(manifest_path: str | Path) -> bool:
    """Read-only integrity check, not source authenticity or research admission.

    v1 uses manifests/<request>-<timestamp>.json and raw/<digest>.<suffix>.
    Directory descriptors anchor reads; symlinks and noncanonical paths fail.
    Snapshot root ancestors must remain trusted/stable during the operation.
    """
    descriptors = []
    try:
        given = Path(manifest_path)
        if ".." in given.parts:
            return False
        path = given.absolute()
        root = path.parent.parent
        if path.parent.name != "manifests" or root.resolve() != root:
            return False
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(root, flags)
        descriptors.append(root_fd)
        manifest_fd = os.open("manifests", flags, dir_fd=root_fd)
        descriptors.append(manifest_fd)
        manifest = json.loads(_read_regular_at(manifest_fd, path.name),
                              object_pairs_hook=_unique_object,
                              parse_constant=_reject_json_constant,
                              parse_float=_finite_json_float)
        manifest_name, raw_name = _manifest_identity(manifest)
        if path.name != manifest_name:
            return False
        raw_fd = os.open("raw", flags, dir_fd=root_fd)
        descriptors.append(raw_fd)
        raw = _read_regular_at(raw_fd, raw_name)
        return (len(raw) == manifest["response"]["bytes"]
                and _sha256(raw) == manifest["response"]["sha256"])
    except (OSError, KeyError, TypeError, ValueError, RecursionError, RuntimeError):
        return False
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
