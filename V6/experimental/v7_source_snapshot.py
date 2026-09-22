"""Immutable, replayable source snapshots for V7 historical PIT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


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
    if not isinstance(response_body, bytes):
        raise SnapshotContractError("response body must be bytes")
    try:
        timestamp = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    except ValueError as error:
        raise SnapshotContractError("retrieved_at must be an ISO timestamp") from error
    if not str(parser_version).strip():
        raise SnapshotContractError("parser_version is required")
    base = Path(root)
    response_sha = _sha256(response_body)
    suffix = ".json" if "json" in content_type.lower() else ".bin"
    raw_relative = Path("raw") / f"{response_sha}{suffix}"
    raw_path = base / raw_relative
    if raw_path.exists():
        if _sha256(raw_path.read_bytes()) != response_sha:
            raise SnapshotContractError("content-addressed raw snapshot is corrupted")
    else:
        _atomic_bytes(raw_path, response_body)
    stamp = timestamp.isoformat().replace(":", "").replace("+", "p")
    manifest = {
        "schema_version": "v7-source-snapshot-v1",
        "request_id": request.request_id,
        "request": request.canonical_request(),
        "retrieved_at": timestamp.isoformat(),
        "parser_version": str(parser_version),
        "response": {
            "status_code": int(status_code),
            "content_type": str(content_type),
            "bytes": len(response_body),
            "sha256": response_sha,
            "path": raw_relative.as_posix(),
        },
    }
    manifest_path = base / "manifests" / f"{request.request_id}-{stamp}.json"
    _atomic_bytes(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return manifest_path


def verify_snapshot(manifest_path: str | Path) -> bool:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        raw_path = path.parent.parent / manifest["response"]["path"]
        raw = raw_path.read_bytes()
        return (
            manifest["schema_version"] == "v7-source-snapshot-v1"
            and len(raw) == int(manifest["response"]["bytes"])
            and _sha256(raw) == manifest["response"]["sha256"]
            and _sha256(_canonical_bytes(manifest["request"])) == manifest["request_id"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
