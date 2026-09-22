"""Read-only V7 data-health publication endpoint."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v7", tags=["v7"])
SCHEMA = "v7-health-summary-v1"
VALID_STATES = {"healthy", "degraded", "blocked"}
DATA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "schema": SCHEMA,
            "state": "error",
            "publish_allowed": False,
            "message": message,
        },
    )


def _validate(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("publication must be an object")
    if document.get("schema") != SCHEMA:
        raise ValueError("unsupported publication schema")
    if document.get("state") not in VALID_STATES:
        raise ValueError("invalid publication state")
    if not isinstance(document.get("publish_allowed"), bool):
        raise ValueError("publish_allowed must be a boolean")
    if document["publish_allowed"] != (document["state"] != "blocked"):
        raise ValueError("publication decision conflicts with state")
    data_id = document.get("data_id")
    if not isinstance(data_id, str) or not DATA_ID.fullmatch(data_id):
        raise ValueError("invalid data identifier")
    generated_at = document.get("generated_at")
    if not isinstance(generated_at, str):
        raise ValueError("missing generation time")
    try:
        timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid generation time") from exc
    if timestamp.tzinfo is None:
        raise ValueError("generation time must include timezone")
    counts = document.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("counts must be an object")
    allowed_counts = ("entries", "affected_dates", "affected_stocks")
    if any(type(counts.get(key)) is not int or counts[key] < 0 for key in allowed_counts):
        raise ValueError("counts must be nonnegative integers")
    # Project only the public contract. The source summary also contains raw
    # paths, stock identifiers, dates and quality details for local research.
    return {
        "schema": SCHEMA,
        "state": document["state"],
        "publish_allowed": document["publish_allowed"],
        "data_id": data_id,
        "generated_at": generated_at,
        "counts": {key: counts[key] for key in allowed_counts},
    }


@router.get("/status")
async def status():
    results_dir = Path(os.environ.get("V7_RESULTS_DIR", "results/v7"))
    publication = results_dir / "health-summary.json"
    if not publication.is_file():
        return {
            "schema": SCHEMA,
            "state": "not_ready",
            "publish_allowed": False,
            "message": "V7 資料健康摘要尚未發布。",
        }

    try:
        document = json.loads(publication.read_text(encoding="utf-8"))
        return _validate(document)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return _error("V7 資料健康摘要無效；沒有沿用舊資料。")
