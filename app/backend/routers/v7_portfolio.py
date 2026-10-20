"""Opt-in loopback-only fixture history. No cache, source-path request or mutation."""
import os
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from V6.experimental.v7_portfolio_history import HistoryError, read_history

router = APIRouter(prefix="/v7", tags=["v7-fixture-history"])

def response(state, status=200, **values):
    return JSONResponse({"state": state, **values}, status_code=status, headers={"Cache-Control": "no-store"})

@router.get("/portfolio-history")
def portfolio_history(request: Request):
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        return response("error", 403, message="Local fixture service only")
    if os.getenv("V7_PORTFOLIO_HISTORY_ENABLED") != "1":
        return response("not_ready", message="Fixture history disabled")
    path = os.getenv("V7_PORTFOLIO_HISTORY_PATH")
    if not path:
        return response("not_ready", message="Fixture history unavailable")
    try:
        # Missing is distinct from unreadable/corrupt; no fallback to any prior value.
        if not Path(path).exists():
            return response("not_ready", message="Fixture history unavailable")
        history = read_history(path, expected_sha256=os.getenv("V7_PORTFOLIO_HISTORY_SHA256", ""))
    except (HistoryError, OSError, ValueError):
        return response("error", 503, message="Fixture history corrupt or identity invalid")
    return response("ready", history=history)
