"""Minimal local launch of the same opt-in fixture router, without legacy startup.

From repository root, with explicit path/digest environment:
python -m uvicorn app.backend.portfolio_history_local:app --host 127.0.0.1 --port 8000
No worker, scheduler, data fetch or production configuration is loaded.
"""
from fastapi import FastAPI
from app.backend.routers.v7_portfolio import router

app = FastAPI(title="MarketMamba synthetic portfolio history (local only)")
app.include_router(router, prefix="/api")
