"""
MarketMamba V6 — FastAPI Backend
=================================
Local:  uvicorn main:app --reload --port 8000
Cloud:  Render (see render.yaml)
Docs:   http://localhost:8000/docs
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import signals, performance, market, portfolio, reports


app = FastAPI(
    title="MarketMamba V6 API",
    description="台股截面因子選股系統 — Alpha 訊號 / 量化績效 / 持倉管理",
    version="6.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Local dev: allow Vite (5173) and any localhost
# Production: set ALLOWED_ORIGINS env var in Render Dashboard
_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:1420"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(signals.router,     prefix="/api")
app.include_router(performance.router, prefix="/api")
app.include_router(market.router,      prefix="/api")
app.include_router(portfolio.router,   prefix="/api")
app.include_router(reports.router,     prefix="/api")



@app.get("/")
@app.head("/")
async def root():
    """Root endpoint — Uptime Robot ping target"""
    return {"status": "ok", "service": "MarketMamba V6"}


@app.get("/health")
@app.head("/health")
@app.get("/api/health")
@app.head("/api/health")
async def health():
    """Health check — GET and HEAD both supported (Uptime Robot compatibility)"""
    return {"status": "ok", "service": "MarketMamba V6", "version": "6.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

