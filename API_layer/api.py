"""
FastAPI REST API for Financial Intelligence System
Endpoints for portfolio scanning, alerts, briefs, and live streaming.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Data.data_model import Portfolio, ThresholdConfig
from Storage.storage import StorageManager

logger = logging.getLogger(__name__)

storage = StorageManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await storage.connect_all()
    except Exception as e:
        logger.warning(f"Storage connection failed (degraded mode): {e}")
    yield
    await storage.close_all()


app = FastAPI(
    title="GeoFinancial Intelligence API",
    description="Portfolio monitoring with news-driven intelligence briefings",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────────

class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    use_sp500: bool = False
    portfolio_name: str = "Custom Portfolio"
    daily_sigma: float = 2.0
    weekly_sigma: float = 2.0
    news_lookback_hours: int = 72
    max_news_per_ticker: int = 20
    skip_news: bool = False


class ScanResponse(BaseModel):
    status: str
    message: str
    moves_detected: int = 0
    task_id: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "version": "2.0.0"}


pipeline_lock = asyncio.Lock()


@app.post("/api/v1/scan", response_model=ScanResponse)
async def trigger_scan(req: ScanRequest):
    """Trigger an on-demand portfolio scan."""
    if pipeline_lock.locked():
        return ScanResponse(status="busy", message="Scan already running")

    if not req.use_sp500 and not req.tickers:
        raise HTTPException(400, "Provide tickers or set use_sp500=true")

    async def _run():
        async with pipeline_lock:
            from Pipeline.run_pipeline import FinPipeline
            portfolio = Portfolio(
                name=req.portfolio_name,
                tickers=req.tickers or [],
                use_sp500=req.use_sp500,
                threshold_config=ThresholdConfig(
                    daily_sigma_threshold=req.daily_sigma,
                    weekly_sigma_threshold=req.weekly_sigma,
                ),
            )
            pipeline = FinPipeline(portfolio)
            await pipeline.run(
                news_lookback_hours=req.news_lookback_hours,
                max_news_per_ticker=req.max_news_per_ticker,
                skip_news=req.skip_news,
            )

    task = asyncio.create_task(_run())
    ticker_count = len(req.tickers or [])
    return ScanResponse(
        status="started",
        message=f"Scan triggered: {'S&P 500' if req.use_sp500 else f'{ticker_count} tickers'}",
        task_id=str(id(task)),
    )


@app.get("/api/v1/alerts")
async def get_alerts(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=200),
    level: Optional[str] = None,
):
    """Retrieve recent alerts."""
    try:
        alerts = await storage.postgres.get_recent_alerts(hours=hours, limit=limit)
        if level:
            alerts = [a for a in alerts if a.get("alert_level") == level]
        return {"count": len(alerts), "alerts": alerts}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/moves")
async def get_moves(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
    ticker: Optional[str] = None,
):
    """Retrieve recent flagged moves."""
    try:
        moves = await storage.postgres.get_recent_moves(hours=hours, limit=limit)
        if ticker:
            moves = [m for m in moves if m.get("ticker") == ticker.upper()]
        return {"count": len(moves), "moves": moves}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/briefs")
async def get_briefs(limit: int = Query(10, ge=1, le=50)):
    """Retrieve recent intelligence briefs."""
    pool = storage.postgres.pool
    if pool is None:
        raise HTTPException(503, "Database unavailable")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, date, portfolio_name, executive_summary, "
                "tickers_monitored, tickers_flagged, total_articles "
                "FROM briefs ORDER BY date DESC LIMIT $1", limit
            )
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/briefs/{brief_id}")
async def get_brief(brief_id: str):
    pool = storage.postgres.pool
    if pool is None:
        raise HTTPException(503, "Database unavailable")

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT brief_json FROM briefs WHERE id = $1", brief_id)
        if not row:
            raise HTTPException(404, "Brief not found")
        return json.loads(row["brief_json"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    try:
        pubsub = await storage.redis.subscribe_alerts()
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                await ws.send_text(msg["data"])
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API_layer.api:app", host="0.0.0.0", port=8000, reload=True)
