"""
FastAPI REST API for GeoFinancial Intelligence System
SQLite-backed endpoints for briefs, alerts, moves, clusters, and SQL explorer.

Endpoints:
    GET  /health                        System health check
    GET  /api/v1/briefs                 List recent briefs
    GET  /api/v1/briefs/latest          Full latest brief (dashboard landing)
    GET  /api/v1/briefs/{id}            Full brief by ID
    GET  /api/v1/moves                  Query flagged moves (filter by ticker, level, sigma)
    GET  /api/v1/moves/{ticker}         Move history for a specific ticker
    GET  /api/v1/alerts/critical        Critical alerts from last N days
    GET  /api/v1/clusters/{brief_id}    Clusters for a specific brief
    POST /api/v1/scan                   Trigger a pipeline scan
    POST /api/v1/sql                    Execute a read-only SQL query
    GET  /api/v1/stats                  Database statistics
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Storage.sqlite_store import BriefDatabase

logger = logging.getLogger(__name__)

db = BriefDatabase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Database ready: {db.stats()}")
    yield


app = FastAPI(
    title="GeoFinancial Intelligence API",
    description="Portfolio monitoring with factor-decomposed intelligence briefings",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    use_sp500: bool = False
    portfolio_name: str = "Custom Portfolio"
    daily_sigma: float = 2.0
    weekly_sigma: float = 2.0
    idio_sigma: float = 1.5
    skip_news: bool = False


class ScanResponse(BaseModel):
    status: str
    message: str
    brief_id: Optional[str] = None


class SqlRequest(BaseModel):
    query: str
    limit: int = 200


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    stats = db.stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.1.0",
        "database": stats,
    }


# ── Briefs ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/briefs")
async def list_briefs(limit: int = Query(20, ge=1, le=100)):
    """List recent briefs (summary metadata, no full alerts)."""
    return db.get_recent_briefs(limit)


@app.get("/api/v1/briefs/latest")
async def get_latest_brief():
    """Full latest brief with all alerts, clusters, recommendations, sectors."""
    brief = db.get_latest_brief()
    if not brief:
        raise HTTPException(404, "No briefs in database. Run a scan first.")
    return brief


@app.get("/api/v1/briefs/{brief_id}")
async def get_brief(brief_id: str):
    """Full brief by ID."""
    brief = db.get_brief_full(brief_id)
    if not brief:
        raise HTTPException(404, f"Brief {brief_id} not found")
    return brief


# ── Moves / Alerts ───────────────────────────────────────────────────────────

@app.get("/api/v1/moves")
async def get_moves(
    ticker: Optional[str] = None,
    level: Optional[str] = None,
    min_sigma: Optional[float] = None,
    brief_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Query flagged moves with optional filters."""
    conditions = []
    params = []

    if brief_id:
        conditions.append("fm.brief_id = ?")
        params.append(brief_id)
    if ticker:
        conditions.append("fm.ticker = ?")
        params.append(ticker.upper())
    if level:
        conditions.append("fm.alert_level = ?")
        params.append(level.lower())
    if min_sigma is not None:
        conditions.append("fm.idiosyncratic_sigma >= ?")
        params.append(min_sigma)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    query = f"""
        SELECT fm.*, b.date as brief_date, b.portfolio_name
        FROM flagged_moves fm
        JOIN briefs b ON fm.brief_id = b.id
        {where}
        ORDER BY fm.idiosyncratic_sigma DESC
        LIMIT ?
    """

    result = db.execute_sql(query.replace("?", "{}").format(*params))
    # Use parameterized query directly instead
    with db._connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return {"count": len(rows), "moves": [dict(r) for r in rows]}


@app.get("/api/v1/moves/{ticker}")
async def get_ticker_history(ticker: str, limit: int = Query(50, ge=1, le=200)):
    """All flagged moves for a specific ticker across all briefs."""
    return db.get_moves_by_ticker(ticker, limit)


@app.get("/api/v1/alerts/critical")
async def get_critical_alerts(days: int = Query(7, ge=1, le=90)):
    """Critical alerts from the last N days."""
    return db.get_critical_alerts(days)


# ── Clusters ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/clusters/{brief_id}")
async def get_clusters(brief_id: str):
    """Causal clusters for a specific brief."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM causal_clusters WHERE brief_id = ? ORDER BY cluster_id",
            (brief_id,)
        ).fetchall()
        clusters = []
        for r in rows:
            c = dict(r)
            c["tickers"] = json.loads(c.get("tickers") or "[]")
            clusters.append(c)
        return clusters


# ── SQL Explorer ─────────────────────────────────────────────────────────────

@app.post("/api/v1/sql")
async def execute_sql(req: SqlRequest):
    """Execute a read-only SQL query against the brief database."""
    result = db.execute_sql(req.query, limit=req.limit)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# ── Scan Trigger ─────────────────────────────────────────────────────────────

pipeline_lock = asyncio.Lock()


@app.post("/api/v1/scan", response_model=ScanResponse)
async def trigger_scan(req: ScanRequest):
    """Trigger an on-demand pipeline scan. Writes results to SQLite."""
    if pipeline_lock.locked():
        return ScanResponse(status="busy", message="Scan already running")

    if not req.use_sp500 and not req.tickers:
        raise HTTPException(400, "Provide tickers or set use_sp500=true")

    async def _run():
        async with pipeline_lock:
            from Data.data_model import Portfolio, ThresholdConfig
            from Pipeline.run_pipeline import FinPipeline

            threshold = ThresholdConfig(
                daily_sigma_threshold=req.daily_sigma,
                weekly_sigma_threshold=req.weekly_sigma,
            )
            portfolio = Portfolio(
                name=req.portfolio_name,
                tickers=req.tickers or [],
                use_sp500=req.use_sp500,
                threshold_config=threshold,
            )
            pipeline = FinPipeline(portfolio)
            brief = await pipeline.run(
                skip_news=req.skip_news,
                idio_sigma_threshold=req.idio_sigma,
            )
            # Store in SQLite
            if brief:
                db.store_brief(brief)

    asyncio.create_task(_run())
    source = "S&P 500" if req.use_sp500 else f"{len(req.tickers or [])} tickers"
    return ScanResponse(status="started", message=f"Scan triggered: {source}")


# ── Stats ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/stats")
async def get_stats():
    """Database row counts per table."""
    return db.stats()


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from importlib import import_module

    uvicorn = import_module("uvicorn")
    uvicorn.run("API_layer.api:app", host="0.0.0.0", port=8000, reload=True)
