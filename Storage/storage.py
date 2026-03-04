"""
Storage Layer for GeoFinancial Intelligence System
PostgreSQL for persistent storage, Redis for caching/pub-sub.
"""

import json
import logging
import importlib
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from Data.data_model import PriceMove, MoveAlert, MarketBrief
from Data.settings import Settings

logger = logging.getLogger(__name__)


class PostgresStore:
    """Persistent storage for moves, alerts, and briefs."""

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS flagged_moves (
        id SERIAL PRIMARY KEY,
        ticker TEXT NOT NULL,
        company_name TEXT,
        sector TEXT,
        period TEXT NOT NULL,
        direction TEXT NOT NULL,
        pct_change FLOAT,
        move_in_sigma FLOAT,
        alert_level TEXT,
        price_start FLOAT,
        price_end FLOAT,
        historical_volatility FLOAT,
        volume_ratio FLOAT,
        raw_json JSONB,
        detected_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        alert_level TEXT NOT NULL,
        title TEXT,
        summary TEXT,
        root_cause_category TEXT,
        root_cause_explanation TEXT,
        recommended_actions JSONB,
        news_count INTEGER,
        raw_json JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS briefs (
        id TEXT PRIMARY KEY,
        date TIMESTAMPTZ NOT NULL,
        portfolio_name TEXT,
        executive_summary TEXT,
        tickers_monitored INTEGER,
        tickers_flagged INTEGER,
        total_articles INTEGER,
        brief_json JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_moves_ticker ON flagged_moves(ticker);
    CREATE INDEX IF NOT EXISTS idx_moves_detected ON flagged_moves(detected_at DESC);
    CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level);
    CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_briefs_date ON briefs(date DESC);
    """

    def __init__(self):
        self.settings = Settings()
        self.pool: Optional[Any] = None

    async def connect(self):
        try:
            asyncpg = importlib.import_module("asyncpg")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL storage. Install dependencies from requirements.txt."
            ) from e

        self.pool = await asyncpg.create_pool(
            host=self.settings.PG_HOST, port=self.settings.PG_PORT,
            user=self.settings.PG_USER, password=self.settings.PG_PASSWORD,
            database=self.settings.PG_DATABASE, min_size=2, max_size=10,
        )
        await self._init_schema()
        logger.info("PostgreSQL connected")

    def _require_pool(self) -> Any:
        if self.pool is None:
            raise RuntimeError("PostgreSQL pool is not connected")
        return self.pool

    async def _init_schema(self):
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(self.SCHEMA_SQL)

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def store_move(self, move: PriceMove):
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO flagged_moves
                   (ticker, company_name, sector, period, direction, pct_change,
                    move_in_sigma, alert_level, price_start, price_end,
                    historical_volatility, volume_ratio, raw_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                move.ticker, move.company_name,
                move.sector.value if move.sector else None,
                move.period.value, move.direction.value, move.pct_change,
                move.move_in_sigma, move.alert_level.value,
                move.price_start, move.price_end,
                move.historical_volatility, move.volume_ratio,
                json.dumps(move.model_dump(), default=str),
            )

    async def store_alert(self, alert: MoveAlert):
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO alerts
                   (id, ticker, alert_level, title, summary, root_cause_category,
                    root_cause_explanation, recommended_actions, news_count, raw_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                   ON CONFLICT (id) DO NOTHING""",
                alert.id, alert.ticker, alert.alert_level.value,
                alert.title, alert.summary,
                alert.root_cause.primary_cause.value if alert.root_cause else None,
                alert.root_cause.explanation if alert.root_cause else None,
                json.dumps([a.model_dump() for a in alert.recommended_actions]),
                alert.news_count,
                json.dumps(alert.model_dump(), default=str),
            )

    async def store_brief(self, brief: MarketBrief):
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO briefs
                   (id, date, portfolio_name, executive_summary,
                    tickers_monitored, tickers_flagged, total_articles, brief_json)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                brief.id, brief.date, brief.portfolio_name,
                brief.executive_summary, brief.tickers_monitored,
                brief.tickers_flagged, brief.total_articles_analyzed,
                json.dumps(brief.model_dump(), default=str),
            )

    async def get_recent_alerts(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            rows = await conn.fetch(
                "SELECT * FROM alerts WHERE created_at > $1 ORDER BY created_at DESC LIMIT $2",
                cutoff, limit,
            )
            return [dict(r) for r in rows]

    async def get_recent_moves(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            rows = await conn.fetch(
                "SELECT * FROM flagged_moves WHERE detected_at > $1 ORDER BY detected_at DESC LIMIT $2",
                cutoff, limit,
            )
            return [dict(r) for r in rows]


class RedisCache:
    """Redis for event dedup, alert pub/sub, and caching."""

    ALERT_CHANNEL = "geofin:alerts"

    def __init__(self):
        self.settings = Settings()
        self.client: Optional[Any] = None

    async def connect(self):
        try:
            redis_asyncio = importlib.import_module("redis.asyncio")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "redis.asyncio is required for Redis cache. Install dependencies from requirements.txt."
            ) from e

        client = redis_asyncio.Redis(
            host=self.settings.REDIS_HOST, port=self.settings.REDIS_PORT,
            db=self.settings.REDIS_DB, decode_responses=True,
        )
        await client.ping()
        self.client = client
        logger.info("Redis connected")

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError("Redis client is not connected")
        return self.client

    async def close(self):
        if self.client:
            await self.client.close()

    async def publish_alert(self, alert_data: Dict):
        client = self._require_client()
        await client.publish(self.ALERT_CHANNEL, json.dumps(alert_data, default=str))

    async def subscribe_alerts(self):
        client = self._require_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self.ALERT_CHANNEL)
        return pubsub

    async def cache_snapshot(self, data: Dict, ttl: int = 300):
        client = self._require_client()
        await client.setex("market:snapshot", ttl, json.dumps(data, default=str))

    async def get_snapshot(self) -> Optional[Dict]:
        client = self._require_client()
        data = await client.get("market:snapshot")
        return json.loads(data) if data else None


class StorageManager:
    """Unified storage interface."""

    def __init__(self):
        self.postgres = PostgresStore()
        self.redis = RedisCache()

    async def connect_all(self):
        await self.postgres.connect()
        await self.redis.connect()
        logger.info("All storage backends connected")

    async def close_all(self):
        await self.postgres.close()
        await self.redis.close()
