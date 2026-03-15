"""
SQLite Storage for GeoFinancial Intelligence System

Lightweight persistent storage for daily intelligence briefs.
Each pipeline run writes to 5 normalized tables:
    - briefs           : one row per pipeline run (summary, market snapshot, metadata)
    - flagged_moves    : one row per alert (ticker, decomposition, root cause)
    - causal_clusters  : one row per cluster (epicenter, coherence, members)
    - recommendations  : one row per recommendation (action, urgency, rationale)
    - sector_analysis  : one row per sector per brief (sector, summary)

Design decisions:
    - SQLite over Postgres: zero-config, single file, stdlib-only. Suitable for
      daily-cadence writes (~1 brief/day) and dashboard reads. Swap to Postgres
      via the existing Storage/storage.py for production multi-user deployments.
    - WAL mode: enables concurrent reads while the pipeline writes.
    - JSON columns for variable-length arrays (related_tickers, cluster member lists)
      that don't need individual indexing.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "geofin.db"

SCHEMA_SQL = """
-- ── Core Brief ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS briefs (
    id                  TEXT PRIMARY KEY,
    date                TEXT NOT NULL,
    portfolio_name      TEXT NOT NULL,
    executive_summary   TEXT,
    tickers_monitored   INTEGER,
    tickers_flagged     INTEGER,
    total_articles      INTEGER,
    generation_time_sec REAL,
    -- Market snapshot (denormalized for fast dashboard reads)
    spy_return          REAL,
    qqq_return          REAL,
    vix_level           REAL,
    vix_change          REAL,
    treasury_10y        REAL,
    -- Full JSON for any fields not in columns
    raw_json            TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

-- ── Flagged Moves / Alerts ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flagged_moves (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id                TEXT NOT NULL REFERENCES briefs(id),
    ticker                  TEXT NOT NULL,
    company_name            TEXT,
    sector                  TEXT,
    alert_level             TEXT NOT NULL,   -- critical / high / medium / low
    direction               TEXT,            -- up / down
    -- Raw move
    pct_change              REAL,
    move_in_sigma           REAL,
    price_start             REAL,
    price_end               REAL,
    -- Factor decomposition
    idiosyncratic_return    REAL,
    idiosyncratic_sigma     REAL,
    market_component        REAL,
    sector_component        REAL,
    r_squared               REAL,
    -- LLM analysis
    title                   TEXT,
    summary                 TEXT,
    root_cause              TEXT,            -- primary category
    root_cause_confidence   REAL,
    root_cause_explanation  TEXT,
    news_count              INTEGER DEFAULT 0,
    related_tickers         TEXT             -- JSON array
);

-- ── Causal Clusters ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS causal_clusters (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id            TEXT NOT NULL REFERENCES briefs(id),
    cluster_id          INTEGER NOT NULL,
    epicenter_ticker    TEXT,
    epicenter_return    REAL,
    coherence_score     REAL,
    dominant_sector     TEXT,
    size                INTEGER,
    tickers             TEXT,               -- JSON array
    is_singleton        INTEGER DEFAULT 0
);

-- ── Recommendations ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id        TEXT NOT NULL REFERENCES briefs(id),
    rank            INTEGER,                -- 1-based priority order
    action_type     TEXT,                   -- exit / hedge / investigate / monitor / opportunistic_buy
    target          TEXT,                   -- ticker(s)
    urgency         TEXT,                   -- immediate / high / medium / low
    rationale       TEXT,
    time_horizon    TEXT                    -- immediate / intraday / this_week / multi_day
);

-- ── Sector Analysis ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sector_analysis (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id    TEXT NOT NULL REFERENCES briefs(id),
    sector      TEXT NOT NULL,
    summary     TEXT
);

-- ── Indices ─────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_briefs_date         ON briefs(date DESC);
CREATE INDEX IF NOT EXISTS idx_moves_brief         ON flagged_moves(brief_id);
CREATE INDEX IF NOT EXISTS idx_moves_ticker        ON flagged_moves(ticker);
CREATE INDEX IF NOT EXISTS idx_moves_level         ON flagged_moves(alert_level);
CREATE INDEX IF NOT EXISTS idx_moves_sigma         ON flagged_moves(idiosyncratic_sigma DESC);
CREATE INDEX IF NOT EXISTS idx_clusters_brief      ON causal_clusters(brief_id);
CREATE INDEX IF NOT EXISTS idx_recs_brief          ON recommendations(brief_id);
CREATE INDEX IF NOT EXISTS idx_sector_brief        ON sector_analysis(brief_id);
"""


class BriefDatabase:
    """
    SQLite interface for storing and querying intelligence briefs.

    Usage:
        db = BriefDatabase()              # uses default geofin.db
        db.store_brief(market_brief)      # after pipeline run
        briefs = db.get_recent_briefs(5)  # dashboard query
        rows = db.execute_sql(query)      # SQL explorer
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Context manager that yields a connection with WAL mode and FK enforcement."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"SQLite database initialized at {self.db_path}")

    # ── Write: Store a Complete Brief ────────────────────────────────────────

    def store_brief(self, brief) -> str:
        """
        Decompose a MarketBrief into all 5 tables.

        Accepts either:
            - a MarketBrief Pydantic model (from pipeline)
            - a dict (from JSON file)
        Returns the brief_id.
        """
        if hasattr(brief, "model_dump"):
            data = brief.model_dump()
        elif isinstance(brief, dict):
            data = brief
        else:
            raise TypeError(f"Expected MarketBrief or dict, got {type(brief)}")

        brief_id = data.get("id", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        snap = data.get("market_snapshot") or {}

        with self._connect() as conn:
            # ── briefs table ──
            conn.execute("""
                INSERT OR REPLACE INTO briefs
                    (id, date, portfolio_name, executive_summary,
                     tickers_monitored, tickers_flagged, total_articles,
                     generation_time_sec, spy_return, qqq_return,
                     vix_level, vix_change, treasury_10y, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                brief_id,
                str(data.get("date", "")),
                data.get("portfolio_name", ""),
                data.get("executive_summary", ""),
                data.get("tickers_monitored", 0),
                data.get("tickers_flagged", 0),
                data.get("total_articles_analyzed", 0),
                data.get("generation_time_seconds", 0),
                snap.get("spy_daily_return"),
                snap.get("qqq_daily_return"),
                snap.get("vix_level"),
                snap.get("vix_change"),
                snap.get("treasury_10y"),
                json.dumps(data, default=str),
            ))

            # ── flagged_moves table ──
            for alert in data.get("alerts", []):
                move = alert.get("move", {})
                rc = alert.get("root_cause", {})
                conn.execute("""
                    INSERT INTO flagged_moves
                        (brief_id, ticker, company_name, sector, alert_level,
                         direction, pct_change, move_in_sigma, price_start, price_end,
                         idiosyncratic_return, idiosyncratic_sigma,
                         market_component, sector_component, r_squared,
                         title, summary, root_cause, root_cause_confidence,
                         root_cause_explanation, news_count, related_tickers)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    brief_id,
                    alert.get("ticker", ""),
                    alert.get("company_name", ""),
                    move.get("sector"),
                    alert.get("alert_level", "medium"),
                    move.get("direction"),
                    move.get("pct_change"),
                    move.get("move_in_sigma"),
                    move.get("price_start"),
                    move.get("price_end"),
                    move.get("idiosyncratic_return"),
                    move.get("idiosyncratic_sigma"),
                    move.get("market_component"),
                    move.get("sector_component"),
                    move.get("r_squared"),
                    alert.get("title", ""),
                    alert.get("summary", ""),
                    rc.get("primary_cause", "unknown"),
                    rc.get("confidence"),
                    rc.get("explanation", ""),
                    alert.get("news_count", 0),
                    json.dumps(rc.get("related_tickers", [])),
                ))

            # ── causal_clusters table ──
            for cluster in data.get("causal_clusters", []):
                conn.execute("""
                    INSERT INTO causal_clusters
                        (brief_id, cluster_id, epicenter_ticker, epicenter_return,
                         coherence_score, dominant_sector, size, tickers, is_singleton)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    brief_id,
                    cluster.get("cluster_id"),
                    cluster.get("epicenter_ticker"),
                    cluster.get("epicenter_idio_return"),
                    cluster.get("coherence_score"),
                    cluster.get("dominant_sector"),
                    cluster.get("size"),
                    json.dumps(cluster.get("tickers", [])),
                    1 if cluster.get("is_singleton") else 0,
                ))

            # ── recommendations table ──
            for i, rec in enumerate(data.get("top_recommendations", []), 1):
                conn.execute("""
                    INSERT INTO recommendations
                        (brief_id, rank, action_type, target, urgency, rationale, time_horizon)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    brief_id, i,
                    rec.get("action_type", ""),
                    rec.get("target", ""),
                    rec.get("urgency", ""),
                    rec.get("rationale", ""),
                    rec.get("time_horizon", ""),
                ))

            # ── sector_analysis table ──
            for sector, summary in (data.get("sector_summary", {}) or {}).items():
                conn.execute("""
                    INSERT INTO sector_analysis (brief_id, sector, summary)
                    VALUES (?,?,?)
                """, (brief_id, sector, summary))

        logger.info(
            f"Stored brief {brief_id}: {len(data.get('alerts',[]))} moves, "
            f"{len(data.get('causal_clusters',[]))} clusters, "
            f"{len(data.get('top_recommendations',[]))} recs"
        )
        return brief_id

    # ── Read: Dashboard Queries ──────────────────────────────────────────────

    def get_recent_briefs(self, limit: int = 20) -> List[Dict]:
        """Return recent briefs (without full JSON) for the archive list."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, date, portfolio_name, executive_summary,
                       tickers_monitored, tickers_flagged, total_articles,
                       generation_time_sec, spy_return, vix_level, created_at
                FROM briefs ORDER BY date DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_brief_full(self, brief_id: str) -> Optional[Dict]:
        """Return a complete brief with all related data (for dashboard view)."""
        with self._connect() as conn:
            brief_row = conn.execute(
                "SELECT * FROM briefs WHERE id = ?", (brief_id,)
            ).fetchone()
            if not brief_row:
                return None

            brief = dict(brief_row)

            brief["alerts"] = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM flagged_moves WHERE brief_id = ? ORDER BY idiosyncratic_sigma DESC",
                    (brief_id,)
                ).fetchall()
            ]

            brief["causal_clusters"] = [
                {**dict(r), "tickers": json.loads(r["tickers"] or "[]")}
                for r in conn.execute(
                    "SELECT * FROM causal_clusters WHERE brief_id = ? ORDER BY cluster_id",
                    (brief_id,)
                ).fetchall()
            ]

            brief["recommendations"] = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM recommendations WHERE brief_id = ? ORDER BY rank",
                    (brief_id,)
                ).fetchall()
            ]

            brief["sector_analysis"] = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM sector_analysis WHERE brief_id = ?",
                    (brief_id,)
                ).fetchall()
            ]

            return brief

    def get_latest_brief(self) -> Optional[Dict]:
        """Return the most recent brief with full data (for dashboard landing)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM briefs ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            return self.get_brief_full(row["id"])

    def get_moves_by_ticker(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Return all flagged moves for a specific ticker across all briefs."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT fm.*, b.date as brief_date, b.portfolio_name
                FROM flagged_moves fm
                JOIN briefs b ON fm.brief_id = b.id
                WHERE fm.ticker = ?
                ORDER BY b.date DESC LIMIT ?
            """, (ticker.upper(), limit)).fetchall()
            return [dict(r) for r in rows]

    def get_critical_alerts(self, days: int = 7) -> List[Dict]:
        """Return critical alerts from the last N days."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT fm.*, b.date as brief_date, b.portfolio_name
                FROM flagged_moves fm
                JOIN briefs b ON fm.brief_id = b.id
                WHERE fm.alert_level = 'critical'
                  AND b.date >= datetime('now', ?)
                ORDER BY fm.idiosyncratic_sigma DESC
            """, (f"-{days} days",)).fetchall()
            return [dict(r) for r in rows]

    # ── Read: SQL Explorer ───────────────────────────────────────────────────

    def execute_sql(self, query: str, limit: int = 200) -> Dict[str, Any]:
        """
        Execute a read-only SQL query from the dashboard SQL explorer.
        Only SELECT statements are allowed. A LIMIT is enforced.

        Returns: {"columns": [...], "rows": [...], "count": int}
        """
        cleaned = query.strip().rstrip(";")

        # Security: only allow SELECT
        first_word = cleaned.split()[0].lower() if cleaned else ""
        if first_word != "select":
            return {"error": "Only SELECT queries are allowed"}

        # Block dangerous patterns
        blocked = ["drop", "delete", "insert", "update", "alter", "create", "attach", "detach"]
        lower = cleaned.lower()
        for word in blocked:
            if word in lower.split():
                return {"error": f"'{word}' statements are not allowed"}

        # Enforce limit
        if "limit" not in lower:
            cleaned += f" LIMIT {limit}"

        with self._connect() as conn:
            try:
                cursor = conn.execute(cleaned)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return {"columns": columns, "rows": rows, "count": len(rows)}
            except sqlite3.Error as e:
                return {"error": str(e)}

    # ── Utility ──────────────────────────────────────────────────────────────

    def import_from_json(self, json_path: str) -> str:
        """Import a market_brief_*.json file into the database."""
        with open(json_path) as f:
            data = json.load(f)
        return self.store_brief(data)

    def stats(self) -> Dict[str, int]:
        """Return row counts for all tables."""
        with self._connect() as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ["briefs", "flagged_moves", "causal_clusters",
                              "recommendations", "sector_analysis"]
            }


# ── CLI: Import existing brief JSONs ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="GeoFinIntel SQLite Storage")
    parser.add_argument("--import-briefs", action="store_true",
                        help="Import all market_brief_*.json files into the database")
    parser.add_argument("--import-file", type=str,
                        help="Import a specific brief JSON file")
    parser.add_argument("--stats", action="store_true",
                        help="Print database statistics")
    parser.add_argument("--query", type=str,
                        help="Execute a SQL query and print results")
    parser.add_argument("--db", type=str, default=None,
                        help="Database file path (default: geofin.db)")
    args = parser.parse_args()

    db = BriefDatabase(db_path=args.db)

    if args.import_briefs:
        files = sorted(glob.glob("market_brief_*.json") + glob.glob("market_breif_*.json"))
        if not files:
            print("No market_brief_*.json files found in current directory")
        for f in files:
            try:
                bid = db.import_from_json(f)
                print(f"  Imported {f} -> {bid}")
            except Exception as e:
                print(f"  Failed {f}: {e}")

    if args.import_file:
        bid = db.import_from_json(args.import_file)
        print(f"Imported -> {bid}")

    if args.stats:
        for table, count in db.stats().items():
            print(f"  {table}: {count} rows")

    if args.query:
        result = db.execute_sql(args.query)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"  {result['count']} rows")
            for row in result["rows"][:20]:
                print(f"  {row}")