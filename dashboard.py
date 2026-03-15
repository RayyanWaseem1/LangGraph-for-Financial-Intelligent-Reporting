"""
GeoFinancial Intelligence Dashboard
SQLite-backed Streamlit interface for portfolio monitoring and intelligence briefings.

Pages:
    1. Dashboard  - Today's brief with summary, charts, clusters, recommendations
    2. Portfolio   - Custom ticker scan with configurable thresholds
    3. Archive    - Browse all stored briefs with full detail drill-down
    4. SQL Explorer - Raw SQL queries against the brief database

Requires:
    - FastAPI running: uvicorn API_layer.api:app --reload
    - Or direct SQLite access (fallback mode when API is offline)
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = Path(__file__).resolve().parent / "geofin.db"

st.set_page_config(
    page_title="GeoFinancial Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #070b14; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: #111827; border: 1px solid #1e293b; border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #9ca3af !important; font-size: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    .stExpander { border: 1px solid #1e293b !important; border-radius: 8px !important; }
    section[data-testid="stSidebar"] { background: #0a0f1c; }
</style>
""", unsafe_allow_html=True)

LEVEL_COLORS = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}
LEVEL_ICONS = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


# ── Data Access Layer ────────────────────────────────────────────────────────

def _api_available() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _db_query(sql: str, params: tuple = ()) -> List[Dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        st.error(f"DB Error: {e}")
        return []
    finally:
        conn.close()


def _api_get(path: str, params: Optional[Dict] = None):
    try:
        import httpx
        resp = httpx.get(f"{API_URL}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _api_post(path: str, data: Dict):
    try:
        import httpx
        resp = httpx.post(f"{API_URL}{path}", json=data, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ── Data Fetchers (API with SQLite fallback) ─────────────────────────────────

def get_brief_list(limit: int = 30) -> List[Dict]:
    data = _api_get("/api/v1/briefs", {"limit": limit})
    if data:
        return data
    return _db_query(
        "SELECT id, date, portfolio_name, executive_summary, tickers_monitored, "
        "tickers_flagged, total_articles, spy_return, vix_level, generation_time_sec "
        "FROM briefs ORDER BY date DESC LIMIT ?", (limit,)
    )


def get_brief_full(brief_id: str) -> Optional[Dict]:
    data = _api_get(f"/api/v1/briefs/{brief_id}")
    if data:
        return data
    rows = _db_query("SELECT * FROM briefs WHERE id = ?", (brief_id,))
    if not rows:
        return None
    brief = rows[0]
    brief["alerts"] = _db_query(
        "SELECT * FROM flagged_moves WHERE brief_id = ? ORDER BY idiosyncratic_sigma DESC",
        (brief_id,)
    )
    clusters = _db_query(
        "SELECT * FROM causal_clusters WHERE brief_id = ? ORDER BY cluster_id", (brief_id,)
    )
    for c in clusters:
        try:
            c["tickers"] = json.loads(c.get("tickers", "[]"))
        except (json.JSONDecodeError, TypeError):
            c["tickers"] = []
    brief["causal_clusters"] = clusters
    brief["recommendations"] = _db_query(
        "SELECT * FROM recommendations WHERE brief_id = ? ORDER BY rank", (brief_id,)
    )
    brief["sector_analysis"] = _db_query(
        "SELECT * FROM sector_analysis WHERE brief_id = ?", (brief_id,)
    )
    return brief


def get_latest_brief() -> Optional[Dict]:
    data = _api_get("/api/v1/briefs/latest")
    if data:
        return data
    rows = _db_query("SELECT id FROM briefs ORDER BY date DESC LIMIT 1")
    if rows:
        return get_brief_full(rows[0]["id"])
    return None


def run_sql_query(query: str) -> Dict:
    data = _api_post("/api/v1/sql", {"query": query, "limit": 200})
    if data:
        return data
    cleaned = query.strip().rstrip(";")
    if not cleaned.lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed"}
    for blocked in ["drop", "delete", "insert", "update", "alter", "create"]:
        if blocked in cleaned.lower().split():
            return {"error": f"'{blocked}' is not allowed"}
    if "limit" not in cleaned.lower():
        cleaned += " LIMIT 200"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(cleaned)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return {"columns": cols, "rows": rows, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📡 GeoFinIntel")
    st.caption("Intelligence System v2.1")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "💼 Portfolio", "📋 Archive", "🔍 SQL Explorer"],
        label_visibility="collapsed",
    )

    st.divider()

    api_online = _api_available()
    if api_online:
        st.success("● API Connected", icon="📡")
    elif DB_PATH.exists():
        st.warning("● API Offline — SQLite Fallback", icon="💾")
    else:
        st.error("● No data source available", icon="❌")

    brief_count = _db_query("SELECT COUNT(*) as n FROM briefs")
    if brief_count:
        st.caption(f"{brief_count[0]['n']} briefs stored")


# ── PAGE: Dashboard ──────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    brief = get_latest_brief()

    if not brief:
        st.title("📊 Market Intelligence Dashboard")
        st.info(
            "No briefs in database. Run a pipeline scan or import existing JSON files:\n\n"
            "```bash\npython -m Storage.sqlite_store --import-briefs\n```"
        )
        st.stop()

    date_str = str(brief.get("date", ""))[:10]
    st.markdown(f"## 📊 Market Intelligence Brief — {date_str}")
    st.caption(
        f"{brief.get('portfolio_name', '')} · "
        f"{brief.get('generation_time_sec', brief.get('generation_time_seconds', '?'))}s pipeline"
    )

    # Market snapshot
    spy = brief.get("spy_return", brief.get("spy_daily_return"))
    vix = brief.get("vix_level")
    tnx = brief.get("treasury_10y")

    snap_cols = st.columns(5)
    with snap_cols[0]:
        if spy is not None:
            st.metric("SPY", f"{spy:+.2f}%", delta=f"{spy:.2f}%", delta_color="inverse")
    with snap_cols[1]:
        qqq = brief.get("qqq_return", brief.get("qqq_daily_return"))
        if qqq is not None:
            st.metric("QQQ", f"{qqq:+.2f}%", delta=f"{qqq:.2f}%", delta_color="inverse")
    with snap_cols[2]:
        if vix is not None:
            st.metric("VIX", f"{vix:.1f}")
    with snap_cols[3]:
        if tnx is not None:
            st.metric("10Y", f"{tnx:.3f}%")
    with snap_cols[4]:
        st.metric("Articles", brief.get("total_articles", brief.get("total_articles_analyzed", 0)))

    # Key stats
    alerts = brief.get("alerts", [])
    clusters = brief.get("causal_clusters", [])
    recs = brief.get("recommendations", brief.get("top_recommendations", []))

    stat_cols = st.columns(5)
    stat_cols[0].metric("Monitored", brief.get("tickers_monitored", 0))
    stat_cols[1].metric("Flagged", brief.get("tickers_flagged", 0))
    stat_cols[2].metric("Critical", sum(1 for a in alerts if a.get("alert_level") == "critical"))
    stat_cols[3].metric("Clusters", len(clusters))
    stat_cols[4].metric("Alerts", len(alerts))

    # Executive summary
    st.markdown("---")
    st.markdown("#### Executive Summary")
    st.markdown(brief.get("executive_summary", "No summary available."))

    # Charts
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Idiosyncratic σ by Ticker")
        if alerts:
            chart_data = []
            for a in alerts[:15]:
                sigma = a.get("idiosyncratic_sigma") or (a.get("move", {}) or {}).get("idiosyncratic_sigma") or 0
                chart_data.append({
                    "ticker": a.get("ticker", "?"),
                    "σ": abs(float(sigma)),
                    "level": a.get("alert_level", "low"),
                })
            df = pd.DataFrame(chart_data).sort_values("σ", ascending=False)
            fig = px.bar(df, x="ticker", y="σ", color="level", color_discrete_map=LEVEL_COLORS)
            fig.update_layout(
                height=320, margin=dict(t=10, b=10, l=10, r=10),
                plot_bgcolor="#111827", paper_bgcolor="#111827",
                font_color="#9ca3af", showlegend=False,
                xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        st.markdown("#### Causal Clusters")
        if clusters:
            for c in clusters:
                tickers = c.get("tickers", [])
                if isinstance(tickers, str):
                    try:
                        tickers = json.loads(tickers)
                    except (json.JSONDecodeError, TypeError):
                        tickers = []
                coherence = c.get("coherence_score", 0) or 0
                color = "#22c55e" if coherence > 0.4 else "#eab308" if coherence > 0.25 else "#ef4444"
                st.markdown(
                    f"**Cluster {c.get('cluster_id', '?')}** · "
                    f"{c.get('dominant_sector', 'mixed')} · "
                    f"<span style='color:{color}'>ρ = {coherence:.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Epicenter: **{c.get('epicenter_ticker', '?')}** · "
                    f"{c.get('size', len(tickers))} tickers · "
                    f"{', '.join(tickers[:8])}"
                    f"{'...' if len(tickers) > 8 else ''}"
                )

    # Recommendations
    if recs:
        st.markdown("---")
        st.markdown("#### Top Recommendations")
        for r in recs:
            urgency = r.get("urgency", "medium")
            icon = "🔴" if urgency == "immediate" else "🟠" if urgency == "high" else "🟡"
            st.markdown(
                f"{icon} **[{urgency.upper()}]** "
                f"**{r.get('action_type', '')}:** {r.get('target', '')}  \n"
                f"_{r.get('rationale', '')}_  \n"
                f"⏱ {r.get('time_horizon', '')}"
            )

    # Alerts
    st.markdown("---")
    st.markdown(f"#### Flagged Moves ({len(alerts)})")
    for a in alerts:
        level = a.get("alert_level", "low")
        icon = LEVEL_ICONS.get(level, "⚪")
        ticker = a.get("ticker", "?")
        pct = a.get("pct_change") or (a.get("move", {}) or {}).get("pct_change", 0) or 0
        sigma = a.get("idiosyncratic_sigma") or (a.get("move", {}) or {}).get("idiosyncratic_sigma", 0) or 0
        title = a.get("title", f"{ticker} {pct:+.1f}%")

        with st.expander(
            f"{icon} **[{level.upper()}]** {ticker}  {pct:+.1f}%  ({sigma:.1f}σ)  —  {title[:80]}",
            expanded=(level == "critical"),
        ):
            summary = a.get("summary") or a.get("root_cause_explanation", "")
            if summary:
                st.write(summary)
            mc = st.columns(4)
            idio_r = a.get("idiosyncratic_return") or (a.get("move", {}) or {}).get("idiosyncratic_return")
            mkt_c = a.get("market_component") or (a.get("move", {}) or {}).get("market_component")
            r2 = a.get("r_squared") or (a.get("move", {}) or {}).get("r_squared")
            cause = a.get("root_cause") or (a.get("move", {}) or {}).get("root_cause", "—")
            if idio_r is not None:
                mc[0].metric("Idiosyncratic", f"{idio_r:+.1f}%")
            if mkt_c is not None:
                mc[1].metric("Market β", f"{mkt_c:+.2f}%")
            if r2 is not None:
                mc[2].metric("R²", f"{r2:.3f}")
            mc[3].metric("Root Cause", str(cause))

            # Source articles with links
            articles = a.get("top_articles", [])
            if isinstance(articles, str):
                try:
                    articles = json.loads(articles)
                except (json.JSONDecodeError, TypeError):
                    articles = []
            if articles:
                st.markdown("**📰 Source Articles:**")
                for art in articles:
                    url = art.get("url", "")
                    title = art.get("title", "Untitled")
                    source = art.get("source", "")
                    sent = art.get("sentiment", 0) or 0
                    sent_icon = "🟢" if sent > 0.1 else "🔴" if sent < -0.1 else "⚪"
                    if url:
                        st.markdown(f"  {sent_icon} [{title}]({url}) — *{source}*")
                    else:
                        st.markdown(f"  {sent_icon} {title} — *{source}*")

    # Sector analysis
    sector_data = brief.get("sector_analysis", [])
    sector_dict = brief.get("sector_summary", {})
    if sector_data or sector_dict:
        st.markdown("---")
        st.markdown("#### Sector Analysis")
        if sector_data:
            for s in sector_data:
                st.markdown(f"**{s.get('sector', '?')}**")
                st.caption(s.get("summary", ""))
        elif sector_dict:
            for sector, summary in sector_dict.items():
                st.markdown(f"**{sector}**")
                st.caption(summary)


# ── PAGE: Portfolio ──────────────────────────────────────────────────────────

elif page == "💼 Portfolio":
    st.markdown("## 💼 Custom Portfolio Scan")
    st.caption("Run the full intelligence pipeline on your tickers")

    tickers_input = st.text_area(
        "Portfolio Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, JPM, GS",
        height=80,
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.caption(f"{len(tickers)} tickers")

    col1, col2, col3 = st.columns(3)
    daily_sigma = col1.slider("Daily σ threshold", 1.0, 4.0, 2.0, 0.25)
    idio_sigma = col2.slider("Idiosyncratic σ threshold", 0.5, 3.0, 1.5, 0.25)
    skip_news = col3.checkbox("Skip news retrieval")
    use_sp500 = st.checkbox("Monitor full S&P 500 instead of custom tickers")

    if st.button("🚀 Run Intelligence Pipeline", type="primary"):
        if not api_online:
            st.error("API is offline. Start it with:\n\n```bash\nuvicorn API_layer.api:app --reload\n```")
        else:
            with st.spinner("Pipeline running... (this takes 1-3 minutes for S&P 500)"):
                req = {
                    "use_sp500": use_sp500,
                    "tickers": tickers if not use_sp500 else None,
                    "portfolio_name": "S&P 500" if use_sp500 else "Custom Portfolio",
                    "daily_sigma": daily_sigma,
                    "idio_sigma": idio_sigma,
                    "skip_news": skip_news,
                }
                result = _api_post("/api/v1/scan", req)
                if result:
                    st.success(f"Scan {result['status']}: {result['message']}")
                    st.info("Brief will appear on the Dashboard and Archive pages once the pipeline completes.")

    with st.expander("Pipeline Steps", expanded=False):
        st.markdown("""
        1. **Market Monitor** → scan for Nσ moves across all tickers
        2. **Factor Decomposition** → R = α + β_mkt·SPY + β_sec·SectorETF⊥ + ε
        3. **Prediction Residual** → OLS contextual signal for LLM
        4. **News Retrieval** → GDELT + NewsAPI (72h lookback)
        5. **Causal Graph** → partial correlation network → spectral clustering
        6. **SLM Agent** → FinBERT classification + sentiment + relevance
        7. **LLM Agent** → Claude Sonnet synthesis → intelligence brief
        """)


# ── PAGE: Archive ────────────────────────────────────────────────────────────

elif page == "📋 Archive":
    st.markdown("## 📋 Brief Archive")
    briefs = get_brief_list(limit=50)

    if not briefs:
        st.info("No briefs stored. Import existing JSON files:\n\n```bash\npython -m Storage.sqlite_store --import-briefs\n```")
        st.stop()

    st.caption(f"{len(briefs)} briefs stored")

    brief_options = {
        f"{str(b.get('date',''))[:10]} — {b.get('portfolio_name','')} — "
        f"{b.get('tickers_flagged',0)} flagged": b.get("id")
        for b in briefs
    }

    selected_label = st.selectbox("Select a brief", list(brief_options.keys()))
    selected_id = brief_options.get(selected_label)

    if selected_id:
        full = get_brief_full(selected_id)
        if full:
            st.markdown(f"### {full.get('portfolio_name', '')} — {str(full.get('date',''))[:10]}")
            mc = st.columns(5)
            mc[0].metric("Monitored", full.get("tickers_monitored", 0))
            mc[1].metric("Flagged", full.get("tickers_flagged", 0))
            mc[2].metric("Articles", full.get("total_articles", full.get("total_articles_analyzed", 0)))
            mc[3].metric("Clusters", len(full.get("causal_clusters", [])))
            mc[4].metric("Pipeline", f"{full.get('generation_time_sec', full.get('generation_time_seconds', '?'))}s")

            st.markdown("#### Executive Summary")
            st.write(full.get("executive_summary", ""))

            alerts = full.get("alerts", [])
            if alerts:
                st.markdown(f"#### Flagged Moves ({len(alerts)})")
                alert_df = pd.DataFrame(alerts)
                display_cols = ["ticker", "alert_level", "pct_change", "idiosyncratic_sigma",
                                "market_component", "r_squared", "root_cause", "title"]
                available = [c for c in display_cols if c in alert_df.columns]
                if available:
                    st.dataframe(
                        alert_df[available].sort_values("idiosyncratic_sigma", ascending=False)
                        if "idiosyncratic_sigma" in available else alert_df[available],
                        use_container_width=True, hide_index=True,
                    )

            clusters = full.get("causal_clusters", [])
            if clusters:
                st.markdown("#### Causal Clusters")
                for c in clusters:
                    tickers = c.get("tickers", [])
                    if isinstance(tickers, str):
                        try:
                            tickers = json.loads(tickers)
                        except (json.JSONDecodeError, TypeError):
                            tickers = []
                    st.markdown(
                        f"**Cluster {c.get('cluster_id', '?')}** — "
                        f"Epicenter: {c.get('epicenter_ticker', '?')} — "
                        f"ρ = {(c.get('coherence_score') or 0):.2f} — "
                        f"{c.get('dominant_sector', 'mixed')} — "
                        f"{c.get('size', len(tickers))} tickers"
                    )
                    st.caption(", ".join(tickers[:12]) + ("..." if len(tickers) > 12 else ""))

            recs = full.get("recommendations", full.get("top_recommendations", []))
            if recs:
                st.markdown("#### Recommendations")
                for r in recs:
                    st.markdown(
                        f"**{r.get('rank', '')}. [{r.get('urgency', '').upper()}] "
                        f"{r.get('action_type', '')}:** {r.get('target', '')}  \n"
                        f"_{r.get('rationale', '')}_"
                    )

            sectors = full.get("sector_analysis", [])
            if sectors:
                st.markdown("#### Sector Analysis")
                for s in sectors:
                    st.markdown(f"**{s['sector']}**: {s['summary']}")

            with st.expander("Raw JSON"):
                display = {k: v for k, v in full.items() if k != "raw_json"}
                st.json(display)


# ── PAGE: SQL Explorer ───────────────────────────────────────────────────────

elif page == "🔍 SQL Explorer":
    st.markdown("## 🔍 SQL Explorer")
    st.caption("Query the SQLite database directly — read-only SELECT statements")

    presets = {
        "Top moves by σ": "SELECT ticker, pct_change, idiosyncratic_sigma, alert_level, title FROM flagged_moves ORDER BY idiosyncratic_sigma DESC LIMIT 20",
        "Critical alerts": "SELECT ticker, pct_change, idiosyncratic_sigma, title FROM flagged_moves WHERE alert_level = 'critical'",
        "All briefs": "SELECT id, date, portfolio_name, tickers_monitored, tickers_flagged, total_articles FROM briefs ORDER BY date DESC",
        "Clusters": "SELECT cluster_id, epicenter_ticker, coherence_score, dominant_sector, size FROM causal_clusters",
        "Recommendations": "SELECT rank, action_type, target, urgency, time_horizon FROM recommendations ORDER BY rank",
        "Ticker: CNC": "SELECT ticker, pct_change, idiosyncratic_sigma, market_component, r_squared, root_cause FROM flagged_moves WHERE ticker = 'CNC'",
        "Sector analysis": "SELECT sector, summary FROM sector_analysis",
        "σ > 5": "SELECT ticker, pct_change, idiosyncratic_sigma, alert_level FROM flagged_moves WHERE idiosyncratic_sigma > 5 ORDER BY idiosyncratic_sigma DESC",
        "Decomposition": "SELECT ticker, pct_change, market_component, sector_component, idiosyncratic_return, r_squared FROM flagged_moves ORDER BY ABS(idiosyncratic_return) DESC LIMIT 15",
    }

    preset_cols = st.columns(3)
    for i, (label, query) in enumerate(presets.items()):
        col = preset_cols[i % 3]
        if col.button(label, key=f"preset_{i}", use_container_width=True):
            st.session_state["sql_query"] = query

    sql_query = st.text_area(
        "SQL Query",
        value=st.session_state.get(
            "sql_query",
            "SELECT ticker, pct_change, idiosyncratic_sigma, alert_level, title "
            "FROM flagged_moves ORDER BY idiosyncratic_sigma DESC LIMIT 20"
        ),
        height=80,
        key="sql_input",
    )
    sql_query_text = sql_query or ""

    if st.button("▶ Run Query", type="primary"):
        result = run_sql_query(sql_query_text)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("rows"):
            st.success(f"{result['count']} rows returned")
            df = pd.DataFrame(result["rows"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "query_results.csv", "text/csv")
        else:
            st.info("Query returned 0 rows.")

    with st.expander("📐 Schema Reference"):
        st.markdown("""
        **briefs** — `id, date, portfolio_name, executive_summary, tickers_monitored, tickers_flagged, total_articles, generation_time_sec, spy_return, qqq_return, vix_level, vix_change, treasury_10y`

        **flagged_moves** — `brief_id, ticker, company_name, sector, alert_level, direction, pct_change, move_in_sigma, price_start, price_end, idiosyncratic_return, idiosyncratic_sigma, market_component, sector_component, r_squared, title, summary, root_cause, root_cause_confidence, root_cause_explanation, news_count, related_tickers`

        **causal_clusters** — `brief_id, cluster_id, epicenter_ticker, epicenter_return, coherence_score, dominant_sector, size, tickers, is_singleton`

        **recommendations** — `brief_id, rank, action_type, target, urgency, rationale, time_horizon`

        **sector_analysis** — `brief_id, sector, summary`

        Supports standard SQLite syntax. Only SELECT queries are allowed.
        """)
