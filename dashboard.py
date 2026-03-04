"""
GeoFinancial Intelligence Dashboard
Portfolio monitoring, move detection, and intelligence briefing viewer.
"""

import os
import json
import httpx
import streamlit as st
import plotly.express as px
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial Intelligence", page_icon="🌐", layout="wide")


def api_get(endpoint, params=None):
    try:
        resp = httpx.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint, json_data=None):
    try:
        resp = httpx.post(f"{API_URL}{endpoint}", json=json_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌐 GeoFinIntel")
    st.caption("Portfolio Intelligence System")
    page = st.radio("Navigation", ["📊 Dashboard", "🚨 Alerts", "📋 Briefs", "🔍 Scan"])
    st.divider()
    health = api_get("/health")
    if health:
        st.success(f"System: {health['status']}")
    else:
        st.error("System: Offline")


# ── Dashboard ────────────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    st.title("📊 Market Intelligence Dashboard")

    alerts_data = api_get("/api/v1/alerts", {"hours": 24, "limit": 200})
    moves_data = api_get("/api/v1/moves", {"hours": 24, "limit": 500})

    alerts = alerts_data.get("alerts", []) if alerts_data else []
    moves = moves_data.get("moves", []) if moves_data else []

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔔 Alerts (24h)", len(alerts))
    col2.metric("📈 Moves Flagged", len(moves))
    col3.metric("🚨 Critical", sum(1 for a in alerts if a.get("alert_level") == "critical"))
    col4.metric("🟠 High", sum(1 for a in alerts if a.get("alert_level") == "high"))

    if moves:
        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Move Magnitude (σ)")
            df = pd.DataFrame(moves)
            if "move_in_sigma" in df.columns and "ticker" in df.columns:
                top = df.nlargest(15, "move_in_sigma")
                fig = px.bar(top, x="ticker", y="move_in_sigma", color="direction",
                             color_discrete_map={"up": "#4CAF50", "down": "#FF4444"})
                fig.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Sector Breakdown")
            if "sector" in df.columns:
                sector_counts = df["sector"].dropna().value_counts()
                fig = px.pie(values=sector_counts.values, names=sector_counts.index)
                fig.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Flagged Moves")
        display_cols = ["ticker", "company_name", "pct_change", "move_in_sigma", "alert_level", "direction", "period"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].head(25), use_container_width=True, hide_index=True)


# ── Alerts ───────────────────────────────────────────────────────────────────

elif page == "🚨 Alerts":
    st.title("🚨 Alert Feed")

    col1, col2 = st.columns(2)
    hours = col1.slider("Lookback (hours)", 1, 168, 24)
    level = col2.selectbox("Level", ["All", "critical", "high", "medium", "low"])

    params: dict[str, int | str] = {"hours": hours, "limit": 100}
    if level != "All":
        params["level"] = level

    data = api_get("/api/v1/alerts", params)
    alerts = data.get("alerts", []) if data else []

    for alert in alerts:
        lvl = alert.get("alert_level", "info")
        icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        with st.expander(f"{icons.get(lvl, '⚪')} [{lvl.upper()}] {alert.get('title', alert['ticker'])}", expanded=(lvl == "critical")):
            st.write(alert.get("summary", ""))
            if alert.get("root_cause_category"):
                st.write(f"**Root cause:** {alert['root_cause_category']}")
            if alert.get("root_cause_explanation"):
                st.write(alert["root_cause_explanation"])


# ── Briefs ───────────────────────────────────────────────────────────────────

elif page == "📋 Briefs":
    st.title("📋 Intelligence Briefs")
    briefs = api_get("/api/v1/briefs", {"limit": 20})
    if briefs:
        for b in briefs:
            with st.expander(
                f"📅 {str(b.get('date', ''))[:10]} — {b.get('portfolio_name', '')} | "
                f"Flagged: {b.get('tickers_flagged', 0)}",
                expanded=(briefs.index(b) == 0),
            ):
                st.write(b.get("executive_summary", ""))
    else:
        st.info("No briefs yet. Run a scan to generate one.")


# ── Scan ─────────────────────────────────────────────────────────────────────

elif page == "🔍 Scan":
    st.title("🔍 Run Portfolio Scan")

    mode = st.radio("Portfolio", ["S&P 500", "Custom Tickers"])

    tickers = None
    if mode == "Custom Tickers":
        tickers_input = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, NVDA, TSLA, AMZN")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        st.write(f"Tickers: {', '.join(tickers)}")

    col1, col2 = st.columns(2)
    daily_sigma = col1.slider("Daily σ threshold", 1.0, 4.0, 2.0, 0.25)
    weekly_sigma = col2.slider("Weekly σ threshold", 1.0, 4.0, 2.0, 0.25)
    skip_news = st.checkbox("Skip news retrieval (moves only)")

    if st.button("🚀 Run Scan", type="primary"):
        req = {
            "use_sp500": mode == "S&P 500",
            "tickers": tickers,
            "portfolio_name": mode,
            "daily_sigma": daily_sigma,
            "weekly_sigma": weekly_sigma,
            "skip_news": skip_news,
        }
        result = api_post("/api/v1/scan", req)
        if result:
            st.success(f"Scan {result['status']}: {result['message']}")
        else:
            st.error("Failed to trigger scan")