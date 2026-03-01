"""
Data Models for Intelligence System
Portfolio monitoring, threshold detection, news analysis, and intelligence briefing.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ── Enums ────────────────────────────────────────────────────────────────────

class MarketImpactCategory(str, Enum):
    """Root cause categories for significant price moves."""
    FED_ANNOUNCEMENT = "fed_announcement"
    RATE_DECISION = "rate_decision"
    TRADE_WAR = "trade_war"
    SANCTIONS = "sanctions"
    MILITARY_CONFLICT = "military_conflict"
    ELECTION = "election"
    BILL_SIGNING = "bill_signing"
    COMMODITY_DISRUPTION = "commodity_disruption"
    REGULATORY_CHANGE = "regulatory_change"
    EARNINGS_SURPRISE = "earnings_surprise"
    MERGER_ACQUISITION = "merger_acquisition"
    CURRENCY_CRISIS = "currency_crisis"
    SOVEREIGN_DEBT = "sovereign_debt"
    CENTRAL_BANK_POLICY = "central_bank_policy"
    PANDEMIC_HEALTH = "pandemic_health"
    TECH_DISRUPTION = "tech_disruption"
    CLIMATE_EVENT = "climate_event"
    LABOR_MARKET = "labor_market"
    INFRASTRUCTURE = "infrastructure"
    GEOPOLITICAL_TENSION = "geopolitical_tension"
    ECONOMIC_DATA = "economic_data"
    SECTOR_ROTATION = "sector_rotation"
    ANALYST_RATING = "analyst_rating"
    INSIDER_ACTIVITY = "insider_activity"
    UNKNOWN = "unknown"


class MoveDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class MovePeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class AlertLevel(str, Enum):
    CRITICAL = "critical"     # > 3σ move
    HIGH = "high"             # > 2.5σ move
    MEDIUM = "medium"         # > 2σ move
    LOW = "low"               # > 1.5σ move


class Sector(str, Enum):
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    DEFENSE = "defense"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIAL = "industrial"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    UTILITIES = "utilities"
    COMMUNICATIONS = "communications"


class EventSource(str, Enum):
    GDELT = "gdelt"
    NEWSAPI = "newsapi"
    RSS = "rss"
    MANUAL = "manual"


# ── Portfolio & Threshold Models ─────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    """Configuration for move detection thresholds."""
    # Volatility-adjusted: flag moves exceeding N standard deviations
    daily_sigma_threshold: float = Field(default=2.0, description="Daily move threshold in σ")
    weekly_sigma_threshold: float = Field(default=2.0, description="Weekly move threshold in σ")
    # Absolute fallback: flag moves exceeding fixed % regardless of vol
    daily_abs_threshold_pct: float = Field(default=5.0, description="Absolute daily % threshold")
    weekly_abs_threshold_pct: float = Field(default=10.0, description="Absolute weekly % threshold")
    # Lookback for historical volatility calculation
    volatility_lookback_days: int = Field(default=60, description="Days of history for σ calc")
    # Minimum volume filter (skip illiquid moves)
    min_avg_volume: int = Field(default=500_000, description="Min average daily volume")


class Portfolio(BaseModel):
    """User portfolio or watchlist."""
    id: str = Field(default_factory=_uid)
    name: str = Field(default="My Portfolio")
    tickers: List[str] = Field(default_factory=list)
    use_sp500: bool = Field(default=False, description="Monitor full S&P 500")
    threshold_config: ThresholdConfig = Field(default_factory=ThresholdConfig)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Price Move Detection ─────────────────────────────────────────────────────

class PriceMove(BaseModel):
    """A detected significant price move for a single ticker."""
    ticker: str
    company_name: str = ""
    sector: Optional[Sector] = None
    period: MovePeriod
    direction: MoveDirection
    # Price data
    price_start: float
    price_end: float
    pct_change: float                    # Raw percentage change
    # Volatility context
    historical_volatility: float         # Annualized σ
    daily_sigma: float                   # σ of daily returns
    move_in_sigma: float                 # How many σ this move represents
    # Threshold info
    threshold_sigma: float               # What threshold was breached
    alert_level: AlertLevel
    # Volume context
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    volume_ratio: Optional[float] = None  # volume / avg_volume
    # Metadata
    period_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketSnapshot(BaseModel):
    """Overall market context at time of detection."""
    spy_daily_return: Optional[float] = None
    qqq_daily_return: Optional[float] = None
    vix_level: Optional[float] = None
    vix_change: Optional[float] = None
    treasury_10y: Optional[float] = None
    snapshot_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── News & Event Models ──────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    """A news article retrieved for a flagged ticker."""
    id: str = Field(default_factory=_uid)
    url: str = ""
    title: str = ""
    description: str = ""
    source_name: str = ""
    source: EventSource = EventSource.GDELT
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment_score: float = 0.0         # -1 to 1
    relevance_score: float = 0.0         # 0 to 1
    matched_ticker: str = ""


class TickerNewsBundle(BaseModel):
    """All news articles retrieved for a single flagged ticker."""
    ticker: str
    company_name: str = ""
    move: PriceMove
    articles: List[NewsArticle] = Field(default_factory=list)
    article_count: int = 0


# ── Intelligence Analysis Models ─────────────────────────────────────────────

class RootCauseAnalysis(BaseModel):
    """LLM-generated root cause analysis for a price move."""
    ticker: str
    primary_cause: MarketImpactCategory = MarketImpactCategory.UNKNOWN
    secondary_causes: List[MarketImpactCategory] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    key_articles: List[str] = Field(default_factory=list)
    is_company_specific: bool = True
    related_tickers: List[str] = Field(default_factory=list)


class SectorImpact(BaseModel):
    """Impact assessment for a market sector."""
    sector: Sector
    impact_summary: str = ""
    affected_tickers: List[str] = Field(default_factory=list)
    direction: MoveDirection = MoveDirection.DOWN
    magnitude: str = "moderate"          # mild, moderate, severe


class ActionRecommendation(BaseModel):
    """Recommended action based on move analysis."""
    action_type: str = ""                # monitor, hedge, reduce_exposure, opportunistic_buy
    target: str = ""
    urgency: str = "medium"
    rationale: str = ""
    time_horizon: str = ""               # intraday, this_week, this_month


# ── Alert & Briefing Models ──────────────────────────────────────────────────

class MoveAlert(BaseModel):
    """Generated alert for a significant price move with full analysis."""
    id: str = Field(default_factory=_uid)
    ticker: str
    company_name: str = ""
    alert_level: AlertLevel
    title: str = ""
    summary: str = ""
    move: PriceMove
    root_cause: Optional[RootCauseAnalysis] = None
    sector_impacts: List[SectorImpact] = Field(default_factory=list)
    recommended_actions: List[ActionRecommendation] = Field(default_factory=list)
    related_moves: List[str] = Field(default_factory=list)
    news_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketBrief(BaseModel):
    """Complete market intelligence briefing."""
    id: str = Field(default_factory=_uid)
    date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    executive_summary: str = ""
    portfolio_name: str = ""
    tickers_monitored: int = 0
    tickers_flagged: int = 0
    market_snapshot: Optional[MarketSnapshot] = None
    alerts: List[MoveAlert] = Field(default_factory=list)
    critical_alerts: List[MoveAlert] = Field(default_factory=list)
    sector_summary: Dict[str, str] = Field(default_factory=dict)
    top_recommendations: List[ActionRecommendation] = Field(default_factory=list)
    total_articles_analyzed: int = 0
    generation_time_seconds: float = 0.0