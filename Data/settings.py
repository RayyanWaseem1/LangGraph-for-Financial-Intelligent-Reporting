"""
Configuration for the Intelligence System
API keys, database connections, market monitoring parameters
"""

import os
from typing import Dict, List 
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application wide configuration loaded from environment variables"""

    #LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    PRIMARY_MODEL: str = "claude-sonnet-4-5-20250929"
    FAST_MODEL: str = "claude-haiku-4-5-20251001"

    #News APIs
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    #Database
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = int(os.getenv("PG_PORT", "5432"))
    PG_USER: str = os.getenv("PG_USER", "geofin")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "")
    PG_DATABASE: str = os.getenv("PG_DATABASE", "geofintel")

    #Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    #Market Monitoring
    DEFAULT_DAILY_SIGMA: float = 2.0
    DEFAULT_WEEKLY_SIGMA: float = 2.0 
    VOLATILITY_LOOKBACK_DAYS: int = 60
    MIN_AVG_VOLUME: int = 500_000
    NEWS_LOOKBACK_HOURS: int = 72 #how far back to search for news
    MAX_NEWS_PER_TICKER: int = 20 #max articles per flagged ticker 

    class Config: 
        env_file = ".env"
        extra = "allow"

# -- S&P500 Sector Mapping --#
#Representative tickers by sector for sector-level analysis

SECTOR_TICKERS: Dict[str, List[str]] = {
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "ADBE"],
    "financials": ["JPM", "BAC", "GS", "MS", "WFC", "BRK-B", "C", "AXP", "SCHW", "BLK"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "consumer_discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
    "consumer_staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS"],
    "industrial": ["CAT", "BA", "HON", "UPS", "RTX", "DE", "LMT", "GE", "MMM", "UNP"],
    "materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC"],
    "real_estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "communications": ["GOOGL", "META", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA"],
}

#Reverse lookup: ticker -> sector 
TICKER_TO_SECTOR: Dict[str, str] = {}
for sector, tickers in SECTOR_TICKERS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

#Major market indices for context
MARKET_INDICES = ["SPY", "QQQ", "DIA", "IWM", "VIX", "TLT", "GLD", "USO"]

#Company name mapping for news search (ticker -> search terms)
TICKER_COMPANY_NAMES: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "BAC": "Bank of America",
    "JNJ": "Johnson Johnson",
    "UNH": "UnitedHealth",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "PFE": "Pfizer",
    "LLY": "Eli Lilly",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "HD": "Home Depot",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "AMD": "AMD Advanced Micro Devices",
    "AVGO": "Broadcom",
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon",
    "MCD": "McDonald",
    "KO": "Coca Cola",
    "PG": "Procter Gamble",
    "WMT": "Walmart",
    "COST": "Costco",
    # For unlisted tickers, the market monitor will use yfinance longName
}
