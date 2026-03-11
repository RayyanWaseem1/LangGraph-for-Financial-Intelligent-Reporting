"""
Market monitor for Financial Intelligence System
Pulls the price data from yfinance, computes volatility-adjusted thresholds,
and flags significant moves for S&P500 or a custom portfolio
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple 

import numpy as np 
import yfinance as yf 
import pandas as pd 

from Data.data_model import (
    Portfolio, ThresholdConfig, PriceMove, MarketSnapshot,
    MoveDirection, MovePeriod, AlertLevel, Sector,
)

from Data.settings import Settings, TICKER_TO_SECTOR, MARKET_INDICES

logger = logging.getLogger(__name__)

#-- S&P500 Constituents --#

def fetch_sp500_tickers() -> List[str]:
    """
    Fetch current S&P500 constituent tickers from Wikipedia
    Falls back to a cached list if fetch fails
    """

    try: 
        import requests
        #Wikipedia tends to block requests without the User-Agent header
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers = {"User-Agent": "Financial Intelligence/1.0 (Financial Research Tool)"},
            timeout = 15,
        )
        resp.raise_for_status()
        table = pd.read_html(resp.text)[0]
        tickers = table["Symbol"].str.replace(".", "-", regex = False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P500 tickers from Wikipedia")
        return tickers 
    except Exception as e:
        logger.warning(f"Failed to fetch S&p500 list: {e}. Using cached list.")
        #Fall back: top ~50 by market cap
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "MCD", "WMT", "CSCO",
            "TMO", "CRM", "ACN", "AMD", "BA", "CAT", "DIS", "GS", "NFLX",
            "LMT", "RTX", "NEE", "GE", "LOW", "SBUX", "ORCL", "BLK", "AXP",
            "NKE", "DE", "C", "PFE", "UPS",
        ]
    
class MarketMonitor:
    """
    Monitors price moves for a portfolio of tickers.
    Flags any ticker whose daily or weekly return breaches a 
    volatility-adjusted threshold (N st.dev from mean)
    """

    def __init__(self, portfolio: Optional[Portfolio] = None):
        self.settings = Settings() 

        if portfolio:
            self.portfolio = portfolio 
        else:
            #Default S&P 500 
            self.portfolio = Portfolio(name = "S&P 500", use_sp500 = True)

        self._resolve_tickers() 

    def _resolve_tickers(self):
        """ Resolve the final ticker list"""
        if self.portfolio.use_sp500:
            self.tickers = fetch_sp500_tickers()
            self.portfolio.name = "S&P 500"
        else:
            self.tickers = [t.upper().strip() for t in self.portfolio.tickers if t.strip()]

        if not self.tickers:
            raise ValueError("No tickers to monitor")
        logger.info(f"Monitoring {len(self.tickers)} tickers ({self.portfolio.name})")

    @property
    def threshold(self) -> ThresholdConfig:
        return self.portfolio.threshold_config
    
    # -- Main Detection -- #

    def detect_significant_moves(self) -> Tuple[List[PriceMove], MarketSnapshot]:
        """
        Pull the recent price data for all tickers and flag moves that breach 
        the volatility-adjusted thresholds. Returns flagged moves and market context
        """
        logger.info(f"Scanning {len(self.tickers)} tickers for significant moves ")
        flagged_moves: List[PriceMove] = []

        #Get market context first
        snapshot = self._get_market_snapshot() 

        #Proces tickers in batches (yfinance supports batch download)
        batch_size = 50
        for i in range(0, len(self.tickers), batch_size):
            batch = self.tickers[i:i + batch_size]
            batch_str = " ".join(batch)

            try:
                lookback = self.threshold.volatility_lookback_days + 10 #padding
                data: Optional[pd.DataFrame] = yf.download(
                    batch_str, 
                    period = f"{lookback}d",
                    group_by = "ticker",
                    progress = False,
                    threads = True,
                )
                if data is None or data.empty:
                    continue

                for ticker in batch:
                    try: 
                        moves = self._analyze_ticker(ticker, data, snapshot)
                        flagged_moves.extend(moves)
                    except Exception as e:
                        logger.debug(f"Skipping {ticker}: {e}")
                        continue 

            except Exception as e:
                logger.error(f"Batch download failed for {batch[:5]} ...: {e}")
                continue 

        #Sort by severity (sigma magnitude)
        flagged_moves.sort(key = lambda m: abs(m.move_in_sigma), reverse = True)

        logger.info(
            f"Scan complete: {len(flagged_moves)} signficant moves detected"
            f"out of {len(self.tickers)} tickers"
        )
        return flagged_moves, snapshot 
    
    def _analyze_ticker(
            self, ticker: str, data: pd.DataFrame, snapshot: MarketSnapshot
    ) -> List[PriceMove]:
        """ Analyze a single ticker for threshold breaches"""
        moves = []

        #Extract ticker data for multi-ticker DataFrame
        try:
            if len(self.tickers) == 1:
                #Single ticker download has different sstructure
                ticker_data = data
            else:
                ticker_data = data[ticker]
        except (KeyError, TypeError):
            return []
        
        if ticker_data is None or ticker_data.empty:
            return []
        
        close = ticker_data["Close"].dropna()
        volume = ticker_data["Volume"].dropna() 

        #Ensure Series (not DataFrame) for newer yfinance versions
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        if len(close) < self.threshold.volatility_lookback_days // 2:
            return [] #not enough history 
        
        #volume filter
        avg_vol = volume.tail(20).mean() 
        if avg_vol < self.threshold.min_avg_volume:
            return []
        
        #compute daily returns and volatility
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) < 20:
            return []
        
        hist_returns = daily_returns.tail(self.threshold.volatility_lookback_days)
        daily_sigma = hist_returns.std() 
        annualized_vol = daily_sigma * np.sqrt(252)

        if daily_sigma == 0 or np.isnan(daily_sigma):
            return [] 
        
        #Get company name 
        company_name = self._get_company_name(ticker) 
        sector = self._get_sector(ticker) 

        # -- Check daily move -- #
        if len(close) >= 2:
            daily_return = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            daily_sigma_move = abs(daily_return) / daily_sigma 

            if (daily_sigma_move >= self.threshold.daily_sigma_threshold or 
                abs(daily_return) * 100 >= self.threshold.daily_abs_threshold_pct):

                alert_level = self._sigma_to_alert_level(daily_sigma_move)
                moves.append(PriceMove(
                    ticker = ticker,
                    company_name = company_name,
                    sector = sector,
                    period = MovePeriod.DAILY,
                    direction = MoveDirection.UP if daily_return > 0 else MoveDirection.DOWN,
                    price_start = round(float(close.iloc[-2]), 2),
                    price_end = round(float(close.iloc[-1]), 2),
                    pct_change = round(float(daily_return * 100), 2),
                    historical_volatility= round(float(annualized_vol), 4),
                    daily_sigma = round(float(daily_sigma), 6),
                    move_in_sigma = round(float(daily_sigma_move), 2),
                    threshold_sigma = self.threshold.daily_sigma_threshold,
                    alert_level = alert_level,
                    volume = int(volume.iloc[-1]) if len(volume) > 0 else None,
                    avg_volume = int(avg_vol),
                    volume_ratio = round(float(volume.iloc[-1] / avg_vol), 2) if avg_vol > 0 else None,
                    period_start = close.index[-2].to_pydatetime().replace(tzinfo = timezone.utc),
                    period_end = close.index[-1].to_pydatetime().replace(tzinfo = timezone.utc),
                ))

        # -- Check weekly move --#
        if len(close) >= 6:
            #use 5 trading days ago as a weekly start
            weekly_return = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
            weekly_sigma = daily_sigma * np.sqrt(5) #scale daily σ to weekly 
            weekly_sigma_move = abs(weekly_return) / weekly_sigma if weekly_sigma > 0 else 0

            if (weekly_sigma_move >= self.threshold.weekly_sigma_threshold or 
                abs(weekly_return) * 100 >= self.threshold.weekly_abs_threshold_pct):

                alert_level = self._sigma_to_alert_level(weekly_sigma_move)
                moves.append(PriceMove(
                    ticker = ticker,
                    company_name = company_name,
                    sector = sector,
                    period = MovePeriod.WEEKLY,
                    direction = MoveDirection.UP if weekly_return > 0 else MoveDirection.DOWN,
                    price_start = round(float(close.iloc[-6]), 2),
                    price_end = round(float(close.iloc[-1]), 2),
                    pct_change = round(float(weekly_return * 100), 2),
                    historical_volatility= round(float(annualized_vol), 4),
                    daily_sigma = round(float(daily_sigma), 6),
                    move_in_sigma = round(float(weekly_sigma_move), 2),
                    threshold_sigma = self.threshold.weekly_sigma_threshold,
                    alert_level = alert_level,
                    volume = int(volume.tail(5).mean()) if len(volume) >= 5 else None, 
                    avg_volume = int(avg_vol),
                    volume_ratio = round(float(volume.tail(5).mean() / avg_vol), 2) if avg_vol > 0 else None,
                    period_start = close.index[-6].to_pydatetime().replace(tzinfo = timezone.utc),
                    period_end = close.index[-1].to_pydatetime().replace(tzinfo = timezone.utc),
                ))

        return moves 
    
    #-- Market Context --#

    def _get_market_snapshot(self) -> MarketSnapshot:
        """ Pulling current market index data for context"""
        snapshot = MarketSnapshot() 
        try:
            idx_data: Optional[pd.DataFrame] = yf.download(
                "SPY QQQ ^VIX ^TNX", period = "5d", progress = False, group_by = "ticker", threads = True,
            )
            if idx_data is None or idx_data.empty:
                return snapshot

            def _get_close(ticker: str) -> Optional[pd.Series]:
                """ Extract close prices handling various yfinance formats"""
                try:
                    if ticker in idx_data.columns.get_level_values(0):
                        return idx_data[ticker]["Close"].dropna()
                except (KeyError, TypeError):
                    pass 
                try:
                    return idx_data[(ticker, "Close")].dropna()
                except (KeyError, TypeError):
                    pass 
                return None 

            spy_close = _get_close("SPY")
            if spy_close is not None and len(spy_close) >= 2:
                snapshot.spy_daily_return = round(
                    float((spy_close.iloc[-1] - spy_close.iloc[-2]) / spy_close.iloc[-2] * 100), 2
                ) 

            qqq_close = _get_close("QQQ")
            if qqq_close is not None and len(qqq_close) >= 2:
                snapshot.qqq_daily_return = round(
                    float((qqq_close.iloc[-1] - qqq_close.iloc[-2]) / qqq_close.iloc[-2] * 100), 2
                )

            vix_close = _get_close("^VIX")
            if vix_close is not None:
                if len(vix_close) >= 1:
                    snapshot.vix_level = round(float(vix_close.iloc[-1]), 2)
                if len(vix_close) >= 2:
                    snapshot.vix_change = round(
                        float(vix_close.iloc[-1] - vix_close.iloc[-2]), 2
                    )

            tnx_close = _get_close("^TNX")
            if tnx_close is not None and len(tnx_close) >= 1:
                snapshot.treasury_10y = round(float(tnx_close.iloc[-1]), 3)

        except Exception as e:
            logger.warning(f"Market snapshot incomplete: {e}")

        return snapshot 
    
    # -- Helpers -- #

    @staticmethod
    def _sigma_to_alert_level(sigma: float) -> AlertLevel:
        """ Map move magnitude in σ to alert level"""
        if sigma >= 3.0:
            return AlertLevel.CRITICAL
        elif sigma >= 2.5:
            return AlertLevel.HIGH
        elif sigma >= 2.0:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
        
    @staticmethod
    def _get_company_name(ticker:str) -> str:
        """ Get company name, with yfinance as a fallback"""
        from Data.settings import TICKER_COMPANY_NAMES
        if ticker in TICKER_COMPANY_NAMES:
            return TICKER_COMPANY_NAMES[ticker]
        try:
            info = yf.Ticker(ticker).info
            return info.get("longName", info.get("shortName", ticker))
        except Exception:
            return ticker 
        
    @staticmethod
    def _get_sector(ticker:str) -> Optional[Sector]:
        """ Look up a sector for the ticker"""
        sector_str = TICKER_TO_SECTOR.get(ticker)
        if sector_str:
            try:
                return Sector(sector_str)
            except ValueError:
                pass
        return None 
    
# -- Convenience Functions --#

def scan_sp500(
        daily_sigma: float = 2.0, weekly_sigma: float = 2.0
) -> Tuple[List[PriceMove], MarketSnapshot]:
    """ Quick scans of S&P500 with specificied thresholds"""
    portfolio = Portfolio(
        name = "S&P 500",
        use_sp500 = True,
        threshold_config = ThresholdConfig(
            daily_sigma_threshold=daily_sigma,
            weekly_sigma_threshold = weekly_sigma,
        ),
    )
    monitor = MarketMonitor(portfolio)
    return monitor.detect_significant_moves() 

def scan_portfolio(
        tickers: List[str],
        name: str = "Custom Portfolio",
        daily_sigma: float = 2.0,
        weekly_sigma: float = 2.0,
) -> Tuple[List[PriceMove], MarketSnapshot]:
    """ Quick scan of a custom ticker list"""
    portfolio = Portfolio(
        name = name,
        tickers = tickers,
        use_sp500 = False,
        threshold_config = ThresholdConfig(
            daily_sigma_threshold = daily_sigma,
            weekly_sigma_threshold = weekly_sigma,
        ),
    )
    monitor = MarketMonitor(portfolio)
    return monitor.detect_significant_moves() 
