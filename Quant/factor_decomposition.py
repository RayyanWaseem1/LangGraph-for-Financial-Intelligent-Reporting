"""
Factor Decomposition for Financial Intelligence System 

Decomposes each flagged price move into
1. Market (systematic) component - attributable to broad market beta 
2. Sector component - attributable to sector-level dynamics 
3. Idiosyncratic (residual) component - the unexplained part 

Only the idiosyncratic residual gets sent to the LLM for explanation. 
This prevents the system from "explaining" market-wide selloffs as company-specific events

Methodology:
    - Fama-French 3-factor regression (Mkt-RF, SMB, HML) over a rolling window
    - Sector ETF beta for sector component isolation 
    - Residual = actual return - predicted return from factor model 
    - Fama-French factors fetched from Kenneth french's data library 

Example: 
    NVDA drops 4.8% on a day SPY drops 2.0. 
    Factor model predicts NVDA should be down ~2.9% (Beta_mkt = 1.45 * SPY return)
    Idiosyncratic residual: -1.9% (This is what the LLM should explain)
"""

import logging 
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple 
from dataclasses import dataclass 

import numpy as np 
import pandas as pd 
import yfinance as yf 
from io import StringIO

from Data.data_model import PriceMove, MoveDirection, MovePeriod, AlertLevel, Sector
from Data.settings import Settings, TICKER_TO_SECTOR

logger = logging.getLogger(__name__)

#-- Sector ETF Mapping --#

SECTOR_ETFS = {
    "technology": "XLK",
    "financials": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrial": "XLI",
    "materials": "XLB",
    "real_estate": "XLRE",
    "utilities": "XLU",
    "communications": "XLC",
    "defense": "XLI", 
}

#-- Decomposed Move --#

@dataclass
class DecomposedMove:
    """ A price move decomposed into factor components"""
    ticker: str
    company_name: str 
    sector: Optional[str]
    period: str 

    #Raw move
    total_return: float #Actual % return 
    move_in_sigma: float #σ magnitude of total move 

    #Factor decomposition
    market_component: float #Return attributable to market (β_mkt × R_mkt)
    sector_component: float #Return attributable to sector (β_sec × R_sec_resid)
    idiosyncratic_return: float #Unexplained residual (what LLM explains)
    idiosyncratic_sigma: float #Residual in units of idiosyncratic σ

    #Factor model diagnostics
    market_beta: float #Estimated beta to market
    sector_beta: float #Estimated beta to sector (after market adjustment)
    r_squared: float #Factor model R^2 (how much is explained)
    factor_model_prediction: float #What the model predicted the return should be 

    #Context
    spy_return: float #SPY return on the same day/week 
    sector_etf_return: float #sector ETF return 
    alert_level: AlertLevel #Based on idiosyncratic σ, NOT the total σ

    #Original PriceMove reference
    original_move: Optional[PriceMove] = None 

    @property
    def pct_explained(self) -> float: 
        """ What % of the total move is explained by factors"""
        if abs(self.total_return) < 1e-8:
            return 0.0
        explained = abs(self.market_component + self.sector_component)
        return min(100.0, (explained / abs(self.total_return)) * 100)
    
    @property
    def is_mostly_systematic(self) -> bool:
        """ True if >70% of the move is explained by market + sector factors"""
        return self.pct_explained > 70.0
    
#-- Factor Model --$

class FactorDecomposer:
    """
    Decomposes price moves using factor regression 

    For each ticker, it estimates:
        R_i = α + β_mkt × R_mkt + β_sector × R_sector_resid + ε

    where R_sector_resid is the sector ETF return that is orthogonalized against
    the market (to avoid any double-counting)

    The idiosyncratic return ε is what the LLM should hope to explain
    """

    def __init__(self, lookback_days: int = 120):
        self.lookback_days = lookback_days 
        self._market_data_cache: Dict[str, pd.Series] = {}
        self._sector_data_cache: Dict[str, pd.Series] = {}
        self._ff_factors: Optional[pd.DataFrame] = None 

    def decompose_moves(
        self,
        moves: List[PriceMove],
        idiosyncratic_sigma_threshold: float = 1.5,
    ) -> Tuple[List[DecomposedMove], List[DecomposedMove]]:
        """ 
        Decompose all flagged moves into factor components

        Returns:
            - significant: Moves with large idiosyncratic residuals (worth explaining with LLM)
            - systematic: Moves mostly explained by market/sector factors (doesn't need LLM)
        """

        logger.info(f"Decomposing {len(moves)} moves into factor components")

        #Prefetch market and sector data
        self._prefetch_data(moves)

        significant = []
        systematic = []

        for move in moves:
            try:
                decomposed = self._decompose_single(move)
                if decomposed is None:
                    #Couldn't decompose - treat as significant by default 
                    significant.append(self._passthrough(move))
                    continue 

                if abs(decomposed.idiosyncratic_sigma) >= idiosyncratic_sigma_threshold:
                    significant.append(decomposed)
                else:
                    systematic.append(decomposed)

            except Exception as e:
                logger.debug(f"Decomposition failed for {move.ticker}: {e}")
                significant.append(self._passthrough(move))

        logger.info(
            f"Decomposition complete: {len(significant)} significant idiosyncratic moves, "
            f"{len(systematic)} mostly systematic (filtered out)"
        )
        return significant, systematic 
    
    def _decompose_single(self, move: PriceMove) -> Optional[DecomposedMove]:
        """ Decompose a single move using factor regression """
        ticker = move.ticker
        sector = move.sector.value if move.sector else TICKER_TO_SECTOR.get(ticker, None)

        #Get historical returns for regression
        ticker_returns = self._get_returns(ticker)
        market_returns = self._get_returns("SPY")

        if ticker_returns is None or market_returns is None:
            return None 
        
        #Align dates
        aligned = pd.DataFrame({
            "ticker": ticker_returns,
            "market": market_returns,
        }).dropna() 

        #Add sector ETF if available
        sector_etf = SECTOR_ETFS.get(sector)
        has_sector = False 
        if sector_etf:
            sector_returns = self._get_returns(sector_etf)
            if sector_returns is not None:
                aligned["sector"] = sector_returns
                aligned = aligned.dropna() 
                has_sector = True 

        if len(aligned) < 30:
            return None 
        
        #Use trailing lookback window 
        aligned = aligned.tail(self.lookback_days)

        #Regression: R_ticker = α + β_mkt × R_mkt + β_sec × R_sec_resid + ε
        Y = aligned["ticker"].values 
        X_mkt = aligned["market"].values 

        if has_sector:
            #orthogonalize sector returns against the market
            #R_sec_resid = R_sec - β(R_sec, R_mkt) × R_mkt
            X_sec = aligned["sector"].values
            sec_on_mkt_beta = np.cov(X_sec, X_mkt)[0, 1] / (np.var(X_mkt) + 1e-10)
            X_sec_resid = X_sec - sec_on_mkt_beta * X_mkt 

            #Build design matrix: [1, R_mkt, R_sec_resid]
            X = np.column_stack([np.ones(len(Y)), X_mkt, X_sec_resid])
        else: 
            X = np.column_stack([np.ones(len(Y)), X_mkt])

        #OLS regression
        try:
            betas, residuals, _, _ = np.linalg.lstsq(X, Y, rcond = None)
        except np.linalg.LinAlgError:
            return None 
        
        alpha = betas[0]
        beta_mkt = betas[1]
        beta_sec = betas[2] if has_sector else 0.0

        #Fitted values and residuals
        Y_hat = X @ betas 
        resid = Y - Y_hat 

        #R^2
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        r_squared = max(0.0, min(1.0, r_squared))

        #Idiosyncratic volatility (st. dev of residuals)
        idio_sigma = np.std(resid) 
        if idio_sigma < 1e-10:
            idio_sigma = 1e-10

        #Decompose today's move 
        total_return = move.pct_change / 100.0 #converting to decimal
        spy_return = market_returns.iloc[-1] if len(market_returns) > 0 else 0.0

        market_component = beta_mkt * spy_return 
        sector_component = 0.0
        sector_etf_return = 0.0 

        if has_sector: 
            sector_etf_return = aligned["sector"].iloc[-1]
            sector_resid = sector_etf_return - sec_on_mkt_beta * spy_return
            sector_component = beta_sec * sector_resid 

        predicted = alpha + market_component + sector_component 
        idiosyncratic = total_return - predicted 

        #Idiosyncratic sigma (how unusual is this residual?)
        idio_sigma_move = abs(idiosyncratic) / idio_sigma 

        #Alert level based on IDIOSYNCRATIC sigma, not total 
        if idio_sigma_move >= 3.0:
            alert_level = AlertLevel.CRITICAL
        elif idio_sigma_move >= 2.5:
            alert_level = AlertLevel.HIGH
        elif idio_sigma_move >= 2.0:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW 

        return DecomposedMove(
            ticker = ticker,
            company_name = move.company_name,
            sector = sector,
            period = move.period.value,
            total_return = round(move.pct_change, 2),
            move_in_sigma = move.move_in_sigma,
            market_component = round(market_component * 100, 2),
            sector_component = round(sector_component * 100, 2),
            idiosyncratic_return = round(idiosyncratic * 100, 2),
            idiosyncratic_sigma= round(idio_sigma_move, 2),
            market_beta = round(beta_mkt, 3),
            sector_beta = round(beta_sec, 3),
            r_squared = round(r_squared, 3),
            factor_model_prediction=round(predicted * 100, 2),
            spy_return = round(spy_return * 100, 2),
            sector_etf_return = round(sector_etf_return * 100, 2),
            alert_level = alert_level,
            original_move = move,
        )
    
    def _prefetch_data(self, moves: List[PriceMove]):
        """ Prefetching market and sector ETF data in bulk"""
        #Always need to use SPY
        tickers_needed = {"SPY"}

        #Collecting unique sector ETFs
        for move in moves:
            sector = move.sector.value if move.sector else TICKER_TO_SECTOR.get(move.ticker)
            if sector and sector in SECTOR_ETFS:
                tickers_needed.add(SECTOR_ETFS[sector])

        #Also need all move tickers 
        for move in moves:
            tickers_needed.add(move.ticker)

        tickers_str = " ".join(tickers_needed)
        lookback = self.lookback_days + 20 # padding

        try:
            data = yf.download(
                tickers_str, period = f"{lookback}d",
                group_by = "ticker", progress = False, threads = True,
            )

            for ticker in tickers_needed:
                try:
                    if len(tickers_needed) == 1:
                        close = data["Close"].dropna() 
                    else:
                        close = data[ticker]["Close"].dropna()
                    returns = close.pct_change().dropna()
                    self._market_data_cache[ticker] = returns 
                except (KeyError, TypeError):
                    continue 

        except Exception as e:
            logger.warning(f"Prefetch failed: {e}")

    def _get_returns(self, ticker: str) -> Optional[pd.Series]:
        """ Get daily return series for a ticker (from cache or fetch)"""
        if ticker in self._market_data_cache:
            return self._market_data_cache[ticker]
        
        try:
            data = yf.download(ticker, period = f"{self.lookback_days + 20}d", progress = False)
            if data.empty:
                return None 
            returns = data["Close"].pct_change().dropna()
            self._market_data_cache[ticker] = returns 
            return returns 
        except Exception:
            return None 
        
    def _passthrough(self, move: PriceMove) -> DecomposedMove:
        """ Create a DecomposedMove when factor decomposition fails
            Treats the entire move as idiosyncratic (conservative)
        """
        return DecomposedMove(
            ticker = move.ticker,
            company_name = move.company_name,
            sector = move.sector.value if move.sector else None,
            period = move.period.value,
            total_return = move.pct_change,
            move_in_sigma = move.move_in_sigma,
            market_component = 0.0,
            sector_component = 0.0,
            idiosyncratic_return= move.pct_change,
            idiosyncratic_sigma= move.move_in_sigma,
            market_beta = 1.0,
            sector_beta = 0.0,
            r_squared = 0.0,
            factor_model_prediction=0.0,
            spy_return = 0.0,
            sector_etf_return= 0.0,
            alert_level = move.alert_level,
            original_move=move,
        )
    
#-- Prediction Residual Model --#

class ReturnPredictor:
    """
    Simple return prediction model that estimates expected daily returns
    The system only flags and explains moves that THIS MODEL CANNOT PREDICT.

    Uses: Sector momentum + volatility regime + market trend features
    This is not mean't to be a great predictor. It is meant to capture the
    "obvious" components so that the LLM only explains what is genuinely surprising

    Features:
        - 5-day sector ETF momentum
        - 20-day sector ETF momentum
        - VIX level (volatility regime)
        - 5-day market (SPY) momentum
        - 20-day realized volatility
        - Day of week effects
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.models: Dict[str, Dict] = {} #ticker -> {betas, sigma_resid}

    def fit_and_predict(
            self, moves: List[PriceMove],
    ) -> Dict[str, Dict[str, float]]:
        """
        For each flagged ticker, this fits a prediction model on historical data 
        and computes the prediction residual for the current move. 

        Returns:
            - {ticker: {predicted, actual, residual, residual_sigma}}
        """

        results = {}

        #Fetch VIX for regime conditioning
        try:
            vix_data = yf.download("^VIX", period = f"{self.lookback_days + 10}d", progress = False)
            vix_series = vix_data["Close"].dropna() if not vix_data.empty else None
        except Exception:
            vix_series = None 

        for move in moves:
            try:
                result = self._fit_predict_single(move, vix_series)
                if result:
                    results[move.ticker] = result
            except Exception as e:
                logger.debug(f"Prediction failed for {move.ticker}: {e}")

        return results 
    
    def _fit_predict_single(
            self, move: PriceMove, vix_series: Optional[pd.Series],
    ) -> Optional[Dict[str, float]]:
        """ Fit the model and compute prediction residual for one ticker"""
        ticker = move.ticker
        sector = move.sector.value if move.sector else TICKER_TO_SECTOR.get(ticker)
        sector_etf = SECTOR_ETFS.get(sector, "SPY")

        #Donwloading historical data
        tickers_needed = f"{ticker} SPY {sector_etf}"
        data = yf.download(
            tickers_needed, period = f"{self.lookback_days + 10}d",
            group_by = "ticker", progress = False,
        )

        if data.empty:
            return None 
        
        #Extract returns
        try:
            if ticker == "SPY" and sector_etf == "SPY":
                #Edge case: ticker is SPY itself
                tk_close = data["SPY"]["Close"].dropna()
            else:
                tk_close = data[ticker]["Close"].dropna()
            spy_close = data["SPY"]["Close"].dropna()
            sec_close = data[sector_etf]["Close"].dropna()
        except (KeyError, TypeError):
            return None
    
        tk_ret = tk_close.pct_change().dropna()
        spy_ret = spy_close.pct_change().dropna()
        sec_ret = sec_close.pct_chang().dropna()

        #Building feature matrix 
        features = pd.DataFrame(index = tk_ret.index)
        features["target"] = tk_ret 
        features["spy_ret"] = spy_ret
        features["sec_mom_5d"] = sec_ret.rolling(5).mean()
        features["sec_mom_20d"] = sec_ret.rolling(20).mean() 
        features["spy_mom_5d"] = spy_ret.rolling(5).mean()
        features["rvol_20d"] = tk_ret.rolling(20).std()
        features["dow"] = features.index.dayofweek / 4.0 #normalized 

        if vix_series is not None:
            features["vix"] = vix_series.reindex(features.index).ffill() / 40.0 #normalized

        features = features.dropna()
        if len(features) < 60:
            return None 
        
        #Train/predict split: Ue all but the last row to fit, predict the last row
        train = features.iloc[:-1]
        test = features.iloc[-1:]

        Y_train = train["target"].values
        X_cols = [c for c in features.columns if c != "target"]
        X_train = train[X_cols].values
        X_test = test[X_cols].values 

        #Add intercept 
        X_train = np.column_stack([np.ones(len(X_train)), X_train])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])

        #OLS
        try:
            betas, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond = None)
        except np.linalg.LinAlgError:
            return None 
        
        #Residual statistics from training period 
        train_pred = X_train @ betas
        train_resid = Y_train - train_pred 
        sigma_resid = np.std(train_resid)
        if sigma_resid < 1e-10:
            sigma_resid = 1e-10

        #Predicting today 
        predicted = float(X_test @ betas)
        actual = move.pct_change / 100.0
        residual = actual - predicted 
        residual_sigma = abs(residual) / sigma_resid 

        return {
            "predicted_return_pct": round(predicted * 100, 2),
            "actual_return_pct": move.pct_change,
            "residual_pct": round(residual * 100, 2),
            "residual_sigma": round(residual_sigma, 2),
            "model_sigma_resid": round(sigma_resid * 100, 4),
        }
