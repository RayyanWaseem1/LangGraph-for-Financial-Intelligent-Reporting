"""
Factor Decomposition Evaluation
Evaluates the quality of the Fama-French factor model:
1. R^2 (in-sample & out-of-sample) - how much return variance is explained
2. Residual normality - Jarque-Bera test for model specification
3. Beta stability - rolling beta consistency over time
4. Calibration - do sigma-based alerts match the empirical frequencies
5. Out-of-sample decomposition accuracy on known event types
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple 
from dataclasses import dataclass, field 

import numpy as np 
import pandas as pd 
import yfinance as yf 

from Quant.factor_decomposition import FactorDecomposer, DecomposedMove, SECTOR_ETFS
from Data.settings import TICKER_TO_SECTOR

logger = logging.getLogger(__name__)

@dataclass
class FactorModelMetrics:
    """ Evaluation metrics for the factor decomposition model"""
    #R^2 statistics
    avg_r_squared: float = 0.0
    median_r_squared: float = 0.0
    r_squared_by_sector: Dict[str, float] = field(default_factory = dict)
    r_squared_distribution: List[float] = field(default_factory = list)

    #Residual normality (Jarque-Bera)
    avg_jarque_bera_stat: float = 0.0
    pct_normal_residuals: float = 0.0 #% of tickers passing the JB test at alpha = 0.05
    avg_excess_kurtosis: float = 0.0
    avg_skewness: float = 0.0 

    #Beta stability
    avg_beta_std: float = 0.0 #Avg std of rolling betas across tickers
    max_beta_range: float = 0.0 #Worst case beta instability
    beta_stability_by_sector: Dict[str, float] = field(default_factory=dict)

    #Calibration
    calibration_curve: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibration_error: float = 0.0 #Avg absolute diff: predicted vs empirical freq

    #Summary
    tickers_evaluated: int = 0
    evaluation_period_days: int = 0

class FactorModelEvaluator:
    """
    Evaluates the factor decomposition model on historical data
    """
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.decomposer = FactorDecomposer(lookback_days = 120)

    def evaluate(
        self,
        tickers: List[str],
        test_period_days: int = 60,
    ) -> FactorModelMetrics:
        """
        Full evaluation of the factor model on a set of tickers.
        Uses walk-forward: fit on [0, T-test_period], evaluate on [T-test_period, T]
        """
        logger.info(f"Evaluating factor model on {len(tickers)} tickers, "
                    f"{test_period_days}d test period")
        
        metrics = FactorModelMetrics(
            tickers_evaluated = len(tickers),
            evaluation_period_days = test_period_days,
        )

        #Fetch all data
        all_data = self._fetch_data(tickers, self.lookback_days + test_period_days + 20)
        if all_data is None:
            return metrics
        
        #Per-ticker evaluation
        r_squared_list = []
        jb_stats = []
        kurtosis_list = []
        skew_list = []
        beta_stds = []
        sector_r2: Dict[str, List[float]] = {}
        sector_beta_std: Dict[str, List[float]] = {}

        for ticker in tickers:
            if ticker not in all_data.columns or "SPY" not in all_data.columns:
                continue 

            sector = TICKER_TO_SECTOR.get(ticker, "unknown")
            result = self._evaluate_single_ticker(
                all_data, ticker, sector, test_period_days,
            )
            if result is None:
                continue 

            r_squared_list.append(result["r_squared"])
            sector_r2.setdefault(sector, []).append(result["r_squared"])

            if result.get("jb_stat") is not None:
                jb_stats.append(result["jb_stat"])
                kurtosis_list.append(result["excess_kurtosis"])
                skew_list.append(result["skewness"])

            if result.get("beta_std") is not None:
                beta_stds.append(result["beta_std"])
                sector_beta_std.setdefault(sector, []).append(result["beta_std"])

        #Aggregate metrics
        if r_squared_list:
            metrics.avg_r_squared = round(float(np.mean(r_squared_list)), 4)
            metrics.median_r_squared = round(float(np.median(r_squared_list)), 4)
            metrics.r_squared_distribution = [round(x, 4) for x in sorted(r_squared_list)]
            metrics.r_squared_by_sector = {
                s: round(float(np.mean(vals)), 4)
                for s, vals in sector_r2.items()
            }

        if jb_stats:
            metrics.avg_jarque_bera_stat = round(float(np.mean(jb_stats)), 2)
            #JB critical value at alpha = 0.05 is ~5.99 (x^2(2))
            metrics.pct_normal_residuals = round(
                sum(1 for jb in jb_stats if jb < 5.99) / len(jb_stats) * 100, 1
            )
            metrics.avg_excess_kurtosis = round(float(np.mean(kurtosis_list)), 3)
            metrics.avg_skewness = round(float(np.mean(skew_list)), 3)

        if beta_stds:
            metrics.avg_beta_std = round(float(np.mean(beta_stds)), 4)
            metrics.max_beta_range = round(float(max(beta_stds)), 4)
            metrics.beta_stability_by_sector = {
                s: round(float(np.mean(vals)), 4)
                for s, vals in sector_beta_std.items()
            }

        #Calibration
        metrics.calibration_curve, metrics.calibration_error = self._evaluate_calibration(
            all_data, tickers, test_period_days,
        )

        return metrics 
    
    def _evaluate_single_ticker(
        self, data: pd.DataFrame, ticker: str, sector: str,
        test_period_days: int,
    ) -> Optional[Dict]:
        """ Evaluate factor model for a single ticker"""
        tk_ret = data[ticker].dropna()
        spy_ret = data["SPY"].dropna()

        #Align
        aligned = pd.DataFrame({"ticker": tk_ret, "market": spy_ret}).dropna()

        sector_etf = SECTOR_ETFS.get(sector)
        if sector_etf and sector_etf in data.columns:
            aligned["sector"] = data[sector_etf]
            aligned = aligned.dropna()

        if len(aligned) < 120 + test_period_days:
            return None 
        
        #Split: train on the first portion, test on the last test_period_days
        train = aligned.iloc[:-test_period_days]
        test = aligned.iloc[-test_period_days:]

        #Fit on training data
        Y_train = train["ticker"].to_numpy(dtype = float)
        X_train = train["market"].to_numpy(dtype = float)

        has_sector = "sector" in aligned.columns
        sec_on_mkt = 0.0
        if has_sector:
            X_sec_train = train["sector"].to_numpy(dtype = float)
            cov_sec_mkt = float(np.mean(
                (X_sec_train - np.mean(X_sec_train)) * (X_train - np.mean(X_train))
            ))
            sec_on_mkt = cov_sec_mkt / (float(np.var(X_train)) + 1e-10)
            X_sec_resid_train = X_sec_train - sec_on_mkt * X_train
            X_mat_train = np.column_stack((
                np.ones(Y_train.shape[0], dtype = float), X_train, X_sec_resid_train
            ))
        else:
            X_mat_train = np.column_stack((
                np.ones(Y_train.shape[0], dtype = float), X_train
            ))

        try:
            betas, _, _, _ = np.linalg.lstsq(X_mat_train, Y_train, rcond = None)
        except np.linalg.LinAlgError:
            return None 
        
        #Evaluating on test data (out of sample R^2)
        Y_test = test["ticker"].to_numpy(dtype = float)
        X_test = test["market"].to_numpy(dtype = float)

        if has_sector:
            X_sec_test = test["sector"].to_numpy(dtype = float)
            X_sec_resid_test = X_sec_test - sec_on_mkt * X_test 
            X_mat_test = np.column_stack((
                np.ones(Y_test.shape[0], dtype = float), X_test, X_sec_resid_test
            ))
        else:
            X_mat_test = np.column_stack((
                np.ones(Y_test.shape[0], dtype = float), X_test
            ))

        Y_pred = X_mat_test @ betas 
        residuals = Y_test - Y_pred 

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
        r_squared = max(0.0, 1 - ss_res / (ss_tot + 1e-10))

        #Residual normality (Jarque-Bera)
        n = len(residuals)
        mean_r = np.mean(residuals)
        std_r = np.std(residuals, ddof = 1)
        if std_r < 1e-10:
            return {"r_squared": r_squared}
        
        skewness = float(np.mean(((residuals - mean_r) / std_r) ** 3))
        excess_kurtosis = float(np.mean(((residuals - mean_r) / std_r) ** 4) - 3)
        jb_stat = (n / 6) * (skewness ** 2 + (excess_kurtosis ** 2) / 4)

        #Beta stability (rolling 30 day betas on training data)
        rolling_betas = []
        window = 30
        for i in range(window, len(train)):
            window_slice = train.iloc[i - window: i]
            y_w = window_slice["ticker"].to_numpy(dtype = float)
            x_w = window_slice["market"].to_numpy(dtype = float)
            cov_xy = float(np.mean((y_w - np.mean(y_w)) * (x_w - np.mean(x_w))))
            var_x = float(np.var(x_w))
            if var_x > 1e-10:
                rolling_betas.append(cov_xy / var_x)

        beta_std = float(np.std(rolling_betas)) if rolling_betas else None 

        return {
            "r_squared": float(r_squared),
            "jb_stat": float(jb_stat),
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "beta_std": beta_std,
        }
    
    def _evaluate_calibration(
        self, data: pd.DataFrame, tickers: List[str],
        test_period_days: int,
    ) -> Tuple[Dict, float]:
        """
        Calibration: do sigma-based thresholds match the empirical frequencies?
        For each sigma bucket (1.5, 2.0, 2.5, 3.0), we compute:
            - the expected frequency (under normal distribution)
            - the empirical frequency (how often residuals actually exceed that sigma)
        """
        from scipy import stats as scipy_stats

        sigma_thresholds = [1.5, 2.0, 2.5, 3.0]
        #Two-tailed probabilities under normal
        expected_freq = {
            "1.5": 2 * (1 - scipy_stats.norm.cdf(1.5)),
            "2.0": 2 * (1 - scipy_stats.norm.cdf(2.0)),
            "2.5": 2 * (1 - scipy_stats.norm.cdf(2.5)),
            "3.0": 2 * (1 - scipy_stats.norm.cdf(3.0)),
        }

        #Collect all residuals across the tickers 
        all_residual_sigmas = []

        for ticker in tickers:
            if ticker not in data.columns or "SPY" not in data.columns:
                continue 

            aligned = pd.DataFrame({
                "ticker": data[ticker], "market": data["SPY"]
            }).dropna()
            if len(aligned) < 120 + test_period_days:
                continue 

            train = aligned.iloc[:-test_period_days]
            test = aligned.iloc[-test_period_days:]

            Y_train = train["ticker"].to_numpy(dtype = float)
            market_train = train["market"].to_numpy(dtype = float)
            X_train = np.column_stack((
                np.ones(Y_train.shape[0], dtype = float), market_train
            ))

            try:
                betas, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond = None)
            except np.linalg.LinAlgError:
                continue 

            train_resid = Y_train - X_train @ betas 
            sigma = np.std(train_resid)
            if sigma < 1e-10:
                continue 

            Y_test = test["ticker"].to_numpy(dtype = float)
            market_test = test["market"].to_numpy(dtype = float)
            X_test = np.column_stack((
                np.ones(Y_test.shape[0], dtype = float), market_test
            ))
            test_resid = Y_test - X_test @ betas 
            test_sigma_vals = np.abs(test_resid) / sigma
            all_residual_sigmas.extend(test_sigma_vals.tolist())

        if not all_residual_sigmas:
            return {}, 0.0
        
        residual_arr = np.array(all_residual_sigmas, dtype = float)
        n_total = len(residual_arr)

        calibration = {}
        total_error = 0.0

        for threshold in sigma_thresholds:
            key = str(threshold)
            empirical = float(np.mean(residual_arr >= threshold))
            expected = float(expected_freq[key])
            error = float(abs(empirical - expected))
            total_error += error 

            calibration[key] = {
                "threshold_sigma": threshold,
                "expected_frequency": round(expected, 5),
                "empirical_frequency": round(empirical, 5),
                "absolute_error": round(error, 5),
                "ratio": round(empirical / (expected + 1e-10), 2),
                "n_exceedances": int(np.sum(residual_arr >= threshold)),
                "n_total": n_total,
            }

        avg_error = total_error / len(sigma_thresholds)

        return calibration, round(avg_error, 5)
    
    def _fetch_data(
        self, tickers: List[str] , days: int,
    ) -> Optional[pd.DataFrame]:
        """ Fetch daily returns for all tickers + SPY + sector ETFS"""
        all_tickers = list(set(
            tickers + ["SPY"] + list(SECTOR_ETFS.values())
        ))

        try:
            data = yf.download(
                " ".join(all_tickers), period = f"{days}d",
                group_by = "ticker", progress = False, threads = True,
            )
            if data is None or data.empty:
                return None

            returns = pd.DataFrame()
            for t in all_tickers:
                try:
                    if len(all_tickers) == 1:
                        close = data["Close"].dropna()
                    else:
                        close = data[t]["Close"].dropna()
                    returns[t] = close.pct_change().dropna()
                except (KeyError, TypeError):
                    continue 

            return returns.dropna() if not returns.empty else None 
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return None 
