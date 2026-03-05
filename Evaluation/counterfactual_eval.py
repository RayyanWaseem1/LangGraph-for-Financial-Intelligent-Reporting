"""
Counterfactual Filtering Rate Evaluation

The factor decomposition filters out moves that are "explained" by market/sector beta.
    1. Filtering rate - what fraction of raw moves get filtered as systematic?
    2. False negative detection - of filtered moves, did any have subsequent company-specific news that the system missed?
    3. Next-day reversion analysis - systematic moves (market beta) should partially
    revert when the market reverst. Idiosyncratic moves should not. This tests whether
    the decomposition is correctly separating the two
    4. Event contamination check - on days with known company-sepcific events (earnings, M&A),
    did the filter correctly pass those through?

If the filter has a high false negative rate, the genuine important moves never end up reaching the LLM
"""

import logging
from datetime import datetime, timedelta 
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field 

import numpy as np 
import pandas as pd
import yfinance as yf 

from Quant.factor_decomposition import FactorDecomposer, DecomposedMove

logger = logging.getLogger(__name__)

@dataclass
class FilteringRateMetrics:
    """ How aggressively the filter operates"""
    total_raw_moves: int = 0
    passed_idiosyncratic: int = 0
    filtered_systematic: int = 0
    filtering_rate_pct: float = 0.0 #% of raw moves filtered out 

    #Breakdown by alert level
    filtered_by_level: Dict[str, int] = field(default_factory=dict)
    passed_by_level: Dict[str, int] = field(default_factory=dict)

    #Breakdown by sector
    filtering_rate_by_sector: Dict[str, float] = field(default_factory=dict)

    #Distribution of idiosyncratic sigma for filtered moves
    avg_filtered_idio_sigma: float = 0.0
    max_filtered_idio_sigma: float = 0.0

@dataclass
class NextDayReversionMetrics:
    """
    Validates decomposition by checking the next-day behavior:
        - systematic moves should partially revert when market reverts
        - idiosyncratic moves should not rever (they are driven by news, nor market)
    """

    #For filtered (systematic) moves
    systematic_reversion_rate: float = 0.0 #% that reversed direction next day
    systematic_avg_nextday_corr_with_spy: float = 0.0 #should be high 

    #For passed (idiosyncratic) moves
    idiosyncratic_reversion_rate: float = 0.0 #should be lower
    idiosyncratic_avg_nextday_corr_with_spy: float = 0.0 #should be low 

    #Separation quality
    reversion_separation: float = 0.0 #systematic rate - idiosyncratic rate
    #positive = good (systematic reverts more, as expected)

    n_systematic_evaluated: int = 0
    n_idiosyncratic_evaluated: int = 0

@dataclass
class EventContaminationMetrics:
    """
    On known event days, did the filter correctly pass through company-specific moves
    """
    n_known_events_tested: int = 0
    correctly_passed: int = 0 #Event move was NOT filtered (correct)
    incorrectly_filtered: int = 0 #Event move WAS filtered (false negative)
    false_negative_rate: float = 0.0
    false_negative_tickers: List[str] = field(default_factory = list) 

@dataclass
class CounterfactualMetrics:
    """ Complete counterfactual filtering evaluation"""
    filtering_rate: FilteringRateMetrics = field(default_factory = FilteringRateMetrics)
    next_day_reversion: NextDayReversionMetrics = field(default_factory=NextDayReversionMetrics)
    event_contamination: EventContaminationMetrics = field(default_factory=EventContaminationMetrics)

    #Overall quality score (0-1, higher = better filtering)
    filter_quality_score: float = 0.0

class CounterfactualEvaluator:
    """
    Evaluates the quality of the factor decomposition filter
    """

    #Known historical events for contamination testing
    #Format: (date, ticker, event_type, expected_idiosyncratic)
    KNOWN_EVENTS = [
        ("2024-01-25", "TSLA", "earnings_miss", True),
        ("2024-01-26", "INTC", "earnings_miss", True),
        ("2024-02-02", "META", "earnings_beat", True),
        ("2024-02-02", "AMZN", "earnings_beat", True),
        ("2024-02-22", "NVDA", "earnings_blowout", True),
        ("2024-03-07", "COST", "earnings_beat", True),
        ("2024-04-19", "NFLX", "subscriber_miss", True),
        ("2024-05-23", "NVDA", "earnings_blowout", True),
        ("2024-07-24", "GOOGL", "earnings_beat", True),
        ("2024-07-24", "TSLA", "earnings_miss", True),
        ("2024-07-30", "MSFT", "cloud_slowdown", True),
        ("2024-08-05", "MULTIPLE", "yen_carry_unwind", False), # Market-wide
        ("2024-09-18", "MULTIPLE", "fed_rate_cut", False), # Market-wide
        ("2024-10-24", "BA", "strike_impact", True),
        ("2024-10-29", "GOOGL", "earnings_beat", True),
        ("2024-11-06", "MULTIPLE", "election_result", False), # Market-wide
        ("2024-12-18", "MULTIPLE", "fed_hawkish_cut", False), # Market-wide
        ("2025-01-27", "NVDA", "deepseek_shock", True),
        ("2025-01-29", "MSFT", "cloud_miss", True),
        ("2025-02-19", "PLTR", "defense_guidance", True),
    ]

    def __init__(self):
        self.decomposer = FactorDecomposer()

    def evaluate(
        self,
        significant_moves: List[DecomposedMove],
        systematic_moves: List[DecomposedMove],
        run_next_day_analysis: bool = True,
    ) -> CounterfactualMetrics:
        """
        Full counterfactual evaluation

        Params:
            - significant_moves: Moves that PASSED the filter (idiosyncratic)
            - systematic_moves: Moves that were FILTERED (systematic)
        """
        metrics = CounterfactualMetrics()

        #Filtering rate
        metrics.filtering_rate = self._compute_filtering_rate(
            significant_moves, systematic_moves,
        )

        #Next day Reversion
        if run_next_day_analysis:
            metrics.next_day_reversion = self._compute_reversion(
                significant_moves, systematic_moves,
            )

        #Event contamination
        metrics.event_contamination = self._check_event_contamination(
            significant_moves, systematic_moves,
        )

        #Overall quality score 
        metrics.filter_quality_score = self._compute_quality_score(metrics)

        return metrics 
    
    def evaluate_historical(
        self,
        tickers: List[str],
        test_dates: Optional[List[str]] = None,
        sigma_threshold: float = 2.0,
        idio_sigma_threshold: float = 1.5,
    ) -> CounterfactualMetrics:
        """
        Running the full pipeline on historical dates and evalute filtering.
        Replays history
        """
        from Data.market_monitor import MarketMonitor
        from Data.data_model import Portfolio, ThresholdConfig

        if test_dates is None:
            #Use known event dates
            test_dates = list(set(d for d, _, _, _ in self.KNOWN_EVENTS))

        all_significant = []
        all_systematic = []

        for date_str in test_dates:
            logger.info(f"Replaying {date_str}...")
            try:
                #Fetch data around this date
                date = datetime.strptime(date_str, "%Y-%m-%d")
                end = date + timedelta(days = 2)
                start = date - timedelta(days = 120)

                #Get moves for this date
                #In production you would use the exact intraday data
                data = yf.download(
                    " ".join(tickers), start = start.strftime("%Y-%m-%d"),
                    end = end.strftime("%Y-%m-%d"), group_by = "ticker",
                    progress = False,
                )

                if data is None or data.empty:
                    continue 

                #build mock PriceMoves for the target date
                from Data.data_model import PriceMove, MoveDirection, MovePeriod, AlertLevel
                moves = []

                for ticker in tickers:
                    try:
                        close = data[ticker]["Close"].dropna()
                        if len(close) < 61:
                            continue 

                        #Find the target date's return
                        target_idx = close.index.get_indexer(
                            [pd.Timestamp(date_str)], method = "nearest"
                        )[0]
                        if target_idx < 1:
                            continue 

                        ret = (close.iloc[target_idx] - close.iloc[target_idx - 1]) / close.iloc[target_idx - 1]
                        vol = close.pct_change().rolling(60).std().iloc[target_idx]

                        if vol > 0 and abs(ret) / vol >= sigma_threshold:
                            move = PriceMove(
                                ticker = ticker,
                                company_name = ticker,
                                period = MovePeriod.DAILY,
                                direction = MoveDirection.UP if ret > 0 else MoveDirection.DOWN,
                                price_start = float(close.iloc[target_idx - 1]),
                                price_end = float(close.iloc[target_idx]),
                                pct_change = round(float(ret*100), 2),
                                historical_volatility = round(float(vol), 4),
                                daily_sigma = round(float(vol), 4),
                                move_in_sigma = round(float(abs(ret) / vol), 2),
                                threshold_sigma = sigma_threshold,
                                alert_level = AlertLevel.MEDIUM,
                            )
                            moves.append(move)

                    except (KeyError, TypeError, IndexError):
                        continue 

                if not moves:
                    continue 

                #Decompose 
                significant, systematic = self.decomposer.decompose_moves(
                    moves, idiosyncratic_sigma_threshold=idio_sigma_threshold,
                )
                all_significant.extend(significant)
                all_systematic.extend(systematic)

            except Exception as e:
                logger.warning(f"Failed to replay {date_str}: {e}")

        return self.evaluate(all_significant, all_systematic, run_next_day_analysis=False)
    
    #--Filtering Rate--#
    def _compute_filtering_rate(
        self,
        significant: List[DecomposedMove],
        systematic: List[DecomposedMove],
    ) -> FilteringRateMetrics:
        total = len(significant) + len(systematic)
        if total == 0:
            return FilteringRateMetrics() 
        
        metrics = FilteringRateMetrics(
            total_raw_moves = total,
            passed_idiosyncratic = len(significant),
            filtered_systematic = len(systematic),
            filtering_rate_pct = round(len(systematic) / total * 100, 1),
        )

        #By alert level
        for m in systematic:
            level = m.alert_level.value
            metrics.filtered_by_level[level] = metrics.filtered_by_level.get(level, 0) + 1
        for m in significant:
            level = m.alert_level.value
            metrics.passed_by_level[level] = metrics.passed_by_level.get(level, 0) + 1

        #By sector
        sector_counts: Dict[str, Dict[str, int]] = {}
        for m in significant + systematic:
            s = m.sector or "unknown"
            sector_counts.setdefault(s, {"passed": 0, "filtered": 0})
        for m in significant:
            s = m.sector or "unknown"
            sector_counts[s]["passed"] += 1
        for m in systematic:
            s = m.sector or "unknown"
            sector_counts[s]["filtered"] += 1
        for s, counts in sector_counts.items():
            total_s = counts["passed"] + counts["filtered"]
            if total_s > 0:
                metrics.filtering_rate_by_sector[s] = round(
                    counts["filtered"] / total_s * 100, 1
                )

        #Filtered move sigma distribution
        if systematic:
            idio_sigmas = [m.idiosyncratic_sigma for m in systematic]
            metrics.avg_filtered_idio_sigma = round(float(np.mean(idio_sigmas)), 2)
            metrics.max_filtered_idio_sigma = round(float(max(idio_sigmas)), 2)

        return metrics 
    
    #-- Next Day Reversion --#

    def _compute_reversion(
        self,
        significant: List[DecomposedMove],
        systematic: List[DecomposedMove],
    ) -> NextDayReversionMetrics:
        """
        Check if systematic moves rever more than idiosyncratic moves
        Logic: if the decomposition is correct, systematic moves should track 
        SPY's next-day reversal, while idiosyncratic moves should not
        """
        metrics = NextDayReversionMetrics()

        all_tickers = list(set(
            [m.ticker for m in significant] + [m.ticker for m in systematic] + ["SPY"]
        ))

        try:
            data = yf.download(
                " ".join(all_tickers), period = "10d",
                group_by = "ticker", progress = False,
            )
        except Exception:
            return metrics 
        
        if data is None or data.empty:
            return metrics
        
        #Check systematic moves
        sys_reversions = []
        sys_spy_corrs = []
        for m in systematic:
            result = self._check_reversion(data, m.ticker, m.total_return)
            if result:
                sys_reversions.append(result["reversed"])
                sys_spy_corrs.append(result["spy_correlation"])

        #Check idiosyncratic moves
        idio_reversions = []
        idio_spy_corrs = []
        for m in significant:
            result = self._check_reversion(data, m.ticker, m.idiosyncratic_return)
            if result:
                idio_reversions.append(result["reversed"])
                idio_spy_corrs.append(result["spy_correlation"])

        if sys_reversions:
            metrics.systematic_reversion_rate = round(
                sum(sys_reversions) / len(sys_reversions) * 100, 1
            )
            metrics.systematic_avg_nextday_corr_with_spy = round(
                float(np.mean(sys_spy_corrs)), 3
            )
            metrics.n_systematic_evaluated = len(sys_reversions)

        if idio_reversions:
            metrics.idiosyncratic_reversion_rate = round(
                sum(idio_reversions) / len(idio_reversions) * 100, 1
            )
            metrics.idiosyncratic_avg_nextday_corr_with_spy = round(
                float(np.mean(idio_spy_corrs)), 3
            )
            metrics.n_idiosyncratic_evaluated = len(idio_reversions)

        metrics.reversion_separation = round(
            metrics.systematic_reversion_rate - metrics.idiosyncratic_reversion_rate, 1
        )
        return metrics 
    
    def _check_reversion(
        self, data: pd.DataFrame, ticker: str, move_return: float,
    ) -> Optional[Dict]:
        """ Check if a ticker's next-day return reversed the move"""
        try:
            if len(set(data.columns.get_level_values(0))) > 1:
                close = data[ticker]["Close"].dropna()
                spy_close = data["SPY"]["Close"].dropna()
            else:
                return None 
            
            if len(close) < 3:
                return None 
            
            #Most recent returns
            ret_today = (close.iloc[-2] - close.iloc[-3]) / close.iloc[-3]
            ret_next = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            spy_next = (spy_close.iloc[-1] - spy_close.iloc[-2]) / spy_close.iloc[-2]

            #Did it reverse?
            reversed_dir = (ret_today > 0 and ret_next < 0) or (ret_today < 0 and ret_next > 0)

            #Correlation with SPY next day move
            spy_corr = 1.0 if (ret_next > 0) == (spy_next > 0) else 0.0

            return {"reversed": reversed_dir, "spy_correlation": spy_corr}
        
        except (KeyError, IndexError, TypeError):
            return None 
        
    #--Event Contamination --#
    def _check_event_contamination(
        self,
        significant: List[DecomposedMove],
        systematic: List[DecomposedMove],
    ) -> EventContaminationMetrics:
        """
        Check if known company-specific event were correctly passed through 
        (not filtered as systematic)
        """
        metrics = EventContaminationMetrics()

        sig_tickers = set(m.ticker for m in significant)
        sys_tickers = set(m.ticker for m in systematic)

        for date, ticker, event_type, should_be_idiosyncratic in self.KNOWN_EVENTS:
            if ticker == "MULTIPLE":
                continue #skip market wide events for this check
            if ticker not in sig_tickers and ticker not in sys_tickers:
                continue #ticker not in this run 

            metrics.n_known_events_tested += 1

            if should_be_idiosyncratic:
                if ticker in sig_tickers:
                    metrics.correctly_passed += 1
                else:
                    metrics.incorrectly_filtered += 1
                    metrics.false_negative_tickers.append(f"{ticker} ({date}: {event_type})")

        if metrics.n_known_events_tested > 0:
            metrics.false_negative_rate = round(
                metrics.incorrectly_filtered / metrics.n_known_events_tested * 100, 1
            )
        return metrics 
    
    #--Quality Score --#
    def _compute_quality_score(self, metrics: CounterfactualMetrics) -> float:
        """
        Compute an overall filter quality score (0-1)
        Combines: filtering rate, reversion separation, false negative rate
        """
        scores = []

        #Filtering rate: 20-60% is healthy. Too low = not filtering enough. Too high = too aggressive
        fr = metrics.filtering_rate.filtering_rate_pct
        if 20 <= fr <= 60:
            scores.append(1.0)
        elif 10 <= fr < 20 or 60 < fr <= 80:
            scores.append(0.7)
        else:
            scores.append(0.3)

        #Reversion separated: positive is good (systematic reverts more)
        sep = metrics.next_day_reversion.reversion_separation
        if sep > 10:
            scores.append(1.0)
        elif sep > 0:
            scores.append(0.7)
        else:
            scores.append(0.3)

        #False negative rate: lower is better
        fnr = metrics.event_contamination.false_negative_rate
        if fnr == 0:
            scores.append(1.0)
        elif fnr < 10:
            scores.append(0.8)
        elif fnr < 25:
            scores.append(0.5)
        else:
            scores.append(0.2)

        return round(float(np.mean(scores)) if scores else 0.0, 3)
