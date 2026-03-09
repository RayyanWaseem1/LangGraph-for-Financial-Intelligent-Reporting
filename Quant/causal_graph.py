"""
Cross Move Causal Graph for Financial Intelligence System 

When multiple tickers get flagged simultaneously, the question that is to be asked is
"what is the causal structure" not "why did each move"

This causal_graph module:
1. Builds a correlation network of flagged moves using rolling returns correlations
2. Computes partial correlations (controlling for market factor) to identify
direct relationships vs. spurious market_driven co-movement
3. Clusters flagged moves into causal groups (moves driven by the same event)
4. Identifies "epicenter" tickers - the ones that moved first or most severely
5. Presents the LLM with entire CLUSTERS rather than individual isolated moves

Example:
- 8 tickers flagged. The causal_graph reveals:
    - Cluster A: {NVDA, AMD, AVGO, QCOM} - semiconductor export restriction
    - Cluster B: {JPM, GS, MS} - Fed minutes release
    - Singleton: {TSLA} - idiosyncratic (Musk tweet)

    The LLM will ingest 3 analysis tasks instead of 8, and can reason about the contagion
    within each cluster 

Methods:
- Pearson correlation on 60-day rolling returns 
- Partial correlation (market-residualized) to remove any spurious links
- Spectral clustering on the partial correlation adjacency matrix 
- Granger-style temporal lead-lag detection within clusters
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set 
from dataclasses import dataclass, field 

import numpy as np 
import pandas as pd 
import yfinance as yf 
from collections import defaultdict 

from Data.data_model import PriceMove, AlertLevel 
from Quant.factor_decomposition import DecomposedMove

logger = logging.getLogger(__name__)

def _extract_close_series(
    data: pd.DataFrame, ticker: str, all_tickers: List[str]
) -> Optional[pd.Series]:
    """Extract close price series across yfinance column layouts."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if len(all_tickers) == 1:
                flat = (
                    data.droplevel("Ticker", axis=1)
                    if "Ticker" in data.columns.names
                    else data.droplevel(0, axis=1)
                )
                result = flat["Close"].dropna()
            else:
                try:
                    result = data[ticker]["Close"].dropna()
                except KeyError:
                    result = data["Close"][ticker].dropna()
        else:
            result = data["Close"].dropna()

        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]

        return result if len(result) > 0 else None
    except (KeyError, TypeError, ValueError):
        return None

def self_extract_close_series(
    data: pd.DataFrame, ticker: str, all_tickers: List[str]
) -> Optional[pd.Series]:
    """Backward-compatible wrapper for older/local callsites."""
    return _extract_close_series(data, ticker, all_tickers)

#-- Data Structure --#

@dataclass 
class CausalEdge:
    """Directed edge within the causal graph """
    source: str #ticker that leads 
    target: str #ticker that follows 
    correlation: float #raw return correlation
    partial_correlation: float #market-residualized correlation
    lead_lag_days: int = 0 #positive = source leads the target 
    edge_strength: float = 0.0 #combined signal strength 

@dataclass
class MoveCluster:
    """ A cluster of flagged moves sharing a common driver"""
    cluster_id: int 
    tickers: List[str]
    #Epicenter: The ticker with the largest idiosyncratic move in the cluster
    epicenter_ticker: str 
    epicenter_idio_return: float 
    #Shared characteristics
    dominant_sector: Optional[str] = None 
    avg_correlation: float = 0.0
    avg_partial_correlation: float = 0.0
    #Decomposed moves within this cluster
    moves: List[DecomposedMove] = field(default_factory = list)
    #Edges within this cluster 
    internal_edges: List[CausalEdge] = field(default_factory = list)
    #Cluster level summary stats
    total_return_range: Tuple[float, float] = (0.0, 0.0)
    coherence_score: float = 0.0 #How tightly correlated is the cluster 

    @property
    def size(self) -> int:
        return len(self.tickers)
    
    @property
    def is_singleton(self) -> bool:
        return self.size == 1
    
    @property 
    def is_sector_cluster(self) -> bool:
        """ True if all tickers share the same sector"""
        sectors = set(m.sector for m in self.moves if m.sector)
        return len(sectors) == 1
    
    @property
    def max_severity(self) -> AlertLevel:
        levels = [m.alert_level for m in self.moves]
        priority = {AlertLevel.CRITICAL: 0, AlertLevel.HIGH: 1,
                    AlertLevel.MEDIUM: 2, AlertLevel.LOW: 3}
        return min(levels, key = lambda l: priority.get(l, 4)) if levels else AlertLevel.LOW
    
@dataclass 
class CausalGraph:
    """ Complete causal graph output"""
    clusters: List[MoveCluster]
    edges: List[CausalEdge]
    correlation_matrix: Optional[np.ndarray] = None 
    partial_correlation_matrix: Optional[np.ndarray] = None 
    tickers: List[str] = field(default_factory=list)

    #Summary
    num_clusters: int = 0
    num_singletons: int = 0
    num_multi_move_clusters: int = 0

#-- Causal Graph Builder --#

class CausalGraphBuilder:
    """
    Builds a causal graph from the flagged moves

    Steps:
        1. Compute the pairwise return correlations (60-day rolling)
        2. Compute the partial correlations (residualized against SPY)
        3. Build adjacency matrix from significant partial correlations
        4. Cluster using spectral methods
        5. Identify epicenter within each of the clusters 
        6. Detect lead-lag relationships
    """

    def __init__(
        self,
        lookback_days: int = 60,
        correlation_threshold: float = 0.4,
        partial_corr_threshold: float = 0.3,
    ):
        self.lookback_days = lookback_days
        self.corr_threshold = correlation_threshold
        self.partial_corr_threshold = partial_corr_threshold

    def build_graph(
        self,
        decomposed_moves: List[DecomposedMove],
    ) -> CausalGraph:
        """ Building the full causal graph from decomposed moves"""
        if len(decomposed_moves) <= 1:
            #Single move - trivial graph
            cluster = MoveCluster(
                cluster_id = 0,
                tickers = [decomposed_moves[0].ticker] if decomposed_moves else [],
                epicenter_ticker = decomposed_moves[0].ticker if decomposed_moves else "",
                epicenter_idio_return= decomposed_moves[0].idiosyncratic_return if decomposed_moves else 0.0,
                moves = decomposed_moves,
            )
            return CausalGraph(
                clusters = [cluster],
                edges = [],
                tickers = [m.ticker for m in decomposed_moves],
                num_clusters = 1,
                num_singletons = 1,
            )
        tickers = [m.ticker for m in decomposed_moves]
        move_lookup = {m.ticker: m for m in decomposed_moves}

        logger.info(f"Building causal graph for {len(tickers)} flagged tickers")

        #Step 1: Fetching return data and compute correlations
        returns_df = self._fetch_returns(tickers)
        if returns_df is None or returns_df.shape[1] < 2:
            return self._fallback_graph(decomposed_moves)
        
        #Step 2: Correlation matrix
        corr_matrix = returns_df.corr().values
        valid_tickers = list(returns_df.columns)
        n = len(valid_tickers)

        #Step 3: Partial correlation matrix (residualize against SPY)
        partial_corr_matrix = self._compute_partial_correlations(returns_df)

        #Defensive alignment for matrix shapes in case of upstream data quirks.
        if partial_corr_matrix.shape != (n, n):
            min_n = min(n, partial_corr_matrix.shape[0], partial_corr_matrix.shape[1], corr_matrix.shape[0], corr_matrix.shape[1])
            logger.warning(
                "Matrix shape mismatch in causal graph builder (corr=%s, partial=%s, tickers=%s). "
                "Truncating to %s.",
                corr_matrix.shape, partial_corr_matrix.shape, n, min_n,
            )
            valid_tickers = valid_tickers[:min_n]
            corr_matrix = corr_matrix[:min_n, :min_n]
            partial_corr_matrix = partial_corr_matrix[:min_n, :min_n]
            n = min_n
            if n < 2:
                return self._fallback_graph(decomposed_moves)

        #Step 4: Building edges from significant partial correlations
        edges = []
        adjacency = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                raw_corr = corr_matrix[i, j]
                partial_corr = partial_corr_matrix[i, j]

                if abs(partial_corr) >= self.partial_corr_threshold:
                    adjacency[i, j] = abs(partial_corr)
                    adjacency[j, i] = abs(partial_corr)

                    #Determine lead-lag 
                    lead_lag = self._detect_lead_lag(
                        returns_df[valid_tickers[i]],
                        returns_df[valid_tickers[j]],
                    )

                    edge = CausalEdge(
                        source = valid_tickers[i] if lead_lag >= 0 else valid_tickers[j],
                        target = valid_tickers[j] if lead_lag >= 0 else valid_tickers[i],
                        correlation = round(raw_corr, 3),
                        partial_correlation= round(partial_corr, 3),
                        lead_lag_days = abs(lead_lag),
                        edge_strength = round(abs(partial_corr), 3),
                    )
                    edges.append(edge)

        #Step 5: Cluster tickers
        cluster_labels = self._spectral_cluster(adjacency, n)

        #Step 6: Build MoveCluster objects
        clusters = self._build_clusters(
            cluster_labels, valid_tickers, move_lookup, edges
        )

        #Include the tickers that weren't in returns_df
        missing_tickers = set(tickers) - set(valid_tickers)
        for ticker in missing_tickers:
            if ticker in move_lookup:
                clusters.append(MoveCluster(
                    cluster_id = max(c.cluster_id for c in clusters) + 1 if clusters else 0,
                    tickers = [ticker],
                    epicenter_ticker = ticker,
                    epicenter_idio_return= move_lookup[ticker].idiosyncratic_return,
                    moves = [move_lookup[ticker]],
                ))

        num_singletons = sum(1 for c in clusters if c.is_singleton)

        graph = CausalGraph(
            clusters = clusters,
            edges = edges,
            correlation_matrix= corr_matrix,
            partial_correlation_matrix= partial_corr_matrix,
            tickers = valid_tickers,
            num_clusters = len(clusters),
            num_singletons=num_singletons,
            num_multi_move_clusters=len(clusters) - num_singletons,
        )

        logger.info(
            f"Causal graph: {graph.num_clusters} clusters "
            f"({graph.num_multi_move_clusters} multi-move, {graph.num_singletons} singletons), "
            f"{len(edges)} edges"
        )
        return graph 
    
    #-- Correlation Computation --#

    def _fetch_returns(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """ Fetching daily returns for all tickers + SPY"""
        all_tickers = list(dict.fromkeys(tickers + ["SPY"]))
        tickers_str = " ".join(all_tickers)

        try:
            data: Optional[pd.DataFrame] = yf.download(
                tickers_str, period = f"{self.lookback_days + 10}d",
                group_by = "tickers", progress = False, threads = True,
            )
            if data is None or data.empty:
                return None

            returns = pd.DataFrame()
            for t in all_tickers:
                try:
                    close = _extract_close_series(data, t, all_tickers)
                    if close is None:
                        continue
                    returns[t] = close.pct_change().dropna()
                except (KeyError, TypeError):
                    continue 

            return returns.dropna() if not returns.empty else None 
        except Exception as e:
            logger.warning(f"Failed to fetch return data: {e}")
            return None 
        
    def _compute_partial_correlations(self, returns_df: pd.DataFrame) -> np.ndarray:
        """ 
        Computing partial correlation matrix, controlling for SPY (market factor)
        
        For each pair (i, j), the partial correlation is:
            p(i, j | SPY) = correlation of residuals after regressing both i and j on SPY

        This removes any spurious correlation driven by shared overlapping market exposure
        """

        tickers = [c for c in returns_df.columns if c != "SPY"]
        n = len(tickers)
        partial_corr = np.zeros((n,n))

        if "SPY" not in returns_df.columns:
            #No market factor available - falling back to raw correlations
            return returns_df[tickers].corr().values
        
        spy = np.asarray(returns_df["SPY"].to_numpy(), dtype=float)

        #Residualize each ticker against SPY
        residuals = {}
        for t in tickers:
            y = np.asarray(returns_df[t].to_numpy(), dtype=float)

            #Simple OLS: y = a + b*SPY + e
            X = np.empty((spy.shape[0], 2), dtype=float)
            X[:, 0] = 1.0
            X[:, 1] = spy

            try:
                betas, _, _, _ = np.linalg.lstsq(X, y, rcond = None)
                residuals[t] = y - X @ betas 
            except np.linalg.LinAlgError:
                residuals[t] = y 

        #Correlation of residuals
        for i in range(n):
            partial_corr[i, i] = 1.0
            for j in range(i + 1, n):
                r_i = residuals[tickers[i]]
                r_j = residuals[tickers[j]]
                corr = self._safe_corr(r_i, r_j)
                partial_corr[i, j] = corr
                partial_corr[j, i] = corr 

        return partial_corr

    @staticmethod
    def _safe_corr(x: object, y: object) -> float:
        """Compute Pearson correlation defensively for typing/runtime stability."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = min(x_arr.shape[0], y_arr.shape[0])
        if n < 2:
            return 0.0

        x_arr = x_arr[-n:]
        y_arr = y_arr[-n:]
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if valid.sum() < 2:
            return 0.0

        x_valid = x_arr[valid]
        y_valid = y_arr[valid]
        x_centered = x_valid - x_valid.mean()
        y_centered = y_valid - y_valid.mean()
        denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        if denom == 0.0:
            return 0.0
        return float(np.dot(x_centered, y_centered) / denom)
    
    #-- Lead Lag Detection -- #

    def _detect_lead_lag(
        self, series_a: pd.Series, series_b: pd.Series, max_lag: int = 5
    ) -> int:
        """
        Detect temporal lead-lag between two return series.
        Returns positive if A leads B, negative if B leads A, 0 if they are simultaneous.
        Uses cross-correlation at different lags
        """

        a = series_a.values
        b = series_b.values 
        n = min(len(a), len(b))
        a = a[-n:]
        b = b[-n:]

        best_lag = 0
        best_corr = abs(self._safe_corr(a, b))

        for lag in range(1, max_lag + 1):
            #A leads B by 'lag' days
            corr_forward = abs(self._safe_corr(a[:-lag], b[lag:]))
            if corr_forward > best_corr:
                best_corr = corr_forward
                best_lag = lag 

            #B leads A by 'lag' days
            corr_backward = abs(self._safe_corr(a[lag:], b[:-lag]))
            if corr_backward > best_corr:
                best_corr = corr_backward
                best_lag = -lag 

        return best_lag 
    
    # -- Clustering -- #

    def _spectral_cluster(self, adjacency: np.ndarray, n: int) -> np.ndarray:
        """
        Spectral clustering on the partial correlation adjacency matrix 
        Determines the number of clusters automatically via eigengap heuristics
        """

        if n <= 2:
            #Trivial case
            if adjacency[0, 1] >= self.partial_corr_threshold:
                return np.array([0,0])
            return np.array([0,1])
        
        #Degree matrix
        D = np.diag(adjacency.sum(axis = 1))

        #Laplacian: L = D - A
        L = D - adjacency 

        #Normalized Laplacian: D^{-1/2} L D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        #Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            #Fallback: Each ticker is its own cluster
            return np.arange(n)
        
        #Eigengap heuristic: find largest gap in the sorted eigenvalues
        #to determine the number of clusters
        sorted_eigs = np.sort(eigenvalues)
        gaps = np.diff(sorted_eigs[:min(n, 10)]) #looking at first 10

        if len(gaps) == 0:
            return np.arange(n)
        
        #Number of clusters = argmax(gap) + 1 
        #But capping at a reasonable number and floor at 1
        k = int(np.argmax(gaps)) + 1
        k = max(1, min(k, n, 8))

        #K-means on first k eigenvectors
        V = eigenvectors[:, :k]

        #Normalize rows 
        row_norms = np.linalg.norm(V, axis = 1, keepdims = True) + 1e-10
        V_norm = V / row_norms

        #Simple k-means (avoiding sklearn dependency)
        labels = self._simple_kmeans(V_norm, k)
        return labels 
    
    @staticmethod
    def _simple_kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """ Minimal k-means implementation"""
        n = X.shape[0]
        if k >= n:
            return np.arange(n)
        
        #initialize centroids randomly
        rng = np.random.RandomState(42)
        idx = rng.choice(n, k, replace=False)
        centroids = X[idx].copy() 

        labels = np.zeros(n, dtype = int)

        for _ in range(max_iter):
            #Assigning to nearest centroid
            new_labels = np.zeros(n, dtype = int)
            for i in range(n):
                dists = np.linalg.norm(centroids - X[i], axis = 1)
                new_labels[i] = np.argmin(dists)

            #Checking convergence
            if np.array_equal(labels, new_labels):
                break 
            labels = new_labels

            #Update centroids 
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = X[mask].mean(axis = 0)

        return labels 
    
    #-- Cluster Construction --#

    def _build_clusters(
        self,
        cluster_labels: np.ndarray,
        tickers: List[str],
        move_lookup: Dict[str, DecomposedMove],
        edges: List[CausalEdge],
    ) -> List[MoveCluster]:
        """ Build MoveCluster objects from the cluster labels"""
        clusters_dict: Dict[int, List[str]] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters_dict[int(label)].append(tickers[i])

        clusters = []
        for cid, cluster_tickers in clusters_dict.items():
            cluster_moves = [
                move_lookup[t] for t in cluster_tickers if t in move_lookup
            ]
            if not cluster_moves:
                continue 

            #Epicenter: Largest absolute idiosyncratic return
            epicenter = max(cluster_moves, key = lambda m: abs(m.idiosyncratic_return))

            #Dominant sector
            sectors = [m.sector for m in cluster_moves if m.sector]
            dominant_sector = max(set(sectors), key = sectors.count) if sectors else None

            #Internal edges
            ticker_set = set(cluster_tickers)
            internal_edges = [
                e for e in edges
                if e.source in ticker_set and e.target in ticker_set
            ]

            #Average correlations
            avg_corr = (
                float(np.mean([e.correlation for e in internal_edges]))
                if internal_edges else 0.0
            )
            avg_partial = (
                float(np.mean([e.partial_correlation for e in internal_edges]))
                if internal_edges else 0.0
            )

            #Return range
            returns = [m.idiosyncratic_return for m in cluster_moves]
            return_range = (round(min(returns), 2), round(max(returns), 2))

            #Coherence: average pairwise partial correlatoin
            coherence = abs(avg_partial) if len(cluster_tickers) > 1 else 1.0

            clusters.append(MoveCluster(
                cluster_id = cid, 
                tickers = cluster_tickers,
                epicenter_ticker = epicenter.ticker,
                epicenter_idio_return=epicenter.idiosyncratic_return,
                dominant_sector=dominant_sector,
                avg_correlation=float(round(avg_corr, 3)),
                avg_partial_correlation=float(round(avg_partial,3)),
                moves = cluster_moves,
                internal_edges = internal_edges,
                total_return_range = return_range,
                coherence_score=float(round(coherence, 3)),
            ))

        #Sort by max severity then epicenter magnitude
        level_order = {AlertLevel.CRITICAL: 0, AlertLevel.HIGH: 1,
                       AlertLevel.MEDIUM: 2, AlertLevel.LOW: 3}
        clusters.sort(key = lambda c: (
            level_order.get(c.max_severity, 4), -abs(c.epicenter_idio_return),
        ))

        return clusters 
    
    def _fallback_graph(self, moves: List[DecomposedMove]) -> CausalGraph:
        """ Fallback when correlation computation fails - each move is its own cluster"""
        clusters = []
        for i, m in enumerate(moves):
            clusters.append(MoveCluster(
                cluster_id = i,
                tickers = [m.ticker],
                epicenter_ticker = m.ticker,
                epicenter_idio_return= m.idiosyncratic_return,
                moves = [m],
            ))

        return CausalGraph(
            clusters = clusters, edges = [], tickers = [m.ticker for m in moves],
            num_clusters = len(clusters), num_singletons=len(clusters),
        )
