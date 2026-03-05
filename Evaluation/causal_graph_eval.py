"""
Causal Graph Evaluation
Evaluates the quality of the cross-move causal clustering:
    1. Adjusted Rand Index - cluster agreement with known sector labels
    2. Silhouette Score - intra-cluster cohesion vs. inter-cluster separation
    3. Temporal Stability - cluster consistency across consecutive days
    4. Epicenter Validation - does the identified epicenter match ground truth?
"""

import logging
from typing import Dict, List, Optional, Tuple 
from dataclasses import dataclass, field 
from collections import Counter 

import numpy as np 

from Quant.causal_graph import CausalGraph, MoveCluster, CausalGraphBuilder
from Quant.factor_decomposition import DecomposedMove

logger = logging.getLogger(__name__)

@dataclass
class CausalGraphMetrics:
    """Complete causal graph evaluation metrics"""
    #Cluster quality
    adjusted_rand_index: float = 0.0 #Agrement with sector labels
    silhouette_score: float = 0.0 #Cluster cohesion / separation
    avg_cluster_coherence: float = 0.0 #Avg intra cluster partial correlation

    #Cluster characteristics
    num_clusters: int = 0
    num_singletons: int = 0
    num_multi_move: int = 0
    avg_cluster_size: float = 0.0
    pct_sector_pure_clusters: float = 0.0 #% of clusters where tickers share a sector

    #Temporal stability (if multi-day data is available)
    temporal_stability_score: float = 0.0 #Jaccard similarity between consecutive days
    cluster_persistence_rate: float = 0.0 #% of clusters that reappear next day

    #Epicenter accuracy
    epicenter_is_max_sigma: float = 0.0 #% of clusters where epicenter has max |sigma|
    epicenter_is_earliest: float = 0.0 #% where epicenter moved first (if lead-lag data)

    #Edge quality
    avg_edge_partial_corr: float = 0.0
    pct_edges_same_sector: float = 0.0 #Sanity: most edges should link same sector tickers

    n_evaluations: int = 0

class CausalGraphEvaluator:
    """
    Evaluates the causal graph quality
    """

    def evaluate_single(self, graph: CausalGraph) -> CausalGraphMetrics:
        """ Evaluating a single causal graph output"""
        metrics = CausalGraphMetrics(n_evaluations = 1)

        if not graph.clusters:
            return metrics 
        
        metrics.num_clusters = graph.num_clusters 
        metrics.num_singletons = graph.num_singletons 
        metrics.num_multi_move = graph.num_multi_move_clusters

        sizes = [c.size for c in graph.clusters]
        metrics.avg_cluster_size = round(float(np.mean(sizes)), 2) if sizes else 0.0

        #Adjusted Rand Index vs Sector Labels
        metrics.adjusted_rand_index = self._compute_ari(graph)

        #Silhouette score
        if graph.partial_correlation_matrix is not None and len(graph.tickers) > 2:
            metrics.silhouette_score = self._compute_silhouette(graph)

        #Cluster coherence
        coherences = [c.coherence_score for c in graph.clusters if not c.is_singleton]
        metrics.avg_cluster_coherence = (
            round(float(np.mean(coherences)), 4) if coherences else 0.0
        )

        #Sector purity
        multi_clusters = [c for c in graph.clusters if not c.is_singleton]
        if multi_clusters:
            pure = sum(1 for c in multi_clusters if c.is_sector_cluster)
            metrics.pct_sector_pure_clusters = round(pure / len(multi_clusters) * 100, 1)

        #Epicenter validation
        metrics.epicenter_is_max_sigma = self._validate_epicenters(graph)

        #Edge quality
        if graph.edges: 
            metrics.avg_edge_partial_corr = round(
                float(np.mean([abs(e.partial_correlation) for e in graph.edges])), 4
            )
            same_sector = 0
            total_edges = 0
            for e in graph.edges:
                s_src = self._get_ticker_sector(e.source, graph)
                s_tgt = self._get_ticker_sector(e.target, graph)
                if s_src and s_tgt:
                    total_edges += 1
                    if s_src == s_tgt:
                        same_sector += 1
            metrics.pct_edges_same_sector = (
                round(same_sector / total_edges * 100, 1) if total_edges > 0 else 0.0
            )

        return metrics
    
    def evaluate_temporal_stability(
        self,
        graphs: List[CausalGraph],
    ) -> Dict[str, float]:
        """
        Evaluate cluster consistency across consecutive time steps.
        Graphs should be causal grpahs from consecutive days
        """
        if len(graphs) < 2:
            return {"temporal_stability": 0.0, "persistence_rate": 0.0}
        
        jaccard_scores = []
        persistence_counts = []

        for i in range(1, len(graphs)):
            prev_clusters = self._clusters_to_sets(graphs[i-1])
            curr_clusters = self._clusters_to_sets(graphs[i])

            #Pairwise max Jaccard between prev and curr clusters
            if prev_clusters and curr_clusters:
                max_jaccards = []
                persistent = 0

                for prev_set in prev_clusters:
                    best_j = 0.0
                    for curr_set in curr_clusters:
                        intersection = len(prev_set & curr_set)
                        union = len(prev_set | curr_set)
                        j = intersection / union if union > 0 else 0.0
                        best_j = max(best_j, j)
                    max_jaccards.append(best_j)
                    if best_j > 0.5: #cluster "persisted" if >50% overlap
                        persistent += 1

                jaccard_scores.append(float(np.mean(max_jaccards)))
                persistence_counts.append(persistent / len(prev_clusters))

        return {
            "temporal_stability": round(float(np.mean(jaccard_scores)), 4) if jaccard_scores else 0.0,
            "persistence_rate": round(float(np.mean(persistence_counts)), 4) if persistence_counts else 0.0,
        }
    
    #-- Adjusted Rand Index --#
    def _compute_ari(self, graph: CausalGraph) -> float:
        """
        Adjusted Rand Index: measures agreement between our clustering
        and the known sector labels. ARI = 1 means a perfect agreement,
        ARI = 0 means its random, ARI < 0 means its worse than random
        """
        #build two label vectors: cluster assignment vs sector
        cluster_labels = []
        sector_labels = []

        for cluster in graph.clusters:
            for move in cluster.moves:
                cluster_labels.append(cluster.cluster_id)
                sector_labels.append(move.sector or "unknown")

        if len(cluster_labels) < 2:
            return 0.0
        
        return self._ari(cluster_labels, sector_labels)
    
    @staticmethod
    def _ari(labels_a: List, labels_b: List) -> float:
        """ Compute Adjusted Rand Index between two clusterings"""
        n = len(labels_a)
        if n < 2:
            return 0.0 
        
        #Contingency table
        contingency: Dict[Tuple, int] = Counter(zip(labels_a, labels_b))

        #Row and Column sums
        a_counts = Counter(labels_a)
        b_counts = Counter(labels_b)

        #Combinatorial terms
        def comb2(x):
            return x * (x-1) / 2
        
        sum_cij2 = sum(comb2(v) for v in contingency.values())
        sum_ai2 = sum(comb2(v) for v in a_counts.values())
        sum_bj2 = sum(comb2(v) for v in b_counts.values())
        comb_n = comb2(n)

        if comb_n == 0:
            return 0.0

        expected = sum_ai2 * sum_bj2 / comb_n 
        max_index = (sum_ai2 + sum_bj2) / 2 
        denominator = max_index - expected 

        if abs(denominator) < 1e-10:
            return 0.0 if abs(sum_cij2 - expected) > 1e-10 else 1.0
        
        return round(float((sum_cij2 - expected) / denominator), 4)
    
    #-- Silhouette Score -- #
    def _compute_silhouette(self, graph: CausalGraph) -> float:
        """ 
        Silhouette score: Measures how similar each point is to its own cluster
        versus the nearest other cluster. Uses partial correlation as similarity.
        Range: [-1, 1]. The higher the score the better
        """
        if graph.partial_correlation_matrix is None:
            return 0.0

        tickers = graph.tickers
        n = len(tickers)
        pcm = graph.partial_correlation_matrix

        #Distance matrix: 1 - |partial_corr|
        dist = 1.0 - np.abs(pcm)

        #Build cluster assignment
        ticker_to_cluster = {}
        for cluster in graph.clusters:
            for ticker in cluster.tickers:
                if ticker in tickers:
                    ticker_to_cluster[ticker] = cluster.cluster_id

        silhouettes = []
        for i, ticker in enumerate(tickers):
            if ticker not in ticker_to_cluster:
                continue 

            my_cluster = ticker_to_cluster[ticker]

            #a(i) = avg distance to own cluster members
            own_dists = []
            other_cluster_dists: Dict[int, List[float]] = {}

            for j, other_ticker in enumerate(tickers):
                if i == j or other_ticker not in ticker_to_cluster:
                    continue 
                other_cluster = ticker_to_cluster[other_ticker]
                d = dist[i,j]

                if other_cluster == my_cluster:
                    own_dists.append(d)
                else:
                    other_cluster_dists.setdefault(other_cluster, []).append(d)

            a_i = float(np.mean(own_dists)) if own_dists else 0.0

            #b(i) = min avg distance to any other cluster
            if other_cluster_dists:
                b_i = min(float(np.mean(ds)) for ds in other_cluster_dists.values())
            else:
                b_i = 0.0

            #s(i) = (b-a) / max(a,b)
            max_ab = max(a_i, b_i)
            s_i = (b_i - a_i) / max_ab if max_ab > 1e-10 else 0.0
            silhouettes.append(s_i)

        return round(float(np.mean(silhouettes)), 4) if silhouettes else 0.0

    # -- Epicenter Validation -- #

    def _validate_epicenters(self, graph: CausalGraph) -> float:
        """
        For each multi-move cluster, check if the identified epicenter is the ticker
        with the largest |idiosyncratic_sigma|
        """
        multi_clusters = [c for c in graph.clusters if not c.is_singleton and c.moves]
        if not multi_clusters:
            return 0.0
        
        correct = 0
        for cluster in multi_clusters:
            actual_max = max(cluster.moves, key = lambda m: abs(m.idiosyncratic_sigma))
            if actual_max.ticker == cluster.epicenter_ticker:
                correct += 1
        return round(correct / len(multi_clusters) * 100, 1)
    
    # -- Helpers -- #
    
    @staticmethod 
    def _clusters_to_sets(graph: CausalGraph) -> List[set]:
        """ Converting clusters to sets of tickers (for Jaccard comparison)"""
        return [
            set(c.tickers) for c in graph.clusters
            if not c.is_singleton
        ]
    
    @staticmethod
    def _get_ticker_sector(ticker: str, graph: CausalGraph) -> Optional[str]:
        for cluster in graph.clusters:
            for move in cluster.moves:
                if move.ticker == ticker:
                    return move.sector 
                
        return None 