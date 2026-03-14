# Financial Intelligence - Evaluation Report

 ** Generated **: 2026-03-14T00:31:12.248277+00:00
 ** Tickers evaluated**: 25

##1. Factor Decomposition Model

| Metric | Value |
|--------|-------|
| Avg R^2 (out-of-sample) | 0.4153 |
| Median R^2 | 0.415 |
| Calibration error | 0.02978 |
| Residuals passing normality test | 52.0% |
| Avg excess kurtosis | 2.79 |
| Avg beta stability (sigma) | 0.3395 |

**Distribution Fit**: Student's t with df=4.52 (Normal calibration error: 0.02211, t-distribution error: 0.02978, improvement: -34.7%)

**Calibration Curve (Normal vs Student's t)**:

| σ Threshold | Expected (Normal) | Expected (t) | Empirical | Ratio (Normal) | Ratio (t) |
|-------------|-------------------|--------------|-----------|----------------|-----------|
| 1.5σ | 0.13361 | 0.19994 | 0.148 | 1.11x | 0.74x |
| 2.0σ | 0.0455 | 0.10798 | 0.07933 | 1.74x | 0.73x |
| 2.5σ | 0.01242 | 0.05967 | 0.036 | 2.9x | 0.6x |
| 3.0σ | 0.0027 | 0.03419 | 0.01933 | 7.16x | 0.57x |

## 2. SLM (Fine-tuned FinBERT)

| Task | Metric | Value |
|------|--------|-------|
| Classification | Accuracy | 0.9267 |
| Classification | Macro-F1 | 0.3643 |
| Classification | Top-3 Accuracy | 0.9553 |
| Sentiment | MAE | 0.092 |
| Sentiment | Spearman ρ | 0.8565 |
| Sentiment | Directional Accuracy | 0.9266 |
| Relevance | Precision | 0.8279 |
| Relevance | Recall | 0.9806 |
| Relevance | F1 | 0.8978 |
| Relevance | AUC-ROC | 0.8737 |

**Teacher Agreement**: cls=0.9267, cls_top3=0.9553, sentiment_rank=0.8565, relevance=0.8553

**Latency**: single=26.3ms, batch100=899.03ms, **55.6x faster** than API

## 3. Causal Graph Clustering

| Metric | Value |
|--------|-------|
| Adjusted Rand Index | 1.0 |
| Silhouette Score | 0.5278 |
| Avg Cluster Coherence | 0.414 |
| Clusters | 4 (2 multi, 2 singletons) |
| Sector-Pure Clusters | 100.0% |
| Epicenter = Max σ | 50.0% |
| Edges Same-Sector | 100.0% |

## 4. LLM-as-Judge (Brief Quality)

| Dimension | Avg Score (1-5) | Std |
|-----------|-----------------|-----|
| Factual Accuracy | 3.0 | 0.0 |
| Causal Reasoning | 4.0 | 0.0 |
| Decomposition Awareness | 5.0 | — |
| Actionability | 4.0 | — |
| Information Density | 4.0 | — |
| **Overall** | **4.0** | 0.0 |

Hallucination rate: 100.0% of briefs
Attribution error rate: 0.0% of briefs
Avg vague recommendations: 2.0 per brief

## 5. Counterfactual Filtering Rate

| Metric | Value |
|--------|-------|
| Raw moves flagged | 11 |
| Passed (idiosyncratic) | 7 |
| Filtered (systematic) | 4 |
| **Filtering rate** | **36.4%** |
| Avg filtered idio σ | 1.07 |

**Next-Day Reversion Analysis** (validates decomposition):

| Category | Reversion Rate | SPY Correlation |
|----------|---------------|-----------------|
| Systematic (filtered) | 0.0% | 0.75 |
| Idiosyncratic (passed) | 57.1% | 0.571 |
| **Separation** | **-57.1pp** | |

Event contamination: 50.0% false negative rate (1/2 events)

**Filter Quality Score: 0.5/1.0**

False negatives:META (2024-02-02: earnings_beat)
