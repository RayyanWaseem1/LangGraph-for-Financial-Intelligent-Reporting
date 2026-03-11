# Financial Intelligence - Evaluation Report

 ** Generated **: 2026-03-10T18:40:07.984682+00:00
 ** Tickers evaluated**: 25

##1. Factor Decomposition Model

| Metric | Value |
|--------|-------|
| Avg R^2 (out-of-sample) | 0.4106 |
| Median R^2 | 0.3876 |
| Calibration error | 0.02411 |
| Residuals passing normality test | 52.0% |
| Avg excess kurtosis | 2.614 |
| Avg beta stability (sigma) | 0.3399 |

**Calibration Curve**:

| Sigma Threshold | Expected Freq | Empirical Freq | Ratio | Error |
|-------------|---------------|----------------|-------|-------|
| 1.5 sigma | 0.13361 | 0.15 | 1.12x | 0.01639 |
| 2.0 sigma | 0.0455 | 0.08267 | 1.82x | 0.03717 |
| 2.5 sigma | 0.01242 | 0.038 | 3.06x | 0.02558 |
| 3.0 sigma | 0.0027 | 0.02 | 7.41x | 0.0173 |

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

**Latency**: single=26.43ms, batch100=862.28ms, **58.0x faster** than API
