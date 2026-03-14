# LangGraph-for-Financial-Intelligent-Reporting
### A quantitative equity monitoring pipeline combining two-factor decomposition (market + orthogonalized sector), spectral causal graph clustering, a fine-tuned multi-task FinBERT SLM, and Claude LLM synthesis, with a five-component evaluation framework 

## Motivation and Design Philosophy
### The Problem with Naive Equity Monitoring
The standard approach to equity monitoring; flag any stock that moves more than some fixed percentage, retrieve news articles, and summarize them with an LLM; conflates three fundamentally different market phenomena into a single undifferentiated signal. For example, when a semiconductor stock drops 4% on a day where the S&P 500 drops 3%, the naive system fires an alert and sends Claude or any other LLM to go find out what happened to that semiconductor company. But there is nothing to explain. The stock has a market beta of roughly 1.3, and 1.3 x 3% is 3.9%. The move is almost entirely systematic, meaning it is just the market moving, not the company.

This conflation is not merely an aesthetic problem. It tends to waste computational resources (API calls to news services and LLMs for moves that have a trivial explanation), generates false alerts that desensitize the end user, and most critically, it obscures the moves that actually matter. When a managed care company drops 20% on a day when the market is flat and the healthcare sector is flat, that is a genuine indiosyncratic event, such as an earnings miss, a regulatory action, or fraud revelation. These events demand immediate analytical attention. But in a naive system, this critical signal is buried alongside dozens of alerts about stocks that simplly moved with the market.

The core idea behind this project is that the quantitative layer should exist as a filtering mechanism, not an analysis mechanism. The factor model, the prediciton residual, and the causal graph are not trying to explain price movements. That is the job of the LLM. They exist to remove noise, so that the LLM can focus on signal. A well-decomposed move tells the LLM: "this stock dropped 15.2%, of which 0.1% was market beta and 0.0% was sector rotation. The remaining 15.1% is unexplained. Here are the news articles. Tell me why."

### The Two-Agent Architecture
The second design principle concerns the division of labor between models. Large language models are remarkably good at synthesis, causal reasoning, and generating structured analytical narratives. They are remarkably bad at being called 500 times in a loop to classify individual news articles. Latency and cost make this impractical. A single Claude API call, for example, to classify one article can take approximately 500ms and costs a fraction of a cent. Multiplied by 252 articles across 47 flagged tickers, this becomes 126 seconds and a non-trivial cost per pipeline run.

My solution is a two-agent architecture where a locally-deployed fine-tuned transformer (the SLM, or Small Language Model) handles all high-volume, narrow-bandwidth tasks, such as classification into 6 event categories, sentiment scoring on a [-1, +1] scale, and binary relevance determination, all in a single forward pass per article at 28 ms with zero marginal cost. The SLM processes the same 252 articles in under one second. The LLM then receives the SLM-enriched data; articles pre-classified, pre-scored, and pre-filtered; and makes a single API call to synthesize the intelligence brief.

This is not merely an optimization. It changes the kind of reasoning the LLM performs. Instead of asking the LLM "What category is this article?" 252 times over, we ask it once: "Given these 47 idiosyncratic moves, organized into 3 causal clusters, with SLM-scored news coverage, generate an executive summary, sector impact analysis, and prioritized action recommendations." The LLM operates at the level of strategic synthesis rather than article-level classification.

### The Evaluation Philosophy
A complex system like this, spanning factor models, graph algorithms, transformer fine-tuning, and LLM generation, could not have been evaluated by a single metric. Each component has its own failure models, and the interactions between components create emergent failure modes that no individual test captures. A factor model might have excelled R-squared but poor calibration at the tails. An SLM might achieve 93% accuracy but fail entirely on rare event categories. A causal graph might produce perfect clusters that the LLM misinterprets. 

The evaluation framework therefore tests five independent dimensions: factor model quality (R-squared, calibration, beta stability), SLM task performance (classification, sentiment, relevance, latency), causal graph clustering quality (ARI, silhouette, sector purity), LLM brief quality via an LLM-as-Judge protocol (factual accuracy, causal reasoning, decomposition awareness, actionability, information density), and counterfactual filtering validation (does the decomposition correctly distinguish systematic from idiosyncratic moves). Each dimension is independently configurable, and the framework supports ablation studies via --sigma and --distribution flags that control detection thresholds and tail distribution models.

## System Architecture
The pipeline processes the full S&P 500 (503 tickers as of my evaluation date) through six sequential stages, each of which progressively filters and enriches the data before passing it downstream. The architecture is implemented as a LangGraph state machine where each stage reads from and writes to a shared PipelineState dictionary. The state carries the full provenance chain, from raw price data through decomposition components, cluster assignments, SLM scores, and finally the LLM's structured output, enabling the evaluation framework to trace any claim in the final brief back to its source data.

┌──────────────────────────────────────────────────────────────────────┐
│                     S&P 500 (503 tickers)                            │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  1. Market Monitor   │  yfinance · volatility-adjusted Nσ thresholds
            │     49 raw moves     │  daily + weekly scans · volume filters
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  2. Factor Decomp    │  R = α + β_mkt·SPY + β_sec·SectorETF + ε
            │     47 idiosyncratic │  OLS regression · 120-day lookback
            │      2 filtered      │  idiosyncratic σ threshold (default 1.5σ)
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  3. Prediction       │  ReturnPredictor: Random Forest on
            │     Residual         │  lagged returns, VIX, day-of-week
            └──────────┬──────────┘  flags moves the ML model can't predict
                       │
            ┌──────────▼──────────┐
            │  4. News Retrieval   │  GDELT (sequential, 1.5s delays)
            │     252 articles     │  + NewsAPI fallback (30-day window)
            └──────────┬──────────┘  72-hour lookback per ticker
                       │
            ┌──────────▼──────────┐
            │  5. Causal Graph     │  Partial correlation network
            │     3 clusters       │  Spectral clustering · epicenter detection
            └──────────┬──────────┘  coherence scoring per cluster
                       │
            ┌──────────▼──────────┐
            │  6a. SLM Agent       │  Fine-tuned FinBERT (local, 28ms/article)
            │  Classification      │  6-class move categorization
            │  Sentiment           │  [-1, +1] financial sentiment
            │  Relevance           │  Binary article-move relevance
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  6b. LLM Agent       │  Claude Sonnet (1 API call, 16K tokens)
            │  Impact Assessment   │  Cluster-level causal analysis
            │  Brief Generation    │  Executive summary + recommendations
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Market Intelligence │  JSON brief with alerts, sector analysis,
            │  Brief               │  5 prioritized recommendations, cluster data
            └─────────────────────┘

## Pipeline: Step-by_Step
### Market Monitor
The market monitor scans all S&P 500 constituents (fetched from Wikipedia with a proper User-Agent header to avoid HTTP 403 blocks) for price moves exceeding volatility-adjusted thresholds. Each ticker's daily return is compared against its own trailing 60-day standard deviation: a move triggers an alert when its absolute value exceeds N standard deviations, where N defaults to 2.0. Alert severity is assigned by sigma magnitude -- LOW for 1.5-2 sigma, MEDIUM for 2-2.5 sigma, HIGH for 2.5-3 sigma, and CRITICAL for moves exceeding 3 sigma. Absolute fallback thresholds of 5% daily and 10% weekly catch moves in low-volatility regimes where the sigma-based threshold might be too permissive. On a typical volatile day, the monitor flags 30-50 of 503 tickers. On a quiet day, it may flag as few as 1-5. This variability was thought out in its design as the system's sensitivity adapts to market conditions through the volatility normalization.

### Factor Decomposition 
For each flagged move, the decomposer fits a two-factor ordinary least squares model on a 120-day trailing window. The model regresses the ticker's daily return on the SPY market return and an orthogonalized sector ETF return. The orthogonalization is a critical detail: without it, the market and sector regressors are collinear (sector ETFs have significant market beta themselves), which inflates standard errors and produces unstable coefficient estimates. The orthogonalized sector return was computed as the residual of regressing the sector ETF on SPY. This captures the pure sector rotation component, independent of broad market movement.

The idiosyncratic return (the OLS residual epsilon) is then standardized by the historical residual volatility to produce an idiosyncratic sigma. Moves whose idiosyncratic sigma falls below the threshold (default 1.5 sigma) are classified as "mostly systematic" and filtered from downstream processing. On the evaluation date (March 13, 2026), this step filtered 2 of 42 raw moves as systematic, a modest filtering rate that reflects a genuinely volatile day with many company-specific events. On quieter days, filtering rates of 40-60% may be typical.

Newer versions of yfinance return MultiIndex DataFrame columns for single-ticker downloads, which silently breaks any code that accesses price data directly. A custom _extract_close() helper was thus implemented to handle both column formats and deployed across all four modules that fetch price data. This is the kind of brittle dependency that causes silent failures in production ML pipelines and underscores the importance of defensive data access patterns.

### Prediction Residual
Beyond the linear factor model, a ReturnPredictor module fits a multivariate OLS regression for each flagged ticker, using lagged features (5-day sector ETF momentum, 20-day sector momentum, 5-day SPY momentum, 20-day realized volatility, VIX level, and day-of-week encoding) to estimate expected returns. The prediction residual, which is the gap between the model's prediction and the actual return, provides additional context for the LLM's synthesis. This step does not filter or reprioritize moves; the residuals are passed as supplementary context in the LLM prompt, helping it distinguish between moves that were predictable from recent momentum patterns and those that were genuinely surprising given all available quantitative features. In practice, with only daily features and no intraday or alternative data, the predictor's explanatory power is limited. On the evaluated pipeline run it produced zero successful fits due to data alignment issues, making this an area for future improvement. The architecture is designed to accommodate richer feature sets and more robust estimators in a production deployment. 

### News Retrieval
GDELT and NewsAPI provide article coverage with a 72-hour lookback window per ticker. The GDELT integration required significant iteration to handle the rate limiting. The initial concurrent approach (5 simultaneous requests) triggered immediate 429 responses. The production implementation uses sequential requests with 1.5 second inter-request delays and a 3-retry exponential backoff (3s, 6s, 9s) for failures. NewsAPI serves as a secondary source, providing coverage within its 30-day API window and catching tickers where GDELT returns empty or rate-limited responses. News retrieval runs only for tickers that passed the factor decomposition filter. This was an important efficiency gain that saves 30-60% of API calls on typical days.

### Causal Graph
The causal graph module constructs a partial correlation network among flagged tickers using 60 days of return history. Unlike raw Pearson correlations, which can be spuriously high when two unrealted stocks both correlate with a third factor, partial correlations isolate direct pairwise relationships by removing shared exposure. The implementaiton residualizes each ticker's returns against SPY via OLS regression, then computes pairwise Pearson correlations of the resulting residuals. This effectively removes the market factor from every pair, so that any remaining co-movement reflects sector-level or idiosyncratic linkages rather than shared beta. An adjacency matrix is formed by thresholding these partial correlations at statistically significant levels.

Spectral clustering on the graph Laplacian groups tickers with strong residual co-movement. Each resulting cluster is characterized by an epicenter (the ticker with the highest idiosyncratic sigma, representing the likely source of the shared shock), a dominant sector, a coherence score (the average internal partial correlation, measuring how tightly the cluster members move together), and internal edge structure with lead-lag estimates.

The causal graph transforms the LLM's analytical task from "explain 40 independent ticker moves" to "explain 3 thematic clusters of related moves." This is a qualitative improvement in the kind of reasoning the LLM can perform. Instead of producing 40 disconnected analyses, it can identify shared catalysts: a cluster of consumer staples companies all declining after one member's earnings miss suggests earnings contagion, not 11 independent events.

### SLM Agent
The locally-deployed SLM performs three tasks per article in a single forward pass through a shared ProsusAI/FinBERT backbone with task-specific projection heads. The backbone's 768-dimensional CLS token representation first passes through a shared projection layer (768 to 256 dimensions, with LayerNorm, GELU activation, and dropout) that produces a common intermediate representation for all three tasks. From this shared 256-dimensional representation, the classification head maps through an additional 256-unit hidden layer to a 6-class probability distribution over event categories. The sentiment head maps through a 128-unit hidden layer to a scalar in [-1, +1] via tanh activation. The relevance head maps through a 128-unit hidden layer to a binary probability via sigmoid. The multi-task architecture shares the expensive backbone computation across all three tasks, and the multi-task loss uses learnable uncertainty weights (Kendall et al., 2018) that automatically balance the gradient contributions from classification cross-entropy, sentiment MSE, and relevance binary cross-entropy.

### LLM Agent
The LLM agent receives the complete SLM-enriched, cluster-organized dataset in a single structured prompt. The system message enforces strict grounding rules developed through iterative evaluation: the LLM must only reference tickers present in the input data, must use exact sigma values and return percentages from the decomposition, must not fabricate cluster structures or causal probabilities beyond what the pipeline computed, and must explicitly acknowledge uncertainty when the evidence is insufficient to support a causal claim. The LLM outputs a Pydantic-validated 'ImpactAndBrief' object with 'max_tokens = 16384' (increased from an initial 4096 after observing truncation-induced validation failures). All output fields use 'default_factory' for graceful degradation.

## Multi-Task FinBERT: Training Pipeline
### Training Data Generation
The training data pipeline identifies historical significant moves by scanning 365 days of S&P 500 price data with a 2 sigma threshold, yielding approximately 449 moves with sufficient context for labeling. For each detected move, Claude receives the ticker, return magnitude, date, sector, and any retrieved GDELT/NewsAPI articles, then produces a classification label (one of 25 fine-grained categories, later consolidated to 6), a sentiment score on a [-1, +1] scale, and a relevance assessment for each article-move pair. Low-confidence labels (confidence < 0.3) are filtered, and the resulting seed dataset contained 449 classification, 179 sentiment, and 179 relevance examples, totaling 807 labeled instances.

A significant practical constraint shaped the quality of this seed data. GDELT's rate limiting meant that 94% of training examples had no retrieved article content. Claude was therefore labeling moves based on ticker, date, and return magnitude, and its parametric training knowledge rather than grounded article text. For classification, this is acceptable as an earnings surprise for AAPL on a specific date is unambiguous given the ticker and the magnitude, regardless of whether the article text is present. But for sentiment and relevance, the absence of real article text means that the SLM trains on impoverished input representations that differ substantially from what it encounters at inference time.

### The Distribution Mismatch Problem
This point deserves emphasis because it motivated the entire augmentation strategy. At training time, 94% of the SLM's input texts looked like this: 

AAPL (Apple Inc.) moved -3.2% on 2025-06-15. Category: earnings.
No article content available.

At inference time, when the SLM processes real pipeline data, the input texts look like this: 

Apple Inc. (AAPL) -3.2% (2.8σ). News: Apple Reports Q3 Revenue Miss,
Cuts Full-Year Guidance Citing Weak iPhone Demand in China

This is a textbook train-test distribution mismatch. The model learns to classify based on ticker symbols, percentage magnitudes, and the literal string "No article content available". None of which carry the linguistic features that distinguish "earnings_surprise" from "analyst_rating" or "regulatory_change" in real news headlines. A model trained exclusively on the seed data would learn correct statistical associations (large negative moves for AAPL in late January are usually earnings) but would fail to generalize to the rich, varied financial language it encounters in production.

The solution was not to collect more real data, as GDELT rate limits make that impractical at scale without a paid API, but to synthetically generate the input text while preserving the verified labels. This is a critical distinction as I did not fabricate labels or invent training signal. The labels were already produced by Claude's analysis of real historical moves. I generated realistic financial headlines that matched those verified labels, giving the SLM actual financial language patterns to learn from.

### LLM-Powered Multi-Task Augmentation
The augmentation module uses Claude Haiku (chosen due to its cost efficiency at the required volume) with 'temperature = 0.7' to generate 4 diverse headlines per labeled move. The prompt provides the ticker, company name, percentage change, sigma magnitude, date, and verified category label, and instructs the model to produce headlines that "sound like real Bloomberg/Reuters/CNBC headlines" with "specific details (quarter numbers, dollar amounts, analyst names, metrics)" and "varied styles: some breaking-news, some analytical, some focused on specific data points." The structured output is validated via a Pydantic 'SyntheticHeadlines' model that enforces headline length (under 120 characters) and sentiment range [-1, +1].

The key design choice was that each synthetic headline generates training examples for all three tasks simultaneously, not just classification. Each headline produces four direct training signals: two classification examples, one sentiment example, and one positive relevance example. Additionally, after all headlines are generated, two negative relevance examples are created per move via cross-move sampling, bringing the total to six training signals per headline-move combination.

First, two classification examples are created from each headline. One pairs the headline with the move context (e.g., "Apple Inc. (AAPL) -3.2% (2.8 sigma). News: Apple Reports Q3 Revenue Miss..."), mimicking the format that the SLM would see at inference time. The second uses the headline alone ("Apple Reports Q3 Revenue Miss, Cuts Full-Year Guidance"), teaching the model to classify from pure linguistic features without relying on the ticker or return magnitude as a crutch. This dual-format strategy prevents the model from learning a shortcut where it ignores the headline and classifies solely based on the numerical features.

Second, one sentiment example is created using the headline text with the sentiment score provided by Claude during generation. Because Claude generates both the headline and the sentiment score in the same API call, the sentiment labels are internally consistent. A headline about "cutting guidance" will reliably receive a negative score, and one about "beating estimates" will receive a positive score.

Third, one positive relevance example pairs the headline with its own move (e.g., "[MOVE: AAPL -3.2% down] Apple Reports Q3 Revenue Miss...") with a high relevance score (uniformly sampled from [0.85, 1.0] to introduce slight label noise that prevents overconfidence).

Fourth, and most critically for the relevance task, two negative relevance examples are generated per move via cross-move sampling. After all headlines have been generated and pooled, each move is paired with 2 headlines randomly sampled from different tickers and different moves. For example, the AAPL earnings move might be paired with "Exxon Mobil Reports Record Quarterly Profit on Higher Oil Prices" as a negative relevance example. These negatives receive low relevance scores (uniformly sampled from [0.0, 0.2]) and teach the model the critical distinction between "this article is about finance" and "this article is relevant to this specific move." Without cross-move negatives, the relevance head would learn to score any financial text as relevant regardless of its relationship to the target ticker.

This augmentation strategy expanded the dataset from 807 to 13,399 total examples: 8,041 classification (from 449), 2,180 sentiment (from 179), and 3,200 relevance (from 179). The expansion ratio differs by task because classification receives 2 examples per headline (with and without context) while sentiment receives 1 and relevance receives 1 positive per headline plus 2 negatives per move.

### Why This Approach is Non-Standard
Standard data augmentation in NLP typically involves surface-level transformations such as synonym replacement, random insertion/deletion, back-translation. These preserve the input's meaning while varying its form. These techniques are effective for general text classification but inappropriate for financial headline generation, where the semantic content must be specific to the verified label (a "regulatory_change" headline must describe a regulatory event, not a random paraphrase of the original text) and the style must match production data (Bloomberg/Reuters headline conventions, not academic prose).

The approach I used is closer to knowledge distillation than traditional augmentation. Claude serves as a teacher model that has deep knowledge of financial events (it knows what kind of headlines accompany earnings misses, regulatory actions, and M&A announcements) and generates training data that transfers this knowledge to the student SLM. The SLM then learns to recognize these patterns at 58x the speed and zero marginal cost. The 'temperature = 0.7' setting ensures sufficient diversity that the SLM learns robust linguistic features rather than memorizing a small set of templates.

The multi-task aspect of the augmentation is also uncommon. Most augmentation strategies target a single task. Here, each API call to Claude Haiku produces training signal for classification, sentiment, and relevance simultaneously, amortizing the generation cost across all three task heads. The cross-move negative sampling for relevance is a form of hard negative mining - the negatives are topically related (all are financial headlines about significant moves) but semantically unrelated to the specific target move, forcing the model to learn fine-grained relevance discrimination rather than a coarse "is this financial text?" classifier.

### The Class Weight Problem
The initial training run used 25 fine-grained classification categories with raw inverse-frequency class weighting. The results were instructive as a case study in what goes wrong with extreme class imbalance. With 5,944 earnings examples and 26 geopolitical examples, the raw inverse-frequency weights produced a 229:1 ratio between the rarest and most common classes. The optimizer responded rationally to this incentive structure: it learned to never predict "earnings" (the majority class), because getting one geopolitical example right was rewarded 229 times more than getting one earnings example right. Classification accuracy was 4-7%, far below the 17% that random guessing would have achieved with 6 classes.

The fix actually involved two interventions. First, the 25 fine-grained categories were consolidated to 6 semantically coherent groups via a 'LABEL_CONSOLIDATION_MAP.' Fifteen of the original categories had zero training examples; consolidation ensured every surviving category had representation while preserving the distinctions the downstream LLM needs: the LLM must know whether a move is earnings-driven, macro-driven, or geopolitically-driven, but it does not need to distinguish "fed_announcements" from "rate_decision". Those are both macro_economic events that the LLM can disambiguate from article context.

Second, the class weighting scheme was changed from raw inverse-frequency to square-root dampened inverse-frequency with a hard cap at 15:1 maximum ratio. The mathematical form is weight_i = sqrt(N / (K x n_i)) where N is total examples, K is number of classes, and n_i is the count for class i, followed by normalization to mean 1.0 and clamping to a 15:1 max ratio. Under this scheme, the earnings weight increased from 0.01 to 0.13 and the geopolitical weight decreased from 2.36 to 1.91. Classification accuracy immediately jumped from 4% to 87% in the first epoch and converged at 91% by epoch 5.

### Training Configuration and Trajectory
The final training run used the ProsusAI/FinBERT backbone (110M parameters, 81M trainable after freezing the first 4 of 12 transformer layers) with AdamW optimization and differential learning rates: 2e-6 for the pre-trained backbone (to preserve the financial pre-training), 2e-5 for the task heads (to learn task-specific mappings quickly), and 1e-5 for the learnable task uncertainty weights. A cosine annealing schedule decayed the learning rate over the full 5 epochs. Dropout of 0.3 was applied to all task heads, and label smoothing of 0.1 on the classification cross-entropy prevented overconfident majority-class predictions. Training ran on an M-series MacBook CPU at approximately 2 hours per epoch, for a total of 10 hours.

The training trajectory showed healthy convergence with no signs of overfitting. Train loss decreased from 3.14 to 2.55 while validation loss tracked closely from 2.82 to 2.51. The small and stable train-val gap indicates the model was learning generalizable patterns rather than memorizing the training set. Classification accuracy rose from 86.7% to 90.9%, with every epoch saving a new best checkpoint based on validation loss. The learnable task weights moved modestly (classification drifting to 0.98, sentiment and relevance to 1.02), reflecting the classification-dominated dataset. With a more balanced dataset, the uncertainty weighting would produce more differentiated task weights, but the modest movement here confirms that the multi-task loss is functioning correctly. It is down-weighting the task with the strongest gradient signal (classification) and up-weighting the tasks with weaker signal (sentiment, relevance).

## Evaluation Framework
### Factor Model Evaluation
The factor model evaluator uses a walk-forward protocol: for each of the 25 evaluation tickers, the OLS model is fit on [0, T-60d] and tested on [T-60d, T]. Metrics include out-of-sample R-squared, calibration curves comparing empirical exceedance frequencies against both Normal (Gaussian) and Student's t-distribution predictions at 1.5/2.0/2.5/3.0 sigma thresholds, Jarque-Bera residual normality tests, and rolling beta stability. The Student's t degrees of freedom were estimated via MLE on the pooled standardized residuals.

### SLM Evaluation
Classification: accuracy, macro-F1, weighted-F1, top-3 accuracy, per-class F1, confusion matrix. Sentiment: MAE, Spearman rank correlation, directional accuracy. Relevance: precision, recall, F1, AUC-ROC. All metrics are additionally computed as "teacher agreement" against Claude's original labels. Latency benchmarks compare single-inference and batch-100 times against estimated API equivalents.

### Causal Graph Evaluation
Adjusted Rand Index (cluster-sector alignment), silhouette score, cluster coherence, sector purity, epicenter accuracy, and same-sector edge percentage. Requires sufficient market volatility to produce >= 2 idiosyncratic moves for clustering.

### LLM-as-Judge
Following the MT-Bench and AlpacaEval paradigm, a separate Claude instance evaluated brief quality on five dimensions (1-5 scale) with chain-of-thought justification: Factual Accuracy, Causal Reasoning, Decomposition Awareness, Actionability, and Information Density. Also detects hallucinations, attribution errors, and vague recommendations. The judge receives the full brief alongside the original input data (with decomposition fields and cluster assignments) for verification.

A critical engineering lesson emerged from developing this evaluator. The initial implementation passed only the executive summary to the judge and provided move data without the decomposition fields. The judge saw moves with 0.0% idiosyncratic returns and correctly flagged the brief's real claims as "hallucinations." Resolving this required adding the decomposition fields to the PriceMove model, persisting causal cluster data in the MarketBrief output, and injecting decomposition values from a source-of-truth lookup. The lesson is that evaluation infrastructure requires the same engineering rigor as the system being evaluated.

### Counterfactual Filtering
Tests the decomposition's filtering decision by examining next-day reversion rates for filtered (systematic) versus passed (idiosyncratic) moves, and testing against 20 hardcoded known events for false negative rate. 

## Results
### Factor Decomposition
The factor model achieves an average out-of-sample R-squared of 0.415 across 25 evaluation tickers, indicating that approximately 42% of daily return variance is explained by market and sector factors. The sector-level distribution reveals an intuitive pattern: energy (R-squared = 0.67) and financials (0.63) are highly systematic, while healthcare (0.14) and technology (0.28) are dominated by idiosyncratic events -- precisely the regime where this system adds the most value.

The calibration analysis produced a genuine research finding. The normal distribution underestimates tail probabilities at all thresholds, with the underestimation becoming severe at 3 sigma (empirical 1.93% vs predicted 0.27%, a 7.16x gap). The MLE-fitted Student's t with df = 4.52 overcorrects, overestimating at all thresholds (ratios 0.57-0.74x). The empirical distribution lies between the two parametric models, with excess kurtosis of 2.79 confirming fat tails that are heavier than Gaussian but lighter than a low-df Student's t. The crossover point is around 2 sigma, and the optimal tail model would be a constrained t with df = 8-10 or a mixture distribution.

### SLM Performance
The fine-tuned FinBERT achieves 92.7% classification accuracy and 95.5% top-3 accuracy on the validation set. Sentiment prediction achieves a Spearman rank correlation of 0.857 and directional accuracy of 92.7%. Relevance detection achieves F1 of 0.898 with recall of 0.981, meaning the model catches virtually every relevant article while maintaining 83% precision. The SLM processes 100 articles in 862ms - a 58x speedup over equivalent API calls at zero marginal cost.

The gap between accuracy (92.7%) and macro-F1 (0.364) reflects a class imbalance: the model effectively learned two categories (earnings F1 = 0.96, macro_economic F1 = 0.92) while the four rare categories have near zero F1. With 26 geopolitical and 59 sector_market training examples, the model lacks sufficient signal to learn those boundaries. This is a data limitation, rather than a modeling limitation.

### Causal Graph 
The causal graph achieves a perfect ARI of 1.0, with 100% sector-pure clusters and 100% same-sector edges. Silhouette score of 0.528 indicates clear cluster separation, and coherence of 0.414 reflects genuine statistical co-movement. These results validate the partial correlation approach: by controlling for shared market exposure, the network correctly identifies within sector co-movement as the primary clustering signal.

### LLM-as-Judge
The final dimension scores are: Decomposition Awareness 5/5 (the brief never misattributes idiosyncratic moves to market conditions), Causal Reasoning 4/5 (the judge notes "appropriately skeptical causal reasoning"), Actionability 4/5, Information Density 4/5, and Factual Accuracy 3/5. The factual accuracy gap reflects residual LLM hallucinations which were fabricated ticker references and invented numerical values. The grounding instructions worked to reduce these but did not fully eliminate them. Attribution errors dropped to 0%.

### Counterfactual Filtering
The factor decomposition filters 28-37% of raw moves as systematic, with 0% false negative rate on known historical events in the best configuration. Filtered moves have average idiosyncratic sigma of 0.73 - 1.07, confirming correct identification of systematic-dominated moves.

### Full Pipeline Performance
On the evaluation date, the pipeline scanned 503 S&P 500 constituents, detected 42 raw moves, filtered 2 as systematic, retrieved 351 articles for 40 idiosyncratic moves, organized them into 3 causal clusters, and produced a 25-alert intelligence brief with 5 prioritized recommendations in 136 seconds. The SLM processed all articles in under 1 second, and the LLM synthesis required a single API call.

## Key Engineering Decisions
**FinBERT over DistilBERT:** ProsusAI/FinBERT is pre-trained on the TRC2 financial corpus and financial news, providing domain-specific token representations. General-purpose models lack financial vocabulary and contextual understanding of terms like "guidance revision" or "margin compression."

**Label Consolidation from 25 to 6 Categories:** With GDELT rate limits producing sparse data, 15 of the original 25 categories had zero examples. Consolidation preserved the analytical distinctions the LLM needs (earnings vs macro vs geopolitical) while ensuring every category had training signal.

**Sqrt-dampened Class Weights:** The raw inverse frequency weighting failure (229:1 ratio causing below-random accuracy) is a known risk in imbalanced classification. The square root dampening preserves the directional intent while preventing the extreme ratios that destabilize optimization.

**Decomposition Data Persistence:** The original architecture silently dropped factor decomposition fields when serializing alerts, creating a provenance gap that caused the LLM-as-Judge to flag correct claims as hallucinations for three evaluation rounds. The fix exemplifies the principle that evaluation infrastructure must maintain the same data integrity as the system under test.

## Honest Limitations
The SLM's macro-F1 of 0.364 reflects a model that has effectively learned two of six categories. The weighted F1 of 0.912 is genuinely strong, but it is dominated by the earnings and macro_economic classes. With production news feeds providing 10-50x more diverse examples, macro-F1 would improve substantially.

The LLM agent continues to hallucinate despite grounding instructions. The judge identifies fabricated ticker references, invented numerical values, and causal narratives beyond what the data supports. This is a fundamental limitation of autoregressive language models and argues for a post-hoc structured output validation layer.

The counterfactual reversion separation is negative in all evaluation runs (idiosyncratic moves revert more than systematic), opposite of theoretical prediction. However, sample sizes (3-7 moves per group) are too small for statistical significance. 

Neither the normal nor the Student's t-distribution provides optimal tail calibration. The normal underestimates by up to 7x and the t overestimates by up to 1.8x. The empirical distribution lies between the two models.

The ReturnPredictor module produced zero successful fits on the evaluated pipeline run due to data alignment issues, making the prediction residual step effectively a no-op. The architecture supports richer feature sets and more robust estimators, but the current OLS implementation with daily features provides limited additional signal beyond the factor model.

## Future Work
The most impactful improvement would be production-grade data feeds (Bloomberg, Refinitiv, SEC EDGAR), simultaneously resolving the SLM's rare-category data scarcity, training text quality, and news coverage reliability.

The calibration finding suggests implementing a constrained Student's t (df >= 6) or Gaussian mixture for better tail modeling. LLM hallucinations could be substantially reduced by a rule-based output validator checking every ticker and numerical value against the input data. Retrieval-Augmented Generation (RAG) with historical precedents would ground causal narratives in evidence. A streaming architecture with real-time detection would enable intraday deployment. The ReturnPredictor could be improved by switching to a nonlinear model (e.g., Random Forest or gradient boosting) and incorporating intraday features, alternative data, or options-implied volatility surfaces.

## Reproduction
### Prerequisites
Python 3.11+
pip install -r requirements.txt

### Required API keys in .env
ANTHROPIC_API_KEY=sk-ant-...
NEWSAPI_KEY=...           # Optional, improves news coverage

### Quick Start
* Run the full pipeline on S&P 500
python -m Pipeline.run_pipeline --sp500

* Run with custom tickers and lower threshold
python -m Pipeline.run_pipeline --tickers AAPL MSFT NVDA TSLA --idio-sigma 1.0

* Skip news retrieval (faster, for testing)
python -m Pipeline.run_pipeline --sp500 --skip-news

### Training
* Generate training data from historical moves
python -m SLM.generate_training_data --lookback 365 --sigma 2.0 --max-moves 500 --output training_data

* Augment with synthetic headlines
python -m SLM.augment_training_data --input-dir training_data --output-dir training_data

* Train the multi-task FinBERT
python -m SLM.train \
    --data-dir training_data \
    --output-dir models/financial_slm \
    --backbone ProsusAI/finbert \
    --epochs 5 --batch-size 16 --lr 2e-5 \
    --freeze-layers 4 --val-split 0.15

### Evaluation
* Full evaluation suite with normal distribution calibration
python -m Evaluation.run_eval --sigma 1.0 --distribution normal

* Compare with Student's t calibration
python -m Evaluation.run_eval --sigma 1.0 --distribution student_t

* Skip specific components
python -m Evaluation.run_eval --skip-factor --skip-counterfactual
