"""
Unified Evaluation Runner for Intelligence System

Runs all the evaluation components and generate a comprehensive report:
    1. Factor Model Eval - R^2, calibration, beta stability, residual normality
    2. SLM Eval - classification F1, sentiment MAE, relevance AUC ROC
    3. Causal Graph Eval - ARI, silhouette, epicenter accuracy
    4. LLM Judge - brief quality scoring (factual, causal, actionability)
    5. Counterfactual Filter - filtering rate, false negative, reversion analysis

Outputs:
    - evaluation_report.json - complete metrics
    - evaluation_report.md - human-readable summary with key findings
"""

import asyncio
import json 
import logging 
import sys
import argparse 
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Dict, List, Optional
from dataclasses import asdict 

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

class EvaluationRunner:
    """ Orchestrates all evalutions and generates reports"""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        test_data_dir: str = "training_data",
        slm_model_path: str = "models/financial_slm",
        output_dir: str = "evaluation_results",
        sigma_threshold: float = 2.0,
    ):
        self.tickers = tickers or self._default_tickers()
        self.test_data_dir = test_data_dir
        self.slm_model_path = slm_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents = True, exist_ok=True)
        self.sigma_threshold = sigma_threshold
        self.results: Dict = {}

    async def run_all(
        self,
        skip_factor: bool = False,
        skip_slm: bool = False,
        skip_causal: bool = False,
        skip_llm_judge: bool = False,
        skip_counterfactual: bool = False,
    ) -> Dict:
        """ Run all evaluation components"""
        logger.info("=" * 80)
        logger.info("Financial Intelligence - Evaluation Suite")
        logger.info("=" * 80)

        timestamp = datetime.now(timezone.utc).isoformat()
        self.results = {
            "evaluation_timestamp": timestamp,
            "tickers_evaluated": self.tickers,
            "components": {},
        }

        #1. Factor Model
        if not skip_factor:
            logger.info("\n [1/5] Evaluating Factor Decomposition Model")
            try:
                from Evaluation.factor_eval import FactorModelEvaluator
                evaluator = FactorModelEvaluator()
                factor_metrics = evaluator.evaluate(self.tickers, test_period_days = 60)
                self.results["components"]["factor_model"] = asdict(factor_metrics)
                logger.info(f" R-squared: {factor_metrics.avg_r_squared} | "
                            f"Calibration error: {factor_metrics.calibration_error} | "
                            f"Normal residuals: {factor_metrics.pct_normal_residuals}%")
            except Exception as e:
                logger.error(f" Factor eval failed: {e}")
                self.results["components"]["factor_model"] = {"error": str(e)}
        else:
            logger.info("\n[1/5] Skipping the factor model evaluation")

        #2. SLM
        if not skip_slm:
            logger.info("\n [2/5] Evaluating SLM (FinBERT multi-fask model)")
            try:
                from Evaluation.slm_eval import SLMEvaluator
                evaluator = SLMEvaluator(test_data_dir = self.test_data_dir)
                slm_metrics = evaluator.evaluate(model_path = self.slm_model_path)
                self.results["components"]["slm"] = asdict(slm_metrics)
                logger.info(
                    f" Cls F1: {slm_metrics.classification.macro_f1} | "
                    f" Sent MAE: {slm_metrics.sentiment.mae} | "
                    f" Rel AUC: {slm_metrics.relevance.auc_roc} | "
                    f" Speedup: {slm_metrics.latency.speedup_factor}x"
                )
            except Exception as e:
                logger.error(f" SLM eval failed: {e}")
                self.results["components"]["slm"] = {"error": str(e)}
        else:
            logger.info("\n [2/5] Skipping SLM evaluation")

        #3. Causal Graph
        if not skip_causal:
            logger.info("\n [3/5] Evaluating Causal Graph Clustering")
            try:
                await self._eval_causal_graph()
            except Exception as e:
                logger.error(f" Causal graph eval failed: {e}")
                self.results["components"]["causal_graph"] = {"error": str(e)}
        else:
            logger.info("\n [3/5] Skipping causal graph evaluation")

        #4. LLM Judge
        if not skip_llm_judge:
            logger.info("\n [4/5] Running LLM-as-Judge evaluation")
            try:
                await self._eval_llm_judge()
            except Exception as e:
                logger.error(f" LLM judge failed: {e}")
                self.results["components"]["llm_judge"] = {"error": str(e)}
        else:
            logger.info("\n [4/5] Skipping LLM judge evaluation")

        #5. Counterfactual Filtering
        if not skip_counterfactual:
            logger.info("\n [5/5] Evaluating Counterfactual Filtering Rate")
            try:
                await self._eval_counterfactual()
            except Exception as e:
                logger.error(f" Counterfactual eval failed: {e}")
                self.results["components"]["counterfactual"] = {"error": str(e)}
        else:
            logger.info("\n [5/5] Skipping counterfactual evaluation")

        #Generate Reports
        self._save_json_report()
        self._save_markdown_report()

        logger.info(f"\n Evlauation complete. Reports saved to {self.output_dir}/")
        return self.results
    
    #Causal Graph Evaluation

    async def _eval_causal_graph(self):
        """ Run causal graph eval by generating a graph from current data"""
        from Data.market_monitor import MarketMonitor
        from Data.data_model import Portfolio, ThresholdConfig
        from Quant.factor_decomposition import FactorDecomposer
        from Quant.causal_graph import CausalGraphBuilder
        from Evaluation.causal_graph_eval import CausalGraphEvaluator

        #Run a live scan to get the actual data
        portfolio = Portfolio(name = "Eval", tickers = self.tickers, use_sp500 = False,
                              threshold_config=ThresholdConfig(daily_sigma_threshold=self.sigma_threshold, weekly_sigma_threshold=self.sigma_threshold))
        monitor = MarketMonitor(portfolio)
        moves, _ = monitor.detect_significant_moves()

        if not moves:
            self.results["components"]["causal_graph"] = {
                "note": "No signifiant moves detecetd for evaluation"
            }
            return 
        
        decomposer = FactorDecomposer()
        significant, _ = decomposer.decompose_moves(moves)

        if len(significant) < 2:
            self.results["components"]["causal_graph"] = {
                "note": f"Only {len(significant)} idiosyncratic moves - need >= 2 for clustering"
            }
            return 
        
        builder = CausalGraphBuilder()
        graph = builder.build_graph(significant)

        evaluator = CausalGraphEvaluator()
        metrics = evaluator.evaluate_single(graph)
        self.results["components"]["causal_graph"] = asdict(metrics)

        logger.info(
            f" ARI: {metrics.adjusted_rand_index} | "
            f"Silhouette: {metrics.silhouette_score} | "
            f"Clusters: {metrics.num_clusters} | "
            f"Epicenter accuracy: {metrics.epicenter_is_max_sigma}%"
        )

    #LLM Judge Evaluation

    async def _eval_llm_judge(self):
        """ Run LLM judge on recent briefs or generate a test brief"""
        from Evaluation.llm_judge import LLMJudge

        judge = LLMJudge()

        #Check for saved briefs
        brief_files = list(Path(".").glob("market_brief_*.json"))

        if not brief_files:
            #Generate a brief to judge 
            logger.info(" No saved briefs found - running pipeline to generate one...")
            brief_data = await self._generate_test_brief()
            if not brief_data:
                self.results["components"]["llm_judge"] = {
                    "note": "Could not generate test brief"
                }
                return 
            briefs_to_judge = [brief_data]
        else:
            #Judge existing briefs
            briefs_to_judge = []
            for bf in brief_files[:5]: #Max 5 briefs
                try:
                    with open(bf) as f:
                        data = json.load(f)
                    briefs_to_judge.append({
                        "brief_text": data.get("executive_summary", ""),
                        "input_data": {
                            "moves": data.get("alerts", []),
                            "decomposed_moves": data.get("alerts", []),
                            "clusters": [],
                        },
                    })
                except Exception:
                    continue 

        if not briefs_to_judge:
            self.results["components"]["llm_judge"] = {
                "note": "No briefs available for evaluation"
            }
            return 
        
        metrics = await judge.evaluate_briefs(briefs_to_judge)
        self.results["components"]["llm_judge"] = asdict(metrics)

        logger.info(
            f" Overall: {metrics.avg_overall_score}/5 | "
            f"Factual: {metrics.avg_factual_accuracy}/5 | "
            f"Causal: {metrics.avg_causal_reasoning}/5 | "
            f"Decomp awareness: {metrics.avg_decomposition_awareness}/5 | "
            f"Hallucination rate: {metrics.pct_briefs_with_hallucinations}%"
        )

    async def _generate_test_brief(self) -> Optional[Dict]:
        """ Generate a brief for judge evaluation"""
        try:
            from Data.market_monitor import MarketMonitor
            from Data.data_model import Portfolio, ThresholdConfig
            from Quant.factor_decomposition import FactorDecomposer

            portfolio = Portfolio(name = "Eval", tickers = self.tickers[:20],
                                  use_sp500 = False, threshold_config=ThresholdConfig(daily_sigma_threshold=self.sigma_threshold, weekly_sigma_threshold=self.sigma_threshold))
            monitor = MarketMonitor(portfolio)
            moves, snapshot = monitor.detect_significant_moves()

            if not moves:
                return None 
            
            decomposer = FactorDecomposer()
            significant, _ = decomposer.decompose_moves(moves)

            return {
                "brief_text": f"Test brief with {len(significant)} idiosyncratic moves detected.",
                "input_data": {
                    "moves": [
                        {"ticker": m.ticker, "pct_change": m.total_return,
                         "idiosyncratic_return": m.idiosyncratic_return,
                         "idiosyncratic_sigma": m.idiosyncratic_sigma,
                         "alert_level": m.alert_level.value}
                         for m in significant
                    ],
                    "decomposed_moves": [
                        {"ticker": m.ticker, "total_return": m.total_return,
                         "market_component": m.market_component,
                         "sector_component": m.sector_component,
                         "idiosyncratic_return": m.idiosyncratic_return,
                         "r_squared": m.r_squared}
                         for m in significant
                    ],
                    "clusters": [],
                },
            }
        except Exception as e:
            logger.warning(f"Could not generate test brief: {e}")
            return None 
        
    #--Counterfactual Evaluation --#

    async def _eval_counterfactual(self):
        """ Run counterfactual filtering evaluation"""
        from Evaluation.counterfactual_eval import CounterfactualEvaluator
        from Data.market_monitor import MarketMonitor
        from Data.data_model import Portfolio, ThresholdConfig
        from Quant.factor_decomposition import FactorDecomposer

        #Run live scan 
        portfolio = Portfolio(name = "Eval", tickers = self.tickers,
                              use_sp500=False, threshold_config=ThresholdConfig(daily_sigma_threshold=self.sigma_threshold, weekly_sigma_threshold=self.sigma_threshold))
        monitor = MarketMonitor(portfolio)
        moves, _ = monitor.detect_significant_moves()

        if not moves:
            self.results["components"]["counterfactual"] = {
                "note": "No moves detected for counterfactual analysis"
            }
            return 
        
        decomposer = FactorDecomposer()
        significant, systematic = decomposer.decompose_moves(moves)

        evaluator = CounterfactualEvaluator()
        metrics = evaluator.evaluate(significant, systematic)
        self.results["components"]["counterfactual"] = asdict(metrics)

        logger.info(
            f"  Filtering rate: {metrics.filtering_rate.filtering_rate_pct}% | "
            f"Quality score: {metrics.filter_quality_score} | "
            f"Reversion separation: {metrics.next_day_reversion.reversion_separation}pp | "
            f"False negatives: {metrics.event_contamination.false_negative_rate}%"
        )

    #-- Report Generation --#

    def _save_json_report(self):
        filepath = self.output_dir / "evaluation_report.json"
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent = 2, default = str)
        logger.info(f"JSON report: {filepath}")

    def _save_markdown_report(self):
        filepath = self.output_dir / "evaluation_report.md"
        r = self.results
        components = r.get("components", {})

        lines = [
            "# Financial Intelligence - Evaluation Report",
            f"\n ** Generated **: {r.get('evaluation_timestamp', 'N/A')}",
            f" ** Tickers evaluated**: {len(r.get('tickers_evaluated', []))}",
            "",
        ]

        # -- Factor Model --#
        fm = components.get("factor_model", {})
        if "error" not in fm and fm:
            lines.extend([
                "##1. Factor Decomposition Model", 
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Avg R^2 (out-of-sample) | {fm.get('avg_r_squared', 'N/A')} |",
                f"| Median R^2 | {fm.get('median_r_squared', 'N/A')} |",
                f"| Calibration error | {fm.get('calibration_error', 'N/A')} |",
                f"| Residuals passing normality test | {fm.get('pct_normal_residuals', 'N/A')}% |",
                f"| Avg excess kurtosis | {fm.get('avg_excess_kurtosis', 'N/A')} |",
                f"| Avg beta stability (sigma) | {fm.get('avg_beta_std', 'N/A')} |",
                "",
            ])

            #Calibration table
            cal = fm.get("calibration_curve", {})
            if cal:
                # Extract distribution fit info
                dist_fit = cal.get("_distribution_fit", {})
                if dist_fit:
                    lines.extend([
                        f"**Distribution Fit**: Student's t with df={dist_fit.get('student_t_df', 'N/A')} "
                        f"(Normal calibration error: {dist_fit.get('avg_calibration_error_normal', 'N/A')}, "
                        f"t-distribution error: {dist_fit.get('avg_calibration_error_t', 'N/A')}, "
                        f"improvement: {dist_fit.get('improvement_pct', 'N/A')}%)",
                        "",
                    ])

                lines.extend([
                    "**Calibration Curve (Normal vs Student's t)**:",
                    "",
                    "| σ Threshold | Expected (Normal) | Expected (t) | Empirical | Ratio (Normal) | Ratio (t) |",
                    "|-------------|-------------------|--------------|-----------|----------------|-----------|",
                ])
                for key in sorted(k for k in cal.keys() if not k.startswith("_")):
                    c = cal[key]
                    lines.append(
                        f"| {c.get('threshold_sigma', key)}σ | "
                        f"{c.get('expected_frequency_normal', c.get('expected_frequency', 'N/A'))} | "
                        f"{c.get('expected_frequency_t', 'N/A')} | "
                        f"{c.get('empirical_frequency', 'N/A')} | "
                        f"{c.get('ratio_vs_normal', c.get('ratio', 'N/A'))}x | "
                        f"{c.get('ratio_vs_t', 'N/A')}x |"
                    )
                lines.append("")

        # -- SLM -- #
        slm = components.get("slm", {})
        if "error" not in slm and slm:
            cls = slm.get("classification", {})
            sent = slm.get("sentiment", {})
            rel = slm.get("relevance", {})
            lat = slm.get("latency", {})
            ta = slm.get("teacher_agreement", {})

            lines.extend([
                "## 2. SLM (Fine-tuned FinBERT)",
                "",
                "| Task | Metric | Value |",
                "|------|--------|-------|",
                f"| Classification | Accuracy | {cls.get('accuracy', 'N/A')} |",
                f"| Classification | Macro-F1 | {cls.get('macro_f1', 'N/A')} |",
                f"| Classification | Top-3 Accuracy | {cls.get('top_3_accuracy', 'N/A')} |",
                f"| Sentiment | MAE | {sent.get('mae', 'N/A')} |",
                f"| Sentiment | Spearman ρ | {sent.get('spearman_correlation', 'N/A')} |",
                f"| Sentiment | Directional Accuracy | {sent.get('directional_accuracy', 'N/A')} |",
                f"| Relevance | Precision | {rel.get('precision', 'N/A')} |",
                f"| Relevance | Recall | {rel.get('recall', 'N/A')} |",
                f"| Relevance | F1 | {rel.get('f1', 'N/A')} |",
                f"| Relevance | AUC-ROC | {rel.get('auc_roc', 'N/A')} |",
                "",
                f"**Teacher Agreement**: cls={ta.get('classification_agreement', 'N/A')}, "
                f"cls_top3={ta.get('classification_top3_agreement', 'N/A')}, "
                f"sentiment_rank={ta.get('sentiment_rank_agreement', 'N/A')}, "
                f"relevance={ta.get('relevance_agreement', 'N/A')}",
                "",
                f"**Latency**: single={lat.get('slm_single_ms', 'N/A')}ms, "
                f"batch100={lat.get('slm_batch_100_ms', 'N/A')}ms, "
                f"**{lat.get('speedup_factor', 'N/A')}x faster** than API",
                "",
            ])

            #Most confused pairs
            confused = cls.get("more_confused_pairs", [])
            if confused:
                lines.extend([
                    "**Most Confused Category Pairs**:",
                    "",
                    "| True | Predicted | Count |",
                    "|------|-----------|-------|",
                ])
                for true, pred, count in confused[:5]:
                    lines.append(f"| {true} | {pred} | {count} |")
                lines.append("")

        #-- Causal Graph --#
        cg = components.get("causal_graph", {})
        if "error" not in cg and "note" not in cg and cg:
            lines.extend([
                "## 3. Causal Graph Clustering",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Adjusted Rand Index | {cg.get('adjusted_rand_index', 'N/A')} |",
                f"| Silhouette Score | {cg.get('silhouette_score', 'N/A')} |",
                f"| Avg Cluster Coherence | {cg.get('avg_cluster_coherence', 'N/A')} |",
                f"| Clusters | {cg.get('num_clusters', 'N/A')} ({cg.get('num_multi_move', 0)} multi, {cg.get('num_singletons', 0)} singletons) |",
                f"| Sector-Pure Clusters | {cg.get('pct_sector_pure_clusters', 'N/A')}% |",
                f"| Epicenter = Max σ | {cg.get('epicenter_is_max_sigma', 'N/A')}% |",
                f"| Edges Same-Sector | {cg.get('pct_edges_same_sector', 'N/A')}% |",
                "",
            ])

        # -- LLM Judge -- #
        lj = components.get("llm_judge", {})
        if "error" not in lj and "note" not in lj and lj:
            lines.extend([
                "## 4. LLM-as-Judge (Brief Quality)",
                "",
                "| Dimension | Avg Score (1-5) | Std |",
                "|-----------|-----------------|-----|",
                f"| Factual Accuracy | {lj.get('avg_factual_accuracy', 'N/A')} | {lj.get('std_factual_accuracy', 'N/A')} |",
                f"| Causal Reasoning | {lj.get('avg_causal_reasoning', 'N/A')} | {lj.get('std_causal_reasoning', 'N/A')} |",
                f"| Decomposition Awareness | {lj.get('avg_decomposition_awareness', 'N/A')} | — |",
                f"| Actionability | {lj.get('avg_actionability', 'N/A')} | — |",
                f"| Information Density | {lj.get('avg_information_density', 'N/A')} | — |",
                f"| **Overall** | **{lj.get('avg_overall_score', 'N/A')}** | {lj.get('std_overall_score', 'N/A')} |",
                "",
                f"Hallucination rate: {lj.get('pct_briefs_with_hallucinations', 'N/A')}% of briefs",
                f"Attribution error rate: {lj.get('pct_briefs_with_attribution_errors', 'N/A')}% of briefs",
                f"Avg vague recommendations: {lj.get('avg_vague_recommendations_per_brief', 'N/A')} per brief",
                "",
            ])

        #-- Counterfactual --#
        cf = components.get("counterfactual", {})
        if "error" not in cf and "note" not in cf and cf:
            fr = cf.get("filtering_rate", {})
            ndr = cf.get("next_day_reversion", {})
            ec = cf.get("event_contamination", {})

            lines.extend([
                "## 5. Counterfactual Filtering Rate",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Raw moves flagged | {fr.get('total_raw_moves', 'N/A')} |",
                f"| Passed (idiosyncratic) | {fr.get('passed_idiosyncratic', 'N/A')} |",
                f"| Filtered (systematic) | {fr.get('filtered_systematic', 'N/A')} |",
                f"| **Filtering rate** | **{fr.get('filtering_rate_pct', 'N/A')}%** |",
                f"| Avg filtered idio σ | {fr.get('avg_filtered_idio_sigma', 'N/A')} |",
                "",
                "**Next-Day Reversion Analysis** (validates decomposition):",
                "",
                "| Category | Reversion Rate | SPY Correlation |",
                "|----------|---------------|-----------------|",
                f"| Systematic (filtered) | {ndr.get('systematic_reversion_rate', 'N/A')}% | {ndr.get('systematic_avg_nextday_corr_with_spy', 'N/A')} |",
                f"| Idiosyncratic (passed) | {ndr.get('idiosyncratic_reversion_rate', 'N/A')}% | {ndr.get('idiosyncratic_avg_nextday_corr_with_spy', 'N/A')} |",
                f"| **Separation** | **{ndr.get('reversion_separation', 'N/A')}pp** | |",
                "",
                f"Event contamination: {ec.get('false_negative_rate', 'N/A')}% false negative rate "
                f"({ec.get('incorrectly_filtered', 0)}/{ec.get('n_known_events_tested', 0)} events)",
                "",
                f"**Filter Quality Score: {cf.get('filter_quality_score', 'N/A')}/1.0**",
                "",
            ])

            if ec.get("false_negative_tickers"):
                lines.append("False negatives:" + ", ".join(ec["false_negative_tickers"]))
                lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown report: {filepath}")

    @staticmethod
    def _default_tickers() -> List[str]:
        """ Representative tickers across sectors for evaluation"""
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", #tech
            "JPM", "GS", "MS", "BAC", #financials
            "XOM", "CVX", "SLB", #energy
            "JNJ", "UNH", "PFE", #healthcare
            "BA", "CAT", "GE", #industrials
            "WMT", "COST", "HD", #consumer
            "NEE", "DUK", #utilities
        ]
    
async def main():
    parser = argparse.ArgumentParser(
        description="Run GeoFinancial Intelligence evaluation suite",
    )
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to evaluate")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--skip-factor", action="store_true")
    parser.add_argument("--skip-slm", action="store_true")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--sigma", type=float, default = 2.0,
                        help = "σ threshold for move detection in eval (lower = more moves detected, default: 2.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    runner = EvaluationRunner(
        tickers=args.tickers,
        output_dir=args.output,
        sigma_threshold=args.sigma,
    )

    results = await runner.run_all(
        skip_factor=args.skip_factor,
        skip_slm=args.skip_slm,
        skip_causal=args.skip_causal,
        skip_llm_judge=args.skip_judge,
        skip_counterfactual=args.skip_counterfactual,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)

    for component, data in results.get("components", {}).items():
        if isinstance(data, dict) and "error" not in data and "note" not in data:
            print(f"\n  {component}:")
            # Print key metrics based on component type
            if component == "factor_model":
                print(f"    R² = {data.get('avg_r_squared', 'N/A')} | "
                      f"Calibration err = {data.get('calibration_error', 'N/A')}")
            elif component == "slm":
                cls = data.get("classification", {})
                print(f"    Cls F1 = {cls.get('macro_f1', 'N/A')} | "
                      f"Rel AUC = {data.get('relevance', {}).get('auc_roc', 'N/A')}")
            elif component == "causal_graph":
                print(f"    ARI = {data.get('adjusted_rand_index', 'N/A')} | "
                      f"Silhouette = {data.get('silhouette_score', 'N/A')}")
            elif component == "llm_judge":
                print(f"    Overall = {data.get('avg_overall_score', 'N/A')}/5 | "
                      f"Hallucinations = {data.get('pct_briefs_with_hallucinations', 'N/A')}%")
            elif component == "counterfactual":
                print(f"    Filter quality = {data.get('filter_quality_score', 'N/A')}/1.0 | "
                      f"Filtering rate = {data.get('filtering_rate', {}).get('filtering_rate_pct', 'N/A')}%")

    print(f"\n  Reports saved to: {args.output}/")


if __name__ == "__main__":
    asyncio.run(main())
