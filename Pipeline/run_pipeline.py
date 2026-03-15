"""
Pipeline Runner for Intelligence System
Orchestrates the entire workflow

1. Market Monitor -> detects moves exceeding volatility-adjusted thresholds
2. Factor Decomposition -> separate systematic vs idiosyncratic components
3. Prediction Residual -> flag only what the model cannot explain 
4. News Retrieval -> fetch news only for genuinely surprising moves
5. Causal Graph -> cluster related moves by shared drivers
6. SLM Agent -> classify, score sentiment/relevance per cluster
7. LLM Agent -> synthesize, assess impact, generate brief

The key insight: Steps 2-3 filter OUT moves that are just beta to the market,
and step 5 groups related moves so the LLM reasons about specific CAUSES not just TICKERS
"""

import asyncio 
import logging 
import sys
import argparse
from datetime import datetime, timezone 
from typing import List, Optional
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data.market_monitor import MarketMonitor
from Data.news_retriever import NewsRetriever
from Data.data_model import (
    Portfolio, ThresholdConfig, PriceMove, MarketSnapshot,
    MarketBrief, TickerNewsBundle,
)

from Quant.factor_decomposition import FactorDecomposer, ReturnPredictor, DecomposedMove
from Quant.causal_graph import CausalGraphBuilder, CausalGraph
from Pipeline.intelligence_workflow import run_intelligence_pipeline

from Storage.sqlite_store import BriefDatabase

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)

class FinPipeline:
    """ Main pipeline orchestrator"""

    def __init__(self, portfolio: Optional[Portfolio] = None):
        self.portfolio = portfolio
        self.monitor = MarketMonitor(portfolio)
        self.news_retriever = NewsRetriever()
        self.decomposer = FactorDecomposer()
        self.predictor = ReturnPredictor()
        self.graph_builder = CausalGraphBuilder() 

    async def run(
        self,
        news_lookback_hours: int = 72,
        max_news_per_ticker: int = 20,
        skip_news: bool = False,
        idio_sigma_threshold: float = 1.5,
    ) -> Optional[MarketBrief]:
        """
        Full pipeline execution:
        1. Scan the portfolio for significant moves
        2. Factor decomposition -> isolate idiosyncratic residuals
        3. Prediction residual -> flag only genuinely surprising moves
        4. Retrieve news for significant idiosyncratic moves 
        5. Build causal graph -> cluster related moves 
        6. Run SLM -> LLM intelligence pipeline on clusters
        """

        logger.info("=" * 70)
        logger.info("Financial Intelligence Pipeline")
        logger.info("=" * 70)

        #Step 1: Market Scan
        logger.info("\n [1/6] Scanning the portfolio for significant moves ...")
        moves, snapshot = self.monitor.detect_significant_moves()

        if not moves:
            logger.info("No significant moves detected. Portfolio is quiet.")
            self._display_snapshot(snapshot)
            return None 
        
        logger.info(f" {len(moves)} raw significant moves detected")

        #Step 2: Factor Decomposition
        logger.info(f"\n [2/6] Decomposing moves into market/sector/idiosyncratic components...")
        significant, systematic = self.decomposer.decompose_moves(
            moves, idiosyncratic_sigma_threshold = idio_sigma_threshold,
        )

        logger.info(
            f"Factor decomposition: {len(significant)} genuine idiosyncratic moves, "
            f"{len(systematic)} filtered (explained by market/sector beta)"
        )
        self._display_decomposition(significant, systematic, snapshot)

        if not significant:
            logger.info("All moves explained by market/sector factors. Nothing idiosyncratic")
            return None 
        
        #Step 3: Prediction Residual
        logger.info(f"[3/6] Computing prediction residuals (what can't the model explain?) ...")
        original_moves = [m.original_move for m in significant if m.original_move]
        prediction_results = self.predictor.fit_and_predict(original_moves)

        logger.info(
            f"Prediction model fitted: {len(prediction_results)} tickers | "
            f"Average residual sigma: {self._avg_pred_sigma(prediction_results):.1f}"
        )

        #Step 4: News Retrieval
        news_bundles: List[TickerNewsBundle] = []
        if not skip_news:
            logger.info(f"\n[4/6] Retrieving news for {len(significant)} idiosyncratic moves")
            #Only retrieves news for idiosyncratic moves (not systematic moves)
            news_moves = [m.original_move for m in significant if m.original_move]
            try:
                news_bundles = await self.news_retriever.retrieve_news_for_moves(
                    news_moves,
                    lookback_hours = news_lookback_hours,
                    max_per_ticker=max_news_per_ticker,
                )
                total_articles = sum(b.article_count for b in news_bundles)
                logger.info(f"Retrieved {total_articles} articles for {len(news_bundles)} tickers")
            except Exception as e:
                logger.error(f"News retrieval failed: {e}")
        else:
            logger.info("\n [4/6] Skipping the news retrieval (--skip-news)")

        #Step 5: Causal Graph 
        logger.info(f"\n [5/6] Building causal grpah (clustering related moves) ...")
        causal_graph = self.graph_builder.build_graph(significant)
        self._display_causal_graph(causal_graph)

        #Step 6: Intelligence Pipeline (SLM + LLM)
        logger.info(f"\n [6/6] Running SLM -> LLM intelligence pipeline on {causal_graph.num_clusters} clusters...")
        try:
            brief = await run_intelligence_pipeline(
                decomposed_moves=significant,
                causal_graph=causal_graph,
                news_bundles=news_bundles,
                market_snapshot=snapshot,
                prediction_results=prediction_results,
                systematic_moves=systematic,
                portfolio_name=self.monitor.portfolio.name,
                tickers_monitored=len(self.monitor.tickers),
            )
        except Exception as e:
            logger.error(f"Intelligence pipeline failed: {e}", exc_info = True)
            return None 
        
        if brief:
            self._display_brief(brief)
            self._save_brief(brief)
            logger.info("Pipeline Completed!")
        else:
            logger.error("Pipeline failed to generate brief")

        await self.news_retriever.close()
        return brief 
    
    #Display Methods
    def _display_snapshot(self, snapshot: MarketSnapshot):
        print(f"\n Market: SPY {snapshot.spy_daily_return or 'N/A'}% | "
              f"QQQ {snapshot.qqq_daily_return or 'N/A'}% | "
              f"VIX {snapshot.vix_level or 'N/A'}")
        
    def _display_decomposition(
            self, significant: List[DecomposedMove],
            systematic: List[DecomposedMove], snapshot: MarketSnapshot,
    ):
        self._display_snapshot(snapshot)
        if systematic:
            print(f"\n Filtered ({len(systematic)} explained by market/sector):")
            for m in systematic[:5]:
                print(
                    f" {m.ticker}: {m.total_return:+.1f}% total -> "
                    f"market {m.market_component:+.1f}% + sector {m.sector_component:+.1f}% "
                    f"+ idio {m.idiosyncratic_return:+.1f}% ({m.idiosyncratic_sigma:.1f} sigma)"
                )
            if len(systematic) > 5:
                print(f" ... and {len(systematic) - 5} more")

        if significant:
            print(f"\n Significant idiosyncratic moves ({len(significant)}): ")
            for m in significant[:10]:
                arrow = "🟢" if m.idiosyncratic_return > 0 else "🔴"
                print(
                    f" {arrow} {m.ticker}: {m.total_return:+.1f}% total -> "
                    f"market {m.market_component:+.1f}% + sector {m.sector_component:+.1f}% "
                    f"+ IDIO {m.idiosyncratic_return:+.1f}% ({m.idiosyncratic_sigma:.1f} sigma) "
                    f"[{m.alert_level.value.upper()}] | R-squared = {m.r_squared:.2f}"
                )

    def _display_causal_graph(self, graph: CausalGraph):
        print(f"\n Causal graph: {graph.num_clusters} clusters "
              f"({graph.num_multi_move_clusters} multi-move, {graph.num_singletons} singletons)")
        
        for cluster in graph.clusters:
            if cluster.is_singleton:
                m = cluster.moves[0] if cluster.moves else None 
                if m:
                    print(f" [Singleton] {m.ticker}: idio {m.idiosyncratic_return:+.1f}%")
            else:
                print(
                    f" [Cluster {cluster.cluster_id}] "
                    f"{', '.join(cluster.tickers)} | "
                    f"epicenter: {cluster.epicenter_ticker} "
                    f"({cluster.epicenter_idio_return:+.1f}% | "
                    f"sector: {cluster.dominant_sector or 'mixed'} | "
                    f"coherence: {cluster.coherence_score:.2f}"
                )

    def _display_brief(self, brief: MarketBrief):
        print(f"\n {'=' * 80}")
        print(f" Market Intelligence Brief")
        print(f"{'='*80}")
        print(f"\n {brief.date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f" {brief.portfolio_name} : {brief.tickers_monitored} monitored -> "
              f"{brief.tickers_flagged} flagged")
        print(f" Generated in: {brief.generation_time_seconds}s")
        print(f"\n Executive Summary \n{'-' * 80}")
        print(brief.executive_summary[:800])

        if brief.critical_alerts:
            print(f"\n Critical Alerts ({len(brief.critical_alerts)})")
            for alert in brief.critical_alerts[:5]:
                print(f" [{alert.alert_level.value.upper()}] {alert.title}")
                print(f" {alert.summary[:200]}")

        if brief.top_recommendations:
            print(f"\n Top Recommendations")
            for i, rec in enumerate(brief.top_recommendations[:5], 1):
                print(f" {i}. [{rec.urgency.upper()}] {rec.action_type}: {rec.target}")
                print(f" {rec.rationale[:150]}")

    def _save_brief(self, brief: MarketBrief):
        import json
        filename = f"market_brief_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(brief.model_dump(), f, indent = 2, default = str)
        logger.info(f" Brief saved to: {filename}")

        from Storage.sqlite_store import BriefDatabase
        try:
            db = BriefDatabase()
            brief_id = db.store_brief(brief)
            logger.info(f" Brief stored in SQLite: {db.db_path} (id={brief_id})")
        except Exception as e:
            logger.warning(f" Database store failed (brief JSON still is saved): {e}")

    @staticmethod
    def _avg_pred_sigma(results: dict) -> float:
        if not results:
            return 0.0
        return sum(r["residual_sigma"] for r in results.values()) / len(results)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Financial Intelligence — Quantitative anomaly detection + AI explanation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Pipeline.run_pipeline --sp500
  python -m Pipeline.run_pipeline --tickers AAPL MSFT NVDA TSLA AMZN
  python -m Pipeline.run_pipeline --sp500 --daily-sigma 1.5 --idio-sigma 1.0
  python -m Pipeline.run_pipeline --sp500 --skip-news
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sp500", action="store_true", help="Monitor S&P 500")
    group.add_argument("--tickers", nargs="+", help="Custom ticker list")

    parser.add_argument("--name", default=None, help="Portfolio name")
    parser.add_argument("--daily-sigma", type=float, default=2.0)
    parser.add_argument("--weekly-sigma", type=float, default=2.0)
    parser.add_argument("--idio-sigma", type=float, default=1.5,
                        help="Idiosyncratic σ threshold after factor decomposition (default: 1.5)")
    parser.add_argument("--news-lookback", type=int, default=72)
    parser.add_argument("--max-news", type=int, default=20)
    parser.add_argument("--skip-news", action="store_true")

    return parser.parse_args()


async def main():
    args = parse_args()

    threshold = ThresholdConfig(
        daily_sigma_threshold=args.daily_sigma,
        weekly_sigma_threshold=args.weekly_sigma,
    )

    if args.sp500:
        portfolio = Portfolio(name=args.name or "S&P 500", use_sp500=True, threshold_config=threshold)
    else:
        portfolio = Portfolio(name=args.name or "Custom Portfolio", tickers=args.tickers,
                              use_sp500=False, threshold_config=threshold)

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Financial Intelligence System                                   ║
    ║  Quant Decomposition → Causal Graph → SLM → LLM Brief            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    print(f"  Portfolio: {portfolio.name}")
    print(f"  Thresholds: raw={args.daily_sigma}σ, idiosyncratic={args.idio_sigma}σ")

    pipeline = FinPipeline(portfolio)
    try:
        await pipeline.run(
            news_lookback_hours=args.news_lookback,
            max_news_per_ticker=args.max_news,
            skip_news=args.skip_news,
            idio_sigma_threshold=args.idio_sigma,
        )
    except KeyboardInterrupt:
        logger.info("\n Interrupted")
    except Exception as e:
        logger.error(f" Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
