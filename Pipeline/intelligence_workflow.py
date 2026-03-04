"""
LangGraph Intelligence Pipeline 

Two Agent Architecture:
    Agent 1 - SLM Agent (local, fast):
        Fine-tuned FinBERT handles classification, sentiment, and relevance
        scoring for all articles in a batch. Incurs zero API cost, should have
        millisecond latency. Falls back to calling Haiku if the SLM model is not 
        yet trained 

    Agent 2 - LLM Agent (Claude Sonnet, deep reasoning):
        Takes SLM-enriched data and performs an impact assessment, cross-move
        synthesis, strategic recommendations, and brief generation

Pipeline: SLM Agent -> LLM Agent -> Output
"""

import logging
import asyncio 
from datetime import datetime, timezone 
from typing import List, Dict, Any, Optional, TypedDict, Annotated, TYPE_CHECKING
import operator 
import json 

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, SecretStr 

from Data.data_model import (
    PriceMove, NewsArticle, TickerNewsBundle, MarketSnapshot,
    MoveAlert, MarketBrief, RootCauseAnalysis, SectorImpact,
    ActionRecommendation, MarketImpactCategory, MoveDirection,
    AlertLevel, Sector, MovePeriod,
)

from Data.settings import Settings
from SLM.model import SLMInference, SLMOutput, CLASSIFICATION_LABELS

if TYPE_CHECKING:
    from Quant.factor_decomposition import DecomposedMove
    from Quant.causal_graph import CausalGraph

logger = logging.getLogger(__name__)
settings = Settings() 

#-- LLM Client (Agent 2 only, Agent 1 only uses local SLM) -- #

primary_llm = ChatAnthropic(
    model_name = settings.PRIMARY_MODEL,
    api_key = SecretStr(settings.ANTHROPIC_API_KEY),
    max_tokens_to_sample = 4096,
    temperature = 0.2,
    timeout = None,
    stop = None,
)

#Fallback to Haiku for when SLM is not available
fallback_llm = ChatAnthropic(
    model_name = settings.FAST_MODEL,
    api_key = SecretStr(settings.ANTHROPIC_API_KEY),
    max_tokens_to_sample = 2048,
    temperature = 0.1,
    timeout = None,
    stop = None,
)

# -- SLM Initialization -- #

_slm: Optional[SLMInference] = None 
_slm_available: bool = False 

def get_slm() -> Optional[SLMInference]:
    """ Lazy load the SLM model. Returns None if not yet trained"""
    global _slm, _slm_available
    if _slm is not None:
        return _slm if _slm_available else None 
    
    try:
        _slm = SLMInference(model_path = "models/financial_slm")
        _slm_available = True 
        logger.info("SLM loaded - using local model for classification/sentiment/relevance")
        return _slm 
    except Exception as e:
        logger.info(f"SLM not available ({e}) - falling back to using Claude Haiku")
        _slm_available = False 
        return None 
    
# -- Pipeline State --#

class PipelineState(TypedDict):
    """ State passed between the two agents"""
    #Inputs
    moves: List[Dict[str, Any]] #Decomposed moves (idiosyncrantic only)
    news_bundles: List[Dict[str, Any]]
    market_snapshot: Dict[str, Any]
    portfolio_name: str
    tickers_monitored: int 
    clusters: List[Dict[str, Any]] #Causal graph clusters
    systematic_summary: List[Dict[str, Any]] #Moves explained by factors (for context)

    #After Agent 1 (SLM)
    enriched_moves: Annotated[List[Dict[str, Any]], operator.add]

    #After Agetn 2 (LLM)
    alerts: Annotated[List[Dict[str, Any]], operator.add]
    brief: Optional[Dict[str, Any]]

#-- Structured Outputs for the LLM Agent --#

class ImpactAndBrief(BaseModel):
    """ Combined impact assessment + brief from the LLM agent"""

    class MoveImpact(BaseModel):
        ticker: str 
        alert_title: str = Field(description = "Concise alert title (max 100 chars)")
        alert_summary: str = Field(description = "3-5 sentence summary of the move, cause, and implications")
        sector_impacts: List[Dict[str,str]] = Field(
            default_factory = list,
            description = "List of {sector, impact_summary, direction, magnitude}"
        )
        recommended_actions: List[Dict[str,str]] = Field(
            default_factory = list,
            description = "List of {action_type, target, urgency, rationale, time_horizon}"
        )
        related_moves: List[str] = Field(default_factory = list)
        forward_looking: str = ""

    #Per move assessments
    move_impacts: List[MoveImpact] = Field(description = "Impact assessment for each move")

    #Synthesized brief
    executive_summary: str = Field(description = "3-5 sentence executive summary")
    sector_summaries: Dict[str, str] = Field(description = "Per-sector summary")
    top_recommendations: List[Dict[str, str]] = Field(
        description = "Top 5 action recommendations across all moves"
    )
    risk_watchlist: str = Field(description = "Key risks to monitor")

#Fallback structured output when SLM is unavailable (For when Agent 1 uses Haiku)
class HaikuClassification(BaseModel):
    ticker: str
    primary_category: str 
    secondary_categories: List[str] = Field(default_factory = list)
    is_company_specific: bool = True 
    explanation: str = ""
    confidence: float = 0.5
    article_sentiments: List[float] = Field(default_factory = list, description = "Sentiment per article, -1 to 1")
    article_relevances: List[float] = Field(default_factory = list, description = "Relevance per article, 0 to 1")

# AGENT 1: SLM AGENT (Fast, Local)
# Handles: Classification, Sentiment, Relevance

async def slm_agent(state: PipelineState) -> Dict: 
    """
    Agent 1 - SLM Agent

    Uses the fine-tuned multi-task SLM for:
    1. Root cause classification (25 categories) per ticker
    2. Financial sentiment scoring per article
    3. Relevance/causality scoring per article 

    False back to Claude Haiku if SLM is not yet trained.
    All outputs are structured identically regardless of backend
    """
    logger.info("=" * 60)
    logger.info("[Agent 1: SLM] Classification + Sentiment + Relevance")
    logger.info("=" * 60)

    moves = state["moves"]
    news_bundles = state["news_bundles"]
    snapshot = state["market_snapshot"]

    if not moves:
        return {"enriched_moves": []}
    
    slm = get_slm() 

    if slm:
        enriched = await _slm_process(slm, moves, news_bundles, snapshot)
    else:
        enriched = await _haiku_fallback_process(moves, news_bundles, snapshot)

    logger.info(f"[Agent 1] Enriched {len(enriched)} moves")
    return {"enriched_moves": enriched}

async def _slm_process(
        slm: SLMInference,
        moves: List[Dict], news_bundles: List[Dict], snapshot: Dict,
) -> List[Dict]:
    """ Process all the moves using local SLM - batch inference, zero API costs"""
    logger.info("[SLM] Running local inference ... ")

    #Group moves by sector for context
    sector_moves: Dict[str, int] = {}
    for m in moves:
        s = m.get("sector", "unknown") or "unknown"
        sector_moves[s] = sector_moves.get(s, 0) + 1

    enriched = []
    bundle_lookup = {b["ticker"]: b for b in news_bundles}

    for move in moves:
        ticker = move["ticker"]
        bundle = bundle_lookup.get(ticker, {})
        articles = bundle.get("articles", [])

        #-- Classification: build the move description text -- #
        move_text = (
            f"{move.get('company_name', ticker)} ({ticker}) stock moved"
            f"{move['pct_change']}% {move['direction']}. "
            f"Magnitude: {move['move_in_sigma']} standard deviations. "
            f"Sector: {move.get('sector', 'unknown')}"
        )
        if articles:
            top_titles = " | ".join(a.get("title", "") for a in articles[:5])
            move_text += f" Related news: {top_titles}"

        cls_output = slm.analyze(move_text)

        #-- Sentiment + Relevance: Scores each article --#
        scored_articles = []
        if articles:
            article_texts = []
            for art in articles:
                #For relevance, prepend move context
                rel_text = (
                    f"[Move: {ticker} {move['pct_change']}% {move['direction']}]"
                    f"{art.get('title', '')} {art.get('description', '')[:200]}"
                )
                article_texts.append(rel_text)

            #Batch inference for articles
            if article_texts:
                article_outputs = slm.analyze_batch(article_texts)

                for art, output in zip(articles, article_outputs):
                    scored_articles.append({
                        **art,
                        "slm_sentiment": output.sentiment_score,
                        "slm_relevance": output.relevance_score,
                        "slm_is_relevant": output.is_relevant,
                    })
        #Filter to relevant articles only
        relevant_articles = [a for a in scored_articles if a.get("slm_is_relevant", False)]
        avg_sentiment = (
            sum(a["slm_sentiment"] for a in scored_articles) / len(scored_articles)
            if scored_articles else 0.0
        )

        enriched.append({
            **move,
            #SLM classification
            "primary_cause": cls_output.predicted_category,
            "classification_confidence": cls_output.classification_confidence,
            "top_3_categories": sorted(
                cls_output.category_probabilities.items(),
                key = lambda x: x[1], reverse = True,
            )[:3],
            #SLM sentiment
            "avg_article_sentiment": round(avg_sentiment, 3),
            "article_sentiments": [a.get("slm_sentiment", 0) for a in scored_articles],
            #SLM relevance
            "relevant_article_count": len(relevant_articles),
            "total_article_count": len(articles),
            "relevant_articles": relevant_articles[:10], #top 10 relevant
            "all_scored_articles": scored_articles,
            #Context
            "sector_move_count": sector_moves.get(move.get("sector", ""), 0),
            "is_broad_market_move": (
                snapshot.get("spy_daily_return") is not None
                and abs(snapshot.get("spy_daily_return", 0)) > 1.5
            ),
            #Preprocessing metadata
            "processed_by": "slm",
        })

    logger.info(f"[SLM] Processed {len(enriched)} moves locally")
    return enriched 

async def _haiku_fallback_process(
        moves: List[Dict], news_bundles: List[Dict], snapshot: Dict,
) -> List[Dict]:
    """ Fallback: use Claude Haiku for classification when SLM not available"""
    logger.info("[Haiku Fallback] SLM not available, using Haiku...")

    classifier = fallback_llm.with_structured_output(HaikuClassification)
    bundle_lookup = {b["ticker"]: b for b in news_bundles}

    sector_moves: Dict[str, int] = {}
    for m in moves:
        s = m.get("sector", "unknown") or "unknown"
        sector_moves[s] = sector_moves.get(s, 0) + 1

    enriched = []
    for move in moves:
        ticker = move["ticker"]
        bundle = bundle_lookup.get(ticker, {})
        articles = bundle.get("articles", [])

        article_text = "\n".join(
            f"[{i}] ({a.get('source_name', '?')}) {a.get('title', '')}"
            for i, a in enumerate(articles[:15])
        ) or "No articles found"

        prompt = f"""Classify this stock move and score the associated news articles.

MOVE: {ticker} ({move.get('company_name', '')}) — {move['pct_change']}% {move['direction']} ({move['move_in_sigma']}σ)
SECTOR: {move.get('sector', 'unknown')}

ARTICLES:
{article_text}

Classify the root cause, provide sentiment scores (-1 to 1) and relevance scores (0 to 1) for each article."""

        try:
            raw_result = await classifier.ainvoke([
                SystemMessage(content = "You are a financial analyst. Classify moves and score articles."),
                HumanMessage(content = prompt),
            ])
            result = (
                raw_result
                if isinstance(raw_result, HaikuClassification)
                else HaikuClassification.model_validate(raw_result)
            )

            scored_articles = []
            for i, art in enumerate(articles):
                sent = result.article_sentiments[i] if i < len(result.article_sentiments) else 0.0
                rel = result.article_relevances[i] if i < len(result.article_relevances) else 0.0
                scored_articles.append({
                    **art,
                    "slm_sentiment": sent,
                    "slm_relevance": rel,
                    "slm_is_relevant": rel >= 0.5,
                })

            relevant = [a for a in scored_articles if a.get("slm_is_relevant")]
            avg_sent = sum(a["slm_sentiment"] for a in scored_articles) / max(len(scored_articles),1)

            enriched.append({
                **move,
                "primary_cause": result.primary_category,
                "classification_confidence": result.confidence,
                "secondary_causes": result.secondary_categories,
                "is_company_specific": result.is_company_specific,
                "explanation": result.explanation,
                "avg_article_sentiment": round(avg_sent, 3),
                "article_sentiments": [a.get("slm_sentiment", 0) for a in scored_articles],
                "relevant_article_count": len(relevant),
                "total_article_count": len(articles),
                "relevant_articles": relevant[:10],
                "all_scored_articles": scored_articles,
                "sector_move_count": sector_moves.get(move.get("sector", ""), 0),
                "is_broad_market_move": abs(snapshot.get("spy_daily_return", 0) or 0),
                "processed_by": "haiku_fallback",
            })

            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Haiku classificaiton failed for {ticker}: {e}")
            enriched.append({
                **move,
                "primary_cause": "unknown",
                "classification_confidence": 0.0,
                "avg_article_sentiment": 0.0,
                "relevant_article_count": 0,
                "total_article_count": len(articles),
                "relevant_articles": [],
                "all_scored_particles": [],
                "sector_move_count": 0,
                "is_broad_market_move": False,
                "processed_by": "haiku_fallback_error",
            })

    logger.info(f"[Haiku] Processed {len(enriched)} moves ({len(enriched)} API Calls)")
    return enriched 

# AGENT 2: LLM AGENT (Claude Sonnet, Deep Reasoning)
# Handles: Impact Assessment, Cross-Move Synthesis, Brief Generation

async def llm_agent(state: PipelineState) -> Dict:
    """
    Agent 2 - LLM Agent (Claude Sonnet)

    Takes SLM enriched moves and performs:
    1. Deep impact assessment per move (sector contagion, broader implications)
    2. Cross-move synthesis (are multiple moves related? Systemic risk?)
    3. Strategic recommendations with specific actions and time horizons
    4. Executive brief generation

    This is a single LLM call that processes all the moves together,
    enabling the model to reason across moves and identify connections
    """

    logger.info("=" * 60)
    logger.info("[Agent 2: LLM] Impact Assessment + Synthesis + Brief")
    logger.info("=" * 60)

    enriched_moves = state["enriched_moves"]
    snapshot = state["market_snapshot"]
    portfolio_name = state.get("portfolio_name", "Portfolio")
    tickers_monitored = state.get("tickers_monitored", 0)

    if not enriched_moves:
        return {
            "alerts": [],
            "brief": {
                "executive_summary": "No significant moves detected.",
                "sector_summaries": {},
                "top_recommendations": [],
                "portfolio_name": portfolio_name,
                "tickers_monitored": tickers_monitored,
                "tickers_flagged": 0,
                "alerts": [],
                "critical_alerts": [],
            },
        }

    #Sort by severity - most important first
    enriched_moves.sort(key = lambda m: abs(m.get("moves_in_sigma", 0)), reverse = True)

    #Limit to the top 25 for LLM context window
    top_moves = enriched_moves[:25]

    #-- Building the comprehensive prompt with SLM enriched + decomposed data --#
    market_ctx = ""
    if snapshot:
        market_ctx = (
            f"SPY: {snapshot.get('spy_daily_return', 'N/A')}% | "
            f"QQQ: {snapshot.get('qqq_daily_return', 'N/A')}% | "
            f"VIX: {snapshot.get('vix_level', 'N/A')} (Delta{snapshot.get('vix_change', 'N/A')}) | "
            f"10Y: {snapshot.get('treasury_10y', 'N/A')}%"
        )

    #-- Formatting by Cluster (not by individual ticker) --#
    clusters = state.get("clusters", [])
    systematic = state.get("systematic_summary", [])

    cluster_descriptions = []
    enriched_lookup = {em["ticker"]: em for em in top_moves}

    for cl in clusters:
        tickers_in_cluster = cl.get("tickers", [])
        cluster_moves = [enriched_lookup[t] for t in tickers_in_cluster if t in enriched_lookup]
        if not cluster_moves:
            continue 

        if cl.get("is_singleton"):
            em = cluster_moves[0]
            #Factor decomposition context 
            factor_ctx = (
                f"Total return: {em.get('total_return', em.get('pct_change', 0)):+.1f}% | "
                f"Market Beta component: {em.get('market_component', 0):+1f}% | "
                f"Sector component: {em.get('sector_component', 0):+.1f}% | "
                f"IDIOSYNCRATIC RESIDUAL: {em.get('idiosyncratic_return', 0):+.1f}% "
                f"({em.get('idiosyncratic_sigma', 0):.1f} sigma)"
            )
            pred_ctx = ""
            if em.get("prediction_residual_pct") is not None:
                pred_ctx = (
                    f"\n Quant model predicted: {em.get('predicted_return_pct', 0):+.1f}% | "
                    f"Prediction residual: {em.get('prediction_residual_pct', 0):+.1f}% "
                    f"({em.get('prediction_residual_sigma', 0):.1f} sigma unexplained)"
                )

            rel_titles = [a.get("title", "") for a in em.get("relevant_articles", [])[:5]]
            rel_text = " | ".join(rel_titles) if rel_titles else "No relevant articles"

            desc = (
                f"[Singleton] {em['ticker']} ({em.get('company_name', '')}) "
                f"[{em.get('alert_level', 'medium').upper()}]\n"
                f" {factor_ctx}{pred_ctx}\n"
                f" SLM classification: {em.get('primary_cause', 'unknown')} "
                f"({em.get('classification_confidence', 0):.0%}) | "
                f"Sentiment: {em.get('avg_article_sentiment', 0):.2f} \n"
                f" Key news: {rel_text}"
            )
        else:
            #Multi move cluster
            members = []
            for em in cluster_moves:
                members.append(
                    f"{em['ticker']} ({em.get('idiosyncratic_return', 0):+.1f}% idio, "
                    f"{em.get('idiosyncratic_sigma', 0):.1f} sigma)"
                )

            edges_text = ""
            if cl.get("edges"):
                edge_strs = [
                    f"{e['source']} -> {e['target']} (p = {e['partial_correlation']:.2f}, "
                    f" lead-lag = {e['lead_lag_days']} d)"
                    for e in cl["edges"][:5]
                ]
                edges_text = f"\n Causal links: {'; '.join(edge_strs)}"

            desc = (
                f"[Cluster {cl['cluster_id']}] {', '.join(cl['tickers'])} "
                f"(epicenter: {cl['epicenter_ticker']}, "
                f"idio: {cl['epicenter_idio_return']:+.1f}%) \n"
                f" Sector: {cl.get('dominant_sector', 'mixed')} | "
                f"Coherence: {cl.get('coherence_score', 0):.2f} | "
                f"Partial corr: {cl.get('avg_partial_correlation', 0):.2f}\n"
                f" Members: {'; '.join(members)}"
                f"{edges_text}"
            )

        cluster_descriptions.append(desc)

    all_clusters_text = "\n\n".join(cluster_descriptions)

    #Systematic context
    systematic_ctx = ""
    if systematic:
        sys_tickers = [f"{s['ticker']} ({s['total_return']:+.1f}%, {s['pct_explained']:.0f}% explained)"
                       for s in systematic[:10]]
        systematic_ctx = (
            f"\n\nSystematic Moves Filtered (explained by market/sector beta, NOT for analysis): \n"
            f" {', '.join(sys_tickers)}"
            f"\n These moves are normal given today's market conditions."
        )

    prompt = f"""You are the Chief Market Strategist analyzing today's UNEXPLAINED portfolio moves.

IMPORTANT: The quantitative pipeline has already:
1. Decomposed every move into market (β), sector, and idiosyncratic components
2. Filtered out moves fully explained by market/sector factors
3. Flagged only the IDIOSYNCRATIC RESIDUALS — what the quant model can't explain
4. Clustered related moves by shared drivers using partial correlation analysis

Your job is to EXPLAIN THE UNEXPLAINED — why did these idiosyncratic residuals occur?

PORTFOLIO: {portfolio_name}
TICKERS MONITORED: {tickers_monitored}
MOVES FLAGGED: {len(enriched_moves)} idiosyncratic (after factor decomposition)
MARKET CONTEXT: {market_ctx}

CAUSAL CLUSTERS (analyze by cluster, not individual ticker):
{all_clusters_text}{systematic_ctx}

For each cluster:
1. Explain what caused the IDIOSYNCRATIC residuals (not the market/sector component)
2. Assess if the SLM classification is correct or needs override
3. For multi-move clusters: what shared event links these tickers?
4. Strategic recommendations considering the causal structure

Generate impact assessments per cluster and a synthesized brief."""

    assessor = primary_llm.with_structured_output(ImpactAndBrief)

    try:
        raw_result = await assessor.ainvoke([
            SystemMessage(content=(
                "You are the Chief Market Strategist at a quantitative hedge fund. "
                "You receive factor-decomposed, cluster-analyzed move data. "
                "The quant pipeline has ALREADY separated market/sector beta from idiosyncratic moves. "
                "Your job is to explain ONLY the idiosyncratic residuals — the part the quant model "
                "can't explain. Do NOT attribute moves to broad market conditions (that's already handled). "
                "For multi-move clusters, identify the shared causal event. "
                "Be specific about actions, tickers, and time horizons."
            )),
            HumanMessage(content=prompt),
        ])
        result = (
            raw_result
            if isinstance(raw_result, ImpactAndBrief)
            else ImpactAndBrief.model_validate(raw_result)
        )

        #-- Building alerts fromt the LLM response
        alerts = []
        impact_lookup = {imp.ticker: imp for imp in result.move_impacts}

        for em in top_moves:
            ticker = em["ticker"]
            impact = impact_lookup.get(ticker)

            alert = {
                "ticker": ticker,
                "company_name": em.get("company_name", ticker),
                "alert_level": em.get("alert_level", "medium"),
                "title": impact.alert_title if impact else f"{ticker} {em['pct_change']}%",
                "summary": impact.alert_summary if impact else "", 
                "move": em,
                "root_cause": {
                    "ticker": ticker,
                    "primary_cause": em.get("primary_cause", "unknown"),
                    "confidence": em.get("classification_confidence", 0),
                    "explanation": impact.alert_summary if impact else "",
                    "is_company_specific": em.get("is_company_specific", True),
                    "related_tickers": impact.related_moves if impact else [],
                },
                "sector_impacts": impact.sector_impacts if impact else [],
                "recommended_actions": impact.recommended_actions if impact else [],
                "related_moves": impact.related_moves if impact else [],
                "news_count": em.get("total_article_count", 0),
                "forward_looking": impact.forward_looking if impact else "",
            }
            alerts.append(alert)

        #Sort by severity
        level_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key = lambda a: (
            level_order.get(a["alert_level"], 4),
            -abs(a.get("move", {}).get("move_in_sigma", 0)),
        ))

        critical_alerts = [a for a in alerts if a["alert_level"] == "critical"]

        brief = {
            "executive_summary": result.executive_summary,
            "sector_summaries": result.sector_summaries,
            "top_recommendations": result.top_recommendations,
            "risk_watchlist": result.risk_watchlist,
            "portfolio_name": portfolio_name,
            "tickers_monitored": tickers_monitored,
            "tickers_flagged": len(enriched_moves),
            "alerts": alerts,
            "critical_alerts": critical_alerts,
            "market_snapshot": snapshot,
            "total_articles_analyzed": sum(em.get("total_article_count", 0) for em in enriched_moves),
        }

        logger.info(
            f"[Agent 2] Generated {len(alerts)} alerts, "
            f"{len(critical_alerts)} critical (1 LLM call)"
        )
        return {"alerts": alerts, "brief": brief}
    
    except Exception as e:
        logger.error(f"LLM agent failed: {e}", exc_info = True)
        #Graceful degradation: build a basic brief form only the SLM data
        alerts = []
        for em in top_moves:
            alerts.append({
                "ticker": em["ticker"],
                "company_name": em.get("company_name", ""),
                "alert_level": em.get("alert_level", "medium"),
                "title": f"{em['ticker']} {em['pct_change']}% ({em.get('primary_cause', 'unknown')})",
                "summary": f"Classified as {em.get('primary_cause', 'unknown')} by SLM. LLM analysis is unavailable",
                "move": em,
                "root_cause": {"ticker": em["ticker"], "primary_cause": em.get("primary_cause", "unknown")},
                "sector_impacts": [],
                "recommended_actions": [],
                "news_count": em.get("total_article_count", 0),
            })

        return {
            "alerts": alerts,
            "brief": {
                "executive_summary": f"LLM analysis failed. {len(alerts)} moves flagged by SLM classification.",
                "sector_summaries": {},
                "top_recommendations": [],
                "portfolio_name": portfolio_name,
                "tickers_monitored": tickers_monitored,
                "tickers_flagged": len(enriched_moves),
                "alerts": alerts,
                "critical_alerts": [a for a in alerts if a["alert_level"] == "critical"],
            },
        }
    
# -- Building the Two Agent Workflow --#
def build_pipeline() -> CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]:
    """Construct the 2-agent LangGraph pipeline: SLM → LLM."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("slm_agent", slm_agent)
    workflow.add_node("llm_agent", llm_agent)

    workflow.set_entry_point("slm_agent")
    workflow.add_edge("slm_agent", "llm_agent")
    workflow.add_edge("llm_agent", END)

    return workflow.compile()


# ── Entry Point ─-#

async def run_intelligence_pipeline(
    decomposed_moves: List["DecomposedMove"],
    causal_graph: "CausalGraph",
    news_bundles: List[TickerNewsBundle],
    market_snapshot: MarketSnapshot,
    prediction_results: Dict[str, Dict[str, float]],
    systematic_moves: Optional[List["DecomposedMove"]] = None,
    portfolio_name: str = "Portfolio",
    tickers_monitored: int = 0,
) -> Optional[MarketBrief]:
    """
    Run the 2-agent pipeline: SLM Agent → LLM Agent.

    Now receives:
    - decomposed_moves: Only idiosyncratic moves (systematic filtered out)
    - causal_graph: Clusters of related moves
    - prediction_results: What the quant model predicted vs. actual
    - systematic_moves: Moves explained by market/sector (for context)
    """
    start_time = datetime.now(timezone.utc)

    # Serialize decomposed moves with factor data
    moves_data = []
    for dm in decomposed_moves:
        move_dict = dm.original_move.model_dump() if dm.original_move else {}
        move_dict.update({
            "market_component": dm.market_component,
            "sector_component": dm.sector_component,
            "idiosyncratic_return": dm.idiosyncratic_return,
            "idiosyncratic_sigma": dm.idiosyncratic_sigma,
            "market_beta": dm.market_beta,
            "sector_beta": dm.sector_beta,
            "r_squared": dm.r_squared,
            "factor_model_prediction": dm.factor_model_prediction,
            "spy_return": dm.spy_return,
            "sector_etf_return": dm.sector_etf_return,
        })
        # Add prediction residual if available
        pred = prediction_results.get(dm.ticker, {})
        if pred:
            move_dict["predicted_return_pct"] = pred.get("predicted_return_pct", 0)
            move_dict["prediction_residual_pct"] = pred.get("residual_pct", 0)
            move_dict["prediction_residual_sigma"] = pred.get("residual_sigma", 0)
        moves_data.append(move_dict)

    bundles_data = [b.model_dump() for b in news_bundles]
    snapshot_data = market_snapshot.model_dump() if market_snapshot else {}

    # Serialize causal graph clusters
    clusters_data = []
    for cluster in causal_graph.clusters:
        clusters_data.append({
            "cluster_id": cluster.cluster_id,
            "tickers": cluster.tickers,
            "epicenter_ticker": cluster.epicenter_ticker,
            "epicenter_idio_return": cluster.epicenter_idio_return,
            "dominant_sector": cluster.dominant_sector,
            "avg_partial_correlation": cluster.avg_partial_correlation,
            "coherence_score": cluster.coherence_score,
            "is_singleton": cluster.is_singleton,
            "is_sector_cluster": cluster.is_sector_cluster,
            "size": cluster.size,
            "edges": [
                {"source": e.source, "target": e.target,
                 "partial_correlation": e.partial_correlation,
                 "lead_lag_days": e.lead_lag_days}
                for e in cluster.internal_edges
            ],
        })

    # Systematic moves summary for context
    systematic_summary = []
    if systematic_moves:
        for sm in systematic_moves[:20]:
            systematic_summary.append({
                "ticker": sm.ticker, "total_return": sm.total_return,
                "market_component": sm.market_component,
                "pct_explained": sm.pct_explained,
            })

    initial_state: PipelineState = {
        "moves": moves_data,
        "news_bundles": bundles_data,
        "market_snapshot": snapshot_data,
        "portfolio_name": portfolio_name,
        "tickers_monitored": tickers_monitored,
        "clusters": clusters_data,
        "systematic_summary": systematic_summary,
        "enriched_moves": [],
        "alerts": [],
        "brief": None,
    }

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state)

    brief_data = result.get("brief")
    if not brief_data:
        return None

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Construct MarketBrief
    brief = MarketBrief(
        executive_summary=brief_data.get("executive_summary", ""),
        portfolio_name=brief_data.get("portfolio_name", portfolio_name),
        tickers_monitored=brief_data.get("tickers_monitored", tickers_monitored),
        tickers_flagged=brief_data.get("tickers_flagged", 0),
        market_snapshot=market_snapshot,
        sector_summary=brief_data.get("sector_summaries", {}),
        total_articles_analyzed=brief_data.get("total_articles_analyzed", 0),
        generation_time_seconds=round(elapsed, 1),
    )

    # Convert alerts to MoveAlert models
    for alert_data in brief_data.get("alerts", []):
        try:
            move_dict = alert_data.get("move", {})
            rc_dict = alert_data.get("root_cause", {})

            root_cause = RootCauseAnalysis(
                ticker=rc_dict.get("ticker", ""),
                primary_cause=MarketImpactCategory(rc_dict.get("primary_cause", "unknown")),
                explanation=rc_dict.get("explanation", ""),
                confidence=rc_dict.get("confidence", 0),
                is_company_specific=rc_dict.get("is_company_specific", True),
                related_tickers=rc_dict.get("related_tickers", []),
            )

            sector_impacts = [
                SectorImpact(
                    sector=Sector(si["sector"]),
                    impact_summary=si.get("impact_summary", ""),
                    direction=MoveDirection(si.get("direction", "down")),
                    magnitude=si.get("magnitude", "moderate"),
                )
                for si in alert_data.get("sector_impacts", [])
                if si.get("sector") in [e.value for e in Sector]
            ]

            actions = [
                ActionRecommendation(**a)
                for a in alert_data.get("recommended_actions", [])
            ]

            move_obj = PriceMove(
                ticker=move_dict.get("ticker", ""),
                company_name=move_dict.get("company_name", ""),
                period=MovePeriod(move_dict.get("period", "daily")),
                direction=MoveDirection(move_dict.get("direction", "down")),
                price_start=move_dict.get("price_start", 0),
                price_end=move_dict.get("price_end", 0),
                pct_change=move_dict.get("pct_change", 0),
                historical_volatility=move_dict.get("historical_volatility", 0),
                daily_sigma=move_dict.get("daily_sigma", 0),
                move_in_sigma=move_dict.get("move_in_sigma", 0),
                threshold_sigma=move_dict.get("threshold_sigma", 2.0),
                alert_level=AlertLevel(move_dict.get("alert_level", "medium")),
            )

            alert = MoveAlert(
                ticker=alert_data["ticker"],
                company_name=alert_data.get("company_name", ""),
                alert_level=AlertLevel(alert_data["alert_level"]),
                title=alert_data.get("title", ""),
                summary=alert_data.get("summary", ""),
                move=move_obj,
                root_cause=root_cause,
                sector_impacts=sector_impacts,
                recommended_actions=actions,
                related_moves=alert_data.get("related_moves", []),
                news_count=alert_data.get("news_count", 0),
            )
            brief.alerts.append(alert)
            if alert.alert_level == AlertLevel.CRITICAL:
                brief.critical_alerts.append(alert)

        except Exception as e:
            logger.warning(f"Alert construction failed for {alert_data.get('ticker')}: {e}")

    for rec in brief_data.get("top_recommendations", []):
        try:
            brief.top_recommendations.append(ActionRecommendation(**rec))
        except Exception:
            continue

    logger.info(
        f"Pipeline complete: {len(brief.alerts)} alerts, "
        f"{len(brief.critical_alerts)} critical, {elapsed:.1f}s | "
        f"SLM → LLM (2-agent architecture)"
    )
    return brief
