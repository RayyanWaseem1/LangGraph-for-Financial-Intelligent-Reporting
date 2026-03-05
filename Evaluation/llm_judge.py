"""
LLM-as-Judge Evaluation
Uses a separate Claude instance to score intelligence brief quality

Evaluates each brief on 5 dimensions (on a 1-5 scale):
    1. Factual Accuracy - does the brief state anything provably wrong?
    2. Causal Reasoning - does it distinguish correlation from causation?
    3. Decomposition Awareness - does it correctly use factor decomposition context?
    4. Actionability - are recommendations specific enough to act on?
    5. Information Density - is every sentence carrying weight, or is there filler?

Also evaluates:
    - Hallucination detection (claims that are not supported by input data)
    - Systematic attribution errors (attributing idiosyncratic moves to market conditions)
    - Recommendation specificity (ticker + direction + timeframe)

Follows MT-Bench/AlpacaEval paradigm: structured rubric, forced scoring, chain of thought justification before each score
"""

import logging 
import asyncio
import json 
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field 
import numpy as np 

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr 

from Data.settings import Settings
from Data.data_model import MarketBrief

logger = logging.getLogger(__name__)
settings = Settings() 

#-- Structured Judge Output --#
class DimensionScore(BaseModel):
    """ Scoring for a single evaluation dimension"""
    dimension: str 
    score: int = Field(ge = 1, le = 5, description = "1 = poor, 5 = excellent")
    justification: str = Field(description = "2-3 sentence reasoning for the score")
    specific_issues: List[str] = Field(
        default_factory=list,
        description = "Specific errors or strenghts identified"
    )

class BriefJudgement(BaseModel):
    """ Complete judgement of a single intelligence brief"""
    factual_accuracy: DimensionScore
    causal_reasoning: DimensionScore
    decomposition_awareness: DimensionScore
    actionability: DimensionScore
    information_density: DimensionScore

    #Specific checks
    hallucinations_detected: List[str] = Field(
        default_factory = list,
        description = "Claims in the brief are not supported by the input data"
    )
    systematic_attribution_errors: List[str] = Field(
        default_factory=list,
        description = "Cases where idiosyncratic moves are attributed to the market/sector"
    )
    vague_recommendations: List[str] = Field(
        default_factory = list,
        description = "Recommendations lacking specific ticker, direction, or timeframe"
    )

    overall_score: float = Field(description = "Weighted average of dimension scores (1-5)")
    overall_assessment: str = Field(description = "2-3 sentence overall quality judgement")

# -- Aggregated Metrics --#

@dataclass 
class LLMJudgeMetrics:
    """ Aggregated LLM-as-Judge results across multiple briefs"""
    #Per dimension averages
    avg_factual_accuracy: float = 0.0
    avg_causal_reasoning: float = 0.0
    avg_decomposition_awareness: float = 0.0
    avg_actionability: float = 0.0
    avg_information_density: float = 0.0
    avg_overall_score: float = 0.0

    #Per dimension standard deviation 
    std_factual_accuracy: float = 0.0
    std_causal_reasoning: float = 0.0
    std_overall_score: float = 0.0

    #Error rates
    pct_briefs_with_hallucinations: float = 0.0
    pct_briefs_with_attribution_errors: float = 0.0
    avg_vague_recommendations_per_brief: float = 0.0

    #Score distribution
    score_distribution: Dict[str, int] = field(default_factory = dict) #1-5 -> count

    n_briefs_evaluated: int = 0
    individual_judgements: List[Dict] = field(default_factory = list) 

#-- LLM Judge --#
class LLMJudge:
    """
    Uses a separate Claude instance to evaluate intelligence brief quality.
    The judge model should be a different model or at least a separate invocation 
    that uses a judge-specific system prompt
    """

    def __init__(self):
        self.judge_llm = ChatAnthropic(
            model_name = settings.PRIMARY_MODEL,
            api_key = SecretStr(settings.ANTHROPIC_API_KEY),
            max_tokens_to_sample = 4096,
            temperature = 0.1, #low temp for consistent scoring
            timeout = 60.0,
            stop = None,
        )

    async def judge_brief(
        self,
        brief_text: str,
        input_data: Dict,
    ) -> Optional[BriefJudgement]:
        """
        Judging a single intelligence brief

        Params:
            - brief_text: The generated executive summary and its recommendations
            - input_data: The raw data that was fed to the pipeline (moves, decomposition, etc)
        """
        judge = self.judge_llm.with_structured_output(BriefJudgement)

        #Format the input data the brief was based on
        moves_summary = self._format_input_moves(input_data.get("moves", []))
        decomp_summary = self._format_decomposition(input_data.get("decomposed_moves", []))
        clusters_summary = self._format_clusters(input_data.get("clusters", []))

        prompt = f"""You are an expert evaluator assessing the quality of a financial intelligence brief.
You will score the brief on 5 dimensions (1-5 scale) with specific justification for each.

IMPORTANT CONTEXT: This brief was generated by an automated system that:
1. First decomposed each price move into market (β), sector, and idiosyncratic components
2. Only the IDIOSYNCRATIC residuals were flagged for analysis
3. Moves were clustered by shared causal drivers using partial correlations
4. The brief should explain the RESIDUALS, not restate market-wide conditions

═══ INPUT DATA THE SYSTEM RECEIVED ═══

FLAGGED MOVES (after factor decomposition):
{moves_summary}

FACTOR DECOMPOSITION DETAILS:
{decomp_summary}

CAUSAL CLUSTERS:
{clusters_summary}

═══ GENERATED BRIEF TO EVALUATE ═══

{brief_text}

═══ SCORING RUBRIC ═══

FACTUAL ACCURACY (1-5):
  5: All claims verifiable from input data; no fabricated details
  3: Mostly accurate but some claims lack support in input data
  1: Multiple fabricated claims or contradictions with input data

CAUSAL REASONING (1-5):
  5: Correctly identifies causes vs. correlations; acknowledges uncertainty
  3: Some causal claims are plausible but not well-supported
  1: Confuses correlation with causation; makes unsupported causal claims

DECOMPOSITION AWARENESS (1-5):
  5: Correctly focuses on idiosyncratic residuals; never attributes them to market/sector
  3: Partially uses decomposition context but sometimes explains the market component
  1: Ignores decomposition entirely; explains total returns as if they're all idiosyncratic

ACTIONABILITY (1-5):
  5: Every recommendation has specific ticker, direction, urgency, and timeframe
  3: Some recommendations are specific, others are generic
  1: All recommendations are vague platitudes ("monitor the situation")

INFORMATION DENSITY (1-5):
  5: Every sentence adds new information; no filler or repetition
  3: Some useful content mixed with filler or restated information
  1: Mostly filler; repeats input data without adding insight

For EACH dimension, provide your reasoning BEFORE your score.
Also identify any hallucinations, attribution errors, or vague recommendations."""
        
        try:
            result = await judge.ainvoke([
                SystemMessage(content = (
                    "You are a rigorous evaluator of financial intelligence briefs. "
                    "You have deep expertise in quantitative finance and know the difference "
                    "between a brief that adds genuine insight versus one that merely restates "
                    "the input data. Be honest and critical - a score of 3 is average, "
                    "5 should be rare, and 1 means genuine failure. "
                    "Always justify your scores with specific examples from the brief."
                )),
                HumanMessage(content = prompt)
            ])
            if isinstance(result, BriefJudgement):
                return result
            if isinstance(result, BaseModel):
                return BriefJudgement.model_validate(result.model_dump())
            if isinstance(result, dict):
                return BriefJudgement.model_validate(result)
            logger.error(f"Unexpected judge output type: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return None 
        
    async def evaluate_briefs(
        self,
        briefs_with_inputs: List[Dict],
    ) -> LLMJudgeMetrics:
        """ 
        Evaluate multiple briefs and aggregate the results

        Each entry in the briefs_with_inputs should have:
            -"brief_text": str
            -"input_data": Dict (moves, decomposition, clusters)
        """
        metrics = LLMJudgeMetrics()
        judgements = []

        for i, entry in enumerate(briefs_with_inputs):
            logger.info(f"Judging brief {i+1}/{len(briefs_with_inputs)} ...")

            judgement = await self.judge_brief(
                entry["brief_text"],
                entry["input_data"],
            )

            if judgement:
                judgements.append(judgement)

            await asyncio.sleep(1.0)

        if not judgements:
            return metrics
        
        metrics.n_briefs_evaluated = len(judgements)

        #Aggregate scores
        factual = [j.factual_accuracy.score for j in judgements]
        causal = [j.causal_reasoning.score for j in judgements]
        decomp = [j.decomposition_awareness.score for j in judgements]
        action = [j.actionability.score for j in judgements]
        density = [j.information_density.score for j in judgements]
        overall = [j.overall_score for j in judgements]

        metrics.avg_factual_accuracy = round(float(np.mean(factual)), 2)
        metrics.avg_causal_reasoning = round(float(np.mean(causal)), 2)
        metrics.avg_decomposition_awareness = round(float(np.mean(decomp)), 2)
        metrics.avg_actionability = round(float(np.mean(action)), 2)
        metrics.avg_information_density = round(float(np.mean(density)), 2)
        metrics.avg_overall_score = round(float(np.mean(overall)), 2) 

        metrics.std_factual_accuracy = round(float(np.std(factual)), 2)
        metrics.std_causal_reasoning = round(float(np.std(causal)), 2)
        metrics.std_overall_score = round(float(np.std(overall)), 2)

        #Error rates
        metrics.pct_briefs_with_hallucinations = round(
            sum(1 for j in judgements if j.hallucinations_detected) / len(judgements) * 100, 1
        )
        metrics.pct_briefs_with_attribution_errors = round(
            sum(1 for j in judgements if j.systematic_attribution_errors) / len(judgements) * 100, 1
        )
        metrics.avg_vague_recommendations_per_brief = round(
            float(np.mean([len(j.vague_recommendations) for j in judgements])), 2
        )

        #Score distribution
        all_scores = factual + causal + decomp + action + density 
        for s in range(1, 6):
            metrics.score_distribution[str(s)] = sum(1 for x in all_scores if x == s)

        #Store individual judgements for detailed analysis
        metrics.individual_judgements = [
            {
                "factual_accuracy": j.factual_accuracy.score,
                "causal_reasoning": j.causal_reasoning.score,
                "decomposition_awareness": j.decomposition_awareness.score,
                "actionability": j.actionability.score,
                "information_density": j.information_density.score,
                "overall": j.overall_score,
                "hallucinations": j.hallucinations_detected,
                "attribution_errors": j.systematic_attribution_errors,
                "vague_recs": j.vague_recommendations,
                "overall_assessment": j.overall_assessment,
            }
            for j in judgements
        ]

        logger.info(
            f"LLM Judge: {len(judgements)} briefs evaluated | "
            f"Overall: {metrics.avg_overall_score}/5 | "
            f"Factual: {metrics.avg_factual_accuracy}/5 | "
            f"Causal: {metrics.avg_causal_reasoning}/5 | "
            f"Decomp: {metrics.avg_decomposition_awareness}/5"
        )

        return metrics 
    
    #-- Formatting Helpers --#

    @staticmethod 
    def _format_input_moves(moves: List[Dict]) -> str:
        lines = []
        for m in moves[:20]:
            lines.append(
                f" {m.get('ticker', '?')}: {m.get('pct_change', 0):+.1f}% total, "
                f"idio {m.get('idiosyncratic_return', 0):+.1f}% "
                f"({m.get('idiosyncratic_sigma', 0):.1f} sigma) "
                f"[{m.get('alert_level', '?')}]"
            )
        return "\n".join(lines) if lines else "No moves"
    
    @staticmethod
    def _format_decomposition(moves: List[Dict]) -> str:
        lines = []
        for m in moves[:10]:
            lines.append(
                f" {m.get('ticker', '?')}: total = {m.get('total_return', 0):+.1f}% -> "
                f"market = {m.get('market_component', 0):+.1f}% + "
                f"sector = {m.get('sector_component', 0):+.1f}% + "
                f"IDIO = {m.get('idiosyncratic_return', 0):+.1f}% | "
                f"R-squared = {m.get('r_squared', 0):2f}"
            )
        return "\n".join(lines) if lines else "No decomposition data"
    
    @staticmethod
    def _format_clusters(clusters: List[Dict]) -> str:
        lines = []
        for c in clusters[:10]:
            if c.get("is_singleton"):
                lines.append(f" [Singleton] {c['tickers'][0]}")
            else:
                lines.append(
                    f" [Cluster {c.get('cluster_id', '?')}] "
                    f"{', '.join(c.get('tickers', []))} | "
                    f"epicenter: {c.get('epicenter_ticker', '?')} | "
                    f"coherence: {c.get('coherence_score', 0):.2f}"
                )
        return "\n".join(lines) if lines else "No cluster data"
    
