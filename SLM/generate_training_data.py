"""
Training Data Generator
Generates labeled training data by:
1. Pulling historical prie moves from yfinance
2. Retrieving contemporaneous news from GDELT/NewsAPI
3. Labeling via Claude (classification, sentiment, relevance)
4. Outputting structured datasets for fine-tuning

This is essentially the "teacher" pipeline that distils Claude's capabilities
into training data for the smaller model
"""

import asyncio
import json 
import logging
import random 
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict 

import pandas as pd 
import numpy as np 
import yfinance as yf 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr

from Data.settings import Settings, SECTOR_TICKERS, TICKER_COMPANY_NAMES
from SLM.model import CLASSIFICATION_LABELS

logger = logging.getLogger(__name__)
settings = Settings()

#-- Structured Output for Labeling --#

class ArticleLabel(BaseModel):
    """ Claude's label for a single article relative to a price move"""
    relevance_score: float = Field(description = "0-1: how relevant is this article to explaining the price move?")
    is_causal: bool = Field(description = "Is this article about a likely cause of the move (vs. coincidenttal)?")
    sentiment_score: float = Field(description = "-1 (very bearish) to 1 (very bullish) financial sentiment")
    primary_category: str = Field(description = f"Root cause category. One of: {', '.join(CLASSIFICATION_LABELS)}")

class MoveLabelBatch(BaseModel):
    """ Claude's labels for a batch of articles for one move"""
    move_category: str = Field(description = f"Primary category for this overall move. One of: {', '.join(CLASSIFICATION_LABELS)}")
    move_confidence: float = Field(description = "0-1 confidence in the move category")
    article_labels: List[ArticleLabel] = Field(description = "Lables for each article in order")

# -- Training Examples Structures -- #
@dataclass
class ClassificationExample:
    """ One training example for the classification head"""
    text: str #Article text or move description
    label: str #Category name
    label_idx: int #Category index
    confidence: float #Teacher confidence
    ticker: str
    move_pct: float 
    move_sigma: float 

@dataclass
class SentimentExample:
    """ One training example for the sentiment head"""
    text: str 
    sentiment: float #-1 to 1
    ticker: str 

@dataclass
class RelevanceExample:
    """ One training example for the relevance head"""
    text: str #Article text
    relevance: float #0 to 1
    is_relevant: bool #Binary label
    ticker: str 
    move_pct: float 

#-- Historical Move Scanner --#
def find_historical_moves(
    tickers: List[str],
    lookback_days: int = 365,
    sigma_threshold: float = 2.0,
    max_moves_per_ticker: int = 10,
) -> List[Dict[str, Any]]:
    """ 
    Scan historical data for significant moves to use as training examples.
    Returns a list of move dicts with ticker, date, pct_change, sigma, etc
    """
    all_moves = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period = f"{lookback_days}d", progress = False)
            if data is None or data.empty or len(data) < 60:
                continue 

            close = data["Close"].dropna()
            returns = close.pct_change().dropna()

            #Rolling 60-day volatility
            rolling_sigma = returns.rolling(60).std()

            moves = []
            for i in range(60, len(returns)):
                ret = returns.iloc[i]
                sigma = rolling_sigma.iloc[i]

                if sigma > 0 and abs(ret) / sigma >= sigma_threshold:
                    date = returns.index[i]
                    moves.append({
                        "ticker": ticker,
                        "company_name": TICKER_COMPANY_NAMES.get(ticker, ticker),
                        "date": date.strftime("%Y-%m-%d"),
                        "pct_change": round(float(ret * 100), 2),
                        "move_in_sigma": round(float(abs(ret) / sigma), 2),
                        "direction": "up" if ret > 0 else "down",
                        "price": round(float(close.iloc[i]), 2),
                        "volume": int(data["Volume"].iloc[i]) if "Volume" in data.columns else 0,
                    })

            #Sample to avoid overrepresenting one ticker
            if len(moves) > max_moves_per_ticker:
                moves = random.sample(moves, max_moves_per_ticker)

            all_moves.extend(moves)
        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")
            continue 

    logger.info(f"Found {len(all_moves)} historical significant moves across {len(tickers)} tickers")
    return all_moves

#-- News Retrieval for Historial Dates --#

async def retrieve_historical_news(
    ticker: str, company_name: str, date: str, max_articles: int = 10
) -> List[Dict[str, str]]:
    """
    Retrieve news articles around a specific historical date for a ticker
    Using GDELT
    """
    import httpx

    GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

    #Searching window: day before to the day after the move 
    move_date = datetime.strptime(date, "%Y-%m-%d")
    start = (move_date - timedelta(days = 1)).strftime("%Y%m%d%H%M%S")
    end = (move_date + timedelta(days = 1)).strftime("%Y%m%d%H%M%S")

    query = f'"{company_name}" OR "{ticker}"'

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_articles * 2),
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
        "sort": "DateDesc",
    }

    try:
        async with httpx.AsyncClient(timeout = 15.0) as client:
            resp = await client.get(GDELT_DOC_API, params = params)
            resp.raise_for_status()
            data = resp.json() 

            articles = []
            for item in data.get("articles", []):
                title = item.get("title", "")
                if title:
                    articles.append({
                        "title": title,
                        "source": item.get("domain", "unknown"),
                        "url": item.get("url", ""),
                        "tone": item.get("tone", 0),
                    })

            return articles[:max_articles]
    except Exception as e:
        logger.debug(f" No GDELT results for {ticker} on {date}: {e}")
        return []
    
#-- Claude Labeling --#

async def label_move_with_claude(
    move: Dict[str, Any],
    articles: List[Dict[str, str]],
) -> Optional[MoveLabelBatch]:
    """
    Using Claude to label the historical move and its associated articles.
    This is the "teacher" that generates training data for the SLM
    """

    llm = ChatAnthropic(
        model_name = settings.FAST_MODEL,
        api_key = SecretStr(settings.ANTHROPIC_API_KEY),
        max_tokens_to_sample = 2048,
        temperature = 0.1,
        timeout = None,
        stop = None,
    )
    labeler = llm.with_structured_output(MoveLabelBatch)

    articles_text = "\n".join(
        f"[{i}] ({a['source']}) {a['title']}"
        for i, a in enumerate(articles)
    ) if articles else "No articles found"

    prompt = f"""Label this historical stock move and its associated news articles.

MOVE:
- Ticker: {move['ticker']} ({move['company_name']})
- Date: {move['date']}
- Change: {move['pct_change']}% ({move['direction']})
- Magnitude: {move['move_in_sigma']}σ

ARTICLES:
{articles_text}

For the overall move, classify the root cause category.
For each article, provide relevance, causality, and sentiment scores.
If no articles are found, still classify the move based on the ticker, date, and magnitude."""

    try:
        raw_result = await labeler.ainvoke([
            SystemMessage(content=(
                "You are a financial data labeler creating training data for an ML model. "
                "Be precise and consistent. For classification, choose the single best category. "
                "For sentiment, consider financial implications (layoffs = bearish, "
                "strong earnings = bullish). For relevance, consider whether the article "
                "plausibly explains the specific price move."
            )),
            HumanMessage(content=prompt),
        ])
        return (
            raw_result
            if isinstance(raw_result, MoveLabelBatch)
            else MoveLabelBatch.model_validate(raw_result)
        )
    except Exception as e:
        logger.warning(f"Claude labeling failed for {move['ticker']} {move['date']}: {e}")
        return None 
    
# --  Dataset Generation Pipeline -- #
async def generate_training_dataset(
    tickers: Optional[List[str]] = None,
    lookback_days: int = 365,
    sigma_threshold: float = 2.0,
    max_moves: int = 500,
    output_dir: str = "training_data",
) -> Dict[str, int]:
    """
    Full training data generation pipeline:
    1. Find historical significant moves
    2. Retrieve news for each move
    3. Label with Claude
    4. Save structured training examples

    Returns counts of examples generated per task
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)

    #Default tickers: Top companies from each sector
    if not tickers:
        tickers = []
        for sector, sector_tickers in SECTOR_TICKERS.items():
            tickers.extend(sector_tickers[:5]) #Top 5 per sector
        tickers = list(set(tickers))

    logger.info(f"Generating training data from {len(tickers)} tickers, {lookback_days}d lookback")

    #Step 1: Find historical moves
    moves = find_historical_moves(
        tickers, lookback_days, sigma_threshold,
        max_moves_per_ticker=max_moves // len(tickers) + 1,
    )
    if len(moves) > max_moves:
        moves = random.sample(moves, max_moves)

    logger.info(f"Processing {len(moves)} historical moves")

    #Storage for training examples
    cls_examples: List[Dict] = []
    sent_examples: List[Dict] = []
    rel_examples: List[Dict] = []

    #Step 2 and Step 3: Retrieve the news + Label with Claude
    for i, move in enumerate(moves):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i+1} / {len(moves)} moves processed")

        #Retrieve historical news
        articles = await retrieve_historical_news(
            move["ticker"], move["company_name"], move["date"],
        )

        #Label with Claude
        labels = await label_move_with_claude(move, articles)
        if not labels:
            continue 

        # -- Extract classificaiton examples -- %
        #Use the move description as a classification example
        move_text = (
            f"{move['company_name']} ({move['ticker']}) stock moved "
            f"{move['pct_change']}% {move['direction']}. "
            f"Magnitude: {move['move_in_sigma']} standard deviations."
        )

        #Add article titles for richer context
        if articles:
            article_titles = " | ".join(a["title"] for a in articles[:5])
            move_text += f" Related news: {article_titles}"

        if labels.move_category in CLASSIFICATION_LABELS:
            cls_examples.append(asdict(ClassificationExample(
                text = move_text,
                label = labels.move_category,
                label_idx = CLASSIFICATION_LABELS.index(labels.move_category),
                confidence = labels.move_confidence,
                ticker = move["ticker"],
                move_pct = move["pct_change"],
                move_sigma = move["move_in_sigma"],
            )))

        # -- Extract per-article example -- #
        for j, art_label in enumerate(labels.article_labels):
            if j >= len(articles):
                break 

            article_text = articles[j]["title"]

            #Sentiment example
            sent_examples.append(asdict(SentimentExample(
                text = article_text,
                sentiment = art_label.sentiment_score,
                ticker = move["ticker"],
            )))

            #Relevance example (with move context prepended)
            rel_text = (
                f"[MOVE: {move['ticker']} {move['pct_change']}% {move['direction']}] "
                f"{article_text}"
            )
            rel_examples.append(asdict(RelevanceExample(
                text = rel_text,
                relevance = art_label.relevance_score,
                is_relevant=art_label.is_causal,
                ticker = move['ticker'],
                move_pct = move["pct_change"]
            )))

        #Rate limit 
        await asyncio.sleep(0.5)

    #Step 4: Saving datasets
    for name, examples in [
        ("classification", cls_examples),
        ("sentiment", sent_examples),
        ("relevance", rel_examples),
    ]:
        filepath = output_path / f"{name}_train.jsonl"
        with open(filepath, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, default = str) + "\n")
        logger.info(f"Saved {len(examples)} {name} examples to {filepath}")

    #Save metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "lookback_days": lookback_days,
        "sigma_threshold": sigma_threshold,
        "total_moves": len(moves),
        "examples": {
            "classification": len(cls_examples),
            "sentiment": len(sent_examples),
            "relevance": len(rel_examples),
        },
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent = 2)

    logger.info(
        f"Dataset generation complete: "
        f"{len(cls_examples)} classification, "
        f"{len(sent_examples)} sentiment, "
        f"{len(rel_examples)} relevance examples"
    )

    return metadata["examples"]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SLM training data")
    parser.add_argument("--lookback", type=int, default=365, help="Days of history")
    parser.add_argument("--sigma", type=float, default=2.0, help="σ threshold for moves")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves to process")
    parser.add_argument("--output", default="training_data", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_training_dataset(
        lookback_days=args.lookback,
        sigma_threshold=args.sigma,
        max_moves=args.max_moves,
        output_dir=args.output,
    ))
