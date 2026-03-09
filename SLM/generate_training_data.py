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
from typing import List, Dict, Any, Optional, Tuple, cast
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

            #Handling multi-level columns from potential newer yfinance versions
            #yfinance can return columns like (Close, AAPL) instead of just Close
            if isinstance(data.columns, pd.MultiIndex):
                #Flatten: take the first level or extrac ticker-specific data
                data = data.droplevel("Ticker", axis = 1) if "Ticker" in data.columns.names else data[ticker] if ticker in data.columns.get_level_values(0) else data.droplevel(0, axis = 1)

            close = data["Close"].dropna()
            returns = close.pct_change().dropna()

            #Ensure we have a scalar series rather than a dataframe
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            #Rolling 60-day volatility
            rolling_sigma = returns.rolling(60).std()

            #Getting volume series
            vol_series = data["Volume"] if "Volume" in data.columns else None
            if vol_series is not None and isinstance(vol_series, pd.DataFrame):
                vol_series = vol_series.iloc[:, 0]

            moves = []
            for i in range(60, len(returns)):
                ret = float(returns.iloc[i])
                sigma = float(rolling_sigma.iloc[i])

                if sigma > 0 and abs(ret) / sigma >= sigma_threshold:
                    date = returns.index[i]
                    vol = int(vol_series.iloc[i]) if vol_series is not None and i < len(vol_series) else 0
                    moves.append({
                        "ticker": ticker,
                        "company_name": TICKER_COMPANY_NAMES.get(ticker, ticker),
                        "date": date.strftime("%Y-%m-%d"),
                        "pct_change": round(float(ret * 100), 2),
                        "move_in_sigma": round(float(abs(ret) / sigma), 2),
                        "direction": "up" if ret > 0 else "down",
                        "price": round(float(close.iloc[i]), 2),
                        "volume": vol,
                    })

            #Sample to avoid overrepresenting one ticker
            if len(moves) > max_moves_per_ticker:
                moves = random.sample(moves, max_moves_per_ticker)

            all_moves.extend(moves)
            logger.info(f" {ticker}: {len(moves)} significant moves found")

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

    articles = []

    #Attempt 1: GDELT with retry + backoff
    GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
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

    for attempt in range(3):
        try: 
            async with httpx.AsyncClient(timeout = 15.0) as client:
                resp = await client.get(GDELT_DOC_API, params = params)
                if resp.status_code == 429:
                    wait = 10.0 * (attempt + 1)
                    logger.debug(f"GDELT 429 for {ticker}/{date}, retrying in {wait}s (attempt {attempt + 1}/3)")
                    await asyncio.sleep(wait)
                    continue 
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("articles", []):
                    title = item.get("title", "")
                    if title:
                        articles.append({
                            "title": title,
                            "source": item.get("domain", "unknown"),
                            "url": item.get("url", ""),
                            "tone": item.get("tone", 0),
                        })
                break
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
            else:
                logger.debug(f"GDELT failed for {ticker}/{date} after 3 attempts: {e}")

    #Attempt 2: NewsAPI (only works for last ~30 days)
    if not articles:
        try:
            days_ago = (datetime.now() - move_date).days
            if days_ago <= 30 and settings.NEWSAPI_KEY:
                from_date = (move_date - timedelta(days = 1)).strftime("%Y-%m-%dT%H:%M:%S")
                to_date = (move_date + timedelta(days = 1)).strftime("%Y-%m-%dT%H:%M:%S")
                async with httpx.AsyncClient(timeout = 15.0) as client:
                    resp = await client.get(
                        "https://newsapi.org/v2/everything",
                        params = {
                            "q": f'"{company_name}" OR "{ticker}"',
                            "from": from_date,
                            "to": to_date,
                            "sortBy": "relevancy",
                            "pageSize": str(max_articles),
                            "language": "en",
                            "apiKey": settings.NEWSAPI_KEY,
                        }
                    )
                    if resp.status_code == 200:
                        for item in resp.json().get("articles", []):
                            title = item.get("title", "")
                            desc = item.get("description", "")
                            if title:
                                articles.append({
                                    "title": title,
                                    "source": item.get("source", {}).get("name", "unknown"),
                                    "url": item.get("url", ""),
                                    "description": desc or "",
                                })
        except Exception as e:
            logger.debug(f"NewsAPI failed for {ticker}/{date}: {e}")

    return articles[:max_articles]
    
#-- Claude Labeling --#

async def label_move_with_claude(
    move: Dict[str, Any],
    articles: List[Dict[str, str]],
) -> Optional[MoveLabelBatch]:
    """
    Using Claude to label the historical move and its associated articles.
    This is the "teacher" that generates training data for the SLM

    When articles are available: label based on article content
    When articles are missing: Claude uses its training knowledge about 
    earnings dates, Fed meetings, geopolitical events, etc. to classify
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
        f"[{i}] ({a['source']}) {a['title']}" + (f" - {a['description'][:200]}" if a.get('descripton') else "")
        for i, a in enumerate(articles)
    ) if articles else "No articles retrieved (API rate limited)."

    #When no articles, ask Claude to use its knowledge
    knowledge_hint = ""
    if not articles:
        knowledge_hint = """
IMPORTANT: No news articles were retrieved due to API rate limits. However, you likely
know from your training data what happened to this stock on or around this date.
Use your knowledge of earnings dates, Fed meetings, geopolitical events, trade policy
announcements, sector rotations, and other market-moving events to make your best
classification. Set confidence based on how certain you are of the cause.
If you genuinely have no idea, use 'unknown' with confidence 0.2."""

    prompt = f"""Label this historical stock move and its associated news articles.

MOVE:
- Ticker: {move['ticker']} ({move['company_name']})
- Date: {move['date']}
- Change: {move['pct_change']}% ({move['direction']})
- Magnitude: {move['move_in_sigma']}σ
{knowledge_hint}

ARTICLES:
{articles_text}

For the overall move, classify the root cause category.
For each article (if any), provide relevance, causality, and sentiment scores.
Be specific with the category - avoid 'unknonw' if you can identify a likely cause."""

    try:
        raw_result: Any = await labeler.ainvoke([
            SystemMessage(content=(
                "You are a financial data labeler creating training data for an ML model. "
                "Be precise and consistent. For classification, choose the single best category. "
                "For sentiment, consider financial implications (layoffs = bearish, "
                "strong earnings = bullish). For relevance, consider whether the article "
                "plausibly explains the specific price move. "
                "Use your knowledge of financial markets to make informed classifications "
                "even when article data is limited."
            )),
            HumanMessage(content=prompt),
        ])

        if isinstance(raw_result, MoveLabelBatch):
            return raw_result
        if isinstance(raw_result, BaseModel):
            return cast(MoveLabelBatch, MoveLabelBatch.model_validate(raw_result.model_dump()))
        if isinstance(raw_result, dict):
            return cast(MoveLabelBatch, MoveLabelBatch.model_validate(raw_result))

        logger.warning(
            "Claude labeling returned unexpected type for %s %s: %s",
            move["ticker"],
            move["date"],
            type(raw_result).__name__,
        )
        return None
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
    news_success = 0
    news_failed = 0

    for i, move in enumerate(moves):
        if (i + 1) % 20 == 0:
            logger.info(
                f"Progress: {i+1}/{len(moves)} moves processed "
                f"({len(cls_examples)} cls, {len(sent_examples)} sent, {len(rel_examples)} rel)"
            )

        #Retrieve historical news
        articles = await retrieve_historical_news(
            move["ticker"], move["company_name"], move["date"],
        )
        if articles:
            news_success += 1
        else:
            news_failed += 1

        #Label with Claude
        labels = await label_move_with_claude(move, articles)
        if not labels:
            continue 

        #Skip "unknown" labels with very low confidence - they add noise
        if labels.move_category == "unknown" and labels.move_confidence < 0.3:
            continue 

        # -- Extract classificaiton examples -- %
        #Use the move description as a classification example
        if articles:
            #use article titles as the training text (this is what the SLM sees at inference)
            article_titles = " | ".join(a["title"] for a in articles[:5])
            move_text = (
                f"{move['company_name']} ({move['ticker']}) {move['pct_change']}% "
                f"{move['direction']} ({move['move_in_sigma']} sigma). "
                f"News: {article_titles}"
            )
        else:
            #Generate context-rich text for Claude labeled moves without articles
            #Include ticker, date, direction, magnitude - the SLM learns from the pattern
            move_text = (
                f"{move['company_name']} ({move['ticker']}) stock moved"
                f"{move['pct_change']}% {move['direction']} on {move['date']}. "
                f"Magnitude: {move['move_in_sigma']} standard deviations"
            )

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
            desc = articles[j].get("description", "")
            if desc:
                article_text = f"{article_text}. {desc[:200]}"

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

        #Rate limit: longer delay if GDELT is struggling
        await asyncio.sleep(2.0 if news_failed > news_success else 0.5)
    logger.info(
        f"News retrieval: {news_success} successful, {news_failed} failed "
        f"({news_success/(news_success + news_failed) * 100:.0f}% hit rate)"
    )

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
