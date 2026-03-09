"""
News Retrieval Client for Financial Intelligence System
Fetches news articles related to specific flagged tickers from GDELT and NewsAPI
"""

import logging
import hashlib
import asyncio 
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional 

import httpx
import yfinance as yf

from Data.data_model import (
    PriceMove, NewsArticle, TickerNewsBundle, EventSource
)

from Data.settings import Settings, TICKER_COMPANY_NAMES

logger = logging.getLogger(__name__)

class NewsRetriever:
    """
    Retrieves news articles for flagged tickers from GDELT API and NewsAPI
    Searches by company name and ticker symbol to maximize recall
    """
    GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    def __init__(self):
        self.settings = Settings()
        self.client = httpx.AsyncClient(timeout = 30.0, follow_redirects = True)
        self._seen_urls: set = set() 

    async def close(self):
        await self.client.aclose() 

    # -- Main Entry Point -- #

    async def retrieve_news_for_moves(
            self, moves: List[PriceMove], lookback_hours: Optional[int] = None,
            max_per_ticker: Optional[int] = None,
    ) -> List[TickerNewsBundle]:
        """
        For each flagged price move, retrieve related news articles.
        Returns a TickerNewsBundle per move
        """

        lookback = lookback_hours or self.settings.NEWS_LOOKBACK_HOURS
        max_articles = max_per_ticker or self.settings.MAX_NEWS_PER_TICKER

        #Deduplicate tickers (a ticker might have both daily and weekly moves)
        seen_tickers = set() 
        unique_moves: List[PriceMove] = []
        for move in moves:
            if move.ticker not in seen_tickers:
                seen_tickers.add(move.ticker)
                unique_moves.append(move)

        logger.info(f"Retrieving news for {len(unique_moves)} unique tickers (lookback = {lookback}h)")

        bundles = []

        #Process sequentially with delay to avoid getting previous GDELT 429 rate limits
        for i, move in enumerate(unique_moves):
            try:
                result = await self._retrieve_for_ticker(move, lookback, max_articles)
                if isinstance(result, TickerNewsBundle):
                    bundles.append(result)
            except Exception as e:
                logger.warning(f"News retrieval error for {move.ticker}: {e}")

            #Staggering requests to stay under rate limit
            if i < len(unique_moves) - 1:
                await asyncio.sleep(1.5)

        total_articles = sum(b.article_count for b in bundles)
        logger.info(f"Retrieved {total_articles} total articles for {len(bundles)} tickers")
        return bundles
    
    async def _retrieve_for_ticker(
            self, move: PriceMove, lookback_hours: int, max_articles:int
    ) -> TickerNewsBundle:
        """
        Retrieve news for a single ticker from all sources
        """
        ticker = move.ticker
        company_name = move.company_name or self._resolve_company_name(ticker)
        search_terms = self._build_search_terms(ticker, company_name)

        articles: List[NewsArticle] = []

        #GDELT
        gdelt_articles = await self._search_gdelt(
            search_terms, ticker, lookback_hours, max_articles
        )
        articles.extend(gdelt_articles)

        #NewsAPI
        if self.settings.NEWSAPI_KEY:
            newsapi_articles = await self._search_newsapi(
                search_terms, ticker, lookback_hours, max_articles
            )
            articles.extend(newsapi_articles)

        #Deduplicate by URL hash 
        unique_articles = self._deduplicate(articles)

        #Sort by relevance then recency
        unique_articles.sort(key = lambda a: (-a.relevance_score, -a.published_at.timestamp()))

        final = unique_articles[:max_articles]

        return TickerNewsBundle(
            ticker = ticker,
            company_name = company_name,
            move = move,
            articles = final,
            article_count = len(final),
        )
    
    # -- GDELT Search --#
    async def _search_gdelt(
            self, search_terms: List[str], ticker: str,
            lookback_hours: int, max_articles: int,
    ) -> List[NewsArticle]:
        """ Search GDELT API for articles matching the company/ticker"""
        #GDELT query: OR together all search terms
        query = " OR ".join(f'"{term}"' for term in search_terms)

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(min(max_articles * 2, 100)), #fetch extra, filter later
            "timespan": f"{lookback_hours * 60} min", 
            "format": "json",
            "sort": "DateDesc",
        }

        try:
            resp = await self.client.get(self.GDELT_DOC_API, params = params)
            resp.raise_for_status()
            data = resp.json() 

            articles = []
            for item in data.get("articles", []):
                try:
                    pub_date = datetime.strptime(
                        item.get("seendate", ""), "%Y%m%dT%H%M%SZ"
                    ).replace(tzinfo = timezone.utc)
                except (ValueError, TypeError):
                    pub_date = datetime.now(timezone.utc)

                title = item.get("title", "")
                relevance = self._compute_relevance(title, search_terms, ticker)
                try:
                    # GDELT tone is roughly -10..10; normalize to NewsArticle's -1..1.
                    sentiment = max(-1.0, min(1.0, float(item.get("tone", 0)) / 10.0))
                except (TypeError, ValueError):
                    sentiment = 0.0

                articles.append(NewsArticle(
                    url = item.get("url", ""),
                    title = title,
                    description = title, #GDELT only returns titles
                    source_name = item.get("domain", "unknown"),
                    source = EventSource.GDELT,
                    published_at = pub_date,
                    sentiment_score = sentiment,
                    relevance_score = relevance,
                    matched_ticker = ticker,
                ))
            return articles 
        
        except httpx.HTTPError as e:
            logger.warning(f"GDELT search failed for {ticker}: {e}")
            return []
        except Exception as e:
            logger.warning(f"GDELT parse error for {ticker}: {e}")
            return []
        
    # -- NewsAPI Search --#
    async def _search_newsapi(
            self, search_terms: List[str], ticker: str,
            lookback_hours: int, max_articles: int,
    ) -> List[NewsArticle]:
        """ Search NewsAPI for articles matching the company/ticker"""
        from_time = datetime.now(timezone.utc) - timedelta(hours = lookback_hours)

        #NewsAPI query: primary company name + ticker
        query = f'"{search_terms[0]}" OR "{ticker}"'

        params = {
            "q": query,
            "from": from_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 50),
            "language": "en",
            "apiKey": self.settings.NEWSAPI_KEY,
        }

        try:
            resp = await self.client.get(self.NEWSAPI_URL, params = params)
            resp.raise_for_status()
            data = resp.json()

            articles = []
            for item in data.get("articles", []):
                try:
                    pub_date = datetime.fromisoformat(
                        item.get("publishedAt", "").replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pub_date = datetime.now(timezone.utc)

                title = item.get("title", "")
                desc = item.get("description", "") or ""
                text = f"{title} {desc}"
                relevance = self._compute_relevance(text, search_terms, ticker)

                articles.append(NewsArticle(
                    url = item.get("url", ""),
                    title = title,
                    description = desc[:500],
                    source_name = item.get("source", {}).get("name", "unknown"),
                    source = EventSource.NEWSAPI,
                    published_at = pub_date,
                    relevance_score = relevance,
                    matched_ticker = ticker,
                ))

            return articles 
        except Exception as e:
            logger.warning(f"NewsAPI search failed for {ticker}: {e}")
            return []
        
    #-- Helpers --#

    @staticmethod
    def _build_search_terms(ticker: str, company_name: str) -> List[str]:
        """ Build search terms for a ticker/company"""
        terms = []
        if company_name and company_name != ticker:
            #Primary: company name
            terms.append(company_name)
        #Ticker symbol (e.g., "AAPL stock")
        terms.append(f"{ticker}")
        # "$TICKER" format common in financial media 
        terms.append(f"${ticker}")
        return terms 
    
    @staticmethod
    def _resolve_company_name(ticker: str) -> str:
        """ Resolve ticker to company name for search"""
        if ticker in TICKER_COMPANY_NAMES:
            return TICKER_COMPANY_NAMES[ticker]
        try:
            info = yf.Ticker(ticker).info
            return info.get("longName", info.get("shortName", ticker))
        except Exception:
            return ticker 
        
    @staticmethod
    def _compute_relevance(text: str, search_terms: List[str], ticker: str) -> float:
        """Compute 0-1 relevance score based on term presence in text"""
        text_lower = text.lower()
        score = 0.0

        #Ticker mentioned directly (high signal)
        if ticker.lower() in text_lower or f"${ticker.lower()}" in text_lower:
            score += 0.5

        #Company name mentioned
        for term in search_terms:
            if term.lower() in text_lower:
                score += 0.3
                break 

        #Financial context keywords
        financial_terms = [
            "stock", "share", "market", "trading", "investor",
            "earnings", "revenue", "profit", "loss", "analyst",
            "upgrade", "downgrade", "target", "rally", "drop",
        ]

        matches = sum(1 for ft in financial_terms if ft in text_lower)
        score += min(0.2, matches * 0.05)

        return min(1.0, score)
    
    def _deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """ Remove duplicate articles by URL"""
        unique = []
        for article in articles:
            url_hash = hashlib.md5(article.url.encode()).hexdigest()
            if url_hash not in self._seen_urls:
                self._seen_urls.add(url_hash)
                unique.append(article)
        return unique
    
