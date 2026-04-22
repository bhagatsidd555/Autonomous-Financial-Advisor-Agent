"""
src/ingestion/news_fetcher.py
==============================
Fetches financial news from multiple sources including RSS feeds
and NewsAPI. Provides structured news items ready for classification
and sentiment analysis.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import feedparser

from config.settings import NEWS_RSS_FEEDS, NEWS_API_KEY, MAX_NEWS_ITEMS

logger = logging.getLogger(__name__)


class NewsItem:
    """Represents a single financial news article."""

    def __init__(
        self,
        headline: str,
        summary: str,
        source: str,
        published_at: str,
        category: str = "general",
        tags: list[str] = None,
        url: str = "",
    ):
        self.id = hashlib.md5(headline.encode()).hexdigest()[:12]
        self.headline = headline
        self.summary = summary
        self.source = source
        self.published_at = published_at
        self.category = category
        self.tags = tags or []
        self.url = url

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at,
            "category": self.category,
            "tags": self.tags,
            "url": self.url,
        }

    def __repr__(self):
        return f"NewsItem(id={self.id}, headline='{self.headline[:60]}...')"


class NewsFetcher:
    """
    Fetches financial news from multiple data sources:
    1. RSS feeds (Economic Times, Moneycontrol, etc.)
    2. NewsAPI (if API key provided)
    3. Sample/local data fallback for development
    """

    def __init__(self, use_sample_fallback: bool = True):
        self.use_sample_fallback = use_sample_fallback
        self.sample_data_path = Path(__file__).parent.parent.parent / "data" / "sample_news.json"
        logger.info("NewsFetcher initialized (fallback=%s)", use_sample_fallback)

    # ── RSS Feed Fetching ────────────────────
    def fetch_from_rss(self, feed_name: str, feed_url: str, max_items: int = 5) -> list[NewsItem]:
        """
        Parse a single RSS feed and return list of NewsItem objects.

        Args:
            feed_name: Human-readable name (e.g., 'economic_times')
            feed_url: RSS feed URL
            max_items: Maximum articles to return

        Returns:
            List of NewsItem objects
        """
        items = []
        try:
            logger.info("Fetching RSS feed: %s", feed_name)
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning("Feed parse warning for %s: %s", feed_name, feed.bozo_exception)

            for entry in feed.entries[:max_items]:
                headline = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()

                # Clean HTML tags from summary (basic cleanup)
                import re
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                summary = summary[:500] if len(summary) > 500 else summary

                published = entry.get("published", datetime.now().isoformat())

                item = NewsItem(
                    headline=headline,
                    summary=summary,
                    source=feed_name.replace("_", " ").title(),
                    published_at=str(published),
                    category="market_news",
                    url=entry.get("link", ""),
                )
                items.append(item)

        except Exception as e:
            logger.error("Error fetching RSS feed %s: %s", feed_name, str(e))

        logger.info("Fetched %d articles from %s", len(items), feed_name)
        return items

    def fetch_all_rss_feeds(self) -> list[NewsItem]:
        """Fetch from all configured RSS feeds."""
        all_items = []
        per_feed = max(2, MAX_NEWS_ITEMS // len(NEWS_RSS_FEEDS))

        for name, url in NEWS_RSS_FEEDS.items():
            items = self.fetch_from_rss(name, url, per_feed)
            all_items.extend(items)

        # Deduplicate by headline similarity
        seen_headlines = set()
        unique_items = []
        for item in all_items:
            key = item.headline[:50].lower()
            if key not in seen_headlines:
                seen_headlines.add(key)
                unique_items.append(item)

        return unique_items[:MAX_NEWS_ITEMS]

    # ── NewsAPI Fetching ─────────────────────
    def fetch_from_newsapi(
        self,
        query: str = "Indian stock market NIFTY NSE BSE",
        days_back: int = 1,
    ) -> list[NewsItem]:
        """
        Fetch news from NewsAPI.org (requires API key).

        Args:
            query: Search query
            days_back: How many days back to search

        Returns:
            List of NewsItem objects
        """
        if not NEWS_API_KEY:
            logger.warning("No NEWS_API_KEY configured; skipping NewsAPI fetch")
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        endpoint = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": MAX_NEWS_ITEMS,
            "apiKey": NEWS_API_KEY,
        }

        items = []
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for article in data.get("articles", []):
                if not article.get("title") or article["title"] == "[Removed]":
                    continue
                item = NewsItem(
                    headline=article["title"],
                    summary=article.get("description", "") or "",
                    source=article.get("source", {}).get("name", "Unknown"),
                    published_at=article.get("publishedAt", ""),
                    category="market_news",
                    url=article.get("url", ""),
                )
                items.append(item)

            logger.info("Fetched %d articles from NewsAPI", len(items))

        except requests.exceptions.RequestException as e:
            logger.error("NewsAPI request failed: %s", str(e))
        except Exception as e:
            logger.error("Unexpected error in NewsAPI fetch: %s", str(e))

        return items

    # ── Sample Data Fallback ─────────────────
    def load_sample_news(self) -> list[NewsItem]:
        """Load news from local sample JSON file (for development/testing)."""
        try:
            with open(self.sample_data_path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)

            items = []
            for raw in raw_items:
                item = NewsItem(
                    headline=raw["headline"],
                    summary=raw["summary"],
                    source=raw["source"],
                    published_at=raw["published_at"],
                    category=raw.get("category", "general"),
                    tags=raw.get("tags", []),
                )
                items.append(item)

            logger.info("Loaded %d sample news items", len(items))
            return items

        except FileNotFoundError:
            logger.warning("Sample news file not found at %s", self.sample_data_path)
            return []
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in sample news file: %s", str(e))
            return []

    # ── Main Fetch Orchestrator ──────────────
    def fetch_latest_news(
        self,
        use_rss: bool = True,
        use_newsapi: bool = True,
    ) -> list[NewsItem]:
        """
        Main method to fetch news from all available sources.
        Falls back to sample data in development environments.

        Returns:
            Consolidated, deduplicated list of NewsItem objects
        """
        all_items: list[NewsItem] = []

        # Try RSS feeds
        if use_rss:
            rss_items = self.fetch_all_rss_feeds()
            all_items.extend(rss_items)

        # Try NewsAPI
        if use_newsapi and NEWS_API_KEY:
            api_items = self.fetch_from_newsapi()
            all_items.extend(api_items)

        # Use sample fallback if no live data obtained
        if not all_items and self.use_sample_fallback:
            logger.info("No live news fetched; using sample data fallback")
            all_items = self.load_sample_news()

        if not all_items:
            logger.warning("No news data available from any source")
            return []

        # Deduplicate
        seen = set()
        unique = []
        for item in all_items:
            key = item.headline[:60].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        logger.info("Total unique news items: %d", len(unique))
        return unique[:MAX_NEWS_ITEMS]

    def extract_stocks_mentioned(self, news_item: NewsItem) -> list[str]:
        """
        Extract stock symbols mentioned in a news item.
        Matches against known stock names and symbols.

        Returns:
            List of stock symbols (e.g., ['TCS.NS', 'INFY.NS'])
        """
        from config.settings import STOCK_SECTOR_MAP

        mentioned = []
        text = f"{news_item.headline} {news_item.summary}".lower()

        stock_name_map = {
            "tcs": "TCS.NS",
            "infosys": "INFY.NS",
            "wipro": "WIPRO.NS",
            "hdfc bank": "HDFCBANK.NS",
            "hdfcbank": "HDFCBANK.NS",
            "icici bank": "ICICIBANK.NS",
            "sbi": "SBIN.NS",
            "state bank": "SBIN.NS",
            "reliance": "RELIANCE.NS",
            "sun pharma": "SUNPHARMA.NS",
            "sunpharma": "SUNPHARMA.NS",
            "maruti": "MARUTI.NS",
            "tatasteel": "TATASTEEL.NS",
            "tata steel": "TATASTEEL.NS",
            "airtel": "BHARTIARTL.NS",
            "bharti airtel": "BHARTIARTL.NS",
            "itc": "ITC.NS",
            "kotak": "KOTAKBANK.NS",
            "axis bank": "AXISBANK.NS",
            "bajaj finance": "BAJFINANCE.NS",
        }

        for keyword, symbol in stock_name_map.items():
            if keyword in text:
                if symbol not in mentioned:
                    mentioned.append(symbol)

        return mentioned

    def extract_sectors_mentioned(self, news_item: NewsItem) -> list[str]:
        """
        Extract sector names mentioned in a news item.

        Returns:
            List of sector names
        """
        sector_keywords = {
            "Banking": ["bank", "banking", "rbi", "nbfc", "credit", "loan", "deposit"],
            "IT": ["it sector", "software", "tech", "technology", "infosys", "tcs", "wipro"],
            "Pharma": ["pharma", "drug", "medicine", "healthcare", "fda", "cipla"],
            "Auto": ["auto", "automobile", "car", "vehicle", "ev", "electric vehicle"],
            "Energy": ["oil", "gas", "petrol", "refinery", "power", "energy", "coal"],
            "FMCG": ["fmcg", "consumer goods", "fmcg sector", "fast moving"],
            "Metal": ["metal", "steel", "aluminium", "copper", "iron"],
            "Telecom": ["telecom", "5g", "spectrum", "jio", "airtel"],
            "Finance": ["nbfc", "insurance", "mutual fund", "stock exchange", "sebi"],
        }

        text = f"{news_item.headline} {news_item.summary}".lower()
        mentioned = []

        for sector, keywords in sector_keywords.items():
            if any(kw in text for kw in keywords):
                mentioned.append(sector)

        return list(set(mentioned))
