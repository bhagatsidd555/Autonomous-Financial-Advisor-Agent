"""
News Classifier (FINAL STABLE VERSION)

✔ Groq LLM support
✔ No crash on API failure
✔ Safe JSON parsing
✔ Handles missing fields
✔ Compatible with financial_advisor.py
✔ Production-ready
✔ Observability added correctly
"""

import json
import logging
from dataclasses import dataclass, field
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class ClassifiedNews:
    id: str
    title: str
    summary: str
    sentiment: str = "neutral"
    scope: str = "market_wide"
    affected_sectors: list = field(default_factory=list)
    affected_stocks: list = field(default_factory=list)
    impact_score: float = 0.5
    is_portfolio_relevant: bool = False
    reasoning: str = ""


# ─────────────────────────────────────────────
# SAFE HELPER
# ─────────────────────────────────────────────

def safe_get(obj, attr, default=""):
    try:
        val = getattr(obj, attr, None)
        return val if val else default
    except Exception:
        return default


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class NewsClassifier:

    def __init__(self):
        if not GROQ_API_KEY:
            logger.warning("⚠ GROQ_API_KEY missing — running in fallback mode")
            self.client = None
        else:
            self.client = Groq(api_key=GROQ_API_KEY)

        self.model = MODEL

    # ✅ MAIN FUNCTION (FIXED)
    def classify_batch(
        self,
        news_items,
        portfolio_symbols=None,
        portfolio_sectors=None,
        portfolio_stocks=None,
        **kwargs
    ):
        # Fix alias issue
        if portfolio_stocks and not portfolio_symbols:
            portfolio_symbols = portfolio_stocks

        portfolio_symbols = portfolio_symbols or []
        portfolio_sectors = portfolio_sectors or []

        # ✅ OBSERVABILITY (correct place)
        logger.info(f"[OBSERVABILITY] Classified {len(news_items)} news articles")

        results = []

        for item in news_items:
            try:
                classified = self._classify_one(
                    item,
                    portfolio_symbols,
                    portfolio_sectors
                )
                results.append(classified)
            except Exception as e:
                logger.error(f"❌ Failed to classify item: {e}")

                results.append(
                    ClassifiedNews(
                        id="fallback",
                        title=safe_get(item, "title"),
                        summary=safe_get(item, "summary"),
                    )
                )

        logger.info(f"✔ Classified {len(results)} news items")
        return results

    # ✅ FILTER FUNCTION
    def filter_portfolio_relevant(self, classified_news, portfolio_sectors, portfolio_symbols):
        return [
            n for n in classified_news
            if n.is_portfolio_relevant
        ]

    # ─────────────────────────────────────────
    # INTERNAL METHODS
    # ─────────────────────────────────────────

    def _classify_one(self, item, portfolio_symbols, portfolio_sectors):
        title = safe_get(item, "title") or safe_get(item, "headline")
        summary = safe_get(item, "summary") or safe_get(item, "content")

        # Fallback if no LLM
        if not self.client:
            return ClassifiedNews(
                id=safe_get(item, "id", "news"),
                title=title,
                summary=summary,
                sentiment="neutral",
                scope="market_wide",
                impact_score=0.5,
                is_portfolio_relevant=True
            )

        prompt = f"""
Analyze this financial news.

Title: {title}
Summary: {summary}

Return ONLY JSON:
{{
 "sentiment": "positive/negative/neutral",
 "scope": "market_wide/sector_specific/stock_specific",
 "affected_sectors": [],
 "affected_stocks": [],
 "impact_score": 0-1,
 "reasoning": "short reason"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            raw = response.choices[0].message.content.strip()
            data = self._parse_json(raw)

        except Exception as e:
            logger.warning(f"⚠ LLM failed: {e}")
            data = {}

        return ClassifiedNews(
            id=safe_get(item, "id", "news"),
            title=title,
            summary=summary,
            sentiment=data.get("sentiment", "neutral"),
            scope=data.get("scope", "market_wide"),
            affected_sectors=data.get("affected_sectors", []),
            affected_stocks=data.get("affected_stocks", []),
            impact_score=self._safe_float(data.get("impact_score", 0.5)),
            is_portfolio_relevant=self._check_relevance(
                data,
                portfolio_symbols,
                portfolio_sectors
            ),
            reasoning=data.get("reasoning", "")
        )

    def _parse_json(self, raw):
        try:
            return json.loads(raw)
        except Exception:
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                return json.loads(raw[start:end])
            except Exception:
                return {}

    def _safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.5

    def _check_relevance(self, data, symbols, sectors):
        try:
            if data.get("scope") == "market_wide":
                return True

            for s in data.get("affected_sectors", []):
                if s in sectors:
                    return True

            for stock in data.get("affected_stocks", []):
                if stock in symbols:
                    return True

        except Exception:
            pass

        return False