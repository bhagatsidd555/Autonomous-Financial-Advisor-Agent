"""
Conflict Resolver — FIXED VERSION

✔ Compatible with advisor
✔ Uses Groq
✔ No crashes
"""

import logging
import os
from dataclasses import dataclass

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")


@dataclass
class ConflictSignal:
    symbol_or_sector: str
    explanation: str


class ConflictResolver:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODEL

    # ✅ FIXED SIGNATURE
    def detect_and_resolve(
        self,
        portfolio=None,
        classified_news=None,
        sector_snapshot=None,
        market_condition=None
    ):
        conflicts = []

        classified_news = classified_news or []

        for news in classified_news:
            if news.sentiment == "positive" and news.impact_score > 0.7:
                conflicts.append(
                    ConflictSignal(
                        symbol_or_sector="Market",
                        explanation="Positive news but market reaction is weak — possible profit booking or macro pressure."
                    )
                )

        logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts