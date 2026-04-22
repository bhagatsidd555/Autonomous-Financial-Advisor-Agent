"""
src/analytics/market_analyzer.py
==================================
Analyzes overall market conditions by examining index movements,
breadth, and volatility to determine bullish/bearish/neutral sentiment.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    HIGH_VOLATILITY_THRESHOLD,
    SIGNIFICANT_LOSS_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class MarketCondition:
    """Summary of current overall market conditions."""
    sentiment: str                    # bullish | bearish | neutral | volatile
    strength: str                     # strong | moderate | weak
    nifty_change_pct: float
    sensex_change_pct: float
    nifty_bank_change_pct: float
    advance_decline_ratio: float
    volatility_level: str             # low | moderate | high | extreme
    key_signals: list[str]
    raw_indices: dict
    confidence: float                 # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "sentiment": self.sentiment,
            "strength": self.strength,
            "nifty_change_pct": self.nifty_change_pct,
            "sensex_change_pct": self.sensex_change_pct,
            "nifty_bank_change_pct": self.nifty_bank_change_pct,
            "advance_decline_ratio": self.advance_decline_ratio,
            "volatility_level": self.volatility_level,
            "key_signals": self.key_signals,
            "confidence": self.confidence,
        }

    def to_text_summary(self) -> str:
        """Generate human-readable market summary."""
        sentiment_emoji = {
            "bullish": "📈",
            "bearish": "📉",
            "neutral": "➡️",
            "volatile": "⚡",
        }.get(self.sentiment, "")

        lines = [
            f"{sentiment_emoji} Market Sentiment: {self.sentiment.upper()} ({self.strength})",
            f"  • NIFTY 50:    {'+' if self.nifty_change_pct >= 0 else ''}{self.nifty_change_pct:.2f}%",
            f"  • SENSEX:      {'+' if self.sensex_change_pct >= 0 else ''}{self.sensex_change_pct:.2f}%",
            f"  • NIFTY Bank:  {'+' if self.nifty_bank_change_pct >= 0 else ''}{self.nifty_bank_change_pct:.2f}%",
            f"  • Volatility:  {self.volatility_level.upper()}",
        ]

        if self.key_signals:
            lines.append("  Key Signals:")
            for signal in self.key_signals:
                lines.append(f"    → {signal}")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────
class MarketAnalyzer:
    """
    Analyzes market-wide conditions from index data.
    Determines overall sentiment and generates structured summaries
    for the reasoning layer.
    """

    def __init__(self):
        logger.info("MarketAnalyzer initialized")

    def analyze(self, indices_data: dict[str, dict]) -> MarketCondition:
        """
        Main analysis method. Takes raw index data and returns
        a MarketCondition assessment.

        Args:
            indices_data: Output from MarketDataFetcher.fetch_all_indices()

        Returns:
            MarketCondition dataclass
        """
        if not indices_data:
            logger.error("No index data provided for market analysis")
            return self._empty_condition()

        # Extract key index changes
        nifty_chg = self._get_change(indices_data, "NIFTY_50")
        sensex_chg = self._get_change(indices_data, "SENSEX")
        bank_chg = self._get_change(indices_data, "NIFTY_BANK")
        it_chg = self._get_change(indices_data, "NIFTY_IT")
        pharma_chg = self._get_change(indices_data, "NIFTY_PHARMA")
        auto_chg = self._get_change(indices_data, "NIFTY_AUTO")
        fmcg_chg = self._get_change(indices_data, "NIFTY_FMCG")
        metal_chg = self._get_change(indices_data, "NIFTY_METAL")

        # Core indices list for breadth analysis
        core_changes = [nifty_chg, sensex_chg, bank_chg, it_chg, pharma_chg, auto_chg]
        valid_changes = [c for c in core_changes if c is not None]

        # Determine sentiment
        sentiment, strength = self._determine_sentiment(nifty_chg, sensex_chg, valid_changes)

        # Advance/Decline ratio (from sector performance)
        sector_changes = [c for c in [bank_chg, it_chg, pharma_chg, auto_chg, fmcg_chg, metal_chg] if c is not None]
        advancing_sectors = sum(1 for c in sector_changes if c > 0)
        declining_sectors = sum(1 for c in sector_changes if c < 0)
        ad_ratio = (
            advancing_sectors / declining_sectors
            if declining_sectors > 0
            else float(advancing_sectors)
        )

        # Volatility assessment
        volatility = self._assess_volatility(nifty_chg, sensex_chg, valid_changes)

        # Key signals
        signals = self._identify_signals(
            nifty_chg, sensex_chg, bank_chg, it_chg,
            metal_chg, fmcg_chg, pharma_chg, auto_chg
        )

        # Confidence score (based on data availability)
        available = sum(1 for c in [nifty_chg, sensex_chg, bank_chg] if c is not None)
        confidence = min(available / 3, 1.0)

        condition = MarketCondition(
            sentiment=sentiment,
            strength=strength,
            nifty_change_pct=round(nifty_chg or 0, 4),
            sensex_change_pct=round(sensex_chg or 0, 4),
            nifty_bank_change_pct=round(bank_chg or 0, 4),
            advance_decline_ratio=round(ad_ratio, 2),
            volatility_level=volatility,
            key_signals=signals,
            raw_indices=indices_data,
            confidence=round(confidence, 2),
        )

        logger.info(
            "Market analysis complete: %s (%s) | NIFTY: %.2f%%",
            condition.sentiment,
            condition.strength,
            condition.nifty_change_pct,
        )
        return condition

    # ── Helper Methods ───────────────────────
    def _get_change(self, indices_data: dict, index_name: str) -> Optional[float]:
        """Safely extract change_pct for a named index."""
        if index_name in indices_data:
            return indices_data[index_name].get("change_pct")
        return None

    def _determine_sentiment(
        self,
        nifty_chg: Optional[float],
        sensex_chg: Optional[float],
        all_changes: list[float],
    ) -> tuple[str, str]:
        """
        Determine market sentiment and its strength.

        Returns:
            Tuple of (sentiment, strength) strings
        """
        if not all_changes:
            return "neutral", "weak"

        avg_change = sum(all_changes) / len(all_changes)
        primary_change = nifty_chg or avg_change

        # Extreme volatility check
        if abs(primary_change) > 2.5:
            return "volatile", "strong"

        # Bullish
        if primary_change > 0:
            if primary_change > 1.0:
                return "bullish", "strong"
            elif primary_change > 0.4:
                return "bullish", "moderate"
            else:
                return "bullish", "weak"

        # Bearish
        elif primary_change < 0:
            if primary_change < -1.0:
                return "bearish", "strong"
            elif primary_change < -0.4:
                return "bearish", "moderate"
            else:
                return "bearish", "weak"

        # Neutral
        return "neutral", "moderate"

    def _assess_volatility(
        self,
        nifty_chg: Optional[float],
        sensex_chg: Optional[float],
        all_changes: list[float],
    ) -> str:
        """Classify volatility level."""
        if not all_changes:
            return "unknown"

        max_abs = max(abs(c) for c in all_changes)
        avg_abs = sum(abs(c) for c in all_changes) / len(all_changes)

        if max_abs > 3.0 or avg_abs > 2.0:
            return "extreme"
        elif max_abs > 1.5 or avg_abs > 1.0:
            return "high"
        elif max_abs > 0.75 or avg_abs > 0.5:
            return "moderate"
        else:
            return "low"

    def _identify_signals(self, *changes) -> list[str]:
        """
        Generate human-readable key signal strings
        from individual sector/index performances.
        """
        (nifty, sensex, bank, it, metal, fmcg, pharma, auto) = changes
        signals = []

        # Broad market signals
        if nifty and nifty > 1.0:
            signals.append(f"NIFTY 50 up strongly ({nifty:+.2f}%) — broad-based buying")
        elif nifty and nifty < -1.0:
            signals.append(f"NIFTY 50 down sharply ({nifty:+.2f}%) — broad selling pressure")

        if sensex and nifty and abs(sensex - nifty) > 0.5:
            signals.append("SENSEX and NIFTY diverging — selective sectoral movement")

        # Sector-specific signals
        if bank is not None:
            if bank > 1.5:
                signals.append(f"Banking sector leading rally ({bank:+.2f}%)")
            elif bank < -1.5:
                signals.append(f"Banking sector dragging market ({bank:+.2f}%)")

        if it is not None:
            if it > 2.0:
                signals.append(f"IT sector outperforming ({it:+.2f}%) — likely USD/global cue")
            elif it < -2.0:
                signals.append(f"IT sector underperforming ({it:+.2f}%) — US macro concerns")

        if metal is not None and abs(metal) > 2.0:
            signals.append(
                f"Metal sector {'surging' if metal > 0 else 'falling'} ({metal:+.2f}%) — commodity cycle signal"
            )

        if fmcg is not None and fmcg > 0.5 and nifty and nifty < 0:
            signals.append("FMCG holding up in falling market — defensive rotation")

        if pharma is not None and pharma > 1.5:
            signals.append(f"Pharma sector rallying ({pharma:+.2f}%) — defensive buying")

        if auto is not None and auto < -1.5:
            signals.append(f"Auto sector under pressure ({auto:+.2f}%) — demand/EV concerns")

        if not signals:
            signals.append("Mixed signals — no dominant theme")

        return signals

    def _empty_condition(self) -> MarketCondition:
        """Return empty condition when no data available."""
        return MarketCondition(
            sentiment="unknown",
            strength="unknown",
            nifty_change_pct=0.0,
            sensex_change_pct=0.0,
            nifty_bank_change_pct=0.0,
            advance_decline_ratio=1.0,
            volatility_level="unknown",
            key_signals=["Insufficient data to assess market conditions"],
            raw_indices={},
            confidence=0.0,
        )
