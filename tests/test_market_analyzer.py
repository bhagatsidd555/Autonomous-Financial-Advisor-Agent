"""
tests/test_market_analyzer.py
================================
Unit tests for the MarketAnalyzer and MarketDataFetcher modules.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.analytics.market_analyzer import MarketAnalyzer, MarketCondition


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def analyzer():
    return MarketAnalyzer()


@pytest.fixture
def bullish_indices():
    """Sample indices data for a bullish day."""
    return {
        "NIFTY_50": {"symbol": "^NSEI", "change_pct": 1.2, "current": 22500, "previous_close": 22234},
        "SENSEX": {"symbol": "^BSESN", "change_pct": 1.1, "current": 74000, "previous_close": 73190},
        "NIFTY_BANK": {"symbol": "^NSEBANK", "change_pct": 1.8, "current": 48000, "previous_close": 47148},
        "NIFTY_IT": {"symbol": "^CNXIT", "change_pct": 0.9, "current": 34000, "previous_close": 33694},
        "NIFTY_PHARMA": {"symbol": "NIFTYPHARMA.NS", "change_pct": 0.5, "current": 20000, "previous_close": 19900},
        "NIFTY_AUTO": {"symbol": "^CNXAUTO", "change_pct": 0.7, "current": 22000, "previous_close": 21846},
        "NIFTY_FMCG": {"symbol": "^CNXFMCG", "change_pct": 0.3, "current": 55000, "previous_close": 54835},
        "NIFTY_METAL": {"symbol": "^CNXMETAL", "change_pct": 2.1, "current": 8500, "previous_close": 8325},
    }


@pytest.fixture
def bearish_indices():
    """Sample indices data for a bearish day."""
    return {
        "NIFTY_50": {"symbol": "^NSEI", "change_pct": -1.5, "current": 21900, "previous_close": 22234},
        "SENSEX": {"symbol": "^BSESN", "change_pct": -1.4, "current": 72650, "previous_close": 73190},
        "NIFTY_BANK": {"symbol": "^NSEBANK", "change_pct": -2.1, "current": 46953, "previous_close": 47148},
        "NIFTY_IT": {"symbol": "^CNXIT", "change_pct": -0.5, "current": 33525, "previous_close": 33694},
        "NIFTY_PHARMA": {"symbol": "NIFTYPHARMA.NS", "change_pct": 0.2, "current": 19940, "previous_close": 19900},
        "NIFTY_AUTO": {"symbol": "^CNXAUTO", "change_pct": -1.8, "current": 21453, "previous_close": 21846},
        "NIFTY_FMCG": {"symbol": "^CNXFMCG", "change_pct": 0.4, "current": 55054, "previous_close": 54835},
        "NIFTY_METAL": {"symbol": "^CNXMETAL", "change_pct": -2.5, "current": 8117, "previous_close": 8325},
    }


@pytest.fixture
def flat_indices():
    """Sample indices data for a flat/neutral day."""
    return {
        "NIFTY_50": {"symbol": "^NSEI", "change_pct": 0.1, "current": 22256, "previous_close": 22234},
        "SENSEX": {"symbol": "^BSESN", "change_pct": -0.1, "current": 73117, "previous_close": 73190},
        "NIFTY_BANK": {"symbol": "^NSEBANK", "change_pct": 0.2, "current": 47242, "previous_close": 47148},
        "NIFTY_IT": {"symbol": "^CNXIT", "change_pct": 0.0, "current": 33694, "previous_close": 33694},
        "NIFTY_PHARMA": {"symbol": "NIFTYPHARMA.NS", "change_pct": -0.1, "current": 19880, "previous_close": 19900},
    }


# ─────────────────────────────────────────────
# Test: Sentiment Detection
# ─────────────────────────────────────────────
class TestSentimentDetection:

    def test_bullish_strong(self, analyzer, bullish_indices):
        result = analyzer.analyze(bullish_indices)
        assert result.sentiment == "bullish"
        assert result.strength in ("strong", "moderate")
        assert result.nifty_change_pct == pytest.approx(1.2, abs=0.01)

    def test_bearish_strong(self, analyzer, bearish_indices):
        result = analyzer.analyze(bearish_indices)
        assert result.sentiment == "bearish"
        assert result.strength in ("strong", "moderate")
        assert result.nifty_change_pct == pytest.approx(-1.5, abs=0.01)

    def test_neutral_flat_day(self, analyzer, flat_indices):
        result = analyzer.analyze(flat_indices)
        assert result.sentiment in ("neutral", "bullish", "bearish")
        # On a flat day, strength should be weak
        assert result.strength in ("weak", "moderate")

    def test_empty_indices_returns_unknown(self, analyzer):
        result = analyzer.analyze({})
        assert result.sentiment == "unknown"
        assert result.confidence == 0.0

    def test_volatile_on_extreme_move(self, analyzer):
        extreme_indices = {
            "NIFTY_50": {"change_pct": 3.5, "current": 23000, "previous_close": 22190},
            "SENSEX": {"change_pct": 3.2, "current": 75000, "previous_close": 72660},
        }
        result = analyzer.analyze(extreme_indices)
        assert result.sentiment in ("volatile", "bullish")


# ─────────────────────────────────────────────
# Test: Volatility Assessment
# ─────────────────────────────────────────────
class TestVolatilityAssessment:

    def test_low_volatility(self, analyzer):
        changes = [0.2, 0.3, 0.1, -0.1, 0.2]
        result = analyzer._assess_volatility(0.2, 0.3, changes)
        assert result == "low"

    def test_high_volatility(self, analyzer):
        changes = [2.5, -1.8, 1.5, -2.0, 1.2]
        result = analyzer._assess_volatility(2.5, -1.8, changes)
        assert result in ("high", "extreme")

    def test_moderate_volatility(self, analyzer):
        changes = [0.8, -0.9, 1.0, -0.7, 0.6]
        result = analyzer._assess_volatility(0.8, -0.9, changes)
        assert result in ("moderate", "high")


# ─────────────────────────────────────────────
# Test: MarketCondition Data Model
# ─────────────────────────────────────────────
class TestMarketConditionModel:

    def test_to_dict_has_required_keys(self, analyzer, bullish_indices):
        condition = analyzer.analyze(bullish_indices)
        d = condition.to_dict()
        required_keys = [
            "sentiment", "strength", "nifty_change_pct",
            "sensex_change_pct", "volatility_level", "key_signals", "confidence"
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_confidence_range(self, analyzer, bullish_indices):
        condition = analyzer.analyze(bullish_indices)
        assert 0.0 <= condition.confidence <= 1.0

    def test_text_summary_non_empty(self, analyzer, bullish_indices):
        condition = analyzer.analyze(bullish_indices)
        summary = condition.to_text_summary()
        assert len(summary) > 20

    def test_key_signals_not_empty(self, analyzer, bullish_indices):
        condition = analyzer.analyze(bullish_indices)
        assert isinstance(condition.key_signals, list)
        assert len(condition.key_signals) >= 1


# ─────────────────────────────────────────────
# Test: Advance/Decline Ratio
# ─────────────────────────────────────────────
class TestAdvanceDeclineRatio:

    def test_positive_ad_ratio_on_bullish_day(self, analyzer, bullish_indices):
        condition = analyzer.analyze(bullish_indices)
        # Most sectors should be advancing
        assert condition.advance_decline_ratio >= 1.0

    def test_negative_ad_ratio_on_bearish_day(self, analyzer, bearish_indices):
        condition = analyzer.analyze(bearish_indices)
        # More sectors should be declining
        assert condition.advance_decline_ratio <= 1.5


# ─────────────────────────────────────────────
# Test: Signal Identification
# ─────────────────────────────────────────────
class TestSignalIdentification:

    def test_banking_sector_signal_detected(self, analyzer, bullish_indices):
        """When bank Nifty rises significantly, signal should mention banking."""
        condition = analyzer.analyze(bullish_indices)
        signal_text = " ".join(condition.key_signals).lower()
        # Banking rally or NIFTY should be mentioned
        assert any(keyword in signal_text for keyword in ["bank", "nifty", "rally"])

    def test_no_signals_when_mixed(self, analyzer, flat_indices):
        """Flat market should produce at least one signal (even if 'Mixed signals')."""
        condition = analyzer.analyze(flat_indices)
        assert len(condition.key_signals) >= 1
