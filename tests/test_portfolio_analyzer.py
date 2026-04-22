"""
tests/test_portfolio_analyzer.py
===================================
Unit tests for PortfolioLoader and PortfolioAnalyzer modules.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from src.ingestion.portfolio_loader import PortfolioLoader, Portfolio, Holding, UserProfile
from src.analytics.portfolio_analyzer import PortfolioAnalyzer, PortfolioAnalysis, RiskAlert
from config.settings import CONCENTRATION_RISK_THRESHOLD


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def sample_portfolio_dict():
    return {
        "user": {
            "id": "test_001",
            "name": "Test User",
            "risk_profile": "moderate",
            "investment_goal": "wealth_creation",
            "investment_horizon_years": 5,
        },
        "portfolio": {
            "total_invested": 500000,
            "cash_balance": 10000,
            "holdings": [
                {
                    "symbol": "HDFCBANK.NS",
                    "name": "HDFC Bank",
                    "quantity": 50,
                    "avg_buy_price": 1600.0,
                    "sector": "Banking",
                    "asset_type": "equity",
                },
                {
                    "symbol": "TCS.NS",
                    "name": "TCS",
                    "quantity": 20,
                    "avg_buy_price": 3800.0,
                    "sector": "IT",
                    "asset_type": "equity",
                },
                {
                    "symbol": "RELIANCE.NS",
                    "name": "Reliance",
                    "quantity": 30,
                    "avg_buy_price": 2900.0,
                    "sector": "Energy",
                    "asset_type": "equity",
                },
            ],
        },
    }


@pytest.fixture
def loaded_portfolio(sample_portfolio_dict):
    loader = PortfolioLoader()
    return loader.load_from_dict(sample_portfolio_dict)


@pytest.fixture
def sample_market_data():
    return {
        "HDFCBANK.NS": {
            "current": 1650.0,
            "previous_close": 1600.0,
            "change": 50.0,
            "change_pct": 3.125,
        },
        "TCS.NS": {
            "current": 3700.0,
            "previous_close": 3800.0,
            "change": -100.0,
            "change_pct": -2.631,
        },
        "RELIANCE.NS": {
            "current": 2950.0,
            "previous_close": 2900.0,
            "change": 50.0,
            "change_pct": 1.724,
        },
    }


@pytest.fixture
def enriched_portfolio(loaded_portfolio, sample_market_data):
    loader = PortfolioLoader()
    return loader.enrich_with_market_data(loaded_portfolio, sample_market_data)


@pytest.fixture
def analyzer():
    return PortfolioAnalyzer()


# ─────────────────────────────────────────────
# Test: Portfolio Loading
# ─────────────────────────────────────────────
class TestPortfolioLoader:

    def test_load_from_dict(self, sample_portfolio_dict):
        loader = PortfolioLoader()
        portfolio = loader.load_from_dict(sample_portfolio_dict)
        assert portfolio is not None
        assert len(portfolio.holdings) == 3
        assert portfolio.user.name == "Test User"

    def test_user_profile_loaded_correctly(self, loaded_portfolio):
        assert loaded_portfolio.user.risk_profile == "moderate"
        assert loaded_portfolio.user.investment_goal == "wealth_creation"
        assert loaded_portfolio.user.investment_horizon_years == 5

    def test_holdings_symbols(self, loaded_portfolio):
        symbols = loaded_portfolio.get_symbols()
        assert "HDFCBANK.NS" in symbols
        assert "TCS.NS" in symbols
        assert "RELIANCE.NS" in symbols

    def test_cash_balance_loaded(self, loaded_portfolio):
        assert loaded_portfolio.cash_balance == 10000.0

    def test_get_holding_by_symbol(self, loaded_portfolio):
        holding = loaded_portfolio.get_holding("TCS.NS")
        assert holding is not None
        assert holding.name == "TCS"
        assert holding.quantity == 20

    def test_get_holding_missing_returns_none(self, loaded_portfolio):
        holding = loaded_portfolio.get_holding("UNKNOWN.NS")
        assert holding is None

    def test_missing_user_raises_error(self):
        loader = PortfolioLoader()
        with pytest.raises(ValueError, match="missing 'user'"):
            loader.load_from_dict({"portfolio": {"holdings": []}})

    def test_missing_portfolio_raises_error(self):
        loader = PortfolioLoader()
        with pytest.raises(ValueError, match="missing 'portfolio'"):
            loader.load_from_dict({"user": {"id": "1", "name": "A"}})

    def test_load_from_file_sample(self):
        """Test loading the bundled sample portfolio."""
        loader = PortfolioLoader()
        portfolio = loader.load_from_file()  # Uses sample by default
        assert portfolio is not None
        assert len(portfolio.holdings) > 0
        assert portfolio.user.name is not None

    def test_load_from_temp_json_file(self, sample_portfolio_dict):
        """Test loading from a custom JSON file path."""
        loader = PortfolioLoader()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_portfolio_dict, f)
            temp_path = f.name

        try:
            portfolio = loader.load_from_file(temp_path)
            assert len(portfolio.holdings) == 3
        finally:
            os.unlink(temp_path)

    def test_file_not_found_raises_error(self):
        loader = PortfolioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path/portfolio.json")


# ─────────────────────────────────────────────
# Test: Market Data Enrichment
# ─────────────────────────────────────────────
class TestPortfolioEnrichment:

    def test_current_prices_populated(self, enriched_portfolio):
        for holding in enriched_portfolio.holdings:
            assert holding.current_price > 0

    def test_daily_pnl_calculated(self, enriched_portfolio):
        hdfc = enriched_portfolio.get_holding("HDFCBANK.NS")
        assert hdfc is not None
        # 50 shares * ₹50 gain = ₹2500
        assert hdfc.daily_pnl == pytest.approx(2500.0, abs=1.0)

    def test_unrealised_pnl_calculated(self, enriched_portfolio):
        tcs = enriched_portfolio.get_holding("TCS.NS")
        assert tcs is not None
        # 20 shares * (3700 - 3800) = -2000
        assert tcs.unrealised_pnl == pytest.approx(-2000.0, abs=1.0)

    def test_current_value_calculated(self, enriched_portfolio):
        reliance = enriched_portfolio.get_holding("RELIANCE.NS")
        assert reliance is not None
        # 30 * 2950 = 88500
        assert reliance.current_value == pytest.approx(88500.0, abs=1.0)

    def test_total_current_value_correct(self, enriched_portfolio):
        # HDFC: 50*1650=82500, TCS: 20*3700=74000, RIL: 30*2950=88500, cash=10000
        expected = 82500 + 74000 + 88500 + 10000
        assert enriched_portfolio.total_current_value == pytest.approx(expected, abs=1.0)


# ─────────────────────────────────────────────
# Test: Portfolio Analysis
# ─────────────────────────────────────────────
class TestPortfolioAnalyzer:

    def test_analysis_returns_object(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        assert isinstance(analysis, PortfolioAnalysis)

    def test_sector_allocation_has_all_sectors(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        assert "Banking" in analysis.sector_allocation
        assert "IT" in analysis.sector_allocation
        assert "Energy" in analysis.sector_allocation

    def test_sector_pct_sums_to_100(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        equity_total = sum(v["pct"] for v in analysis.sector_allocation.values())
        assert equity_total == pytest.approx(100.0, abs=0.5)

    def test_daily_pnl_sum_correct(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        # HDFC: +2500, TCS: -2000, RIL: +1500 = +2000
        expected = 2500 + (-2000) + 1500
        assert analysis.total_daily_pnl == pytest.approx(expected, abs=10.0)

    def test_top_performers_sorted(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        if len(analysis.top_performers) >= 2:
            # Top performer should have higher daily change than second
            assert analysis.top_performers[0]["daily_change_pct"] >= analysis.top_performers[1]["daily_change_pct"]

    def test_risk_alerts_list(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        assert isinstance(analysis.risk_alerts, list)
        for alert in analysis.risk_alerts:
            assert alert.level in ("high", "medium", "low")
            assert alert.category in ("concentration", "volatility", "drawdown", "diversification")

    def test_overall_risk_level_valid(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        assert analysis.overall_risk_level in ("low", "moderate", "high", "very_high")

    def test_to_dict_structure(self, analyzer, enriched_portfolio):
        analysis = analyzer.analyze(enriched_portfolio)
        d = analysis.to_dict()
        assert "pnl" in d
        assert "sector_allocation" in d
        assert "risk_alerts" in d
        assert "top_performers" in d
        assert "overall_risk_level" in d


# ─────────────────────────────────────────────
# Test: Concentration Risk Detection
# ─────────────────────────────────────────────
class TestConcentrationRisk:

    def test_high_concentration_detected(self):
        """A portfolio with 70% in one sector should trigger high risk alert."""
        concentrated_dict = {
            "user": {"id": "x", "name": "X", "risk_profile": "moderate",
                     "investment_goal": "growth", "investment_horizon_years": 5},
            "portfolio": {
                "total_invested": 100000,
                "cash_balance": 0,
                "holdings": [
                    {"symbol": "HDFCBANK.NS", "name": "HDFC", "quantity": 40,
                     "avg_buy_price": 1600.0, "sector": "Banking", "asset_type": "equity"},
                    {"symbol": "SBIN.NS", "name": "SBI", "quantity": 30,
                     "avg_buy_price": 600.0, "sector": "Banking", "asset_type": "equity"},
                    {"symbol": "TCS.NS", "name": "TCS", "quantity": 5,
                     "avg_buy_price": 3800.0, "sector": "IT", "asset_type": "equity"},
                ],
            },
        }
        loader = PortfolioLoader()
        portfolio = loader.load_from_dict(concentrated_dict)

        # Enrich with market data
        market_data = {
            "HDFCBANK.NS": {"current": 1600.0, "previous_close": 1600.0, "change": 0, "change_pct": 0},
            "SBIN.NS": {"current": 600.0, "previous_close": 600.0, "change": 0, "change_pct": 0},
            "TCS.NS": {"current": 3800.0, "previous_close": 3800.0, "change": 0, "change_pct": 0},
        }
        portfolio = loader.enrich_with_market_data(portfolio, market_data)

        analyzer = PortfolioAnalyzer()
        analysis = analyzer.analyze(portfolio)

        # Should have a concentration alert for Banking
        concentration_alerts = [
            a for a in analysis.risk_alerts
            if a.category == "concentration" and "Banking" in str(a.affected)
        ]
        assert len(concentration_alerts) >= 1
        assert any(a.level == "high" for a in concentration_alerts)


# ─────────────────────────────────────────────
# Test: Benchmark Comparison
# ─────────────────────────────────────────────
class TestBenchmarkComparison:

    def test_outperforming_benchmark(self, analyzer):
        result = analyzer.compare_vs_benchmark(
            portfolio_daily_pct=1.5, nifty_daily_pct=0.8
        )
        assert result["outperforming"] is True
        assert result["alpha"] == pytest.approx(0.7, abs=0.01)

    def test_underperforming_benchmark(self, analyzer):
        result = analyzer.compare_vs_benchmark(
            portfolio_daily_pct=-0.5, nifty_daily_pct=0.8
        )
        assert result["outperforming"] is False
        assert result["alpha"] == pytest.approx(-1.3, abs=0.01)
