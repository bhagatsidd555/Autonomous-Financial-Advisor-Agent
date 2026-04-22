"""
tests/test_reasoning.py
=========================
Unit tests for the reasoning layer:
  - NewsClassifier
  - CausalReasoner
  - ConflictResolver
  - SelfEvaluator
"""

import pytest
from unittest.mock import patch, MagicMock

from src.ingestion.news_fetcher import NewsItem, NewsFetcher
from src.reasoning.news_classifier import ClassifiedNews, NewsClassifier
from src.reasoning.conflict_resolver import ConflictResolver, ConflictSignal
from src.agent.self_evaluator import SelfEvaluator, EvaluationResult
from src.reasoning.causal_reasoner import ReasoningOutput, CausalLink


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def sample_news_item():
    return NewsItem(
        headline="RBI keeps repo rate unchanged at 6.5%",
        summary="The Reserve Bank of India kept the benchmark repo rate unchanged at 6.5% at its bi-monthly monetary policy meeting, as expected by most analysts.",
        source="Economic Times",
        published_at="2025-04-22T10:00:00",
        category="monetary_policy",
        tags=["RBI", "repo rate", "banking"],
    )


@pytest.fixture
def positive_it_news():
    return NewsItem(
        headline="TCS Q4 results beat estimates; revenue up 15%",
        summary="Tata Consultancy Services reported a strong fourth quarter with revenues beating analyst estimates by 5%. The company saw strong growth in North America and Europe.",
        source="Business Standard",
        published_at="2025-04-22T09:00:00",
        category="earnings",
        tags=["TCS", "IT", "Q4 results", "earnings beat"],
    )


@pytest.fixture
def negative_auto_news():
    return NewsItem(
        headline="Maruti Suzuki guides for lower margins in FY26 due to EV investments",
        summary="Maruti Suzuki warned that margins would be under pressure in FY26 as the company ramps up investments in electric vehicle platforms.",
        source="LiveMint",
        published_at="2025-04-21T15:00:00",
        category="sector_news",
        tags=["Maruti", "Auto", "EV", "margins"],
    )


@pytest.fixture
def mock_classification_response():
    return """{
        "sentiment": "neutral",
        "scope": "sector-specific",
        "impact_level": "high",
        "affected_sectors": ["Banking"],
        "affected_stocks": [],
        "key_themes": ["RBI policy", "repo rate", "monetary policy"],
        "confidence": 0.9,
        "reasoning": "RBI rate decision is sector-specific to banking with high impact"
    }"""


@pytest.fixture
def mock_anthropic_client(mock_classification_response):
    """Mock the Anthropic client for unit tests."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_classification_response)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_reasoning_output():
    """Create a sample ReasoningOutput for evaluation tests."""
    return ReasoningOutput(
        executive_summary="Portfolio declined 1.2% today driven by banking sector weakness following RBI rate decision.",
        market_narrative="Indian markets traded lower as the banking sector came under pressure after the RBI's policy announcement. NIFTY 50 fell 0.8% while Bank Nifty led declines at -1.5%.",
        portfolio_impact_narrative="Rahul's portfolio was impacted primarily through the banking sector holdings. HDFC Bank and SBI both declined, contributing to the portfolio's 1.2% daily loss. The IT sector acted as a partial hedge.",
        actionable_insights=[
            "Consider reducing Banking concentration from 40% to under 30% given ongoing rate uncertainty.",
            "IT holdings are providing portfolio resilience — maintain current allocation.",
        ],
        causal_chain=[
            CausalLink(
                cause="RBI kept repo rates unchanged with hawkish tone",
                effect="Banking sector under selling pressure",
                mechanism="Higher-for-longer rates compress net interest margins for banks",
                confidence=0.85,
                scope="sector",
            ),
            CausalLink(
                cause="Banking sector decline (-1.5%)",
                effect="Portfolio fell 1.2% today",
                mechanism="Portfolio has 40% exposure to banking sector",
                confidence=0.90,
                scope="portfolio",
            ),
        ],
        key_positive_signals=["IT sector resilient", "Global markets stable"],
        key_negative_signals=["Banking sector weakness", "RBI hawkish tone"],
        conflicting_signals=[],
        confidence_score=0.82,
        data_quality="high",
        reasoning_depth="deep",
    )


# ─────────────────────────────────────────────
# Test: NewsItem
# ─────────────────────────────────────────────
class TestNewsItem:

    def test_news_item_creation(self, sample_news_item):
        assert sample_news_item.headline == "RBI keeps repo rate unchanged at 6.5%"
        assert sample_news_item.source == "Economic Times"
        assert sample_news_item.id is not None
        assert len(sample_news_item.id) == 12

    def test_news_item_id_deterministic(self, sample_news_item):
        """Same headline should always produce same ID."""
        item2 = NewsItem(
            headline="RBI keeps repo rate unchanged at 6.5%",
            summary="...",
            source="ET",
            published_at="2025-01-01",
        )
        assert sample_news_item.id == item2.id

    def test_news_item_to_dict(self, sample_news_item):
        d = sample_news_item.to_dict()
        assert "id" in d
        assert "headline" in d
        assert "sentiment" not in d  # Not classified yet


# ─────────────────────────────────────────────
# Test: NewsFetcher
# ─────────────────────────────────────────────
class TestNewsFetcher:

    def test_load_sample_news(self):
        fetcher = NewsFetcher(use_sample_fallback=True)
        items = fetcher.load_sample_news()
        assert len(items) >= 1
        assert all(isinstance(i, NewsItem) for i in items)

    def test_extract_stocks_mentioned(self):
        fetcher = NewsFetcher()
        item = NewsItem(
            headline="TCS and Infosys report strong Q4 numbers",
            summary="TCS earnings beat estimates while Infosys raised guidance",
            source="ET",
            published_at="2025-04-22",
        )
        stocks = fetcher.extract_stocks_mentioned(item)
        assert "TCS.NS" in stocks
        assert "INFY.NS" in stocks

    def test_extract_sectors_mentioned(self):
        fetcher = NewsFetcher()
        item = NewsItem(
            headline="RBI keeps repo rate unchanged",
            summary="Banking sector cheers as RBI holds rate",
            source="ET",
            published_at="2025-04-22",
        )
        sectors = fetcher.extract_sectors_mentioned(item)
        assert "Banking" in sectors

    def test_fallback_when_no_live_data(self):
        """Fetcher should return sample data when live fetch fails."""
        fetcher = NewsFetcher(use_sample_fallback=True)
        # Calling with both sources disabled should fall back to sample
        items = fetcher.fetch_latest_news(use_rss=False, use_newsapi=False)
        assert len(items) >= 1


# ─────────────────────────────────────────────
# Test: NewsClassifier
# ─────────────────────────────────────────────
class TestNewsClassifier:

    @patch("src.reasoning.news_classifier.anthropic.Anthropic")
    def test_classify_single_returns_classified_news(
        self, mock_anthropic, sample_news_item, mock_classification_response
    ):
        """Test that classify_single returns a ClassifiedNews object."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=mock_classification_response)]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        classifier = NewsClassifier()
        result = classifier.classify_single(sample_news_item)

        assert isinstance(result, ClassifiedNews)
        assert result.sentiment in ("positive", "negative", "neutral")
        assert result.scope in ("market-wide", "sector-specific", "stock-specific")
        assert result.impact_level in ("high", "medium", "low")

    @patch("src.reasoning.news_classifier.anthropic.Anthropic")
    def test_classify_banking_news(
        self, mock_anthropic, sample_news_item, mock_classification_response
    ):
        """RBI rate news should classify as sector-specific to Banking."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=mock_classification_response)]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        classifier = NewsClassifier()
        result = classifier.classify_single(
            sample_news_item,
            portfolio_sectors=["Banking", "IT"],
            portfolio_stocks=["HDFCBANK.NS", "TCS.NS"],
        )

        assert "Banking" in result.affected_sectors or result.scope == "sector-specific"
        # Banking news should be relevant to a portfolio with banking holdings
        assert result.relevance_score > 0.0

    @patch("src.reasoning.news_classifier.anthropic.Anthropic")
    def test_fallback_on_api_error(self, mock_anthropic, sample_news_item):
        """Classifier should return basic result on API failure."""
        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

        classifier = NewsClassifier()
        result = classifier.classify_single(sample_news_item)

        assert isinstance(result, ClassifiedNews)
        assert result.sentiment == "neutral"
        assert result.classification_confidence < 0.5  # Low confidence on fallback

    def test_portfolio_relevance_market_wide(self):
        """Market-wide news should be relevant regardless of portfolio."""
        item = NewsItem("Global rally", "Global markets rising", "ET", "2025-04-22")
        classified = ClassifiedNews(
            news_item=item,
            sentiment="positive",
            scope="market-wide",
            impact_level="high",
        )
        assert classified.is_portfolio_relevant(["Banking", "IT"], ["HDFCBANK.NS"])

    def test_portfolio_relevance_sector_match(self):
        """Sector-specific news should be relevant if sector is in portfolio."""
        item = NewsItem("Banking news", "RBI decision", "ET", "2025-04-22")
        classified = ClassifiedNews(
            news_item=item,
            sentiment="negative",
            scope="sector-specific",
            impact_level="high",
            affected_sectors=["Banking"],
        )
        assert classified.is_portfolio_relevant(["Banking", "IT"], ["HDFCBANK.NS"])
        assert not classified.is_portfolio_relevant(["Pharma", "Metal"], ["SUNPHARMA.NS"])

    def test_portfolio_relevance_stock_match(self):
        """Stock-specific news should be relevant if stock is in portfolio."""
        item = NewsItem("TCS earnings", "TCS beats", "ET", "2025-04-22")
        classified = ClassifiedNews(
            news_item=item,
            sentiment="positive",
            scope="stock-specific",
            impact_level="medium",
            affected_stocks=["TCS.NS"],
        )
        assert classified.is_portfolio_relevant([], ["TCS.NS", "INFY.NS"])
        assert not classified.is_portfolio_relevant(["Banking"], ["HDFCBANK.NS"])


# ─────────────────────────────────────────────
# Test: ReasoningOutput
# ─────────────────────────────────────────────
class TestReasoningOutput:

    def test_to_dict_has_required_keys(self, sample_reasoning_output):
        d = sample_reasoning_output.to_dict()
        required = [
            "executive_summary", "market_narrative", "portfolio_impact_narrative",
            "causal_chain", "key_positive_signals", "key_negative_signals",
            "confidence_score", "reasoning_depth",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_full_report_generation(self, sample_reasoning_output):
        report = sample_reasoning_output.to_full_report()
        assert len(report) > 200
        assert "Executive Summary" in report
        assert "Causal Chain" in report
        assert "Actionable Insights" in report

    def test_causal_chain_serialization(self, sample_reasoning_output):
        for link in sample_reasoning_output.causal_chain:
            d = link.to_dict()
            assert "cause" in d
            assert "effect" in d
            assert "mechanism" in d
            assert 0.0 <= d["confidence"] <= 1.0


# ─────────────────────────────────────────────
# Test: SelfEvaluator
# ─────────────────────────────────────────────
class TestSelfEvaluator:

    @patch("src.agent.self_evaluator.anthropic.Anthropic")
    def test_evaluate_returns_result(self, mock_anthropic, sample_reasoning_output):
        eval_response = """{
            "overall_score": 0.82,
            "reasoning_quality": 0.85,
            "factual_consistency": 0.80,
            "actionability": 0.80,
            "clarity": 0.85,
            "data_coverage": 0.78,
            "strengths": ["Clear causal chain", "Specific actionable insights"],
            "weaknesses": ["Could mention more stocks"],
            "improvement_suggestions": ["Include sector comparison"]
        }"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=eval_response)]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        evaluator = SelfEvaluator()
        result = evaluator.evaluate(sample_reasoning_output)

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.quality_grade in ("A", "B", "C", "D")

    def test_simple_evaluate_score_range(self, sample_reasoning_output):
        evaluator = SelfEvaluator()
        score = evaluator.evaluate_simple(sample_reasoning_output)
        assert 0.0 <= score <= 1.0

    def test_high_quality_output_scores_high(self, sample_reasoning_output):
        evaluator = SelfEvaluator()
        score = evaluator.evaluate_simple(sample_reasoning_output)
        # Our sample output has all components → should score high
        assert score >= 0.8

    def test_empty_output_scores_low(self):
        evaluator = SelfEvaluator()
        empty_output = ReasoningOutput(
            executive_summary="",
            market_narrative="",
            portfolio_impact_narrative="",
            actionable_insights=[],
            causal_chain=[],
            key_positive_signals=[],
            key_negative_signals=[],
            conflicting_signals=[],
            confidence_score=0.0,
            data_quality="low",
            reasoning_depth="shallow",
        )
        score = evaluator.evaluate_simple(empty_output)
        assert score == 0.0

    def test_score_to_grade_mapping(self):
        evaluator = SelfEvaluator()
        assert evaluator._score_to_grade(0.90) == "A"
        assert evaluator._score_to_grade(0.75) == "B"
        assert evaluator._score_to_grade(0.60) == "C"
        assert evaluator._score_to_grade(0.40) == "D"

    def test_passed_flag_based_on_min_score(self):
        evaluator = SelfEvaluator()
        # Direct construction to avoid LLM call
        result = EvaluationResult(
            overall_score=0.75,
            quality_grade="B",
            passed=0.75 >= evaluator.min_score,
            reasoning_quality=0.75,
            factual_consistency=0.75,
            actionability=0.75,
            clarity=0.75,
            data_coverage=0.75,
        )
        assert result.passed is True

        result_fail = EvaluationResult(
            overall_score=0.3,
            quality_grade="D",
            passed=0.3 >= evaluator.min_score,
            reasoning_quality=0.3,
            factual_consistency=0.3,
            actionability=0.3,
            clarity=0.3,
            data_coverage=0.3,
        )
        assert result_fail.passed is False
