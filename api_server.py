"""
FastAPI Backend Server — Autonomous Financial Advisor Agent
Run: uvicorn api_server:app --reload --port 8000
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config.settings import validate_config
from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.news_fetcher import NewsFetcher
from src.ingestion.portfolio_loader import PortfolioLoader
from src.analytics.market_analyzer import MarketAnalyzer
from src.analytics.sector_analyzer import SectorAnalyzer
from src.analytics.portfolio_analyzer import PortfolioAnalyzer
from src.reasoning.news_classifier import NewsClassifier
from src.reasoning.causal_reasoner import CausalReasoner
from src.reasoning.conflict_resolver import ConflictResolver
from src.agent.self_evaluator import SelfEvaluator

app = FastAPI(title="Autonomous Financial Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_components = {}

def get_components():
    global _components
    if not _components:
        missing = validate_config()
        if missing:
            raise RuntimeError(f"Missing config: {missing}")
        _components = {
            "market_fetcher":     MarketDataFetcher(),
            "news_fetcher":       NewsFetcher(use_sample_fallback=True),
            "portfolio_loader":   PortfolioLoader(),
            "market_analyzer":    MarketAnalyzer(),
            "sector_analyzer":    SectorAnalyzer(),
            "portfolio_analyzer": PortfolioAnalyzer(),
            "news_classifier":    NewsClassifier(),
            "causal_reasoner":    CausalReasoner(),
            "conflict_resolver":  ConflictResolver(),
            "self_evaluator":     SelfEvaluator(),
        }
    return _components

class PortfolioAnalyzeRequest(BaseModel):
    portfolio_path: Optional[str] = "data/sample_portfolio.json"

def _safe(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _holding_to_dict(h) -> dict:
    return {
        "symbol":             getattr(h, "symbol", ""),
        "quantity":           getattr(h, "quantity", 0),
        "avg_buy_price":      _safe(getattr(h, "avg_buy_price", 0)),
        "current_price":      _safe(getattr(h, "current_price", getattr(h, "cmp", 0))),
        "current_value":      _safe(getattr(h, "current_value", 0)),
        "unrealised_pnl":     _safe(getattr(h, "unrealised_pnl", 0)),
        "unrealised_pnl_pct": _safe(getattr(h, "unrealised_pnl_pct", 0)),
        "daily_pnl":          _safe(getattr(h, "daily_pnl", 0)),
        "daily_pnl_pct":      _safe(getattr(h, "daily_pnl_pct", getattr(h, "daily_change_pct", 0))),
        "sector":             getattr(h, "sector", "Unknown"),
    }

def _news_to_dict(n) -> dict:
    impact = getattr(n, "impact_score", getattr(n, "impact_level", 0.5))
    if isinstance(impact, float):
        impact_label = "HIGH" if impact >= 0.7 else "MED" if impact >= 0.4 else "LOW"
    else:
        impact_label = str(impact).upper()
    return {
        "title":            getattr(n, "title", ""),
        "source":           getattr(n, "source", ""),
        "sentiment":        getattr(n, "sentiment", "neutral"),
        "scope":            getattr(n, "scope", "market-wide"),
        "impact":           impact_label,
        "impact_score":     _safe(impact),
        "affected_sectors": getattr(n, "affected_sectors", []),
        "affected_stocks":  getattr(n, "affected_stocks", []),
        "is_relevant":      getattr(n, "is_portfolio_relevant", False),
    }

def _index_to_dict(name: str, idx) -> dict:
    if isinstance(idx, dict):
        change_pct = idx.get("change_pct", idx.get("pct_change", 0.0))
        price = idx.get("price", idx.get("current_price", 0.0))
    else:
        change_pct = getattr(idx, "change_pct", getattr(idx, "pct_change", 0.0))
        price = getattr(idx, "price", getattr(idx, "current_price", 0.0))
    return {
        "name":       name,
        "price":      _safe(price),
        "change_pct": _safe(change_pct),
        "direction":  "up" if _safe(change_pct) > 0 else "down" if _safe(change_pct) < 0 else "flat",
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Autonomous Financial Advisor API"}

@app.get("/api/portfolio/sample")
def get_sample_portfolio():
    import json
    path = Path("data/sample_portfolio.json")
    if not path.exists():
        raise HTTPException(404, "sample_portfolio.json not found")
    return json.loads(path.read_text())

@app.post("/api/analyze")
def analyze_portfolio(req: PortfolioAnalyzeRequest):
    start = time.time()
    try:
        c = get_components()

        portfolio = c["portfolio_loader"].load_from_file(req.portfolio_path)
        if not portfolio:
            raise HTTPException(400, "Portfolio not loaded")

        indices_raw = c["market_fetcher"].fetch_all_indices()
        stocks_raw  = c["market_fetcher"].fetch_multiple_stocks(portfolio.get_symbols())
        portfolio   = c["portfolio_loader"].enrich_with_market_data(portfolio, stocks_raw)

        market_condition   = c["market_analyzer"].analyze(indices_raw)
        sector_snapshot    = c["sector_analyzer"].analyze(indices_raw, stocks_raw)
        portfolio_analysis = c["portfolio_analyzer"].analyze(portfolio)

        raw_news          = c["news_fetcher"].fetch_latest_news() or []
        portfolio_sectors = list((getattr(portfolio_analysis, "sector_allocation", {}) or {}).keys())
        portfolio_symbols = portfolio.get_symbols()
        classified_news   = c["news_classifier"].classify_batch(
            raw_news,
            portfolio_symbols=portfolio_symbols,
            portfolio_sectors=portfolio_sectors,
        )
        relevant_news      = [n for n in classified_news if getattr(n, "is_portfolio_relevant", False)]
        news_for_reasoning = relevant_news if relevant_news else classified_news[:5]

        user_name = getattr(getattr(portfolio, "user", None), "name", "Investor")
        reasoning = c["causal_reasoner"].reason(
            market_condition=market_condition,
            sector_snapshot=sector_snapshot,
            portfolio_analysis=portfolio_analysis,
            classified_news=news_for_reasoning,
            user_profile={"name": user_name},
        )

        evaluation = c["self_evaluator"].evaluate(
            reasoning,
            portfolio_data=portfolio_analysis,
            news_count=len(news_for_reasoning),
        )

        conflicts = []
        try:
            conflicts = c["conflict_resolver"].detect_and_resolve(
                portfolio=portfolio,
                classified_news=relevant_news,
                sector_snapshot=sector_snapshot,
                market_condition=market_condition,
            )
        except Exception as ce:
            logger.warning(f"Conflict resolver: {ce}")

        sector_alloc_raw = getattr(portfolio_analysis, "sector_allocation", {}) or {}
        sector_allocation = []
        for sector, data in sector_alloc_raw.items():
            if isinstance(data, dict):
                weight = data.get("weight_pct", data.get("allocation_pct", 0))
                pnl    = data.get("daily_pnl", 0)
            else:
                weight = getattr(data, "weight_pct", getattr(data, "allocation_pct", 0))
                pnl    = getattr(data, "daily_pnl", 0)
            sector_allocation.append({
                "sector":    sector,
                "weight":    round(_safe(weight), 1),
                "daily_pnl": round(_safe(pnl), 0),
            })

        risk_alerts = [
            getattr(a, "message", str(a))
            for a in (getattr(portfolio_analysis, "risk_alerts", []) or [])
        ]

        causal_chains = [
            {
                "cause":      getattr(ch, "cause", ""),
                "effect":     getattr(ch, "effect", ""),
                "mechanism":  getattr(ch, "mechanism", ""),
                "confidence": round(_safe(getattr(ch, "confidence", 0.5)), 2),
                "scope":      getattr(ch, "scope", "macro"),
            }
            for ch in (getattr(reasoning, "causal_chains", []) or [])
        ]

        indices_dict = getattr(market_condition, "indices", {}) or {}
        if not isinstance(indices_dict, dict):
            indices_dict = {}

        elapsed = round(time.time() - start, 2)

        return {
            "success":    True,
            "run_time_s": elapsed,
            "user_name":  user_name,
            "holdings":   [_holding_to_dict(h) for h in (portfolio.holdings or [])],
            "summary": {
                "total_invested":  round(_safe(getattr(portfolio_analysis, "total_invested", 0)), 0),
                "current_value":   round(_safe(getattr(portfolio_analysis, "total_current_value", 0)), 0),
                "daily_pnl":       round(_safe(getattr(portfolio_analysis, "total_daily_pnl", 0)), 0),
                "daily_pnl_pct":   round(_safe(getattr(portfolio_analysis, "daily_pnl_pct", 0)), 2),
                "overall_pnl":     round(_safe(getattr(portfolio_analysis, "total_unrealised_pnl", 0)), 0),
                "overall_pnl_pct": round(_safe(getattr(portfolio_analysis, "unrealised_pnl_pct", 0)), 2),
                "risk_level":      str(getattr(portfolio_analysis, "risk_level", "moderate")),
            },
            "sector_allocation": sorted(sector_allocation, key=lambda x: -x["weight"]),
            "risk_alerts": risk_alerts,
            "market": {
                "sentiment":    str(getattr(market_condition, "sentiment", "neutral")),
                "strength":     str(getattr(market_condition, "strength", "moderate")),
                "nifty_change": round(_safe(getattr(market_condition, "nifty_change", 0)), 2),
                "leaders":      getattr(market_condition, "leaders",  getattr(sector_snapshot, "leaders", [])),
                "laggards":     getattr(market_condition, "laggards", getattr(sector_snapshot, "laggards", [])),
                "indices": [_index_to_dict(k, v) for k, v in list(indices_dict.items())[:8]],
            },
            "news": [_news_to_dict(n) for n in news_for_reasoning[:6]],
            "reasoning": {
                "market_narrative":    getattr(reasoning, "market_narrative", ""),
                "portfolio_impact":    getattr(reasoning, "portfolio_impact", ""),
                "actionable_insights": getattr(reasoning, "actionable_insights", []),
                "positive_signals":    getattr(reasoning, "positive_signals", []),
                "negative_signals":    getattr(reasoning, "negative_signals", []),
                "causal_chains":       causal_chains,
                "key_drivers":         getattr(reasoning, "key_drivers", []),
                "confidence_score":    round(_safe(getattr(reasoning, "confidence_score", 0.8)), 2),
            },
            "conflicts": [
                {
                    "entity":      getattr(cf, "symbol_or_sector", str(cf)),
                    "explanation": getattr(cf, "explanation", str(cf)),
                }
                for cf in conflicts
            ],
            "evaluation": {
                "grade":               getattr(evaluation, "grade", "B"),
                "score":               round(_safe(getattr(evaluation, "score", 0.7)), 2),
                "reasoning_quality":   round(_safe(getattr(evaluation, "reasoning_quality", 0)), 2),
                "factual_consistency": round(_safe(getattr(evaluation, "factual_consistency", 0)), 2),
                "actionability":       round(_safe(getattr(evaluation, "actionability", 0)), 2),
                "clarity":             round(_safe(getattr(evaluation, "clarity", 0)), 2),
                "data_coverage":       round(_safe(getattr(evaluation, "data_coverage", 0)), 2),
                "areas_to_improve":    getattr(evaluation, "areas_to_improve", []),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze endpoint failed")
        raise HTTPException(500, str(e))
