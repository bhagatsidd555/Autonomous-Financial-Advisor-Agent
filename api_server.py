"""
FastAPI Backend Server — Autonomous Financial Advisor Agent
Run locally: uvicorn api_server:app --reload --port 8000
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Groq for chat endpoint ──
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    GROQ_MODEL   = os.getenv("MODEL", "llama-3.3-70b-versatile")
    if groq_client:
        logger.info("Groq client initialised — model: %s", GROQ_MODEL)
    else:
        logger.warning("GROQ_API_KEY not set — /api/chat will be unavailable")
except ImportError:
    groq_client = None
    GROQ_MODEL  = ""
    logger.warning("groq package not installed — run: pip install groq")

# ── Project imports ──
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

# ─────────────────────────────────────────────
app = FastAPI(title="Autonomous Financial Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded singleton components ──
_components: dict = {}

def get_components() -> dict:
    global _components
    if not _components:
        missing = validate_config()
        if missing:
            raise RuntimeError(f"Missing config keys: {missing}")
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
        logger.info("All components initialised successfully")
    return _components


# ─────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────

class PortfolioAnalyzeRequest(BaseModel):
    portfolio_path: Optional[str] = "data/sample_portfolio.json"

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None    # last /api/analyze response
    history: Optional[list] = None    # past [{role, content}, ...] messages


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _safe(v, default: float = 0.0) -> float:
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
        "daily_pnl_pct":      _safe(getattr(h, "daily_pnl_pct",
                                    getattr(h, "daily_change_pct", 0))),
        "sector":             getattr(h, "sector", "Unknown"),
    }


def _news_to_dict(n) -> dict:
    impact = getattr(n, "impact_score", getattr(n, "impact_level", 0.5))
    if isinstance(impact, float):
        impact_label = "HIGH" if impact >= 0.7 else "MED" if impact >= 0.4 else "LOW"
    else:
        impact_label = str(impact).upper()
    return {
        "title":            getattr(n, "title", getattr(
                                getattr(n, "news_item", None), "headline", "")),
        "source":           getattr(n, "source", getattr(
                                getattr(n, "news_item", None), "source", "")),
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
        price      = idx.get("price", idx.get("current", idx.get("current_price", 0.0)))
    else:
        change_pct = getattr(idx, "change_pct", getattr(idx, "pct_change", 0.0))
        price      = getattr(idx, "price", getattr(idx, "current", 0.0))
    chg = _safe(change_pct)
    return {
        "name":       name,
        "price":      _safe(price),
        "change_pct": chg,
        "direction":  "up" if chg > 0 else "down" if chg < 0 else "flat",
    }


def _build_sector_allocation(portfolio_analysis) -> list[dict]:
    """Safely extract sector allocation from portfolio_analysis."""
    raw = getattr(portfolio_analysis, "sector_allocation", {}) or {}
    result = []
    for sector, data in raw.items():
        if isinstance(data, dict):
            weight = data.get("pct", data.get("weight_pct", data.get("allocation_pct", 0)))
            pnl    = data.get("daily_pnl", 0)
            value  = data.get("value", 0)
        else:
            weight = getattr(data, "pct", getattr(data, "weight_pct", 0))
            pnl    = getattr(data, "daily_pnl", 0)
            value  = getattr(data, "value", 0)
        result.append({
            "sector":    sector,
            "weight":    round(_safe(weight), 1),
            "value":     round(_safe(value), 0),
            "daily_pnl": round(_safe(pnl), 0),
        })
    return sorted(result, key=lambda x: -x["weight"])


def _build_causal_chains(reasoning) -> list[dict]:
    """Extract causal chain from either causal_chain or causal_chains attribute."""
    chains = (
        getattr(reasoning, "causal_chain", None)
        or getattr(reasoning, "causal_chains", None)
        or []
    )
    result = []
    for ch in chains:
        result.append({
            "cause":      getattr(ch, "cause", ""),
            "effect":     getattr(ch, "effect", ""),
            "mechanism":  getattr(ch, "mechanism", ""),
            "confidence": round(_safe(getattr(ch, "confidence", 0.5)), 2),
            "scope":      getattr(ch, "scope", "macro"),
        })
    return result


def _build_evaluation(evaluation) -> dict:
    """Safely extract self-evaluation fields."""
    dims = getattr(evaluation, "dimensions", {}) or {}

    def _dim(key, fallback_attr):
        return round(_safe(dims.get(key, getattr(evaluation, fallback_attr, 0))), 2)

    return {
        "grade":               getattr(evaluation, "quality_grade",
                                       getattr(evaluation, "grade", "B")),
        "score":               round(_safe(getattr(evaluation, "overall_score",
                                            getattr(evaluation, "score", 0.7))), 2),
        "reasoning_quality":   _dim("reasoning_quality",   "reasoning_quality"),
        "factual_consistency": _dim("factual_consistency", "factual_consistency"),
        "actionability":       _dim("actionability",       "actionability"),
        "clarity":             _dim("clarity",             "clarity"),
        "data_coverage":       _dim("data_coverage",       "data_coverage"),
        "areas_to_improve":    (
            getattr(evaluation, "weaknesses", None)
            or getattr(evaluation, "areas_to_improve", [])
        ),
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status":          "ok",
        "service":         "Autonomous Financial Advisor API",
        "version":         "1.0.0",
        "groq_available":  groq_client is not None,
        "groq_model":      GROQ_MODEL,
    }


@app.get("/api/portfolio/sample")
def get_sample_portfolio():
    import json
    path = Path("data/sample_portfolio.json")
    if not path.exists():
        raise HTTPException(404, "sample_portfolio.json not found")
    return json.loads(path.read_text())


# ── POST /api/analyze ──────────────────────────────────────────────────────
@app.post("/api/analyze")
def analyze_portfolio(req: PortfolioAnalyzeRequest):
    """
    Run the full autonomous agent pipeline:
    portfolio → market data → news → causal reasoning → self-evaluation
    """
    start = time.time()
    try:
        c = get_components()

        # 1. Portfolio
        portfolio = c["portfolio_loader"].load_from_file(req.portfolio_path)
        if not portfolio:
            raise HTTPException(400, "Portfolio could not be loaded")

        # 2. Market data
        indices_raw = c["market_fetcher"].fetch_all_indices()
        stocks_raw  = c["market_fetcher"].fetch_multiple_stocks(portfolio.get_symbols())
        portfolio   = c["portfolio_loader"].enrich_with_market_data(portfolio, stocks_raw)

        # 3. Analytics
        market_condition   = c["market_analyzer"].analyze(indices_raw)
        sector_snapshot    = c["sector_analyzer"].analyze(indices_raw, stocks_raw)
        portfolio_analysis = c["portfolio_analyzer"].analyze(portfolio)

        # 4. News
        raw_news          = c["news_fetcher"].fetch_latest_news() or []
        portfolio_sectors = list((getattr(portfolio_analysis, "sector_allocation", {}) or {}).keys())
        portfolio_symbols = portfolio.get_symbols()

        classified_news = c["news_classifier"].classify_batch(
            raw_news,
            portfolio_sectors=portfolio_sectors,
            portfolio_stocks=portfolio_symbols,
        )
        relevant_news      = [n for n in classified_news
                               if getattr(n, "relevance_score", 0) > 0.3
                               or getattr(n, "is_portfolio_relevant", False)]
        news_for_reasoning = relevant_news if relevant_news else classified_news[:5]

        # 5. Causal reasoning
        user_name = getattr(getattr(portfolio, "user", None), "name", "Investor")
        reasoning = c["causal_reasoner"].reason(
            market_condition=market_condition,
            sector_snapshot=sector_snapshot,
            portfolio_analysis=portfolio_analysis,
            classified_news=news_for_reasoning,
            user_profile={
                "name":            user_name,
                "risk_profile":    getattr(getattr(portfolio, "user", None), "risk_profile", "moderate"),
                "investment_goal": getattr(getattr(portfolio, "user", None), "investment_goal", "wealth_creation"),
            },
        )

        # 6. Self-evaluation
        evaluation = c["self_evaluator"].evaluate(reasoning)

        # 7. Conflict resolution
        conflicts = []
        try:
            conflicts = c["conflict_resolver"].detect_and_resolve(
                portfolio=portfolio,
                classified_news=relevant_news,
                sector_snapshot=sector_snapshot,
                market_condition=market_condition,
            )
        except Exception as ce:
            logger.warning("Conflict resolver skipped: %s", ce)

        # 8. Build indices dict for response
        indices_dict = {}
        for name, idx_data in indices_raw.items():
            indices_dict[name] = idx_data

        elapsed = round(time.time() - start, 2)
        logger.info("Analysis complete in %.2fs", elapsed)

        return {
            "success":    True,
            "run_time_s": elapsed,
            "user_name":  user_name,

            # Holdings
            "holdings": [_holding_to_dict(h) for h in (portfolio.holdings or [])],

            # P&L Summary
            "summary": {
                "total_invested":  round(_safe(getattr(portfolio_analysis, "total_invested", 0)), 0),
                "current_value":   round(_safe(getattr(portfolio_analysis, "total_current_value", 0)), 0),
                "daily_pnl":       round(_safe(getattr(portfolio_analysis, "total_daily_pnl", 0)), 0),
                "daily_pnl_pct":   round(_safe(getattr(portfolio_analysis, "total_daily_pnl_pct", 0)), 2),
                "overall_pnl":     round(_safe(getattr(portfolio_analysis, "total_unrealised_pnl", 0)), 0),
                "overall_pnl_pct": round(_safe(getattr(portfolio_analysis, "total_unrealised_pnl_pct", 0)), 2),
                "risk_level":      str(getattr(portfolio_analysis, "overall_risk_level",
                                       getattr(portfolio_analysis, "risk_level", "moderate"))),
                "cash_balance":    round(_safe(getattr(portfolio_analysis, "cash_balance", 0)), 0),
            },

            # Sector allocation
            "sector_allocation": _build_sector_allocation(portfolio_analysis),

            # Risk alerts
            "risk_alerts": [
                getattr(a, "message", str(a))
                for a in (getattr(portfolio_analysis, "risk_alerts", []) or [])
            ],

            # Market
            "market": {
                "sentiment":    str(getattr(market_condition, "sentiment", "neutral")),
                "strength":     str(getattr(market_condition, "strength", "moderate")),
                "nifty_change": round(_safe(getattr(market_condition, "nifty_change_pct",
                                            getattr(market_condition, "nifty_change", 0))), 2),
                "sensex_change": round(_safe(getattr(market_condition, "sensex_change_pct", 0)), 2),
                "bank_change":   round(_safe(getattr(market_condition, "nifty_bank_change_pct", 0)), 2),
                "volatility":   str(getattr(market_condition, "volatility_level", "moderate")),
                "leaders":      getattr(sector_snapshot, "leaders", []),
                "laggards":     getattr(sector_snapshot, "laggards", []),
                "rotation":     getattr(sector_snapshot, "rotation_signal", "mixed"),
                "key_signals":  getattr(market_condition, "key_signals", []),
                "indices":      [_index_to_dict(k, v) for k, v in list(indices_dict.items())[:8]],
                "sectors":      [s.to_dict() for s in (getattr(sector_snapshot, "sectors", []) or [])],
            },

            # News
            "news": [_news_to_dict(n) for n in news_for_reasoning[:6]],

            # Reasoning
            "reasoning": {
                "executive_summary":   getattr(reasoning, "executive_summary", ""),
                "market_narrative":    getattr(reasoning, "market_narrative", ""),
                "portfolio_impact":    getattr(reasoning, "portfolio_impact_narrative",
                                               getattr(reasoning, "portfolio_impact", "")),
                "actionable_insights": getattr(reasoning, "actionable_insights", []),
                "positive_signals":    getattr(reasoning, "key_positive_signals",
                                               getattr(reasoning, "positive_signals", [])),
                "negative_signals":    getattr(reasoning, "key_negative_signals",
                                               getattr(reasoning, "negative_signals", [])),
                "conflicting_signals": getattr(reasoning, "conflicting_signals", []),
                "causal_chains":       _build_causal_chains(reasoning),
                "confidence_score":    round(_safe(getattr(reasoning, "confidence_score", 0.8)), 2),
                "reasoning_depth":     getattr(reasoning, "reasoning_depth", "moderate"),
            },

            # Conflicts
            "conflicts": [
                {
                    "entity":      getattr(cf, "symbol_or_sector",
                                           getattr(cf, "symbol", str(cf))),
                    "conflict_type": getattr(cf, "conflict_type", "unknown"),
                    "expected":    getattr(cf, "expected_behaviour", ""),
                    "actual":      getattr(cf, "actual_behaviour", ""),
                    "explanation": getattr(cf, "explanation", str(cf)),
                    "confidence":  round(_safe(getattr(cf, "confidence", 0.5)), 2),
                }
                for cf in conflicts
            ],

            # Self-evaluation
            "evaluation": _build_evaluation(evaluation),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze endpoint failed")
        raise HTTPException(500, str(e))


# ── POST /api/chat ─────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Interactive Q&A — user asks any question about their portfolio / market.
    Uses Groq LLM with the latest analysis as context.
    """
    if groq_client is None:
        raise HTTPException(
            503,
            "Groq client not configured. "
            "Add GROQ_API_KEY to your .env file and restart the server."
        )

    # ── System prompt ──────────────────────────────────────────────────────
    system = (
        "You are FinAdvisor AI — an expert autonomous Indian stock market advisor. "
        "You help users understand their portfolio, market trends, and investment decisions. "
        "Be concise, specific, and actionable. Use Indian market context (NSE/BSE, SEBI, RBI). "
        "When asked about specific stocks provide data-backed analysis. "
        "Format your response in clear paragraphs. Maximum 4 paragraphs."
    )

    # Inject portfolio context if available
    if req.context:
        ctx       = req.context
        summary   = ctx.get("summary", {})
        market    = ctx.get("market", {})
        reasoning = ctx.get("reasoning", {})
        holdings  = ctx.get("holdings", [])

        system += f"""

Current portfolio context:
- User: {ctx.get('user_name', 'Investor')}
- Daily P&L: ₹{summary.get('daily_pnl', 0):,.0f} ({summary.get('daily_pnl_pct', 0):.2f}%)
- Total invested: ₹{summary.get('total_invested', 0):,.0f}
- Current value: ₹{summary.get('current_value', 0):,.0f}
- Overall P&L: ₹{summary.get('overall_pnl', 0):,.0f} ({summary.get('overall_pnl_pct', 0):.2f}%)
- Risk level: {summary.get('risk_level', 'moderate')}
- Market sentiment: {market.get('sentiment', 'Neutral')}
- NIFTY change today: {market.get('nifty_change', 0):.2f}%
- Sector leaders: {', '.join(market.get('leaders', []))}
- Sector laggards: {', '.join(market.get('laggards', []))}
- AI market narrative: {reasoning.get('market_narrative', '')}
- Portfolio impact: {reasoning.get('portfolio_impact', '')}
- Holdings: {', '.join([h.get('symbol', '') for h in holdings])}
- Risk alerts: {'; '.join(ctx.get('risk_alerts', []))}
- Actionable insights: {'; '.join(reasoning.get('actionable_insights', []))}
- Positive signals: {'; '.join(reasoning.get('positive_signals', []))}
- Negative signals: {'; '.join(reasoning.get('negative_signals', []))}"""

    # ── Build messages list ────────────────────────────────────────────────
    messages = [{"role": "system", "content": system}]

    # Safely parse conversation history (handles both dict and string items)
    for msg in (req.history or [])[-6:]:
        if isinstance(msg, dict):
            role    = msg.get("role", "user")
            content = msg.get("content", "")
        elif isinstance(msg, str):
            # Plain string fallback — treat as user message
            role    = "user"
            content = msg
        else:
            continue

        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Add the current question
    messages.append({"role": "user", "content": req.message})

    # ── Call Groq ─────────────────────────────────────────────────────────
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=800,
            temperature=0.4,
            messages=messages,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("Chat response generated (%d chars)", len(answer))
        return {"answer": answer, "model": GROQ_MODEL}

    except Exception as e:
        logger.error("Groq chat error: %s", e)
        raise HTTPException(500, f"LLM error: {str(e)}")


# ── Local dev entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)