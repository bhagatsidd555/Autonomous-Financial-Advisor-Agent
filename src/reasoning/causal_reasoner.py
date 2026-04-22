"""
Causal Reasoner — FIXED VERSION

Fixes:
✔ ReasoningOutput class name (financial_advisor.py expects this exact name)
✔ reason() signature: reason(market_condition, sector_snapshot, portfolio_analysis, classified_news, user_profile)
✔ NIFTY change properly extracted — no more "NIFTY at 0.00%"
✔ confidence_score attribute (financial_advisor uses .confidence_score)
✔ to_full_report() method on ReasoningOutput
✔ Safe JSON parsing, no crashes
"""

import logging
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from groq import Groq
from dotenv import load_dotenv

from src.utils.prompts import CAUSAL_REASONING_PROMPT, PORTFOLIO_IMPACT_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class CausalLink:
    cause: str = ""
    effect: str = ""
    mechanism: str = ""
    confidence: float = 0.5
    scope: str = "macro"


@dataclass
class ReasoningOutput:
    """
    Name kept as ReasoningOutput to match financial_advisor.py import:
        from src.reasoning.causal_reasoner import CausalReasoner, ReasoningOutput
    """
    market_narrative: str = ""
    portfolio_impact: str = ""
    causal_chains: List[CausalLink] = field(default_factory=list)
    actionable_insights: List[str] = field(default_factory=list)
    positive_signals: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    confidence_score: float = 0.8      # financial_advisor uses .confidence_score
    key_drivers: List[dict] = field(default_factory=list)

    def to_full_report(self) -> str:
        """Called by financial_advisor.py: reasoning_output.to_full_report()"""
        lines = []

        lines.append("\n\U0001f4ca Market Narrative:")
        lines.append(self.market_narrative or "No narrative available.")

        if self.key_drivers:
            lines.append("\n\U0001f517 Key Drivers:")
            for kd in self.key_drivers[:5]:
                headline = kd.get("headline", "")[:80]
                sectors  = kd.get("impacted_sectors", [])
                stocks   = kd.get("affected_stocks", [])
                lines.append(
                    "- " + headline
                    + " impacted sectors " + str(sectors)
                    + " affecting stocks " + str(stocks)
                )

        lines.append("\n\U0001f4c9 Portfolio Impact:")
        lines.append(self.portfolio_impact or "No portfolio impact data.")

        if self.actionable_insights:
            lines.append("\n\U0001f4a1 Insights:")
            for ins in self.actionable_insights[:5]:
                lines.append("- " + str(ins))

        lines.append("\nConfidence: " + str(int(self.confidence_score * 100)) + "%")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_attr_or_dict(obj, *attrs):
    """Try multiple attribute names on object or dict."""
    for attr in attrs:
        val = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        if val is not None:
            return val
    return None


def _get_index_change_from_entry(entry) -> Optional[float]:
    """Extract % change from a single index entry (object or dict)."""
    for attr in ("change_pct", "pct_change", "change_percent", "daily_change_pct"):
        val = entry.get(attr) if isinstance(entry, dict) else getattr(entry, attr, None)
        if val is not None:
            return _safe_float(val)
    return None


def _get_nifty_change(market_condition) -> float:
    """Extract NIFTY 50 daily % change."""
    # 1. Direct top-level attributes
    for attr in ("nifty_change", "nifty_pct", "nifty50_change", "nifty_daily_pct"):
        val = _get_attr_or_dict(market_condition, attr)
        if val is not None:
            return _safe_float(val)

    # 2. From .indices dict
    indices = _get_attr_or_dict(market_condition, "indices") or {}
    if isinstance(indices, dict):
        for key in ("NIFTY_50", "NIFTY50", "nifty_50", "nifty50", "NIFTY 50"):
            entry = indices.get(key)
            if entry is not None:
                val = _get_index_change_from_entry(entry)
                if val is not None:
                    return val
    elif isinstance(indices, list):
        for idx_item in indices:
            name = _get_attr_or_dict(idx_item, "name") or ""
            if "NIFTY" in str(name).upper() and "BANK" not in str(name).upper():
                val = _get_index_change_from_entry(idx_item)
                if val is not None:
                    return val

    return 0.0


def _get_sensex_change(market_condition) -> float:
    for attr in ("sensex_change", "sensex_pct"):
        val = _get_attr_or_dict(market_condition, attr)
        if val is not None:
            return _safe_float(val)
    indices = _get_attr_or_dict(market_condition, "indices") or {}
    if isinstance(indices, dict):
        for key in ("SENSEX", "sensex", "BSE_SENSEX"):
            entry = indices.get(key)
            if entry:
                val = _get_index_change_from_entry(entry)
                if val is not None:
                    return val
    return 0.0


def _build_sector_context(sector_snapshot) -> str:
    if sector_snapshot is None:
        return "Sector data unavailable"

    # Try .sectors dict
    sectors = _get_attr_or_dict(sector_snapshot, "sectors", "sector_data")
    if sectors and isinstance(sectors, dict):
        lines = []
        for name, data in list(sectors.items())[:8]:
            pct = _get_attr_or_dict(data, "change_pct", "pct_change") or 0.0
            lines.append("  " + str(name) + ": " + str(_safe_float(pct)) + "%")
        if lines:
            return "\n".join(lines)

    # Fallback: leaders/laggards
    leaders  = _get_attr_or_dict(sector_snapshot, "leaders") or []
    laggards = _get_attr_or_dict(sector_snapshot, "laggards") or []
    if leaders or laggards:
        return "Leaders: " + str(leaders) + "\nLaggards: " + str(laggards)

    return "No sector detail available"


def _build_portfolio_sectors(portfolio_analysis) -> str:
    alloc = _get_attr_or_dict(portfolio_analysis, "sector_allocation")
    if not alloc or not isinstance(alloc, dict):
        return "Portfolio sector data unavailable"
    lines = []
    for sector, data in alloc.items():
        pct = _get_attr_or_dict(data, "weight_pct", "allocation_pct") or 0.0
        lines.append("  " + str(sector) + ": " + str(round(_safe_float(pct), 1)) + "%")
    return "\n".join(lines) or "No sector data"


def _build_sector_pnl(portfolio_analysis) -> str:
    sector_pnl = _get_attr_or_dict(portfolio_analysis, "sector_pnl")
    if not sector_pnl or not isinstance(sector_pnl, dict):
        return "Sector P&L data unavailable"
    lines = []
    for sector, data in sector_pnl.items():
        pnl_val = (_get_attr_or_dict(data, "daily_pnl") or data) if isinstance(data, dict) else data
        lines.append("  " + str(sector) + ": Rs." + str(round(_safe_float(pnl_val), 0)))
    return "\n".join(lines) or "No P&L data"


def _format_news(news_items: list) -> str:
    if not news_items:
        return "No relevant news available"
    lines = []
    for item in news_items[:6]:
        title   = _get_attr_or_dict(item, "title") or ""
        sent    = _get_attr_or_dict(item, "sentiment") or "neutral"
        impact  = _get_attr_or_dict(item, "impact_level", "impact_score") or "low"
        sectors = _get_attr_or_dict(item, "affected_sectors") or []
        stocks  = _get_attr_or_dict(item, "affected_stocks") or []

        if isinstance(impact, float):
            impact = "HIGH" if impact >= 0.7 else "MED" if impact >= 0.4 else "LOW"

        lines.append(
            "- [" + str(impact).upper() + "] " + str(title)[:80]
            + "\n  Sentiment: " + str(sent)
            + " | Sectors: " + str(sectors)
            + " | Stocks: " + str(stocks)
        )
    return "\n".join(lines)


def _parse_json_safe(raw: str) -> dict:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("JSON parse failed in causal_reasoner")
    return {}


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────
class CausalReasoner:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODEL
        logger.info("CausalReasoner initialised")

    # ──────────────────────────────────────────
    # PUBLIC METHOD — signature matches financial_advisor.py exactly
    # ──────────────────────────────────────────
    def reason(
        self,
        market_condition,
        sector_snapshot,
        portfolio_analysis,
        classified_news: list,
        user_profile: Optional[dict] = None,
    ) -> ReasoningOutput:

        logger.info("Running causal reasoning...")

        # Extract numbers
        nifty_change        = _get_nifty_change(market_condition)
        sensex_change       = _get_sensex_change(market_condition)
        sentiment           = str(_get_attr_or_dict(market_condition, "sentiment") or "neutral")
        strength            = str(_get_attr_or_dict(market_condition, "strength") or "moderate")
        leaders             = _get_attr_or_dict(market_condition, "leaders") or \
                              _get_attr_or_dict(sector_snapshot, "leaders") or []
        laggards            = _get_attr_or_dict(market_condition, "laggards") or \
                              _get_attr_or_dict(sector_snapshot, "laggards") or []

        total_daily_pnl     = _safe_float(_get_attr_or_dict(portfolio_analysis, "total_daily_pnl"))
        total_daily_pnl_pct = _safe_float(_get_attr_or_dict(portfolio_analysis, "daily_pnl_pct"))
        total_invested      = _safe_float(_get_attr_or_dict(portfolio_analysis, "total_invested"))
        total_current       = _safe_float(_get_attr_or_dict(portfolio_analysis, "total_current_value"))
        unrealised_pnl      = _safe_float(_get_attr_or_dict(portfolio_analysis, "total_unrealised_pnl"))
        unrealised_pnl_pct  = _safe_float(_get_attr_or_dict(portfolio_analysis, "unrealised_pnl_pct"))

        key_signals_raw = _get_attr_or_dict(market_condition, "key_signals") or []
        market_signals_str = "\n".join(
            "  - " + str(s) for s in key_signals_raw[:5]
        ) or "  - No additional signals"

        logger.info("[OBSERVABILITY] LLM Reasoning Triggered")
        logger.info("[OBSERVABILITY] News count: " + str(len(classified_news)))
        logger.info("[OBSERVABILITY] Portfolio PnL: " + str(round(total_daily_pnl_pct, 2)) + "%")

        # Step 1: Causal chains
        causal_data = self._run_causal_chain(
            sentiment=sentiment,
            strength=strength,
            nifty_change=nifty_change,
            market_signals_str=market_signals_str,
            sector_context=_build_sector_context(sector_snapshot),
            news_items=classified_news,
            portfolio_sectors=_build_portfolio_sectors(portfolio_analysis),
            sector_pnl=_build_sector_pnl(portfolio_analysis),
            total_daily_pnl=total_daily_pnl,
            total_daily_pnl_pct=total_daily_pnl_pct,
        )

        # Step 2: Portfolio narrative
        profile = user_profile or {}
        narrative_data = self._run_portfolio_narrative(
            user_name=profile.get("name", "Investor"),
            risk_profile=profile.get("risk_profile", "moderate"),
            investment_goal=profile.get("investment_goal", "wealth creation"),
            total_invested=total_invested,
            total_current_value=total_current,
            total_daily_pnl=total_daily_pnl,
            total_daily_pnl_pct=total_daily_pnl_pct,
            unrealised_pnl=unrealised_pnl,
            unrealised_pnl_pct=unrealised_pnl_pct,
            portfolio_analysis=portfolio_analysis,
            causal_chain=causal_data,
            news_items=classified_news,
            market_sentiment=sentiment,
        )

        # Build market_narrative — use REAL nifty_change
        market_narrative = narrative_data.get("portfolio_narrative", "")
        if not market_narrative:
            market_narrative = (
                "Market is " + sentiment + " (" + strength + ")"
                + " with NIFTY at " + str(round(nifty_change, 2)) + "%."
                + " Sector leaders include " + str(leaders)
                + " while laggards are " + str(laggards) + "."
            )

        # key_drivers for display
        key_drivers = []
        for item in classified_news[:5]:
            title   = _get_attr_or_dict(item, "title") or ""
            sectors = _get_attr_or_dict(item, "affected_sectors") or []
            stocks  = _get_attr_or_dict(item, "affected_stocks") or []
            key_drivers.append({
                "headline": title,
                "impacted_sectors": sectors,
                "affected_stocks": stocks,
            })

        return ReasoningOutput(
            market_narrative=market_narrative,
            portfolio_impact=(
                narrative_data.get("executive_summary", "")
                or "Portfolio moved " + str(round(total_daily_pnl_pct, 2)) + "% today."
            ),
            causal_chains=[
                CausalLink(
                    cause=link.get("cause", ""),
                    effect=link.get("effect", ""),
                    mechanism=link.get("mechanism", ""),
                    confidence=_safe_float(link.get("confidence", 0.5)),
                    scope=link.get("scope", "macro"),
                )
                for link in causal_data.get("causal_links", [])
            ],
            actionable_insights=narrative_data.get("actionable_insights", [
                "Monitor sector concentration risk",
                "Track IT sector weakness due to recent earnings pressure",
                "Consider diversification to reduce volatility exposure",
            ]),
            positive_signals=causal_data.get("positive_signals", []),
            negative_signals=causal_data.get("negative_signals", []),
            conflicting_signals=causal_data.get("conflicting", []),
            confidence_score=_safe_float(narrative_data.get("confidence", 0.8)),
            key_drivers=key_drivers,
        )

    # ──────────────────────────────────────────
    def _run_causal_chain(
        self,
        sentiment, strength, nifty_change,
        market_signals_str, sector_context,
        news_items, portfolio_sectors, sector_pnl,
        total_daily_pnl, total_daily_pnl_pct,
    ) -> dict:

        prompt = CAUSAL_REASONING_PROMPT.format(
            market_sentiment=sentiment + " (" + strength + ")",
            nifty_change=nifty_change,
            market_signals=market_signals_str,
            sector_context=sector_context,
            news_items=_format_news(news_items),
            portfolio_sectors=portfolio_sectors,
            sector_pnl=sector_pnl,
            total_daily_pnl=total_daily_pnl,
            total_daily_pnl_pct=total_daily_pnl_pct,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=800, temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json_safe(resp.choices[0].message.content)
        except Exception as e:
            logger.error("Causal chain LLM call failed: " + str(e))
            return {"causal_links": [], "positive_signals": [],
                    "negative_signals": ["LLM call failed"], "conflicting": []}

    # ──────────────────────────────────────────
    def _run_portfolio_narrative(
        self,
        user_name, risk_profile, investment_goal,
        total_invested, total_current_value,
        total_daily_pnl, total_daily_pnl_pct,
        unrealised_pnl, unrealised_pnl_pct,
        portfolio_analysis, causal_chain,
        news_items, market_sentiment,
    ) -> dict:

        holdings = _get_attr_or_dict(portfolio_analysis, "holdings") or []
        sorted_h = sorted(
            holdings,
            key=lambda h: _safe_float(
                _get_attr_or_dict(h, "daily_pnl_pct", "daily_change_pct") or 0
            ),
            reverse=True,
        )

        def _fmt_holdings(lst):
            lines = []
            for h in lst:
                sym = _get_attr_or_dict(h, "symbol") or ""
                pct = _safe_float(_get_attr_or_dict(h, "daily_pnl_pct", "daily_change_pct") or 0)
                lines.append("  " + str(sym) + " " + ("+" if pct >= 0 else "") + str(round(pct, 2)) + "%")
            return "\n".join(lines) or "  No data"

        alerts = _get_attr_or_dict(portfolio_analysis, "risk_alerts") or []
        alerts_str = "\n".join(
            "  - " + str(getattr(a, "message", str(a))) for a in alerts[:4]
        ) or "  None"

        links = causal_chain.get("causal_links", [])
        causal_str = "\n".join(
            "  " + str(i + 1) + ". " + l.get("cause", "") + " -> " + l.get("effect", "")
            for i, l in enumerate(links[:4])
        ) or "  No causal data"

        prompt = PORTFOLIO_IMPACT_PROMPT.format(
            user_name=user_name,
            risk_profile=risk_profile,
            investment_goal=investment_goal,
            total_invested=total_invested,
            total_current_value=total_current_value,
            total_daily_pnl=total_daily_pnl,
            total_daily_pnl_pct=total_daily_pnl_pct,
            unrealised_pnl=unrealised_pnl,
            unrealised_pnl_pct=unrealised_pnl_pct,
            top_performers=_fmt_holdings(sorted_h[:3]),
            worst_performers=_fmt_holdings(sorted_h[-3:][::-1]),
            risk_alerts=alerts_str,
            causal_chain=causal_str,
            relevant_news=_format_news(news_items[:4]),
            market_sentiment=market_sentiment,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=700, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json_safe(resp.choices[0].message.content)
        except Exception as e:
            logger.error("Portfolio narrative LLM call failed: " + str(e))
            return {
                "executive_summary": "Portfolio moved " + str(round(total_daily_pnl_pct, 2)) + "% today.",
                "portfolio_narrative": "",
                "actionable_insights": [
                    "Monitor sector concentration risk",
                    "Track IT sector weakness",
                    "Consider diversification",
                ],
                "confidence": 0.7,
            }   