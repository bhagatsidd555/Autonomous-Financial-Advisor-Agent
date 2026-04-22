"""
src/analytics/portfolio_analyzer.py
=====================================
Computes portfolio-level analytics:
  - Daily P&L and unrealised P&L
  - Asset and sector allocation
  - Concentration risk detection
  - Performance attribution by sector
  - Risk metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    CONCENTRATION_RISK_THRESHOLD,
    MODERATE_CONCENTRATION_THRESHOLD,
    BASE_CURRENCY_SYMBOL,
)
from src.ingestion.portfolio_loader import Portfolio, Holding

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class RiskAlert:
    """Represents a risk flag detected in portfolio."""
    level: str            # high | medium | low
    category: str         # concentration | volatility | drawdown | diversification
    message: str
    affected: list[str] = field(default_factory=list)  # symbols or sectors

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "affected": self.affected,
        }


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis result."""
    # P&L Summary
    total_invested: float
    total_current_value: float
    total_unrealised_pnl: float
    total_unrealised_pnl_pct: float
    total_daily_pnl: float
    total_daily_pnl_pct: float
    cash_balance: float

    # Allocation
    sector_allocation: dict        # sector → {value, pct}
    asset_allocation: dict         # asset_type → {value, pct}

    # Performance Attribution
    sector_daily_pnl: dict         # sector → daily_pnl
    sector_total_return: dict      # sector → total return %
    top_performers: list[dict]     # sorted by daily % gain
    worst_performers: list[dict]   # sorted by daily % loss

    # Risk Alerts
    risk_alerts: list[RiskAlert]
    overall_risk_level: str        # low | moderate | high | very_high

    def to_dict(self) -> dict:
        return {
            "pnl": {
                "total_invested": round(self.total_invested, 2),
                "total_current_value": round(self.total_current_value, 2),
                "total_unrealised_pnl": round(self.total_unrealised_pnl, 2),
                "total_unrealised_pnl_pct": round(self.total_unrealised_pnl_pct, 2),
                "total_daily_pnl": round(self.total_daily_pnl, 2),
                "total_daily_pnl_pct": round(self.total_daily_pnl_pct, 2),
                "cash_balance": round(self.cash_balance, 2),
            },
            "sector_allocation": self.sector_allocation,
            "asset_allocation": self.asset_allocation,
            "sector_daily_pnl": {k: round(v, 2) for k, v in self.sector_daily_pnl.items()},
            "sector_total_return": {k: round(v, 2) for k, v in self.sector_total_return.items()},
            "top_performers": self.top_performers[:5],
            "worst_performers": self.worst_performers[:5],
            "risk_alerts": [r.to_dict() for r in self.risk_alerts],
            "overall_risk_level": self.overall_risk_level,
        }

    def to_text_summary(self) -> str:
        """Human-readable portfolio summary."""
        sym = BASE_CURRENCY_SYMBOL
        daily_sign = "+" if self.total_daily_pnl >= 0 else ""
        pnl_sign = "+" if self.total_unrealised_pnl >= 0 else ""

        lines = [
            "═══ Portfolio Summary ═══",
            f"  Invested    : {sym}{self.total_invested:,.0f}",
            f"  Current     : {sym}{self.total_current_value:,.0f}",
            f"  Total P&L   : {sym}{pnl_sign}{self.total_unrealised_pnl:,.0f} ({pnl_sign}{self.total_unrealised_pnl_pct:.2f}%)",
            f"  Today's P&L : {sym}{daily_sign}{self.total_daily_pnl:,.0f} ({daily_sign}{self.total_daily_pnl_pct:.2f}%)",
            "",
            "═══ Sector Allocation ═══",
        ]
        for sector, alloc in sorted(self.sector_allocation.items(), key=lambda x: x[1]["pct"], reverse=True):
            lines.append(f"  {sector:<14} {sym}{alloc['value']:>10,.0f}  ({alloc['pct']:.1f}%)")

        if self.risk_alerts:
            lines.append("")
            lines.append("⚠️  Risk Alerts")
            for alert in self.risk_alerts:
                icon = "🔴" if alert.level == "high" else "🟡"
                lines.append(f"  {icon} [{alert.level.upper()}] {alert.message}")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────
class PortfolioAnalyzer:
    """
    Computes comprehensive portfolio analytics from an enriched Portfolio object.
    """

    def __init__(self):
        logger.info("PortfolioAnalyzer initialized")

    def analyze(self, portfolio: Portfolio) -> PortfolioAnalysis:
        """
        Run full portfolio analysis.

        Args:
            portfolio: Enriched Portfolio (must have current prices)

        Returns:
            PortfolioAnalysis dataclass
        """
        holdings = portfolio.holdings

        # ── P&L ──────────────────────────────
        total_invested = sum(h.invested_value for h in holdings)
        total_current_value = sum(h.current_value for h in holdings) + portfolio.cash_balance
        total_unrealised_pnl = total_current_value - total_invested - portfolio.cash_balance
        total_unrealised_pnl_pct = (total_unrealised_pnl / total_invested * 100) if total_invested else 0

        total_daily_pnl = sum(h.daily_pnl for h in holdings)
        previous_total = sum((h.previous_close or h.current_price) * h.quantity for h in holdings)
        total_daily_pnl_pct = (total_daily_pnl / previous_total * 100) if previous_total else 0

        # ── Allocation ────────────────────────
        sector_values: dict[str, float] = {}
        for h in holdings:
            sector_values[h.sector] = sector_values.get(h.sector, 0) + h.current_value

        equity_total = sum(h.current_value for h in holdings)
        sector_allocation = {
            sector: {
                "value": round(val, 2),
                "pct": round((val / equity_total * 100) if equity_total else 0, 2),
            }
            for sector, val in sector_values.items()
        }

        asset_values: dict[str, float] = {}
        for h in holdings:
            asset_values[h.asset_type] = asset_values.get(h.asset_type, 0) + h.current_value
        if portfolio.cash_balance > 0:
            asset_values["cash"] = portfolio.cash_balance

        total_with_cash = equity_total + portfolio.cash_balance
        asset_allocation = {
            asset: {
                "value": round(val, 2),
                "pct": round((val / total_with_cash * 100) if total_with_cash else 0, 2),
            }
            for asset, val in asset_values.items()
        }

        # ── Performance Attribution ───────────
        sector_daily_pnl: dict[str, float] = {}
        sector_invested: dict[str, float] = {}
        sector_current: dict[str, float] = {}

        for h in holdings:
            s = h.sector
            sector_daily_pnl[s] = sector_daily_pnl.get(s, 0) + h.daily_pnl
            sector_invested[s] = sector_invested.get(s, 0) + h.invested_value
            sector_current[s] = sector_current.get(s, 0) + h.current_value

        sector_total_return = {
            s: round(((sector_current[s] - sector_invested[s]) / sector_invested[s] * 100), 2)
            if sector_invested[s] else 0
            for s in sector_invested
        }

        # ── Individual Performers ─────────────
        performers = sorted(
            [h.to_dict() for h in holdings],
            key=lambda x: x["daily_change_pct"],
            reverse=True,
        )
        top_performers = performers[:5]
        worst_performers = performers[-5:][::-1]

        # ── Risk Alerts ───────────────────────
        risk_alerts = self._detect_risks(
            holdings, sector_allocation, total_daily_pnl_pct
        )
        overall_risk = self._compute_overall_risk(risk_alerts)

        analysis = PortfolioAnalysis(
            total_invested=round(total_invested, 2),
            total_current_value=round(total_current_value, 2),
            total_unrealised_pnl=round(total_unrealised_pnl, 2),
            total_unrealised_pnl_pct=round(total_unrealised_pnl_pct, 2),
            total_daily_pnl=round(total_daily_pnl, 2),
            total_daily_pnl_pct=round(total_daily_pnl_pct, 2),
            cash_balance=portfolio.cash_balance,
            sector_allocation=sector_allocation,
            asset_allocation=asset_allocation,
            sector_daily_pnl=sector_daily_pnl,
            sector_total_return=sector_total_return,
            top_performers=top_performers,
            worst_performers=worst_performers,
            risk_alerts=risk_alerts,
            overall_risk_level=overall_risk,
        )

        logger.info(
            "Portfolio analysis: daily P&L=%+.2f%% | overall_pnl=%+.2f%% | risk=%s",
            analysis.total_daily_pnl_pct,
            analysis.total_unrealised_pnl_pct,
            overall_risk,
        )
        return analysis

    # ── Risk Detection ───────────────────────
    def _detect_risks(
        self,
        holdings: list[Holding],
        sector_allocation: dict,
        daily_pnl_pct: float,
    ) -> list[RiskAlert]:
        """Detect risk conditions in the portfolio."""
        alerts = []

        # 1. Sector concentration risk
        for sector, alloc in sector_allocation.items():
            pct = alloc["pct"] / 100
            if pct > CONCENTRATION_RISK_THRESHOLD:
                alerts.append(RiskAlert(
                    level="high",
                    category="concentration",
                    message=f"High concentration in {sector} ({alloc['pct']:.1f}% of portfolio). "
                            f"Exceeds 40% threshold — consider diversifying.",
                    affected=[sector],
                ))
            elif pct > MODERATE_CONCENTRATION_THRESHOLD:
                alerts.append(RiskAlert(
                    level="medium",
                    category="concentration",
                    message=f"Moderate concentration in {sector} ({alloc['pct']:.1f}%). "
                            f"Monitor and consider rebalancing.",
                    affected=[sector],
                ))

        # 2. Large daily loss
        if daily_pnl_pct < -2.0:
            alerts.append(RiskAlert(
                level="high",
                category="drawdown",
                message=f"Portfolio is down {abs(daily_pnl_pct):.2f}% today — significant daily drawdown.",
                affected=[],
            ))
        elif daily_pnl_pct < -1.0:
            alerts.append(RiskAlert(
                level="medium",
                category="drawdown",
                message=f"Portfolio is down {abs(daily_pnl_pct):.2f}% today.",
                affected=[],
            ))

        # 3. Single stock over-weight
        total_equity = sum(h.current_value for h in holdings)
        for h in holdings:
            stock_pct = (h.current_value / total_equity * 100) if total_equity else 0
            if stock_pct > 25:
                alerts.append(RiskAlert(
                    level="high",
                    category="concentration",
                    message=f"{h.name} ({h.symbol}) represents {stock_pct:.1f}% of portfolio — very high single-stock exposure.",
                    affected=[h.symbol],
                ))
            elif stock_pct > 15:
                alerts.append(RiskAlert(
                    level="medium",
                    category="concentration",
                    message=f"{h.name} represents {stock_pct:.1f}% of portfolio.",
                    affected=[h.symbol],
                ))

        # 4. Number of sectors / diversification
        num_sectors = len(sector_allocation)
        if num_sectors < 3:
            alerts.append(RiskAlert(
                level="medium",
                category="diversification",
                message=f"Portfolio spread across only {num_sectors} sector(s). "
                        f"Low diversification increases sector-specific risk.",
                affected=list(sector_allocation.keys()),
            ))

        # 5. Significant single stock loss today
        for h in holdings:
            if h.daily_change_pct < -4.0:
                alerts.append(RiskAlert(
                    level="medium",
                    category="volatility",
                    message=f"{h.name} fell {abs(h.daily_change_pct):.1f}% today — unusual daily move.",
                    affected=[h.symbol],
                ))

        return alerts

    def _compute_overall_risk(self, alerts: list[RiskAlert]) -> str:
        """Compute overall risk level from individual alerts."""
        high_count = sum(1 for a in alerts if a.level == "high")
        medium_count = sum(1 for a in alerts if a.level == "medium")

        if high_count >= 2:
            return "very_high"
        elif high_count >= 1:
            return "high"
        elif medium_count >= 2:
            return "moderate"
        elif medium_count >= 1:
            return "low"
        return "low"

    def compare_vs_benchmark(
        self,
        portfolio_daily_pct: float,
        nifty_daily_pct: float,
    ) -> dict:
        """Compare portfolio performance against NIFTY 50 benchmark."""
        alpha = portfolio_daily_pct - nifty_daily_pct
        return {
            "portfolio_pct": round(portfolio_daily_pct, 4),
            "benchmark_pct": round(nifty_daily_pct, 4),
            "alpha": round(alpha, 4),
            "outperforming": alpha > 0,
            "summary": (
                f"Portfolio {'outperformed' if alpha > 0 else 'underperformed'} "
                f"NIFTY 50 by {abs(alpha):.2f}% today"
            ),
        }
