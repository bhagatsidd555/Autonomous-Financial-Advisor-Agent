"""
src/ingestion/portfolio_loader.py
===================================
Loads and validates user portfolio data from JSON files.
Provides a clean Portfolio data model for analytics processing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import STOCK_SECTOR_MAP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class Holding:
    """Represents a single stock holding in the portfolio."""
    symbol: str
    name: str
    quantity: int
    avg_buy_price: float
    sector: str
    asset_type: str = "equity"

    # Populated after market data fetch
    current_price: float = 0.0
    current_value: float = 0.0
    invested_value: float = 0.0
    unrealised_pnl: float = 0.0
    unrealised_pnl_pct: float = 0.0
    daily_change: float = 0.0
    daily_change_pct: float = 0.0
    daily_pnl: float = 0.0
    previous_close: float = 0.0

    def __post_init__(self):
        self.invested_value = self.avg_buy_price * self.quantity

    def update_with_market_data(self, market_data: dict):
        """Populate live price fields from market data dict."""
        self.current_price = market_data.get("current", self.avg_buy_price)
        self.previous_close = market_data.get("previous_close", self.current_price)
        self.current_value = self.current_price * self.quantity
        self.invested_value = self.avg_buy_price * self.quantity
        self.unrealised_pnl = self.current_value - self.invested_value
        self.unrealised_pnl_pct = (self.unrealised_pnl / self.invested_value * 100) if self.invested_value else 0
        self.daily_change = market_data.get("change", 0)
        self.daily_change_pct = market_data.get("change_pct", 0)
        self.daily_pnl = self.daily_change * self.quantity

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "quantity": self.quantity,
            "avg_buy_price": self.avg_buy_price,
            "sector": self.sector,
            "asset_type": self.asset_type,
            "current_price": round(self.current_price, 2),
            "current_value": round(self.current_value, 2),
            "invested_value": round(self.invested_value, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct, 2),
            "daily_change": round(self.daily_change, 2),
            "daily_change_pct": round(self.daily_change_pct, 2),
            "daily_pnl": round(self.daily_pnl, 2),
        }


@dataclass
class UserProfile:
    """User profile and risk preferences."""
    id: str
    name: str
    risk_profile: str  # conservative, moderate, aggressive
    investment_goal: str
    investment_horizon_years: int


@dataclass
class Portfolio:
    """Complete user portfolio with all holdings."""
    user: UserProfile
    holdings: list[Holding]
    cash_balance: float = 0.0
    total_invested: float = 0.0

    # Populated after market data
    total_current_value: float = 0.0
    total_unrealised_pnl: float = 0.0
    total_unrealised_pnl_pct: float = 0.0
    total_daily_pnl: float = 0.0
    sector_allocation: dict = field(default_factory=dict)
    asset_allocation: dict = field(default_factory=dict)

    def get_symbols(self) -> list[str]:
        """Return list of all stock symbols in portfolio."""
        return [h.symbol for h in self.holdings]

    def get_holding(self, symbol: str) -> Optional[Holding]:
        """Get a specific holding by symbol."""
        for h in self.holdings:
            if h.symbol == symbol:
                return h
        return None

    def recalculate_totals(self):
        """Recompute portfolio-level totals from holding data."""
        self.total_current_value = sum(h.current_value for h in self.holdings) + self.cash_balance
        self.total_invested = sum(h.invested_value for h in self.holdings)
        self.total_unrealised_pnl = sum(h.unrealised_pnl for h in self.holdings)
        self.total_unrealised_pnl_pct = (
            (self.total_unrealised_pnl / self.total_invested * 100)
            if self.total_invested else 0
        )
        self.total_daily_pnl = sum(h.daily_pnl for h in self.holdings)

        # Sector allocation
        sector_values: dict[str, float] = {}
        for h in self.holdings:
            sector_values[h.sector] = sector_values.get(h.sector, 0) + h.current_value

        equity_total = sum(h.current_value for h in self.holdings)
        self.sector_allocation = {
            sector: {
                "value": round(val, 2),
                "pct": round((val / equity_total * 100) if equity_total else 0, 2),
            }
            for sector, val in sector_values.items()
        }

        # Asset type allocation
        asset_values: dict[str, float] = {}
        for h in self.holdings:
            asset_values[h.asset_type] = asset_values.get(h.asset_type, 0) + h.current_value
        if self.cash_balance > 0:
            asset_values["cash"] = self.cash_balance

        portfolio_total = equity_total + self.cash_balance
        self.asset_allocation = {
            asset: {
                "value": round(val, 2),
                "pct": round((val / portfolio_total * 100) if portfolio_total else 0, 2),
            }
            for asset, val in asset_values.items()
        }

    def to_summary_dict(self) -> dict:
        """Compact summary for LLM reasoning."""
        return {
            "user": {
                "name": self.user.name,
                "risk_profile": self.user.risk_profile,
                "investment_goal": self.user.investment_goal,
            },
            "overview": {
                "total_invested": round(self.total_invested, 2),
                "total_current_value": round(self.total_current_value, 2),
                "total_unrealised_pnl": round(self.total_unrealised_pnl, 2),
                "total_unrealised_pnl_pct": round(self.total_unrealised_pnl_pct, 2),
                "total_daily_pnl": round(self.total_daily_pnl, 2),
                "cash_balance": round(self.cash_balance, 2),
            },
            "holdings": [h.to_dict() for h in self.holdings],
            "sector_allocation": self.sector_allocation,
            "asset_allocation": self.asset_allocation,
        }


# ─────────────────────────────────────────────
# Portfolio Loader
# ─────────────────────────────────────────────
class PortfolioLoader:
    """
    Loads user portfolio data from JSON files or dictionaries.
    Validates structure and returns clean Portfolio objects.
    """

    def __init__(self):
        self.sample_path = Path(__file__).parent.parent.parent / "data" / "sample_portfolio.json"

    def load_from_file(self, path: Optional[str] = None) -> Portfolio:
        """
        Load portfolio from a JSON file.

        Args:
            path: File path. Uses sample data if None.

        Returns:
            Portfolio object (without live prices - needs market data enrichment)
        """
        file_path = Path(path) if path else self.sample_path

        if not file_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            logger.info("Loaded portfolio from %s", file_path)
            return self._parse_portfolio(raw)

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in portfolio file: %s", str(e))
            raise

    def load_from_dict(self, data: dict) -> Portfolio:
        """Load portfolio from a Python dictionary."""
        return self._parse_portfolio(data)

    def _parse_portfolio(self, raw: dict) -> Portfolio:
        """Parse raw dict into Portfolio dataclass."""
        self._validate_structure(raw)

        # User profile
        user_raw = raw["user"]
        user = UserProfile(
            id=user_raw.get("id", "unknown"),
            name=user_raw.get("name", "Unknown User"),
            risk_profile=user_raw.get("risk_profile", "moderate"),
            investment_goal=user_raw.get("investment_goal", "growth"),
            investment_horizon_years=user_raw.get("investment_horizon_years", 5),
        )

        # Holdings
        holdings = []
        for raw_holding in raw["portfolio"]["holdings"]:
            symbol = raw_holding["symbol"]
            sector = raw_holding.get("sector") or STOCK_SECTOR_MAP.get(symbol, "Unknown")

            holding = Holding(
                symbol=symbol,
                name=raw_holding.get("name", symbol),
                quantity=int(raw_holding["quantity"]),
                avg_buy_price=float(raw_holding["avg_buy_price"]),
                sector=sector,
                asset_type=raw_holding.get("asset_type", "equity"),
            )
            holdings.append(holding)

        portfolio = Portfolio(
            user=user,
            holdings=holdings,
            cash_balance=float(raw["portfolio"].get("cash_balance", 0)),
            total_invested=float(raw["portfolio"].get("total_invested", 0)),
        )

        logger.info(
            "Parsed portfolio for %s with %d holdings",
            user.name,
            len(holdings),
        )
        return portfolio

    def _validate_structure(self, raw: dict):
        """Validate required fields in raw portfolio data."""
        if "user" not in raw:
            raise ValueError("Portfolio JSON missing 'user' section")
        if "portfolio" not in raw:
            raise ValueError("Portfolio JSON missing 'portfolio' section")
        if "holdings" not in raw["portfolio"]:
            raise ValueError("Portfolio JSON missing 'portfolio.holdings'")

        for i, h in enumerate(raw["portfolio"]["holdings"]):
            if "symbol" not in h:
                raise ValueError(f"Holding {i} missing 'symbol'")
            if "quantity" not in h:
                raise ValueError(f"Holding {i} missing 'quantity'")
            if "avg_buy_price" not in h:
                raise ValueError(f"Holding {i} missing 'avg_buy_price'")

    def enrich_with_market_data(
        self, portfolio: Portfolio, stocks_market_data: dict[str, dict]
    ) -> Portfolio:
        """
        Enrich portfolio holdings with live market prices.

        Args:
            portfolio: Portfolio object (without prices)
            stocks_market_data: Dict mapping symbol → market data

        Returns:
            Enriched Portfolio with live prices and P&L calculated
        """
        enriched_count = 0
        for holding in portfolio.holdings:
            if holding.symbol in stocks_market_data:
                market_data = stocks_market_data[holding.symbol]
                holding.update_with_market_data(market_data)
                enriched_count += 1
            else:
                logger.warning("No market data for %s; using buy price", holding.symbol)
                holding.current_price = holding.avg_buy_price
                holding.current_value = holding.avg_buy_price * holding.quantity

        # Recalculate portfolio totals
        portfolio.recalculate_totals()

        logger.info(
            "Enriched %d/%d holdings with market data",
            enriched_count,
            len(portfolio.holdings),
        )
        return portfolio
