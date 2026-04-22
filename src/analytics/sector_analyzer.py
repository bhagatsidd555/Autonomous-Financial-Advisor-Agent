"""
src/analytics/sector_analyzer.py
==================================
Analyzes sector-level performance from index data and stock prices.
Identifies sector leaders, laggards, rotation patterns, and
generates structured sector insights for the reasoning layer.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class SectorPerformance:
    """Performance data for a single market sector."""
    sector: str
    change_pct: float
    sentiment: str               # bullish | bearish | neutral
    index_symbol: str = ""
    index_change_pct: float = 0.0
    top_gainers: list[str] = field(default_factory=list)
    top_losers: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "change_pct": round(self.change_pct, 4),
            "sentiment": self.sentiment,
            "index_symbol": self.index_symbol,
            "index_change_pct": round(self.index_change_pct, 4),
            "top_gainers": self.top_gainers,
            "top_losers": self.top_losers,
            "notes": self.notes,
        }


@dataclass
class SectorSnapshot:
    """Full snapshot of all sectors at a point in time."""
    sectors: list[SectorPerformance]
    leaders: list[str]           # best performing sectors
    laggards: list[str]          # worst performing sectors
    rotation_signal: str         # defensive | growth | cyclical | mixed
    breadth: str                 # broad-based | selective | narrow

    def to_dict(self) -> dict:
        return {
            "sectors": [s.to_dict() for s in self.sectors],
            "leaders": self.leaders,
            "laggards": self.laggards,
            "rotation_signal": self.rotation_signal,
            "breadth": self.breadth,
        }

    def get_sector(self, name: str) -> SectorPerformance:
        for s in self.sectors:
            if s.sector.lower() == name.lower():
                return s
        return None

    def to_text_summary(self) -> str:
        lines = [f"Sector Rotation: {self.rotation_signal.upper()} | Breadth: {self.breadth}"]
        lines.append(f"  Leaders  : {', '.join(self.leaders) if self.leaders else 'None'}")
        lines.append(f"  Laggards : {', '.join(self.laggards) if self.laggards else 'None'}")
        lines.append("")
        for s in sorted(self.sectors, key=lambda x: x.change_pct, reverse=True):
            arrow = "▲" if s.change_pct >= 0 else "▼"
            lines.append(f"  {arrow} {s.sector:<12} {s.change_pct:+.2f}%")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────
class SectorAnalyzer:
    """
    Derives sector-level performance from market index data
    and individual stock price movements.
    """

    # Mapping from index key → sector name
    INDEX_TO_SECTOR = {
        "NIFTY_BANK": "Banking",
        "NIFTY_IT": "IT",
        "NIFTY_PHARMA": "Pharma",
        "NIFTY_AUTO": "Auto",
        "NIFTY_FMCG": "FMCG",
        "NIFTY_METAL": "Metal",
        "NIFTY_REALTY": "Realty",
        "NIFTY_ENERGY": "Energy",
    }

    # Defensive vs growth sector classification
    DEFENSIVE_SECTORS = {"FMCG", "Pharma", "Telecom"}
    GROWTH_SECTORS = {"IT", "Auto", "Realty"}
    CYCLICAL_SECTORS = {"Banking", "Metal", "Energy"}

    def __init__(self):
        logger.info("SectorAnalyzer initialized")

    def analyze(
        self,
        indices_data: dict[str, dict],
        stocks_data: dict[str, dict] = None,
    ) -> SectorSnapshot:
        """
        Build a full sector snapshot.

        Args:
            indices_data: Output from MarketDataFetcher.fetch_all_indices()
            stocks_data: Optional stock-level data for within-sector analysis

        Returns:
            SectorSnapshot with all sector performance data
        """
        sectors = []

        # Derive sectors from index data
        for index_key, sector_name in self.INDEX_TO_SECTOR.items():
            if index_key in indices_data:
                idx = indices_data[index_key]
                chg = idx.get("change_pct", 0) or 0
                sentiment = self._change_to_sentiment(chg)

                sp = SectorPerformance(
                    sector=sector_name,
                    change_pct=round(chg, 4),
                    sentiment=sentiment,
                    index_symbol=idx.get("symbol", ""),
                    index_change_pct=round(chg, 4),
                    notes=self._generate_sector_note(sector_name, chg),
                )
                sectors.append(sp)

        # Enrich with stock-level data if available
        if stocks_data:
            sectors = self._enrich_with_stocks(sectors, stocks_data)

        # If no sector data from indices, derive from stocks only
        if not sectors and stocks_data:
            sectors = self._derive_sectors_from_stocks(stocks_data)

        # Sort by performance
        sectors.sort(key=lambda s: s.change_pct, reverse=True)

        # Leaders and Laggards
        leaders = [s.sector for s in sectors if s.change_pct > 0.5][:3]
        laggards = [s.sector for s in sectors if s.change_pct < -0.5][:3]

        # Rotation signal
        rotation = self._identify_rotation(sectors)
        breadth = self._assess_breadth(sectors)

        snapshot = SectorSnapshot(
            sectors=sectors,
            leaders=leaders,
            laggards=laggards,
            rotation_signal=rotation,
            breadth=breadth,
        )

        logger.info(
            "Sector analysis: %d sectors | Leaders: %s | Laggards: %s",
            len(sectors), leaders, laggards
        )
        return snapshot

    def get_sectors_for_portfolio(
        self, snapshot: SectorSnapshot, portfolio_sectors: list[str]
    ) -> list[SectorPerformance]:
        """
        Filter sector snapshot to only sectors in user's portfolio.

        Args:
            snapshot: Full market sector snapshot
            portfolio_sectors: List of sector names from user's holdings

        Returns:
            Filtered list of SectorPerformance objects
        """
        portfolio_sector_set = {s.lower() for s in portfolio_sectors}
        return [
            s for s in snapshot.sectors
            if s.sector.lower() in portfolio_sector_set
        ]

    # ── Private Helpers ──────────────────────
    def _change_to_sentiment(self, change_pct: float) -> str:
        """Classify % change into sentiment label."""
        if change_pct > 0.3:
            return "bullish"
        elif change_pct < -0.3:
            return "bearish"
        return "neutral"

    def _generate_sector_note(self, sector: str, change_pct: float) -> str:
        """Generate contextual note based on sector and movement."""
        notes_map = {
            "Banking": {
                "up_strong": "Bank Nifty rally — positive RBI/credit signal",
                "down_strong": "Banking under pressure — rate/NPA concerns",
                "neutral": "Banking sector consolidating",
            },
            "IT": {
                "up_strong": "IT outperforming — USD strength or positive US data",
                "down_strong": "IT under pressure — US slowdown fears or INR appreciation",
                "neutral": "IT sector range-bound",
            },
            "Pharma": {
                "up_strong": "Pharma rallying — defensive buying or positive USFDA news",
                "down_strong": "Pharma falling — regulatory concerns or profit booking",
                "neutral": "Pharma sector stable",
            },
            "Auto": {
                "up_strong": "Auto sector strong — positive sales data or EV optimism",
                "down_strong": "Auto under pressure — input cost/EV transition concerns",
                "neutral": "Auto sector mixed",
            },
            "Metal": {
                "up_strong": "Metal sector surging — global commodity tailwind",
                "down_strong": "Metal falling — China slowdown or commodity weakness",
                "neutral": "Metal sector consolidating",
            },
            "Energy": {
                "up_strong": "Energy sector rallying — crude oil price rise",
                "down_strong": "Energy under pressure — crude correction",
                "neutral": "Energy sector neutral",
            },
        }

        defaults = {
            "up_strong": f"{sector} sector rallying",
            "down_strong": f"{sector} sector falling",
            "neutral": f"{sector} sector neutral",
        }

        template = notes_map.get(sector, defaults)

        if change_pct > 1.0:
            return template["up_strong"]
        elif change_pct < -1.0:
            return template["down_strong"]
        else:
            return template["neutral"]

    def _enrich_with_stocks(
        self,
        sectors: list[SectorPerformance],
        stocks_data: dict[str, dict],
    ) -> list[SectorPerformance]:
        """Add stock-level gainers/losers within each sector."""
        sector_stocks: dict[str, list] = {}
        for symbol, data in stocks_data.items():
            sector = data.get("sector", "Unknown")
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append((symbol, data.get("change_pct", 0)))

        for sp in sectors:
            stocks = sector_stocks.get(sp.sector, [])
            stocks.sort(key=lambda x: x[1], reverse=True)
            sp.top_gainers = [s[0] for s in stocks[:3] if s[1] > 0]
            sp.top_losers = [s[0] for s in stocks[-3:] if s[1] < 0]

        return sectors

    def _derive_sectors_from_stocks(
        self, stocks_data: dict[str, dict]
    ) -> list[SectorPerformance]:
        """Build sector performance from individual stock data."""
        sector_changes: dict[str, list[float]] = {}
        sector_stocks: dict[str, list] = {}

        for symbol, data in stocks_data.items():
            sector = data.get("sector", "Unknown")
            chg = data.get("change_pct", 0) or 0
            if sector not in sector_changes:
                sector_changes[sector] = []
                sector_stocks[sector] = []
            sector_changes[sector].append(chg)
            sector_stocks[sector].append((symbol, chg))

        sectors = []
        for sector, changes in sector_changes.items():
            avg_chg = sum(changes) / len(changes)
            stocks = sorted(sector_stocks[sector], key=lambda x: x[1], reverse=True)

            sp = SectorPerformance(
                sector=sector,
                change_pct=round(avg_chg, 4),
                sentiment=self._change_to_sentiment(avg_chg),
                top_gainers=[s[0] for s in stocks[:2] if s[1] > 0],
                top_losers=[s[0] for s in stocks[-2:] if s[1] < 0],
                notes=self._generate_sector_note(sector, avg_chg),
            )
            sectors.append(sp)

        return sectors

    def _identify_rotation(self, sectors: list[SectorPerformance]) -> str:
        """Identify sector rotation signal."""
        if not sectors:
            return "mixed"

        positive = {s.sector for s in sectors if s.change_pct > 0}
        negative = {s.sector for s in sectors if s.change_pct < 0}

        defensive_up = bool(positive & self.DEFENSIVE_SECTORS)
        growth_up = bool(positive & self.GROWTH_SECTORS)
        cyclical_up = bool(positive & self.CYCLICAL_SECTORS)
        defensive_down = bool(negative & self.DEFENSIVE_SECTORS)

        if defensive_up and not growth_up and not cyclical_up:
            return "defensive"
        elif growth_up and cyclical_up and not defensive_down:
            return "growth"
        elif cyclical_up and not defensive_up:
            return "cyclical"
        elif len(positive) > len(negative):
            return "broad-based rally"
        elif len(negative) > len(positive):
            return "broad-based decline"
        return "mixed"

    def _assess_breadth(self, sectors: list[SectorPerformance]) -> str:
        """Assess market breadth from sector data."""
        if not sectors:
            return "unknown"

        positive = sum(1 for s in sectors if s.change_pct > 0)
        total = len(sectors)
        pct = positive / total

        if pct > 0.75:
            return "broad-based"
        elif pct > 0.5:
            return "moderate"
        elif pct > 0.25:
            return "selective"
        return "narrow"
