"""
src/ingestion/market_data.py
=============================
Fetches live and historical market data for Indian indices and stocks
using the yfinance library. Handles NIFTY 50, SENSEX, sector indices,
and individual stock prices with caching support.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache

import yfinance as yf
import pandas as pd
import numpy as np

from config.settings import MARKET_INDICES, STOCK_SECTOR_MAP, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Simple in-memory cache with TTL
# ─────────────────────────────────────────────
_cache: dict = {}

def _get_cache(key: str):
    """Return cached value if not expired."""
    if key in _cache:
        value, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return value
    return None

def _set_cache(key: str, value):
    """Store value in cache with current timestamp."""
    _cache[key] = (value, time.time())


# ─────────────────────────────────────────────
# Core Market Data Fetcher
# ─────────────────────────────────────────────
class MarketDataFetcher:
    """
    Fetches market index data, sector indices, and stock prices
    from Yahoo Finance. Provides clean structured output for the
    analytics layer.
    """

    def __init__(self):
        self.indices = MARKET_INDICES
        logger.info("MarketDataFetcher initialized with %d indices", len(self.indices))

    # ── Indices ──────────────────────────────
    def fetch_index_data(self, symbol: str, period: str = "5d") -> Optional[dict]:
        """
        Fetch OHLCV data for a single index symbol.

        Args:
            symbol: Yahoo Finance ticker symbol (e.g., '^NSEI')
            period: Time period - '1d', '5d', '1mo', '3mo', '1y'

        Returns:
            dict with 'current', 'previous_close', 'change', 'change_pct',
            'high', 'low', 'volume', 'history' (DataFrame)
        """
        cache_key = f"index_{symbol}_{period}"
        cached = _get_cache(cache_key)
        if cached:
            logger.debug("Cache hit: %s", cache_key)
            return cached

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                logger.warning("No data returned for symbol: %s", symbol)
                return None

            current_close = float(hist["Close"].iloc[-1])
            previous_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_close
            change = current_close - previous_close
            change_pct = (change / previous_close) * 100 if previous_close != 0 else 0

            result = {
                "symbol": symbol,
                "current": round(current_close, 2),
                "previous_close": round(previous_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 4),
                "high": round(float(hist["High"].iloc[-1]), 2),
                "low": round(float(hist["Low"].iloc[-1]), 2),
                "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                "52w_high": round(float(hist["High"].max()), 2) if len(hist) >= 200 else None,
                "52w_low": round(float(hist["Low"].min()), 2) if len(hist) >= 200 else None,
                "history": hist,
                "fetched_at": datetime.now().isoformat(),
            }

            _set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error("Error fetching index %s: %s", symbol, str(e))
            return None

    def fetch_all_indices(self) -> dict[str, dict]:
        """
        Fetch data for all configured market indices.

        Returns:
            Dictionary mapping index name → index data dict
        """
        results = {}
        for name, symbol in self.indices.items():
            logger.info("Fetching index: %s (%s)", name, symbol)
            data = self.fetch_index_data(symbol)
            if data:
                data["name"] = name
                results[name] = data
            else:
                logger.warning("Failed to fetch data for %s", name)

        logger.info("Successfully fetched %d/%d indices", len(results), len(self.indices))
        return results

    # ── Individual Stocks ────────────────────
    def fetch_stock_data(self, symbol: str, period: str = "5d") -> Optional[dict]:
        """
        Fetch price data for an individual stock.

        Args:
            symbol: NSE symbol (e.g., 'HDFCBANK.NS')
            period: Time period

        Returns:
            dict with price info + fundamental data
        """
        cache_key = f"stock_{symbol}_{period}"
        cached = _get_cache(cache_key)
        if cached:
            return cached

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                logger.warning("No price data for stock: %s", symbol)
                return None

            current_close = float(hist["Close"].iloc[-1])
            previous_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_close
            change = current_close - previous_close
            change_pct = (change / previous_close) * 100 if previous_close != 0 else 0

            # Try to get basic info
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                pass

            result = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "current": round(current_close, 2),
                "previous_close": round(previous_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 4),
                "high": round(float(hist["High"].iloc[-1]), 2),
                "low": round(float(hist["Low"].iloc[-1]), 2),
                "volume": int(hist["Volume"].iloc[-1]),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "sector": STOCK_SECTOR_MAP.get(symbol, info.get("sector", "Unknown")),
                "history": hist,
                "fetched_at": datetime.now().isoformat(),
            }

            _set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error("Error fetching stock %s: %s", symbol, str(e))
            return None

    def fetch_multiple_stocks(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch data for multiple stocks efficiently using yfinance batch download.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary mapping symbol → stock data dict
        """
        results = {}

        # Batch download for efficiency
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period="5d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error("Batch download failed: %s", str(e))
            raw = pd.DataFrame()

        for symbol in symbols:
            try:
                if not raw.empty and symbol in raw.columns.get_level_values(0):
                    stock_hist = raw[symbol].dropna()
                else:
                    # Fallback to individual fetch
                    individual = self.fetch_stock_data(symbol)
                    if individual:
                        results[symbol] = individual
                    continue

                if stock_hist.empty:
                    continue

                current_close = float(stock_hist["Close"].iloc[-1])
                previous_close = float(stock_hist["Close"].iloc[-2]) if len(stock_hist) > 1 else current_close
                change = current_close - previous_close
                change_pct = (change / previous_close) * 100 if previous_close != 0 else 0

                results[symbol] = {
                    "symbol": symbol,
                    "current": round(current_close, 2),
                    "previous_close": round(previous_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 4),
                    "high": round(float(stock_hist["High"].iloc[-1]), 2),
                    "low": round(float(stock_hist["Low"].iloc[-1]), 2),
                    "volume": int(stock_hist["Volume"].iloc[-1]),
                    "sector": STOCK_SECTOR_MAP.get(symbol, "Unknown"),
                    "history": stock_hist,
                    "fetched_at": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error("Error processing stock %s from batch: %s", symbol, str(e))
                # Try individual fetch as fallback
                individual = self.fetch_stock_data(symbol)
                if individual:
                    results[symbol] = individual

        logger.info("Fetched %d/%d stocks", len(results), len(symbols))
        return results

    # ── Volatility & Technical Signals ───────
    def calculate_volatility(self, history: pd.DataFrame, window: int = 20) -> float:
        """Calculate annualised volatility from price history."""
        try:
            returns = history["Close"].pct_change().dropna()
            daily_vol = returns.std()
            annualised_vol = daily_vol * np.sqrt(252)
            return round(float(annualised_vol), 4)
        except Exception:
            return 0.0

    def get_market_breadth(self, stocks_data: dict[str, dict]) -> dict:
        """
        Calculate market breadth: advancers vs decliners.

        Returns:
            dict with advancing, declining, unchanged counts and ratio
        """
        advancing = sum(1 for s in stocks_data.values() if s.get("change_pct", 0) > 0)
        declining = sum(1 for s in stocks_data.values() if s.get("change_pct", 0) < 0)
        unchanged = len(stocks_data) - advancing - declining

        total = len(stocks_data)
        ratio = advancing / declining if declining > 0 else float("inf")

        return {
            "total": total,
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "advance_decline_ratio": round(ratio, 2),
        }
