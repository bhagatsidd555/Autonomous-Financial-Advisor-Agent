
import os
from dotenv import load_dotenv

load_dotenv()



GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "dummy_not_needed")  # kept for compatibility
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


APP_ENV: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
ENABLE_OBSERVABILITY: bool = os.getenv("ENABLE_OBSERVABILITY", "false").lower() == "true"
ENABLE_SELF_EVALUATION: bool = os.getenv("ENABLE_SELF_EVALUATION", "true").lower() == "true"


CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))


DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "INR")
BASE_CURRENCY_SYMBOL: str = os.getenv("BASE_CURRENCY_SYMBOL", "₹")


MARKET_INDICES = {
    "NIFTY_50":     "^NSEI",
    "SENSEX":       "^BSESN",
    "NIFTY_BANK":   "^NSEBANK",
    "NIFTY_IT":     "^CNXIT",
    "NIFTY_PHARMA": "NIFTYPHARMA.NS",
    "NIFTY_AUTO":   "^CNXAUTO",
    "NIFTY_FMCG":   "^CNXFMCG",
    "NIFTY_METAL":  "^CNXMETAL",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY",
}


STOCK_SECTOR_MAP = {
    "HDFCBANK.NS":  "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS":      "Banking",
    "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS":  "Banking",
    "BAJFINANCE.NS":  "Finance",
    "BAJAJFINSV.NS":  "Finance",
    # IT
    "TCS.NS":     "IT",
    "INFY.NS":    "IT",
    "WIPRO.NS":   "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS":   "IT",
    # Pharma
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS":   "Pharma",
    "CIPLA.NS":     "Pharma",
    "DIVISLAB.NS":  "Pharma",
    # Auto
    "MARUTI.NS":     "Auto",
    "TATAMOTORS.NS": "Auto",
    "M&M.NS":        "Auto",
    "HEROMOTOCO.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    # Energy / Oil & Gas
    "RELIANCE.NS": "Energy",
    "ONGC.NS":     "Energy",
    "BPCL.NS":     "Energy",
    "IOC.NS":      "Energy",
    "NTPC.NS":     "Energy",
    "POWERGRID.NS":"Energy",
    # FMCG
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS":        "FMCG",
    "NESTLEIND.NS":  "FMCG",
    "BRITANNIA.NS":  "FMCG",
    # Metal
    "TATASTEEL.NS": "Metal",
    "HINDALCO.NS":  "Metal",
    "JSWSTEEL.NS":  "Metal",
    "SAIL.NS":      "Metal",
    # Telecom
    "BHARTIARTL.NS": "Telecom",
    # Conglomerate
    "ADANIENT.NS":   "Conglomerate",
    "ADANIPORTS.NS": "Conglomerate",
}

# ─────────────────────────────────────────────
# Risk Thresholds
# ─────────────────────────────────────────────
CONCENTRATION_RISK_THRESHOLD: float = 0.40      # >40% in one sector = high risk
MODERATE_CONCENTRATION_THRESHOLD: float = 0.25  # >25% = moderate risk
HIGH_VOLATILITY_THRESHOLD: float = 0.03         # >3% daily move = high volatility
SIGNIFICANT_LOSS_THRESHOLD: float = -0.02       # <-2% = significant daily loss

# ─────────────────────────────────────────────
# News Feed URLs (RSS)
# ─────────────────────────────────────────────
NEWS_RSS_FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint":       "https://www.livemint.com/rss/markets",
}

# ─────────────────────────────────────────────
# Agent Behaviour Constants
# ─────────────────────────────────────────────
MAX_NEWS_ITEMS: int = 10           # Max news items to process per run
MAX_REASONING_RETRIES: int = 2     # Retries if reasoning quality is low
MIN_CONFIDENCE_SCORE: float = 0.6  # Minimum acceptable confidence for insights

# ─────────────────────────────────────────────
# Validation — now checks GROQ_API_KEY
# ─────────────────────────────────────────────
def validate_config() -> list[str]:
    """Returns list of missing critical config keys."""
    missing = []
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("dummy") or GROQ_API_KEY == "":
        missing.append("GROQ_API_KEY")
    if ENABLE_OBSERVABILITY and (not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY):
        missing.append("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
    return missing