# 🤖 Autonomous Financial Advisor Agent

An intelligent, explainable AI financial advisor for Indian equity markets (NSE/BSE).  
The agent ingests market data (NIFTY 50, SENSEX), sector indices, financial news, and your portfolio — then uses **causal reasoning** to explain *why* your portfolio performed the way it did today.

---

## 🏗️ Architecture

```
autonomous-financial-advisor/
│
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py                  # All config & constants
│
├── data/
│   ├── sample_portfolio.json        # Demo portfolio
│   └── sample_news.json             # Demo news
│
├── src/
│   ├── ingestion/
│   │   ├── market_data.py           # Fetches NIFTY, SENSEX, stock prices (yfinance)
│   │   ├── news_fetcher.py          # RSS feeds + NewsAPI + sample fallback
│   │   └── portfolio_loader.py      # Loads & validates portfolio JSON
│   │
│   ├── analytics/
│   │   ├── market_analyzer.py       # Market sentiment (bullish/bearish/neutral)
│   │   ├── sector_analyzer.py       # Sector-level performance & rotation
│   │   └── portfolio_analyzer.py    # P&L, allocation, concentration risk
│   │
│   ├── reasoning/
│   │   ├── news_classifier.py       # LLM-based news classification
│   │   ├── causal_reasoner.py       # NEWS → SECTOR → STOCK → PORTFOLIO chain
│   │   └── conflict_resolver.py     # Explains news vs price divergences
│   │
│   ├── agent/
│   │   ├── financial_advisor.py     # Main orchestration agent
│   │   └── self_evaluator.py        # Agent evaluates its own output quality
│   │
│   ├── observability/
│   │   └── langfuse_tracker.py      # Langfuse tracing (prompts, latency, scores)
│   │
│   └── utils/
│       ├── prompts.py               # All LLM prompts (centralised)
│       └── helpers.py               # Formatting, logging, decorators
│
└── tests/
    ├── test_market_analyzer.py
    ├── test_portfolio_analyzer.py
    └── test_reasoning.py
```

---

## 🚀 Quick Start

### 1. Clone & Set Up

```bash
git clone <repo-url>
cd autonomous-financial-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx       # Required
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx             # Optional (observability)
LANGFUSE_SECRET_KEY=sk-lf-xxxxx             # Optional (observability)
NEWS_API_KEY=your_newsapi_key               # Optional (live news)
```

### 3. Run the Agent

```bash
# Run with sample portfolio (no setup required except API key)
python main.py

# Run with your own portfolio
python main.py --portfolio my_portfolio.json

# Run + Interactive Q&A mode
python main.py --interactive

# Skip self-evaluation (faster)
python main.py --no-eval

# More verbose logging
python main.py --log-level DEBUG
```

---

## 📊 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT PIPELINE                            │
│                                                             │
│  [1] Portfolio Load  ─→  JSON → Portfolio object           │
│  [2] Market Data     ─→  yfinance → NIFTY, SENSEX, stocks  │
│  [3] Analytics       ─→  Sentiment + Sector + P&L analysis │
│  [4] News            ─→  RSS/API + LLM Classification       │
│  [5] Reasoning       ─→  Causal Chain (3 LLM calls)        │
│      ├── Market Narrative                                   │
│      ├── Causal Chain Builder                               │
│      └── Portfolio Impact Generator                         │
│  [6] Conflict        ─→  Detect & explain divergences      │
│  [7] Self-Evaluate   ─→  Grade own output (A/B/C/D)        │
│  [8] Observability   ─→  Langfuse spans, scores, traces    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Portfolio JSON Format

```json
{
  "user": {
    "id": "user_001",
    "name": "Your Name",
    "risk_profile": "moderate",
    "investment_goal": "wealth_creation",
    "investment_horizon_years": 5
  },
  "portfolio": {
    "total_invested": 500000,
    "cash_balance": 25000,
    "holdings": [
      {
        "symbol": "HDFCBANK.NS",
        "name": "HDFC Bank Ltd",
        "quantity": 50,
        "avg_buy_price": 1580.0,
        "sector": "Banking",
        "asset_type": "equity"
      }
    ]
  }
}
```

**NSE Stock symbols** must include `.NS` suffix (e.g., `TCS.NS`, `INFY.NS`, `RELIANCE.NS`).

---

## 🔍 Sample Output

```
═══ AI Advisor Report ═══

📊 Executive Summary
  Portfolio declined ₹2,500 (-0.8%) today, driven by HDFC Bank and SBI 
  underperforming after RBI's hawkish monetary policy stance dampened banking 
  sector sentiment.

🌍 Market Overview
  Indian markets traded lower on 22nd April 2025 with NIFTY 50 down 0.6% and 
  Bank Nifty leading declines at -1.2%. The RBI's decision to maintain a 
  withdrawal of accommodation stance weighed heavily on rate-sensitive banking 
  stocks...

🔗 Causal Chain
  1. RBI kept repo rate with hawkish tone
     → Banking sector fell 1.2%
     Mechanism: Higher-for-longer rates compress NIMs for banks

  2. Banking sector decline
     → Portfolio lost ₹4,200 from banking holdings
     Mechanism: 40% portfolio exposure to Banking sector

✅ Positive Signals
  • TCS gained 0.9% on strong US tech demand
  • Reliance resilient on strong Jio subscriber numbers

⚠️ Negative Signals
  • Banking sector under pressure post-RBI
  • Auto sector weak on EV transition concerns

💡 Actionable Insights
  → Consider reducing Banking concentration from 40% to under 30%
  → TCS position is acting as a portfolio stabiliser — maintain allocation
  → Watch for RBI commentary in next MPC meeting for banking sector cues

📈 Confidence Score: 82% | Data Quality: high | Depth: deep
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_portfolio_analyzer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🔭 Observability with Langfuse

1. Sign up at [langfuse.com](https://langfuse.com)
2. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to `.env`
3. Set `ENABLE_OBSERVABILITY=true`

The agent tracks:
- **Traces**: Full end-to-end agent run per session
- **Spans**: Each pipeline step (data fetch, analysis, reasoning)
- **Generations**: Every LLM API call with prompt + response
- **Scores**: Confidence scores and self-evaluation grades

---

## ⚙️ Configuration

All settings are in `config/settings.py` and `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `ENABLE_OBSERVABILITY` | `true` | Langfuse tracking |
| `ENABLE_SELF_EVALUATION` | `true` | Self-evaluation step |
| `MAX_REASONING_RETRIES` | `2` | Retry reasoning if grade < B |
| `MIN_CONFIDENCE_SCORE` | `0.6` | Min acceptable confidence |
| `CONCENTRATION_RISK_THRESHOLD` | `0.40` | 40% = high concentration |

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `anthropic` | Claude AI for all reasoning steps |
| `yfinance` | Live Indian stock + index prices |
| `langfuse` | Observability and prompt tracing |
| `feedparser` | RSS news feed parsing |
| `rich` | Beautiful terminal output |
| `pandas` / `numpy` | Data processing |

---

## 🤝 Extending the Agent

**Add a new data source**: Add a new class in `src/ingestion/`  
**Add a new analysis**: Add a new module in `src/analytics/`  
**Modify reasoning prompts**: Edit `src/utils/prompts.py`  
**Add a new risk check**: Extend `PortfolioAnalyzer._detect_risks()`  

---

## 📄 License

MIT License — Free for personal and educational use.
# financial-advisor-agent
