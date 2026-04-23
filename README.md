# 🤖 Autonomous Financial Advisor Agent

> An AI-powered financial advisor that doesn't just report data — it **reasons** through it.
> Causal chain: **Market News → Sector Trends → Stock Movements → Portfolio Impact**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Sample Output](#sample-output)
8. [How the Reasoning Works](#how-the-reasoning-works)
9. [Observability](#observability)
10. [Running Tests](#running-tests)py
11. [Design Decisions](#design-decisions)

---

## Overview

The **Autonomous Financial Advisor Agent** is a multi-phase reasoning pipeline built for Indian equity markets (NSE/BSE). Given a user portfolio, it:

1. **Fetches live market data** — NIFTY 50, SENSEX, 9 sectoral indices
2. **Ingests real financial news** — from Economic Times, MoneyControl, Mint RSS feeds
3. **Classifies news** by sentiment (positive/negative/neutral) and scope (market-wide / sector-specific / stock-specific)
4. **Analyzes portfolio** — daily P&L, sector allocation, concentration risk
5. **Reasons causally** — links macro news → sector movement → stock performance → portfolio impact
6. **Detects conflicts** — explains when positive news co-exists with falling prices
7. **Self-evaluates** — scores its own reasoning quality (Reasoning Quality, Factual Consistency, Actionability, Clarity, Data Coverage)


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Autonomous Financial Advisor                    │
│                    (financial_advisor.py)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  INGESTION   │   │  ANALYTICS   │   │    REASONING     │
│              │   │              │   │                  │
│ market_data  │   │ market_      │   │ news_classifier  │
│ news_fetcher │   │ analyzer     │   │ causal_reasoner  │
│ portfolio_   │   │ sector_      │   │ conflict_        │
│ loader       │   │ analyzer     │   │ resolver         │
│              │   │ portfolio_   │   │                  │
└──────────────┘   │ analyzer     │   └──────────────────┘
                   └──────────────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │  AGENT LAYER     │
                 │                  │
                 │ self_evaluator   │
                 │ (Grade A–D)      │
                 └──────────────────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │  OBSERVABILITY   │
                 │  langfuse_       │
                 │  tracker         │
                 └──────────────────┘
```

### Causal Reasoning Pipeline

```
Market Event (News)
       │
       ▼
Sector Impact (NIFTY Bank -2.3%)
       │
       ▼
Stock Movement (HDFCBANK -1.8%, SBIN -1.5%)
       │
       ▼
Portfolio Impact (Your banking holdings lost ₹8,400 today)
       │
       ▼
Personalized Insight + Confidence Score
```

---

## Features

| Phase | Feature | Status |
|-------|---------|--------|
| **Phase 1** | NIFTY 50 + SENSEX sentiment analysis | ✅ |
| **Phase 1** | 9 sectoral indices (Bank, IT, Pharma, Auto, FMCG, Metal, Realty, Energy) | ✅ |
| **Phase 1** | News classification: sentiment + scope | ✅ |
| **Phase 2** | Daily P&L (absolute + percentage) | ✅ |
| **Phase 2** | Asset allocation by sector | ✅ |
| **Phase 2** | Concentration risk detection (>25% moderate, >40% high) | ✅ |
| **Phase 3** | Causal chain reasoning (macro → sector → stock → portfolio) | ✅ |
| **Phase 3** | Conflict resolution (positive news + falling price) | ✅ |
| **Phase 3** | High-impact signal prioritization | ✅ |
| **Phase 4** | Langfuse observability integration | ✅ |
| **Phase 4** | LLM-based self-evaluation with letter grade | ✅ |

---

## Project Structure

```
autonomous-financial-advisor/
│
├── main.py                          # Entry point — run this
├── requirements.txt
├── .env.example                     # Copy to .env and fill keys
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # All config, API keys, thresholds
│
├── data/
│   ├── sample_portfolio.json        # Demo portfolio (Rahul Sharma, 10 holdings)
│   └── sample_news.json             # Fallback news when RSS unavailable
│
├── src/
│   ├── ingestion/                   # Phase 1: Data ingestion
│   │   ├── market_data.py           # yfinance → NIFTY/SENSEX/sectoral indices
│   │   ├── news_fetcher.py          # RSS feeds → raw news articles
│   │   └── portfolio_loader.py      # JSON → Portfolio dataclass
│   │
│   ├── analytics/                   # Phase 2: Analytics engine
│   │   ├── market_analyzer.py       # Sentiment: bullish/bearish/neutral
│   │   ├── sector_analyzer.py       # Leaders/laggards, sector P&L
│   │   └── portfolio_analyzer.py    # P&L, allocation, risk alerts
│   │
│   ├── reasoning/                   # Phase 3: Agent intelligence
│   │   ├── news_classifier.py       # LLM: sentiment + scope classification
│   │   ├── causal_reasoner.py       # LLM: causal chain + portfolio narrative
│   │   └── conflict_resolver.py     # Detects/explains conflicting signals
│   │
│   ├── agent/                       # Phase 4: Orchestration + evaluation
│   │   ├── financial_advisor.py     # Master orchestrator (AutonomousFinancialAdvisor)
│   │   ├── self_evaluator.py        # LLM-based quality scoring (Grade A–D)
│   │   └── langfuse_tracker.py      # Observability: prompt/response tracing
│   │
│   ├── observability/
│   │   └── langfuse_tracker.py      # Langfuse integration (optional)
│   │
│   └── utils/
│       ├── helpers.py               # Rich terminal formatting, INR formatting
│       └── prompts.py               # All LLM prompts in one place
│
└── tests/
    ├── test_market_analyzer.py
    ├── test_portfolio_analyzer.py
    └── test_reasoning.py
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A **Groq API key** (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repository

```bash
git clone https://github.com/bhagatsidd555/financial-advisor-agent.git
cd financial-advisor-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```env
# Required
GROQ_API_KEY=gsk_your_groq_key_here

# Optional — for Langfuse observability dashboard
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
ENABLE_OBSERVABILITY=true

# Optional — enable/disable self-evaluation
ENABLE_SELF_EVALUATION=true
```

### 5. Run the agent

```bash
python main.py
```

The agent will:
- Fetch live NSE/BSE data (takes ~5–8 seconds)
- Pull latest news from RSS feeds
- Analyze the sample portfolio
- Print a full advisory report with grade

### 6. Try different portfolios

Edit `data/sample_portfolio.json` or point to another file:

```python
# In main.py
advisor = AutonomousFinancialAdvisor(portfolio_path="data/sector_heavy_portfolio.json")
```

---

## Configuration

All configuration lives in `config/settings.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key for LLaMA 3.3 |
| `ENABLE_SELF_EVALUATION` | `true` | LLM-based quality scoring after each run |
| `ENABLE_OBSERVABILITY` | `false` | Enable Langfuse tracing |
| `CONCENTRATION_RISK_THRESHOLD` | `0.40` | >40% sector = high concentration alert |
| `MODERATE_CONCENTRATION_THRESHOLD` | `0.25` | >25% sector = moderate alert |
| `MAX_NEWS_ITEMS` | `10` | News articles to process per run |
| `MIN_CONFIDENCE_SCORE` | `0.60` | Minimum passing score for self-evaluation |

### Supported Market Indices

| Key | Symbol | Index |
|-----|--------|-------|
| `NIFTY_50` | `^NSEI` | NIFTY 50 |
| `SENSEX` | `^BSESN` | BSE SENSEX |
| `NIFTY_BANK` | `^NSEBANK` | NIFTY Bank |
| `NIFTY_IT` | `^CNXIT` | NIFTY IT |
| `NIFTY_PHARMA` | `^CNXPHARMA` | NIFTY Pharma |
| `NIFTY_AUTO` | `^CNXAUTO` | NIFTY Auto |
| `NIFTY_FMCG` | `^CNXFMCG` | NIFTY FMCG |
| `NIFTY_METAL` | `^CNXMETAL` | NIFTY Metal |
| `NIFTY_REALTY` | `^CNXREALTY` | NIFTY Realty |
| `NIFTY_ENERGY` | `^CNXENERGY` | NIFTY Energy |

---

## Sample Output

```
╭───────────────────────────╮
│ Financial Advisor Running │
╰───────────────────────────╯

── Holdings ──
╭────────────────┬─────┬───────────┬───────────┬───────────────┬───────────────────┬───────────╮
│ Symbol         │ Qty │   Avg Buy │       CMP │ Current Value │    Unrealised P&L │ Daily P&L │
├────────────────┼─────┼───────────┼───────────┼───────────────┼───────────────────┼───────────┤
│ HDFCBANK.NS    │  50 │  ₹1,580.0 │    ₹799.9 │       ₹39,995 │ -₹39,005 (-49.4%) │     -₹592 │
│ TCS.NS         │  20 │  ₹3,750.0 │  ₹2,538.5 │       ₹50,770 │ -₹24,230 (-32.3%) │   -₹1,440 │
│ SBIN.NS        │ 100 │    ₹620.0 │  ₹1,103.3 │       ₹1.10L  │ +₹48,330 (+78.0%) │     -₹855 │
│ ...            │ ... │       ... │       ... │           ... │               ... │       ... │
╰────────────────┴─────┴───────────┴───────────┴───────────────┴───────────────────┴───────────╯

── Portfolio Summary ──
  Invested:     ₹5.87L
  Current:      ₹5.59L
  Today's P&L:  -₹6,441 (-1.19%)

── Risk Alerts ──
⚠  Moderate concentration in Banking (28.1%). Monitor and consider rebalancing.
⚠  State Bank of India represents 20.7% of portfolio.

── AI Advisor Report ──

📊 Market Narrative:
Rahul, the IT sector's underperformance — led by HCL Tech's declining growth premium —
significantly impacted your portfolio today. US macro concerns triggered defensive rotation
into FMCG, while your Banking exposure added pressure from RBI rate uncertainty.

🔗 Key Drivers:
- HCL Tech's growth premium is vanishing → IT sector → HCL Tech, Infosys affected
- FII outflows of ₹3,842 cr → broad market → Banking + IT heaviest hit
- Defensive rotation to FMCG → Nestle India, ITC partially offset losses

📉 Portfolio Impact:
Your portfolio declined ₹6,441 (-1.19%) today. IT holdings (TCS, Infosys) drove
-₹3,224 of the loss. Banking concentration in SBIN contributed -₹855.
Sun Pharma (+₹115) and Tata Steel (+₹82) partially cushioned the decline.

💡 Insights:
- Consider reducing HCL Tech exposure — growth premium declining, AI pivot unproven
- Rebalance Banking below 25% — SBIN at 20.7% is near concentration threshold
- Add FMCG/Pharma names to benefit from defensive rotation in volatile markets

Confidence: 80%

── Evaluation ──
Grade: B (80%)
- Reasoning Quality   : 0.90
- Factual Consistency : 0.90
- Actionability       : 0.80
- Clarity             : 0.80
- Data Coverage       : 0.70

── Run Stats ──
  Time: 11.66s  |  Confidence: 80%
```

---

## How the Reasoning Works

### Causal Chain (Phase 3 Core Logic)

The `CausalReasoner` builds explicit cause-effect links:

```
Trigger: "HCL Tech flags weak Q4, bets on AI"
    │
    ├─► Sector: IT sector underperforms (-1.2%)
    │       │
    │       └─► Stock: TCS -0.9%, Infosys -1.4% (peer impact)
    │               │
    │               └─► Portfolio: IT holdings lost ₹3,224 today
    │
    └─► Macro: US tech spending slowdown confirmed
            │
            └─► FII: Selling in tech-heavy indices
                    │
                    └─► Market: NIFTY IT index -1.0%
```

### Conflict Resolution

When positive news coincides with falling prices, the `ConflictResolver` explains it:

> *"Sensex/Nifty gain headlines are from the previous session — today's data shows a reversal. This is likely profit-booking after the 3-day rally, amplified by FII outflows. The positive news has been priced in."*

### Self-Evaluation Rubric

The `SelfEvaluator` uses `SELF_EVALUATION_PROMPT` to grade its own output:

| Dimension | What it checks |
|-----------|---------------|
| **Reasoning Quality** | Are cause-effect links logically sound? |
| **Factual Consistency** | No internal contradictions? Numbers match? |
| **Actionability** | Are recommendations stock-specific, not generic? |
| **Clarity** | Plain language, no unexplained jargon? |
| **Data Coverage** | Were all key signals (news + market + portfolio) used? |

---

## Observability

### Langfuse Integration (Optional)

Set `ENABLE_OBSERVABILITY=true` in `.env` and add Langfuse keys.

Tracks:
- Every LLM prompt and response
- Token usage per call
- End-to-end latency
- Confidence score per run

View your traces at [cloud.langfuse.com](https://cloud.langfuse.com)

### Structured Logging

Every component logs with consistent format:

```
15:54:15 | INFO  | src.analytics.market_analyzer  | Market analysis complete: bearish (moderate) | NIFTY: -0.81%
15:54:18 | INFO  | src.reasoning.causal_reasoner  | [OBSERVABILITY] Portfolio PnL: -1.19%
15:54:20 | INFO  | src.agent.self_evaluator        | [OBSERVABILITY] Self-evaluation triggered
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_portfolio_analyzer.py -v
pytest tests/test_market_analyzer.py -v
pytest tests/test_reasoning.py -v
```

---

## Design Decisions

### Why Groq + LLaMA 3.3?
- **Speed**: Groq hardware delivers ~500 tok/s — critical for <15s end-to-end latency
- **Cost**: Free tier sufficient for development and demo
- **Quality**: LLaMA 3.3 70B matches GPT-4o for structured financial reasoning tasks

### Why RSS feeds over NewsAPI?
- Zero cost, no API key required for basic runs
- ET, MoneyControl, Mint cover 95% of relevant Indian market news
- NewsAPI key optional for broader coverage

### Why separate prompts in `utils/prompts.py`?
- Single place to tune/version all prompts
- Easy A/B testing of prompt variants
- Prevents prompt drift across modules

### Latency Optimization
- Market data fetched in parallel (yfinance batch calls)
- News classification done in single batch LLM call (not per-article)
- Self-evaluation runs only when `ENABLE_SELF_EVALUATION=true`
- Result: ~10–13s end-to-end on typical run

### Graceful Degradation
- If RSS feed fails → falls back to `data/sample_news.json`
- If yfinance 404 → skips that index, continues with rest
- If LLM call fails → rule-based fallback with real non-zero scores
- If portfolio enrichment fails → uses avg_buy_price as proxy

---

## Requirements

```
groq>=0.9.0
yfinance>=0.2.44
feedparser>=6.0.11
python-dotenv>=1.0.1
rich>=13.7.0
langfuse>=2.25.0       # optional, for observability
pytest>=8.3.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Environment Variables Reference

```env
# ── Required ──────────────────────────────────
GROQ_API_KEY=gsk_...               # Groq API key (free at console.groq.com)

# ── Optional: Observability ───────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
ENABLE_OBSERVABILITY=false          # Set true to enable Langfuse tracing

# ── Optional: Feature Flags ───────────────────
ENABLE_SELF_EVALUATION=true         # LLM-based quality grading
ENABLE_CACHE=true                   # Cache market data for 5 min

# ── Optional: Tuning ──────────────────────────
MAX_TOKENS=4096
TEMPERATURE=0.3
LOG_LEVEL=INFO
```

---

## Evaluation Rubric Coverage

| Criteria | Weight | Implementation |
|----------|--------|---------------|
| **Reasoning Quality** | 35% | `CausalReasoner` builds 3–5 explicit causal links per run. `ConflictResolver` handles ambiguous signals. |
| **Code Design** | 20% | Fully modular (ingestion / analytics / reasoning / agent). Type hints throughout. Dataclasses for all data models. |
| **Observability** | 15% | Langfuse integration in `langfuse_tracker.py`. Structured logging with `[OBSERVABILITY]` tags on every key event. |
| **Edge Case Handling** | 15% | Missing index data → skip gracefully. Positive news + down market → explained. Fallback news when RSS fails. |
| **Evaluation Layer** | 15% | `SelfEvaluator` uses LLM to grade Reasoning Quality, Factual Consistency, Actionability, Clarity, Data Coverage (0–1 each). Letter grade A–D. |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improve-reasoning`)
3. Commit changes (`git commit -m "Improve causal chain depth"`)
4. Push (`git push origin feature/improve-reasoning`)
5. Open a Pull Request

---


**Siddheshwar Bhagat**
GitHub: [@bhagatsidd555](https://github.com/bhagatsidd555)

---

*Built for the Backend Engineering Challenge: Autonomous Financial Advisor Agent*