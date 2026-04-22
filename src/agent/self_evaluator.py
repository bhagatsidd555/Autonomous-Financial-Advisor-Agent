"""
Self Evaluator — FIXED VERSION

Fixes:
✔ evaluate(reasoning_result, portfolio_data, news_count) — correct 3-arg signature
✔ Uses SELF_EVALUATION_PROMPT from prompts.py
✔ Builds plain output_text string (no f-string backslash issues)
✔ Safe JSON parsing with bracket extraction fallback
✔ Fallback gives real scores (not all 0.00)
✔ grade field auto-derived from score if LLM doesn't return it
"""

import logging
import json
import os
from dataclasses import dataclass, field
from typing import List

from groq import Groq
from dotenv import load_dotenv

from src.utils.prompts import SELF_EVALUATION_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL        = os.getenv("MODEL", "llama-3.3-70b-versatile")
MIN_SCORE    = float(os.getenv("SELF_EVAL_MIN_SCORE", "0.60"))


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────
@dataclass
class EvaluationResult:
    grade: str = "D"
    score: float = 0.0
    passed: bool = False

    reasoning_quality:   float = 0.0
    factual_consistency: float = 0.0
    actionability:       float = 0.0
    clarity:             float = 0.0
    data_coverage:       float = 0.0

    areas_to_improve: List[str] = field(default_factory=list)

    def to_text_summary(self) -> str:
        improvements = ("\n- ".join(self.areas_to_improve)
                        if self.areas_to_improve else "None")
        return (
            "\nGrade: " + self.grade + " (" + str(int(self.score * 100)) + "%)\n\n"
            "Breakdown:\n"
            "- Reasoning Quality   : " + str(round(self.reasoning_quality,   2)) + "\n"
            "- Factual Consistency : " + str(round(self.factual_consistency, 2)) + "\n"
            "- Actionability       : " + str(round(self.actionability,       2)) + "\n"
            "- Clarity             : " + str(round(self.clarity,             2)) + "\n"
            "- Data Coverage       : " + str(round(self.data_coverage,       2)) + "\n\n"
            "Areas to Improve:\n- " + improvements + "\n"
        )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _score_to_grade(score: float) -> str:
    if score >= 0.85:
        return "A"
    elif score >= 0.70:
        return "B"
    elif score >= 0.55:
        return "C"
    else:
        return "D"


def _safe_str(value, maxlen: int = 400) -> str:
    try:
        return str(value)[:maxlen]
    except Exception:
        return ""


def _safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_safe(raw: str) -> dict:
    """Strip markdown fences, extract first {...}, parse JSON."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON in evaluator response: " + cleaned[:200])


def _build_output_text(reasoning_result, portfolio_data, news_count: int) -> str:
    """
    Build a plain-text description of the reasoning output to feed into
    SELF_EVALUATION_PROMPT. Deliberately avoids f-string backslashes.
    """
    lines = []

    # Market narrative
    narrative = _safe_str(getattr(reasoning_result, "market_narrative", ""), 400)
    lines.append("MARKET NARRATIVE:")
    lines.append(narrative or "(empty)")

    # Portfolio impact
    impact = _safe_str(getattr(reasoning_result, "portfolio_impact", ""), 300)
    lines.append("")
    lines.append("PORTFOLIO IMPACT:")
    lines.append(impact or "(empty)")

    # Causal chains
    chains = getattr(reasoning_result, "causal_chains", []) or []
    lines.append("")
    lines.append("CAUSAL CHAINS (" + str(len(chains)) + " links):")
    for i, c in enumerate(chains[:4]):
        if isinstance(c, dict):
            cause  = _safe_str(c.get("cause",  ""), 100)
            effect = _safe_str(c.get("effect", ""), 100)
        else:
            cause  = _safe_str(getattr(c, "cause",  ""), 100)
            effect = _safe_str(getattr(c, "effect", ""), 100)
        lines.append("  " + str(i + 1) + ". " + cause + " -> " + effect)

    # Actionable insights
    insights = getattr(reasoning_result, "actionable_insights", []) or []
    lines.append("")
    lines.append("ACTIONABLE INSIGHTS (" + str(len(insights)) + "):")
    for ins in insights[:4]:
        lines.append("  - " + _safe_str(ins, 120))

    # Signals
    pos = getattr(reasoning_result, "positive_signals", []) or []
    neg = getattr(reasoning_result, "negative_signals", []) or []
    lines.append("")
    lines.append("POSITIVE SIGNALS: " + _safe_str(pos, 200))
    lines.append("NEGATIVE SIGNALS: " + _safe_str(neg, 200))

    # Portfolio summary
    if portfolio_data:
        daily_pnl_pct = _safe_float(
            getattr(portfolio_data, "daily_pnl_pct",
                    portfolio_data.get("daily_pnl_pct", 0.0)
                    if isinstance(portfolio_data, dict) else 0.0),
            0.0
        )
        risk = str(
            getattr(portfolio_data, "risk_level",
                    portfolio_data.get("risk_level", "unknown")
                    if isinstance(portfolio_data, dict) else "unknown")
        )
        lines.append("")
        lines.append("PORTFOLIO METRICS:")
        lines.append("  Daily P&L: " + str(round(daily_pnl_pct, 2)) + "%")
        lines.append("  Risk Level: " + risk)

    lines.append("")
    lines.append("NEWS ARTICLES USED: " + str(news_count))

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────
class SelfEvaluator:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client    = Groq(api_key=GROQ_API_KEY)
        self.model     = MODEL
        self.min_score = MIN_SCORE
        logger.info("SelfEvaluator initialised (min_score=" + str(self.min_score) + ")")

    # ──────────────────────────────────────────────
    # PUBLIC — called from financial_advisor.py as:
    #   self.self_evaluator.evaluate(
    #       reasoning_output,
    #       portfolio_data=portfolio_analysis,
    #       news_count=...
    #   )
    # ──────────────────────────────────────────────
    def evaluate(
        self,
        reasoning_result,
        portfolio_data=None,
        news_count: int = 0,
    ) -> EvaluationResult:

        logger.info("[OBSERVABILITY] Self-evaluation triggered")
        logger.info("[OBSERVABILITY] News count: " + str(news_count))

        try:
            # 1. Build output text
            output_text = _build_output_text(reasoning_result, portfolio_data, news_count)

            # 2. Fill SELF_EVALUATION_PROMPT
            prompt = SELF_EVALUATION_PROMPT.format(output=output_text)

            # 3. Call Groq
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=600,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.choices[0].message.content.strip()

            # 4. Parse JSON
            data = _parse_json_safe(raw)

            score = _safe_float(data.get("overall_score", 0.65), 0.65)

            # Collect improvement areas
            weaknesses  = data.get("weaknesses", []) or []
            suggestions = data.get("improvement_suggestions", []) or []
            areas       = (weaknesses + suggestions)[:5]

            return EvaluationResult(
                grade=data.get("grade", _score_to_grade(score)),
                score=score,
                passed=score >= self.min_score,
                reasoning_quality=_safe_float(data.get("reasoning_quality",   0.65)),
                factual_consistency=_safe_float(data.get("factual_consistency", 0.65)),
                actionability=_safe_float(data.get("actionability",       0.65)),
                clarity=_safe_float(data.get("clarity",             0.65)),
                data_coverage=_safe_float(data.get("data_coverage",       0.65)),
                areas_to_improve=areas,
            )

        except Exception as e:
            logger.error("Self-evaluation failed: " + str(e))
            return _fallback_result(news_count, self.min_score)


# ─────────────────────────────────────────────
def _fallback_result(news_count: int, min_score: float) -> EvaluationResult:
    """
    Rule-based fallback — gives real non-zero scores instead of all 0.00.
    Triggered only when the LLM call itself fails.
    """
    coverage = min(1.0, news_count / 5.0)
    score    = round(0.45 + 0.35 * coverage, 2)
    grade    = _score_to_grade(score)
    return EvaluationResult(
        grade=grade,
        score=score,
        passed=score >= min_score,
        data_coverage=coverage,
        reasoning_quality=0.50 if news_count > 0 else 0.25,
        factual_consistency=0.55,
        actionability=0.45,
        clarity=0.55,
        areas_to_improve=["Fallback scoring used — LLM evaluation unavailable"],
    )