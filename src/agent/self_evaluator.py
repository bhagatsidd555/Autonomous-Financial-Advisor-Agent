"""
Self Evaluator — FINAL VERSION (FIXED)

✔ Groq based
✔ No f-string errors
✔ Observability added
✔ Safe parsing
✔ Clean output
"""

import logging
import json
import os
from dataclasses import dataclass

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
MIN_SCORE = float(os.getenv("SELF_EVAL_MIN_SCORE", 0.60))


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────
@dataclass
class EvaluationResult:
    grade: str = "D"
    score: float = 0.0
    passed: bool = False

    reasoning_quality: float = 0.0
    factual_consistency: float = 0.0
    actionability: float = 0.0
    clarity: float = 0.0
    data_coverage: float = 0.0

    areas_to_improve: list = None

    def __post_init__(self):
        if self.areas_to_improve is None:
            self.areas_to_improve = []

    def to_text_summary(self):
        improvements = "\n- ".join(self.areas_to_improve) if self.areas_to_improve else "None"

        return f"""
Grade: {self.grade} ({self.score*100:.0f}%)

Breakdown:
- Reasoning Quality   : {self.reasoning_quality:.2f}
- Factual Consistency : {self.factual_consistency:.2f}
- Actionability       : {self.actionability:.2f}
- Clarity             : {self.clarity:.2f}
- Data Coverage       : {self.data_coverage:.2f}

Areas to Improve:
- {improvements}
"""


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────
class SelfEvaluator:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODEL
        self.min_score = MIN_SCORE

        logger.info(f"SelfEvaluator initialised (min_score={self.min_score:.2f})")

    def evaluate(self, reasoning_result, portfolio_data=None, news_count=0):

        logger.info("[OBSERVABILITY] Self-evaluation triggered")
        logger.info(f"[OBSERVABILITY] News count: {news_count}")

        try:
            prompt = f"""
Evaluate this financial reasoning output.

Market narrative: {reasoning_result.market_narrative[:200]}
Portfolio impact: {reasoning_result.portfolio_impact[:200]}
Causal chains: {len(reasoning_result.causal_chains)}
Insights: {reasoning_result.actionable_insights[:3]}
News used: {news_count}

Return JSON:
{{
 "reasoning_quality": 0-1,
 "factual_consistency": 0-1,
 "actionability": 0-1,
 "clarity": 0-1,
 "data_coverage": 0-1,
 "overall_score": 0-1,
 "grade": "A/B/C/D",
 "areas_to_improve": []
}}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(raw)
            except:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                data = json.loads(raw[start:end])

            score = float(data.get("overall_score", 0.5))

            return EvaluationResult(
                grade=data.get("grade", "C"),
                score=score,
                passed=score >= self.min_score,
                reasoning_quality=float(data.get("reasoning_quality", 0.5)),
                factual_consistency=float(data.get("factual_consistency", 0.5)),
                actionability=float(data.get("actionability", 0.5)),
                clarity=float(data.get("clarity", 0.5)),
                data_coverage=float(data.get("data_coverage", 0.5)),
                areas_to_improve=data.get("areas_to_improve", []),
            )

        except Exception as e:
            logger.error(f"Self-evaluation failed: {e}")

            # fallback scoring
            coverage = min(1.0, news_count / 5)
            score = 0.6 * coverage

            grade = "A" if score >= 0.8 else "B" if score >= 0.6 else "C"

            return EvaluationResult(
                grade=grade,
                score=round(score, 2),
                passed=score >= self.min_score,
                data_coverage=coverage,
                areas_to_improve=["Fallback scoring used"],
            )