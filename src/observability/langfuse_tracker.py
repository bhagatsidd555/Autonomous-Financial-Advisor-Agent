"""
src/observability/langfuse_tracker.py
=======================================
Integrates Langfuse for full observability of the Financial Advisor Agent.
Tracks all LLM prompts, responses, latency, token usage, confidence scores,
and system-level traces for debugging and quality monitoring.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

from config.settings import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    ENABLE_OBSERVABILITY,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Langfuse client (optional dependency)
# ─────────────────────────────────────────────
_langfuse_client = None

def _get_langfuse_client():
    """Lazy-initialize Langfuse client."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not ENABLE_OBSERVABILITY:
        return None

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning("Langfuse keys not configured; observability disabled")
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("Langfuse client initialized at %s", LANGFUSE_HOST)
        return _langfuse_client
    except ImportError:
        logger.warning("Langfuse library not installed; run: pip install langfuse")
        return None
    except Exception as e:
        logger.error("Failed to initialize Langfuse: %s", str(e))
        return None


# ─────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────
class ObservabilityTracker:
    """
    Wraps Langfuse tracing with a simple interface.
    Degrades gracefully when Langfuse is unavailable — the agent
    continues to function without observability.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.client = _get_langfuse_client()
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.current_trace = None
        self._local_log: list[dict] = []  # Always-on local log fallback
        logger.info("ObservabilityTracker ready (session=%s, langfuse=%s)",
                    self.session_id, self.client is not None)

    # ── Trace Management ─────────────────────
    def start_trace(
        self,
        name: str,
        user_id: str = None,
        metadata: dict = None,
    ) -> str:
        """Start a new Langfuse trace for a full agent run."""
        trace_id = str(uuid.uuid4())[:12]

        if self.client:
            try:
                self.current_trace = self.client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id or "anonymous",
                    session_id=self.session_id,
                    metadata=metadata or {},
                )
                logger.debug("Langfuse trace started: %s", trace_id)
            except Exception as e:
                logger.warning("Langfuse trace creation failed: %s", str(e))
                self.current_trace = None

        self._local_log.append({
            "type": "trace_start",
            "trace_id": trace_id,
            "name": name,
            "timestamp": time.time(),
        })
        return trace_id

    def end_trace(self, output: Any = None, level: str = "DEFAULT"):
        """End the current trace."""
        if self.current_trace:
            try:
                self.current_trace.update(output=str(output)[:500] if output else None, level=level)
            except Exception as e:
                logger.warning("Langfuse trace end failed: %s", str(e))

        self._local_log.append({
            "type": "trace_end",
            "timestamp": time.time(),
        })

    # ── Span Tracking ────────────────────────
    @contextmanager
    def span(self, name: str, input_data: Any = None, metadata: dict = None):
        """
        Context manager for tracking a sub-step (span) within a trace.

        Usage:
            with tracker.span("market_analysis") as span:
                result = analyze_market()
                span.output = result
        """
        start_time = time.time()
        span_obj = _SpanProxy(name=name)

        langfuse_span = None
        if self.client and self.current_trace:
            try:
                langfuse_span = self.current_trace.span(
                    name=name,
                    input=str(input_data)[:500] if input_data else None,
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.warning("Langfuse span creation failed: %s", str(e))

        try:
            yield span_obj
        finally:
            elapsed = time.time() - start_time

            if langfuse_span:
                try:
                    langfuse_span.end(output=str(span_obj.output)[:500] if span_obj.output else None)
                except Exception:
                    pass

            self._local_log.append({
                "type": "span",
                "name": name,
                "duration_s": round(elapsed, 3),
                "timestamp": start_time,
            })
            logger.debug("Span [%s] completed in %.2fs", name, elapsed)

    # ── LLM Call Tracking ────────────────────
    def track_llm_call(
        self,
        name: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
        metadata: dict = None,
    ):
        """Track a specific LLM API call with full prompt/response logging."""
        if self.client and self.current_trace:
            try:
                self.current_trace.generation(
                    name=name,
                    model=model,
                    model_parameters={"max_tokens": 1000},
                    input=[{"role": "user", "content": prompt[:2000]}],
                    output=response[:2000],
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                    },
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.warning("Langfuse generation tracking failed: %s", str(e))

        self._local_log.append({
            "type": "llm_call",
            "name": name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "timestamp": time.time(),
        })

    # ── Score / Metrics Tracking ─────────────
    def track_score(
        self,
        name: str,
        value: float,
        comment: str = None,
        trace_id: str = None,
    ):
        """Track a numeric quality score (e.g., confidence score)."""
        if self.client:
            try:
                self.client.score(
                    trace_id=trace_id or getattr(self.current_trace, "id", "unknown"),
                    name=name,
                    value=value,
                    comment=comment,
                )
            except Exception as e:
                logger.warning("Langfuse score tracking failed: %s", str(e))

        self._local_log.append({
            "type": "score",
            "name": name,
            "value": value,
            "comment": comment,
            "timestamp": time.time(),
        })
        logger.debug("Score tracked: %s = %.3f", name, value)

    def track_event(self, name: str, data: dict = None):
        """Track a custom event."""
        if self.client and self.current_trace:
            try:
                self.current_trace.event(
                    name=name,
                    metadata=data or {},
                )
            except Exception:
                pass

        self._local_log.append({
            "type": "event",
            "name": name,
            "data": data,
            "timestamp": time.time(),
        })

    # ── Local Log Access ─────────────────────
    def get_local_log(self) -> list[dict]:
        """Return the local in-memory log (always available regardless of Langfuse)."""
        return self._local_log

    def print_session_summary(self):
        """Print a summary of the current session's tracked calls."""
        spans = [e for e in self._local_log if e["type"] == "span"]
        llm_calls = [e for e in self._local_log if e["type"] == "llm_call"]
        scores = [e for e in self._local_log if e["type"] == "score"]

        print("\n═══ Session Observability Summary ═══")
        print(f"  Session ID   : {self.session_id}")
        print(f"  Spans        : {len(spans)}")
        print(f"  LLM Calls    : {len(llm_calls)}")
        print(f"  Scores Logged: {len(scores)}")

        total_ms = sum(e.get("duration_s", 0) for e in spans) * 1000
        print(f"  Total Time   : {total_ms:.0f}ms")

        if llm_calls:
            total_in = sum(e.get("input_tokens", 0) for e in llm_calls)
            total_out = sum(e.get("output_tokens", 0) for e in llm_calls)
            print(f"  Tokens Used  : {total_in} in / {total_out} out")

        if scores:
            print("  Quality Scores:")
            for s in scores:
                print(f"    • {s['name']}: {s['value']:.3f}")
        print("═" * 38)

    def flush(self):
        """Flush Langfuse client to ensure all events are sent."""
        if self.client:
            try:
                self.client.flush()
            except Exception:
                pass


class _SpanProxy:
    """Lightweight proxy to allow setting output on a span within a context manager."""
    def __init__(self, name: str):
        self.name = name
        self.output = None
