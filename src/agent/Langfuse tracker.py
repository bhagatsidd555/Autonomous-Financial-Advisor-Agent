"""
ObservabilityTracker — simple local file logging (no Langfuse needed)
Replaces Langfuse with a lightweight JSON log file so the rest of
the codebase works without any paid service.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ENABLE_OBSERVABILITY = os.getenv("ENABLE_OBSERVABILITY", "false").lower() == "true"
LOG_DIR = Path("logs")


class ObservabilityTracker:
    """
    Drop-in replacement for the Langfuse tracker.
    Writes structured JSON logs to ./logs/sessions.jsonl
    """

    def __init__(self):
        self.session_id   = uuid.uuid4().hex[:8]
        self.spans: list  = []
        self.scores: list = []
        self.llm_calls    = 0
        self.start_time   = time.time()
        self.enabled      = ENABLE_OBSERVABILITY

        if self.enabled:
            LOG_DIR.mkdir(exist_ok=True)
            logger.info(f"ObservabilityTracker ready (session={self.session_id}, local_logging=True)")
        else:
            logger.info(f"ObservabilityTracker ready (session={self.session_id}, observability=disabled)")

    # ── span helpers (Langfuse-compatible API) ───────

    def start_span(self, name: str, metadata: dict | None = None) -> str:
        span_id = uuid.uuid4().hex[:8]
        span = {
            "id": span_id,
            "name": name,
            "start": time.time(),
            "metadata": metadata or {},
        }
        self.spans.append(span)
        return span_id

    def end_span(self, span_id: str, output: str | None = None) -> None:
        for span in self.spans:
            if span["id"] == span_id and "end" not in span:
                span["end"]      = time.time()
                span["duration"] = round((span["end"] - span["start"]) * 1000, 1)
                span["output"]   = output or ""
                break

    # ── LLM call logging ─────────────────────────────

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str = "",
        latency_ms: float = 0,
        metadata: dict | None = None,
    ) -> None:
        self.llm_calls += 1
        if not self.enabled:
            return
        entry = {
            "type": "llm_call",
            "session": self.session_id,
            "ts": datetime.utcnow().isoformat(),
            "model": model,
            "latency_ms": latency_ms,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "metadata": metadata or {},
        }
        self._write(entry)

    # ── score logging ─────────────────────────────────

    def log_score(self, name: str, value: float, comment: str = "") -> None:
        self.scores.append({"name": name, "value": value})
        if not self.enabled:
            return
        entry = {
            "type": "score",
            "session": self.session_id,
            "ts": datetime.utcnow().isoformat(),
            "name": name,
            "value": value,
            "comment": comment,
        }
        self._write(entry)

    # ── session summary ───────────────────────────────

    def print_summary(self) -> None:
        elapsed = round((time.time() - self.start_time) * 1000)
        print("\n" + "═" * 38)
        print("  Session Observability Summary")
        print("═" * 38)
        print(f"  Session ID  : {self.session_id}")
        print(f"  Spans       : {len(self.spans)}")
        print(f"  LLM Calls   : {self.llm_calls}")
        print(f"  Scores      : {len(self.scores)}")
        print(f"  Total Time  : {elapsed}ms")
        if self.scores:
            print("  Quality Scores:")
            for s in self.scores:
                print(f"    • {s['name']}: {s['value']:.3f}")
        if self.enabled:
            print(f"  Log file    : {LOG_DIR}/sessions.jsonl")
        print("═" * 38 + "\n")

    # ── Langfuse compatibility stubs ──────────────────
    # (so existing code that calls these doesn't break)

    def flush(self) -> None:
        pass  # no-op

    def track_agent_run(self, *args, **kwargs) -> str:
        return self.start_span("agent_run", kwargs.get("metadata"))

    def end_agent_run(self, trace_id: str, *args, **kwargs) -> None:
        self.end_span(trace_id)

    # ── internal ──────────────────────────────────────

    def _write(self, entry: dict) -> None:
        try:
            with open(LOG_DIR / "sessions.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Log write failed: {e}")