"""Latency, token usage and approximate cost per ticket.

* LangChain calls (planner, re-ranking, answer, judge, graph planning, translations) are
  counted by a callback that is active for everything run inside `track_usage()` - also
  in worker threads started with asyncio.to_thread, because the context is copied.
* The AutoGen agents use their own OpenAI client; its usage (`total_usage()`) is added by
  `agents.run_agent_team` just before the client is closed.
* Cost = tokens x the list price per 1M tokens in MODEL_PRICES_PER_1M (USD, approximate;
  edit the table if your prices differ). Embedding calls are not counted: at
  text-embedding-3-small prices they cost a tiny fraction of a cent per ticket.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

# USD per 1M tokens (input, output). Approximate public list prices; update as needed.
MODEL_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}


def price_for(model: str) -> tuple[float, float] | None:
    """Longest matching prefix, so 'gpt-4o-mini-2024-07-18' uses the gpt-4o-mini price."""
    model = (model or "").lower()
    for name in sorted(MODEL_PRICES_PER_1M, key=len, reverse=True):
        if model.startswith(name):
            return MODEL_PRICES_PER_1M[name]
    return None


@dataclass(eq=False)   # identity equality/hash: LangChain may keep handlers in sets
class UsageTracker(BaseCallbackHandler):
    """Collects token usage per model for one ticket (thread-safe)."""

    calls: int = 0
    by_model: dict[str, list[int]] = field(default_factory=dict)   # model -> [input, output]
    started: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, input_tokens: int, output_tokens: int, calls: int = 1) -> None:
        with self._lock:
            self.calls += calls
            row = self.by_model.setdefault(model or "unknown", [0, 0])
            row[0] += int(input_tokens or 0)
            row[1] += int(output_tokens or 0)

    # LangChain callback: one call per LLM response
    def on_llm_end(self, response, **kwargs: Any) -> None:  # noqa: ANN001
        model = (response.llm_output or {}).get("model_name", "") if response.llm_output else ""
        inp = out = 0
        found = False
        for gens in response.generations or []:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg is not None else None
                if usage:
                    inp += usage.get("input_tokens", 0)
                    out += usage.get("output_tokens", 0)
                    found = True
                    meta = getattr(msg, "response_metadata", {}) or {}
                    model = model or meta.get("model_name", "")
        if not found and response.llm_output:
            tu = response.llm_output.get("token_usage") or {}
            inp, out = tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)
        self.add(model, inp, out)

    def summary(self, latency_s: float | None = None) -> dict:
        with self._lock:
            tokens_in = sum(v[0] for v in self.by_model.values())
            tokens_out = sum(v[1] for v in self.by_model.values())
            cost, priced = 0.0, True
            for model, (i, o) in self.by_model.items():
                p = price_for(model)
                if p is None:
                    priced = False
                    continue
                cost += i / 1e6 * p[0] + o / 1e6 * p[1]
            return {
                "latency_s": round(latency_s if latency_s is not None else time.perf_counter() - self.started, 2),
                "llm_calls": self.calls,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost, 5),
                "cost_complete": priced,
                "by_model": {m: {"in": v[0], "out": v[1]} for m, v in self.by_model.items()},
            }


_tracker: ContextVar[UsageTracker | None] = ContextVar("support_usage_tracker", default=None)
register_configure_hook(_tracker, inheritable=True)   # LangChain adds it to every run's callbacks


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """Count every LangChain LLM call made inside the block (and in threads started from it)."""
    tracker = UsageTracker()
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


def current_tracker() -> UsageTracker | None:
    return _tracker.get()


def format_metrics(m: dict) -> str:
    """Short human-readable line, e.g. '18.4 s · 12 model calls · 9.8k tokens · ≈ $0.0031'."""
    if not m:
        return ""
    tokens = m.get("tokens_in", 0) + m.get("tokens_out", 0)
    tok = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
    cost = f" · ≈ ${m.get('cost_usd', 0):.4f}" if m.get("cost_complete", True) else ""
    return f"{m.get('latency_s', 0):.1f} s · {m.get('llm_calls', 0)} model calls · {tok} tokens{cost}"
