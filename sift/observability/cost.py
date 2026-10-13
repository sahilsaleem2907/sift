# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal per-review token/cost capture (opt-in via config.SIFT_COST_TRACKING).

When a review is wrapped in `track_review(...)`, every LiteLLM call records its token
usage into a run-scoped accumulator held in a ContextVar. A ContextVar (not a module
global) keeps totals correct across the async per-file fan-out — each review gets its
own accumulator and concurrent reviews never mix. Off by default with effectively zero
overhead: `record_response` is a no-op when no accumulator is active.

This is deliberately small (see plan "build minimal fresh"): raw token counts are the
ground truth we log; the USD figure is a convenience computed from a built-in price
table. Cache tokens on the summary line are the caching-is-live signal — a nonzero
`cache_read` means the ephemeral prefix is being reused.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# model-id substring -> (usd_per_input_token, usd_per_output_token). Substring, lowercased.
# Only the models we run on the bench; extend as needed.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0e-6, 25.0e-6),
    "claude-opus-4-7": (5.0e-6, 25.0e-6),
    "claude-opus-4-6": (5.0e-6, 25.0e-6),
    "claude-sonnet-4-6": (3.0e-6, 15.0e-6),
    "claude-haiku-4-5": (1.0e-6, 5.0e-6),
    "claude-haiku-4.5": (1.0e-6, 5.0e-6),
}
# Cache-read tokens bill at ~0.1x input; 1h-TTL cache writes at ~2x (5m writes ~1.25x).
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 2.0


def _price_for(model: str) -> Optional[tuple[float, float]]:
    m = (model or "").lower()
    for key, prices in _PRICES.items():
        if key in m:
            return prices
    return None


@dataclass
class _ModelUsage:
    prompt_tokens: int = 0        # total input reported by LiteLLM (incl. cache read/write)
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    calls: int = 0


@dataclass
class UsageAccumulator:
    """Per-review token/cost totals, keyed by resolved model id.

    Keying by model id gives the candidate-vs-critic split for free: the bench pairs
    Opus (primary/agentic) with Sonnet (critic/holistic), so each role lands in its own
    bucket. If one id is reused for both roles the buckets collapse into one.
    """

    by_model: dict[str, _ModelUsage] = field(default_factory=dict)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> None:
        mu = self.by_model.setdefault(model, _ModelUsage())
        mu.prompt_tokens += prompt_tokens
        mu.completion_tokens += completion_tokens
        mu.cache_read_tokens += cache_read_tokens
        mu.cache_write_tokens += cache_write_tokens
        mu.calls += 1
        prices = _price_for(model)
        if prices:
            price_in, price_out = prices
            # LiteLLM's prompt_tokens for Anthropic includes the cached + cache-creation
            # tokens; price those at their reduced/premium rates and the remainder at full.
            uncached = max(prompt_tokens - cache_read_tokens - cache_write_tokens, 0)
            mu.cost += (
                uncached * price_in
                + cache_read_tokens * price_in * _CACHE_READ_MULT
                + cache_write_tokens * price_in * _CACHE_WRITE_MULT
                + completion_tokens * price_out
            )

    def total_cost(self) -> float:
        return sum(mu.cost for mu in self.by_model.values())


_current: contextvars.ContextVar[Optional[UsageAccumulator]] = contextvars.ContextVar(
    "sift_usage_accumulator", default=None
)


def _extract_usage(response: Any) -> Optional[tuple[int, int, int, int]]:
    """Pull (prompt, completion, cache_read, cache_write) from a LiteLLM response, or None."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    # Anthropic cache-creation tokens (LiteLLM surfaces them on usage for Anthropic models).
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return prompt, completion, cache_read, cache_write


def record_response(model: str, response: Any) -> None:
    """Record one LLM response into the active accumulator. No-op when tracking is off."""
    acc = _current.get()
    if acc is None:
        return
    extracted = _extract_usage(response)
    if extracted is None:
        return
    prompt, completion, cache_read, cache_write = extracted
    acc.record(model, prompt, completion, cache_read, cache_write)


@contextlib.contextmanager
def track_review(label: str) -> Iterator[Optional[UsageAccumulator]]:
    """Scope a review for token/cost capture. Enabled only when config.SIFT_COST_TRACKING.

    On exit, emits one greppable summary line per review:
      [cost] review <label> | <model> in=.. out=.. cache_read=.. cache_write=.. calls=.. $x | ... | total $y
    """
    from sift import config

    if not getattr(config, "SIFT_COST_TRACKING", False):
        yield None
        return

    acc = UsageAccumulator()
    token = _current.set(acc)
    try:
        yield acc
    finally:
        _current.reset(token)
        parts = []
        for model, mu in acc.by_model.items():
            parts.append(
                f"{model} in={mu.prompt_tokens} out={mu.completion_tokens} "
                f"cache_read={mu.cache_read_tokens} cache_write={mu.cache_write_tokens} "
                f"calls={mu.calls} ${mu.cost:.4f}"
            )
        body = " | ".join(parts) if parts else "(no LLM calls)"
        logger.info("[cost] review %s | %s | total $%.4f", label, body, acc.total_cost())
