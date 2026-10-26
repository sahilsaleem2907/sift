# SPDX-License-Identifier: AGPL-3.0-only
"""Process-wide adaptive concurrency limiter + retry loop for LLM requests.

Every LLM request (generator, repair, critic, agentic, holistic) goes through
``call_with_limits``, which bounds in-flight requests *per endpoint* and owns the
only retry layer. LiteLLM / the OpenAI SDK are configured with ``num_retries=0`` so
nothing sleeps while holding a slot: on a 429 the slot is released first, the limit
is lowered (AIMD), and the request backs off before re-queueing. This is what keeps
a large PR from tripping a provider's concurrent-request cap (e.g. Ollama Cloud's
"too many concurrent requests") and losing whole files.
"""
import asyncio
import logging
import random
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, TypeVar

import httpx
import litellm

from sift import config

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Consecutive successes before the limit grows back by one (additive increase).
_INCREASE_AFTER = 8
# Several in-flight requests usually 429 together; count that burst as one signal.
_DECREASE_COOLDOWN_SECONDS = 2.0
# Total wall-clock a single request may spend retrying before the error is raised.
_RETRY_BUDGET_SECONDS = 180.0
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_CAP_SECONDS = 30.0
_RETRY_AFTER_CAP_SECONDS = 60.0

# Indirection so tests can make backoff instant.
_sleep = asyncio.sleep


class AdaptiveLimiter:
    """Concurrency gate whose limit halves on rate limits and creeps back on success."""

    def __init__(self, key: str, max_limit: int) -> None:
        self.key = key
        self.max_limit = max(1, max_limit)
        self.limit = self.max_limit
        self.in_flight = 0
        self._waiters: Deque[asyncio.Future] = deque()
        self._successes = 0
        self._last_decrease = 0.0

    async def acquire(self) -> None:
        if self.in_flight < self.limit and not self._waiters:
            self.in_flight += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # Slot was granted just as we were cancelled; hand it on.
                self.release()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self.in_flight -= 1
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_flight < self.limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_flight += 1
                fut.set_result(None)

    def record_success(self) -> None:
        self._successes += 1
        if self.limit < self.max_limit and self._successes >= _INCREASE_AFTER:
            self._successes = 0
            self.limit += 1
            logger.info("[rate_limit] endpoint=%s limit %d→%d after sustained success",
                        self.key, self.limit - 1, self.limit)
            self._wake()

    def record_rate_limit(self) -> None:
        self._successes = 0
        now = time.monotonic()
        if now - self._last_decrease < _DECREASE_COOLDOWN_SECONDS:
            return
        self._last_decrease = now
        new_limit = max(1, self.limit // 2)
        if new_limit != self.limit:
            logger.info("[rate_limit] endpoint=%s limit %d→%d after 429",
                        self.key, self.limit, new_limit)
            self.limit = new_limit


_limiters: Dict[str, AdaptiveLimiter] = {}


def get_limiter(key: str) -> AdaptiveLimiter:
    limiter = _limiters.get(key)
    if limiter is None:
        limiter = AdaptiveLimiter(key, config.SIFT_LLM_MAX_CONCURRENCY)
        _limiters[key] = limiter
    return limiter


def reset_limiters() -> None:
    """Drop all limiter state (tests)."""
    _limiters.clear()


def endpoint_key(kwargs: Dict[str, Any]) -> str:
    """Limiter key for a LiteLLM kwargs dict: the API base, else the provider prefix."""
    base = kwargs.get("api_base")
    if base:
        return str(base).rstrip("/")
    model = str(kwargs.get("model") or "")
    return model.split("/", 1)[0] if "/" in model else model or "default"


def _status_code(exc: BaseException) -> Optional[int]:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, litellm.RateLimitError) or _status_code(exc) == 429:
        return True
    msg = str(exc).lower()
    return "too many concurrent requests" in msg or "rate limit" in msg


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        httpx.TransportError,
    )):
        return True
    code = _status_code(exc)
    return code is not None and code >= 500


def permanent_failure_reason(exc: BaseException) -> Optional[str]:
    """Short reason when the provider will reject this request again unchanged, else None.

    4xx responses other than 408/409/429 (bad request, auth, content policy, context length)
    are deterministic, so retrying the same file is a wasted call.
    """
    from sift.intelligence.budget import overflow_shrink_factor

    if overflow_shrink_factor(exc) is not None:
        return "prompt exceeds the model's context window"
    if _is_rate_limit(exc) or _is_transient(exc):
        return None
    code = _status_code(exc)
    if code is not None and 400 <= code < 500 and code not in (408, 409):
        return f"provider rejected the request ({code})"
    return None


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    headers: Any = getattr(exc, "litellm_response_headers", None)
    if not headers:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return min(float(value), _RETRY_AFTER_CAP_SECONDS) if value else None
    except (TypeError, ValueError, AttributeError):
        return None


def _backoff_seconds(attempt: int) -> float:
    base = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
    return base * random.uniform(0.5, 1.0)


async def call_with_limits(key: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run ``fn`` under the endpoint's limiter, retrying rate-limit and transient errors.

    Rate limits get ``SIFT_LLM_RATE_LIMIT_RETRIES`` attempts and transient errors
    (timeouts, connection errors, 5xx) ``SIFT_LLM_NUM_RETRIES``, both bounded by a
    total wall-clock budget. Any other exception propagates immediately.
    """
    limiter = get_limiter(key)
    deadline = time.monotonic() + _RETRY_BUDGET_SECONDS
    rate_limit_attempts = 0
    transient_attempts = 0
    while True:
        await limiter.acquire()
        try:
            result = await fn()
        except Exception as e:  # noqa: BLE001 — classified below, re-raised if not retryable
            exc: Optional[Exception] = e
        else:
            exc = None
        finally:
            limiter.release()
        if exc is None:
            limiter.record_success()
            return result

        if _is_rate_limit(exc):
            limiter.record_rate_limit()
            rate_limit_attempts += 1
            attempt, allowed, kind = rate_limit_attempts, config.SIFT_LLM_RATE_LIMIT_RETRIES, "rate limit"
        elif _is_transient(exc):
            transient_attempts += 1
            attempt, allowed, kind = transient_attempts, config.SIFT_LLM_NUM_RETRIES, "transient error"
        else:
            raise exc
        if attempt > allowed:
            raise exc
        delay = _retry_after_seconds(exc) or _backoff_seconds(attempt)
        if time.monotonic() + delay > deadline:
            raise exc
        logger.info(
            "[rate_limit] endpoint=%s %s (attempt %d/%d), retrying in %.1fs (limit=%d)",
            key, kind, attempt, allowed, delay, limiter.limit,
        )
        await _sleep(delay)
