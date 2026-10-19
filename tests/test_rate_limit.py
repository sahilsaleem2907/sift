"""Adaptive per-endpoint LLM limiter + retry loop (sift/intelligence/rate_limit.py)."""
import asyncio
from typing import List
from unittest import mock

import httpx
import litellm
import pytest

from sift import config
from sift.intelligence import rate_limit
from sift.intelligence.rate_limit import call_with_limits, endpoint_key, get_limiter


@pytest.fixture(autouse=True)
def _fresh_limiters():
    rate_limit.reset_limiters()
    with mock.patch.multiple(
        config,
        SIFT_LLM_MAX_CONCURRENCY=4,
        SIFT_LLM_RATE_LIMIT_RETRIES=6,
        SIFT_LLM_NUM_RETRIES=3,
    ):
        yield
    rate_limit.reset_limiters()


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    """Make backoff instant; record requested delays and in-flight count while sleeping."""
    recorded: List[float] = []

    async def _fake_sleep(delay: float) -> None:
        recorded.append(delay)
        assert get_limiter("ep").in_flight == 0, "slot must be released before backing off"

    monkeypatch.setattr(rate_limit, "_sleep", _fake_sleep)
    return recorded


def _rate_limit_error(retry_after: str = "") -> litellm.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x"))
    return litellm.RateLimitError(
        "too many concurrent requests", llm_provider="openai", model="m", response=response
    )


def _flaky(failures: List[BaseException], result: str = "ok"):
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return result

    return fn, calls


@pytest.mark.asyncio
async def test_in_flight_never_exceeds_limit():
    peak = {"now": 0, "max": 0}

    async def fn():
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        await asyncio.sleep(0.01)
        peak["now"] -= 1
        return 1

    with mock.patch.object(config, "SIFT_LLM_MAX_CONCURRENCY", 3):
        results = await asyncio.gather(*[call_with_limits("ep", fn) for _ in range(12)])
    assert results == [1] * 12
    assert peak["max"] == 3


@pytest.mark.asyncio
async def test_rate_limit_halves_limit_and_retries_outside_slot(sleeps):
    fn, calls = _flaky([_rate_limit_error()])
    assert await call_with_limits("ep", fn) == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1
    assert get_limiter("ep").limit == 2


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured(sleeps):
    fn, _ = _flaky([_rate_limit_error(retry_after="7")])
    await call_with_limits("ep", fn)
    assert sleeps == [7.0]


@pytest.mark.asyncio
async def test_rate_limit_retry_budget_exhausted_raises(sleeps):
    fn, calls = _flaky([_rate_limit_error() for _ in range(10)])
    with mock.patch.object(config, "SIFT_LLM_RATE_LIMIT_RETRIES", 2):
        with pytest.raises(litellm.RateLimitError):
            await call_with_limits("ep", fn)
    assert calls["n"] == 3
    assert get_limiter("ep").in_flight == 0


@pytest.mark.asyncio
async def test_transient_errors_use_num_retries(sleeps):
    fn, calls = _flaky([httpx.ConnectTimeout("boom") for _ in range(10)])
    with pytest.raises(httpx.ConnectTimeout):
        await call_with_limits("ep", fn)
    assert calls["n"] == 1 + config.SIFT_LLM_NUM_RETRIES
    # Transient errors don't shrink the limit; only rate limits do.
    assert get_limiter("ep").limit == 4


@pytest.mark.asyncio
async def test_non_retryable_error_propagates_immediately(sleeps):
    fn, calls = _flaky([ValueError("bad request")])
    with pytest.raises(ValueError):
        await call_with_limits("ep", fn)
    assert calls["n"] == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_limit_recovers_after_sustained_success(sleeps):
    fn, _ = _flaky([_rate_limit_error()])
    await call_with_limits("ep", fn)
    assert get_limiter("ep").limit == 2
    ok, _ = _flaky([])
    for _ in range(rate_limit._INCREASE_AFTER):
        await call_with_limits("ep", ok)
    assert get_limiter("ep").limit == 3


def test_burst_of_429s_counts_as_one_decrease():
    limiter = get_limiter("ep")
    limiter.record_rate_limit()
    limiter.record_rate_limit()
    limiter.record_rate_limit()
    assert limiter.limit == 2


def test_endpoint_key_prefers_api_base_then_provider():
    assert endpoint_key({"model": "openai/x", "api_base": "https://ollama.com/v1/"}) == "https://ollama.com/v1"
    assert endpoint_key({"model": "groq/openai/gpt-oss-120b", "api_base": None}) == "groq"
    assert endpoint_key({"model": "gpt-4o"}) == "gpt-4o"
