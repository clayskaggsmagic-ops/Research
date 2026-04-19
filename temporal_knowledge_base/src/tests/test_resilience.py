"""Tests for the resilience module — retry, circuit breaker, safe_gather."""

from __future__ import annotations

import asyncio

import pytest

from src.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    circuit_breaker,
    retry_async,
    safe_gather,
)


# ---------------------------------------------------------------------------
# retry_async tests
# ---------------------------------------------------------------------------

class TestRetryAsync:
    """Test the retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retries(self):
        """Function that succeeds on first call is not retried."""
        call_count = 0

        @retry_async(max_retries=3, backoff_base=0.1, retryable_exceptions=(ValueError,))
        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeeds()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        """Function that fails twice then succeeds is retried correctly."""
        call_count = 0

        @retry_async(max_retries=3, backoff_base=0.1, jitter=False, retryable_exceptions=(ValueError,))
        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally"

        result = await fails_twice()
        assert result == "finally"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self):
        """Function that always fails raises after max retries."""
        call_count = 0

        @retry_async(max_retries=2, backoff_base=0.1, jitter=False, retryable_exceptions=(RuntimeError,))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            await always_fails()
        assert call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        """Exception not in retryable_exceptions is raised immediately."""
        call_count = 0

        @retry_async(max_retries=3, backoff_base=0.1, retryable_exceptions=(ValueError,))
        async def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await raises_type_error()
        assert call_count == 1  # no retries


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Test the circuit breaker pattern via the decorator."""

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self):
        """Circuit stays closed when calls succeed."""

        @circuit_breaker(failure_threshold=3, reset_timeout=0.1)
        async def ok():
            return "yes"

        for _ in range(5):
            result = await ok()
            assert result == "yes"

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit opens after failure_threshold failures."""

        @circuit_breaker(failure_threshold=2, reset_timeout=0.1)
        async def fails():
            raise RuntimeError("boom")

        # First two failures trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await fails()

        # Third call should hit the open circuit
        with pytest.raises(CircuitBreakerOpen):
            await fails()

    @pytest.mark.asyncio
    async def test_recovers_after_timeout(self):
        """Circuit transitions to half-open and recovers after reset_timeout."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, reset_timeout=0.05)
        async def fails_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "recovered"

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await fails_then_ok()

        # Wait past reset_timeout, breaker should let the next call through
        await asyncio.sleep(0.1)

        result = await fails_then_ok()
        assert result == "recovered"

    def test_manual_state_tracking(self):
        """CircuitBreaker instance tracks failures/successes manually."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        assert cb.is_open is False

        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False  # not yet at threshold

        cb.record_failure()
        assert cb.is_open is True

        cb.record_success()
        assert cb.is_open is False


# ---------------------------------------------------------------------------
# safe_gather tests
# ---------------------------------------------------------------------------

class TestSafeGather:
    """Test safe_gather (partial failure tolerance via *coros star-args)."""

    @pytest.mark.asyncio
    async def test_all_succeed(self):
        """All tasks succeeding returns all results."""

        async def double(n):
            return n * 2

        results = await safe_gather(double(1), double(2), double(3))
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_partial_failure_returns_none(self):
        """Failed tasks return None, successful ones return values."""

        async def ok():
            return "yes"

        async def fail():
            raise ValueError("nope")

        results = await safe_gather(ok(), fail(), ok())
        assert results == ["yes", None, "yes"]

    @pytest.mark.asyncio
    async def test_return_exceptions_true(self):
        """return_exceptions=True returns the exceptions as values."""

        async def ok():
            return "yes"

        async def fail():
            raise ValueError("nope")

        results = await safe_gather(ok(), fail(), return_exceptions=True)
        assert results[0] == "yes"
        assert isinstance(results[1], ValueError)

    @pytest.mark.asyncio
    async def test_empty_input(self):
        results = await safe_gather()
        assert results == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """max_concurrency caps simultaneous task execution."""
        state = {"active": 0, "peak": 0}

        async def track():
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return True

        await safe_gather(*[track() for _ in range(10)], max_concurrency=3)
        assert state["peak"] <= 3
