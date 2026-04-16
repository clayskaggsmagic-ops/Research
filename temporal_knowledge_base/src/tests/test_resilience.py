"""Tests for the resilience module — retry, circuit breaker, safe_gather."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
    safe_gather,
)


# ---------------------------------------------------------------------------
# retry_with_backoff tests
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:
    """Test the retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retries(self):
        """Function that succeeds on first call is not retried."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
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

        @retry_with_backoff(max_retries=3, base_delay=0.01)
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

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            await always_fails()
        assert call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_backoff_timing(self):
        """Verify exponential backoff is applied (basic timing check)."""
        call_count = 0
        start = time.monotonic()

        @retry_with_backoff(max_retries=2, base_delay=0.05, max_delay=1.0)
        async def fails_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry me")
            return "done"

        await fails_then_ok()
        elapsed = time.monotonic() - start
        # Should have waited ~0.05s for the retry
        assert elapsed >= 0.04, f"Expected at least 40ms delay, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Test the circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self):
        """Circuit stays closed when calls succeed."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        async def ok():
            return "yes"

        for _ in range(5):
            result = await cb.call(ok)
            assert result == "yes"

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit opens after failure_threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def fails():
            raise RuntimeError("boom")

        # First two failures — circuit stays closed during attempts
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fails)

        # Third call should hit the open circuit
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(fails)

    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """Circuit recovers to half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        call_count = 0

        async def fails_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "recovered"

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fails_then_ok)

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Should try half-open and succeed
        result = await cb.call(fails_then_ok)
        assert result == "recovered"


# ---------------------------------------------------------------------------
# safe_gather tests
# ---------------------------------------------------------------------------

class TestSafeGather:
    """Test safe_gather (partial failure tolerance)."""

    @pytest.mark.asyncio
    async def test_all_succeed(self):
        """All tasks succeeding returns all results."""

        async def ok(n):
            return n * 2

        results = await safe_gather([ok(1), ok(2), ok(3)])
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_partial_failure_returns_none(self):
        """Failed tasks return None, successful ones return values."""

        async def ok():
            return "yes"

        async def fail():
            raise ValueError("nope")

        results = await safe_gather([ok(), fail(), ok()])
        assert results == ["yes", None, "yes"]

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """All tasks failing returns all None."""

        async def fail():
            raise ValueError("nope")

        results = await safe_gather([fail(), fail()])
        assert results == [None, None]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        results = await safe_gather([])
        assert results == []
