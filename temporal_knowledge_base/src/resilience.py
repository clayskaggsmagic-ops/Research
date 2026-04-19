"""CHRONOS resilience layer — retry, circuit breaker, and safe concurrency.

All external calls (LLM, search API, article fetch, embedding, database)
are wrapped with exponential-backoff retries and circuit breakers to
survive transient failures without losing pipeline progress.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
    OSError,
)

# Try to add library-specific exceptions (may not all be installed)
try:
    import httpx
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)
except ImportError:
    pass

try:
    import asyncpg
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, asyncpg.PostgresError, asyncpg.InterfaceError)
except ImportError:
    pass

try:
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, ServiceUnavailable
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, GoogleAPIError)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_async(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[BaseException], ...] | None = None,
):
    """Decorator: exponential backoff with jitter for async functions.

    Usage:
        @retry_async(max_retries=3)
        async def call_api():
            ...
    """
    exceptions = retryable_exceptions or RETRYABLE_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            f"[retry] {func.__name__} failed after {max_retries + 1} attempts: {exc}"
                        )
                        raise
                    # Exponential backoff
                    delay = backoff_base ** attempt
                    if jitter:
                        delay *= 0.8 + random.random() * 0.4  # ±20%
                    logger.warning(
                        f"[retry] {func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed ({type(exc).__name__}: {str(exc)[:80]}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore  # unreachable but makes mypy happy

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Trips open after `failure_threshold` consecutive failures.

    When open, calls fail immediately for `reset_timeout` seconds,
    then half-open (allows one call) to test recovery.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0
        self._state = "closed"  # closed | open | half_open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = "half_open"
                return False
            return True
        return False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"[circuit_breaker] OPEN after {self._consecutive_failures} consecutive failures. "
                f"Will retry in {self.reset_timeout}s."
            )


class CircuitBreakerOpen(Exception):
    """Raised when a circuit breaker is open."""
    pass


def circuit_breaker(failure_threshold: int = 5, reset_timeout: float = 60.0):
    """Decorator: circuit breaker pattern for async functions."""
    cb = CircuitBreaker(failure_threshold, reset_timeout)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cb.is_open:
                raise CircuitBreakerOpen(
                    f"{func.__name__} circuit breaker is OPEN — "
                    f"too many consecutive failures"
                )
            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as exc:
                cb.record_failure()
                raise

        wrapper._circuit_breaker = cb  # type: ignore
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Safe gather — partial failure tolerance
# ---------------------------------------------------------------------------

async def safe_gather(
    *coros,
    return_exceptions: bool = False,
    max_concurrency: int | None = None,
) -> list[Any]:
    """Like asyncio.gather but with optional concurrency limiting.

    Args:
        *coros: Coroutines to run concurrently.
        return_exceptions: If True, exceptions are returned as values instead of raised.
        max_concurrency: If set, uses a semaphore to limit concurrent tasks.

    Returns:
        List of results (or exceptions if return_exceptions=True).
        Failed tasks return None if return_exceptions is False.
    """
    if max_concurrency:
        sem = asyncio.Semaphore(max_concurrency)

        async def _limited(coro):
            async with sem:
                return await coro

        coros = tuple(_limited(c) for c in coros)

    results = await asyncio.gather(*coros, return_exceptions=True)

    if not return_exceptions:
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.warning(f"[safe_gather] Task {i} failed: {type(r).__name__}: {r}")
                processed.append(None)
            else:
                processed.append(r)
        return processed

    return list(results)


# ---------------------------------------------------------------------------
# Convenience: retry-wrapped LLM call
# ---------------------------------------------------------------------------

@retry_async(max_retries=3, backoff_base=2.0)
async def resilient_llm_call(llm, messages: list, **kwargs) -> Any:
    """Call an LLM with retry logic baked in."""
    return await llm.ainvoke(messages, **kwargs)
