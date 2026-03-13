# app/tooling/retry_rate_limit.py
from __future__ import annotations

import time
from typing import Any, Callable

ERROR_PREFIX = "ERROR: "


def is_error_result(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ERROR_PREFIX)


def retry(fn: Callable[[], Any], retries: int = 3, base_delay: float = 0.5):
    """
    Run a function with exponential backoff retries.

    Returns:
      - fn() result on success
      - "ERROR: <message>" string on final failure (keeps current behavior compatibility)
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                return f"{ERROR_PREFIX}{str(e)}"
            time.sleep(base_delay * (2 ** (attempt - 1)))


class RateLimiter:
    """
    Simple token-bucket rate limiter to throttle internal API/tool usage.

    Notes:
      - This is for outbound calls (e.g., weather/news providers), not for HTTP endpoint rate limiting.
      - Endpoint rate limiting should remain in SlowAPI (Limiter/get_remote_address).
    """

    def __init__(self, max_per_interval: int, interval_seconds: float):
        self.max_per_interval = max_per_interval
        self.interval_seconds = interval_seconds
        self.tokens = max_per_interval
        self.last_refill = time.time()

    def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        now = time.time()
        elapsed = now - self.last_refill

        intervals = int(elapsed // self.interval_seconds)
        if intervals > 0:
            self.tokens = min(
                self.max_per_interval,
                self.tokens + intervals * self.max_per_interval,
            )
            self.last_refill = now

        if self.tokens == 0:
            sleep_for = self.interval_seconds - (now - self.last_refill)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.tokens = self.max_per_interval
            self.last_refill = time.time()

        self.tokens -= 1
