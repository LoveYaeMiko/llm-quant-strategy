"""Thread-safe token-bucket rate limiter (blueprint §3) + named singletons.

AlphaFeed is paid (generous quota), so the defaults are comfortable rather than
throttling; Baostock is free and should be treated gently. Every vendor call
goes through one of the :class:`RateLimiters` instances.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ...config import Config


class RateLimiter:
    """Token bucket: ``rate_per_minute`` refills continuously, ``burst`` caps it."""

    def __init__(self, rate_per_minute: int, burst: Optional[int] = None) -> None:
        self.rate_per_minute = rate_per_minute
        self._rate = rate_per_minute / 60.0  # tokens per second
        self.burst = burst or rate_per_minute
        self._tokens = float(self.burst)
        self._last = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """Block until ``tokens`` are available; returns the time the call may start.

        Sleeps OUTSIDE the lock: a waiter should not stall every other caller on
        the bucket while it sleeps out its token deficit.
        """
        while True:
            with self._lock:
                now = time.time()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return now
                wait = (tokens - self._tokens) / self._rate
            time.sleep(wait)  # re-check after sleeping — tokens may have refilled

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        return None


class RateLimiters:
    """Named token buckets for each endpoint family."""

    alphafeed_daily = RateLimiter(300)        # single-symbol klines
    alphafeed_daily_batch = RateLimiter(120)  # batch klines (paid quota → generous)
    alphafeed_quote = RateLimiter(300)        # quotes / universe cross-check
    alphafeed_adjust = RateLimiter(120)       # ex_factors
    baostock = RateLimiter(120)               # free source — be gentle

    @classmethod
    def configure(cls, config: Optional[Config] = None) -> None:
        """Overwrite per-minute rates from ``data.rate_limits`` in the master config."""
        if config is None:
            return
        rates = config.get("data.rate_limits", {}) or {}
        for name, limiter in cls.__dict__.items():
            if isinstance(limiter, RateLimiter) and name in rates:
                limiter.rate_per_minute = int(rates[name])
                limiter._rate = limiter.rate_per_minute / 60.0
