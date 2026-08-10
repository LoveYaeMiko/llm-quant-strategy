"""Token-bucket rate limiter tests."""

from __future__ import annotations

import time

from src.config import Config
from src.data.schema.rate_limiter import RateLimiters, RateLimiter


def test_acquire_returns_float_timestamp():
    lim = RateLimiter(rate_per_minute=1000, burst=1000)
    t = lim.acquire()
    assert isinstance(t, float)


def test_bucket_drains_and_waits():
    # 600/min = 10 tokens/s; burst 1 means the second acquire must wait ~0.1s
    lim = RateLimiter(rate_per_minute=600, burst=1)
    lim.acquire(1)
    start = time.monotonic()
    lim.acquire(1)
    assert time.monotonic() - start >= 0.05


def test_context_manager_acquires():
    lim = RateLimiter(rate_per_minute=1000, burst=1000)
    with lim:
        pass  # entry acquired a token


def test_rate_limiters_configure_reads_config():
    cfg = Config({"data": {"rate_limits": {"baostock": 500}}})
    RateLimiters.configure(cfg)
    assert RateLimiters.baostock.rate_per_minute == 500
