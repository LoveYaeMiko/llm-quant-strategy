"""Exponential-backoff retry for flaky scraping feeds (AKShare web scraping).

AKShare hits EastMoney's website; its endpoints break on site changes and are
slow. Every AKShare call goes through :func:`retry_call` so a transient failure
degrades to a graceful empty result instead of killing the whole ingestion.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


def retry_call(
    fn: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    on_error: Optional[Callable[[Exception], Any]] = None,
) -> Any:
    """Call ``fn`` with exponential backoff on exception.

    On the final failure, return ``on_error(exc)`` if provided (use a lambda that
    returns an empty ``pd.DataFrame()``), else re-raise.
    """
    last: Optional[Exception] = None
    delay = base_delay
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — scraping feeds raise anything
            last = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= backoff
    assert last is not None
    if on_error is not None:
        return on_error(last)
    raise last
