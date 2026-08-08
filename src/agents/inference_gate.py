"""AgenticAITA Inference Gating Protocol (review.md §2.1).

Calls to the shared cognitive resource (the LLM) are serialised behind a mutex
and every entry/exit is recorded with a monotonic sequence number. This gives
two things AgenticAITA emphasises:

* **reproducibility** — an audit trail of *which* agent used the model, when,
  and with what budget effect;
* **determinism** — no two model calls interleave, so a replay of the log
  reconstructs the exact order of reasoning steps.
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional


class InferenceGate:
    """Mutex-serialised access to the LLM with an append-only audit log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._audit: list[dict] = []

    @contextmanager
    def reserve(self, agent_name: str, resource: str = "llm") -> Iterator[None]:
        """Acquire exclusive access to ``resource``; record the window."""
        seq = next(self._seq)
        acquired = False
        self._lock.acquire()
        try:
            t0 = time.monotonic()
            acquired = True
            yield
        finally:
            dt = time.monotonic() - t0 if acquired else 0.0
            self._audit.append(
                {
                    "seq": seq,
                    "agent": agent_name,
                    "resource": resource,
                    "started_at": time.time(),
                    "duration_s": round(dt, 6),
                }
            )
            self._lock.release()

    def audit_log(self) -> list[dict]:
        """Immutable copy of the audit trail (ordering = replay order)."""
        return list(self._audit)

    def replay(self, log: Optional[list[dict]] = None) -> list[str]:
        """Reconstruct the chronological agent sequence from an audit log."""
        entries = log if log is not None else self._audit
        return [f"{e['seq']}: {e['agent']}" for e in entries]

    def clear(self) -> None:
        self._audit.clear()
