"""Broker boundary — the one place a real order would ever leave this system.

Today only :class:`PaperBroker` is wired into the D track: it records simulated
fills in the ledger exactly like the live trader does. :class:`RealBroker` is the
future plug-in point for a broker adapter and it FAILS CLOSED: its constructor
calls :func:`src.deploy.assert_simulated_only`, so while
``deployment.mode = observe`` (and ``real_money_enabled = false``) it cannot even
be instantiated, let alone submit. That gives the deployment gate a real
execution point instead of a purely declarative one (2026-09-09 audit: the gate
had zero callers).

No broker SDK is imported here — the adapter implementation is deliberately
absent until the LIVE_READINESS checklist is green.
"""
from __future__ import annotations

from typing import Any, Protocol

from ..deploy import assert_simulated_only


class Broker(Protocol):
    """Minimal order interface the execution layer would use."""

    def submit(self, symbol: str, side: str, shares: float, price: float) -> dict[str, Any]:
        ...


class PaperBroker:
    """Simulated broker: never talks to a network, only records what happened.

    The live trader and the paper runner own the ledger writes, so this class is
    a thin, explicit marker of the boundary rather than a second code path.
    """

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.simulated = True

    def submit(self, symbol: str, side: str, shares: float, price: float) -> dict[str, Any]:
        return {
            "accepted": False,
            "simulated": True,
            "symbol": symbol,
            "side": side,
            "shares": float(shares),
            "price": float(price),
            "note": "paper boundary — the ledger is written by the caller",
        }


class RealBroker:
    """Real-money broker adapter placeholder — construction is gated.

    Raises :class:`src.deploy.RealMoneyNotEnabled` while the deployment channel is
    ``observe``. Wire the actual SDK only after ``docs/LIVE_READINESS.md`` A/B/C/D
    are fully checked and ``deployment.mode`` is switched to ``live``.
    """

    def __init__(self, cfg=None) -> None:
        assert_simulated_only(cfg, "RealBroker construction")
        raise NotImplementedError(
            "no broker adapter is implemented — see docs/LIVE_READINESS.md C1"
        )


__all__ = ["Broker", "PaperBroker", "RealBroker"]
