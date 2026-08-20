"""Control-scaling wrapper — apply the kill-switch gross multiplier to a book.

The three-layer portfolio gross-normalises to 1.0; this wrapper multiplies that
book by the live control-state multiplier so a ``de_risk`` decision halves the
gross exposure and a ``halt`` decision flattens it. It is the single integration
point between the autopilot state and the execution layer — the runner sees only
``compute_weights`` and needs no knowledge of the kill-switch.
"""

from __future__ import annotations

from typing import Callable, Iterable


class ControlScaledPortfolio:
    """Wrap a weight source and scale its book by a live gross multiplier.

    ``scale_getter`` returns the current multiplier (1.0 normal, 0.5 de-risk,
    0.0 halt). When it returns ``<= 0`` the book goes **flat** — every symbol in
    ``symbols`` is mapped to ``0.0`` so the executor *sells* existing positions
    rather than merely skipping the rebalance (which would freeze the book at
    its last weights).
    """

    def __init__(self, inner, scale_getter: Callable[[], float]) -> None:
        self.inner = inner
        self.scale_getter = scale_getter

    def compute_weights(self, symbols: Iterable[str], date) -> dict[str, float]:
        symbols = list(symbols)
        scale = float(self.scale_getter() or 0.0)
        if scale <= 0.0:
            return {s: 0.0 for s in symbols}
        weights = self.inner.compute_weights(symbols, date)
        if scale == 1.0:
            return weights
        return {k: v * scale for k, v in weights.items()}


__all__ = ["ControlScaledPortfolio"]
