"""Three-layer fusion engine (PHASE10 §3.4).

Orchestrates the layers in fixed priority order:

    1. :class:`~src.portfolio.alpha_core.AlphaCore`   — the base long-only book
    2. :class:`~src.portfolio.seasonal_tilt.PEADSeasonalTilt`  — ±20% SUE tilt
    3. :class:`~src.portfolio.risk_overlay.SentimentRiskOverlay` — 50% risk cut

Because the risk layer runs **last**, its position cuts override any tilt applied
to the same name (priority: risk > tilt > alpha, PHASE10 §2.1). After the layers
the book is re-normalised to gross 1 by the *signed* sum — meaningful because the
alpha book is long-only (all weights positive).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class ThreeLayerPortfolio:
    """Alpha → tilt → risk fusion; ``compute_weights`` yields the final book."""

    def __init__(self, alpha, tilt=None, risk=None) -> None:
        self.alpha = alpha
        self.tilt = tilt
        self.risk = risk

    def compute_weights(self, symbols, date: str | pd.Timestamp) -> dict[str, float]:
        """Layered book for one date (alpha → tilt → risk → normalise)."""
        weights = self.alpha.weights_on(date, symbols=symbols)
        if self.tilt is not None:
            weights = self.tilt.apply(weights, date)
        if self.risk is not None:
            weights = self.risk.apply(weights, date)
        return self._normalize(weights)

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        """Gross-normalise to 1 — the invariant of the (now market-neutral) book.

        With the alpha core long the top decile and short the bottom decile, the
        signed sum is ≈ 0, so dividing by it is meaningless; dividing by the gross
        (sum |w|) restores the target leverage after the overlays change it. For a
        pure long-only book this reduces to the old signed-sum normalisation.
        """
        total = sum(abs(v) for v in weights.values())
        if total <= 0:
            return weights
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------ sweep
    def weights_frame(
        self,
        symbols,
        dates,
    ) -> dict[str, pd.DataFrame]:
        """Run the fused book over every date and return the weight frame."""
        import numpy as np

        rows = {}
        for d in dates:
            t = pd.Timestamp(d)
            w = self.compute_weights(symbols, t)
            rows[t] = pd.Series(w, dtype=float)
        frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
        # from_dict does not guarantee a chronological row order — sort so any
        # downstream cumprod/cummax (drawdown) is computed in time order.
        return frame.sort_index()
