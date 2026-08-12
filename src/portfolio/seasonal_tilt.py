"""Seasonal tilt — PEAD reversal tactical overlay (layer 2, PHASE10 §3.3).

The 2022-2025 diagnosis (``cmd_pead --direction reversal``) showed A-share HS300
exhibits **negative** post-earnings drift: high-SUE names *underperform* after
the announcement. The tactical tilt therefore flips the classic PEAD bet, with
the amplitude sign-aware of the position direction (the alpha core is
market-neutral long-short since 2026-08-12):

* high SUE (> 80th percentile) → cut a **long**, add to a **short** (they revert down);
* low SUE (< 20th percentile) → add to a **long**, cut a **short** (they drift up).

Only positions whose |weight| ≥ ``min_weight`` are touched (the blueprint's
"持仓权重 > 3% 的股票"), and only during earnings-season months (1, 2, 4, 8, 10 —
the blueprint's whole-month simplification of "公告后 5 个交易日"). SUE comes
from the PIT :class:`~src.factors.pead.PEADFactor` snapshot, so no look-ahead:
only reports with ``pubDate <= as_of`` are visible.

**Coverage caveat**: the cached Baostock profit panel covers 2020-2025 (the
Phase 9.2 fetch scope), so the tilt is a no-op before 2020 — its marginal
contribution is measured on the 2020-2025 sub-period.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Earnings-season months (annual report 1-2月, Q1 4月, half-year 8月, Q3 10月).
DEFAULT_MONTHS = (1, 2, 4, 8, 10)


class PEADSeasonalTilt:
    """±``amplitude`` weight tilt on SUE extremes during earnings seasons.

    ``universe`` is the full cross-section the SUE percentile is measured over
    (the blueprint's "SUE 在全市场的分位" — ``get_all_sues`` then rank), so a
    held name's tilt reflects its *market* rank, not its rank among the book.
    """

    def __init__(
        self,
        pead,
        universe: list[str],
        *,
        amplitude: float = 0.20,
        min_weight: float = 0.015,
        months: tuple[int, ...] = DEFAULT_MONTHS,
    ) -> None:
        self.pead = pead
        self.universe = list(universe)
        self.amplitude = float(amplitude)
        self.min_weight = float(min_weight)
        self.months = tuple(months)

    def is_earnings_season(self, date: str | pd.Timestamp) -> bool:
        return pd.Timestamp(date).month in self.months

    def apply(self, weights: dict[str, float], date: str | pd.Timestamp) -> dict[str, float]:
        """Tilt the long-only book for ``date``; no-op outside earnings season."""
        if not self.is_earnings_season(date):
            return weights
        if not weights:
            return weights
        sues = self.pead.sue_snapshot(self.universe, date)
        if sues.empty:
            return weights
        pct = sues.rank(pct=True)  # cross-sectional SUE percentile over the universe
        out = dict(weights)
        for symbol, w in weights.items():
            if abs(w) < self.min_weight:
                continue
            p = pct.get(symbol, np.nan)
            if not np.isfinite(p):
                continue
            if p > 0.8 or p < 0.2:
                # sign-aware amplitude: the reversal bet is "high SUE reverts
                # down, low SUE drifts up", so it *adds* to a position whose
                # direction agrees with the bet (long low-SUE, short high-SUE)
                # and *cuts* one that disagrees.
                boost = (w > 0) == (p < 0.2)
                out[symbol] = w * (1.0 + self.amplitude if boost else 1.0 - self.amplitude)
        return out
