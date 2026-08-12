"""Alpha core — the Phase 8 low-vol + low-turnover long-only book (layer 1).

Phase 10 layer 1 is the **only validated alpha direction**: a pool of 5
``Avg(Neg(Rank(TS_Std(Close, N))), Neg(Rank(TS_Mean(Volume, M))))`` formulas
(low realized volatility ∧ low turnover → expected excess return). The core
fuses them by date-wise z-scored equal weight — the same composite
:func:`~src.pool.combination_backtest` builds with ``weights="equal"`` — then
turns the composite into a **long-only** top-decile book:

    composite[d, s] = mean_f zscore(factor_f[d, s])
    book[d]        = top `long_pct` of the composite cross-section, equal weight,
                     gross-normalised to 1, per-name capped at `max_position_pct`

The long-only (not long-short) choice follows the PHASE10 blueprint's semantics:
positions are "持仓权重" (holdings), the overlays cut/hold positions (减仓 50%),
and the integration layer normalises by the *signed* sum — all of which only make
sense for a portfolio of positive weights earning the equity risk premium. The
book is recomputed daily from the slow (120/240-day) factor lookbacks, so daily
recomputation is a close approximation of the blueprint's 月度调仓.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _zscore(scores: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date (neutralises the market level)."""
    s = scores.astype(float)
    mean = s.groupby(level=0).transform("mean")
    std = s.groupby(level=0).transform("std").replace(0.0, np.nan)
    return (s - mean) / (std.fillna(1.0) + 1e-12)


def long_book_weights(
    scores_day: pd.Series,
    *,
    long_pct: float = 0.10,
    max_position_pct: float = 0.05,
) -> dict[str, float]:
    """Long-only top-decile book from one day's cross-sectional scores.

    Drops NaN scores, equal-weights the top ``long_pct`` names, normalises the
    gross to 1 and caps any over-cap name (mirroring ``PointInTimeBacktest``'s
    cap-after-normalisation rule). Returns ``{symbol: weight}``.

    ``scores_day`` may arrive as a ``groupby(level=0)`` group — pandas keeps
    the full ``(date, symbol)`` MultiIndex there — so a leading date level is
    dropped so the book keys are plain symbols.
    """
    s = scores_day.dropna()
    if isinstance(s.index, pd.MultiIndex):
        s = s.droplevel(0)
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="first")]
    if s.empty:
        return {}
    n_long = max(1, int(round(len(s) * long_pct)))
    top = s.sort_values(ascending=False).index[:n_long]
    w = pd.Series(0.0, index=s.index)
    w[top] = 1.0 / n_long
    g = w.sum()
    if g > 0:
        w = w / g
    cap = float(max_position_pct)
    over = w[w > cap]
    if not over.empty:
        scale = cap / over.max()
        w = w.copy()
        w[over.index] = over * scale
    return {sym: float(v) for sym, v in w.items() if v > 0}


class AlphaCore:
    """Equal-weight composite of the Phase 8 pool -> per-date long-only book."""

    def __init__(
        self,
        fctx,
        formulas: Iterable[str],
        *,
        long_pct: float = 0.10,
        max_position_pct: float = 0.05,
    ) -> None:
        self.formulas = list(formulas)
        self.long_pct = float(long_pct)
        self.max_position_pct = float(max_position_pct)
        from ..factors.code_generator import eval_expression

        zscores = [_zscore(eval_expression(f, fctx)) for f in self.formulas]
        self.composite: pd.Series = (
            sum(zscores) / len(zscores) if zscores else pd.Series(dtype=float)
        )
        # index -> date cross-section, built once so weights_on is a dict lookup
        self._by_date: dict[pd.Timestamp, dict[str, float]] = {}
        for d, day in self.composite.groupby(level=0):
            self._by_date[pd.Timestamp(d)] = long_book_weights(
                day, long_pct=self.long_pct, max_position_pct=self.max_position_pct
            )

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(self._by_date)

    def weights_on(self, date: str | pd.Timestamp, symbols: Optional[Iterable[str]] = None) -> dict[str, float]:
        """The alpha book for one trading date (empty dict when out of sample)."""
        w = self._by_date.get(pd.Timestamp(date), {})
        if symbols is None:
            return dict(w)
        keep = set(symbols)
        return {s: v for s, v in w.items() if s in keep}

    def weights_frame(self, dates: Iterable[str | pd.Timestamp]) -> pd.DataFrame:
        """date × symbol book-weight frame over ``dates`` (0.0 where no position)."""
        rows = {pd.Timestamp(d): self.weights_on(d) for d in dates}
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
