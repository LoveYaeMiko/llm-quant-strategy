"""PEAD factor — post-earnings announcement drift (validation_BLUEPRINT §4.x / PHASE9).

PEAD is the most robust fundamental anomaly: stocks that beat the consensus
drift upward for weeks after the announcement; misses drift down. The classic
signal is the **standardized unexpected earnings** — here a seasonal
same-quarter-a-year-ago comparison, per the PHASE9 blueprint's
``(actual - expected) / |expected|``:

    SUE = (eps_cum(Q) - eps_cum(Q-1y)) / |eps_cum(Q-1y)|

where ``eps_cum`` is the *cumulative* EPS of the reporting period
(netProfit / totalShare, matching the statement convention). Using cumulative
EPS directly is deliberate: the difference against the same quarter one year
earlier cancels the year-to-date accumulation, so no single-quarter
differencing is needed and the seasonal baseline handles Q1-vs-Q4 scale.

**PIT guarantee**: only reports whose ``pubDate <= as_of`` are visible, and a
signal older than ``signal_expiry_days`` (60) is dropped — an announcement
cannot trade before it exists, and its edge decays.

This factor is **cross-sectional data-driven** — it is NOT an
``eval_expression`` formula, so it cannot flow through the string-based
``--factor-pool`` combination path. ``score_panel`` emits the aligned
``(date, symbol)`` signal series the single-factor harness and the
``PointInTimeBacktest`` consume directly.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

DATE_OFFSET_1Y = pd.DateOffset(years=1)


class PEADFactor:
    """Cross-sectional PEAD signal from a PIT quarterly-profit panel."""

    def __init__(
        self,
        panel: pd.DataFrame,
        signal_expiry_days: int = 60,
        min_eps_history: int = 8,
    ) -> None:
        self.expiry_days = int(signal_expiry_days)
        self.min_history = int(min_eps_history)
        self._per_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]] = {}
        for symbol, g in panel.groupby("symbol"):
            g = g.dropna(subset=["eps_cum"]).sort_values("pubDate")
            if g.empty:
                continue
            smap = g.drop_duplicates("statDate", keep="last").set_index("statDate")["eps_cum"]
            self._per_symbol[symbol] = (
                g["pubDate"].to_numpy(dtype="datetime64[ns]"),
                g["eps_cum"].to_numpy(dtype=float),
                g["statDate"].to_numpy(dtype="datetime64[ns]"),
                smap,
            )

    @property
    def symbols(self) -> list[str]:
        return list(self._per_symbol.keys())

    def sue(self, symbol: str, as_of: str | pd.Timestamp) -> tuple[float, Optional[pd.Timestamp]]:
        """Standardized unexpected earnings for ``symbol`` as of ``as_of``.

        Returns ``(sue, pub_date)``, or ``(nan, None)`` when there is no signal:
        no report yet, the latest announcement is older than ``expiry_days``,
        fewer than ``min_history`` reports exist, or the prior-year same-quarter
        baseline is missing / zero. PIT: ``pubDate <= as_of`` enforced inside.
        """
        entry = self._per_symbol.get(symbol)
        if entry is None:
            return float("nan"), None
        pubs, eps, stats, smap = entry
        t = pd.Timestamp(as_of)
        i = int(np.searchsorted(pubs, t.to_datetime64(), side="right")) - 1
        if i < 0:
            return float("nan"), None
        if (t - pd.Timestamp(pubs[i])).days > self.expiry_days:
            return float("nan"), None
        if i + 1 < self.min_history:
            return float("nan"), None
        base_stat = pd.Timestamp(stats[i]) - DATE_OFFSET_1Y
        base_eps = smap.get(base_stat)
        if base_eps is None or not np.isfinite(base_eps) or base_eps == 0:
            return float("nan"), None
        sue = (float(eps[i]) - float(base_eps)) / abs(float(base_eps))
        return sue, pd.Timestamp(pubs[i])

    def sue_snapshot(self, symbols: Iterable[str], as_of: str | pd.Timestamp) -> pd.Series:
        """Cross-sectional SUE for ``symbols`` at one ``as_of`` date."""
        vals: dict[str, float] = {}
        for s in symbols:
            sue, _ = self.sue(s, as_of)
            vals[s] = sue
        return pd.Series(vals)

    def score_panel(
        self,
        dates: Iterable[str | pd.Timestamp],
        symbols: Iterable[str],
    ) -> pd.Series:
        """Full ``(date, symbol)`` signal panel: percentile rank of SUE per date.

        Symbols with no valid recent SUE get NaN (excluded from the long/short
        book on that date). The returned series is indexed like the backtest's
        ``(date, symbol)`` forward-return panel.
        """
        syms = list(symbols)
        records: list[tuple[pd.Timestamp, str, float]] = []
        for d in dates:
            t = pd.Timestamp(d)
            r = self.sue_snapshot(syms, t).rank(pct=True)
            for s, v in r.items():
                records.append((t, s, float(v)))
        if not records:
            return pd.Series(dtype=float)
        idx = pd.MultiIndex.from_tuples([(a, b) for a, b, _ in records], names=["date", "symbol"])
        return pd.Series([v for _, _, v in records], index=idx)
