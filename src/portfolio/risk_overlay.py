"""Risk overlay — text-sentiment circuit breaker (layer 3, PHASE10 §3.2).

The highest-priority layer: when a held name's *latest* research-report sentiment
collapses to an extreme negative z-score (Z < ``zscore_threshold``, the
blueprint's 历史 5% 分位 ≈ −2.5), the position is **cut 50% for ``freeze_days``
trading days** — and because this layer runs last, its cut overrides any tilt the
seasonal layer applied to that name (priority: risk > tilt > alpha).

The sentiment time series is the PIT carry-forward panel built by
:func:`~src.sentiment.triagent.build_report_signal` over the cached TriAgent
report scores — a symbol only sees reports with ``report_date <= as_of``, so no
look-ahead. The z-score contrasts the last-3-day mean against the trailing
252-day distribution:

    z = (mean(sent[-3d, d]) - mean(sent[d-252, d))) / std(sent[d-252, d))

``min_trigger_samples`` guards a degenerate std (too few history points → no
trigger). **Coverage caveat**: report sentiment only exists 2022-2025, so the
overlay is a no-op before 2022; its drawdown-reduction is measured there.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class SentimentRiskOverlay:
    """Halve any position whose recent sentiment z-score is extreme-negative."""

    def __init__(
        self,
        sentiment_panel: pd.Series,
        *,
        zscore_threshold: float = -2.5,
        position_cut: float = 0.50,
        freeze_days: int = 5,
        min_trigger_samples: int = 20,
        hist_window: int = 252,
        recent_window: int = 3,
    ) -> None:
        self.panel = sentiment_panel  # (date, symbol) [0,1] Series (PIT, NaN outside coverage)
        self.zscore_threshold = float(zscore_threshold)
        self.position_cut = float(position_cut)
        self.freeze_days = int(freeze_days)
        self.min_trigger_samples = int(min_trigger_samples)
        self.hist_window = int(hist_window)
        self.recent_window = int(recent_window)
        self._freeze_remaining: dict[str, int] = {}  # symbol -> trading days still cut
        self._triggers: list[dict] = []  # audit log
        self._series_cache: dict[str, pd.Series] = {}  # per-symbol (memoised xs)

    # -------------------------------------------------------------- per-name
    def _series(self, symbol: str) -> pd.Series:
        """PIT sentiment series for one symbol, indexed by trading date (cached)."""
        s = self._series_cache.get(symbol)
        if s is None:
            s = self.panel.xs(symbol, level="symbol").dropna()
            self._series_cache[symbol] = s
        return s

    def zscore(self, symbol: str, date: str | pd.Timestamp) -> Optional[float]:
        """Recent-vs-history sentiment z-score as of ``date`` (None = insufficient)."""
        t = pd.Timestamp(date)
        series = self._series(symbol)
        if series.empty:
            return None
        hist = series[(series.index > t - pd.Timedelta(days=self.hist_window)) & (series.index <= t)]
        if len(hist) < self.min_trigger_samples:
            return None
        recent = series[(series.index > t - pd.Timedelta(days=self.recent_window)) & (series.index <= t)]
        if recent.empty:
            return None
        std = float(hist.std())
        if std == 0:
            return None
        return float((float(recent.mean()) - float(hist.mean())) / std)

    # ------------------------------------------------------------- portfolio
    def apply(self, weights: dict[str, float], date: str | pd.Timestamp) -> dict[str, float]:
        """Cut 50% of any triggered (or frozen) position for ``date``."""
        t = pd.Timestamp(date)
        out = dict(weights)
        # advance freezes (a triggered day counts as day 1)
        for symbol in list(self._freeze_remaining):
            if self._freeze_remaining[symbol] <= 0:
                del self._freeze_remaining[symbol]
        for symbol, w in weights.items():
            if abs(w) < 1e-9:
                continue
            if self._freeze_remaining.get(symbol, 0) > 0:
                out[symbol] = w * self.position_cut
                self._freeze_remaining[symbol] -= 1
                continue
            z = self.zscore(symbol, t)
            if z is not None and z < self.zscore_threshold:
                out[symbol] = w * self.position_cut
                self._freeze_remaining[symbol] = self.freeze_days
                self._triggers.append(
                    {"symbol": symbol, "date": str(t.date()), "zscore": round(z, 3),
                     "weight": round(w, 4), "cut_weight": round(w * self.position_cut, 4)}
                )
        return out

    # ---------------------------------------------------------------- audit
    def trigger_log(self) -> list[dict]:
        return list(self._triggers)
