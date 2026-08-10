"""Factor-decay monitoring over rolling windows (Q5, blueprint §8).

A factor's predictive power erodes as markets adapt. DecayTracker scores a
signal's cross-sectional IC in a rolling window and flags decay when the
annualised ICIR falls below the configured floor — by default
``research.decay.icir_threshold: 0.30``, aligned with
``factor_thresholds.yaml icir.keep_threshold`` so "monitor" and "keep" agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.metrics import daily_ic, icir
from ..config import Config

_PERIODS_PER_YEAR = 252.0


@dataclass
class DecayWindow:
    """One scored rolling window of a signal's IC history."""

    start: pd.Timestamp
    end: pd.Timestamp
    ic: float
    icir: float
    n_days: int
    decayed: bool


@dataclass
class DecayMonitorResult:
    """Outcome of one :meth:`DecayTracker.monitor` call."""

    windows: list[DecayWindow]
    recent_icir: float
    decayed: bool
    first_decayed_at: Optional[pd.Timestamp]
    summary: str

    @property
    def healthy(self) -> bool:
        return not self.decayed


class DecayTracker:
    """Rolling-window ICIR monitor for a single factor signal.

    ``window_days`` counts trading observations (integer rolling window), so the
    default 90 is 90 sessions (~4 months of A-share trading), not 90 calendar
    days (~64 sessions) — that is what the ``√252`` annualisation assumes.
    """

    def __init__(
        self,
        window_days: int = 90,
        icir_threshold: float = 0.30,
        min_days: int = 20,
    ) -> None:
        self.window_days = window_days
        self.icir_threshold = icir_threshold
        self.min_days = min_days

    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> "DecayTracker":
        """Build from ``research.decay`` in the master config."""
        decay = config.section("research.decay") if config is not None else {}
        return cls(
            window_days=int(decay.get("window_days", 90)),
            icir_threshold=float(decay.get("icir_threshold", 0.30)),
        )

    def monitor(self, signal: pd.Series, forward_returns: pd.Series) -> DecayMonitorResult:
        """Score ``signal`` vs ``forward_returns`` on a rolling ``window_days`` span.

        Both series are indexed ``(date, symbol)``. The IC series is computed once
        via :func:`daily_ic`, then rolled; ICIR per window is
        ``mean(IC)/std(IC)·√252`` — vectorised, so a 15-year history is cheap.
        """
        ic = daily_ic(signal, forward_returns, method="spearman").sort_index()
        if ic.empty:
            return DecayMonitorResult(
                windows=[], recent_icir=0.0, decayed=False,
                first_decayed_at=None,
                summary="no overlapping (date, symbol) signal/forward data to score",
            )
        # integer window = `window_days` TRADING observations (A-share ~242/yr);
        # a calendar offset would shrink the window to ~64 observations per 90 days
        roll_mean = ic.rolling(self.window_days, min_periods=self.min_days).mean()
        roll_std = ic.rolling(self.window_days, min_periods=self.min_days).std().replace(0.0, np.nan)
        roll_icir = (roll_mean / roll_std * np.sqrt(_PERIODS_PER_YEAR)).dropna()
        if roll_icir.empty:
            return DecayMonitorResult(
                windows=[], recent_icir=0.0, decayed=False,
                first_decayed_at=None,
                summary=f"<{self.min_days} days of IC history — cannot form a window",
            )

        recent_icir = float(roll_icir.iloc[-1])
        below = roll_icir < self.icir_threshold
        decayed = bool(below.any())
        first_decayed_at = roll_icir.index[below].min() if decayed else None

        # window records at a monthly stride (bounded list for the monitor CLI)
        stride = 21  # trading days
        idxs = list(range(0, len(roll_icir), stride))
        if idxs and idxs[-1] != len(roll_icir) - 1:
            idxs.append(len(roll_icir) - 1)
        windows = []
        for i in idxs:
            end = roll_icir.index[i]
            seg = ic[ic.index <= end].iloc[-self.window_days:]
            windows.append(
                DecayWindow(
                    start=seg.index[0] if len(seg) else end,
                    end=end,
                    ic=float(roll_mean.iloc[i]),
                    icir=float(roll_icir.iloc[i]),
                    n_days=int(len(seg)),
                    decayed=bool(roll_icir.iloc[i] < self.icir_threshold),
                )
            )

        state = "DECAYED" if decayed else "healthy"
        summary = (
            f"recent {self.window_days}d ICIR {recent_icir:.2f} ({state}, "
            f"floor {self.icir_threshold:.2f}); first below floor "
            f"{first_decayed_at.date() if first_decayed_at is not None else 'never'}"
        )
        return DecayMonitorResult(
            windows=windows, recent_icir=recent_icir, decayed=decayed,
            first_decayed_at=first_decayed_at, summary=summary,
        )

    # -- convenience aliases (a factor that decayed is one to drop or re-mine) --

    def should_keep(self, signal: pd.Series, forward_returns: pd.Series) -> bool:
        return not self.monitor(signal, forward_returns).decayed


__all__ = ["DecayTracker", "DecayMonitorResult", "DecayWindow"]
