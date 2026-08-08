"""Sleipnir Dynamic Router (review.md §2.1).

The router observes the market regime and chooses which agent gets to run first,
because a strategy that works in a bull market should not be driving decisions
in a sell-off. The regime is classified from a return series via a simple
trend/volatility rule, and ``route`` returns the ordered list of agents that
best fit the current state. Interface-less by design — the router is a pure
function over (agents, market_state).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence

BULL = "bull"
BEAR = "bear"
SIDEWAYS = "sideways"


def classify_market_state(
    returns: pd.Series,
    window: int = 60,
    vol_high_q: float = 0.8,
) -> str:
    """Classify regime from a return series: bull / bear / sideways.

    Uses the sign of the annualised mean return and a high-volatility flag.
    A zero-volatility series is treated as a clean trend (not high-vol), so a
    steady drift is classified bull/bear rather than sideways.
    """
    r = returns.dropna().tail(window)
    if len(r) < 20:
        return SIDEWAYS
    ann = float(r.mean() * 252)
    vol = float(r.std() * np.sqrt(252))
    full = returns.dropna()
    if len(full) > 20:
        roll_vol = full.rolling(20, min_periods=20).std() * np.sqrt(252)
        high_vol = vol > 0 and vol >= roll_vol.quantile(vol_high_q)
    else:
        high_vol = False
    if ann > 0.05 and not high_vol:
        return BULL
    if ann < -0.05 or (ann < 0 and high_vol):
        return BEAR
    return SIDEWAYS


class DynamicRouter:
    """Route agents by market regime."""

    # Preferred agent run-order per regime. The heavy, idea-generating agent is
    # deprioritised in distressed regimes where execution discipline matters more.
    PREFERENCE: dict[str, tuple[str, ...]] = {
        BULL: ("signal", "code", "eval", "risk"),
        BEAR: ("risk", "eval", "code", "signal"),
        SIDEWAYS: ("eval", "signal", "code", "risk"),
    }

    def __init__(self, agents: dict[str, object], market_state: str = SIDEWAYS) -> None:
        self.agents = dict(agents)
        self.market_state = market_state

    def route(self) -> list[object]:
        order = self.PREFERENCE.get(self.market_state, self.PREFERENCE[SIDEWAYS])
        return [self.agents[name] for name in order if name in self.agents]

    def update_state(self, returns: pd.Series, window: int = 60) -> str:
        self.market_state = classify_market_state(returns, window=window)
        return self.market_state
