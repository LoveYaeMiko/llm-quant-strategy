"""LLM cost / budget tracking (review.md §3.2, blueprint checklist "Cost Check").

The blueprint demands the estimated monthly LLM API cost stay under $500. This
module records every model call routed through the pipeline, prices it against a
configurable per-token table, and exposes the running total so the loop can
refuse expensive steps and lean on the code-agent cache instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

# USD per 1M tokens. Placeholder rates — override via constructor.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "deepseek-v3": {"in": 0.27, "out": 1.10},        # cheap generator
    "deepseek-v4-flash": {"in": 0.27, "out": 1.10},  # active API model
    "gpt-4o": {"in": 2.50, "out": 10.00},            # accurate code
    "claude-3.5-sonnet": {"in": 3.00, "out": 15.00},  # critic / EvoQuant diagnosis
    "qwen2.5-7b": {"in": 0.10, "out": 0.20},         # sentence sentiment
    "default": {"in": 1.00, "out": 4.00},
}


@dataclass
class UsageEntry:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: float
    purpose: str = ""


class CostTracker:
    """Cumulative LLM usage ledger with a hard monthly budget."""

    def __init__(
        self,
        monthly_budget_usd: float = 500.0,
        prices: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self.monthly_budget = float(monthly_budget_usd)
        self.prices = {**DEFAULT_PRICES, **(prices or {})}
        self._entries: list[UsageEntry] = []

    # -- recording ----------------------------------------------------------

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        purpose: str = "",
    ) -> float:
        """Price a call, log it, and return the cost in USD."""
        rate = self.prices.get(model, self.prices["default"])
        cost = tokens_in / 1e6 * rate["in"] + tokens_out / 1e6 * rate["out"]
        self._entries.append(
            UsageEntry(
                model=model,
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                cost_usd=cost,
                timestamp=time.time(),
                purpose=purpose,
            )
        )
        return cost

    # -- reporting ----------------------------------------------------------

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self._entries)

    def cost_by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self._entries:
            out[e.model] = out.get(e.model, 0.0) + e.cost_usd
        return out

    def under_budget(self) -> bool:
        return self.total_cost() <= self.monthly_budget

    def budget_remaining(self) -> float:
        return self.monthly_budget - self.total_cost()

    def monthly_projection(self, days_elapsed: float = 1.0) -> float:
        """Extrapolate the current spend to a full month (30 days)."""
        if days_elapsed <= 0:
            return self.total_cost()
        return self.total_cost() / max(days_elapsed, 1.0) * 30.0

    def snapshot(self) -> dict:
        return {
            "monthly_budget_usd": self.monthly_budget,
            "total_cost_usd": round(self.total_cost(), 4),
            "by_model": {k: round(v, 4) for k, v in self.cost_by_model().items()},
            "under_budget": self.under_budget(),
            "n_calls": len(self._entries),
        }

    def entries(self) -> list[UsageEntry]:
        return list(self._entries)
