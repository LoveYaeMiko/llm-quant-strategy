"""AlphaSchema — a structured space of trading semantics.

Each point in the space is a **schema plan** made of five components (blueprint
Phase 2, review.md §2.2 AlphaSchema):

* ``event``      — what triggers the trade (e.g. "Earnings Surprise");
* ``context``    — the regime it is conditioned on ("Bull Market");
* ``qualities``  — the behaviour being captured ("Momentum", "Mean Reversion");
* ``direction``  — long / short / long-short;
* ``output``     — continuous score or binary signal.

Exploration and implementation are decoupled: the LLM navigates *this* space and
a deterministic translator turns the chosen plan into a formula. The catalog
below is the prior knowledge — everything the explorer may propose is drawn from
it, which keeps the search bounded and reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# Component catalogs (the prior over trading semantics)
# ---------------------------------------------------------------------------

EVENTS: tuple[str, ...] = (
    "Earnings Surprise",
    "Momentum Breakout",
    "Mean Reversion",
    "Liquidity Shock",
    "Analyst Revision",
    "Volume Surge",
    "Gap Fill",
    "Trend Following",
    "Volatility Expansion",
    "Volatility Contraction",
    "Price Breakout",
    "Distribution Day",
    "Cumulative Flow",
    "Cross-Asset Spillover",
    "Seasonality",
    "Index Rebalancing",
    "Sentiment Extremes",
    "Funding Rate Spike",
)

CONTEXTS: tuple[str, ...] = (
    "Bull Market",
    "Bear Market",
    "Sideways Market",
    "High Volatility",
    "Low Volatility",
    "High Volume",
    "Low Volume",
    "Rising Rates",
    "Post-Earnings Drift",
    "Pre-Earnings",
    "Market-Neutral",
    "Sector Rotation",
    "Distressed Credit",
    "Liquidity-Rich",
    "Normal Regime",
)

QUALITIES: tuple[str, ...] = (
    "Momentum",
    "Mean Reversion",
    "Value",
    "Quality",
    "Low Volatility",
    "Liquidity",
    "Growth",
    "Sentiment",
    "Carry",
    "Short-Term Reversal",
    "Trend",
    "Size",
    "Accruals",
    "Analyst Dispersion",
)

DIRECTIONS: tuple[str, ...] = ("long", "short", "long_short")
OUTPUTS: tuple[str, ...] = ("score", "signal", "rank")


# ---------------------------------------------------------------------------
# Schema plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaPlan:
    """A single point in the semantic space (the five AlphaSchema components)."""

    event: str
    context: str
    qualities: tuple[str, ...] = ("Momentum",)
    direction: str = "long"
    output: str = "score"

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"invalid direction {self.direction!r}")
        if self.output not in OUTPUTS:
            raise ValueError(f"invalid output {self.output!r}")
        if not self.qualities:
            raise ValueError("qualities must be non-empty")

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaPlan":
        return cls(
            event=data["event"],
            context=data["context"],
            qualities=tuple(data.get("qualities", ("Momentum",))),
            direction=data.get("direction", "long"),
            output=data.get("output", "score"),
        )

    def key(self) -> str:
        """Canonical serialization — identical plans produce identical keys."""
        return json.dumps(
            {
                "event": self.event,
                "context": self.context,
                "qualities": list(self.qualities),
                "direction": self.direction,
                "output": self.output,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def natural_language(self) -> str:
        """A human-readable rendering used to prompt the LLM."""
        q = ", ".join(self.qualities)
        return (
            f"Trade {self.event} in a {self.context} regime, capturing {q}, "
            f"{self.direction}, emitting a {self.output}."
        )


# ---------------------------------------------------------------------------
# Semantic space
# ---------------------------------------------------------------------------


class SemanticSpace:
    """Catalog-driven space with sampling and local neighbourhoods."""

    def __init__(
        self,
        events: Iterable[str] = EVENTS,
        contexts: Iterable[str] = CONTEXTS,
        qualities: Iterable[str] = QUALITIES,
        directions: Iterable[str] = DIRECTIONS,
        outputs: Iterable[str] = OUTPUTS,
    ) -> None:
        self.events = tuple(events)
        self.contexts = tuple(contexts)
        self.qualities = tuple(qualities)
        self.directions = tuple(directions)
        self.outputs = tuple(outputs)

    # -- exploration primitives --------------------------------------------

    def sample(self, rng: Optional[random.Random] = None) -> SchemaPlan:
        rng = rng or random
        n_qualities = rng.randint(1, 2)
        return SchemaPlan(
            event=rng.choice(self.events),
            context=rng.choice(self.contexts),
            qualities=tuple(rng.sample(self.qualities, n_qualities)),
            direction=rng.choice(self.directions),
            output=rng.choice(self.outputs),
        )

    def neighbors(self, plan: SchemaPlan, rng: Optional[random.Random] = None, k: int = 4) -> list[SchemaPlan]:
        """Local perturbations — change exactly one component (AlphaJungle's
        'frequent subtree avoidance' needs a well-defined notion of 'close')."""
        rng = rng or random
        out: list[SchemaPlan] = []
        field_choices: list[tuple[tuple[str, ...], str]] = [
            (self.events, "event"),
            (self.contexts, "context"),
            (self.qualities, "qualities"),
            (self.directions, "direction"),
            (self.outputs, "output"),
        ]
        for pool, field in field_choices:
            for _ in range(k):
                choice = rng.choice(pool)
                if field == "qualities":
                    new_q = list(plan.qualities)
                    idx = rng.randrange(len(new_q))
                    new_q[idx] = choice
                    candidate = SchemaPlan(
                        event=plan.event, context=plan.context,
                        qualities=tuple(new_q), direction=plan.direction, output=plan.output,
                    )
                else:
                    kwargs = plan.to_dict()
                    kwargs[field] = choice
                    candidate = SchemaPlan.from_dict(kwargs)
                if candidate != plan and candidate not in out:
                    out.append(candidate)
        return out[: max(0, k * 4)]

    def validate(self, plan: SchemaPlan) -> bool:
        return (
            plan.event in self.events
            and plan.context in self.contexts
            and all(q in self.qualities for q in plan.qualities)
            and plan.direction in self.directions
            and plan.output in self.outputs
        )
