"""ML promote gate — the production barrier a model must clear before shadow.

Pure functions over metric dicts, so the decision is deterministic and
testable. The gate is deliberately stricter than the incumbent comparison
alone: a model replaces the deployed pool only if it beats the incumbent on
net Sharpe AND max drawdown AND clears the absolute floors on the test window.

Rules (config-backed defaults, never auto-relaxed):
* ``min_sharpe``       — absolute net-Sharpe floor (default 1.0);
* ``max_drawdown``     — absolute drawdown ceiling (default 15%);
* ``min_ic``           — test-window rank_ic floor (default 0.02);
* ``beat_incumbent``   — must exceed the incumbent on BOTH net Sharpe and maxDD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromoteDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def blocked_by(self) -> list[str]:
        return self.reasons


DEFAULT_GATES: dict[str, float] = {
    "min_sharpe": 1.0,
    "max_drawdown": 0.15,
    "min_ic": 0.02,
    "beat_incumbent": 1.0,
}


def evaluate_promotion(
    model: dict[str, Any],
    incumbent: dict[str, Any] | None,
    gates: dict[str, float] | None = None,
) -> PromoteDecision:
    """Decide whether ``model`` may replace ``incumbent`` in the live book.

    ``model`` / ``incumbent`` carry portfolio metrics (``sharpe``, ``max_dd``)
    and, for the model, the test-window ``rank_ic``. ``incumbent=None`` means
    no live pool exists (first deployment) — absolute floors still apply.
    """
    g = dict(DEFAULT_GATES)
    if gates:
        g.update(gates)

    reasons: list[str] = []
    sharpe = float(model.get("sharpe", 0.0) or 0.0)
    max_dd = float(model.get("max_dd", 1.0) or 1.0)
    rank_ic = float(model.get("rank_ic", 0.0) or 0.0)

    if sharpe < float(g["min_sharpe"]):
        reasons.append(f"net Sharpe {sharpe:.2f} < floor {g['min_sharpe']}")
    if max_dd > float(g["max_drawdown"]):
        reasons.append(f"maxDD {max_dd:.2%} > ceiling {g['max_drawdown']:.0%}")
    if abs(rank_ic) < float(g["min_ic"]):
        reasons.append(f"|rank_ic| {rank_ic:.4f} < floor {g['min_ic']}")

    if incumbent is not None:
        inc_sharpe = float(incumbent.get("sharpe", 0.0) or 0.0)
        inc_dd = float(incumbent.get("max_dd", 1.0) or 1.0)
        if not (sharpe > inc_sharpe and max_dd < inc_dd):
            reasons.append(
                f"does not beat incumbent on both axes "
                f"(sharpe {sharpe:.2f} vs {inc_sharpe:.2f}, maxDD {max_dd:.2%} vs {inc_dd:.2%})"
            )

    return PromoteDecision(promote=not reasons, reasons=reasons)


__all__ = ["PromoteDecision", "DEFAULT_GATES", "evaluate_promotion"]
