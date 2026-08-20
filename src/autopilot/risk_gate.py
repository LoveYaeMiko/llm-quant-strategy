"""Risk gate — turn live shadow metrics into a kill-switch decision.

Pure function of ``(shadow_status, current_state, thresholds)`` so it is trivial
to test and deterministic given the same inputs. It decides the operating mode:

* ``normal``   — gross 1.0 (the full three-layer book);
* ``de_risk``  — gross ``de_risk_scale`` (default 0.5);
* ``halt``     — gross 0.0 (flat).

**Escalation is immediate and monotonic** — a breach moves the book up a level
this same run. **De-escalation is conservative** — it only steps down one level
at a time, and only after the live drawdown/return have recovered below the
hysteresis band *and* a cooldown has elapsed, so the gate never flaps on noise.

The primary signal is the *current* drawdown from peak (``equity_curve``'s
``drawdown`` field), not the historical max-drawdown metric: a kill-switch must
answer "is the book underwater right now", not "was it ever underwater".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .state import (
    ControlState,
    MODE_DE_RISK,
    MODE_HALT,
    MODE_NORMAL,
    mode_level,
)


@dataclass
class RiskDecision:
    mode: str
    gross_scale: float
    changed: bool
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# signal extraction from the shadow status payload
# --------------------------------------------------------------------------- #
def _equity_curve(status: dict) -> tuple[pd.Series, pd.Series]:
    """Return ``(equity, drawdown)`` series from ``status["equity_curve"]``.

    ``drawdown`` is the per-day drawdown-from-peak already persisted by
    :func:`src.paper.shadow.build_shadow_status`; empty series when the shadow
    has not yet accumulated a curve.
    """
    curve = status.get("equity_curve") or []
    if not curve:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    eq = pd.Series(
        [float(r.get("equity", 0.0)) for r in curve],
        index=[r.get("date") for r in curve],
        dtype=float,
    )
    dd = pd.Series(
        [float(r.get("drawdown", 0.0)) for r in curve],
        index=[r.get("date") for r in curve],
        dtype=float,
    )
    return eq.sort_index(), dd.sort_index()


def _current_drawdown(eq: pd.Series, dd: pd.Series) -> float:
    """Current drawdown-from-peak as a positive magnitude (0.10 = 10% down)."""
    if len(dd):
        return abs(float(dd.iloc[-1]))
    if len(eq) >= 2 and eq.iloc[-1] > 0:
        peak = float(eq.cummax().iloc[-1])
        if peak > 0:
            return max(0.0, 1.0 - float(eq.iloc[-1]) / peak)
    return 0.0


def _trailing_return(eq: pd.Series, window: int) -> float:
    """Return over the trailing ``window`` days (0.0 when the window is short)."""
    if window <= 0 or len(eq) < 2:
        return 0.0
    tail = eq.iloc[-window:]
    base = float(tail.iloc[0])
    if base <= 0:
        return 0.0
    return float(tail.iloc[-1]) / base - 1.0


def _consecutive_loss_days(eq: pd.Series) -> int:
    """Consecutive down-days at the tail of the equity curve."""
    if len(eq) < 2:
        return 0
    n = 0
    for r in eq.pct_change().dropna().iloc[::-1]:
        if r < 0:
            n += 1
        else:
            break
    return n


def _days_since(since_date: Optional[str], now: pd.Timestamp) -> float:
    if not since_date:
        return float("inf")
    try:
        return float((now.normalize() - pd.Timestamp(since_date).normalize()).days)
    except (ValueError, TypeError):
        return float("inf")


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #
def evaluate_risk_gate(
    status: dict,
    current: ControlState,
    risk_cfg: dict,
    *,
    now: Optional[pd.Timestamp] = None,
) -> RiskDecision:
    """Decide the operating mode from the latest shadow status.

    ``risk_cfg`` is the ``autopilot.risk_gate`` section; ``now`` is injectable for
    deterministic tests (defaults to today).
    """
    now = now or pd.Timestamp.today()
    eq, dd = _equity_curve(status)
    current_dd = _current_drawdown(eq, dd)

    min_history = int(risk_cfg.get("min_history_days", 20))
    dd_de_risk = float(risk_cfg.get("drawdown_de_risk", 0.10))
    dd_halt = float(risk_cfg.get("drawdown_halt", 0.15))
    trail_window = int(risk_cfg.get("trailing_window_days", 60))
    trail_de_risk = float(risk_cfg.get("trailing_return_de_risk", -0.10))
    trail_halt = float(risk_cfg.get("trailing_return_halt", -0.15))
    loss_halt = int(risk_cfg.get("consecutive_loss_days_halt", 20))
    de_risk_scale = float(risk_cfg.get("de_risk_scale", 0.5))
    hysteresis = float(risk_cfg.get("recovery_hysteresis", 0.5))
    cooldown = int(risk_cfg.get("cooldown_days", 5))

    trail = _trailing_return(eq, trail_window)
    consec = _consecutive_loss_days(eq)

    reasons: list[str] = []
    # Not enough live history — refuse to react to a handful of noisy days.
    if len(eq) < min_history:
        reasons.append(
            f"history {len(eq)}d < min {min_history}d — hold {current.mode} "
            f"(no kill-switch on a thin curve)"
        )
        return RiskDecision(current.mode, current.gross_scale, False, reasons)

    halt = current_dd >= dd_halt or trail <= trail_halt or consec >= loss_halt
    derisk = current_dd >= dd_de_risk or trail <= trail_de_risk

    # A decayed deployed factor pool is itself a reason to de-risk: the alpha is
    # losing predictive power, so at minimum halve the gross until the next
    # monitor pass re-scores it (this only raises the floor — never lowers it).
    if bool(getattr(current, "factor_decayed", False)):
        derisk = True
        reasons.append("deployed factor pool decayed → hold at least DE_RISK")

    cur_level = mode_level(current.mode)
    new_mode = current.mode

    if halt:
        new_mode = MODE_HALT
        reasons.append(
            f"currentDD {current_dd:.1%} / trail {trail:.1%} / consec {consec}d → HALT"
        )
    elif derisk:
        # still elevated; never step *down* below de_risk while a breach persists
        new_mode = MODE_DE_RISK if cur_level < mode_level(MODE_DE_RISK) else current.mode
        reasons.append(f"currentDD {current_dd:.1%} / trail {trail:.1%} → DE_RISK")
    elif cur_level > 0:
        # no breach — consider one-level de-escalation (needs recovery + cooldown)
        recovered = (
            current_dd < dd_de_risk * hysteresis
            and trail > trail_de_risk * hysteresis
            and consec == 0
        )
        cooled = _days_since(current.since_date, now) >= cooldown
        if recovered and cooled:
            new_mode = MODE_NORMAL if current.mode == MODE_DE_RISK else MODE_DE_RISK
            reasons.append(
                f"recovered (DD {current_dd:.1%}, trail {trail:+.1%}) + cooled → {new_mode}"
            )
        else:
            reasons.append(
                f"holding {current.mode} (recovered={recovered}, cooled={cooled})"
            )
    else:
        reasons.append(f"no breach (DD {current_dd:.1%}, trail {trail:+.1%}) → normal")

    new_scale = {
        MODE_NORMAL: 1.0,
        MODE_DE_RISK: de_risk_scale,
        MODE_HALT: 0.0,
    }[new_mode]

    changed = new_mode != current.mode or abs(new_scale - current.gross_scale) > 1e-9
    return RiskDecision(new_mode, new_scale, changed, reasons)


__all__ = ["RiskDecision", "evaluate_risk_gate"]
