"""Autopilot — the end-to-end adaptive closed loop.

Turns the daily shadow run into a self-adjusting system:

* :class:`~src.autopilot.state.ControlState` — the persistent operating mode
  (normal / de_risk / halt) and gross multiplier;
* :func:`~src.autopilot.risk_gate.evaluate_risk_gate` — the kill-switch decision
  from live shadow metrics;
* :class:`~src.autopilot.control.ControlScaledPortfolio` — applies the multiplier
  to the three-layer book.

The orchestration (daily shadow → risk gate → periodic §7 re-calibration →
factor-decay monitor → opt-in re-mine) lives in ``src.cli.cmd_autopilot``, which
reuses the same market/portfolio builders as ``shadow`` and ``calibrate``.
"""

from .control import ControlScaledPortfolio
from .risk_gate import RiskDecision, evaluate_risk_gate
from .state import (
    ControlState,
    MODE_DE_RISK,
    MODE_HALT,
    MODE_NORMAL,
    mode_level,
)

__all__ = [
    "ControlState",
    "MODE_NORMAL",
    "MODE_DE_RISK",
    "MODE_HALT",
    "mode_level",
    "RiskDecision",
    "evaluate_risk_gate",
    "ControlScaledPortfolio",
]
