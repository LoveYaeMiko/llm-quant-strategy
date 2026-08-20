"""Autopilot control state — the persistent kill-switch operating mode.

The daily shadow run reads this state before building the book so it honours the
last risk-gate decision even across process restarts and machine reboots. The
orchestrator writes it back after each evaluation.

Only ``mode`` and ``gross_scale`` are acted on by the portfolio layer (the
:class:`~src.autopilot.control.ControlScaledPortfolio` wrapper); the remaining
fields are cadence bookkeeping (``last_*``) plus the ``factor_decayed`` flag the
decay monitor raises to de-risk the book when the deployed alpha degrades.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

MODE_NORMAL = "normal"
MODE_DE_RISK = "de_risk"
MODE_HALT = "halt"

# monotonic ordering so escalation/de-escalation can step levels one at a time.
_MODE_LEVEL = {MODE_NORMAL: 0, MODE_DE_RISK: 1, MODE_HALT: 2}


def mode_level(mode: str) -> int:
    return _MODE_LEVEL.get(mode, 0)


@dataclass
class ControlState:
    """One persisted snapshot of the autopilot's operating decision."""

    mode: str = MODE_NORMAL
    gross_scale: float = 1.0
    reason: str = ""
    since_date: Optional[str] = None   # date the current mode was entered
    last_evaluated: Optional[str] = None
    last_calibrate: Optional[str] = None
    last_monitor: Optional[str] = None
    last_mine: Optional[str] = None
    factor_decayed: bool = False
    extra: dict = field(default_factory=dict)

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ControlState":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "ControlState":
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()  # corrupt state -> start fresh at NORMAL, never crash
        return cls.from_dict(data) if isinstance(data, dict) else cls()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


__all__ = [
    "ControlState",
    "MODE_NORMAL",
    "MODE_DE_RISK",
    "MODE_HALT",
    "mode_level",
]
