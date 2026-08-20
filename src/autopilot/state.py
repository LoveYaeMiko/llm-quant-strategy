"""Autopilot control state — the persistent kill-switch operating mode.

The daily shadow run reads this state before building the book so it honours the
last risk-gate decision even across process restarts and machine reboots. The
orchestrator writes it back after each evaluation.

Only ``mode`` and ``gross_scale`` are acted on by the portfolio layer (the
:class:`~src.autopilot.control.ControlScaledPortfolio` wrapper); the remaining
fields are cadence bookkeeping (``last_*``) plus the ``factor_decayed`` flag the
decay monitor raises to de-risk the book when the deployed alpha degrades.

Persistence is **fail-closed** by design:

* a *missing* file is a clean first run and yields ``normal`` / gross 1.0;
* a *present but corrupt* file (bad JSON, non-object, unknown ``mode``, or a
  non-numeric ``gross_scale``) yields ``de_risk`` / gross 0.5 — the book may be
  under-exposed but is never silently re-armed to full gross on corrupted state;
* ``save`` writes atomically (temp file + ``os.replace``) and keeps the previous
  good snapshot as ``*.bak`` so a later corrupt load can fall back to it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
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
    factor_decayed: bool = False
    extra: dict = field(default_factory=dict)

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def fail_closed(cls, reason: str) -> "ControlState":
        """State that could not be trusted — de-risk, never re-arm to full.

        ``since_date`` is stamped so the cooldown clock can eventually let the
        gate step this back up to ``normal`` on a clean evaluation, instead of
        pinning the book at ``de_risk`` forever.
        """
        return cls(
            mode=MODE_DE_RISK,
            gross_scale=0.5,
            reason=reason,
            since_date=date.today().isoformat(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ControlState":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls._sanitize(known)

    @classmethod
    def _sanitize(cls, fields: dict) -> "ControlState":
        """Validate the two fields the portfolio layer acts on.

        An unknown ``mode`` would otherwise KeyError the risk gate and a null /
        non-numeric ``gross_scale`` would TypeError it — both crash the daily run.
        Instead we fail closed to ``de_risk`` so the book is under-exposed but
        the loop keeps turning.
        """
        mode = fields.get("mode", MODE_NORMAL)
        if mode not in _MODE_LEVEL:
            return cls.fail_closed(f"unknown mode {mode!r} in state file")
        try:
            gross = float(fields.get("gross_scale", 1.0))
        except (TypeError, ValueError):
            return cls.fail_closed("non-numeric gross_scale in state file")
        if not math.isfinite(gross) or not (0.0 <= gross <= 1.0):
            return cls.fail_closed(f"gross_scale {gross!r} out of [0, 1] in state file")
        fields["mode"] = mode
        fields["gross_scale"] = gross
        return cls(**fields)

    @staticmethod
    def _read_json(path: Path) -> Optional[object]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @classmethod
    def load(cls, path: str | Path) -> "ControlState":
        p = Path(path)
        if not p.is_file():
            return cls()  # clean first run — NORMAL is the correct initial state

        data = cls._read_json(p)
        if data is None:
            # primary is corrupt — fall back to the previous good snapshot before
            # giving up, so a transient write fault doesn't cost us the mode.
            bak = p.with_suffix(p.suffix + ".bak")
            if bak.is_file():
                data = cls._read_json(bak)
        if data is None:
            return cls.fail_closed("state file unreadable/corrupt")
        if not isinstance(data, dict):
            return cls.fail_closed("state file is not a JSON object")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

        if p.is_file():
            # keep the previous good state so a later corrupt load can fall back
            # to it rather than re-arming the book to full gross.
            try:
                shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            except OSError:
                pass
        # atomic: readers see either the old or the new file, never a truncation.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, p)


__all__ = [
    "ControlState",
    "MODE_NORMAL",
    "MODE_DE_RISK",
    "MODE_HALT",
    "mode_level",
]
