"""Deployment channel gate — observe (simulated) vs real money.

The system must be able to plug into a real broker WITHOUT any rule violation,
and nothing in the current pipeline may pretend to be live when it is not. This
module is the single place that answers "which channel is this run?" and refuses
to let a real-money path execute while the channel is still ``observe``.

Contract:

* ``deployment.mode: observe`` — everything is simulated/paper. Fills are the
  strategy's own paper fills; nothing is submitted anywhere.
* ``deployment.real_money_enabled: false`` — the hard gate. A future broker
  adapter MUST call :func:`assert_simulated_only` (or check
  :func:`deployment_status`) before submitting an order; with the gate closed it
  raises instead of trading.

The gate NEVER rewrites history: it only constrains actions taken from now on.
Existing simulated/shadow fills are evidence and stay exactly as recorded —
switching channels must not retroactively change any past fill, price or P&L.
"""
from __future__ import annotations

from typing import Any

#: Channels. ``observe`` = simulated/paper only; ``live`` = real money allowed.
MODE_OBSERVE = "observe"
MODE_LIVE = "live"

_DEFAULTS: dict[str, Any] = {
    "mode": MODE_OBSERVE,
    "real_money_enabled": False,
    "observe_since": "2026-01-01",
    "note": "影子/模拟盘：未接入任何券商接口，不产生真实委托",
}


class RealMoneyNotEnabled(RuntimeError):
    """Raised when a real-money path runs with the deployment gate closed."""


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse a YAML/env boolean without the ``bool("false") is True`` trap.

    The config layer interpolates ``${ENV}`` values, so ``real_money_enabled``
    can arrive as the STRING "false" — which ``bool()`` would turn into True and
    silently open the gate. Anything unrecognised falls back to ``default``
    (fail-closed for this module).
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def deployment_status(cfg=None) -> dict[str, Any]:
    """Resolved deployment channel (config section ``deployment``)."""
    section: dict[str, Any] = {}
    if cfg is not None:
        try:
            section = dict(cfg.section("deployment") or {})
        except Exception:  # noqa: BLE001 — a missing section is not an error
            section = {}
    mode = str(section.get("mode", _DEFAULTS["mode"]) or MODE_OBSERVE).strip().lower()
    if mode not in (MODE_OBSERVE, MODE_LIVE):
        mode = MODE_OBSERVE
    enabled = _as_bool(section.get("real_money_enabled", _DEFAULTS["real_money_enabled"]))
    # mode=observe is authoritative: even with the flag flipped (mis-edit, env
    # interpolation, manual change), an observe deployment never submits.
    if mode == MODE_OBSERVE:
        enabled = False
    return {
        "mode": mode,
        "real_money_enabled": enabled,
        "observe_since": section.get("observe_since", _DEFAULTS["observe_since"]),
        "note": section.get("note", _DEFAULTS["note"]),
        "simulated_only": not enabled,
    }


def assert_simulated_only(cfg, what: str = "order submission") -> dict[str, Any]:
    """Raise :class:`RealMoneyNotEnabled` when a real-money action is attempted.

    Returns the status dict when the action is allowed (i.e. the gate is open).
    """
    status = deployment_status(cfg)
    if status["real_money_enabled"]:
        return status
    raise RealMoneyNotEnabled(
        f"{what} blocked: deployment.mode={status['mode']} and "
        f"real_money_enabled=false (simulated/paper only — see docs/LIVE_READINESS.md)"
    )


__all__ = [
    "MODE_LIVE",
    "MODE_OBSERVE",
    "RealMoneyNotEnabled",
    "assert_simulated_only",
    "deployment_status",
]
