"""Autopilot — the adaptive closed loop (control state + risk gate + scaling).

Covers the three pure pieces of the kill-switch:

* :class:`ControlState` persistence (round-trip + corrupt-file fallback);
* :func:`evaluate_risk_gate` (escalation / de-escalation / hysteresis / cooldown);
* :class:`ControlScaledPortfolio` (gross multiplier → flat on halt).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.autopilot.control import ControlScaledPortfolio
from src.autopilot.risk_gate import evaluate_risk_gate
from src.autopilot.state import (
    ControlState,
    MODE_DE_RISK,
    MODE_HALT,
    MODE_NORMAL,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _status(equities, start="2026-01-01"):
    """Build a shadow-status-shaped dict with an equity curve + per-day drawdown."""
    dates = pd.bdate_range(start, periods=len(equities))
    eq = pd.Series(equities, index=dates, dtype=float)
    dd = eq / eq.cummax() - 1.0
    curve = [
        {"date": str(d.date()), "equity": float(e), "drawdown": float(dd.loc[d])}
        for d, e in eq.items()
    ]
    return {
        "equity_curve": curve,
        "equity": {"max_drawdown": 0.0, "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1)},
    }


def _risk_cfg(**overrides) -> dict:
    cfg = {
        "min_history_days": 20,
        "drawdown_de_risk": 0.10,
        "drawdown_halt": 0.15,
        "trailing_window_days": 60,
        "trailing_return_de_risk": -0.10,
        "trailing_return_halt": -0.15,
        "consecutive_loss_days_halt": 20,
        "de_risk_scale": 0.5,
        "recovery_hysteresis": 0.5,
        "cooldown_days": 5,
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# ControlState
# --------------------------------------------------------------------------- #
def test_control_state_roundtrip(tmp_path):
    s = ControlState(mode=MODE_DE_RISK, gross_scale=0.5, reason="x",
                     since_date="2026-01-01")
    p = tmp_path / "state.json"
    s.save(p)
    s2 = ControlState.load(p)
    assert s2.mode == MODE_DE_RISK
    assert s2.gross_scale == pytest.approx(0.5)
    assert s2.reason == "x"
    assert s2.since_date == "2026-01-01"


def test_control_state_missing_file_is_normal(tmp_path):
    assert ControlState.load(tmp_path / "nope.json").mode == MODE_NORMAL


def test_control_state_corrupt_file_fails_closed(tmp_path):
    # a present-but-corrupt file must NOT re-arm the book to full gross
    p = tmp_path / "state.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = ControlState.load(p)
    assert s.mode == MODE_DE_RISK
    assert s.gross_scale == pytest.approx(0.5)


def test_control_state_non_object_json_fails_closed(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert ControlState.load(p).mode == MODE_DE_RISK


def test_control_state_unknown_mode_fails_closed(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"mode": "yolo", "gross_scale": 1.0}', encoding="utf-8")
    assert ControlState.load(p).mode == MODE_DE_RISK


def test_control_state_null_gross_scale_fails_closed(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"mode": "normal", "gross_scale": null}', encoding="utf-8")
    assert ControlState.load(p).mode == MODE_DE_RISK


def test_control_state_out_of_range_gross_scale_fails_closed(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"mode": "normal", "gross_scale": 3.0}', encoding="utf-8")
    assert ControlState.load(p).mode == MODE_DE_RISK


def test_control_state_load_falls_back_to_bak(tmp_path):
    # corrupt primary, valid .bak -> recover the previous good snapshot
    p = tmp_path / "state.json"
    (tmp_path / "state.json.bak").write_text(
        json.dumps({"mode": "halt", "gross_scale": 0.0}), encoding="utf-8"
    )
    p.write_text("{corrupt", encoding="utf-8")
    s = ControlState.load(p)
    assert s.mode == MODE_HALT
    assert s.gross_scale == pytest.approx(0.0)


def test_control_state_save_is_atomic_and_keeps_bak(tmp_path):
    p = tmp_path / "state.json"
    ControlState(mode=MODE_NORMAL).save(p)
    ControlState(mode=MODE_HALT, gross_scale=0.0).save(p)
    # the second save kept the previous snapshot as .bak and left no stray .tmp
    assert (tmp_path / "state.json.bak").is_file()
    assert ControlState.load(tmp_path / "state.json.bak").mode == MODE_NORMAL
    assert not (tmp_path / "state.json.tmp").exists()


# --------------------------------------------------------------------------- #
# risk gate — escalation
# --------------------------------------------------------------------------- #
def test_risk_gate_holds_on_thin_history():
    s = _status([100.0, 90.0])  # only 2 days
    d = evaluate_risk_gate(s, ControlState(), _risk_cfg(), now=pd.Timestamp("2026-01-10"))
    assert d.mode == MODE_NORMAL
    assert d.changed is False


def test_risk_gate_de_risks_on_drawdown():
    # 40 days, last 12% below peak -> de-risk
    eq = [100.0] * 28 + [88.0] * 12
    d = evaluate_risk_gate(_status(eq), ControlState(), _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    assert d.mode == MODE_DE_RISK
    assert d.gross_scale == pytest.approx(0.5)


def test_risk_gate_halts_on_deep_drawdown():
    eq = [100.0] * 28 + [82.0] * 12  # 18% below peak
    d = evaluate_risk_gate(_status(eq), ControlState(), _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    assert d.mode == MODE_HALT
    assert d.gross_scale == pytest.approx(0.0)


def test_risk_gate_halts_on_trailing_return():
    # steady -0.5%/day over 60 days -> trailing return below -15%
    eq = [100.0 * (0.995 ** i) for i in range(40)]
    d = evaluate_risk_gate(_status(eq), ControlState(), _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    assert d.mode == MODE_HALT


def test_risk_gate_de_risks_on_factor_decay():
    # a decayed deployed pool raises the floor to de-risk even with a healthy curve
    current = ControlState(factor_decayed=True)
    eq = [100.0] * 40
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    assert d.mode == MODE_DE_RISK
    assert d.gross_scale == pytest.approx(0.5)


def test_risk_gate_de_risks_on_factor_decay_even_with_thin_history():
    # the decay signal comes from the frozen test window, not the live curve, so
    # a thin curve (below min_history) must not dodge it
    current = ControlState(factor_decayed=True)
    s = _status([100.0, 90.0])  # 2 days < min_history
    d = evaluate_risk_gate(s, current, _risk_cfg(), now=pd.Timestamp("2026-01-10"))
    assert d.mode == MODE_DE_RISK
    assert d.gross_scale == pytest.approx(0.5)


def test_risk_gate_halts_on_consecutive_losses():
    # 20 straight down days at the tail (small enough to avoid drawdown halt? no—
    # use tiny steps so drawdown stays under 15%, but 20 consecutive losses fire)
    eq = [100.0] + [100.0 * (0.999 ** i) for i in range(1, 21)]
    d = evaluate_risk_gate(_status(eq), ControlState(), _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    # drawdown ~2%, trailing return tiny — only the consecutive-loss trigger fires
    assert d.mode == MODE_HALT


# --------------------------------------------------------------------------- #
# risk gate — de-escalation / hysteresis / cooldown
# --------------------------------------------------------------------------- #
def test_risk_gate_de_escalates_after_recovery_and_cooldown():
    # currently de-risked; book fully recovered (flat) and cooldown elapsed
    current = ControlState(mode=MODE_DE_RISK, gross_scale=0.5, since_date="2026-01-01")
    eq = [100.0] * 40
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-01-20"))
    assert d.mode == MODE_NORMAL
    assert d.changed is True


def test_risk_gate_holds_without_cooldown():
    current = ControlState(mode=MODE_DE_RISK, gross_scale=0.5, since_date="2026-01-15")
    eq = [100.0] * 40
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-01-17"))  # 2 days < 5 cooldown
    assert d.mode == MODE_DE_RISK
    assert d.changed is False


def test_risk_gate_hysteresis_requires_full_recovery():
    # 6% drawdown: above the 0.10*0.5 = 5% hysteresis recovery band -> hold de-risk
    current = ControlState(mode=MODE_DE_RISK, gross_scale=0.5, since_date="2026-01-01")
    eq = [100.0] * 28 + [94.0] * 12
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-01-20"))
    assert d.mode == MODE_DE_RISK
    assert d.changed is False


def test_risk_gate_de_escalates_from_halt_after_recovery():
    # halted on a deep drawdown; the book is flat afterward so the frozen peak
    # keeps currentDD ~18% — the gate must still step down to DE_RISK once the
    # trailing window is flat and the cooldown has elapsed (a halt must not be
    # an absorbing state).
    current = ControlState(mode=MODE_HALT, gross_scale=0.0, since_date="2026-01-01")
    eq = [100.0] * 10 + [82.0] * 60
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-03-20"))
    assert d.mode == MODE_DE_RISK
    assert d.gross_scale == pytest.approx(0.5)


def test_risk_gate_holds_de_risk_without_since_date():
    # a de-risked state with no since_date must NOT de-escalate — the cooldown is
    # a safety latch and fails closed, not open
    current = ControlState(mode=MODE_DE_RISK, gross_scale=0.5, since_date=None)
    eq = [100.0] * 40
    d = evaluate_risk_gate(_status(eq), current, _risk_cfg(),
                           now=pd.Timestamp("2026-02-20"))
    assert d.mode == MODE_DE_RISK
    assert d.changed is False


# --------------------------------------------------------------------------- #
# ControlScaledPortfolio
# --------------------------------------------------------------------------- #
class _FakePortfolio:
    def compute_weights(self, symbols, date):
        return {symbols[0]: 0.5, symbols[1]: -0.5}


def test_control_scaled_normal():
    w = ControlScaledPortfolio(_FakePortfolio(), lambda: 1.0)
    assert w.compute_weights(["A", "B"], None) == {"A": 0.5, "B": -0.5}


def test_control_scaled_de_risk():
    w = ControlScaledPortfolio(_FakePortfolio(), lambda: 0.5)
    out = w.compute_weights(["A", "B"], None)
    assert out == {"A": pytest.approx(0.25), "B": pytest.approx(-0.25)}


def test_control_scaled_halt_flattens_book():
    w = ControlScaledPortfolio(_FakePortfolio(), lambda: 0.0)
    out = w.compute_weights(["A", "B"], None)
    assert out == {"A": 0.0, "B": 0.0}
