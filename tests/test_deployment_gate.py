"""Deployment gate + broker boundary (audit follow-ups).

The system must be able to plug into a real broker without violating any rule,
and nothing may present simulated fills as live. Pinned here:

* ``real_money_enabled`` arriving as the STRING "false" (config ``${ENV}``
  interpolation) must not open the gate via ``bool("false") is True``;
* ``mode: observe`` is authoritative regardless of the flag;
* ``assert_simulated_only`` had zero callers in ``src/`` — ``RealBroker``
  (src/live/broker.py) is now a real, tested call site that fails closed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.deploy import (
    MODE_OBSERVE,
    RealMoneyNotEnabled,
    assert_simulated_only,
    deployment_status,
)

ROOT = Path(__file__).resolve().parents[1]


class _Cfg:
    def __init__(self, section: dict | None) -> None:
        self._section = section

    def section(self, name: str) -> dict:
        if name != "deployment" or self._section is None:
            raise KeyError(name)
        return self._section


def test_missing_section_defaults_to_simulated():
    status = deployment_status(None)
    assert status["mode"] == MODE_OBSERVE
    assert status["real_money_enabled"] is False
    assert status["simulated_only"] is True
    assert deployment_status(_Cfg(None))["simulated_only"] is True


def test_gate_closed_raises():
    with pytest.raises(RealMoneyNotEnabled):
        assert_simulated_only(_Cfg({"mode": "observe", "real_money_enabled": False}), "buy 600000.SH")


def test_gate_open_allows():
    cfg = _Cfg({"mode": "live", "real_money_enabled": True})
    status = assert_simulated_only(cfg, "buy 600000.SH")
    assert status["real_money_enabled"] is True
    assert status["simulated_only"] is False


def test_unknown_mode_falls_back_to_observe():
    assert deployment_status(_Cfg({"mode": "yolo", "real_money_enabled": False}))["mode"] == MODE_OBSERVE


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off", ""])
def test_string_false_does_not_open_the_gate(raw):
    status = deployment_status(_Cfg({"mode": "live", "real_money_enabled": raw}))
    assert status["real_money_enabled"] is False
    assert status["simulated_only"] is True


@pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
def test_string_true_opens_the_gate_only_with_mode_live(raw):
    cfg = _Cfg({"mode": "live", "real_money_enabled": raw})
    assert deployment_status(cfg)["real_money_enabled"] is True


def test_mode_observe_is_authoritative():
    """A flipped flag in an observe deployment must NOT open the gate."""
    status = deployment_status(_Cfg({"mode": "observe", "real_money_enabled": True}))
    assert status["mode"] == MODE_OBSERVE
    assert status["real_money_enabled"] is False


def test_shipped_config_has_the_gate_closed():
    cfg = yaml.safe_load((ROOT / "configs" / "master_config.yaml").read_text(encoding="utf-8"))
    dep = cfg["deployment"]
    assert dep["mode"] == "observe"
    assert dep["real_money_enabled"] is False


def test_shadow_status_reports_the_channel():
    from src.config import load_config
    from src.paper.shadow import _deployment_status

    status = _deployment_status(load_config())
    assert status["mode"] == "observe"
    assert status["simulated_only"] is True


def test_real_broker_cannot_be_constructed_while_observe():
    from src.live.broker import PaperBroker, RealBroker

    cfg = _Cfg({"mode": "observe", "real_money_enabled": False})
    with pytest.raises(RealMoneyNotEnabled):
        RealBroker(cfg)
    # the paper boundary is always available and never touches a network
    res = PaperBroker(cfg).submit("600000.SH", "sell", 100.0, 10.0)
    assert res["simulated"] is True


def test_real_broker_reaches_the_adapter_gap_when_gate_open():
    """With the gate open the gate no longer blocks; the missing SDK does."""
    from src.live.broker import RealBroker

    with pytest.raises(NotImplementedError):
        RealBroker(_Cfg({"mode": "live", "real_money_enabled": True}))
