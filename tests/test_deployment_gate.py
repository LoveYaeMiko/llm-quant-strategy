"""Deployment-channel gate tests (landing item 4.1).

The system must be able to plug into a real broker without violating any rule,
and nothing may present simulated fills as live. ``deployment.mode`` is
``observe`` and ``real_money_enabled`` is ``false`` until the LIVE_READINESS
checklist is fully green; the gate only constrains FUTURE actions and never
rewrites a recorded fill.
"""
from __future__ import annotations

import pytest
import yaml

from pathlib import Path

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
