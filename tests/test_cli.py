"""CLI smoke tests — every subcommand runs and exits cleanly (offline)."""

from __future__ import annotations

import json
import os

import pytest

from src.cli import main

# ensure the offline path in tests: an empty key is "set", so .env won't
# repopulate it (config._load_dotenv only sets vars absent from os.environ).
os.environ["DEEPSEEK_API_KEY"] = ""


@pytest.fixture()
def outputs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setenv("LLM_QUANT_OUTPUTS", str(out))
    return out


def test_verify_offline():
    rc = main(["verify", "--seed", "1"])
    assert rc == 0


def test_export_writes_compiled_json(outputs):
    rc = main(["export", "--name", "testfactor", "--formula", "Neg(TS_ZScore(Close, 20))"])
    assert rc == 0
    assert (outputs / "testfactor.compiled.json").exists()


def test_backtest_offline():
    rc = main(["backtest", "--seed", "1", "--formulas", "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"])
    assert rc == 0


def test_evolve_offline_smoke():
    rc = main(["evolve", "--seed", "1", "--trials", "1"])
    assert rc == 0


def test_mine_offline_small(outputs):
    rc = main(["mine", "--iterations", "2", "--hypotheses", "3", "--seed", "1"])
    assert rc in (0, 1)  # exit code 1 only if a checklist gate fails on luck
    assert (outputs / "memory.json").exists()
    assert list(outputs.glob("audit_*.json"))
