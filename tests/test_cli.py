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


def test_mine_window_flag_all(outputs):
    rc = main(["mine", "--iterations", "1", "--hypotheses", "2", "--seed", "1", "--window", "all"])
    assert rc in (0, 1)


def test_monitor_offline_smoke():
    rc = main(["monitor", "--seed", "1", "--window", "all"])
    assert rc in (0, 1)  # decayed or healthy are both valid exits


def test_market_data_audit_store_carries_universe(tmp_path):
    """B4 must see universe records, not just price — the CLI market is price-only."""
    import pandas as pd

    from src.checklist import survivorship_check
    from src.cli import _market_data
    from src.config import Config
    from src.data.point_in_time_loader import from_url

    url = f"sqlite:///{tmp_path / 'pit.db'}"
    store = from_url(url)
    store.upsert(
        pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "valid_from": pd.to_datetime(["2019-01-02", "2019-01-03"]),
                "valid_to": pd.to_datetime(["2019-01-03", "2019-01-04"]),
                "close": [10.0, 11.0],
                "record_type": "price",
            }
        )
    )
    store.upsert(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "AAA"],
                "valid_from": pd.to_datetime(["2015-01-05", "2015-01-05", "2019-01-02"]),
                "valid_to": pd.to_datetime(["2015-01-06", "2015-01-06", "2019-01-03"]),
                "record_type": "universe",
            }
        )
    )
    cfg = Config({"data": {"pit_database_url": url}, "research": {"universe": "all"}})
    mkt = _market_data(cfg, seed=1)
    assert mkt.audit_store is not None
    res = survivorship_check(mkt.audit_store, as_of="2015-01-05")
    assert res.passed
    assert res.meta["n_delisted_since"] == 1  # BBB alive in 2015, gone by 2019
