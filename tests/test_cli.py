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


def test_verify_offline(capsys):
    # Smoke test: verify runs and prints the checklist. Its exit code reflects
    # the store's data state (with data.real_data=true, B5 freshness depends on
    # --mode and the backfill horizon), so any clean exit is acceptable.
    rc = main(["verify", "--seed", "1"])
    assert rc in (0, 1)
    assert "blueprint verification checklist" in capsys.readouterr().out


def test_forward_returns_do_not_fill_across_suspension_gaps():
    # Regression for the Phase 8.1 systematic `reject_high_risk`: forward
    # returns built with the default fill_method='pad' forward-fill close across
    # a suspension/resumption gap and fabricate a multi-hundred-percent "return"
    # on the day a name resumes (up to +1970% on real data). Those outliers
    # dominate return-based Sharpe/max-drawdown while rank-IC stays intact, so
    # every factor was rejected on a metric that had nothing to do with signal
    # quality. The fix computes returns only between consecutive trading days.
    import pandas as pd

    from src.cli import _market_from_records

    dates = pd.bdate_range("2020-01-01", periods=8)
    # A: trades days 0-2, suspended days 3-5, resumes day 6 with a huge jump.
    # With pad the day-5->day-6 forward return would be (500/102 - 1) ~ +390%.
    # B: dense, normal drift.
    rows = []
    for i, (sym, closes) in enumerate(
        {
            "A": [100.0, 101.0, 102.0, None, None, None, 500.0, 505.0],
            "B": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5],
        }.items()
    ):
        for d, c in zip(dates, closes):
            if c is None:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "valid_from": d,
                    "valid_to": d + pd.Timedelta(days=1),
                    "open": c, "high": c, "low": c, "close": c, "volume": 1000.0,
                }
            )
    market = _market_from_records(pd.DataFrame(rows))

    fwd = market.forward_returns
    assert fwd.index.names == ["date", "symbol"]
    # No forward-return row across the suspension gap for A: the day-5 -> day-6
    # "return" must be absent, not a fabricated +390% (the pad behaviour).
    a_fwd = fwd.xs("A", level="symbol")
    assert dates[5] not in a_fwd.index
    assert a_fwd.max() < 0.5
    # B stays dense and correct (roughly +0.5% per day).
    b = fwd.xs("B", level="symbol")
    assert abs(b.dropna().max()) < 0.2


def test_verify_backfill_mode_offline(capsys):
    # --mode backfill is the B5 historical-run mode: it must parse and wire
    # freshness_as_of=project.end_date through run_all without crashing, and
    # still print the checklist.
    rc = main(["verify", "--mode", "backfill", "--seed", "1"])
    assert rc in (0, 1)
    assert "blueprint verification checklist" in capsys.readouterr().out


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
