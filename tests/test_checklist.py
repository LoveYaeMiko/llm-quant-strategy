"""Blueprint verification checklist tests."""

from __future__ import annotations

import pandas as pd

from src.checklist import (
    adjustment_consistency_check,
    cost_check,
    data_freshness_check,
    diversity_check,
    fincad_check,
    no_future_leak_check,
    pit_check,
    run_all,
    survivorship_check,
)
from src.config import Config
from src.cost_tracker import CostTracker
from src.data.point_in_time_loader import PointInTimeStore
from src.data.synthetic import make_synthetic_market


def test_pit_check_passes():
    m = make_synthetic_market(symbols=8, days=80, seed=1)
    res = pit_check(m.pit_store)
    assert res.passed
    assert res.meta["future_facts_leaked"] == 0


def test_fincad_check_passes():
    res = fincad_check()
    assert res.passed
    assert res.meta["reduction"] > 0.5


def test_diversity_check():
    formulas = [
        "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))",
        "Neg(TS_ZScore(Close, 20))",
        "Inv(TS_Std(Close, 30))",
    ]
    res = diversity_check(formulas, min_distance=0.4)
    assert res.passed


def test_diversity_check_fails_for_copies():
    f = "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    res = diversity_check([f, f, f], min_distance=0.4)
    assert res.passed is False


def test_cost_check_under_budget():
    res = cost_check(budget=500.0)
    assert res.passed


def test_run_all_aggregates(market):
    checks = run_all(store=market.pit_store, config=None)
    names = {c.name for c in checks}
    assert names == {"pit", "fincad", "diversity", "cost"}
    assert all(c.passed for c in checks)


# ---------------------------------------------------------------------------
# B1-B5 real-data checks
# ---------------------------------------------------------------------------


def test_no_future_leak_check_passes(market):
    res = no_future_leak_check(market.pit_store, boundary="2019-06-30")
    assert res.passed
    assert res.meta["future_facts_leaked"] == 0


def _price_store():
    store = PointInTimeStore()
    dates = pd.bdate_range("2024-01-01", periods=20)
    rows = [
        {
            "symbol": "AAA", "valid_from": d, "valid_to": d + pd.Timedelta("1D"),
            "close": 10.0, "raw_close": 10.0, "adjust_factor": 1.0, "record_type": "price",
        }
        for d in dates
    ]
    store.upsert(pd.DataFrame(rows))
    return store


def test_adjustment_consistency_passes_on_clean_data():
    res = adjustment_consistency_check(_price_store(), sample=10, price_limit_band=0.30)
    assert res.passed
    assert res.meta["days_audited"] == 20


def test_adjustment_consistency_fails_on_price_spike():
    store = _price_store()
    # inject a +50% adjusted jump far beyond the ±30% band
    rec = store.records.copy()
    rec.loc[rec["valid_from"] == pd.Timestamp("2024-01-10"), "close"] = 15.0
    store.upsert(rec)
    res = adjustment_consistency_check(store, sample=10, price_limit_band=0.30)
    assert res.passed is False
    assert res.meta["band_breaks"] >= 1


def test_survivorship_check_counts_delisted():
    store = PointInTimeStore()
    store.upsert(
        pd.DataFrame(
            [
                {"symbol": "AAA", "valid_from": "2015-01-05", "valid_to": "2015-01-06", "record_type": "universe"},
                {"symbol": "BBB", "valid_from": "2015-01-05", "valid_to": "2015-01-06", "record_type": "universe"},
                {"symbol": "AAA", "valid_from": "2024-06-28", "valid_to": "2024-06-29", "record_type": "universe"},
            ]
        )
    )
    res = survivorship_check(store, as_of="2015-01-05")
    assert res.passed
    assert res.meta["n_delisted_since"] == 1


def test_data_freshness_check():
    store = _price_store()  # 20 business days ending ~2024-01-29
    res = data_freshness_check(store, as_of="2024-02-01", max_staleness_days=7, min_coverage=0.5)
    assert res.passed
    stale = data_freshness_check(store, as_of="2024-04-01", max_staleness_days=7, min_coverage=0.5)
    assert stale.passed is False


def test_run_all_real_data_extends_checks():
    store = _price_store()
    cfg = Config(
        {
            "data": {"real_data": True, "checks": {"survivorship_date": "2024-01-05"}},
            "research": {"train_end": "2024-01-20"},
        }
    )
    checks = run_all(store=store, config=cfg, real_data_audit=True)
    names = {c.name for c in checks}
    assert names == {
        "pit", "fincad", "diversity", "cost",
        "no_future_leak", "adjustment_consistency", "survivorship", "data_freshness",
    }


def test_run_all_research_loop_skips_real_data_audit():
    """Research loops (mine/backtest/evolve/monitor) audit a window-sliced,
    universe-bounded store — B5 would always look stale there and B4's universe
    snapshots are out of scope. Even with data.real_data=true, they must run
    only the four standing checks; only verify opts into the B1-B5 audit."""
    store = _price_store()
    cfg = Config(
        {
            "data": {"real_data": True, "checks": {"survivorship_date": "2024-01-05"}},
            "research": {"train_end": "2024-01-20"},
        }
    )
    checks = run_all(store=store, config=cfg, real_data_audit=False)
    assert {c.name for c in checks} == {"pit", "fincad", "diversity", "cost"}


def test_run_all_real_data_requires_store():
    cfg = Config({"data": {"real_data": True}})
    try:
        run_all(store=None, config=cfg, real_data_audit=True)
        assert False, "expected ValueError"
    except ValueError:
        pass
