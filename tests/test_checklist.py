"""Blueprint verification checklist tests."""

from __future__ import annotations

import pandas as pd

from src.checklist import (
    cost_check,
    diversity_check,
    fincad_check,
    pit_check,
    run_all,
)
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
