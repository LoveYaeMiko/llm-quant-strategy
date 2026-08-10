"""src/pool.py tests — Phase 8 factor-pool management.

Covers re-scoring (evaluate_pool), validation gates (filter_pool), AST-diversity
screening (diversify_pool), composite combination backtests and the decay
watchlist — all against the shared synthetic market.
"""

from __future__ import annotations

import json

import pytest

from src.factors.code_generator import eval_expression
from src.pool import (
    combination_backtest,
    diversify_pool,
    evaluate_pool,
    filter_pool,
    load_pool,
    monitor_watchlist,
    write_json,
)

GOOD = "TS_Return(Close, 5)"
GOOD_2 = "Rank(Close)"
GOOD_3 = "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
BAD = "Unknown_Op(Close, 5)"


def _entry(formula: str, **extra) -> dict:
    e = {"factor": {"formula": formula}, "metrics": {"ic": 0.02, "icir": 0.31}}
    e.update(extra)
    return e


def _high_val_entry(formula: str, ic: float, icir: float) -> dict:
    return {
        "factor": {"formula": formula},
        "metrics": {"ic": 0.01, "icir": 0.1},
        "val_metrics": {"ic": ic, "icir": icir, "sharpe": 1.2, "max_drawdown": -0.1},
    }


# ---------------------------------------------------------------------------
# evaluate_pool
# ---------------------------------------------------------------------------


def test_evaluate_pool_scores_valid_formula(fctx, forward):
    res = evaluate_pool(fctx, forward, [GOOD], n_trials=1)
    assert GOOD in res
    m = res[GOOD]
    assert {"ic", "rank_ic", "icir", "sharpe", "n_days"}.issubset(m)
    assert "error" not in m


def test_evaluate_pool_reports_bad_formula_without_raising(fctx, forward):
    res = evaluate_pool(fctx, forward, [BAD], n_trials=1)
    assert BAD in res
    assert "error" in res[BAD]


def test_evaluate_pool_skips_garbage_never_crashes(fctx, forward):
    res = evaluate_pool(fctx, forward, [BAD, "", None, GOOD], n_trials=1)
    assert "error" in res[BAD]
    assert "error" in res.get("", {})
    assert GOOD in res


# ---------------------------------------------------------------------------
# filter_pool
# ---------------------------------------------------------------------------


def test_filter_keeps_factors_clearing_gates():
    pool = [_entry(GOOD), _entry(GOOD_2)]
    metrics = {GOOD: {"ic": 0.03, "icir": 0.45}, GOOD_2: {"ic": 0.01, "icir": 0.2}}
    kept = filter_pool(pool, metrics, min_ic=0.02, min_icir=0.30)
    assert [e["factor"]["formula"] for e in kept] == [GOOD]
    assert kept[0]["val_metrics"]["icir"] == 0.45


def test_filter_drops_error_and_missing_formulas():
    pool = [_entry(GOOD), _entry(GOOD_2), _entry(BAD)]
    metrics = {GOOD: {"ic": 0.03, "icir": 0.45}, BAD: {"error": "boom"}}
    kept = filter_pool(pool, metrics)
    assert [e["factor"]["formula"] for e in kept] == [GOOD]


def test_filter_gate_uses_or_style_minimums():
    pool = [_entry(GOOD), _entry(GOOD_2)]
    metrics = {GOOD: {"ic": 0.05, "icir": 0.10}, GOOD_2: {"ic": 0.01, "icir": 0.50}}
    kept = filter_pool(pool, metrics, min_ic=0.02, min_icir=0.30)
    # Both clear IC but fail ICIR (GOOD) or clear ICIR but fail IC (GOOD_2).
    assert kept == []


def test_filter_does_not_mutate_input_pool():
    pool = [_entry(GOOD)]
    metrics = {GOOD: {"ic": 0.03, "icir": 0.45}}
    filter_pool(pool, metrics)
    assert "val_metrics" not in pool[0]


# ---------------------------------------------------------------------------
# diversify_pool
# ---------------------------------------------------------------------------


def test_diversify_drops_identical_duplicates():
    pool = [_entry(GOOD), _entry(GOOD)]
    kept = diversify_pool(pool, min_distance=0.40)
    assert len(kept) == 1
    assert kept[0]["factor"]["formula"] == GOOD


def test_diversify_keeps_distinct_formulas_with_low_gate():
    pool = [_entry(GOOD), _entry(GOOD_2), _entry(GOOD_3)]
    kept = diversify_pool(pool, min_distance=0.01)
    assert len(kept) == 3


def test_diversify_sorts_by_val_ic_before_sweep():
    # Same operator, different lookback -> AST distance 0.5, so a 0.60 gate keeps
    # only one of them; order_by=val_ic decides which one wins the slot.
    low = _high_val_entry("TS_Return(Close, 5)", 0.025, 0.5)
    high = _high_val_entry("TS_Return(Close, 10)", 0.06, 1.2)
    pool = [low, high]
    kept = diversify_pool(pool, min_distance=0.60, order_by="ic")
    assert len(kept) == 1
    assert kept[0]["factor"]["formula"] == "TS_Return(Close, 10)"


def test_diversify_drops_unparsable_formulas():
    pool = [_entry(GOOD), _entry(BAD)]
    kept = diversify_pool(pool, min_distance=0.01)
    assert [e["factor"]["formula"] for e in kept] == [GOOD]


# ---------------------------------------------------------------------------
# combination_backtest
# ---------------------------------------------------------------------------


def test_combination_equal_returns_composite_and_per_factor(fctx, forward):
    pool = [_high_val_entry(GOOD, 0.03, 0.5), _high_val_entry(GOOD_2, 0.025, 0.4)]
    res = combination_backtest(fctx, forward, pool, weights="equal", n_trials=1)
    assert res["n_factors"] == 2
    assert res["weights"] == "equal"
    assert "sharpe" in res["composite"]
    assert set(res["per_factor"]) == {GOOD, GOOD_2}


def test_combination_icir_weights_are_normalized(fctx, forward):
    pool = [_high_val_entry(GOOD, 0.03, 0.6), _high_val_entry(GOOD_2, 0.02, 0.2)]
    res = combination_backtest(fctx, forward, pool, weights="icir", n_trials=1)
    assert res["n_factors"] == 2
    assert "sharpe" in res["composite"]


def test_combination_dynamic_weights_run(fctx, forward):
    pool = [_high_val_entry(GOOD, 0.03, 0.5), _high_val_entry(GOOD_2, 0.02, 0.4)]
    res = combination_backtest(fctx, forward, pool, weights="dynamic", n_trials=1)
    assert res["n_factors"] == 2
    assert "sharpe" in res["composite"]


def test_combination_empty_pool_returns_error(fctx, forward):
    res = combination_backtest(fctx, forward, [], weights="equal", n_trials=1)
    assert res["n_factors"] == 0
    assert "error" in res


def test_combination_skips_unscorable_formulas(fctx, forward):
    pool = [_high_val_entry(GOOD, 0.03, 0.5), _entry(BAD)]
    res = combination_backtest(fctx, forward, pool, weights="equal", n_trials=1)
    assert res["n_factors"] == 1
    assert set(res["per_factor"]) == {GOOD}


# ---------------------------------------------------------------------------
# monitor_watchlist
# ---------------------------------------------------------------------------


def test_watchlist_reports_structure(fctx, forward):
    pool = [_high_val_entry(GOOD, 0.03, 0.5)]
    res = monitor_watchlist(fctx, forward, pool, window_days=30, icir_threshold=0.30)
    r = res[GOOD]
    assert {"recent_icir", "decayed", "summary"}.issubset(r)
    assert isinstance(r["decayed"], bool)


def test_watchlist_reports_error_for_bad_formula(fctx, forward):
    pool = [_entry(BAD)]
    res = monitor_watchlist(fctx, forward, pool, window_days=30)
    assert "error" in res[BAD]


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------


def test_write_json_and_load_pool_roundtrip(tmp_path):
    path = tmp_path / "pool.json"
    write_json(path, [{"factor": {"formula": GOOD}, "metrics": {}}])
    assert load_pool(path)[0]["factor"]["formula"] == GOOD


def test_load_pool_accepts_factors_wrapper(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"factors": [{"factor": {"formula": GOOD}}]}), encoding="utf-8")
    assert load_pool(path)[0]["factor"]["formula"] == GOOD
