"""Offline unit tests for ``scripts/bias_stress_test.py`` (no DB, no network).

Every pure helper of the survivorship-bias stress test is exercised here: the
drag/break-even algebra, the hazard compounding, the missing-name and snapshot
plumbing, the de-bias rule, the metric normalisation and the exact verdict-key
contract consumed by the alpha-evolution gate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.bias_stress_test as bst


# --------------------------------------------------------------------------- #
# drag algebra
# --------------------------------------------------------------------------- #
def test_trading_year_fraction():
    assert bst.trading_year_fraction(252) == pytest.approx(1.0)
    assert bst.trading_year_fraction(40) == pytest.approx(40.0 / 252.0)
    assert bst.trading_year_fraction(0) == 0.0
    with pytest.raises(ValueError):
        bst.trading_year_fraction(10, trading_days=0)


def test_entries_per_year_annualises():
    assert bst.entries_per_year(53, 82) == pytest.approx(53 * 252 / 82)
    assert bst.entries_per_year(252, 252) == pytest.approx(252.0)
    assert bst.entries_per_year(0, 100) == 0.0
    with pytest.raises(ValueError):
        bst.entries_per_year(10, 0)


def test_annual_drag_is_signed_and_scales_with_slot_weight():
    # 200 entries/yr, 0.1% doom probability, 6 slots, -50% loss
    frac = bst.annual_drag_frac(200, 0.001, k=6, loss=-0.5)
    assert frac == pytest.approx(-200 * 0.001 * (1 / 6) * 0.5)
    assert bst.annual_drag_pp(200, 0.001, k=6, loss=-0.5) == pytest.approx(frac * 100)
    # a bigger slot count dilutes the per-entry hit
    assert abs(bst.annual_drag_frac(200, 0.001, k=12, loss=-0.5)) == pytest.approx(
        abs(frac) / 2
    )
    with pytest.raises(ValueError):
        bst.annual_drag_frac(200, 0.001, k=0, loss=-0.5)


def test_drag_grid_shape_and_values():
    grid = bst.drag_grid(200, k=6, xs=(0.001, 0.01), losses=(-0.3, -0.5))
    assert grid["x_grid"] == [0.001, 0.01]
    assert grid["loss_grid"] == [-0.3, -0.5]
    assert set(grid["drag_pp"]) == {"-0.30", "-0.50"}
    assert grid["drag_pp"]["-0.50"]["0.001"] == pytest.approx(
        -200 * 0.001 / 6 * 0.5 * 100, abs=1e-4
    )
    assert grid["drag_pp"]["-0.30"]["0.01"] == pytest.approx(-200 * 0.01 / 6 * 0.3 * 100, abs=1e-4)
    # loss scales linearly: -0.8 is 8/3 of -0.3
    full = bst.drag_grid(200, k=6, xs=(0.01,), losses=(-0.3, -0.8))
    assert full["drag_pp"]["-0.80"]["0.01"] / full["drag_pp"]["-0.30"]["0.01"] == pytest.approx(
        0.8 / 0.3, abs=1e-4
    )


def test_breakeven_x_inverts_the_drag():
    epy, k, loss, thr = 200.0, 6, -0.5, 0.024
    x = bst.breakeven_x(epy, k=k, loss=loss, threshold_frac=thr)
    assert x == pytest.approx(0.024 / (200 * (1 / 6) * 0.5))
    # plugging the break-even x back in reproduces the threshold magnitude
    assert abs(bst.annual_drag_frac(epy, x, k=k, loss=loss)) == pytest.approx(thr)
    # a wider loss reaches the threshold with a smaller x
    x_big = bst.breakeven_x(epy, k=k, loss=-0.8, threshold_frac=thr)
    assert x_big < x
    # undefined instead of invented when no drag is possible
    assert bst.breakeven_x(0.0, k=k, loss=loss, threshold_frac=thr) is None
    assert bst.breakeven_x(epy, k=k, loss=0.0, threshold_frac=thr) is None


# --------------------------------------------------------------------------- #
# hazard compounding
# --------------------------------------------------------------------------- #
def test_annual_disappearance_rate_compounds_geometrically():
    assert bst.annual_disappearance_rate(100, 10, 1.0) == pytest.approx(0.10)
    assert bst.annual_disappearance_rate(100, 0, 5.0) == pytest.approx(0.0)
    # 50% gone over two years -> 1 - 0.5**0.5 per year
    assert bst.annual_disappearance_rate(100, 50, 2.0) == pytest.approx(1 - 0.5 ** 0.5)
    # clamped when more names left than existed at the snapshot
    assert bst.annual_disappearance_rate(100, 250, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bst.annual_disappearance_rate(0, 0, 1.0)
    with pytest.raises(ValueError):
        bst.annual_disappearance_rate(100, 1, 0.0)


def test_rule_of_three_upper_rate():
    assert bst.rule_of_three_upper_rate(490, 11.5866) == pytest.approx(3 / (490 * 11.5866))
    assert bst.rule_of_three_upper_rate(0, 10) is None
    assert bst.rule_of_three_upper_rate(10, 0) is None
    # only valid for zero observed events
    assert bst.rule_of_three_upper_rate(490, 11.5866, events=2) is None


def test_x_upper_bound_from_hazard_scales_with_exposure():
    x = bst.x_upper_bound_from_hazard(0.008, hold_days=40, trading_days=252)
    assert x == pytest.approx(0.008 * 40 / 252)
    # double the holding window -> double the per-entry doom probability
    assert bst.x_upper_bound_from_hazard(0.008, hold_days=80) == pytest.approx(2 * x)
    # pick_probability < 1 only lowers the bound
    assert bst.x_upper_bound_from_hazard(0.008, hold_days=40, pick_prob=0.2) == pytest.approx(
        0.2 * x
    )
    assert bst.x_upper_bound_from_hazard(0.0, hold_days=40) == 0.0


# --------------------------------------------------------------------------- #
# universe snapshots / missing names
# --------------------------------------------------------------------------- #
def _universe_frame(rows):
    return pd.DataFrame(rows, columns=["symbol", "valid_from", "valid_to", "name"])


def test_snapshot_sets_groups_by_valid_from():
    df = _universe_frame([
        ("000001.SZ", "2015-01-05", "2015-01-06", "甲"),
        ("000002.SZ", "2015-01-05", "2015-01-06", "乙"),
        ("000001.SZ", "2026-08-07", "2026-08-08", "甲"),
    ])
    snaps = bst.snapshot_sets(df)
    assert set(snaps) == {"2015-01-05", "2026-08-07"}
    assert snaps["2015-01-05"] == {"000001.SZ", "000002.SZ"}
    assert snaps["2026-08-07"] == {"000001.SZ"}
    assert bst.snapshot_sets(pd.DataFrame()) == {}


def test_snapshots_covering_respects_validity_intervals():
    df = _universe_frame([
        ("A", "2015-01-05", "2015-01-06", "a"),      # expired long before
        ("B", "2025-10-01", None, "b"),              # open-ended -> covers
        ("C", "2026-01-01", "2026-01-02", "c"),      # born after the window
        ("D", "2025-01-01", "2025-09-15", "d"),      # overlaps window start
    ])
    assert bst.snapshots_covering(df, "2025-09-01", "2025-12-31") == ["2025-01-01", "2025-10-01"]
    assert bst.snapshots_covering(df, "2016-01-01", "2016-12-31") == []
    assert bst.snapshots_covering(pd.DataFrame(), "2025-09-01", "2025-12-31") == []


def test_missing_names_returns_universe_rows_without_price_bars():
    df = _universe_frame([
        ("A", "2015-01-05", "2015-01-06", "a"),
        ("B", "2015-01-05", "2015-01-06", "b退"),
        ("C", "2026-08-07", "2026-08-08", "c"),
    ])
    miss = bst.missing_names(df, ["A", "C"])
    assert list(miss["symbol"]) == ["B"]
    assert bst.missing_names(df, ["A", "B", "C"]).empty
    assert bst.missing_names(pd.DataFrame(), ["A"]).empty


def test_missing_summary_counts_and_samples():
    df = _universe_frame([
        ("A", "2015-01-05", "2015-01-06", "甲"),
        ("B", "2015-01-05", "2015-01-06", "乙退"),
        ("C", "2026-08-07", "2026-08-08", "丙"),
        ("D", "2026-08-07", "2026-08-08", "丁"),
    ])
    out = bst.missing_summary(df, ["A", "C", "D"], ["A", "B"], "2025-09-01", "2025-12-31")
    assert out["n_universe_symbols"] == 4
    assert out["n_universe_with_price_bars"] == 3
    assert out["n_missing_zero_price_bars"] == 1
    assert out["share_of_universe_missing"] == pytest.approx(0.25)
    assert out["n_missing_in_d_universe"] == 1
    assert out["n_missing_in_window_snapshots"] == 0
    assert out["window_snapshot_dates"] == []
    assert out["snapshot_dates"] == ["2015-01-05", "2026-08-07"]
    assert out["snapshot_sizes"] == {"2015-01-05": 2, "2026-08-07": 2}
    assert out["n_missing_absent_from_newest_snapshot"] == 1
    assert out["n_missing_present_in_newest_snapshot"] == 0
    assert out["earliest_missing_valid_from"] == "2015-01-05"
    assert out["latest_missing_valid_from"] == "2015-01-05"
    assert out["missing_symbols"] == ["B"]
    assert out["n_missing_name_samples_with_delist_tag"] == 1
    assert out["missing_name_samples"] == [
        {"symbol": "B", "valid_from": "2015-01-05", "name": "乙退"}
    ]


def test_missing_summary_handles_empty_universe():
    out = bst.missing_summary(_universe_frame([]), [], [], "2025-09-01", "2025-12-31")
    assert out["n_universe_symbols"] == 0
    assert out["n_missing_zero_price_bars"] == 0
    assert out["share_of_universe_missing"] is None
    assert out["earliest_missing_valid_from"] is None


def test_hazard_summary_measures_both_segments():
    df = _universe_frame([
        ("A", "2015-01-01", "2015-01-02", "甲"),
        ("B", "2015-01-01", "2015-01-02", "乙退"),
        ("C", "2015-01-01", "2015-01-02", "丙"),
        ("A", "2026-01-01", "2026-01-02", "甲"),
        ("C", "2026-01-01", "2026-01-02", "丙"),
        ("D", "2026-01-01", "2026-01-02", "丁"),
    ])
    out = bst.hazard_summary(df, ["A", "B", "C"])
    assert out["measured"] is True
    assert out["oldest_snapshot"] == "2015-01-01"
    assert out["newest_snapshot"] == "2026-01-01"
    assert out["full_universe"]["n_present_at_oldest"] == 3
    assert out["full_universe"]["n_absent_from_newest"] == 1
    assert out["full_universe"]["annual_disappearance_rate"] == pytest.approx(
        bst.annual_disappearance_rate(3, 1, out["years"]), abs=1e-6
    )
    # the D segment here is A/B/C; only B disappeared
    assert out["d_universe"]["n_present_at_oldest_and_in_d"] == 3
    assert out["d_universe"]["n_absent_from_newest"] == 1
    assert out["d_universe"]["rule_of_three_upper_rate"] is None  # events > 0
    assert [g["symbol"] for g in out["gone_name_samples"]] == ["B"]


def test_hazard_summary_zero_events_uses_rule_of_three():
    df = _universe_frame([
        ("A", "2015-01-01", "2015-01-02", "甲"),
        ("B", "2015-01-01", "2015-01-02", "乙"),
        ("A", "2026-01-01", "2026-01-02", "甲"),
        ("B", "2026-01-01", "2026-01-02", "乙"),
    ])
    out = bst.hazard_summary(df, ["A", "B"])
    assert out["d_universe"]["n_absent_from_newest"] == 0
    assert out["d_universe"]["annual_disappearance_rate"] == 0.0
    assert out["d_universe"]["rule_of_three_upper_rate"] == pytest.approx(
        3 / (2 * out["years"]), abs=1e-6
    )


def test_hazard_summary_needs_two_snapshots():
    df = _universe_frame([("A", "2026-08-07", "2026-08-08", "甲")])
    out = bst.hazard_summary(df, ["A"])
    assert out["measured"] is False
    assert "reason" in out


# --------------------------------------------------------------------------- #
# de-bias rule
# --------------------------------------------------------------------------- #
def test_risk_segment_penny_and_bottom_decile_liquidity():
    px = pd.Series({"A": 10.0, "B": 2.5, "C": 5.0, "D": 8.0, "E": np.nan})
    amt = pd.Series({"A": 1000.0, "B": 900.0, "C": 100.0, "D": 200.0, "E": np.nan})
    out = bst.risk_segment(px, amt, price_floor=3.0, decile=0.10)
    # linear 10th percentile of [100, 200, 900, 1000] = 130
    assert out["amount_decile_threshold_cny"] == pytest.approx(130.0)
    assert out["low_price_symbols"] == ["B"]
    assert out["low_amount_symbols"] == ["C", "E"]  # NaN amount = least liquid
    assert out["excluded_symbols"] == ["B", "C", "E"]
    assert out["n_symbols"] == 5
    assert out["n_excluded"] == 3
    assert out["share_excluded"] == pytest.approx(0.6)
    assert out["n_missing_amount_data"] == 1
    assert out["share_low_price"] == pytest.approx(0.2)


def test_risk_segment_is_deterministic_and_union_based():
    px = pd.Series({"A": 2.0, "B": 20.0})
    amt = pd.Series({"A": 500.0, "B": 500.0})
    out = bst.risk_segment(px, amt, price_floor=3.0, decile=0.10)
    # equal amounts -> both sit at the threshold, so both are "bottom decile"
    assert out["low_price_symbols"] == ["A"]
    assert out["low_amount_symbols"] == ["A", "B"]
    assert out["excluded_symbols"] == ["A", "B"]
    assert out["share_bottom_decile_amount"] == pytest.approx(1.0)
    again = bst.risk_segment(px, amt, price_floor=3.0, decile=0.10)
    assert out == again


def test_risk_segment_empty_inputs():
    out = bst.risk_segment(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n_symbols"] == 0
    assert out["excluded_symbols"] == []
    assert out["share_excluded"] is None
    assert out["amount_decile_threshold_cny"] is None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_block_normalises_and_counts_entries():
    equity = {
        "total_return": 0.05, "annualized_return": 0.12, "sharpe": 1.5,
        "max_drawdown": 0.08, "n_days": 100, "n_fills": 3,
        "total_commission": 5.5, "latest": 105000.0, "fills_by_source": {"close": 3},
    }
    fills = pd.DataFrame({"shares": [100.0, -50.0, 200.0]})
    out = bst.metrics_block(equity, fills)
    assert out["cumulative_return"] == pytest.approx(0.05)
    assert out["n_entries"] == 2
    assert out["n_exits"] == 1
    assert out["n_fills"] == 3
    assert out["final_equity"] == pytest.approx(105000.0)
    assert out["fills_by_source"] == {"close": 3}


def test_metrics_block_tolerates_missing_values():
    out = bst.metrics_block({"sharpe": float("nan"), "total_return": None})
    assert out["sharpe"] is None
    assert out["cumulative_return"] is None
    assert out["n_entries"] is None


def test_metric_delta_sign_and_scale():
    base = {"annualized_return": 0.05, "cumulative_return": 0.016, "sharpe": 0.4,
            "max_drawdown": 0.05, "n_fills": 106, "n_entries": 53}
    debi = {"annualized_return": 0.09, "cumulative_return": 0.03, "sharpe": 0.7,
            "max_drawdown": 0.04, "n_fills": 98, "n_entries": 49}
    d = bst.metric_delta(base, debi)
    assert d["annualized_return_pp"] == pytest.approx(4.0)
    assert d["cumulative_return_pp"] == pytest.approx(1.4)
    assert d["sharpe"] == pytest.approx(0.3)
    assert d["max_drawdown_pp"] == pytest.approx(-1.0)
    assert d["n_fills"] == -8
    assert d["n_entries"] == -4


def test_metric_delta_returns_none_for_unmeasured_keys():
    d = bst.metric_delta({"annualized_return": None}, {"annualized_return": 0.1})
    assert d["annualized_return_pp"] is None
    assert d["sharpe"] is None


# --------------------------------------------------------------------------- #
# verdict contract
# --------------------------------------------------------------------------- #
def test_verdict_block_exact_keys_and_threshold_logic():
    v = bst.verdict_block(
        target_alpha=0.08, threshold_ratio=0.3,
        x_upper_bound=0.0012667, epy=202.0, k=6, loss_mid=-0.5,
        h_annual=0.008, breakeven=0.001424,
    )
    assert set(v) == {
        "target_alpha_annual", "threshold_ratio", "threshold_pp", "x_upper_bound",
        "drag_at_upper_bound_pp", "bias_vs_alpha_ratio", "bias_blocking_evolution",
        "verdict_text",
    }
    assert v["threshold_pp"] == pytest.approx(2.4)
    assert v["drag_at_upper_bound_pp"] == pytest.approx(202 * 0.0012667 / 6 * 0.5 * 100, rel=1e-4)
    assert v["bias_vs_alpha_ratio"] == pytest.approx(v["drag_at_upper_bound_pp"] / 8.0, abs=1e-4)
    assert v["bias_blocking_evolution"] is (v["drag_at_upper_bound_pp"] > 2.4)
    assert "NOT BLOCKING" in v["verdict_text"]


def test_verdict_block_flags_blocking_when_drag_exceeds_threshold():
    v = bst.verdict_block(
        target_alpha=0.08, threshold_ratio=0.3, x_upper_bound=0.008, epy=202.0,
        k=6, loss_mid=-0.5, h_annual=0.05, breakeven=0.001424,
    )
    assert v["drag_at_upper_bound_pp"] > 2.4
    assert v["bias_blocking_evolution"] is True
    assert v["verdict_text"].startswith("BLOCKING")
    assert v["bias_vs_alpha_ratio"] > 0.3


def test_verdict_sentence_mentions_break_even_and_hazard():
    txt = bst.verdict_sentence(
        blocking=False, drag_pp=2.13, threshold_pp=2.4, ratio=0.27,
        x_upper_bound=0.0012667, h_annual=0.008, epy=202.0, breakeven=0.001424,
    )
    assert "0.80%/yr" in txt
    assert "0.1424%" in txt
    assert "2.4pp threshold" in txt
    # the margin to the break-even hazard is stated when it is known
    txt3 = bst.verdict_sentence(
        blocking=False, drag_pp=1.75, threshold_pp=2.4, ratio=0.22,
        x_upper_bound=0.0012667, h_annual=0.00798, epy=166.0, breakeven=0.001735,
        breakeven_h=0.01093,
    )
    assert "73% of the break-even hazard" in txt3
    assert "1.09%/yr" in txt3
    # unmeasured inputs must not fabricate numbers
    txt2 = bst.verdict_sentence(
        blocking=False, drag_pp=0.0, threshold_pp=2.4, ratio=0.0,
        x_upper_bound=0.0, h_annual=None, epy=0.0, breakeven=None,
    )
    assert "n/a" in txt2


def test_verdict_block_threads_break_even_hazard_into_the_sentence():
    v = bst.verdict_block(
        target_alpha=0.08, threshold_ratio=0.3, x_upper_bound=0.0012667, epy=166.0,
        k=6, loss_mid=-0.5, h_annual=0.00798, breakeven=0.001735, breakeven_h=0.01093,
    )
    assert "break-even hazard" in v["verdict_text"]
    assert v["bias_blocking_evolution"] is False


def test_float_and_int_coercion_helpers():
    assert bst._f("1.5") == pytest.approx(1.5)
    assert bst._f(float("nan")) is None
    assert bst._f(float("inf")) is None
    assert bst._f(None) is None
    assert bst._f("nope") is None
    assert bst._i(3.7) == 3
    assert bst._i(None) is None
    assert bst._i("x") is None


def test_module_constants_match_the_d_book_config():
    # D_5W shape: 6 slots, 40-day max hold, daily rebalance
    assert bst.K_SLOTS == 6
    assert bst.MAX_HOLD_DAYS == 40
    assert bst.TRADING_DAYS == 252
    assert bst.TARGET_ALPHA_DEFAULT == pytest.approx(0.08)
    assert bst.THRESHOLD_RATIO == pytest.approx(0.3)
    assert bst.THRESHOLD_RATIO * bst.TARGET_ALPHA_DEFAULT * 100 == pytest.approx(2.4)
    assert bst.LOSS_MID == -0.5
    assert set(bst.X_GRID) == {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
    assert set(bst.LOSS_GRID) == {-0.30, -0.50, -0.80}


def test_pure_helpers_do_not_touch_network_or_db(monkeypatch):
    """The pure half of the module must never open a socket or a database."""
    import socket

    def _boom(*_a, **_k):  # pragma: no cover - only fires on a regression
        raise AssertionError("pure helper attempted network/DB access")

    monkeypatch.setattr(socket, "socket", _boom)
    assert bst.annual_drag_pp(100, 0.01, k=6, loss=-0.5) == pytest.approx(
        -100 * 0.01 / 6 * 0.5 * 100
    )
    assert math.isfinite(bst.annual_disappearance_rate(10, 1, 1.0))
    assert bst.risk_segment(pd.Series({"A": 1.0}), pd.Series({"A": 1.0}))["n_excluded"] == 1
