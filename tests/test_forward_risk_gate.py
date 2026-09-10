"""Forward risk-gate tests (audit items 1.1-1.3, 2).

These cover the arithmetic that decides whether a forward window passes. The
guiding rule: an UNMEASURED gate must fail, never silently pass.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.forward.risk_gate import (
    GateThresholds,
    availability,
    binom_two_sided_p,
    cost_deviation,
    data_freshness,
    evaluate_gate,
    fill_violations,
    paired_comparison,
    panel_universe_health,
    power_table,
    sharpe_standard_error,
    soft_metrics,
    symbol_minute_coverage,
    tracking_error,
    years_for_t,
)


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #
def test_power_matches_the_audit_numbers():
    # SE = sqrt(252/N) ⇒ N = 252 (t/SR)^2
    assert years_for_t(1.0, 2.0) == pytest.approx(4.0)
    assert years_for_t(1.0, 2.5) == pytest.approx(6.25, abs=0.05)
    assert years_for_t(0.5, 2.0) == pytest.approx(16.0)
    assert years_for_t(0.5, 2.5) == pytest.approx(25.0)
    assert power_table(1.0)["t_2_5"] == pytest.approx(6.2, abs=0.1)
    assert sharpe_standard_error(252) == pytest.approx(1.0)
    assert sharpe_standard_error(0) == float("inf")


def test_sign_bias_p_value():
    assert binom_two_sided_p(6, 6) == pytest.approx(1.0)
    assert binom_two_sided_p(10, 2) < 0.05
    assert binom_two_sided_p(0, 0) == 1.0
    assert binom_two_sided_p(20, 0) < 1e-4


# --------------------------------------------------------------------------- #
# tracking error
# --------------------------------------------------------------------------- #
def test_tracking_error_zero_when_identical():
    s = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2026-01-01", periods=3))
    out = tracking_error(s, s)
    assert out["n_days"] == 3 and out["mean_abs_pp"] == 0.0
    assert out["sign_bias_p"] == 1.0


def test_tracking_error_flags_systematic_bias():
    idx = pd.date_range("2026-01-01", periods=12)
    rec = pd.Series([0.0] * 12, index=idx)
    replay = pd.Series([0.001] * 12, index=idx)   # the book lags its own replay by 10bp = 0.1pp/day
    out = tracking_error(rec, replay)
    # diff = recorded - replay ⇒ the recorded pipeline is 0.1pp/day WORSE
    assert out["mean_signed_pp"] == pytest.approx(-0.1)
    assert out["n_pos"] == 0 and out["n_neg"] == 12
    assert out["sign_bias_p"] < 0.05
    assert out["worst_date"] == "2026-01-01"


def test_tracking_error_unmeasured_is_reported_as_zero_days():
    out = tracking_error(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n_days"] == 0


def test_tracking_error_excludes_live_fill_days():
    """A live-print fill is an external event the bar replay cannot reproduce."""
    idx = pd.date_range("2026-01-05", periods=4, freq="B")
    rec = pd.Series([0.010, 0.010, 0.010, 0.010], index=idx)
    rep = pd.Series([0.010, 0.000, 0.010, 0.010], index=idx)   # differs on day 2
    raw = tracking_error(rec, rep)
    assert raw["n_days"] == 4 and raw["mean_abs_pp"] > 0
    out = tracking_error(rec, rep, exclude_dates=[idx[1]])
    assert out["n_days"] == 3 and out["mean_abs_pp"] == 0.0
    assert out["n_excluded"] == 1 and out["excluded_mean_abs_pp"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
SPEC = {"commission_bps": 2.5, "transfer_fee_bps": 0.1, "stamp_tax_sell_bps": 5.0,
        "min_commission": 5.0}


def _fills(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", "side", "shares", "price",
                                       "commission", "notional", "source"])


def test_cost_deviation_zero_when_charged_equals_spec():
    notional = 20_000.0
    expected = max(5.0, notional * 2.5 / 10_000) + notional * 0.1 / 10_000 + notional * 5.0 / 10_000
    f = _fills([["2026-01-05", "600000.SH", "sell", -1000, 20.0, expected, notional, "auction"]])
    out = cost_deviation(f, SPEC)
    assert out["fee_deviation_pct"] == pytest.approx(0.0, abs=1e-6)
    assert out["n_fills"] == 1


def test_cost_deviation_detects_config_drift():
    notional = 20_000.0
    f = _fills([["2026-01-05", "600000.SH", "buy", 1000, 20.0, 5.0, notional, "close"]])
    out = cost_deviation(f, SPEC)              # spec implies 5 + 0.2 = 5.2
    assert out["fee_deviation_pct"] < 0        # charged less than the spec implies
    assert out["fee_deviation_pct"] == pytest.approx((5.0 - 5.2) / 5.2 * 100, abs=0.01)


def test_cost_price_integrity_uses_reference_prices():
    f = _fills([
        ["2026-01-05", "600000.SH", "buy", 1000, 20.10, 5.2, 20_100.0, "live"],
        ["2026-01-06", "600001.SH", "buy", 1000, 19.90, 5.2, 19_900.0, "live"],
    ])
    refs = {("2026-01-05", "600000.SH"): 20.0, ("2026-01-06", "600001.SH"): 20.0}
    out = cost_deviation(f, SPEC, reference_prices=refs, modeled_slippage_bps=2.0)
    # +50bp then -50bp → mean 0, mean absolute 50bp
    assert out["price_integrity_bps"] == pytest.approx(0.0, abs=1e-6)
    assert out["price_integrity_abs_bps"] == pytest.approx(50.0, abs=1e-6)
    assert out["n_price_checked"] == 2
    assert "market_impact_bps" in out["unmeasured"]


def test_replay_fills_are_not_price_checked():
    """A replayed stop is booked at min(stop, bar_close) — a model, not a print."""
    f = _fills([["2026-01-05", "600000.SH", "sell", -1000, 19.5, 15.4, 19_500.0, "replay"]])
    out = cost_deviation(f, SPEC, reference_prices={("2026-01-05", "600000.SH"): 20.0})
    assert out["n_price_checked"] == 0 and out["n_price_skipped"] == 1
    assert out["price_integrity_abs_bps"] is None


def test_cost_without_reference_prices_is_not_a_pass():
    f = _fills([["2026-01-05", "600000.SH", "buy", 1000, 20.0, 5.2, 20_000.0, "close"]])
    out = cost_deviation(f, SPEC)
    assert out["price_integrity_abs_bps"] is None
    gate = evaluate_gate({"cost": out, "tracking_error": {"n_days": 5, "mean_abs_pp": 0.0,
                                                          "sign_bias_p": 1.0},
                          "violations": {"total": 0}, "availability": {"availability": 1.0},
                          "data_freshness": {"lag_days": 0},
                          "symbol_coverage": {"min_coverage": 1.0}})
    assert not gate["hard"]["cost_price_integrity"]["ok"]
    assert "cost_price_integrity" in gate["failed"]


# --------------------------------------------------------------------------- #
# violations
# --------------------------------------------------------------------------- #
def _panel(days=3, symbols=("600000.SH", "600001.SH"), last=20.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=days, freq="B")
    return pd.DataFrame({s: np.linspace(last, last + 1, days) for s in symbols}, index=idx)


def test_clean_fills_have_no_violations():
    f = _fills([["2026-01-05", "600000.SH", "buy", 1000, 20.0, 5.2, 20_000.0, "close"],
                ["2026-01-06", "600000.SH", "sell", -1000, 20.5, 15.4, 20_500.0, "auction"]])
    out = fill_violations(f, _panel())
    assert out["total"] == 0, out


def test_same_day_round_trip_is_a_t_plus_1_violation():
    f = _fills([["2026-01-05", "600000.SH", "buy", 1000, 20.0, 5.2, 20_000.0, "close"],
                ["2026-01-05", "600000.SH", "sell", -1000, 20.1, 15.4, 20_100.0, "auction"]])
    out = fill_violations(f, _panel())
    assert out["t_plus_1"] == 1


def test_odd_lot_and_off_tick_are_flagged():
    f = _fills([["2026-01-05", "600000.SH", "buy", 150, 20.005, 5.2, 3_000.75, "close"]])
    out = fill_violations(f, _panel())
    assert out["lot_size"] == 1 and out["tick_size"] == 1


def test_fill_on_a_suspended_day_is_flagged():
    f = _fills([["2026-01-08", "600000.SH", "buy", 1000, 20.0, 5.2, 20_000.0, "close"]])
    out = fill_violations(f, _panel(days=3))          # panel ends 2026-01-07
    assert out["suspension"] == 1


def test_limit_locked_is_direction_aware():
    panel = _panel(days=2)
    panel.loc[panel.index[1], "600000.SH"] = 20.0 * 1.10   # +10% limit-up day
    f = _fills([["2026-01-06", "600000.SH", "buy", 1000, 22.0, 5.2, 22_000.0, "close"]])
    out = fill_violations(f, panel, limit_fn=lambda *a, **k: 0.10)
    assert out["limit_locked"] == 1
    # selling into the same limit-up is legal
    f2 = _fills([["2026-01-06", "600000.SH", "sell", -1000, 22.0, 15.4, 22_000.0, "auction"]])
    assert fill_violations(f2, panel, limit_fn=lambda *a, **k: 0.10)["limit_locked"] == 0


# --------------------------------------------------------------------------- #
# availability / freshness / coverage
# --------------------------------------------------------------------------- #
def test_availability_full_and_outage():
    day = pd.Timestamp("2026-01-05")
    stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                           day + pd.Timedelta(hours=15), freq="60s")
    full = availability(pd.DataFrame({"ts": stamps}), [day])
    assert full["availability"] == 1.0
    partial = availability(pd.DataFrame({"ts": stamps[:100]}), [day])
    assert partial["availability"] < 1.0 and partial["n_days_down"] == 1


def test_availability_is_unmeasured_without_any_heartbeat():
    day = pd.Timestamp("2026-01-05")
    out = availability(pd.DataFrame({"ts": []}), [day])
    assert out["availability"] is None and out["unmeasured"] is True
    gate = evaluate_gate({"availability": out})
    assert not gate["hard"]["availability"]["ok"]


def test_an_empty_window_is_unmeasured_not_perfect():
    """A window that has not started yet must not report full availability."""
    out = availability(pd.DataFrame({"ts": []}), [])
    assert out["availability"] is None and out["unmeasured"] is True
    cov = symbol_minute_coverage({"tail_vol": pd.DataFrame()}, pd.DataFrame(), "2026-09-11",
                                 "2027-03-11")
    assert cov["min_coverage"] is None and cov.get("unmeasured") is True
    gate = evaluate_gate({"availability": out, "symbol_coverage": cov})
    assert not gate["hard"]["availability"]["ok"]
    assert not gate["hard"]["symbol_minute_coverage"]["ok"]


def test_availability_counts_a_long_gap():
    day = pd.Timestamp("2026-01-05")
    stamps = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                                day + pd.Timedelta(hours=10, minutes=30), freq="60s"))
    stamps += list(pd.date_range(day + pd.Timedelta(hours=13),
                                 day + pd.Timedelta(hours=15), freq="60s"))
    out = availability(pd.DataFrame({"ts": stamps}), [day])
    assert 0.5 < out["availability"] < 0.9
    assert out["down_minutes"] > 60


def test_data_freshness():
    assert data_freshness("2026-01-05", "2026-01-06")["lag_days"] == 1
    assert data_freshness("2026-01-05", "2026-01-05")["lag_days"] == 0
    assert data_freshness("2026-01-01", "2026-01-06")["lag_days"] == 5


def test_symbol_coverage_ignores_pre_listing_days():
    idx = pd.date_range("2026-01-05", periods=4, freq="B")
    panel = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [np.nan, np.nan, 2.0, 2.1]}, index=idx)
    tail = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.1], "B": [np.nan, np.nan, 0.2, 0.2]}, index=idx)
    out = symbol_minute_coverage({"tail_vol": tail}, panel, idx[0], idx[-1], threshold=0.95)
    assert out["min_coverage"] == 1.0 and out["n_symbols"] == 2


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
def _bundle(**over) -> dict:
    base = {
        "prereg": {"ok": True, "rule_id": "d_forward_test",
                   "frozen_at": "2026-09-09T23:04:47", "window_match": True,
                   "frozen_before_window": True, "policy_sha256_match": True,
                   "code_commit_match": True, "issues": []},
        "tracking_error": {"n_days": 40, "mean_abs_pp": 0.05, "sign_bias_p": 0.4,
                           "n_pos": 20, "n_neg": 20},
        "cost": {"fee_deviation_pct": 1.0, "price_integrity_abs_bps": 0.4,
                 "price_integrity_max_bps": 0.8, "price_integrity_bps_max": 2.0,
                 "n_price_checked": 40, "n_price_skipped": 3,
                 "unmeasured": ["market_impact_bps"]},
        "violations": {"total": 0},
        "availability": {"availability": 0.995},
        "data_freshness": {"lag_days": 0},
        "symbol_coverage": {"min_coverage": 0.99},
        "universe": {"n_columns": 800, "n_with_price": 800, "n_warm_20": 800,
                     "n_warm_60": 800, "effective_ratio": 1.0},
        "soft": {"sharpe": 0.3, "max_drawdown": -0.05},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def test_gate_passes_on_a_healthy_bundle():
    gate = evaluate_gate(_bundle())
    assert gate["verdict"] == "pass" and gate["failed"] == []
    assert gate["soft"]["note"].startswith("no forward-window power")


@pytest.mark.parametrize("mutation,failed_key", [
    ({"tracking_error": {"mean_abs_pp": 0.5}}, "tracking_error_daily_pp"),
    ({"tracking_error": {"sign_bias_p": 0.001}}, "tracking_error_sign_bias"),
    ({"cost": {"fee_deviation_pct": 25.0}}, "cost_fee_deviation"),
    ({"cost": {"price_integrity_abs_bps": 40.0}}, "cost_price_integrity"),
    ({"violations": {"total": 1}}, "violations"),
    ({"availability": {"availability": 0.9}}, "availability"),
    ({"data_freshness": {"lag_days": 3}}, "data_freshness"),
    ({"symbol_coverage": {"min_coverage": 0.8}}, "symbol_minute_coverage"),
    ({"universe": {"effective_ratio": 0.38, "n_columns": 800, "n_warm_20": 301}},
     "effective_universe"),
])
def test_each_hard_gate_fails_independently(mutation, failed_key):
    gate = evaluate_gate(_bundle(**mutation))
    assert gate["verdict"] == "fail"
    assert failed_key in gate["failed"]


def test_unmeasured_metrics_fail_rather_than_pass():
    gate = evaluate_gate({})
    assert gate["verdict"] == "fail"
    for key in ("prereg_binding", "tracking_error_daily_pp", "cost_fee_deviation",
                "violations", "availability", "data_freshness",
                "symbol_minute_coverage", "effective_universe"):
        assert key in gate["failed"]


def test_prereg_binding_is_a_hard_gate():
    """An evaluation that is not bound to a frozen record is not evidence."""
    ok = evaluate_gate(_bundle())
    assert ok["verdict"] == "pass" and ok["hard"]["prereg_binding"]["ok"] is True
    for mutation in (
        {"prereg": {"ok": False, "issues": ["no verified pre-registration"]}},
        {"prereg": {"ok": False, "policy_sha256_match": False,
                    "issues": ["the live policy fingerprint does not match"]}},
        {"prereg": {"ok": False, "code_commit_match": False,
                    "issues": ["code drifted since the freeze"]}},
    ):
        gate = evaluate_gate(_bundle(**mutation))
        assert gate["verdict"] == "fail"
        assert "prereg_binding" in gate["failed"]


def test_bool_hard_gates_are_not_silently_dropped():
    """``failed`` used to filter on ``isinstance(v, Mapping)`` — a plain bool
    hard gate could never fail. Both shapes must gate now."""
    from src.forward.risk_gate import evaluate_gate as ev

    th = GateThresholds()
    metrics = _bundle()
    gate = ev(metrics, th)
    assert gate["verdict"] == "pass"
    # inject a bool gate by monkeypatching the shape through a tiny subclass-free
    # path: evaluate a bundle whose prereg entry is a bare bool
    metrics_bad = dict(metrics, prereg=False)
    assert ev(metrics_bad, th)["verdict"] == "fail"


def test_unknown_threshold_key_is_refused():
    """A typo must raise, not silently fall back to the built-in default."""

    class _Cfg:
        def get(self, path, default=None):
            if path == "forward.risk_gate.hard":
                return {"tracking_eror_daily_pp_max": 0.5}   # typo
            return default

    with pytest.raises(ValueError) as exc:
        GateThresholds.from_config(_Cfg())
    assert "tracking_eror_daily_pp_max" in str(exc.value)


def test_panel_universe_health_counts_warm_names():
    """A panel that 'knows' 4 names but only 3 have history is 75% effective.

    Regression for the 2026-09-10 finding: the D track is declared as 800 names
    while the 2026 price panel carried 301 — a shrunken cross-section must be
    measured, not assumed away.
    """
    idx = pd.date_range("2026-06-01", periods=60, freq="B")
    panel = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 3.0, "D": np.nan}, index=idx)
    panel.loc[idx[-1], "D"] = 4.0            # one bar only — not warm
    out = panel_universe_health(panel, lookback_days=120)
    assert out["n_columns"] == 4 and out["n_with_price"] == 4
    assert out["n_warm_20"] == 3 and out["n_warm_60"] == 3
    assert out["effective_ratio"] == pytest.approx(0.75)
    # the real 2026 shape: 800 columns, 301 warm
    wide = pd.DataFrame({f"S{i}": (1.0 if i < 301 else np.nan) for i in range(800)}, index=idx)
    bad = panel_universe_health(wide, lookback_days=120)
    assert bad["effective_ratio"] == pytest.approx(0.376, abs=0.01)
    assert evaluate_gate(_bundle(universe=bad))["failed"] == ["effective_universe"]


def test_panel_universe_health_on_an_empty_panel():
    out = panel_universe_health(pd.DataFrame())
    assert out["n_columns"] == 0 and out["effective_ratio"] is None


def test_soft_metrics_never_gate():
    gate = evaluate_gate(_bundle(soft={"sharpe": -2.0, "max_drawdown": -0.9}))
    assert gate["verdict"] == "pass"


def test_tracking_error_needs_a_minimum_number_of_days():
    """A 2-day 'perfect' tracking error is not evidence of fidelity."""
    gate = evaluate_gate(_bundle(tracking_error={"n_days": 2, "mean_abs_pp": 0.0,
                                                 "sign_bias_p": 1.0, "n_pos": 1, "n_neg": 1}))
    assert gate["verdict"] == "fail"
    assert "tracking_error_daily_pp" in gate["failed"]
    assert gate["hard"]["tracking_error_daily_pp"]["min_days"] == 5
    # the same numbers over enough days pass
    ok = evaluate_gate(_bundle(tracking_error={"n_days": 6, "mean_abs_pp": 0.0,
                                               "sign_bias_p": 1.0, "n_pos": 3, "n_neg": 3}))
    assert ok["verdict"] == "pass"


def test_zero_min_days_cannot_switch_the_measurement_off():
    """``tracking_error_min_days: 0`` + ``--no-replay`` used to yield PASS on
    empty series (n_days=0, mean_abs_pp=0.0, p=1.0) — the exact 'unmeasured read
    as fine' failure the gate exists to prevent."""
    th = GateThresholds(tracking_error_min_days=0)
    empty = _bundle(tracking_error={"n_days": 0, "mean_abs_pp": 0.0, "sign_bias_p": 1.0,
                                    "n_pos": 0, "n_neg": 0})
    gate = evaluate_gate(empty, th)
    assert gate["verdict"] == "fail"
    assert gate["hard"]["tracking_error_daily_pp"]["min_days"] == 1   # unconditional floor
    assert "tracking_error_daily_pp" in gate["failed"]


def test_soft_metrics_arithmetic():
    idx = pd.date_range("2026-01-05", periods=60, freq="B")
    eq = pd.Series(np.linspace(100_000, 110_000, 60), index=idx)
    out = soft_metrics(eq)
    assert out["n_days"] == 60 and out["total_return"] > 0
    assert out["sharpe"] is not None and out["max_drawdown"] == 0.0
    assert out["sharpe_standard_error"] == pytest.approx(math.sqrt(252 / 60), abs=1e-3)


def test_thresholds_from_config_shape():
    class _Cfg:
        def get(self, path, default=None):
            if path == "forward.risk_gate.hard":
                return {"tracking_error_daily_pp_max": 0.3, "violations_max": 0}
            if path == "forward.risk_gate.soft.record_only":
                return ["sharpe"]
            return default
    th = GateThresholds.from_config(_Cfg())
    assert th.tracking_error_daily_pp_max == 0.3
    assert th.soft_record_only == ("sharpe",)
    assert th.availability_min == 0.99          # untouched default


# --------------------------------------------------------------------------- #
# paired candidate comparison
# --------------------------------------------------------------------------- #
def test_paired_comparison_detects_a_real_edge():
    idx = pd.date_range("2026-01-05", periods=130, freq="B")
    rng = np.random.default_rng(7)
    common = pd.Series(rng.normal(0.0005, 0.01, 130), index=idx)
    a = common + pd.Series(rng.normal(0.0, 0.004, 130), index=idx)
    b = common + pd.Series(rng.normal(0.0012, 0.004, 130), index=idx)
    out = paired_comparison(a, b, window_days=120, t_min=1.5)
    assert out["ready"] and out["corr"] > 0.5
    assert out["mean_diff_pp"] > 0 and out["verdict"] in {"switch", "hold"}
    assert out["t_stat"] is not None


def test_paired_comparison_holds_when_indistinguishable():
    idx = pd.date_range("2026-01-05", periods=130, freq="B")
    rng = np.random.default_rng(3)
    common = pd.Series(rng.normal(0.0005, 0.01, 130), index=idx)
    a = common + pd.Series(rng.normal(0.0, 0.004, 130), index=idx)
    b = common + pd.Series(rng.normal(0.0, 0.004, 130), index=idx)
    out = paired_comparison(a, b, window_days=120)
    assert out["verdict"] == "hold"
    assert "never separate" in out["note"]


def test_paired_comparison_needs_enough_days():
    a = pd.Series([0.01, 0.02], index=pd.date_range("2026-01-05", periods=2))
    out = paired_comparison(a, a, window_days=120)
    assert out["ready"] is False and out["switch"] is False
    # the key set must match the full result so consumers need no special case
    idx = pd.date_range("2026-01-05", periods=3)
    full = paired_comparison(pd.Series([0.01, 0.02, 0.03], index=idx),
                             pd.Series([0.01, 0.02, 0.03], index=idx))
    assert set(out) - {"reason"} == set(full)


def test_paired_comparison_reports_days_needed():
    idx = pd.date_range("2026-01-05", periods=130, freq="B")
    rng = np.random.default_rng(11)
    common = pd.Series(rng.normal(0.0, 0.01, 130), index=idx)
    a = common + pd.Series(rng.normal(0.0, 0.004, 130), index=idx)
    b = common + pd.Series(rng.normal(0.0005, 0.004, 130), index=idx)
    out = paired_comparison(a, b, window_days=120, t_min=1.5)
    assert out["days_needed_for_t"] is None or out["days_needed_for_t"] > 0
