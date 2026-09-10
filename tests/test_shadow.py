"""Shadow mode + §7 calibration — status red lines, cost model, config write-back.

Covers the pieces added on top of the paper runner that are pure / cheap to
exercise offline: the four canonical red lines (regression for the missing-``level``
crash), the real-A-share cost deviation, ``paper_runner_kwargs``, and
``apply_to_config``'s line-edit write-back (comment preservation, skip-unchanged,
integral-float formatting).
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from src.calibration import _fmt, _same_value, apply_to_config
from src.config import load_config
from src.data.synthetic import make_synthetic_market
from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger
from src.paper.shadow import (
    _drawdown_series,
    _fmt_date,
    _threshold_level,
    build_shadow_status,
    compute_cost_deviation,
    enrich_positions,
    fetch_benchmark_index,
    load_benchmark_index,
    paper_runner_kwargs,
    real_cost_model,
)


# ---------------------------------------------------------------------------
# red lines — level is always present (regression for the KeyError crash)
# ---------------------------------------------------------------------------


def test_threshold_level_mapping():
    assert _threshold_level(5.0, 10.0, 20.0) == "ok"
    assert _threshold_level(15.0, 10.0, 20.0) == "warning"
    assert _threshold_level(25.0, 10.0, 20.0) == "critical"


def _build_status(tmp_path, result_extra: dict | None = None):
    cfg = load_config()
    market = make_synthetic_market(symbols=10, days=30, seed=0)
    syms = sorted(market.price_panel.columns)
    d = str(sorted(market.price_panel.index)[1].date())
    led = PaperLedger(tmp_path / "s.sqlite")
    led.record_day(
        d, cash=90_000.0, equity=101_000.0, positions={syms[0]: 100.0},
        fills=[Fill(d, syms[0], "buy", 100.0, 10.0, 2.5, 1000.0)],
        gross_exposure=1000.0,
    )
    result = {"metrics": {"final_equity": 101_000.0, "final_cash": 90_000.0,
                          "total_return": 0.01, "annualized_return": 0.12,
                          "sharpe": 1.0, "max_drawdown": -0.03, "n_days": 5,
                          "n_fills": 1, "total_commission": 2.5}}
    result.update(result_extra or {})
    status = build_shadow_status(cfg, led, market, result, {"pead": None}, {})
    led.close()
    return status


def test_build_shadow_status_surfaces_clipped_orders(tmp_path):
    """A refused order must be visible in the artifact, not only in memory.

    The executor clips a sell that exceeds the holding (never a short,
    ``docs/EXECUTION_INVARIANTS.md``); the runner collects those into
    ``result['skipped_orders']`` and the status now carries them, so a reviewer
    can see that the submitted list asked for something impossible.
    """
    skipped = [{"symbol": "600000.SH", "reason": "sell_exceeds_holding",
                "wanted": -250.0, "clipped": -200.0, "date": "2026-09-10",
                "layer": "auction"}]
    status = _build_status(tmp_path, {"skipped_orders": skipped})
    assert status["equity"]["skipped_orders"] == skipped
    # and a clean run reports an empty list rather than omitting the key
    assert _build_status(tmp_path / "clean").get("equity", {}).get("skipped_orders") == []


def test_build_shadow_status_red_lines_have_level(tmp_path):
    status = _build_status(tmp_path)
    rls = status["red_lines"]
    assert [rl["name"] for rl in rls] == [
        "cost_deviation", "short_leg_deviation", "regime_switch", "pead_anomaly",
    ]
    for rl in rls:
        assert rl["level"] in {"ok", "warning", "critical"}
        assert rl.get("label")


# ---------------------------------------------------------------------------
# cost model — real A-share fees vs the fixed flat-commission model
# ---------------------------------------------------------------------------


def test_compute_cost_deviation_empty():
    dev = compute_cost_deviation(pd.DataFrame(), {})
    assert dev == {"current_total": 0.0, "real_total": 0.0, "deviation_pct": 0.0}


def test_compute_cost_deviation_real_fees_higher():
    fills = pd.DataFrame([
        {"seq": 1, "date": "2024-01-02", "symbol": "A01", "side": "buy",
         "shares": 100.0, "price": 10.0, "commission": 2.5, "notional": 1000.0},
        {"seq": 2, "date": "2024-01-03", "symbol": "A01", "side": "sell",
         "shares": 100.0, "price": 10.0, "commission": 2.5, "notional": 1000.0},
    ])
    real = {"commission_bps": 2.5, "min_commission": 5.0,
            "stamp_tax_sell_bps": 5.0, "transfer_fee_bps": 0.1}
    dev = compute_cost_deviation(fills, real)
    assert dev["current_total"] == pytest.approx(5.0)  # flat 2.5 + 2.5
    # real: buy max(5, 0.25) + 0.01 = 5.01 ; sell 5 + 0.01 + 0.5 = 5.51 -> 10.52
    assert dev["real_total"] == pytest.approx(10.52, abs=1e-9)
    assert dev["deviation_pct"] > 0


def test_real_cost_model():
    real = real_cost_model(load_config())
    assert real["commission_bps"] == 2.5
    assert real["min_commission"] == 5.0
    assert real["stamp_tax_sell_bps"] == 5.0
    assert real["transfer_fee_bps"] == 0.1


def test_paper_runner_kwargs():
    kw = paper_runner_kwargs(load_config())
    assert kw["cash"] == 100000.0
    # §7 calibration writes the real A-share cost structure back into the
    # ``paper`` section (see test_real_cost_model), so these are now 2.5 / 5.0
    # / 0.1 rather than the pre-calibration flat defaults.
    assert kw["commission_bps"] == 2.5
    assert kw["stamp_tax_sell_bps"] == 5.0
    assert kw["transfer_fee_bps"] == 0.1
    assert kw["max_position_pct"] == 0.05
    assert kw["pit_strict"] is True


def test_paper_section_is_single_sourced():
    # the ``paper`` section must live in exactly one config file; a duplicate in
    # factor_thresholds.yaml (shallow override) would shadow the cost-model keys.
    pcfg = load_config().section("paper")
    assert "stamp_tax_sell_bps" in pcfg
    assert "transfer_fee_bps" in pcfg


# ---------------------------------------------------------------------------
# config write-back — line edit, comment preservation, skip-unchanged
# ---------------------------------------------------------------------------


def test_fmt_integral_floats():
    assert _fmt(5.0) == "5"
    assert _fmt(2.5) == "2.5"
    assert _fmt(0.1) == "0.1"
    assert _fmt(-2.5) == "-2.5"
    assert _fmt(True) == "true"
    assert _fmt(5) == "5"


def test_same_value_numeric_tolerant():
    assert _same_value("2.0", 2.0) is True
    assert _same_value("0.20", 0.2) is True
    assert _same_value("5", 5.0) is True
    assert _same_value("0.0", 5.0) is False
    assert _same_value("abc", "abc") is True


def test_apply_to_config(tmp_path):
    cfg_path = tmp_path / "master_config.yaml"
    cfg_path.write_text(
        "paper:\n"
        "  commission_bps: 5.0      # 券商佣金\n"
        "  min_commission: 1.0      # 最低佣金\n"
        "  stamp_tax_sell_bps: 0.0  # 印花税\n",
        encoding="utf-8",
    )
    res = apply_to_config(
        {
            "paper.commission_bps": 2.5,
            "paper.min_commission": 5.0,
            "paper.stamp_tax_sell_bps": 0.0,  # value unchanged -> skip
            "seasonal_tilt.amplitude": 0.25,  # section absent -> unchanged
        },
        path=cfg_path,
    )
    text = cfg_path.read_text(encoding="utf-8")
    assert "commission_bps: 2.5" in text
    assert "commission_bps: 5.0" not in text
    assert "min_commission: 5" in text          # integral float -> no ".0"
    assert "min_commission: 1.0" not in text
    assert "stamp_tax_sell_bps: 0.0" in text    # untouched
    assert "# 券商佣金" in text                 # trailing comments preserved
    assert "# 最低佣金" in text
    assert "# 印花税" in text
    assert res["changed"] == {
        "paper.commission_bps": {"old": "5.0", "new": "2.5"},
        "paper.min_commission": {"old": "1.0", "new": "5"},
    }
    assert "seasonal_tilt" in res["unchanged"]


# ---------------------------------------------------------------------------
# positions — moving average cost basis (regression for the trade-price bug)
# ---------------------------------------------------------------------------


def _fills_frame(rows: list[dict]) -> pd.DataFrame:
    """Minimal ``ledger.fills()``-shaped frame (signed shares, sell negative)."""
    base = {"seq": 1, "side": "buy", "commission": 0.0, "notional": 0.0}
    out = []
    for i, r in enumerate(rows):
        row = {**base, **r}
        row["seq"] = i + 1
        out.append(row)
    return pd.DataFrame(out)


def test_enrich_positions_sell_keeps_avg_cost():
    # buy 100 @ 10, sell 40 @ 12 -> remaining 60 avg stays 10 (not pulled by px)
    fills = _fills_frame([
        {"date": "2024-01-02", "symbol": "A01", "side": "buy", "shares": 100.0, "price": 10.0},
        {"date": "2024-01-03", "symbol": "A01", "side": "sell", "shares": -40.0, "price": 12.0},
    ])
    out = enrich_positions(fills, {"A01": 60.0})
    assert out["A01"]["entry_price"] == pytest.approx(10.0)
    assert out["A01"]["pnl_basis"] == pytest.approx(600.0)


def test_enrich_positions_reopen_uses_new_price():
    # buy 100 @ 10, sell 100 @ 12, buy 100 @ 15 -> entry 15 (not 13)
    fills = _fills_frame([
        {"date": "2024-01-02", "symbol": "A01", "side": "buy", "shares": 100.0, "price": 10.0},
        {"date": "2024-01-03", "symbol": "A01", "side": "sell", "shares": -100.0, "price": 12.0},
        {"date": "2024-01-04", "symbol": "A01", "side": "buy", "shares": 100.0, "price": 15.0},
    ])
    out = enrich_positions(fills, {"A01": 100.0})
    assert out["A01"]["entry_price"] == pytest.approx(15.0)


def test_enrich_positions_short_cover_keeps_avg_cost():
    # open short 100 @ 10 (sell -100), cover 50 @ 12 -> remaining short avg 10
    fills = _fills_frame([
        {"date": "2024-01-02", "symbol": "A01", "side": "sell", "shares": -100.0, "price": 10.0},
        {"date": "2024-01-03", "symbol": "A01", "side": "buy", "shares": 50.0, "price": 12.0},
    ])
    out = enrich_positions(fills, {"A01": -50.0})
    assert out["A01"]["entry_price"] == pytest.approx(10.0)
    assert out["A01"]["pnl_basis"] == pytest.approx(500.0)


def test_enrich_positions_short_reversal():
    # short 100 @ 10, then buy 150 @ 12 -> flips to long 50 opened at 12
    fills = _fills_frame([
        {"date": "2024-01-02", "symbol": "A01", "side": "sell", "shares": -100.0, "price": 10.0},
        {"date": "2024-01-03", "symbol": "A01", "side": "buy", "shares": 150.0, "price": 12.0},
    ])
    out = enrich_positions(fills, {"A01": 50.0})
    assert out["A01"]["entry_price"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# drawdown / date helpers / benchmark
# ---------------------------------------------------------------------------


def test_drawdown_series():
    eq = pd.Series([100.0, 110.0, 99.0, 121.0], index=["d1", "d2", "d3", "d4"])
    dd = _drawdown_series(eq)
    assert dd.loc["d1"] == 0.0
    assert dd.loc["d2"] == 0.0
    assert dd.loc["d3"] == pytest.approx(99.0 / 110.0 - 1.0)
    assert dd.loc["d4"] == 0.0


def test_fmt_date():
    assert _fmt_date("20240101") == "2024-01-01"
    assert _fmt_date("2024-01-02") == "2024-01-02"
    assert _fmt_date(None) is None
    assert _fmt_date("") is None


def test_fetch_benchmark_index_parses_and_filters(monkeypatch):
    fake = types.ModuleType("akshare")
    fake.stock_zh_index_daily = lambda symbol: pd.DataFrame({
        "date": ["2023-12-29", "2024-01-02", "2024-01-03"],
        "close": [3500.0, 3600.0, 3700.0],
    })
    monkeypatch.setitem(sys.modules, "akshare", fake)
    s = fetch_benchmark_index(load_config(), start_date="20240101", end_date="20240103")
    assert list(s.index) == ["2024-01-02", "2024-01-03"]
    assert list(s.values) == [3600.0, 3700.0]


def test_fetch_benchmark_index_returns_none_on_failure(monkeypatch):
    fake = types.ModuleType("akshare")
    def _boom(symbol):
        raise RuntimeError("network down")
    fake.stock_zh_index_daily = _boom
    monkeypatch.setitem(sys.modules, "akshare", fake)
    assert fetch_benchmark_index(load_config()) is None


def test_load_benchmark_index_roundtrip(tmp_path, monkeypatch):
    cache = tmp_path / "hs300.csv"
    monkeypatch.setattr("src.paper.shadow._benchmark_cache", lambda cfg: cache)
    assert load_benchmark_index(load_config()) is None  # missing -> None
    cache.write_text(",close\n2024-01-02,3600.0\n2024-01-03,3700.0\n", encoding="utf-8")
    s = load_benchmark_index(load_config())
    assert list(s.index) == ["2024-01-02", "2024-01-03"]
    assert list(s.values) == [3600.0, 3700.0]


# ---------------------------------------------------------------------------
# curves — equity / benchmark / excess emitted from build_shadow_status
# ---------------------------------------------------------------------------


def test_build_shadow_status_emits_curves(tmp_path):
    cfg = load_config()
    initial_cash = float(cfg.section("paper").get("initial_cash", 100_000.0))
    market = make_synthetic_market(symbols=3, days=5, seed=0)
    syms = sorted(market.price_panel.columns)
    dates = sorted(market.price_panel.index)
    d0, d1 = str(dates[0].date()), str(dates[1].date())
    led = PaperLedger(tmp_path / "curves.sqlite")
    led.record_day(d0, cash=90_000.0, equity=initial_cash, positions={syms[0]: 100.0},
                   fills=[], gross_exposure=1000.0)
    led.record_day(d1, cash=90_000.0, equity=initial_cash * 1.02, positions={syms[0]: 100.0},
                   fills=[], gross_exposure=1000.0)
    bench = pd.Series([1.0, 1.01], index=[d0, d1])
    status = build_shadow_status(cfg, led, market, {"metrics": {}}, {"pead": None}, {},
                                 benchmark=bench)
    led.close()
    assert len(status["equity_curve"]) == 2
    assert len(status["benchmark"]) == 2
    assert len(status["excess_curve"]) == 2
    assert status["benchmark"][0]["equity"] == pytest.approx(initial_cash)
    # excess = port_nav − bench_nav = (1.02) − (1.01/1.0)
    assert status["excess_curve"][1]["excess"] == pytest.approx(0.02 - 0.01, abs=1e-6)
