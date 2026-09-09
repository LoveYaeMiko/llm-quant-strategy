"""Regression tests for defect D-8 (2026-09-08 audit): mixed-basis ATR.

ADR-0002 fixes the PIT panel's basis: ``open``/``high``/``low`` are RAW, while
``close`` is backward-adjusted (``close = raw_close × adjust_factor``, factor
anchored at the newest bar). The pullback book's true range mixed the two — on a
corporate-action bar (e.g. a 10:1 split) ``high - prev_close`` exploded, the
ATR% became nonsense, and the ATR-based stop distance was clipped to its 4% cap.

The book now scales high/low onto the close's basis with the per-bar factor, so
a name with a large adjustment factor gets the same ATR% as an identical name
without one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.paper.pullback_book import PullbackParams, PullbackPortfolio

DAYS = 60


def _market(factor_b: float = 0.1):
    """Two identical price paths; B carries a corporate-action factor."""
    dates = pd.bdate_range("2026-01-01", periods=DAYS)
    rng = np.random.default_rng(3)
    path = 10.0 + np.cumsum(rng.normal(0, 0.05, DAYS))

    closes = pd.DataFrame({"A": path, "B": path * factor_b}, index=dates)
    rows = []
    records = []
    for d, raw_close in zip(dates, path):
        for sym in ("A", "B"):
            f = 1.0 if sym == "A" else factor_b
            rows.append({
                "date": d, "symbol": sym,
                # ±1% intraday range → TR ≈ 2% of price → ATR% ≈ 2% → 1.5×ATR
                # lands INSIDE the [2.5%, 4%] band (a wider range would clip to
                # the cap and let a hardcoded 0.04 pass the assertion below).
                "open": raw_close, "high": raw_close * 1.01,
                "low": raw_close * 0.99, "close": raw_close * f,
                "volume": 1_000_000.0,
            })
            records.append({"date": d, "symbol": sym, "adjust_factor": f})
    long = pd.DataFrame(rows).set_index(["date", "symbol"])
    rec = pd.DataFrame(records)
    return type("M", (), {"price_panel": closes, "long": long, "records": rec})()


def _book(factor_b: float = 0.1, **overrides) -> PullbackPortfolio:
    params = PullbackParams(k=1, rank_source="momentum", rank_min=0.0, vol_shrink=False, **overrides)
    return PullbackPortfolio(_market(factor_b), params, symbols=["A", "B"])


def test_default_stop_is_flat_3p5():
    """The CODE default must be the deployed flat 3.5% stop.

    Deleting ``pb_stop_lo``/``pb_stop_hi`` from the YAML must not silently revert
    the book to the unvalidated ATR-adaptive band: with the class default the ATR
    channel has to be inert (stop_lo == stop_hi).
    """
    params = PullbackParams()
    assert params.stop_lo == params.stop_hi == pytest.approx(0.035)
    book = _book()
    last = book._dates[-1]
    assert book._stop_dist("A", last) == pytest.approx(0.035)
    assert book._stop_dist("B", last) == pytest.approx(0.035)


def test_adjust_factor_frame_reads_records():
    book = _book()
    last = book._dates[-1]
    factor = book._basis_factor(last)
    assert factor["A"] == pytest.approx(1.0)
    assert factor["B"] == pytest.approx(0.1)


def test_atr_pct_is_basis_consistent():
    # explicit ATR band: this test covers the RESEARCH path (the default is now
    # the deployed flat 3.5% — see test_default_stop_is_flat_3p5)
    book = _book(stop_lo=0.025, stop_hi=0.04)
    last = book._dates[-1]
    assert isinstance(book._atr_pct, pd.DataFrame)      # not a per-date Series
    assert book._atr_pct.shape[1] == 2
    atr_a = float(book._atr_pct.loc[last, "A"])
    atr_b = float(book._atr_pct.loc[last, "B"])
    assert np.isfinite(atr_a) and np.isfinite(atr_b)
    # Identical paths → identical ATR% regardless of the adjustment factor.
    assert atr_b == pytest.approx(atr_a, rel=1e-9)
    # The stop distance follows 1.5×ATR% clipped to [2.5%, 4%] — i.e. the
    # adaptive path is alive, not the flat 2.5% floor (defect D-8b).
    assert book._stop_dist("A", last) == pytest.approx(book._stop_dist("B", last))
    expected = float(np.clip(1.5 * atr_b, 0.025, 0.04))
    assert book._stop_dist("B", last) == pytest.approx(expected)
    # The fixture is tuned so the result is NOT the 4% cap: a hardcoded 0.04
    # would fail here (the audit flagged the earlier version as vacuous).
    assert expected < 0.04
    assert book._stop_dist("B", last) > 0.025


def test_mixed_basis_would_have_inflated_atr():
    """Documents the defect: raw high/low against an adjusted close."""
    book = _book()
    last = book._dates[-1]
    raw_high = 10.0 * 1.02
    adjusted_close = 10.0 * 0.1
    naive_tr = abs(raw_high - adjusted_close)
    naive_atr_pct = naive_tr / adjusted_close
    assert naive_atr_pct > 5.0                      # absurd
    assert float(book._atr_pct.loc[last, "B"]) < 0.1  # fixed: sane


def test_missing_records_falls_back_to_factor_one():
    market = _market()
    market.records = None
    params = PullbackParams(k=1, rank_source="momentum", rank_min=0.0, vol_shrink=False)
    book = PullbackPortfolio(market, params, symbols=["A", "B"])
    assert book._basis_factor(book._dates[-1])["B"] == pytest.approx(1.0)


def test_minute_prints_are_compared_as_is():
    """The minute cache is ALREADY adjustment-scaled — never convert it again.

    Verified against the PIT store (2026-09-09): AlphaFeed's minute endpoint
    returns 前复权 bars (for 000001.SZ on 2026-01-05 the cache's last print is
    11.1336 — exactly the PIT adjusted close — while the raw close was 11.50).
    Scaling a print again by ``adjust_factor`` double-adjusts it (~3% too low on
    this fixture) and fires stops on noise.
    """
    book = _book()
    last = book._dates[-1]
    sym = "B"
    book._open[sym] = type("L", (), {
        "symbol": sym, "entry_date": last, "entry_price": 1.0,
        "stop": 1.0, "stop_dist": 0.03, "peak": 1.0,
        "trail_active": False, "qty": 0.0, "cost": 0.0,
    })()
    # 1.01 is above the stop → no exit, even though 1.01 × 0.1 (this symbol's
    # adjustment factor) would look like a breach if the print were scaled.
    raw = pd.DataFrame({
        "timestamp": pd.to_datetime([f"{last.date()} 10:00:00"]),
        "open": [1.01], "high": [1.01], "low": [1.01], "close": [1.01],
    })
    book._minute_provider = lambda d, s: raw
    assert book.intraday_exits(last) == []
    assert sym in book._open
    # A print AT the stop does exit, at the print itself.
    raw2 = raw.assign(open=1.0, high=1.0, low=1.0, close=1.0)
    book._minute_provider = lambda d, s: raw2
    exits = book.intraday_exits(last)
    assert len(exits) == 1
    assert exits[0]["price"] == pytest.approx(1.0)
