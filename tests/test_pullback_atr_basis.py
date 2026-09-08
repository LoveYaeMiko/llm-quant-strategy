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
                "open": raw_close, "high": raw_close * 1.02,
                "low": raw_close * 0.98, "close": raw_close * f,
                "volume": 1_000_000.0,
            })
            records.append({"date": d, "symbol": sym, "adjust_factor": f})
    long = pd.DataFrame(rows).set_index(["date", "symbol"])
    rec = pd.DataFrame(records)
    return type("M", (), {"price_panel": closes, "long": long, "records": rec})()


def _book(factor_b: float = 0.1) -> PullbackPortfolio:
    params = PullbackParams(k=1, rank_source="momentum", rank_min=0.0, vol_shrink=False)
    return PullbackPortfolio(_market(factor_b), params, symbols=["A", "B"])


def test_adjust_factor_frame_reads_records():
    book = _book()
    last = book._dates[-1]
    factor = book._basis_factor(last)
    assert factor["A"] == pytest.approx(1.0)
    assert factor["B"] == pytest.approx(0.1)


def test_atr_pct_is_basis_consistent():
    book = _book()
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
    assert book._stop_dist("B", last) == pytest.approx(min(0.04, 1.5 * atr_b))
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


def test_intraday_prints_are_converted_to_panel_basis():
    """Raw minute prints must be compared with adjusted stops after scaling."""
    book = _book()
    last = book._dates[-1]
    sym = "B"
    book._open[sym] = type("L", (), {
        "symbol": sym, "entry_date": last, "entry_price": 1.0,
        "stop": 1.0, "stop_dist": 0.03, "peak": 1.0,
        "trail_active": False, "qty": 0.0, "cost": 0.0,
    })()
    # A raw print of 9.9 (= 0.99 adjusted) breaches a stop of 1.0; the same
    # raw print without the basis conversion would be compared as 9.9.
    raw = pd.DataFrame({
        "timestamp": pd.to_datetime([f"{last.date()} 10:00:00"]),
        "open": [9.9], "high": [9.9], "low": [9.9], "close": [9.9],
    })
    book._minute_provider = lambda d, s: raw
    exits = book.intraday_exits(last)
    assert len(exits) == 1
    assert exits[0]["price"] == pytest.approx(0.99)
