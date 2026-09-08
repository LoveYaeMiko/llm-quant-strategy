"""Regression tests for defect D-6 (2026-09-08 audit): kill-switch wiring.

The autopilot's risk gate publishes a gross multiplier (1.0 normal, 0.5 de-risk,
0.0 halt) that the ML books honoured through ``ControlScaledPortfolio``. The
pullback (D) book ignored it entirely, so a de-risk/halt decision was inert on
the only live track. ``PullbackPortfolio(scale_getter=...)`` now applies the
multiplier — shrinking only, never levering up — and a halt flattens the book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.paper.pullback_book import PullbackParams, PullbackPortfolio


def _market(symbols=("000001.SZ", "000002.SZ"), days: int = 90):
    """A minimal PIT-like market: flat-ish closes plus high/low/volume."""
    dates = pd.bdate_range("2026-01-01", periods=days)
    rng = np.random.default_rng(7)
    # Upward drift so the 60-day market trend clears the entry gate.
    closes = pd.DataFrame(
        {s: 10.0 + np.arange(days) * 0.02 + np.cumsum(rng.normal(0, 0.02, days))
         for s in symbols}, index=dates
    )
    rows = []
    for d in dates:
        for s in symbols:
            c = float(closes.loc[d, s])
            rows.append(
                {
                    "date": d, "symbol": s,
                    "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                    "volume": 1_000_000.0,
                }
            )
    long = pd.DataFrame(rows).set_index(["date", "symbol"])
    return type("M", (), {"price_panel": closes, "long": long})()


def _book(scale: float | None, k: int = 1) -> PullbackPortfolio:
    market = _market()
    params = PullbackParams(k=k, rank_source="momentum", rank_min=0.0, vol_shrink=False)
    getter = None if scale is None else (lambda: scale)
    book = PullbackPortfolio(market, params, symbols=["000001.SZ", "000002.SZ"],
                             scale_getter=getter)
    # Bypass the entry filters: this test is about the gross multiplier only.
    book._entry_candidates = lambda d: pd.DataFrame(  # type: ignore[method-assign]
        [{"symbol": "000001.SZ", "px": float(book._close.loc[d, "000001.SZ"]), "rank": 1.0}]
    )
    return book


def _last_date(book: PullbackPortfolio) -> pd.Timestamp:
    return book._dates[-1]


def test_full_scale_is_unchanged():
    book = _book(1.0)
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert w == {"000001.SZ": pytest.approx(1.0)}


def test_no_getter_is_full_scale():
    book = _book(None)
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert w == {"000001.SZ": pytest.approx(1.0)}


def test_de_risk_halves_the_book():
    book = _book(0.5)
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert w == {"000001.SZ": pytest.approx(0.5)}


def test_halt_flattens_every_symbol_the_executor_sees():
    book = _book(0.0)
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert w["000001.SZ"] == 0.0
    assert w["000002.SZ"] == 0.0


def test_multiplier_never_levers_up():
    book = _book(1.5)
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert w == {"000001.SZ": pytest.approx(1.0)}


def test_unreadable_state_fails_closed():
    book = _book(None)
    book._scale_getter = lambda: "not-a-number"  # type: ignore[assignment]
    w = book.compute_weights(["000001.SZ", "000002.SZ"], _last_date(book))
    assert set(w.values()) == {0.0}


def test_pullback_book_opts_into_always_rebalance():
    assert PullbackPortfolio.always_rebalance is True
    assert _book(None).always_rebalance is True
