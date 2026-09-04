"""PullbackPortfolio ledger seeding — the 2026-09-04 ghost-lot regression.

The old seeding subtracted NEGATIVE sell shares (``qty -= shares``), so a fully
closed position was resurrected as a doubled "ghost" lot and re-bought by the
next close rebalance (observed: D bought back 000596.SZ on 09-04 without any
entry signal). The fix accumulates SIGNED shares with moving-average cost —
closed positions must leave NO lot behind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.paper.pullback_book import PullbackParams, PullbackPortfolio


class _LedgerStub:
    def __init__(self, rows: list[dict]):
        self._df = pd.DataFrame(rows)

    def fills(self):
        return self._df


def _book(n_symbols=8, days=40, rows=None, **params):
    market = make_synthetic_market(symbols=n_symbols, days=days, seed=11)
    syms = sorted(market.price_panel.columns)
    p = PullbackParams(**params)
    return PullbackPortfolio(
        market, p, symbols=syms, ledger=_LedgerStub(rows or [])
    )


def _fill(date, symbol, side, shares, price, seq=0):
    return {"date": date, "seq": seq, "symbol": symbol, "side": side,
            "shares": shares, "price": price}


def test_closed_position_leaves_no_ghost_lot():
    # buy 200 → sell 200 (signed) must reconstruct NOTHING, not a 400-share lot
    rows = [
        _fill("2024-01-02", "A01", "buy", 200.0, 10.0, 1),
        _fill("2024-01-03", "A01", "sell", -200.0, 12.0, 2),
    ]
    book = _book(rows=rows)
    assert "A01" not in book._open, "ghost lot resurrected from a closed position"


def test_partial_sell_keeps_moving_average_entry():
    rows = [
        _fill("2024-01-02", "A01", "buy", 100.0, 10.0, 1),
        _fill("2024-01-03", "A01", "buy", 100.0, 20.0, 2),
        _fill("2024-01-04", "A01", "sell", -100.0, 25.0, 3),
        _fill("2024-01-02", "B02", "buy", 300.0, 20.0, 4),
    ]
    book = _book(rows=rows)
    lot = book._open["A01"]
    assert lot.qty == pytest.approx(100.0)
    assert lot.entry_price == pytest.approx(15.0)  # (10+20)/2 — untouched by the sell
    assert lot.stop < lot.entry_price  # stop is below entry, never == entry
    assert book._open["B02"].qty == pytest.approx(300.0)


def test_open_short_position_seeds_negative_qty():
    # the accumulation is sign-based; a net-short book must seed negative qty
    rows = [
        _fill("2024-01-02", "A01", "sell", -100.0, 50.0, 1),
    ]
    book = _book(rows=rows)
    lot = book._open.get("A01")
    assert lot is not None
    assert lot.qty == pytest.approx(-100.0)


def test_reentry_after_flat_restarts_at_trade_price():
    rows = [
        _fill("2024-01-02", "A01", "buy", 200.0, 10.0, 1),
        _fill("2024-01-03", "A01", "sell", -200.0, 12.0, 2),
        _fill("2024-01-05", "A01", "buy", 100.0, 14.0, 3),
    ]
    book = _book(rows=rows)
    lot = book._open["A01"]
    assert lot.qty == pytest.approx(100.0)
    assert lot.entry_price == pytest.approx(14.0)  # fresh block, fresh basis
    assert str(lot.entry_date.date()) == "2024-01-05"
