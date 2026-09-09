"""Closing-auction order layer — execute_orders + runner preclose provider."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.online.order_executor import OrderExecutor
from src.paper.ledger import PaperLedger
from src.paper.runner import PaperRunner


def _ex(**kw):
    kw.setdefault("cash", 100_000.0)
    kw.setdefault("slippage_bps", 2.0)
    kw.setdefault("commission_bps", 2.5)
    kw.setdefault("min_commission", 5.0)
    kw.setdefault("stamp_tax_sell_bps", 5.0)
    kw.setdefault("transfer_fee_bps", 0.1)
    return OrderExecutor(**kw)


def test_execute_orders_fills_at_auction_close():
    ex = _ex()
    # A-share long-only account: A02 is HELD before the sell order arrives
    # (selling shares you do not own is not available to this account type).
    ex.restore(100_000.0, {"A02": 100.0})
    prices = pd.Series({"A01": 10.00, "A02": 20.00})
    res = ex.execute_orders(
        [{"symbol": "A01", "shares": 100.0}, {"symbol": "A02", "shares": -100.0}],
        prices, pd.Timestamp("2026-09-07"),
    )
    assert len(res.fills) == 2
    # sells settle first (cash guard), then buys
    assert [f.side for f in res.fills] == ["sell", "buy"]
    sell = next(f for f in res.fills if f.side == "sell")
    buy = next(f for f in res.fills if f.side == "buy")
    assert buy.price == pytest.approx(10.00 * 1.0002, abs=0.01)  # bid/ask slippage, tick
    assert sell.price == pytest.approx(20.00 * 0.9998, abs=0.01)
    assert buy.time == "15:00"  # auction fill timestamp
    assert ex.positions == {"A01": 100.0}  # A02 closed, no short opened
    # fees: commission min 5 + transfer + stamp on the sell side
    assert buy.commission > 0 and sell.commission > buy.commission


def test_execute_orders_never_opens_a_short():
    """A sell the account cannot cover is clipped/skipped and REPORTED.

    Regression (2026-09-09): a forward candidate started from an empty ledger
    executed production's sell list and silently opened a short position.
    """
    ex = _ex()
    ex.restore(100_000.0, {"A02": 100.0})
    prices = pd.Series({"A01": 10.00, "A02": 20.00, "A03": 5.00})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -250.0},   # more than held → clipped
         {"symbol": "A03", "shares": -100.0}],  # not held → skipped
        prices, pd.Timestamp("2026-09-07"),
    )
    assert [f.shares for f in res.fills] == [-100.0]
    assert ex.positions == {}
    reasons = {s["symbol"]: s["reason"] for s in res.skipped}
    assert reasons == {"A02": "sell_exceeds_holding", "A03": "sell_without_holding"}
    assert all(v >= 0 for v in ex.positions.values())


def test_execute_orders_skips_locked_and_suspended():
    ex = _ex()
    ex.restore(100_000.0, {})
    prices = pd.Series({"A01": 10.00, "A02": np.nan})
    locked = pd.Series({"A01": 0.098, "A02": 0.0})  # A01 at limit-up
    res = ex.execute_orders(
        [{"symbol": "A01", "shares": 100.0}, {"symbol": "A02", "shares": 100.0}],
        prices, pd.Timestamp("2026-09-07"), limit_locked=locked,
    )
    assert res.fills == []  # buy into limit-up lapses; suspended has no print
    assert ex.positions == {}


def _runner(portfolio, market, db, provider=None, **kw):
    kw.setdefault("symbols", sorted(market.price_panel.columns))
    kw.setdefault("cash", 100_000.0)
    kw.setdefault("max_position_pct", 1.0)
    kw.setdefault("rebalance_days", 1)
    kw.setdefault("pit_strict", True)
    return PaperRunner(
        portfolio, market, PaperLedger(db), preclose_provider=provider, **kw
    )


class _StaticPortfolio:
    def __init__(self, symbols, switch_date=None):
        self.weights = {symbols[0]: 0.5}
        self.switch_date = pd.Timestamp(switch_date) if switch_date else None

    def compute_weights(self, symbols, date):
        if self.switch_date is not None and pd.Timestamp(date) > self.switch_date:
            return {symbols[2]: 0.5}
        return dict(self.weights)


def test_runner_executes_preclose_orders_instead_of_book(tmp_path):
    market = make_synthetic_market(symbols=4, days=10, seed=5)
    syms = sorted(market.price_panel.columns)
    dates = sorted(market.price_panel.index)
    d0 = str(dates[3].date())

    def provider(d):
        if str(pd.Timestamp(d).date()) == d0:
            return [{"symbol": syms[1], "shares": 200.0}]  # NOT the book's A01
        return "__normal__"

    r = _runner(_StaticPortfolio(syms), market, tmp_path / "l.sqlite", provider=provider)
    r.run()
    fills_d0 = r.ledger.fills_for_date(d0)
    assert len(fills_d0) == 1
    assert fills_d0[0].symbol == syms[1]  # the preclose order won over the book
    assert fills_d0[0].time == "15:00"
    # other days used the normal book (A01)
    other = [f for d in dates if str(d.date()) != d0 for f in r.ledger.fills_for_date(d)]
    assert any(f.symbol == syms[0] for f in other)
    r.ledger.close()


def test_runner_skips_close_trades_when_no_orders_on_live_date(tmp_path):
    market = make_synthetic_market(symbols=4, days=10, seed=6)
    syms = sorted(market.price_panel.columns)
    dates = sorted(market.price_panel.index)
    d0 = str(dates[3].date())

    def provider(d):
        if str(pd.Timestamp(d).date()) == d0:
            return None  # 14:55 job never ran → no close trades, as in reality
        return "__normal__"

    # switch AT d0: the normal path WOULD trade on d0 (sell A01, buy C03) —
    # the provider-None must suppress exactly those trades (regression for the
    # sentinel bug where None fell through to the normal path, 2026-09-07).
    r = _runner(
        _StaticPortfolio(syms, switch_date=dates[2]), market,
        tmp_path / "l.sqlite", provider=provider,
    )
    r.run()
    assert r.ledger.fills_for_date(d0) == []
    assert len(r.ledger.fills_for_date(dates[4])) >= 2  # next day back to normal
    r.ledger.close()
