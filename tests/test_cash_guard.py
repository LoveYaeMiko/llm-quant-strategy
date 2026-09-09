"""Cash guard — no implicit leverage (audit finding V-1, 2026-09-09).

The production ledger carried 12 days of NEGATIVE cash (2026-01-08 −4,376 on a
50,000 account ≈ 8.7% leverage) because ``OrderExecutor.execute`` walked the
target frame in column order and debited every fill unconditionally: a rebalance
that switched from A to B could buy B before selling A. Fixes pinned here:

* sells execute before buys within a day;
* every buy is clipped to the cash actually available (reserving the fee);
* a buy that cannot afford one board lot is skipped (odd-lot buys are illegal).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.online.order_executor import OrderExecutor


def _day() -> pd.Timestamp:
    return pd.Timestamp("2026-09-09")


def test_switch_never_goes_negative_even_when_the_buy_column_comes_first():
    day = _day()
    ex = OrderExecutor(cash=50_000.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(0.0, {"600000.SH": 5_000.0})  # fully invested, no cash
    prices = pd.DataFrame([{"600000.SH": 10.0, "000001.SZ": 10.0}], index=[day])
    # B (buy) is the FIRST column — the old code would buy before selling A
    targets = pd.DataFrame([{"000001.SZ": 1.0, "600000.SH": 0.0}], index=[day])
    res = ex.execute(targets, prices, equity=50_000.0)
    assert res.cash >= 0.0
    sides = [f.side for f in res.fills]
    assert sides == ["sell", "buy"]          # sell first
    bought = [f for f in res.fills if f.side == "buy"][0]
    sold = [f for f in res.fills if f.side == "sell"][0]
    assert sold.shares == pytest.approx(-5_000.0)
    # buys are funded by the sale proceeds: ~50k / 10 = 5,000 shares (minus fees)
    assert bought.shares <= 5_000.0
    assert bought.shares % 100 == 0
    assert res.positions["000001.SZ"] > 0
    assert "600000.SH" not in res.positions


def test_buy_is_clipped_to_available_cash():
    day = _day()
    ex = OrderExecutor(cash=1_100.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(1_100.0, {})
    prices = pd.DataFrame([{"600000.SH": 10.0}], index=[day])
    targets = pd.DataFrame([{"600000.SH": 1.0}], index=[day])
    res = ex.execute(targets, prices, equity=1_100.0)
    assert len(res.fills) == 1
    fill = res.fills[0]
    assert fill.side == "buy"
    assert fill.shares * fill.price + fill.commission <= 1_100.0 + 1e-9
    assert res.cash >= 0.0
    assert fill.shares == pytest.approx(100.0)  # 100 shares ≈ 1000 + fee


def test_exactly_insufficient_cash_skips_the_buy():
    """1,000 cash cannot fund a 100-share lot at 10.00 plus the 5-yuan minimum
    commission (1005.01) — the old code bought it anyway and went negative."""
    day = _day()
    ex = OrderExecutor(cash=1_000.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(1_000.0, {})
    prices = pd.DataFrame([{"600000.SH": 10.0}], index=[day])
    targets = pd.DataFrame([{"600000.SH": 1.0}], index=[day])
    res = ex.execute(targets, prices, equity=1_000.0)
    assert res.fills == []
    assert res.cash == pytest.approx(1_000.0)


def test_unaffordable_buy_is_skipped_entirely():
    day = _day()
    ex = OrderExecutor(cash=500.0, notional_floor=0.0, max_position_pct=1.0)
    ex.restore(500.0, {})
    prices = pd.DataFrame([{"600000.SH": 10.0}], index=[day])
    targets = pd.DataFrame([{"600000.SH": 1.0}], index=[day])
    res = ex.execute(targets, prices, equity=500.0)
    # 500/10 = 50 shares < one 100-share lot → no fill, cash untouched
    assert res.fills == []
    assert res.cash == pytest.approx(500.0)


def test_auction_orders_sell_first_and_clip_buys():
    day = _day()
    ex = OrderExecutor(cash=0.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(0.0, {"600000.SH": 5_000.0})
    prices = pd.Series({"600000.SH": 10.0, "000001.SZ": 10.0})
    orders = [
        {"symbol": "000001.SZ", "shares": 5_000.0},   # buy listed first
        {"symbol": "600000.SH", "shares": -5_000.0},  # sell
    ]
    res = ex.execute_orders(orders, prices, day, fill_time="15:00")
    assert res.cash >= 0.0
    assert [f.side for f in res.fills] == ["sell", "buy"]
    assert "600000.SH" not in res.positions
    assert res.positions["000001.SZ"] > 0


def test_auction_buy_clipped_when_sell_proceeds_are_insufficient():
    day = _day()
    ex = OrderExecutor(cash=1_100.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(1_100.0, {})
    prices = pd.Series({"000001.SZ": 10.0})
    res = ex.execute_orders([{"symbol": "000001.SZ", "shares": 5_000.0}], prices, day)
    assert res.cash >= 0.0
    assert len(res.fills) == 1
    fill = res.fills[0]
    assert fill.shares == pytest.approx(100.0)   # clipped to one affordable lot
    assert fill.shares * fill.price + fill.commission <= 1_100.0 + 1e-9


def test_auction_buy_below_one_lot_is_dropped():
    day = _day()
    ex = OrderExecutor(cash=500.0, notional_floor=0.0, max_position_pct=1.0,
                       commission_bps=2.5, min_commission=5.0)
    ex.restore(500.0, {})
    prices = pd.Series({"000001.SZ": 10.0})
    res = ex.execute_orders([{"symbol": "000001.SZ", "shares": 50.0}], prices, day)
    assert res.fills == []          # odd-lot buy is illegal outside a full close
    assert res.cash == pytest.approx(500.0)
