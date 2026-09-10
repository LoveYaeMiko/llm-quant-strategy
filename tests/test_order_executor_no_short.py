"""Long-only execution invariant — sells can never open a short.

Independent audit, 2026-09-10. The guard added by ``91bf94b`` clipped a sell
against ``self.positions`` *per order*, so an order list carrying the same sell
twice (the 14:50 list and the persisted account state disagreeing, a duplicated
line, two "reduce" directives in one auction, ...) filled BOTH lines and left
an naked short:

    ex.restore(10_000.0, {"A02": 100.0})
    ex.execute_orders([{"symbol": "A02", "shares": -100.0},
                       {"symbol": "A02", "shares": -100.0}], prices, day)
    # BEFORE the fix → positions={'A02': -100.0}, fills=[-100, -100], skipped=[]
    # AFTER  the fix → positions={},           fills=[-100],      skipped=[1 entry]

A-share rules forbid this account type from holding a short, and the short was
invisible: ``OrderResult.skipped`` was also missing from ``to_dict()``. Both are
pinned here. Offline: no DB, no network, no cache.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.online.order_executor import OrderExecutor

DAY = pd.Timestamp("2026-09-10")


def _ex(**kw) -> OrderExecutor:
    """Executor with the production-like cost model (min commission 5 yuan)."""
    kw.setdefault("max_position_pct", 1.0)
    kw.setdefault("commission_bps", 2.5)
    kw.setdefault("min_commission", 5.0)
    kw.setdefault("stamp_tax_sell_bps", 5.0)
    kw.setdefault("transfer_fee_bps", 0.1)
    return OrderExecutor(**kw)


def _sell_shares(res):
    return [f.shares for f in res.fills if f.side == "sell"]


# -- execute_orders: cumulative clip ---------------------------------------


def test_duplicate_sell_cannot_open_a_short():
    """Two sells of 100 against a 100-share holding → ONE fill, no short.

    Reproduces the audit finding exactly: the second line had nothing left to
    cover and used to be filled as a naked short.
    """
    ex = _ex(cash=10_000.0)
    ex.restore(10_000.0, {"A02": 100.0})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -100.0}, {"symbol": "A02", "shares": -100.0}],
        pd.Series({"A02": 10.0}),
        DAY,
    )
    assert len(res.fills) == 1, "the duplicate sell must not produce a second fill"
    assert sum(f.shares for f in res.fills) == pytest.approx(-100.0)
    assert _sell_shares(res) == [-100.0]
    assert "A02" not in ex.positions and res.positions == {}
    assert all(v >= 0.0 for v in res.positions.values())
    assert len(res.skipped) == 1
    assert res.skipped[0]["symbol"] == "A02"
    assert res.skipped[0]["reason"] == "sell_without_holding"
    assert res.skipped[0]["clipped"] == pytest.approx(0.0)
    assert res.skipped[0]["wanted"] == pytest.approx(-100.0)


def test_partial_clip_sells_only_what_is_held():
    """Sell 250 of a 200-share holding → fill -200, one reported clip."""
    ex = _ex(cash=10_000.0)
    ex.restore(10_000.0, {"A02": 200.0})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -250.0}], pd.Series({"A02": 10.0}), DAY
    )
    assert _sell_shares(res) == [-200.0]
    assert res.positions == {}
    assert len(res.skipped) == 1
    entry = res.skipped[0]
    assert entry["symbol"] == "A02"
    assert entry["reason"] == "sell_exceeds_holding"
    assert entry["wanted"] == pytest.approx(-250.0)
    assert entry["clipped"] == pytest.approx(-200.0)


def test_sell_of_an_unheld_symbol_is_never_filled():
    ex = _ex(cash=10_000.0)
    ex.restore(10_000.0, {})
    res = ex.execute_orders(
        [{"symbol": "A03", "shares": -100.0}], pd.Series({"A03": 5.0}), DAY
    )
    assert res.fills == []
    assert res.positions == {}
    assert res.cash == pytest.approx(10_000.0)
    assert res.skipped == [
        {"symbol": "A03", "reason": "sell_without_holding",
         "wanted": -100.0, "clipped": 0.0}
    ]


def test_covered_sell_fills_fully_and_reports_nothing():
    ex = _ex(cash=1_000.0)
    ex.restore(1_000.0, {"A02": 200.0})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -200.0}], pd.Series({"A02": 10.0}), DAY
    )
    assert _sell_shares(res) == [-200.0]
    assert res.skipped == []
    assert res.positions == {}
    assert res.cash > 1_000.0  # proceeds landed


def test_buy_after_a_clipped_sell_still_executes_within_cash():
    """A clipped sell funds the buys — the buy pass is unaffected by the clip."""
    ex = _ex(cash=100.0)
    ex.restore(100.0, {"A02": 200.0})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -250.0},   # clipped to -200
         {"symbol": "A01", "shares": 300.0}],   # funded by the proceeds
        pd.Series({"A02": 10.0, "A01": 5.0}),
        DAY,
    )
    assert [f.side for f in res.fills] == ["sell", "buy"]  # sells settle first
    assert res.fills[0].shares == pytest.approx(-200.0)
    assert res.fills[1].shares > 0.0
    assert res.fills[1].shares % 100 == 0
    assert res.positions["A01"] == res.fills[1].shares
    assert res.cash >= 0.0
    assert [s["symbol"] for s in res.skipped] == ["A02"]


# -- execute(): opt-in long-only clip --------------------------------------


def _targets_and_prices():
    prices = pd.DataFrame([{"A02": 10.0}], index=[DAY])
    targets = pd.DataFrame([{"A02": -0.5}], index=[DAY])  # wants a short leg
    return targets, prices


def test_execute_long_only_clips_the_planned_short():
    targets, prices = _targets_and_prices()
    ex = OrderExecutor(cash=10_000.0, max_position_pct=0.5, notional_floor=0.0,
                       commission_bps=2.5, min_commission=5.0, long_only=True)
    ex.restore(10_000.0, {"A02": 100.0})
    res = ex.execute(targets, prices, equity=10_000.0)
    # plan would have been -600 (0.5 * 10_000 / 10 = 500 short + 100 held)
    assert _sell_shares(res) == [-100.0]
    assert res.positions == {}
    assert all(v >= 0.0 for v in res.positions.values())
    assert len(res.skipped) == 1
    assert res.skipped[0]["reason"] == "sell_exceeds_holding"
    assert res.skipped[0]["wanted"] == pytest.approx(-600.0)
    assert res.skipped[0]["clipped"] == pytest.approx(-100.0)


def test_execute_default_still_allows_the_short_book():
    """Retired long/short factor books (A/B/C) keep their semantics: the
    long-only clip is OFF unless explicitly requested."""
    targets, prices = _targets_and_prices()
    ex = OrderExecutor(cash=10_000.0, max_position_pct=0.5, notional_floor=0.0,
                       commission_bps=2.5, min_commission=5.0)  # long_only default
    assert ex.long_only is False
    ex.restore(10_000.0, {"A02": 100.0})
    res = ex.execute(targets, prices, equity=10_000.0)
    assert _sell_shares(res) == [-600.0]
    assert res.positions == {"A02": -500.0}
    assert res.skipped == []


# -- reporting -------------------------------------------------------------


def test_to_dict_exposes_skipped_orders():
    ex = _ex(cash=10_000.0)
    ex.restore(10_000.0, {"A02": 100.0})
    res = ex.execute_orders(
        [{"symbol": "A02", "shares": -100.0}, {"symbol": "A02", "shares": -100.0}],
        pd.Series({"A02": 10.0}),
        DAY,
    )
    d = res.to_dict()
    assert "skipped" in d
    assert len(d["skipped"]) == 1
    assert d["skipped"][0]["reason"] == "sell_without_holding"
    json.dumps(d)  # must stay JSON-serialisable for the ledger/report layer
    # a clean execution reports an empty list, not a missing key
    clean = _ex(cash=1_000.0)
    clean.restore(1_000.0, {"A02": 200.0})
    ok = clean.execute_orders(
        [{"symbol": "A02", "shares": -200.0}], pd.Series({"A02": 10.0}), DAY
    )
    assert ok.to_dict()["skipped"] == []
