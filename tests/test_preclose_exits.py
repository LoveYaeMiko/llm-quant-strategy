"""Regression tests for defect D-5 (2026-09-08 audit).

The 14:55 preclose order layer built its target row from
``PullbackPortfolio.compute_weights`` ONLY. That dict contains just the names
the book still WANTS to hold, while ``OrderExecutor.execute`` iterates over the
target frame's COLUMNS. A name the book exited (stop / trail / trend gate /
max_hold / strength exit) was therefore absent from the frame and the executor
never sold it — the closing-auction layer could not liquidate anything.

``merge_targets`` pins every currently-held symbol at 0.0 before the book's
weights override, so an exit becomes an explicit 0.0 target weight.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.online.order_executor import OrderExecutor
from src.preclose import merge_targets


def test_held_but_unwanted_name_is_pinned_to_zero():
    weights = {"600000.SH": 0.16, "000001.SZ": 0.16}
    positions = {"600000.SH": 1300.0, "601872.SH": 1300.0}  # 601872 exited
    row = merge_targets(weights, positions)
    assert row["601872.SH"] == 0.0          # exit is now expressible
    assert row["600000.SH"] == pytest.approx(0.16)  # held + still wanted
    assert row["000001.SZ"] == pytest.approx(0.16)  # new entry


def test_book_weight_overrides_zero_pin():
    # A held name the book still wants must keep its weight, not stay at 0.0.
    row = merge_targets({"600000.SH": 0.4}, {"600000.SH": 100.0})
    assert row == {"600000.SH": 0.4}


def test_empty_book_with_positions_still_sells_everything():
    row = merge_targets({}, {"600000.SH": 100.0, "601872.SH": 1300.0})
    assert row == {"600000.SH": 0.0, "601872.SH": 0.0}


def test_no_positions_and_no_candidates_yields_no_orders():
    assert merge_targets({}, {}) == {}


def test_executor_ignores_names_absent_from_targets():
    """Documents the underlying executor contract that caused D-5."""
    day = pd.Timestamp("2026-09-08")
    ex = OrderExecutor(cash=50_000.0, notional_floor=2_000.0, max_position_pct=0.4)
    ex.restore(50_000.0, {"601872.SH": 1300.0})
    prices = pd.DataFrame([{"601872.SH": 19.08}], index=[day])
    # Old behaviour: the exited name is simply missing from the frame.
    res_absent = ex.execute(pd.DataFrame([{}], index=[day]), prices, equity=50_000.0)
    assert res_absent.fills == []

    # Fixed behaviour: the same name pinned at 0.0 produces the exit fill.
    ex2 = OrderExecutor(cash=50_000.0, notional_floor=2_000.0, max_position_pct=0.4)
    ex2.restore(50_000.0, {"601872.SH": 1300.0})
    targets = pd.DataFrame([merge_targets({}, {"601872.SH": 1300.0})], index=[day])
    res_pinned = ex2.execute(targets, prices, equity=50_000.0)
    assert len(res_pinned.fills) == 1
    fill = res_pinned.fills[0]
    assert fill.side == "sell"
    assert fill.symbol == "601872.SH"
    assert fill.shares == pytest.approx(-1300.0)
