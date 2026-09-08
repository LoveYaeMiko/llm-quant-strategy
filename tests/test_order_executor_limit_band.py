"""Regression tests for defect 2.4 (2026-09-08 audit): hardcoded limit bands.

``OrderExecutor`` used ``0.20 if symbol[:3] in ("688","689","300","301") else
0.10`` in two places — date-unaware (创业板 traded a 10% band before
2020-08-24) and missing 北交所 (30%). Both call sites now use the shared
``board_limit`` helper, the same source of truth as the backtest mask.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.limit_locked import board_limit
from src.online.order_executor import OrderExecutor


def test_board_limit_is_date_and_board_aware():
    pre = pd.Timestamp("2020-01-02")
    post = pd.Timestamp("2021-01-04")
    assert board_limit("300001.SZ", pre) == pytest.approx(0.095)
    assert board_limit("300001.SZ", post) == pytest.approx(0.195)
    assert board_limit("301001.SZ", pre) == pytest.approx(0.095)
    assert board_limit("688001.SH", pre) == pytest.approx(0.195)
    assert board_limit("830001.BJ", post) == pytest.approx(0.295)
    assert board_limit("430001.BJ", post) == pytest.approx(0.295)
    assert board_limit("600000.SH", post) == pytest.approx(0.095)
    assert board_limit("300001.SZ", pre, dynamic=False) == pytest.approx(0.095)


def _sell_executor() -> OrderExecutor:
    ex = OrderExecutor(cash=100_000.0, notional_floor=0.0, max_position_pct=1.0)
    ex.restore(100_000.0, {"300001.SZ": 1000.0})
    return ex


def test_chinext_10pct_drop_was_locked_before_2020_but_not_after():
    pre = pd.Timestamp("2020-01-02")
    post = pd.Timestamp("2021-01-04")
    prices = pd.DataFrame([{"300001.SZ": 90.0}], index=[pre])
    res = _sell_executor().execute_orders(
        [{"symbol": "300001.SZ", "shares": -1000.0}], prices.iloc[0], pre,
        limit_locked=pd.Series({"300001.SZ": -0.10}),
    )
    assert res.fills == []                       # 10% band → locked

    prices2 = pd.DataFrame([{"300001.SZ": 90.0}], index=[post])
    res2 = _sell_executor().execute_orders(
        [{"symbol": "300001.SZ", "shares": -1000.0}], prices2.iloc[0], post,
        limit_locked=pd.Series({"300001.SZ": -0.10}),
    )
    assert len(res2.fills) == 1                  # 20% band → tradeable


def test_bse_30pct_band_is_not_treated_as_locked():
    day = pd.Timestamp("2026-09-08")
    ex = OrderExecutor(cash=100_000.0, notional_floor=0.0, max_position_pct=1.0)
    ex.restore(100_000.0, {"830001.BJ": 1000.0})
    prices = pd.DataFrame([{"830001.BJ": 9.0}], index=[day])
    res = ex.execute_orders(
        [{"symbol": "830001.BJ", "shares": -1000.0}], prices.iloc[0], day,
        limit_locked=pd.Series({"830001.BJ": -0.10}),
    )
    assert len(res.fills) == 1                   # 30% band → not locked


def test_close_executor_uses_the_same_band():
    pre = pd.Timestamp("2020-01-02")
    post = pd.Timestamp("2021-01-04")
    for day, expect in ((pre, 0), (post, 1)):
        ex = OrderExecutor(cash=100_000.0, notional_floor=0.0, max_position_pct=1.0)
        ex.restore(100_000.0, {"300001.SZ": 1000.0})
        targets = pd.DataFrame([{"300001.SZ": 0.0}], index=[day])
        prices = pd.DataFrame([{"300001.SZ": 90.0}], index=[day])
        res = ex.execute(targets, prices, equity=90_000.0,
                         limit_locked=pd.Series({"300001.SZ": -0.10}))
        assert len(res.fills) == expect
