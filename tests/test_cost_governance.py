"""Tests for the cost-governance executor (notional floor + band rebalancing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.online.order_executor import OrderExecutor


def _targets(w: dict[str, float], date: str = "2024-01-02") -> pd.DataFrame:
    return pd.DataFrame([w], index=pd.Index([date], name="date"))


def _prices(px: dict[str, float], date: str = "2024-01-02") -> pd.DataFrame:
    return pd.DataFrame([px], index=pd.Index([date], name="date"))


def test_notional_floor_skips_dust_but_not_exits():
    ex = OrderExecutor(cash=2_000_000, seed=1, notional_floor=3000)
    ex.restore(2_000_000, {"AAA": 100})  # 100 shares at 1000 = 5% weight (at the cap)
    # tiny adjustment: +2 shares (~2k yuan) below the floor → skipped
    t = _targets({"AAA": 0.051})
    r = ex.execute(t, _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r.fills) == 0
    assert ex.positions["AAA"] == 100  # untouched
    # exit always executes regardless of size (odd lot allowed on full close)
    t2 = _targets({"AAA": 0.0})
    r2 = ex.execute(t2, _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r2.fills) == 1
    assert "AAA" not in ex.positions


def test_notional_floor_allows_large_entry():
    ex = OrderExecutor(cash=2_000_000, seed=1, notional_floor=3000)
    t = _targets({"BBB": 0.05})  # 1000 shares / 100k notional > floor
    r = ex.execute(t, _prices({"BBB": 100.0}), equity=2_000_000)
    assert len(r.fills) == 1
    assert "BBB" in ex.positions
    assert ex.positions["BBB"] % 100 == 0


def test_band_rebalancing_holds_small_drift():
    ex = OrderExecutor(cash=2_000_000, seed=1, band_frac=0.002)
    ex.restore(2_000_000, {"AAA": 200})  # 2 board lots at 1000 (10% weight)
    # target 9.9% — drift 0.1% inside the band → held
    r = ex.execute(_targets({"AAA": 0.099}), _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r.fills) == 0
    # target 5.0% — drift 5% outside the band → one lot sold (board-lot rounding)
    r2 = ex.execute(_targets({"AAA": 0.05}), _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r2.fills) == 1
    assert ex.positions["AAA"] == 100


def test_board_lot_rounds_tiny_fills_away():
    ex = OrderExecutor(cash=2_000_000, seed=1)
    t = _targets({"AAA": 0.001})  # 2 shares → rounds to 0, no fill
    r = ex.execute(t, _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r.fills) == 0
    t2 = _targets({"AAA": 0.05})  # 100 shares → a whole lot
    r2 = ex.execute(t2, _prices({"AAA": 1000.0}), equity=2_000_000)
    assert len(r2.fills) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
