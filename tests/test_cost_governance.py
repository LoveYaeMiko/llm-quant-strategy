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
    ex = OrderExecutor(cash=100_000, seed=1, notional_floor=3000)
    ex.restore(100_000, {"AAA": 5})  # 5 shares at 1000 = 5% weight (at the cap)
    # tiny adjustment: +0.1 share (~100 yuan) below the floor → skipped
    t = _targets({"AAA": 0.051})
    r = ex.execute(t, _prices({"AAA": 1000.0}), equity=100_000)
    assert len(r.fills) == 0
    assert ex.positions["AAA"] == 5  # untouched
    # exit always executes regardless of size
    t2 = _targets({"AAA": 0.0})
    r2 = ex.execute(t2, _prices({"AAA": 1000.0}), equity=100_000)
    assert len(r2.fills) == 1
    assert "AAA" not in ex.positions


def test_notional_floor_allows_large_entry():
    ex = OrderExecutor(cash=100_000, seed=1, notional_floor=3000)
    t = _targets({"BBB": 0.05})  # 5k notional > floor
    r = ex.execute(t, _prices({"BBB": 100.0}), equity=100_000)
    assert len(r.fills) == 1
    assert "BBB" in ex.positions


def test_band_rebalancing_holds_small_drift():
    ex = OrderExecutor(cash=100_000, seed=1, band_frac=0.002)
    ex.restore(100_000, {"AAA": 5})  # 5% weight at 1000
    # target 4.9% — drift 0.1% inside the 0.2% band → held
    r = ex.execute(_targets({"AAA": 0.049}), _prices({"AAA": 1000.0}), equity=100_000)
    assert len(r.fills) == 0
    # target 4.0% — drift 1.0% outside the band → traded
    r2 = ex.execute(_targets({"AAA": 0.04}), _prices({"AAA": 1000.0}), equity=100_000)
    assert len(r2.fills) == 1


def test_defaults_keep_legacy_behaviour():
    ex = OrderExecutor(cash=100_000, seed=1)
    t = _targets({"AAA": 0.001})  # tiny fill with no governance → still trades
    r = ex.execute(t, _prices({"AAA": 1000.0}), equity=100_000)
    assert len(r.fills) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
