"""PIT-aware long-short backtester tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, PointInTimeBacktest


def _panel(n_dates=40, n_symbols=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_symbols)]], names=["date", "symbol"])
    sig = rng.normal(0, 1, len(idx))
    fwd = 0.3 * sig + rng.normal(0, 1, len(idx))
    return pd.Series(sig, index=idx), pd.Series(fwd, index=idx)


def test_backtest_runs_and_reports():
    sig, fwd = _panel()
    bt = PointInTimeBacktest()
    res = bt.run(sig, fwd)
    assert res.weights.shape[0] == 40
    assert res.weights.shape[1] <= 12
    assert "sharpe" in res.metrics
    assert res.metrics["n_days"] == 40


def test_signal_with_predictive_power_beats_noise():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=60)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(16)]], names=["date", "symbol"])
    fwd = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    predictive = fwd + rng.normal(0, 0.001, len(idx))  # almost perfect signal
    noise = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    bt = PointInTimeBacktest()
    good = bt.run(predictive, fwd).metrics
    bad = bt.run(noise, fwd).metrics
    assert good["sharpe"] > bad["sharpe"]


def test_position_cap_enforced():
    sig, fwd = _panel()
    bt = PointInTimeBacktest(BacktestConfig(max_position_pct=0.05))
    res = bt.run(sig, fwd)
    assert res.weights.abs().max().max() <= 0.05 + 1e-9


def test_turnover_reported():
    sig, fwd = _panel()
    res = PointInTimeBacktest().run(sig, fwd)
    assert res.metrics["turnover"] >= 0.0


def test_benchmark_metrics():
    sig, fwd = _panel()
    bench = pd.Series(np.full(40, 0.0005), index=sorted(fwd.index.get_level_values(0).unique()))
    res = PointInTimeBacktest().run(sig, fwd, benchmark=bench)
    assert "excess_sharpe" in res.metrics
