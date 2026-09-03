"""Deterministic online layer tests (signal calc / optimizer / executor)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.online.order_executor import OrderExecutor
from src.online.portfolio_optimizer import (
    PortfolioOptimizer,
    break_correlations,
    neutralize_pca,
)
from src.online.signal_calculator import compile_factor, compute_signal


def test_compile_and_compute(fctx, forward):
    compiled = compile_factor("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))", name="mom")
    assert compiled.formula == "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    assert 10 in compiled.lookbacks
    scores = compute_signal(compiled, fctx.data)
    assert scores.index.equals(fctx.data.index)


def test_compile_rejects_bad_formula():
    with pytest.raises(Exception):
        compile_factor("bad_op(Close)")


def test_compiled_factor_json_roundtrip():
    compiled = compile_factor("Neg(TS_ZScore(Close, 20))", name="rev")
    blob = compiled.to_json()
    restored = type(compiled).from_json(blob)
    assert restored.formula == compiled.formula
    assert restored.lookbacks == compiled.lookbacks


def _wide_returns(n_dates=20, n_symbols=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_symbols)), index=dates,
                        columns=[f"S{i}" for i in range(n_symbols)])
    return rets


def test_neutralize_pca_removes_market_factor():
    rets = _wide_returns()
    # signal = market factor exactly (all symbols share it)
    mkt = rets.mean(axis=1)
    long = rets.stack()
    long.index.names = ["date", "symbol"]
    mkt_long = mkt.reindex(long.index.get_level_values(0)).values
    sig = pd.Series(mkt_long, index=long.index)
    neut = neutralize_pca(sig, n_components=2)
    # residual should have far smaller cross-sectional dispersion than raw
    assert abs(neut.mean()) < 1e-9


def test_break_correlations_caps_cluster():
    rets = _wide_returns()
    # S0, S1 perfectly correlated -> one cluster
    rets["S1"] = rets["S0"]
    weights = pd.Series(0.5, index=rets.columns)  # S0 and S1 both 0.5
    out = break_correlations(weights, rets, corr_threshold=0.6, cluster_cap=0.3)
    assert abs(out["S0"] + out["S1"]) <= 0.3 + 1e-9


def test_portfolio_optimizer_weights(scores_panel, market):
    opt = PortfolioOptimizer(long_pct=0.2, short_pct=0.2)
    res = opt.optimize(scores_panel)
    gross = res.weights.abs().sum(axis=1)
    assert (gross >= 0).all()
    assert gross.sum() > 0  # at least some dates have positions
    assert res.weights.shape[1] == market.n_symbols
    # no single position above the cap
    cap = opt.max_position_pct
    assert (res.weights.abs() <= cap + 1e-9).all().all()


def test_neutralize_scores_preserves_shape(scores_panel):
    neut = neutralize_pca(scores_panel, n_components=3)
    assert len(neut) == len(scores_panel)
    assert neut.index.equals(scores_panel.index)


def test_order_executor_fills_and_caps(market):
    rng = np.random.default_rng(0)
    dates = sorted(market.long.index.get_level_values(0).unique())[:10]
    symbols = market.long.index.get_level_values(1).unique().tolist()
    targets = pd.DataFrame(
        rng.choice([0.0, 0.05, -0.05], size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    prices = market.price_panel.reindex(dates)
    ex = OrderExecutor(cash=2_000_000.0, max_position_pct=0.05, seed=0)
    res = ex.execute(targets, prices)
    assert len(res.fills) > 0
    # every position within the cap *at the price it was established* (the cap
    # is enforced per execution day; the loosest bound is the window-min price),
    # and every position a whole 100-share board lot
    for sym, shares in res.positions.items():
        assert abs(shares) <= 0.05 * 2_000_000.0 / prices[sym].min() + 1e-6
        assert abs(shares) % 100 == 0
    for f in res.fills:
        assert abs(f.shares) % 100 == 0  # entries are whole lots (no full closes here)
        assert round(f.price * 100, 6) % 1 == 0  # 0.01 tick


def test_order_executor_blacklist(market):
    dates = sorted(market.long.index.get_level_values(0).unique())[:5]
    symbols = market.long.index.get_level_values(1).unique().tolist()
    targets = pd.DataFrame(0.05, index=dates, columns=symbols)
    prices = market.price_panel.reindex(dates)
    ex = OrderExecutor(blacklist={symbols[0]}, cash=100_000.0, seed=0)
    res = ex.execute(targets, prices)
    assert symbols[0] not in res.positions


def test_order_executor_deterministic():
    from src.data.synthetic import make_synthetic_market

    m1 = make_synthetic_market(symbols=12, days=30, seed=3)
    m2 = make_synthetic_market(symbols=12, days=30, seed=3)
    dates = sorted(m1.long.index.get_level_values(0).unique())[:8]
    syms = m1.long.index.get_level_values(1).unique().tolist()
    targets = pd.DataFrame(np.tile([0.05, -0.05, 0.0], (len(dates), len(syms)))[:, : len(syms)],
                           index=dates, columns=syms)
    r1 = OrderExecutor(cash=100_000.0, seed=0).execute(targets, m1.price_panel.reindex(dates))
    r2 = OrderExecutor(cash=100_000.0, seed=0).execute(targets, m2.price_panel.reindex(dates))
    assert r1.to_dict() == r2.to_dict()
