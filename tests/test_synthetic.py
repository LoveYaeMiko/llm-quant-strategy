"""Synthetic market generator sanity tests."""

from __future__ import annotations

import pandas as pd

from src.data.synthetic import make_synthetic_market


def test_panels_well_formed():
    m = make_synthetic_market(symbols=20, days=100, seed=2)
    assert m.price_panel.shape == (100, 20)
    assert not m.price_panel.isna().all().all()
    assert m.long.index.names == ["date", "symbol"]
    assert {"open", "high", "low", "close", "volume"} <= set(m.long.columns)
    assert m.n_symbols == 20
    assert m.n_days == 100


def test_forward_returns_aligned():
    m = make_synthetic_market(symbols=10, days=60, seed=3)
    assert m.forward_returns.index.names == ["date", "symbol"]
    # stack() drops all-NaN rows, so the last trading date (no next-day
    # return) is absent; everything present must be finite
    dates = sorted(m.forward_returns.index.get_level_values(0).unique())
    assert dates[-1] < m.long.index.get_level_values(0).max()
    assert m.forward_returns.isna().sum() == 0


def test_pit_query_never_sees_future():
    m = make_synthetic_market(symbols=10, days=120, seed=4)
    from src.data.point_in_time_loader import VALID_FROM

    q = m.pit_store.query("2020-04-01")
    assert (pd.to_datetime(q[VALID_FROM]) <= pd.Timestamp("2020-04-01")).all()


def test_synthetic_factor_has_measurable_ic():
    """A momentum factor on the synthetic market should carry non-zero IC."""
    m = make_synthetic_market(symbols=30, days=200, seed=9)
    from src.factors.code_generator import FactorContext, eval_expression

    fctx = FactorContext(m.long)
    scores = eval_expression("TS_Return(Close, 10)", fctx)
    from src.backtest.metrics import mean_ic

    ic = mean_ic(scores, m.forward_returns)
    assert abs(ic) > 0.01  # persistent market factor => measurable signal
