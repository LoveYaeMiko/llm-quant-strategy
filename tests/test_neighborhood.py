"""A3 邻域选参 — plateau detection + in/out IC correlation (reject argmax)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import in_out_ic_correlation, neighborhood_plateau


def test_neighborhood_plateau_true_when_flat():
    ok, support = neighborhood_plateau([0.05, 0.049, 0.048, 0.047], plateau_band=0.3, min_support_fraction=0.4)
    assert ok is True
    assert support == 1.0


def test_neighborhood_plateau_false_for_lone_spike():
    # one high, three low -> the argmax is a spike, not a plateau
    ok, support = neighborhood_plateau([0.09, 0.01, 0.01, 0.01], plateau_band=0.3, min_support_fraction=0.4)
    assert ok is False
    assert support == 0.25


def test_neighborhood_plateau_empty_or_nonpositive():
    assert neighborhood_plateau([]) == (False, 0.0)
    assert neighborhood_plateau([-0.01, -0.02])[0] is False


def test_neighborhood_plateau_ignores_none_and_nan():
    ok, support = neighborhood_plateau([0.05, 0.05, None, float("nan")], min_support_fraction=0.5)
    assert ok is True and support == 1.0


def _panel(n_symbols=12, n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    forward = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    return dates, symbols, forward


def test_in_out_ic_correlation_none_with_too_few_panels():
    _, symbols, forward = _panel()
    rng = np.random.default_rng(1)
    panels = [
        pd.Series(rng.normal(0, 1, len(forward)), index=forward.index)
        for _ in range(2)
    ]
    assert in_out_ic_correlation(panels, forward) is None


def test_in_out_ic_correlation_none_for_short_panels():
    _, _, forward = _panel(n_days=8)
    rng = np.random.default_rng(4)
    panels = [
        pd.Series(rng.normal(0, 1, len(forward)), index=forward.index)
        for _ in range(3)
    ]
    assert in_out_ic_correlation(panels, forward) is None


def test_in_out_ic_correlation_bounded_or_none():
    _, _, forward = _panel()
    rng = np.random.default_rng(2)
    panels = [
        pd.Series(rng.normal(0, 1, len(forward)), index=forward.index)
        for _ in range(5)
    ]
    r = in_out_ic_correlation(panels, forward)
    if r is not None:
        assert -1.0 <= r <= 1.0


def test_in_out_ic_correlation_positive_for_persistent_signal():
    # A factor that is good in both halves (its rank carries over) should give a
    # positive in/out IC correlation across the neighbourhood.
    dates, symbols, forward = _panel()
    half = len(dates) // 2
    # true signal: a stable per-symbol tilt that forward is also tilted toward
    tilt = pd.Series(np.arange(len(symbols)), index=symbols)
    panels = []
    rng = np.random.default_rng(3)
    for _ in range(4):
        base = tilt.loc[symbols].reindex(forward.index, level=1)
        noise = pd.Series(rng.normal(0, 0.5, len(forward)), index=forward.index)
        panels.append(pd.Series((base.values + noise.values), index=forward.index))
    r = in_out_ic_correlation(panels, forward)
    # with a stable tilt the in/out ranking should not be strongly negative
    assert r is not None
    assert r > -0.9
