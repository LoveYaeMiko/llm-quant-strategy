"""Tests for the model ensemble (src/ml/ensemble.py) — offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble import cross_sectional_rank, ensemble_scores, rank_ensemble


def _scores(values: np.ndarray, seed: int = 0) -> pd.Series:
    dates = pd.bdate_range("2024-01-01", periods=30)
    syms = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=len(idx)), index=idx)


def test_cross_sectional_rank_per_date():
    s = _scores(np.zeros(30 * 10))
    r = cross_sectional_rank(s)
    # every date ranks 0..1
    for d in s.index.get_level_values(0).unique():
        day = r.loc[d]
        assert day.min() > 0.0 and day.max() <= 1.0


def test_rank_ensemble_is_rank_average():
    s1 = _scores(None, seed=1)
    s2 = _scores(None, seed=2)
    e = rank_ensemble([s1, s2])
    expect = (cross_sectional_rank(s1) + cross_sectional_rank(s2)) / 2
    assert np.allclose(e, expect, atol=1e-12)


def test_ensemble_ignores_empty_model():
    s1 = _scores(None, seed=1)
    s2 = pd.Series(dtype=float, index=s1.index)
    e, used = ensemble_scores([("m1", s1), ("m2", s2)])
    assert used == ["m1"]
    assert np.allclose(e, cross_sectional_rank(s1), atol=1e-12)


def test_ensemble_all_empty_raises():
    idx = _scores(None).index
    with pytest.raises(ValueError):
        rank_ensemble([pd.Series(dtype=float, index=idx)])
