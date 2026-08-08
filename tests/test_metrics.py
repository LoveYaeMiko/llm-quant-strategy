"""Metrics tests — IC, ICIR, Sharpe, MaxDD, Bonferroni significance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as M


def _panel(n_dates=60, n_symbols=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_symbols)]], names=["date", "symbol"])
    sig = rng.normal(0, 1, len(idx))
    fwd = sig + rng.normal(0, 1, len(idx))  # correlated with signal
    return pd.Series(sig, index=idx), pd.Series(fwd, index=idx)


def test_daily_ic_positive_when_correlated():
    sig, fwd = _panel()
    ic = M.daily_ic(sig, fwd)
    assert ic.mean() > 0.1
    assert ic.index.name == "date"


def test_mean_ic_and_icir():
    sig, fwd = _panel()
    assert M.mean_ic(sig, fwd) > 0.1
    ic = M.daily_ic(sig, fwd)
    assert M.icir(ic) > 0.0


def test_sharpe_and_drawdown():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    assert M.sharpe_ratio(rets) > 0.0
    assert 0.0 <= M.max_drawdown(rets) < 0.3
    assert M.annualized_return(rets) > 0.0


def test_annualized_return_robust_to_large_losses():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0, 0.05, 300))
    rets.iloc[0] = -0.999  # near-total loss — must not go complex
    val = M.annualized_return(rets)
    assert isinstance(val, float) and val <= 0.0


def test_turnover_of_constant_weights_zero():
    w = pd.DataFrame(0.1, index=pd.date_range("2020-01-01", periods=5), columns=list("ABC"))
    assert M.turnover(w) == 0.0


def test_bonferroni_raises_bar_with_trials():
    single = M.significance_threshold_sharpe(n_days=250, n_trials=1)
    many = M.significance_threshold_sharpe(n_days=250, n_trials=100)
    assert many > single


def test_is_significant_respects_trials():
    rng = np.random.default_rng(0)
    # drift chosen so the realised Sharpe (≈2.1) clears the 1-trial bar (~1.39)
    # but not the 10k-trial Bonferroni bar (~3.24)
    rets = pd.Series(rng.normal(0.0016, 0.01, 500))
    assert bool(M.is_significant(rets, n_trials=1)[0]) is True
    assert bool(M.is_significant(rets, n_trials=10000)[0]) is False


def test_factor_eval_bundle():
    sig, fwd = _panel()
    fe = M.factor_eval(sig, fwd, n_trials=1)
    for key in ("ic", "rank_ic", "icir", "sharpe", "max_drawdown", "t_stat", "n_days", "significant"):
        assert key in fe
    assert fe["n_days"] > 0


def test_factor_eval_empty():
    empty = pd.Series(dtype=float)
    fe = M.factor_eval(empty, empty)
    assert fe["n_days"] == 0
    assert fe["significant"] is False
