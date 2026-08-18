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


# -- Deflated Sharpe (Bailey & López de Prado) -------------------------------


def _returns(seed=0, mean=0.002, std=0.01, n=500):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_expected_max_sharpe_monotonic():
    # more trials OR wider trial variance both raise the luck baseline
    assert M.expected_max_sharpe(10, 0.5) < M.expected_max_sharpe(1000, 0.5)
    assert M.expected_max_sharpe(100, 0.1) < M.expected_max_sharpe(100, 1.0)
    assert M.expected_max_sharpe(1, 1.0) == 0.0  # no trials -> no correction


def test_deflated_sharpe_significant_vs_insignificant():
    strong = _returns(seed=0, mean=0.002, std=0.01)  # ann Sharpe ≈ 3.2
    weak = _returns(seed=1, mean=0.0003, std=0.01)   # ann Sharpe ≈ 0.5
    tight = [1.0, 1.05, 0.95, 1.02]
    hi = M.deflated_sharpe_ratio(
        strong, n_trials=4, trial_sharpe_variance=float(np.var(tight, ddof=1))
    )
    lo = M.deflated_sharpe_ratio(weak, n_trials=1000, trial_sharpe_variance=0.64)
    assert hi["deflated_sharpe"] is not None and hi["deflated_sharpe"] > 0.9
    assert lo["deflated_sharpe"] is not None and lo["deflated_sharpe"] < 0.5


def test_deflated_sharpe_returns_none_when_undefined():
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0, 0.01, 10))  # too few days
    flat = pd.Series(np.zeros(100))             # no variance
    assert M.deflated_sharpe_ratio(short, 100, 0.5)["deflated_sharpe"] is None
    assert M.deflated_sharpe_ratio(flat, 100, 0.5)["deflated_sharpe"] is None


def test_factor_eval_tail_spread_and_deflated():
    sig, fwd = _panel()
    fe = M.factor_eval(sig, fwd, n_trials=1)
    # tail spread surfaced; DSR absent without a trial set
    assert "tail_spread" in fe and fe["tail_spread"] > 0.0
    assert fe["deflated_sharpe"] is None
    fe2 = M.factor_eval(sig, fwd, n_trials=4, trial_sharpes=[1.0, 1.1, 0.9, 1.05])
    assert fe2["deflated_sharpe"] is not None
    assert 0.0 <= fe2["deflated_sharpe"] <= 1.0
