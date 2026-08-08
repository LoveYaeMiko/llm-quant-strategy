"""Evaluation metrics — IC, RankIC, ICIR, Sharpe, MaxDD, turnover.

Thresholds live in ``configs/factor_thresholds.yaml`` (single source of truth);
this module only computes. FINSABER compliance: multiple-hypothesis correction
is applied to Sharpe significance when many factors were tried (blueprint §4C).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IC metrics
# ---------------------------------------------------------------------------


def daily_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Per-date IC between a signal and forward returns.

    Both series are indexed by ``(date, symbol)``. IC is computed cross-sectionally
    per date, which is what quant literature means by factor IC.
    """
    df = pd.DataFrame({"sig": signal, "fwd": forward_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        if method == "spearman":
            # rank-based Pearson — numpy/pandas only, no scipy required
            return float(g["sig"].rank().corr(g["fwd"].rank(), method="pearson"))
        return float(g["sig"].corr(g["fwd"], method=method))

    out = df.groupby(level=0).apply(_ic)
    if isinstance(out.index, pd.MultiIndex):
        out = out.droplevel(-1)
    return out.rename("ic")


def mean_ic(signal: pd.Series, forward_returns: pd.Series, method: str = "spearman") -> float:
    return float(daily_ic(signal, forward_returns, method).mean())


def icir(ic_series: pd.Series, periods_per_year: int = 252) -> float:
    """ICIR = mean(IC) / std(IC), annualised by sqrt(periods)."""
    ic = ic_series.dropna()
    if len(ic) < 2 or float(ic.std()) == 0:
        return 0.0
    return float(ic.mean() / ic.std() * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Return-based metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or float(r.std()) == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Peak-to-trough drawdown of a return series (positive fraction)."""
    r = returns.fillna(0.0)
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(-dd.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    # log-based compounding: robust to large returns and never yields complex
    if (r <= -1).any():
        return float("-inf")
    log_total = float(np.log1p(r).sum())
    return float(np.exp(log_total * periods_per_year / max(len(r), 1)) - 1.0)


def turnover(weights: pd.DataFrame) -> float:
    """Mean gross weight change per rebalance — a proxy for trading cost."""
    if weights.empty:
        return 0.0
    w = weights.fillna(0.0)
    changes = w.diff().abs().sum(axis=1)
    return float(changes.mean())


def t_statistic(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or float(r.std()) == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(len(r)))


# ---------------------------------------------------------------------------
# Multiple-hypothesis significance (FINSABER compliance)
# ---------------------------------------------------------------------------


def significance_threshold_sharpe(
    n_days: int,
    n_trials: int,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Bonferroni-adjusted Sharpe required for significance given ``n_trials``.

    Trying 100 factors and keeping the best 5 is data snooping; Bonferroni makes
    the bar: required t ≈ 1.96 becomes wider as trials grow (blueprint §4C).
    """
    if n_trials <= 1:
        return 1.96 * np.sqrt(periods_per_year) / np.sqrt(max(n_days, 1))
    # Bonferroni-critical z for alpha/2 per trial
    z = _normal_quantile(1.0 - alpha / (2.0 * n_trials))
    return z * np.sqrt(periods_per_year) / np.sqrt(max(n_days, 1))


def _normal_quantile(p: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:  # pragma: no cover - scipy absent
        # crude approximation valid for p in (0.9, 0.9999)
        t = np.sqrt(np.log(1.0 / (1.0 - p) ** 2))
        return float(t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                     / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))


def is_significant(
    returns: pd.Series,
    n_trials: int = 1,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> tuple[bool, float]:
    """Bonferroni-adjusted significance check. Returns (pass, required_sharpe)."""
    r = returns.dropna()
    if len(r) < 2:
        return False, 0.0
    required = significance_threshold_sharpe(len(r), n_trials, alpha, periods_per_year)
    return float(sharpe_ratio(r, periods_per_year)) >= required, required


def factor_eval(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    n_trials: int = 1,
) -> dict:
    """One-shot evaluation bundle used by the eval agent."""
    ic = daily_ic(signal, forward_returns, method="spearman")
    forward = forward_returns.reindex(signal.index)
    df = pd.DataFrame({"sig": signal, "fwd": forward}).dropna()
    if df.empty:
        return {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "n_days": 0, "significant": False}
    # portfolio long-short return proxy for Sharpe/maxdd
    sig = df["sig"]
    fwd = df["fwd"]
    rank = sig.groupby(level=0).rank(pct=True)
    top = rank >= 0.9
    bot = rank <= 0.1
    ls_ret = ((top.astype(float) - bot.astype(float)) * fwd).groupby(level=0).mean()
    return {
        "ic": float(ic.mean()),
        "rank_ic": mean_ic(signal, forward_returns, "spearman"),
        "icir": icir(ic),
        "sharpe": sharpe_ratio(ls_ret),
        "max_drawdown": max_drawdown(ls_ret),
        "t_stat": t_statistic(ls_ret),
        "n_days": int(len(ic.dropna())),
        "significant": is_significant(ls_ret, n_trials)[0],
        "required_sharpe": significance_threshold_sharpe(
            max(len(ls_ret), 1), n_trials
        ),
    }
