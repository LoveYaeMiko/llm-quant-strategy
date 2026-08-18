"""Evaluation metrics — IC, RankIC, ICIR, Sharpe, MaxDD, turnover.

Thresholds live in ``configs/factor_thresholds.yaml`` (single source of truth);
this module only computes. FINSABER compliance: multiple-hypothesis correction
is applied to Sharpe significance when many factors were tried (blueprint §4C).
"""

from __future__ import annotations

import math
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


def max_drawdown_from_curve(curve: pd.Series) -> float:
    """Peak-to-trough drawdown of a level series (positive fraction).

    Used for the validation_BLUEPRINT excess drawdown: ``curve`` is the cumulative
    strategy/benchmark ratio minus 1 (a level that can cross zero), so the drop
    is the largest *difference* between the level and its running maximum —
    ``(curve - curve.cummax()).min()`` — matching the blueprint's
    ``excess_returns - running_max``. A ratio form (``curve / running_max``)
    would explode when the level crosses zero.
    """
    c = curve.fillna(0.0)
    return float(-(c - c.cummax()).min())


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


def _normal_cdf(x: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.cdf(x))
    except ImportError:  # pragma: no cover - scipy absent
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & López de Prado) — corrects the "best of N" luck
# ---------------------------------------------------------------------------


def expected_max_sharpe(
    n_trials: int,
    trial_sharpe_variance: float,
    periods_per_year: int = 252,
) -> float:
    """SR₀ — max annualised Sharpe reachable by pure luck over ``n_trials``.

    ``trial_sharpe_variance`` is the variance of the annualised Sharpe across all
    trials tried. ``n_trials`` must be the **effective independent** trial count
    (after family blocking), *not* the raw generated count — the latter
    over-penalises and kills real factors (blueprint §4C / EP004 lesson).
    """
    if n_trials <= 1:
        return 0.0
    var = float(trial_sharpe_variance)
    if not np.isfinite(var) or var <= 0:
        return 0.0
    n = int(n_trials)
    gamma = 0.5772156649
    e = math.e
    z1 = _normal_quantile(1.0 - 1.0 / n)
    z2 = _normal_quantile(1.0 - 1.0 / (n * e))
    sr0_daily = math.sqrt(var / periods_per_year) * ((1.0 - gamma) * z1 + gamma * z2)
    return float(sr0_daily * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_variance: float,
    periods_per_year: int = 252,
) -> dict:
    """DSR — PSR of the observed Sharpe vs the max-Sharpe-by-luck baseline.

    Returns a dict so callers carry the diagnosis alongside the value.
    ``deflated_sharpe >= 0.95`` = significant; ``< 0.90`` = "probably searched-out
    luck". Non-positive PSR denominator or short samples yield ``deflated_sharpe``
    of ``None`` (DSR undefined), never a fabricated number.
    """
    r = returns.dropna()
    base = {
        "deflated_sharpe": None,
        "expected_max_sharpe": 0.0,
        "observed_sharpe": 0.0,
        "n_trials": int(n_trials),
    }
    if len(r) < 30 or float(r.std()) == 0:
        return base
    sr_daily = float(r.mean() / r.std())
    sr_annual = float(sr_daily * math.sqrt(periods_per_year))
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0  # pandas returns *excess* kurtosis
    sr0_annual = expected_max_sharpe(n_trials, trial_sharpe_variance, periods_per_year)
    sr0_daily = sr0_annual / math.sqrt(periods_per_year)
    denom = 1.0 - g3 * sr_daily + (g4 - 1.0) / 4.0 * sr_daily ** 2
    if denom <= 0:
        return base
    z = (sr_daily - sr0_daily) * math.sqrt(len(r) - 1) / math.sqrt(denom)
    return {
        "deflated_sharpe": float(_normal_cdf(z)),
        "expected_max_sharpe": sr0_annual,
        "observed_sharpe": sr_annual,
        "n_trials": int(n_trials),
        "t_days": int(len(r)),
        "skew": g3,
        "kurtosis": g4,
    }


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
    # coerce BOTH sides to Python float: when ``required`` is a numpy float64 the
    # reflected comparison (np.float64.__le__) would return a numpy ``np.True_``
    # scalar — whose __class__.__name__ is "bool" but which json.dump cannot
    # serialize ("Object of type bool is not JSON serializable").
    return bool(float(sharpe_ratio(r, periods_per_year)) >= float(required)), float(required)


def tail_long_short_returns(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> pd.Series:
    """Daily top/bottom-decile long-short return — the tradeability proxy.

    EP004: rank-IC is corrupted by mid-book noise; the *traded tail spread* is
    the true criterion (a positive IC with a negative tail spread is a non-
    tradeable level effect). Returns a date-indexed series (empty if nothing
    aligns).
    """
    forward = forward_returns.reindex(signal.index)
    df = pd.DataFrame({"sig": signal, "fwd": forward}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    rank = df["sig"].groupby(level=0).rank(pct=True)
    top = rank >= 0.9
    bot = rank <= 0.1
    return ((top.astype(float) - bot.astype(float)) * df["fwd"]).groupby(level=0).mean()


def factor_eval(
    signal: pd.Series,
    forward_returns: pd.Series,
    *,
    n_trials: int = 1,
    trial_sharpes: Optional[Sequence[float]] = None,
    periods_per_year: int = 252,
) -> dict:
    """One-shot evaluation bundle used by the eval agent.

    ``trial_sharpes``, when given, is the annualised Sharpe of every trial tried
    (post family-blocking); its variance drives the Deflated Sharpe correction.
    ``tail_spread`` is the annualised top/bottom-decile long-short return — the
    true tradeability criterion (EP004: IC is corrupted by mid-book noise, a
    positive IC with a negative tail spread is a non-tradeable level effect).
    """
    ic = daily_ic(signal, forward_returns, method="spearman")
    ls_ret = tail_long_short_returns(signal, forward_returns)
    if ls_ret.empty:
        return {
            "ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "n_days": 0,
            "significant": False, "tail_spread": 0.0, "deflated_sharpe": None,
        }
    out = {
        "ic": float(ic.mean()),
        "rank_ic": mean_ic(signal, forward_returns, "spearman"),
        "icir": icir(ic),
        "sharpe": sharpe_ratio(ls_ret, periods_per_year),
        "max_drawdown": max_drawdown(ls_ret),
        "t_stat": t_statistic(ls_ret, periods_per_year),
        "n_days": int(len(ic.dropna())),
        "tail_spread": annualized_return(ls_ret, periods_per_year),
        "significant": is_significant(ls_ret, n_trials, periods_per_year=periods_per_year)[0],
        "required_sharpe": significance_threshold_sharpe(
            max(len(ls_ret), 1), n_trials, periods_per_year=periods_per_year
        ),
        "deflated_sharpe": None,
    }
    if trial_sharpes is not None:
        vals = [float(v) for v in trial_sharpes if v is not None and np.isfinite(v)]
        if len(vals) >= 2:
            out.update(deflated_sharpe_ratio(
                ls_ret, n_trials, float(np.var(vals, ddof=1)), periods_per_year
            ))
    return out
