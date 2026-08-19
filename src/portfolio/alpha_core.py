"""Alpha core — the Phase 8 low-vol + low-turnover book, market-neutral (layer 1).

Phase 10 layer 1 is the **only validated alpha direction**: a pool of 5
``Avg(Neg(Rank(TS_Std(Close, N))), Neg(Rank(TS_Mean(Volume, M))))`` formulas
(low realized volatility ∧ low turnover → expected excess return). The core
fuses them by date-wise z-scored equal weight — the same composite
:func:`~src.pool.combination_backtest` builds with ``weights="equal"`` — then
turns the composite into a **market-neutral** book (user decision 2026-08-12,
option A): long the top ``long_pct``, short the bottom ``short_pct``, equal
weight, gross-normalised to 1, per-name capped at ``max_position_pct``.

The market-neutral form is a deliberate reversal of the blueprint's long-only
sketch: the Phase 8 factors were validated on a long-short book (abs_dd
≈ 0.105-0.109), and the long-only conversion inherited full market β — the
2010-2025 backtest showed maxDD 34.5% (Sharpe 0.94), both failing the Phase 10
gate. Shorting the bottom decile restores the hedged profile the gate targets.

**Momentum neutralization** (paper-backed, MLMultiFactorBiasCorrection
arXiv:2507.07107): the raw composite embeds a structural negative correlation
with intermediate/long momentum (mom252 cross-sectional corr -0.25 overall,
-0.44 in 2025) — the short leg shorts momentum winners in bull years (2015,
2021, 2025) and gets squeezed, which was the residual 20.1% maxDD. The
composite is therefore orthogonalised per date against 20/60/120/252-day past
returns (cross-sectional OLS, residuals kept), which on the 2010-2025 sample
raised Sharpe 1.02 -> 1.58 and cut maxDD 20.1% -> 12.7% (probe-verified).

**Regime-adaptive short-leg control** (AlphaCrafter γ): the low-vol short leg
earns in bear/sideways markets (2011 +17.9%, 2018 +15.8%) and bleeds only in
momentum bulls. When the equal-weight market 60-day trend exceeds
``trend_gate``, the short leg is scaled to ``short_scale`` before gross
normalisation — a mild net-long tilt in strong uptrends. On the 2010-2025
sample this closes the gate: Sharpe 1.67, maxDD 9.2% (trend_gate=0.03,
short_scale=0.5), and moves the maxDD trough out of the 2025 squeeze.

The book is recomputed daily from the slow (120/240-day) factor lookbacks, so
daily recomputation is a close approximation of the blueprint's 月度调仓.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _zscore(scores: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date (neutralises the market level)."""
    s = scores.astype(float)
    mean = s.groupby(level=0).transform("mean")
    std = s.groupby(level=0).transform("std").replace(0.0, np.nan)
    return (s - mean) / (std.fillna(1.0) + 1e-12)


def _momentum_panel(close_wide: pd.DataFrame, lookbacks) -> pd.DataFrame:
    """(date, symbol) × lookback past-return frame. PIT-safe by construction
    (a lookback-day return at date ``d`` uses only closes at or before ``d``)."""
    cols = {}
    for n in lookbacks:
        cols[f"mom{n}"] = (close_wide / close_wide.shift(n) - 1.0).stack().rename(f"mom{n}")
    return pd.concat(cols.values(), axis=1)


def _beta_panel(close_wide: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """(date, symbol) rolling market beta vs the equal-weight cross-section.

    beta[symbol, d] = cov(r_symbol, r_mkt) / var(r_mkt) over the trailing
    ``lookback`` days, where r_mkt is the daily cross-sectional mean return.
    PIT-safe: only closes at or before ``d`` enter the window. This is the
    exposure a low-vol book picks up structurally — long low-vol (low beta),
    short high-vol (high beta) — which the momentum projection does NOT remove
    (momentum and beta are distinct exposures).
    """
    ret_wide = close_wide.pct_change(fill_method=None)
    mkt = ret_wide.mean(axis=1)
    mkt_var = mkt.rolling(lookback).var()
    cov = ret_wide.rolling(lookback).cov(mkt)
    return cov.div(mkt_var, axis=0).stack().rename("beta")


def _neutralize_composite(
    composite: pd.Series, close_wide: pd.DataFrame, lookbacks,
    beta_neutralize: bool = False, beta_lookback: int = 252,
) -> pd.Series:
    """Per-date cross-sectional OLS of the composite on momentum (and optionally
    market beta); keep residuals.

    The projection removes the composite's structural exposure to recent
    returns (the momentum squeeze on the low-vol short leg), and — with
    ``beta_neutralize`` — the market-beta exposure (low-vol = low beta long /
    high beta short → a large negative net beta that bleeds on up-days).
    Preserving only the alpha orthogonal to these exposures makes the resulting
    dollar-neutral book beta-neutral as well. Returns a ``(date, symbol)``
    Series aligned to ``composite``'s index; early dates (warm-up) are dropped,
    and dates with fewer than ``min_samples`` names are skipped.
    """
    mom = _momentum_panel(close_wide, lookbacks)
    feats = list(mom.columns)
    if beta_neutralize:
        mom = pd.concat([mom, _beta_panel(close_wide, beta_lookback)], axis=1)
        feats.append("beta")
    joint = pd.concat([composite.rename("comp"), mom], axis=1).dropna()
    rows: list[tuple[tuple, float]] = []
    for d, day in joint.groupby(level=0):
        if len(day) < 30:
            continue
        X = day[feats].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        y = day["comp"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for (dt, sym), v in zip(day.index, y - X @ beta):
            rows.append(((dt, sym), float(v)))
    out = pd.Series(dict(rows))
    out.index = pd.MultiIndex.from_tuples(out.index, names=composite.index.names)
    return out.sort_index()


def _market_trend(close_wide: pd.DataFrame, days: int) -> pd.Series:
    """Equal-weight market ``days``-day cumulative return, PIT-safe.

    Compounds the daily cross-sectional mean return (``close[t]/close[t-1]-1``
    averaged over names present). A level ratio ``mean(close[d])/mean(close[d-60])``
    is *not* used: it is biased by cross-section composition shifts (names
    dropping in/out change the mean level independent of returns). ``daily[d]``
    is the return realised over ``[d-1, d]`` (known at close ``d``), so the
    ``days``-window ending at ``d`` is PIT-safe by construction.
    """
    daily = close_wide.pct_change(fill_method=None).mean(axis=1)  # over names present
    return (1.0 + daily).rolling(int(days)).apply(
        lambda x: x.prod() - 1.0, raw=True
    )


def long_book_weights(
    scores_day: pd.Series,
    *,
    long_pct: float = 0.10,
    short_pct: float = 0.0,
    max_position_pct: float = 0.05,
    short_scale: float = 1.0,
) -> dict[str, float]:
    """Cross-sectional book from one day's scores — long-only or market-neutral.

    Long the top ``long_pct`` names at +1/n and (when ``short_pct > 0``) short
    the bottom ``short_pct`` names at -1/n, then gross-normalise to 1 and cap
    any over-cap name (mirroring ``PointInTimeBacktest``'s construction — the
    same semantics that validated Phase 8's low-vol/low-turnover pool at
    abs_dd ≈ 0.105-0.109). ``short_pct=0`` reproduces the long-only book.
    ``short_scale`` scales the short leg *before* gross normalisation (the
    AlphaCrafter regime-adaptive γ — trims short exposure in strong uptrends).
    Returns ``{symbol: weight}``.

    ``scores_day`` may arrive as a ``groupby(level=0)`` group — pandas keeps
    the full ``(date, symbol)`` MultiIndex there — so a leading date level is
    dropped so the book keys are plain symbols.
    """
    s = scores_day.dropna()
    if isinstance(s.index, pd.MultiIndex):
        s = s.droplevel(0)
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="first")]
    if s.empty:
        return {}
    n = len(s)
    n_long = max(1, int(round(n * long_pct)))
    n_short = max(0, int(round(n * short_pct))) if short_pct > 0 else 0
    order = s.sort_values(ascending=False).index
    w = pd.Series(0.0, index=s.index)
    w[order[:n_long]] = 1.0 / n_long
    if n_short and n_short < n:
        w[order[-n_short:]] = -float(short_scale) / n_short
    g = w.abs().sum()
    if g > 0:
        w = w / g
    cap = float(max_position_pct)
    over = w[w.abs() > cap]
    if not over.empty:
        scale = cap / over.abs().max()
        w = w.copy()
        w[over.index] = over * scale
    return {sym: float(v) for sym, v in w.items() if abs(v) > 1e-12}


class AlphaCore:
    """Momentum-neutralized composite -> regime-adaptive market-neutral book."""

    def __init__(
        self,
        fctx,
        formulas: Iterable[str],
        *,
        long_pct: float = 0.10,
        short_pct: float = 0.10,
        max_position_pct: float = 0.05,
        neutralize: bool = True,
        momentum_lookbacks: tuple[int, ...] = (20, 60, 120, 252),
        beta_neutralize: bool = False,
        beta_lookback: int = 252,
        regime_short: bool = True,
        trend_days: int = 60,
        trend_gate: float = 0.03,
        short_scale: float = 0.5,
        trend_series: Optional[pd.Series] = None,
    ) -> None:
        self.formulas = list(formulas)
        self.long_pct = float(long_pct)
        self.short_pct = float(short_pct)
        self.max_position_pct = float(max_position_pct)
        self.neutralize = bool(neutralize)
        self.momentum_lookbacks = tuple(momentum_lookbacks)
        self.beta_neutralize = bool(beta_neutralize)
        self.beta_lookback = int(beta_lookback)
        self.regime_short = bool(regime_short)
        self.trend_days = int(trend_days)
        self.trend_gate = float(trend_gate)
        self.short_scale = float(short_scale)
        from ..factors.code_generator import eval_expression

        zscores = [_zscore(eval_expression(f, fctx)) for f in self.formulas]
        composite: pd.Series = (
            sum(zscores) / len(zscores) if zscores else pd.Series(dtype=float)
        )
        close_wide = fctx.data["close"].unstack()
        if self.neutralize:
            composite = _neutralize_composite(
                composite, close_wide, self.momentum_lookbacks,
                beta_neutralize=self.beta_neutralize, beta_lookback=self.beta_lookback,
            )
        self.composite = composite
        # Callers with the tradable-forward panel inject the exact validated trend
        # (limit-locked masked daily mean returns compounded) — reproducing the
        # probe that hit maxDD 9.2%. Without it, fall back to the close-based one.
        if trend_series is not None:
            trend = trend_series
        elif self.regime_short:
            trend = _market_trend(close_wide, self.trend_days)
        else:
            trend = None
        # index -> date cross-section, built once so weights_on is a dict lookup
        self._by_date: dict[pd.Timestamp, dict[str, float]] = {}
        for d, day in composite.groupby(level=0):
            ss = 1.0
            if trend is not None and pd.Timestamp(d) in trend.index:
                t = trend[pd.Timestamp(d)]
                if np.isfinite(t) and t > self.trend_gate:
                    ss = self.short_scale
            self._by_date[pd.Timestamp(d)] = long_book_weights(
                day, long_pct=self.long_pct, short_pct=self.short_pct,
                max_position_pct=self.max_position_pct, short_scale=ss,
            )

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(self._by_date)

    def weights_on(self, date: str | pd.Timestamp, symbols: Optional[Iterable[str]] = None) -> dict[str, float]:
        """The alpha book for one trading date (empty dict when out of sample)."""
        w = self._by_date.get(pd.Timestamp(date), {})
        if symbols is None:
            return dict(w)
        keep = set(symbols)
        return {s: v for s, v in w.items() if s in keep}

    def weights_frame(self, dates: Iterable[str | pd.Timestamp]) -> pd.DataFrame:
        """date × symbol book-weight frame over ``dates`` (0.0 where no position)."""
        rows = {pd.Timestamp(d): self.weights_on(d) for d in dates}
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
