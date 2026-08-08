"""Custom PIT-aware long-short backtester (blueprint §Phase 1, review.md §4.1).

The engine never sees the future:

* **Sizing** uses only ``scores`` — factor values that the caller already
  computed from *point-in-time* data at date ``d``;
* **Measurement** uses ``forward_returns`` at ``(d, i)`` — the return realised
  over [d, d+1) — which is legitimate because it is applied *after* the position
  is set, exactly as a live trader would experience it.

The result exposes the daily returns, the weight history (for turnover cost) and
a metrics bundle consistent with :mod:`src.backtest.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import metrics as M

DATE, SYMBOL = "date", "symbol"


@dataclass
class BacktestConfig:
    long_pct: float = 0.10          # top decile long
    short_pct: float = 0.10         # bottom decile short
    max_position_pct: float = 0.05  # per-name cap (online_execution.max_position_pct)
    transaction_cost_bps: float = 10.0
    annualization: int = 252


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    scores: pd.DataFrame
    config: BacktestConfig
    metrics: dict = field(default_factory=dict)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe", 0.0)


class PointInTimeBacktest:
    """Long-short backtester over a ``(date, symbol)`` panel."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        scores: pd.Series,
        forward_returns: pd.Series,
        *,
        benchmark: Optional[pd.Series] = None,
    ) -> BacktestResult:
        cfg = self.config
        df = pd.DataFrame({"score": scores, "fwd": forward_returns}).dropna()

        dates = sorted(df.index.get_level_values(0).unique())
        weights_list: list[pd.Series] = []

        for d in dates:
            day = df.xs(d, level=0)
            day = day.sort_values("score", ascending=False)
            rank = day["score"].rank(method="first", ascending=False)
            n = len(day)
            n_long = max(1, int(round(n * cfg.long_pct)))
            n_short = max(0, int(round(n * cfg.short_pct)))
            w = pd.Series(0.0, index=day.index)
            if n_long:
                longs = day.index[:n_long]
                w[longs] = 1.0 / n_long
            if n_short and n_short < n:
                shorts = day.index[-n_short:]
                w[shorts] = -1.0 / n_short
            # fully-invested normalisation FIRST so the per-name cap is enforced
            # on the final weights (applying the cap before the gross division
            # would re-inflate every weight back above the cap).
            g = w.abs().sum()
            if g > 0:
                w = w / g
            cap = cfg.max_position_pct
            over = w[w.abs() > cap]
            if not over.empty:
                scale = cap / over.abs().max()
                w = w.copy()
                w[over.index] = over * scale
            weights_list.append(w.rename(d))

        weights = pd.DataFrame(weights_list).fillna(0.0)          # date x symbol
        gross = weights.abs().sum(axis=1)

        # portfolio return on date d = sum_i w[i,d] * fwd[i,d]
        fwd_wide = df["fwd"].unstack(fill_value=0.0)
        port_ret = (weights * fwd_wide.reindex(columns=weights.columns, fill_value=0.0)).sum(axis=1)

        # transaction costs on weight turnover
        if cfg.transaction_cost_bps > 0:
            chg = weights.diff().abs().sum(axis=1).fillna(0.0)
            cost = chg * cfg.transaction_cost_bps / 1e4
            port_ret = port_ret - cost

        result = BacktestResult(
            returns=port_ret,
            weights=weights,
            scores=df["score"].unstack(fill_value=0.0),
            config=cfg,
        )
        result.metrics = self._summarize(port_ret, benchmark, turnover=M.turnover(weights))
        return result

    def _summarize(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        turnover: float = 0.0,
    ) -> dict:
        ann = self.config.annualization
        metrics: dict = {
            "total_return": float((1 + returns.fillna(0)).prod() - 1),
            "annualized_return": M.annualized_return(returns, ann),
            "sharpe": M.sharpe_ratio(returns, ann),
            "max_drawdown": M.max_drawdown(returns),
            "t_stat": M.t_statistic(returns, ann),
            "n_days": int(len(returns.dropna())),
            "turnover": turnover,
        }
        if benchmark is not None:
            joint = pd.concat([returns, benchmark], axis=1, keys=["strat", "bench"]).dropna()
            if len(joint) > 2:
                excess = joint["strat"] - joint["bench"]
                metrics["excess_sharpe"] = M.sharpe_ratio(excess, ann)
                metrics["alpha"] = float(excess.mean() * ann)
                cov = joint.cov()
                var = cov.loc["bench", "bench"]
                metrics["beta"] = float(cov.loc["strat", "bench"] / var) if var > 0 else 0.0
                metrics["information_ratio"] = M.sharpe_ratio(excess, ann)
        return metrics
