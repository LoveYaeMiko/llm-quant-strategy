"""Phase 8 factor-pool management — filter, diversify, combine, monitor.

The mining loop (``mine``) writes *accepted* factors to ``outputs/factors.json``
as ``[{"factor": {...}, "metrics": {...}, "risk": {...}}]``; the ``metrics`` are
training-window numbers. This module turns that raw pool into a managed,
deployable factor list and the evidence needed to promote it:

* **evaluate_pool** — re-score every formula on a chosen walk-forward window with
  the same ``factor_eval`` bundle the miner uses, so validation/test numbers are
  directly comparable to the training numbers.
* **filter_pool** — keep factors whose out-of-sample IC and ICIR clear the
  configured floors (blueprint 8.2: ``--min-val-ic 0.02 --min-icir 0.30``).
* **diversify_pool** — greedy selection that keeps minimum pairwise AST distance
  (blueprint 8.2: ``--min-distance 0.40``), so the final pool is structurally
  diverse rather than one formula family.
* **combination_backtest** — combine date-wise z-scored factor scores into a
  composite signal and run it through the PIT long-short engine under three
  weighting schemes: ``equal``, ``icir`` (validation ICIR), ``dynamic``
  (trailing-window ICIR reweighting, 1-day shifted to avoid look-ahead).
* **monitor_watchlist** — rolling-window ICIR decay for every pool factor.

No industry/size fundamentals are ingested yet, so the blueprint's
``--neutralize industry,size`` cannot be honoured; the composite is z-scored
cross-sectionally per date, which neutralises the market level. We build what the
data supports and document what it does not (blueprint rule: adapt, don't force).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .backtest.engine import BacktestConfig, PointInTimeBacktest
from .backtest.metrics import (
    daily_ic,
    deflated_sharpe_ratio,
    factor_eval,
    icir,
    tail_long_short_returns,
)
from .factors.code_generator import CodeGenerator, ast_distance, eval_expression
from .monitoring.decay_tracker import DecayTracker

# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_pool(
    fctx,
    forward_returns: pd.Series,
    formulas: Sequence[str],
    *,
    n_trials: int = 1,
) -> dict[str, dict]:
    """Score every formula on a window with the miner's ``factor_eval`` bundle.

    ``fctx`` is a ``FactorContext`` built on the window-sliced market; any formula
    that fails to evaluate is reported as ``{"error": ...}`` and never treated as
    a passing factor downstream.
    """
    out: dict[str, dict] = {}
    scores_by_f: dict[str, pd.Series] = {}
    for f in formulas:
        try:
            scores = eval_expression(f, fctx)
            scores_by_f[f] = scores
            out[f] = factor_eval(scores, forward_returns, n_trials=n_trials)
        except Exception as exc:  # noqa: BLE001 — a bad formula must not kill the pool
            out[f] = {"error": str(exc)}
    # Deflated Sharpe (Bailey & López de Prado): the "best of N" luck correction.
    # Trial-Sharpe variance comes from every *valid* formula scored this pass;
    # ``n_trials`` is the effective independent count (post family-blocking), so
    # the correction targets snooping without over-penalising real factors.
    trial_sharpes = [
        float(m["sharpe"])
        for f, m in out.items()
        if f in scores_by_f and isinstance(m, dict) and m.get("sharpe") is not None
    ]
    if n_trials > 1 and len(trial_sharpes) >= 2:
        var = float(np.var(trial_sharpes, ddof=1))
        for f, scores in scores_by_f.items():
            ls_ret = tail_long_short_returns(scores, forward_returns)
            if ls_ret.empty:
                continue
            out[f].update(deflated_sharpe_ratio(ls_ret, n_trials, var))
    return out


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def filter_pool(
    pool: list[dict],
    metrics_by_formula: dict[str, dict],
    *,
    min_ic: float = 0.02,
    min_icir: float = 0.30,
) -> list[dict]:
    """Keep factors whose out-of-sample ``metrics_by_formula`` clear the gates.

    Each surviving entry is copied and gains a ``val_metrics`` key so downstream
    stages (diversify / backtest / promote) read one consistent number.
    """
    kept: list[dict] = []
    for entry in pool:
        formula = str(entry.get("factor", {}).get("formula", ""))
        m = metrics_by_formula.get(formula)
        if m is None or "error" in m:
            continue
        if float(m.get("ic", 0.0)) >= min_ic and float(m.get("icir", 0.0)) >= min_icir:
            out = deepcopy(entry)
            out["val_metrics"] = m
            kept.append(out)
    return kept


# ---------------------------------------------------------------------------
# diversify
# ---------------------------------------------------------------------------


def diversify_pool(
    pool: list[dict],
    *,
    min_distance: float = 0.40,
    order_by: Optional[str] = None,
) -> list[dict]:
    """Greedy AST-diversity selection: keep a factor iff its parse tree is at
    least ``min_distance`` away from every already-kept tree.

    ``order_by`` sorts the pool before the sweep so the best-valued factors win a
    slot first (e.g. ``"val_ic"``). The parse tree is the same ``CodeGenerator``
    the pipeline uses, so the distance gate matches ``diversity_check``.
    """
    if order_by:
        pool = sorted(
            pool,
            key=lambda e: float(
                e.get("val_metrics", {}).get(order_by, 0.0)
                or e.get("metrics", {}).get(order_by, 0.0)
            ),
            reverse=True,
        )
    gen = CodeGenerator()
    kept: list[dict] = []
    nodes: list = []
    for entry in pool:
        formula = str(entry.get("factor", {}).get("formula", ""))
        try:
            node = gen.parse(formula)
        except Exception:  # noqa: BLE001 — unparsable formulas are dropped
            continue
        if all(ast_distance(node, k) >= min_distance for k in nodes):
            kept.append(entry)
            nodes.append(node)
    return kept


# ---------------------------------------------------------------------------
# combination backtest
# ---------------------------------------------------------------------------


def _date_zscore(scores: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date (neutralises the market level)."""
    s = scores.astype(float)
    mean = s.groupby(level=0).transform("mean")
    std = s.groupby(level=0).transform("std").replace(0.0, np.nan)
    return (s - mean) / (std.fillna(1.0) + 1e-12)


def _score_each(
    fctx,
    pool: list[dict],
) -> dict[str, pd.Series]:
    """Compute a date-wise z-scored signal per formula that evaluates cleanly."""
    out: dict[str, pd.Series] = {}
    for entry in pool:
        formula = str(entry.get("factor", {}).get("formula", ""))
        try:
            out[formula] = _date_zscore(eval_expression(formula, fctx))
        except Exception:  # noqa: BLE001 — skip un-scorable formulas
            continue
    return out


def _weight_schemes(
    scores_by_formula: dict[str, pd.Series],
    pool: list[dict],
    forward_returns: pd.Series,
    *,
    weights: str,
    dynamic_window: int = 60,
) -> dict[str, float]:
    """Resolve the weight vector (or per-date weight frame) for one scheme."""
    names = list(scores_by_formula)
    pool_map = {str(e.get("factor", {}).get("formula", "")): e for e in pool}

    if weights == "icir":
        raw = {}
        for f in names:
            icir_v = float(pool_map.get(f, {}).get("val_metrics", {}).get("icir", 0.0))
            raw[f] = max(icir_v, 0.0)
        tot = sum(raw.values())
        if tot <= 0:
            return {f: 1.0 / len(names) for f in names}
        return {f: raw[f] / tot for f in names}

    if weights == "dynamic":
        # Trailing-window ICIR per factor, 1 day shifted so the weight applied on
        # date d uses only information up to d-1.  Undefined windows -> equal.
        ic_by_f = {}
        for f in names:
            ic_by_f[f] = daily_ic(scores_by_formula[f], forward_returns, method="spearman")
        dates = sorted({d for ic in ic_by_f.values() for d in ic.index})
        w_frames: dict[str, pd.Series] = {}
        for f, ic in ic_by_f.items():
            roll = ic.rolling(dynamic_window, min_periods=20).apply(
                lambda x: float(icir(pd.Series(x)))
            ).shift(1)
            w_frames[f] = roll.reindex(dates).fillna(0.0)
        table = pd.DataFrame(w_frames).fillna(0.0)
        row_sum = table.sum(axis=1).replace(0.0, np.nan)
        norm = table.div(row_sum, axis=0).fillna(1.0 / len(names))
        return norm  # DataFrame date x factor

    return {f: 1.0 / len(names) for f in names}  # equal (default)


def combination_backtest(
    fctx,
    forward_returns: pd.Series,
    pool: list[dict],
    *,
    weights: str = "equal",
    n_trials: int = 1,
    bt_config: Optional[BacktestConfig] = None,
    forward_tradable: Optional[pd.Series] = None,
) -> dict:
    """Backtest the weighted composite of every pool factor.

    Returns per-factor single-factor backtests (for the correlation read), the
    composite backtest metrics, the weight scheme and the number of factors.

    ``forward_tradable`` (LIMIT_DOWN blueprint 方案 B) is the price-limit-locked
    masked series used for the portfolio Sharpe / max-drawdown; the weight
    schemes (``daily_ic``) stay on the raw ``forward_returns``.
    """
    scores = _score_each(fctx, pool)
    if not scores:
        return {"error": "no scorable factors in the pool", "n_factors": 0}
    names = list(scores)
    fwd_bt = forward_tradable if forward_tradable is not None else forward_returns

    w = _weight_schemes(scores, pool, forward_returns, weights=weights)
    if isinstance(w, dict):
        composite = sum(w[f] * scores[f] for f in names)
    else:  # dynamic returns a date x factor weight frame; align on the date level
        wdf = w.reindex(columns=names).fillna(0.0)
        composite = pd.Series(0.0, index=next(iter(scores.values())).index, dtype=float)
        for f in names:
            composite = composite.add(scores[f].mul(wdf[f], level=0).fillna(0.0))
        composite = composite.rename("composite")

    bt = PointInTimeBacktest(bt_config)
    comp = bt.run(composite, fwd_bt)
    per_factor = {}
    for f in names:
        try:
            per_factor[f] = bt.run(scores[f], fwd_bt).metrics
        except Exception as exc:  # noqa: BLE001
            per_factor[f] = {"error": str(exc)}
    return {
        "n_factors": len(names),
        "weights": str(weights),
        "composite": comp.metrics,
        "per_factor": per_factor,
    }


# ---------------------------------------------------------------------------
# decay watchlist
# ---------------------------------------------------------------------------


def monitor_watchlist(
    fctx,
    forward_returns: pd.Series,
    pool: list[dict],
    *,
    tracker: Optional[DecayTracker] = None,
    window_days: int = 90,
    icir_threshold: float = 0.30,
) -> dict:
    """Rolling-window ICIR decay per pool factor."""
    tr = tracker or DecayTracker(window_days=window_days, icir_threshold=icir_threshold)
    out: dict = {}
    for entry in pool:
        formula = str(entry.get("factor", {}).get("formula", ""))
        try:
            res = tr.monitor(eval_expression(formula, fctx), forward_returns)
            out[formula] = {
                "recent_icir": res.recent_icir,
                "decayed": res.decayed,
                "first_decayed_at": str(res.first_decayed_at.date())
                if res.first_decayed_at is not None else None,
                "summary": res.summary,
            }
        except Exception as exc:  # noqa: BLE001
            out[formula] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------


def load_pool(path: Path | str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("factors", data.get("pool", []))
    return list(data)


def write_json(path: Path | str, obj) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


__all__ = [
    "evaluate_pool",
    "filter_pool",
    "diversify_pool",
    "combination_backtest",
    "monitor_watchlist",
    "load_pool",
    "write_json",
]
