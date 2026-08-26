"""Systematic operator sweep — the deterministic half of the exploration track.

Builds a grid of candidate formulas over the NEW exploration-tier operators
(higher moments / volatility structure / illiquidity), each cross-sectionally
normalised via ``Rank`` in BOTH directions (long and short exposure), then hands
them to the walk-forward validator. This is the "massive historical validation"
arm: every new operator is stress-tested across train/val/test before it is
even considered for promotion.
"""

from __future__ import annotations

from typing import Any


def _raw_formula(op: str, lb: Any) -> str:
    """Raw (un-normalised) formula for one operator × lookback.

    ``lb`` is an int (single window) or a (short, long) tuple for ``ts_vol_ratio``.
    Volatility / moment operators are applied to the 1-day return series; level
    / structure operators to the raw price or OHLC fields.
    """
    if op == "ts_semi_std":
        return f"TS_Semi_Std(TS_Return(Close, 1), {lb})"
    if op == "ts_max_drawdown":
        return f"TS_Max_Drawdown(Close, {lb})"
    if op == "ts_autocorr":
        return f"TS_AutoCorr(TS_Return(Close, 1), {lb})"
    if op == "ts_rsq":
        return f"TS_Rsq(Close, {lb})"
    if op == "ts_vol_ratio":
        w1, w2 = lb
        return f"TS_Vol_Ratio(TS_Return(Close, 1), {w1}, {w2})"
    if op == "ts_illiquidity":
        return f"TS_Illiquidity(Close, Volume, {lb})"
    if op == "ts_garman_klass":
        return f"TS_Garman_Klass(Open, High, Low, Close, {lb})"
    if op == "ts_parkinson":
        return f"TS_Parkinson(High, Low, {lb})"
    if op == "ts_range":
        return f"TS_Range(High, Low, Close, {lb})"
    if op == "ts_price_position":
        return f"TS_Price_Position(Close, {lb})"
    if op == "ts_rel_volume":
        return f"TS_Rel_Volume(Volume, {lb})"
    if op == "ts_sharpe":
        return f"TS_Sharpe(TS_Return(Close, 1), {lb})"
    raise ValueError(f"unknown sweep operator {op!r}")


def build_sweep_formulas(sweep_cfg: dict) -> list[dict]:
    """Expand the sweep config into ``(name, formula, meaning)`` candidates."""
    lookbacks = list(sweep_cfg.get("lookbacks", [60, 120, 240]))
    vol_pairs = [tuple(p) for p in sweep_cfg.get("vol_ratio_pairs", [])]
    ops = list(sweep_cfg.get("operators", []))
    signs = list(sweep_cfg.get("signs", ["long", "short"]))
    out: list[dict] = []
    for op in ops:
        lbs: list[Any] = vol_pairs if op == "ts_vol_ratio" else lookbacks
        for lb in lbs:
            raw = _raw_formula(op, lb)
            for sign in signs:
                norm = f"Rank({raw})" if sign == "long" else f"Neg(Rank({raw}))"
                key = f"{op}_{lb if not isinstance(lb, tuple) else f'{lb[0]}x{lb[1]}'}_{sign}"
                out.append(
                    {
                        "name": key,
                        "formula": norm,
                        "meaning": f"{op} lookback={lb} exposure={sign}",
                        "operator": op,
                        "source": "sweep",
                    }
                )
    return out


def exploration_verdict(metrics: dict, gates: dict) -> str:
    """Loose exploration gate — decoupled from the production factor_thresholds.

    Returns one of ``pass`` / ``reject_high_risk`` / ``reject``. This gate only
    decides whether a candidate is worth *studying further*; promotion to the
    production pool still runs the hard production gates.
    """
    dd = float(metrics.get("max_drawdown", 1.0))
    rk = float(metrics.get("rank_ic", 0.0))
    icir = float(metrics.get("icir", 0.0))
    ic = float(metrics.get("ic", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    if dd > float(gates.get("max_drawdown_limit", 0.30)):
        return "reject_high_risk"
    if rk >= float(gates.get("rank_ic_keep", 0.02)) and icir >= float(gates.get("icir_keep", 0.20)):
        return "pass"
    if ic >= float(gates.get("ic_keep", 0.01)) and sharpe >= float(gates.get("min_sharpe", 0.50)):
        return "pass"
    return "reject"


def is_reliable(train: dict, test: dict, gates: dict) -> bool:
    """A candidate is *reliable* iff it passes the exploration gate on BOTH the
    train and test windows with a consistent IC sign (no regime flip)."""
    t = train.get("exploration")
    te = test.get("exploration")
    if t != "pass" or te != "pass":
        return False
    trk = float(train.get("rank_ic", 0.0))
    terk = float(test.get("rank_ic", 0.0))
    return (trk > 0) == (terk > 0) or abs(terk) < 1e-9
