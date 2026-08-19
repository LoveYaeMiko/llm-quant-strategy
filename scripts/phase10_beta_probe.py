"""Phase 10 — market-beta exposure probe + beta-neutralization backtest.

The live shadow (2026) is NOT beta-neutral: net daily PnL correlates -0.675 with
the equal-weight market (short leg -0.828). The mechanism is structural to the
low-vol factor: long top-decile = low-vol = LOW beta, short bottom-decile =
high-vol = HIGH beta, so the dollar-neutral book carries a large NEGATIVE beta.

This probe (a) measures how much of the momentum-neutralized composite's
cross-section is beta, (b) confirms the short leg is high-beta, and (c) prices a
beta-neutralized variant (cross-sectional OLS on [1, mom20/60/120/252, beta252])
against the baseline to see whether removing beta helps or hurts on 2010-2025.

Prints a console report; no files written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_book(composite, close_wide, trend, *, long_pct, short_pct, max_pos,
                short_scale, trend_gate):
    from src.portfolio.alpha_core import long_book_weights
    by_date: dict = {}
    for d, day in composite.groupby(level=0):
        ss = 1.0
        if trend is not None and pd.Timestamp(d) in trend.index:
            t = trend[pd.Timestamp(d)]
            if np.isfinite(t) and t > trend_gate:
                ss = short_scale
        by_date[pd.Timestamp(d)] = long_book_weights(
            day, long_pct=long_pct, short_pct=short_pct,
            max_position_pct=max_pos, short_scale=ss,
        )
    return by_date


def _price(by_date, market, dates):
    from src.backtest.metrics import (
        annualized_return, max_drawdown, sharpe_ratio, t_statistic,
    )
    wf = pd.DataFrame.from_dict(
        {d: by_date.get(d, {}) for d in dates}, orient="index"
    ).fillna(0.0)
    fwd = getattr(market, "forward_returns_tradable", market.forward_returns)
    fwd_wide = fwd.unstack(fill_value=0.0)
    aligned = fwd_wide.reindex(index=wf.index, columns=wf.columns, fill_value=0.0)
    port_ret = (wf * aligned).sum(axis=1).sort_index().dropna()
    if port_ret.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "annualized_return": 0.0}
    return {
        "total_return": float((1.0 + port_ret).prod() - 1.0),
        "annualized_return": float(annualized_return(port_ret)),
        "sharpe": float(sharpe_ratio(port_ret)),
        "max_drawdown": float(max_drawdown(port_ret)),
        "t_stat": float(t_statistic(port_ret)),
        "n_days": int(len(port_ret)),
    }


def main() -> int:
    from src.cli import _cached_universe_json
    from src.config import load_config
    from src.factors.code_generator import FactorContext
    from src.portfolio.alpha_core import (
        AlphaCore, _market_trend, _momentum_panel, _neutralize_composite,
    )
    from src.portfolio.backtest_runner import load_hs300_market

    cfg = load_config()
    symbols = _cached_universe_json("hs300", cfg)
    t0 = time.time()
    market = load_hs300_market(cfg, symbols, start="2009-01-06", end="2025-12-31")
    formulas = [x.get("factor", {}).get("formula", x.get("formula"))
                for x in json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))]
    fctx = FactorContext(market.long)
    close = market.price_panel
    print(f"loaded market in {time.time() - t0:.1f}s  (factors={len(formulas)})")

    # --- baseline: momentum-neutralized composite (production today) ---
    alpha = AlphaCore(fctx, formulas, long_pct=0.10, short_pct=0.10,
                      max_position_pct=0.05)
    comp = alpha.composite

    # --- market beta (252d rolling, equal-weight market) ---
    ret_wide = close.pct_change(fill_method=None)
    mkt = ret_wide.mean(axis=1)
    mkt_var = mkt.rolling(252).var()
    cov = ret_wide.rolling(252).cov(mkt)
    beta_wide = cov.div(mkt_var, axis=0)
    beta = beta_wide.stack().rename("beta")

    # --- (a) how beta is the composite cross-section? ---
    mom = _momentum_panel(close, (20, 60, 120, 252))
    merged = pd.concat([comp.rename("comp"), beta], axis=1).dropna()
    rows = []
    for d, day in merged.groupby(level=0):
        if len(day) < 30:
            continue
        rows.append((d, np.corrcoef(day["beta"].rank(), day["comp"].rank())[0, 1]))
    corr = pd.Series(dict(rows))
    print(f"\n=== composite vs beta252 cross-sectional spearman (2010-2025) ===")
    print(f"  mean={corr.mean():+.3f}  med={corr.median():+.3f}  "
          f"|r|>0.3: {(corr.abs() > 0.3).mean():.1%}")

    # --- (b) short-leg vs long-leg beta gap ---
    print("\n=== leg beta (252d) by composite decile ===")
    for y in (2011, 2015, 2018, 2021, 2025):
        yday = merged[merged.index.get_level_values(0).year == y]
        if not len(yday):
            continue
        bottom = yday[yday["comp"] <= yday["comp"].quantile(0.10)]["beta"]
        top = yday[yday["comp"] >= yday["comp"].quantile(0.90)]["beta"]
        print(f"  {y}: short-leg beta={bottom.mean():+.3f}  long-leg beta={top.mean():+.3f}  "
              f"gap={bottom.mean() - top.mean():+.3f}")

    # --- (c) beta-neutralized composite ---
    joint = pd.concat([comp.rename("comp"), mom, beta.rename("beta")], axis=1).dropna()
    rows = []
    for d, day in joint.groupby(level=0):
        if len(day) < 30:
            continue
        feats = list(mom.columns) + ["beta"]
        X = day[feats].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        y = day["comp"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        for (dt, sym), v in zip(day.index, y - X @ coef):
            rows.append(((dt, sym), float(v)))
    comp_beta = pd.Series(dict(rows))
    comp_beta.index = pd.MultiIndex.from_tuples(comp_beta.index, names=comp.index.names)
    comp_beta = comp_beta.sort_index()

    # beta exposure of the neutralized composite
    m2 = pd.concat([comp_beta.rename("comp"), beta], axis=1).dropna()
    c2 = []
    for d, day in m2.groupby(level=0):
        if len(day) >= 30:
            c2.append((d, np.corrcoef(day["beta"].rank(), day["comp"].rank())[0, 1]))
    c2 = pd.Series(dict(c2))
    print(f"\n=== beta-neutralized composite vs beta252 corr ===")
    print(f"  mean={c2.mean():+.3f}  med={c2.median():+.3f}")

    # --- (d) price both books ---
    trend = _market_trend(close, 60)
    dates = sorted(market.forward_returns.index.get_level_values(0).unique())
    base_book = _build_book(comp, close, trend, long_pct=0.10, short_pct=0.10,
                            max_pos=0.05, short_scale=0.5, trend_gate=0.03)
    beta_book = _build_book(comp_beta, close, trend, long_pct=0.10, short_pct=0.10,
                            max_pos=0.05, short_scale=0.5, trend_gate=0.03)
    m_base = _price(base_book, market, dates)
    m_beta = _price(beta_book, market, dates)

    print("\n=== backtest comparison (2010-2025, tradable fwd, regime short on) ===")
    print(f"  {'metric':<18}{'baseline':>12}{'beta-neutral':>14}")
    for k in ("sharpe", "max_drawdown", "annualized_return", "total_return", "t_stat"):
        print(f"  {k:<18}{m_base.get(k, 0):>12.3f}{m_beta.get(k, 0):>14.3f}")

    # sub-period robustness
    print("\n=== sub-period Sharpe (base vs beta-neutral) ===")
    for lo, hi in (("2010", "2017"), ("2018", "2025"), ("2020", "2025")):
        ds = [d for d in dates if pd.Timestamp(lo) <= d <= pd.Timestamp(hi + "-12-31")]
        print(f"  {lo}-{hi}: base={_price(base_book, market, ds)['sharpe']:.2f}  "
              f"betaN={_price(beta_book, market, ds)['sharpe']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
