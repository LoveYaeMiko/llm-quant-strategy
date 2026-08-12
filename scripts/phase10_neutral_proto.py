"""Phase 10 — momentum-neutralization prototype (paper-backed evidence test).

MLMultiFactorBiasCorrection (arXiv:2507.07107) reports that cross-sectional
neutralization of raw factors against systematic exposures raises mean IC
(0.023->0.041) and IR (0.147->0.461) on 2010-2024 A-shares. The probe
(phase10_neutral_probe.py) showed the Phase 10 market-neutral composite is
structurally short intermediate/long momentum (mom252 corr -0.25 overall,
-0.44 in 2025), which is the 2025 short-leg squeeze behind the residual 20.1%
maxDD. This prototype measures whether neutralizing the composite against
momentum actually improves Sharpe / maxDD before committing any code change.

Variants (all PIT-safe — momentum uses only past closes):
  V0  baseline (no neutralization)                          — must reproduce 1.04/20.1%
  V1  neutralize on [mom20, mom120]
  V2  neutralize on [mom60, mom252]
  V3  neutralize on [mom20, mom60, mom120, mom252]
  V4  neutralize on [mom120] (single strongest exposure)

For each: residual z-scores -> market-neutral book (same long/short/cap as
baseline) -> weight-driven backtest 2010-2025. Prints a comparison table.
No files written, no code changed.
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

from src.backtest.metrics import annualized_return, max_drawdown, sharpe_ratio
from src.cli import _cached_universe_json
from src.config import load_config
from src.factors.code_generator import FactorContext
from src.portfolio.alpha_core import AlphaCore, _zscore, long_book_weights
from src.portfolio.backtest_runner import load_hs300_market


def _momentum_stack(close, lookbacks=(20, 60, 120, 252)):
    """(date, symbol) x lookback past-return frame. PIT-safe by construction."""
    cols = {}
    for n in lookbacks:
        m = (close / close.shift(n) - 1.0).stack().rename(f"mom{n}")
        cols[f"mom{n}"] = m
    return pd.concat(cols.values(), axis=1)


def _neutralize(composite: pd.Series, mom: pd.DataFrame, lookbacks):
    """Per-date OLS of composite z on [1, mom cols], return residual z-scores."""
    resid = pd.Series(np.nan, index=composite.index)
    xc = mom[list(lookbacks)].astype(float)
    joint = pd.concat([composite.rename("comp"), xc], axis=1).dropna()
    for d, day in joint.groupby(level=0):
        if len(day) < 30:
            continue
        X = day[list(lookbacks)].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        y = day["comp"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        idx = [(d, s) for s in day.index.get_level_values(1)]
        resid.loc[pd.MultiIndex.from_tuples(idx, names=composite.index.names)] = r
    return resid.dropna()


def main() -> int:
    cfg = load_config()
    symbols = _cached_universe_json("hs300", cfg)
    t0 = time.time()
    market = load_hs300_market(cfg, symbols, start="2009-01-06", end="2025-12-31")
    formulas = [x.get("factor", {}).get("formula", x.get("formula"))
                for x in json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))]
    alpha = AlphaCore(FactorContext(market.long), formulas,
                      long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
    comp = alpha.composite
    mom = _momentum_stack(market.price_panel)
    print(f"proto: loaded market + composite in {time.time() - t0:.1f}s")

    fwd = market.forward_returns_tradable
    fwd_wide = fwd.unstack(fill_value=0.0)
    symbols_uni = sorted(comp.index.get_level_values(1).unique())
    aligned = fwd_wide.reindex(columns=symbols_uni).fillna(0.0)
    # forward-return dates only — the composite's last price day has no next-day
    # return, so drop dates missing from the aligned forward panel.
    dates = sorted(comp.index.get_level_values(0).unique())
    dates = [d for d in dates if pd.Timestamp("2010-01-01") <= d <= pd.Timestamp("2025-12-31")
             and pd.Timestamp(d) in aligned.index]

    variants = {
        "V0 baseline": None,
        "V1 [20,120]": ("mom20", "mom120"),
        "V2 [60,252]": ("mom60", "mom252"),
        "V3 [20,60,120,252]": ("mom20", "mom60", "mom120", "mom252"),
        "V4 [120]": ("mom120",),
    }

    print(f"\n{'variant':<18} {'sharpe':>7} {'ann':>7} {'maxDD':>7} {'corr_mkt':>8} "
          f"{'2015':>7} {'2021':>7} {'2025':>7} {'n_pos':>6}")
    bench = aligned.mean(axis=1)
    for name, lbs in variants.items():
        if lbs is None:
            resid = comp
            run_dates = dates
        else:
            resid = _neutralize(comp, mom, lbs)
            # neutralized residuals start later (momentum warm-up) — use their dates
            run_dates = sorted({pd.Timestamp(d) for d in resid.index.get_level_values(0)})
            run_dates = [d for d in run_dates if pd.Timestamp("2010-01-01") <= d <= pd.Timestamp("2025-12-31")
                         and d in aligned.index]
        # rebuild book from residual scores -> weight-driven returns
        port_ret = pd.Series(0.0, index=pd.to_datetime(run_dates))
        npos = []
        for d in run_dates:
            day = resid.xs(d, level=0) if isinstance(resid.index, pd.MultiIndex) else resid
            w = long_book_weights(day, long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
            if not w:
                continue
            syms = [s for s in aligned.columns if s in w]
            port_ret.loc[pd.Timestamp(d)] = sum(w[s] * aligned.loc[pd.Timestamp(d), s] for s in syms)
            npos.append(sum(1 for v in w.values() if abs(v) > 0))
        port_ret = port_ret.dropna()
        y = lambda yr: (1 + port_ret[port_ret.index.year == yr]).prod() - 1 if (port_ret.index.year == yr).any() else float("nan")
        mkt = bench.reindex(port_ret.index)
        print(f"{name:<18} {port_ret.mean()/port_ret.std()*np.sqrt(252):7.2f} "
              f"{annualized_return(port_ret):7.1%} {max_drawdown(port_ret):7.1%} "
              f"{port_ret.corr(mkt):8.3f} {y(2015):7.1%} {y(2021):7.1%} {y(2025):7.1%} "
              f"{np.mean(npos):6.0f}")
        # drawdown structure for the top-2 variants
        if name in ("V3 [20,60,120,252]", "V0 baseline"):
            cum = (1 + port_ret).cumprod()
            dd = cum / cum.cummax() - 1.0
            for i, (dt, v) in enumerate(dd.sort_values().head(2).items(), 1):
                peak = cum[:dt].idxmax()
                print(f"    DD#{i}: {v:.1%} trough={dt.date()} peak={peak.date()} "
                      f"year={dt.year}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
