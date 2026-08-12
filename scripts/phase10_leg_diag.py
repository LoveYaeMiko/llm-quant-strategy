"""Phase 10 — V3 residual drawdown leg attribution + regime-adaptive short control.

After momentum neutralization (V3) the residual 12.7% maxDD is still a 2025
Jan-Sep bleed. Decide WHAT to cut next: is the bleed the long leg or the short
leg? Then test AlphaCrafter-style regime-adaptive short-leg control (reduce
short exposure in strong-trend regimes) on top of V3.

Prints a console report; no files written, no code changed.
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

from src.backtest.metrics import annualized_return, max_drawdown
from src.cli import _cached_universe_json
from src.config import load_config
from src.factors.code_generator import FactorContext
from src.portfolio.alpha_core import AlphaCore, long_book_weights
from src.portfolio.backtest_runner import load_hs300_market

LOOKBACKS = ("mom20", "mom60", "mom120", "mom252")


def _momentum_stack(close):
    cols = {}
    for n in (20, 60, 120, 252):
        cols[f"mom{n}"] = (close / close.shift(n) - 1.0).stack().rename(f"mom{n}")
    return pd.concat(cols.values(), axis=1)


def _neutralize(composite, mom):
    resid = []
    joint = pd.concat([composite.rename("comp"), mom[list(LOOKBACKS)]], axis=1).dropna()
    for d, day in joint.groupby(level=0):
        if len(day) < 30:
            continue
        X = day[list(LOOKBACKS)].to_numpy(float)
        X = np.column_stack([np.ones(len(X)), X])
        y = day["comp"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        for (dt, sym), v in zip(day.index, r):
            resid.append(((dt, sym), v))
    out = pd.Series(dict(resid))
    out.index = pd.MultiIndex.from_tuples(out.index, names=composite.index.names)
    return out.sort_index()


def main() -> int:
    cfg = load_config()
    symbols = _cached_universe_json("hs300", cfg)
    t0 = time.time()
    market = load_hs300_market(cfg, symbols, start="2009-01-06", end="2025-12-31")
    formulas = [x.get("factor", {}).get("formula", x.get("formula"))
                for x in json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))]
    alpha = AlphaCore(FactorContext(market.long), formulas,
                      long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
    resid = _neutralize(alpha.composite, _momentum_stack(market.price_panel))
    fwd = market.forward_returns_tradable
    fwd_wide = fwd.unstack(fill_value=0.0)
    aligned = fwd_wide.reindex(columns=sorted(alpha.composite.index.get_level_values(1).unique())).fillna(0.0)
    run_dates = sorted({pd.Timestamp(d) for d in resid.index.get_level_values(0)})
    run_dates = [d for d in run_dates if pd.Timestamp("2010-01-01") <= d <= pd.Timestamp("2025-12-31")
                 and d in aligned.index]
    print(f"diag: loaded + neutralized in {time.time() - t0:.1f}s, {len(run_dates)} days")

    # market trend signal (equal-weight 60d cum return) — for regime gate.
    # PIT-safe: bench[d] is the forward return over (d, d+1], NOT known at close d.
    # The last realised market return at close d is bench[d-1], so shift by 1.
    bench = aligned.mean(axis=1)
    trend60 = (1 + bench).shift(1).rolling(60).apply(lambda x: x.prod() - 1, raw=True)

    # ------- build per-day book, with optional short-leg regime cut -------
    def book_returns(short_scale, trend_gate, short_scale_trend):
        """short_scale: uniform short-leg scale. trend_gate: scale short leg by
        short_scale_trend only when market 60d trend > gate (AlphaCrafter γ)."""
        port = pd.Series(0.0, index=pd.to_datetime(run_dates))
        long_c = pd.Series(0.0, index=port.index)
        short_c = pd.Series(0.0, index=port.index)
        for d in run_dates:
            day = resid.xs(pd.Timestamp(d), level=0)
            w = long_book_weights(day, long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
            if not w:
                continue
            ss = short_scale
            if trend_gate is not None:
                tr = trend60.loc[pd.Timestamp(d)] if pd.Timestamp(d) in trend60.index else np.nan
                if np.isfinite(tr) and tr > trend_gate:
                    ss = short_scale_trend
            w2 = {s: (v * ss if v < 0 else v) for s, v in w.items()}
            g = sum(abs(v) for v in w2.values())
            if g > 0:
                w2 = {s: v / g for s, v in w2.items()}
            syms = [s for s in aligned.columns if s in w2]
            rets = {s: aligned.loc[pd.Timestamp(d), s] for s in syms}
            port.loc[pd.Timestamp(d)] = sum(w2[s] * rets[s] for s in syms)
            long_c.loc[pd.Timestamp(d)] = sum(max(w2[s], 0) * rets[s] for s in syms)
            short_c.loc[pd.Timestamp(d)] = sum(min(w2[s], 0) * rets[s] for s in syms)
        return port.dropna(), long_c.dropna(), short_c.dropna()

    print("\n=== V3 residual: long-leg vs short-leg annual contribution ===")
    port, long_c, short_c = book_returns(1.0, None, 1.0)
    for yr in (2011, 2015, 2018, 2021, 2024, 2025):
        if (port.index.year == yr).any():
            l = (1 + long_c[long_c.index.year == yr]).prod() - 1
            s = (1 + short_c[short_c.index.year == yr]).prod() - 1
            print(f"  {yr}: long {l:+.1%}   short {s:+.1%}   total {((1+port[port.index.year==yr]).prod()-1):+.1%}")

    # ---- short-leg squeeze signal (PIT-safe: uses short-leg returns through d-1) ----
    _, _, short_c_base = book_returns(1.0, None, 1.0)
    short_trail = pd.Series(np.nan, index=port.index)
    sc = (1 + short_c_base)
    short_trail = sc.rolling(20).apply(lambda x: x.prod() - 1, raw=True).shift(1)

    def book_returns_squeeze(scale_when_squeezed, squeeze_thresh):
        """cut short leg when the short book's own trailing 20d return < thresh."""
        port = pd.Series(0.0, index=pd.to_datetime(run_dates))
        for d in run_dates:
            day = resid.xs(pd.Timestamp(d), level=0)
            w = long_book_weights(day, long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
            if not w:
                continue
            ss = 1.0
            st = short_trail.loc[pd.Timestamp(d)] if pd.Timestamp(d) in short_trail.index else np.nan
            if np.isfinite(st) and st < squeeze_thresh:
                ss = scale_when_squeezed
            w2 = {s: (v * ss if v < 0 else v) for s, v in w.items()}
            g = sum(abs(v) for v in w2.values())
            if g > 0:
                w2 = {s: v / g for s, v in w2.items()}
            syms = [s for s in aligned.columns if s in w2]
            port.loc[pd.Timestamp(d)] = sum(w2[s] * aligned.loc[pd.Timestamp(d), s] for s in syms)
        return port.dropna()

    def report(name, p):
        s = p.mean() / p.std() * np.sqrt(252)
        y25 = (1 + p[p.index.year == 2025]).prod() - 1
        cum = (1 + p).cumprod()
        dd = cum / cum.cummax() - 1.0
        trough = dd.idxmin()
        peak = cum[:trough].idxmax()
        print(f"{name:<38} {s:7.2f} {annualized_return(p):7.1%} {max_drawdown(p):7.1%} "
              f"{y25:7.1%}   DD@{trough.date()} peak@{peak.date()}")

    print("\n=== regime-adaptive short-leg control on V3 (PIT-safe) ===")
    print(f"{'config':<38} {'sharpe':>7} {'ann':>7} {'maxDD':>7} {'2025':>7}  location")
    report("V3 baseline (short x1.0)", port)
    for gate in (0.03, 0.04, 0.05):
        for ss in (0.5, 0.6):
            p, _, _ = book_returns(1.0, gate, ss)
            report(f"trend>{gate:.0%} short x{ss}", p)
    print("\n=== short-leg squeeze trigger (trailing 20d short return < thresh) ===")
    for thresh in (-0.10, -0.08, -0.06, -0.04):
        for ss in (0.5, 0.6):
            p = book_returns_squeeze(ss, thresh)
            report(f"squeeze<{thresh:.0%} short x{ss}", p)

    # ---- robustness: sub-period check for the passing configs ----
    print("\n=== sub-period robustness (V3 + trend gates) ===")
    print(f"{'config':<30} {'seg':<12} {'sharpe':>7} {'ann':>7} {'maxDD':>7} {'n':>6}")
    for gate, ss in ((0.03, 0.5), (0.03, 0.6)):
        p, _, _ = book_returns(1.0, gate, ss)
        for lbl, (lo, hi) in (("2010-2017", ("2010-01-01", "2017-12-31")),
                              ("2018-2025", ("2018-01-01", "2025-12-31")),
                              ("2020-2025", ("2020-01-01", "2025-12-31"))):
            sub = p[(p.index >= lo) & (p.index <= hi)]
            s = sub.mean() / sub.std() * np.sqrt(252) if len(sub) > 50 else float("nan")
            print(f"{f'trend>{gate:.0%} x{ss}':<30} {lbl:<12} {s:7.2f} "
                  f"{annualized_return(sub):7.1%} {max_drawdown(sub):7.1%} {len(sub):6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
