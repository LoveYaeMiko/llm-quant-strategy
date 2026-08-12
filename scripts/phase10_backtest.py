"""Phase 10 full-sample backtest — 4-layer scenarios on HS300 2010-2025.

Runs the PHASE10_BLUEPRINT §4.1 comparison: baseline (pure Alpha core) /
+risk (sentiment circuit breaker) / +tilt (PEAD reversal) / three-layer full.
Each scenario prices the same weight-driven backtest; the gate (Sharpe > 1.6
AND maxDD < 10%) is evaluated per scenario.

Pipeline (all PIT, no look-ahead):
    HS300 universe  → lean SQL market loader (windowed bars, NOT the 12.48M-row
                      full snapshot)  →  AlphaCore(5 Phase 8 formulas)
                    → PEADFactor(cached profit panel)  →  seasonal tilt
                    → build_report_signal(cached report sentiment)  →  risk overlay
    port_ret[d] = Σ_i weights[d,i] · fwd[d,i]        (weights from data at d)

Usage:
    python scripts/phase10_backtest.py [--symbols hs300] [--start ...] [--end ...]

Outputs: outputs/phase10_backtest.json + console comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_universe(config, name: str) -> list[str]:
    from src.cli import _cached_universe_json

    return _cached_universe_json(name, config)


def _load_factor_formulas() -> list[str]:
    """The 5 accepted Phase 8 formulas from outputs/factors.json."""
    path = ROOT / "outputs" / "factors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    formulas = [x.get("factor", {}).get("formula", x.get("formula")) for x in data]
    formulas = [f for f in formulas if isinstance(f, str) and f.strip()]
    return formulas


def _build_sentiment_panel(config, symbols: list[str], trading_dates) -> pd.Series:
    """PIT carry-forward report-sentiment panel from the cached title scores."""
    from src.sentiment.ingestion import ReportIngestor
    from src.sentiment.triagent import build_report_signal

    sent_cfg = config.get("sentiment") or {}
    report_dir = str(sent_cfg.get("report_dir", "data/reports"))
    score_cache = str(sent_cfg.get("score_cache", "data/reports/report_sentiment.parquet"))
    decay = int(sent_cfg.get("decay_days", 10))

    ingestor = ReportIngestor(report_dir)
    cached = ingestor.cached_symbols()
    have = set(symbols) & cached
    if not have:
        print("WARNING: no cached reports for the universe — risk overlay is a no-op")
        return pd.Series(dtype=float)
    reports = ingestor.load(symbols=sorted(have))
    scores = pd.read_parquet(score_cache)
    sig = build_report_signal(reports, scores, trading_dates, sorted(have), decay_days=decay)
    print(f"sentiment: {len(reports)} reports, {int(sig.notna().sum())} signal cells "
          f"({len(have)}/{len(symbols)} symbols covered)")
    return sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="hs300")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--config", action="append", default=[])
    args = ap.parse_args()

    from src.config import load_config
    from src.data.financials import ensure_profit_panel
    from src.factors.code_generator import FactorContext
    from src.factors.pead import PEADFactor
    from src.portfolio.alpha_core import AlphaCore
    from src.portfolio.backtest_runner import (
        load_hs300_market,
        run_phase10_scenarios,
        write_json,
    )

    cfg = load_config(*args.config) if args.config else load_config()

    t0 = time.time()
    symbols = _load_universe(cfg, args.symbols)
    print(f"universe: {len(symbols)} symbols ({args.symbols})")

    # ------------------------------------------------------------ market
    # Warm-up buffer: the alpha lookbacks are 120/240-day rolling, so load 360
    # calendar days *before* the window (the backtest slice still starts at
    # `args.start`). Without it the book is empty until the lookbacks fill.
    warmup_start = (pd.Timestamp(args.start) - pd.Timedelta(days=360)).date().isoformat()
    print(f"loading market (windowed SQL, warmup from {warmup_start})...")
    market = load_hs300_market(cfg, symbols, start=warmup_start, end=args.end)
    trading_dates = sorted(market.forward_returns.index.get_level_values(0).unique())
    print(f"market: {len(market.price_panel.columns)} symbols x "
          f"{len(trading_dates)} days [{trading_dates[0].date()} .. {trading_dates[-1].date()}] "
          f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------ alpha core
    formulas = _load_factor_formulas()
    print(f"alpha core: {len(formulas)} Phase 8 formulas")
    fctx = FactorContext(market.long)
    # regime trend for the short-leg control: equal-weight tradable-market 60d
    # compound return, PIT-safe — bench[d-1] is the last realised return at
    # close d. Reproduces the probe that hit maxDD 9.2% (trend>3% short x0.5).
    _fwd_wide = market.forward_returns_tradable.unstack(fill_value=0.0)
    _bench = _fwd_wide.reindex(columns=symbols).mean(axis=1)
    _trend = (1 + _bench).shift(1).rolling(60).apply(lambda x: x.prod() - 1, raw=True)
    alpha = AlphaCore(fctx, formulas, long_pct=0.10, max_position_pct=0.05,
                      trend_series=_trend)
    print(f"composite: {alpha.composite.notna().sum()} non-NaN cells "
          f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------ PEAD tilt
    pead_cfg = cfg.get("pead") or {}
    years = [int(y) for y in range(2020, 2026)]
    panel = ensure_profit_panel(symbols, years, cache_dir=str(pead_cfg.get("cache_dir", "data/financials")))
    pead = PEADFactor(
        panel,
        signal_expiry_days=int(pead_cfg.get("signal_expiry_days", 60)),
        min_eps_history=int(pead_cfg.get("min_eps_history", 8)),
    )
    print(f"pead: {len(pead.symbols)} symbols with quarterly EPS, panel rows={len(panel)} "
          f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------ risk overlay
    sentiment_panel = _build_sentiment_panel(cfg, symbols, trading_dates)
    _RK = {"zscore_threshold", "position_cut", "freeze_days", "min_trigger_samples"}
    risk_kwargs = {k: v for k, v in (cfg.get("risk_overlay") or {}).items() if k in _RK}

    # ------------------------------------------------------------ 4 scenarios
    results = run_phase10_scenarios(
        market, alpha, pead=pead, sentiment_panel=sentiment_panel,
        symbols=symbols, start=args.start, end=args.end,
        risk_kwargs=risk_kwargs,
    )
    results["window"] = {"start": args.start, "end": args.end, "universe": args.symbols}
    results["runtime_s"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------ report
    print("\n=== PHASE 10 SCENARIOS ===")
    header = f"{'scenario':<12} {'sharpe':>7} {'ann_ret':>9} {'maxDD':>7} {'turnover':>8} {'days':>6}  gate"
    print(header)
    print("-" * len(header))
    for row in results["table"]:
        print(
            f"{row['scenario']:<12} {row['sharpe']:>7.2f} {row['annualized_return']:>8.1%} "
            f"{row['max_drawdown']:>6.1%} {row['turnover']:>8.2f} {row['n_days']:>6}  "
            f"{'PASS' if row['gate_passed'] else 'FAIL'}"
        )
    print(f"\nrisk triggers per scenario: "
          f"{ {k: v['risk_triggers'] for k, v in results['scenarios'].items() if 'risk' in k} }")

    out = ROOT / "outputs" / "phase10_backtest.json"
    write_json(out, results)
    print(f"\nwrote {out} ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
