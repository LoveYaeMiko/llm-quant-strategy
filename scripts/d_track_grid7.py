"""D-track grid v7 — smarter intraday stop triggers at cap 0.40.

v6 showed the naive "low breaches → sell" trigger whipsaws (ann 33.2% → 9.4%).
This grid tests confirmed-breach (minute CLOSE below stop), a buffer, and an
open-auction exemption — keeping the realistic intraday execution while
filtering wick noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.paper.pullback_book import PullbackParams, PullbackPortfolio

BASE = dict(
    k=6, rank_source="ml", rank_min=0.8, mom_window=63, mom_long_rank_min=0.0,
    bounce_confirm=False, ema_fast=21, ema_zone=21, zone_band=0.02,
    pullback_min=0.03, vol_shrink=True, atr_mult=1.5, stop_lo=0.025,
    stop_hi=0.04, breakeven_r=1.0, trail_r=1.5, exit_into_strength_r=3.0,
    max_hold=40, entry_gate=0.0, exit_gate=-0.03, trend_days=60,
    vwap_filter=0.0, stop_rv=False, tail_vol_max=0.5,
    open30_max=0.0, range_max=0.0, full_invest=True,
    stop_trigger="low", stop_buffer=0.0, stop_open_minutes=0,
)

VARIANTS = {
    "D_close_only": {"no_intraday": True},
    "D_low": {},
    "D_close_trig": {"stop_trigger": "close"},
    "D_close_buf": {"stop_trigger": "close", "stop_buffer": 0.005},
    "D_close_skip30": {"stop_trigger": "close", "stop_open_minutes": 30},
    "D_close_buf_skip30": {"stop_trigger": "close", "stop_buffer": 0.005, "stop_open_minutes": 30},
    "D_low_skip30": {"stop_open_minutes": 30},
}


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.data.intraday import make_minute_provider
    from src.ml import load_artifact, score_artifact
    from src.paper.ledger import PaperLedger
    from src.paper.runner import PaperRunner
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    market = _market_data(cfg, seed=1)
    symbols = [s for s in symbols if s in market.price_panel.columns]

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    booster = load_artifact(meta_path.with_suffix(".txt"))
    frame = pd.read_parquet(ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet")
    scores = score_artifact(booster, frame)
    provider = make_minute_provider(cfg)

    results = {}
    for label, over in VARIANTS.items():
        no_intraday = bool(over.pop("no_intraday", False))
        params = PullbackParams(**{**BASE, **over})
        portfolio = PullbackPortfolio(
            market, params, symbols=symbols, scores=scores,
            minute_provider=None if no_intraday else provider,
        )
        ledger_path = ROOT / "outputs" / f"_dgrid7_{label}.sqlite"
        ledger_path.unlink(missing_ok=True)
        ledger = PaperLedger(str(ledger_path))
        runner = PaperRunner(
            portfolio, market, ledger, symbols=symbols,
            cash=50_000, slippage_bps=2.0, commission_bps=2.5, min_commission=5.0,
            stamp_tax_sell_bps=5.0, transfer_fee_bps=0.1,
            rebalance_days=1, notional_floor=2000, band_frac=0.0,
            max_position_pct=0.40, pit_strict=True, seed=7,
        )
        out = runner.run(start="2026-01-01", end="2026-08-28")
        fills = ledger.fills()
        n_intraday = int((fills.get("time", "").fillna("") != "").sum()) if len(fills) else 0
        ledger.close()
        ledger_path.unlink(missing_ok=True)
        m = out["metrics"]
        results[label] = {
            "cum_return": m.get("total_return", 0.0),
            "ann_return": m.get("annualized_return", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_dd": m.get("max_drawdown", 0.0),
            "n_fills": m.get("n_fills", 0),
            "cost": m.get("total_commission", 0.0),
            "n_intraday": n_intraday,
        }
        r = results[label]
        print(f"{label:18s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"intraday={r['n_intraday']} cost={r['cost']:,.0f}", flush=True)

    out_path = ROOT / "outputs" / "d_track_grid7.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(results, key=lambda k: results[k]["ann_return"])
    print(f"\nbest: {best} ann={results[best]['ann_return']:+.2%}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
