"""D-track grid v3 — ML-scanner pullback (Martin Luk discipline + ML strong-stock
scanner). 2026 first (using the shadow feature cache — instant, exact), then a
full-panel build to validate the winner on 2025.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.paper.pullback_book import PullbackParams, PullbackPortfolio

COST = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}

BASE = dict(
    k=8, rank_source="ml", rank_min=0.8, mom_window=63, mom_long_rank_min=0.0,
    bounce_confirm=False, ema_fast=9, ema_zone=21, zone_band=0.02,
    pullback_min=0.03, vol_shrink=True, atr_mult=1.5, stop_lo=0.025,
    stop_hi=0.04, breakeven_r=1.0, trail_r=1.5, exit_into_strength_r=0.0,
    max_hold=40, entry_gate=0.0, exit_gate=-0.03, trend_days=60,
)

VARIANTS = {
    "D_ml": {},
    "D_ml_rank09": {"rank_min": 0.9},
    "D_ml_k6": {"k": 6},
    "D_ml_str3": {"exit_into_strength_r": 3.0},
    "D_ml_bounce": {"bounce_confirm": True},
    "D_ml_gate": {"entry_gate": 0.005, "exit_gate": 0.0},
    "D_ml_zone34": {"ema_zone": 34},
    "D_ml_novol": {"vol_shrink": False},
}


def round_trip_stats(fills: pd.DataFrame) -> dict:
    if fills is None or len(fills) == 0:
        return {"round_trips": 0, "win_rate": None, "avg_win_pct": None, "avg_loss_pct": None}
    fills = fills.sort_values("date")
    lots: dict[str, dict] = {}
    trips = []
    for _, f in fills.iterrows():
        sym = str(f["symbol"])
        side = str(f["side"]).lower()
        px = float(f["price"])
        sh = abs(float(f["shares"]))
        if side == "buy":
            lot = lots.get(sym)
            if lot is None:
                lots[sym] = {"entry": px, "qty": sh, "cost": sh * px}
            else:
                lot["qty"] += sh
                lot["cost"] += sh * px
                lot["entry"] = lot["cost"] / lot["qty"]
        else:
            lot = lots.get(sym)
            if lot is None:
                continue
            lot["qty"] -= sh
            if lot["qty"] <= 0:
                trips.append(px / lot["entry"] - 1.0)
                lots.pop(sym)
    if not trips:
        return {"round_trips": 0, "win_rate": None, "avg_win_pct": None, "avg_loss_pct": None}
    wins = [t for t in trips if t > 0]
    losses = [t for t in trips if t <= 0]
    return {
        "round_trips": len(trips),
        "win_rate": len(wins) / len(trips),
        "avg_win_pct": float(np.mean(wins)) if wins else None,
        "avg_loss_pct": float(np.mean(losses)) if losses else None,
    }


def run_variant(market, symbols, scores, params, wstart, wend, label) -> dict:
    from src.paper.ledger import PaperLedger
    from src.paper.runner import PaperRunner

    portfolio = PullbackPortfolio(market, params, symbols=symbols, scores=scores)
    ledger_path = ROOT / "outputs" / f"_dtune_{label}.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = PaperLedger(str(ledger_path))
    runner = PaperRunner(
        portfolio, market, ledger, symbols=symbols,
        cash=50_000, slippage_bps=COST["slippage"], commission_bps=COST["commission"],
        min_commission=COST["min"], stamp_tax_sell_bps=COST["stamp"],
        transfer_fee_bps=COST["transfer"], rebalance_days=1,
        notional_floor=2000, band_frac=0.0, pit_strict=True, seed=7,
    )
    out = runner.run(start=wstart, end=wend)
    fills = ledger.fills()
    ledger.close()
    ledger_path.unlink(missing_ok=True)
    m = out["metrics"]
    rt = round_trip_stats(fills)
    return {
        "cum_return": m.get("total_return", 0.0),
        "ann_return": m.get("annualized_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "n_fills": m.get("n_fills", 0),
        "cost": m.get("total_commission", 0.0),
        **rt,
    }


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.ml import load_artifact, score_artifact
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    market = _market_data(cfg, seed=1)
    symbols = [s for s in symbols if s in market.price_panel.columns]
    print(f"market: {len(symbols)} symbols", flush=True)

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    booster = load_artifact(meta_path.with_suffix(".txt"))

    # --- stage 1: 2026 grid from the shadow feature cache (exact, instant) ---
    cache_dir = ROOT / "outputs" / "cache" / "ml_book_features"
    cached = sorted(cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_size)
    frame26 = None
    for p in reversed(cached):
        f = pd.read_parquet(p)
        if list(f.columns) == meta["features"]:
            frame26 = f
            print(f"reused shadow feature cache {p.name} ({len(f)} rows)", flush=True)
            break
    if frame26 is None:
        raise SystemExit("no shadow feature cache matched the artifact — run the shadow first")

    scores26 = score_artifact(booster, frame26)
    print("scored 2026 (cached features)", flush=True)

    results: dict[str, dict] = {}
    for label, overrides in VARIANTS.items():
        params = PullbackParams(**{**BASE, **overrides})
        r = run_variant(market, symbols, scores26, params, "2026-01-01", "2026-08-28", f"{label}_2026")
        results[label] = {"2026": r}
        win_txt = "—" if r["win_rate"] is None else f"{r['win_rate']:.0%}"
        print(f"{label:12s} 2026: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"cost={r['cost']:,.0f} trips={r['round_trips']} win={win_txt}", flush=True)

    best = max(results, key=lambda k: results[k]["2026"]["ann_return"])
    print(f"\nstage1 best(2026): {best} ann={results[best]['2026']['ann_return']:+.2%}", flush=True)

    # --- stage 2: full-panel build → 2025 validation of the winner ---
    from src.data.margin import load_margin_extras_from_store
    from src.ml import build_feature_matrix

    frame_cache = ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet"
    if frame_cache.exists():
        frame_full = pd.read_parquet(frame_cache)
        print(f"reused full-panel frame cache ({len(frame_full)} rows)", flush=True)
    else:
        frame_full = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=12)
        extras = meta.get("metadata", {}).get("extra_features") or []
        if extras:
            margin = load_margin_extras_from_store(cfg)
            frame_full = frame_full.join(
                pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left"
            )
        frame_full.to_parquet(frame_cache)
        print("built full-panel frame", flush=True)

    scores_full = score_artifact(booster, frame_full)
    print("scored full panel", flush=True)
    r25 = run_variant(market, symbols, scores_full, PullbackParams(**{**BASE, **VARIANTS[best]}),
                      "2025-01-01", "2025-12-31", f"{best}_2025")
    results[best]["2025"] = r25
    win_txt = "—" if r25["win_rate"] is None else f"{r25['win_rate']:.0%}"
    print(f"{best:12s} 2025: cum={r25['cum_return']:+.2%} ann={r25['ann_return']:+.2%} "
          f"sharpe={r25['sharpe']:+.2f} maxDD={r25['max_dd']:.2%} fills={r25['n_fills']} "
          f"cost={r25['cost']:,.0f} trips={r25['round_trips']} win={win_txt}", flush=True)

    out_path = ROOT / "outputs" / "d_track_grid.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwinner: {best} | 2026 ann={results[best]['2026']['ann_return']:+.2%} "
          f"| 2025 cum={results[best]['2025']['cum_return']:+.2%}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
