"""D-track tune round 3 — A-share-adapted wider stops (daily ±10% limit market
caps single-day gaps; wider stops + fewer slots keep per-name risk ≈ 0.5%).

# 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
# 注意：本脚本 BASE 未设 tail_vol_max（默认 0.0 = 门槛关闭），接入特征包只是保证
# 与生产同一条装配路径；如需生产门槛口径需在 BASE 显式加 tail_vol_max=0.5。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.paper.pullback_book import PullbackParams

COST = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}

BASE = dict(
    k=8, rank_source="ml", rank_min=0.8, mom_window=63, mom_long_rank_min=0.0,
    bounce_confirm=False, ema_fast=21, ema_zone=21, zone_band=0.02,
    pullback_min=0.03, vol_shrink=True, atr_mult=1.5, stop_lo=0.025,
    stop_hi=0.04, breakeven_r=1.0, trail_r=1.5, exit_into_strength_r=3.0,
    max_hold=40, entry_gate=0.0, exit_gate=-0.03, trend_days=60,
)

VARIANTS = {
    "D_t21": {},                                     # round-4 winner (baseline)
    "D_t21_wide": {"atr_mult": 2.5, "stop_lo": 0.04, "stop_hi": 0.06},
    "D_t21_k4_wide": {"k": 4, "atr_mult": 2.5, "stop_lo": 0.04, "stop_hi": 0.06},
    "D_t21_k6": {"k": 6},
    "D_t21_str5": {"exit_into_strength_r": 5.0},
}


def round_trip_stats(fills: pd.DataFrame) -> dict:
    if fills is None or len(fills) == 0:
        return {"round_trips": 0, "win_rate": None}
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
        return {"round_trips": 0, "win_rate": None}
    wins = [t for t in trips if t > 0]
    return {"round_trips": len(trips), "win_rate": len(wins) / len(trips)}


def run_variant(market, symbols, scores, params, wstart, wend, label, intraday=None) -> dict:
    from src.paper.ledger import PaperLedger
    from src.paper.pullback_book import PullbackPortfolio
    from src.paper.runner import PaperRunner

    portfolio = PullbackPortfolio(
        market, params, symbols=symbols, scores=scores, intraday=intraday,
    )
    ledger_path = ROOT / "outputs" / f"_dtune3_{label}.sqlite"
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
    from src.data.intraday import load_intraday_frames
    from src.ml import load_artifact, score_artifact
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    market = _market_data(cfg, seed=1)
    symbols = [s for s in symbols if s in market.price_panel.columns]

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    booster = load_artifact(meta_path.with_suffix(".txt"))

    cache_dir = ROOT / "outputs" / "cache" / "ml_book_features"
    frame26 = None
    for p in sorted(cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True):
        f = pd.read_parquet(p)
        if list(f.columns) == meta["features"]:
            frame26 = f
            break
    if frame26 is None:
        raise SystemExit("no shadow feature cache matched")
    scores26 = score_artifact(booster, frame26)

    # 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
    intraday = load_intraday_frames(cfg, symbols)
    print(f"market: {len(symbols)} | scores26 ready | intraday frames: {list(intraday)}",
          flush=True)

    results: dict[str, dict] = {}
    for label, overrides in VARIANTS.items():
        params = PullbackParams(**{**BASE, **overrides})
        r = run_variant(market, symbols, scores26, params, "2026-01-01", "2026-08-28",
                        f"{label}_2026", intraday=intraday)
        results[label] = {"2026": r}
        win_txt = "—" if r["win_rate"] is None else f"{r['win_rate']:.0%}"
        print(f"{label:14s} 2026: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"cost={r['cost']:,.0f} trips={r['round_trips']} win={win_txt}", flush=True)

    ranked = sorted(results, key=lambda k: results[k]["2026"]["ann_return"], reverse=True)
    top1 = ranked[0]

    frame_full_path = ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet"
    if frame_full_path.exists():
        frame_full = pd.read_parquet(frame_full_path)
        scores_full = score_artifact(booster, frame_full)
        r = run_variant(market, symbols, scores_full, PullbackParams(**{**BASE, **VARIANTS[top1]}),
                        "2025-01-01", "2025-12-31", f"{top1}_2025", intraday=intraday)
        results[top1]["2025"] = r
        win_txt = "—" if r["win_rate"] is None else f"{r['win_rate']:.0%}"
        print(f"{top1:14s} 2025: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"cost={r['cost']:,.0f} trips={r['round_trips']} win={win_txt}", flush=True)

    out_path = ROOT / "outputs" / "d_track_grid3.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwinner: {top1} | 2026 ann={results[top1]['2026']['ann_return']:+.2%}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
