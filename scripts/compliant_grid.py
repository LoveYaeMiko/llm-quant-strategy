"""Compliant-executor re-grid (2026 full-panel replay, legal fills).

Re-optimizes caps / rebalance / floors after the A-share compliance fixes
(100-share lots, ticks, limit-lock blocks). ML books are built from the cached
full-panel scores with the same long_book_weights semantics as the shadow.

# 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
# 注意：D_VARIANTS 的 base 未设 tail_vol_max（默认 0.0 = 门槛关闭），接入特征包只是
# 保证与生产同一条装配路径；如需生产门槛口径需在 base 显式加 tail_vol_max=0.5。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.portfolio.alpha_core import _market_trend, long_book_weights

COST = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}
START, END = "2026-01-01", "2026-08-28"

ML_VARIANTS = {
    "A_rb10":   dict(cash=2_000_000, uni="hs300", long_pct=0.10, cap=0.05, rb=10, floor=0.0, band=0.0),
    "A_rb15":   dict(cash=2_000_000, uni="hs300", long_pct=0.10, cap=0.05, rb=15, floor=0.0, band=0.0),
    "A_rb20":   dict(cash=2_000_000, uni="hs300", long_pct=0.10, cap=0.05, rb=20, floor=0.0, band=0.0),
    "B_cap08_rb10": dict(cash=100_000, uni="hs300_500", long_pct=0.05, cap=0.08, rb=10, floor=2000.0, band=0.001),
    "B_cap10_rb10": dict(cash=100_000, uni="hs300_500", long_pct=0.05, cap=0.10, rb=10, floor=2000.0, band=0.001),
    "B_cap12_rb10": dict(cash=100_000, uni="hs300_500", long_pct=0.05, cap=0.12, rb=10, floor=2000.0, band=0.001),
    "B_cap08_rb15": dict(cash=100_000, uni="hs300_500", long_pct=0.05, cap=0.08, rb=15, floor=2000.0, band=0.001),
    "B_cap10_rb15": dict(cash=100_000, uni="hs300_500", long_pct=0.05, cap=0.10, rb=15, floor=2000.0, band=0.001),
    "C_cap08_rb10": dict(cash=50_000, uni="hs300_500", long_pct=0.05, cap=0.08, rb=10, floor=1000.0, band=0.001),
    "C_cap10_rb10": dict(cash=50_000, uni="hs300_500", long_pct=0.05, cap=0.10, rb=10, floor=1000.0, band=0.001),
    "C_cap12_rb10": dict(cash=50_000, uni="hs300_500", long_pct=0.05, cap=0.12, rb=10, floor=1000.0, band=0.001),
    "C_cap10_rb15": dict(cash=50_000, uni="hs300_500", long_pct=0.05, cap=0.10, rb=15, floor=1000.0, band=0.001),
    "C_l4cap10":  dict(cash=50_000, uni="hs300_500", long_pct=0.04, cap=0.10, rb=10, floor=1000.0, band=0.001),
}

D_VARIANTS = {
    "D_k6": dict(k=6),
    "D_k5": dict(k=5),
    "D_k4": dict(k=4),
    "D_k8": dict(k=8),
}


class _BooksPortfolio:
    def __init__(self, books):
        self._books = books

    def compute_weights(self, symbols, date):
        return self._books.get(pd.Timestamp(date), {})


def ml_books(scores, symbols_used, long_pct, cap, market):
    comp = scores[
        (scores.index.get_level_values(0) >= pd.Timestamp(START))
        & (scores.index.get_level_values(0) <= pd.Timestamp(END))
    ]
    close = market.price_panel.reindex(columns=symbols_used)
    trend = _market_trend(close, 60)
    books = {}
    for d, day in comp.groupby(level=0):
        day = day[day.index.get_level_values(1).isin(symbols_used)]
        ss = 1.0
        if pd.Timestamp(d) in trend.index:
            t = trend[pd.Timestamp(d)]
            if np.isfinite(t) and t > 0.03:
                ss = 0.5
        books[pd.Timestamp(d)] = long_book_weights(
            day, long_pct=long_pct, short_pct=0.0, max_position_pct=cap, short_scale=ss
        )
    return books


def replay(label, portfolio, market, symbols, cash, rb, floor, band, cap) -> dict:
    from src.paper.ledger import PaperLedger
    from src.paper.runner import PaperRunner

    ledger_path = ROOT / "outputs" / f"_cgrid_{label}.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = PaperLedger(str(ledger_path))
    runner = PaperRunner(
        portfolio, market, ledger, symbols=symbols, cash=cash,
        slippage_bps=2.0, commission_bps=2.5, min_commission=5.0,
        stamp_tax_sell_bps=5.0, transfer_fee_bps=0.1,
        rebalance_days=rb, notional_floor=floor, band_frac=band,
        max_position_pct=cap, pit_strict=True, seed=7,
    )
    out = runner.run(start=START, end=END)
    ledger.close()
    ledger_path.unlink(missing_ok=True)
    m = out["metrics"]
    return {
        "cum_return": m.get("total_return", 0.0),
        "ann_return": m.get("annualized_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "n_fills": m.get("n_fills", 0),
        "cost": m.get("total_commission", 0.0),
    }


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.data.intraday import load_intraday_frames
    from src.ml import load_artifact, score_artifact
    from src.paper.pullback_book import PullbackParams, PullbackPortfolio
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    uni800 = resolve_shadow_universe(cfg, "hs300_500")
    uni300 = resolve_shadow_universe(cfg, "hs300")

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    booster = load_artifact(meta_path.with_suffix(".txt"))
    frame = pd.read_parquet(ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet")
    scores = score_artifact(booster, frame)

    # 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
    intraday = load_intraday_frames(cfg, uni800)
    print(f"market: {len(uni800)} symbols | scores ready | "
          f"intraday frames: {list(intraday)}", flush=True)

    results = {}
    for label, v in ML_VARIANTS.items():
        uni = uni300 if v["uni"] == "hs300" else uni800
        books = ml_books(scores, uni, v["long_pct"], v["cap"], market)
        r = replay(label, _BooksPortfolio(books), market, uni, v["cash"],
                   v["rb"], v["floor"], v["band"], v["cap"])
        results[label] = r
        print(f"{label:14s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"cost={r['cost']:,.0f}", flush=True)

    base = dict(rank_source="ml", rank_min=0.8, mom_window=63, mom_long_rank_min=0.0,
                bounce_confirm=False, ema_fast=21, ema_zone=21, zone_band=0.02,
                pullback_min=0.03, vol_shrink=True, atr_mult=1.5, stop_lo=0.025,
                stop_hi=0.04, breakeven_r=1.0, trail_r=1.5, exit_into_strength_r=3.0,
                max_hold=40, entry_gate=0.0, exit_gate=-0.03, trend_days=60,
                # 生产口径：尾盘 30 分钟量比门槛（2026-09-09 起显式对齐部署配置）
                tail_vol_max=0.5)
    for label, over in D_VARIANTS.items():
        params = PullbackParams(**{**base, **over})
        portfolio = PullbackPortfolio(
            market, params, symbols=uni800, scores=scores, intraday=intraday,
        )
        r = replay(label, portfolio, market, uni800, 50_000, 1, 2000.0, 0.0, 0.20)
        results[label] = r
        print(f"{label:14s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"cost={r['cost']:,.0f}", flush=True)

    out_path = ROOT / "outputs" / "compliant_grid.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for track, prefix in (("A", "A_"), ("B", "B_"), ("C", "C_"), ("D", "D_")):
        cands = {k: v for k, v in results.items() if k.startswith(prefix)}
        best = max(cands, key=lambda k: cands[k]["ann_return"])
        print(f"best {track}: {best} ann={cands[best]['ann_return']:+.2%} "
              f"cum={cands[best]['cum_return']:+.2%} sharpe={cands[best]['sharpe']:+.2f}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
