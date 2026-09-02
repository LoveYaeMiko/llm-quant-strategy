"""Replicate the tune's B_5_5_gov on the FULL panel vs the shadow's sliced path.

Decides whether the shadow/tune return gap comes from the warmup truncation
(Jan book divergence) or from a pipeline bug.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

START, END = "2026-01-01", "2026-08-28"
COST = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}
V = {"cash": 100_000, "long_pct": 0.05, "short_pct": 0.05,
     "rebalance_days": 20, "notional_floor": 2000, "band_frac": 0.001}


class _BooksPortfolio:
    def __init__(self, books):
        self._books = books

    def compute_weights(self, symbols, date):
        return self._books.get(pd.Timestamp(date), {})


def build_books(market, scores, v):
    from src.portfolio.alpha_core import _market_trend, long_book_weights

    comp = scores[
        (scores.index.get_level_values(0) >= pd.Timestamp(START))
        & (scores.index.get_level_values(0) <= pd.Timestamp(END))
    ]
    trend = _market_trend(market.price_panel, 60)
    books = {}
    for d, day in comp.groupby(level=0):
        ss = 1.0
        if pd.Timestamp(d) in trend.index:
            t = trend[pd.Timestamp(d)]
            if np.isfinite(t) and t > 0.03:
                ss = 0.5
        books[pd.Timestamp(d)] = long_book_weights(
            day, long_pct=v["long_pct"], short_pct=v["short_pct"],
            max_position_pct=0.05, short_scale=ss,
        )
    return books


def run_variant(label, market, scores, v, seed):
    from src.paper.ledger import PaperLedger
    from src.paper.runner import PaperRunner

    books = build_books(market, scores, v)
    ledger_path = ROOT / "outputs" / f"_rep_{label}.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = PaperLedger(str(ledger_path))
    runner = PaperRunner(
        _BooksPortfolio(books), market, ledger,
        symbols=list(market.price_panel.columns),
        cash=v["cash"], slippage_bps=COST["slippage"], commission_bps=COST["commission"],
        min_commission=COST["min"], stamp_tax_sell_bps=COST["stamp"],
        transfer_fee_bps=COST["transfer"], rebalance_days=v["rebalance_days"],
        notional_floor=v["notional_floor"], band_frac=v["band_frac"],
        pit_strict=True, seed=seed,
    )
    out = runner.run(start=START, end=END)
    ledger.close()
    ledger_path.unlink(missing_ok=True)
    m = out["metrics"]
    print(f"{label:18s} cum={m.get('total_return', 0):+.2%} ann={m.get('annualized_return', 0):+.2%} "
          f"sharpe={m.get('sharpe', 0):+.2f} maxDD={m.get('max_drawdown', 0):+.2%} "
          f"fills={m.get('n_fills', 0)} cost={m.get('total_commission', 0):,.0f}", flush=True)
    return m


def main() -> int:
    from src.cli import _market_data, _slice_market
    from src.config import load_config
    from src.data.margin import load_margin_extras_from_store
    from src.ml import build_feature_matrix, load_artifact, score_artifact
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    extras = meta.get("metadata", {}).get("extra_features") or []
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    booster = load_artifact(meta_path.with_suffix(".txt"))
    margin = load_margin_extras_from_store(cfg)
    ext = pd.concat([margin[k].rename(k) for k in extras], axis=1)

    def scores_for(market):
        frame = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=12)
        frame = frame.join(ext, how="left")
        return score_artifact(booster, frame)

    # 1) tune replication: 1200d warmup (all 252-bar lookbacks complete -> full-panel values)
    full = _slice_market(_market_data(cfg, seed=1), "2022-09-19", None)
    s_full = scores_for(full)
    print("full(1200d) scored:", len(s_full), "rows", flush=True)
    run_variant("tune_repl_full", full, s_full, V, seed=7)

    # 2) old shadow replication: 360d warmup slice (truncated Jan lookbacks)
    sliced360 = _slice_market(_market_data(cfg, seed=1), "2025-01-06", None)
    s360 = scores_for(sliced360)
    print("slice360 scored:", len(s360), "rows", flush=True)
    run_variant("shadow_slice360", sliced360, s360, V, seed=7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
