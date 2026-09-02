"""Dual-track parameter tuning — replay the 2026 shadow window per variant.

One market load, one artifact scoring, then a small grid of book/rebalance/
governance variants per track. The winner per track is written back to
``configs/master_config.yaml`` (``shadow.accounts``). Variants:

* A track (2M): rebalance 10 vs 20 days, with/without governance;
* B track (100k): 5%/5% vs 5%/7% deciles, governance on (floor 2000, band 0.1%).

Usage:  python scripts/dual_track_tune.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ml import build_feature_matrix, load_artifact, score_artifact  # noqa: E402
from src.paper.ledger import PaperLedger  # noqa: E402
from src.paper.ml_book import MLBookPortfolio  # noqa: E402
from src.paper.runner import PaperRunner  # noqa: E402
from src.data.margin import load_margin_extras_from_store  # noqa: E402
from src.portfolio.alpha_core import _market_trend, long_book_weights  # noqa: E402

START, END = "2026-01-01", "2026-08-28"
COST = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}

VARIANTS = [
    {"label": "A_rb10_raw",   "cash": 2_000_000, "long_pct": 0.10, "short_pct": 0.10,
     "rebalance_days": 10, "notional_floor": 0,     "band_frac": 0.0},
    {"label": "A_rb20_gov",   "cash": 2_000_000, "long_pct": 0.10, "short_pct": 0.10,
     "rebalance_days": 20, "notional_floor": 5000,  "band_frac": 0.001},
    {"label": "B_5_5_gov",    "cash": 100_000, "long_pct": 0.05, "short_pct": 0.05,
     "rebalance_days": 20, "notional_floor": 2000,  "band_frac": 0.001},
    {"label": "B_5_7_gov",    "cash": 100_000, "long_pct": 0.05, "short_pct": 0.07,
     "rebalance_days": 20, "notional_floor": 2000,  "band_frac": 0.001},
    {"label": "B_7_7_gov",    "cash": 100_000, "long_pct": 0.07, "short_pct": 0.07,
     "rebalance_days": 20, "notional_floor": 2000,  "band_frac": 0.001},
]


class _BooksPortfolio:
    def __init__(self, books):
        self._books = books

    def compute_weights(self, symbols, date):
        return self._books.get(pd.Timestamp(date), {})


def _books_for(market, scores, variant) -> dict:
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
            day, long_pct=variant["long_pct"], short_pct=variant["short_pct"],
            max_position_pct=0.05, short_scale=ss,
        )
    return books


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write winners back to master_config.yaml")
    args = ap.parse_args()

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    symbols = list(market.price_panel.columns)
    print(f"market: {len(symbols)} symbols", flush=True)

    # one artifact scoring: newest LightGBM (345 features + margin extras)
    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=None)
    extras = meta.get("metadata", {}).get("extra_features") or []
    if extras:
        margin = load_margin_extras_from_store(cfg)
        frame = frame.join(pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left")
    booster = load_artifact(meta_path.with_suffix(".txt"))
    scores = score_artifact(booster, frame)
    print(f"scored artifact {meta_path.name}", flush=True)

    results = {}
    for v in VARIANTS:
        books = _books_for(market, scores, v)
        ledger_path = ROOT / "outputs" / f"_tune_{v['label']}.sqlite"
        if ledger_path.exists():
            ledger_path.unlink()
        ledger = PaperLedger(str(ledger_path))
        runner = PaperRunner(
            _BooksPortfolio(books), market, ledger, symbols=symbols,
            cash=v["cash"], slippage_bps=COST["slippage"], commission_bps=COST["commission"],
            min_commission=COST["min"], stamp_tax_sell_bps=COST["stamp"],
            transfer_fee_bps=COST["transfer"], rebalance_days=v["rebalance_days"],
            notional_floor=v["notional_floor"], band_frac=v["band_frac"],
            pit_strict=True, seed=7,
        )
        out = runner.run(start=START, end=END)
        ledger.close()
        ledger_path.unlink(missing_ok=True)
        m = out["metrics"]
        results[v["label"]] = {
            "ann_return": m.get("annualized_return", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_dd": m.get("max_drawdown", 0.0),
            "n_fills": m.get("n_fills", 0),
            "cost": m.get("total_commission", 0.0),
            "variant": {k: val for k, val in v.items() if k != "label"},
        }
        print(f"{v['label']:14s} ann={results[v['label']]['ann_return']:+.2%} "
              f"sharpe={results[v['label']]['sharpe']:+.2f} "
              f"maxDD={results[v['label']]['max_dd']:.2%} "
              f"fills={results[v['label']]['n_fills']} "
              f"cost={results[v['label']]['cost']:,.0f}", flush=True)

    out = ROOT / "outputs" / "dual_track_tune.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # winners per track (best Sharpe among that track's variants)
    a_best = max((k for k in results if k.startswith("A_")), key=lambda k: results[k]["sharpe"])
    b_best = max((k for k in results if k.startswith("B_")), key=lambda k: results[k]["sharpe"])
    print(f"\nwinners: A -> {a_best}, B -> {b_best}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
