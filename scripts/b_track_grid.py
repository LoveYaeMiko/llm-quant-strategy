"""B-track follow-up grid: rebalance 10 vs 20 days x short leg 0% vs 5%.

Full-panel replay of the 2026 window (same as dual_track_tune) — the baseline
B_5_5_gov (20d) reproduced +13.45% ann there; check whether 10d rebalance or
dropping the short leg lifts the 10W track further.
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

VARIANTS = [
    {"label": "B_5_5_gov_rb20", "cash": 100_000, "long_pct": 0.05, "short_pct": 0.05,
     "rebalance_days": 20, "notional_floor": 2000, "band_frac": 0.001},
    {"label": "B_5_5_gov_rb10", "cash": 100_000, "long_pct": 0.05, "short_pct": 0.05,
     "rebalance_days": 10, "notional_floor": 2000, "band_frac": 0.001},
    {"label": "B_5_0_gov_rb20", "cash": 100_000, "long_pct": 0.05, "short_pct": 0.0,
     "rebalance_days": 20, "notional_floor": 2000, "band_frac": 0.001},
    {"label": "B_5_0_gov_rb10", "cash": 100_000, "long_pct": 0.05, "short_pct": 0.0,
     "rebalance_days": 10, "notional_floor": 2000, "band_frac": 0.001},
]


class _BooksPortfolio:
    def __init__(self, books):
        self._books = books

    def compute_weights(self, symbols, date):
        return self._books.get(pd.Timestamp(date), {})


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.data.margin import load_margin_extras_from_store
    from src.ml import build_feature_matrix, load_artifact, score_artifact
    from src.paper.ledger import PaperLedger
    from src.paper.runner import PaperRunner
    from src.portfolio.alpha_core import _market_trend, long_book_weights

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    symbols = list(market.price_panel.columns)
    print(f"market: {len(symbols)} symbols", flush=True)

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=12)
    extras = meta.get("metadata", {}).get("extra_features") or []
    if extras:
        margin = load_margin_extras_from_store(cfg)
        frame = frame.join(pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left")
    booster = load_artifact(meta_path.with_suffix(".txt"))
    scores = score_artifact(booster, frame)
    comp = scores[
        (scores.index.get_level_values(0) >= pd.Timestamp(START))
        & (scores.index.get_level_values(0) <= pd.Timestamp(END))
    ]
    trend = _market_trend(market.price_panel, 60)
    print(f"scored {meta_path.name}", flush=True)

    results = {}
    for v in VARIANTS:
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
        ledger_path = ROOT / "outputs" / f"_btune_{v['label']}.sqlite"
        ledger_path.unlink(missing_ok=True)
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
            "cum_return": m.get("total_return", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_dd": m.get("max_drawdown", 0.0),
            "n_fills": m.get("n_fills", 0),
            "cost": m.get("total_commission", 0.0),
        }
        print(f"{v['label']:16s} cum={results[v['label']]['cum_return']:+.2%} "
              f"ann={results[v['label']]['ann_return']:+.2%} "
              f"sharpe={results[v['label']]['sharpe']:+.2f} "
              f"maxDD={results[v['label']]['max_dd']:.2%} "
              f"fills={results[v['label']]['n_fills']} cost={results[v['label']]['cost']:,.0f}", flush=True)

    out_path = ROOT / "outputs" / "b_track_grid.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(results, key=lambda k: results[k]["ann_return"])
    print(f"\nbest: {best}  ann={results[best]['ann_return']:+.2%}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
