"""Warmup-sensitivity test: does the 360d shadow warmup truncate long-lookback
features and change the 2026 books vs a 540d warmup (max window = 252 bars)?

Compares artifact scores between two market slices that differ ONLY in warmup
length. High rank-corr in Feb+ but low in Jan => warmup truncation is the cause
of the B-track shadow/tune gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _slice_market_direct(market, start: str, warmup_calendar_days: int):
    """Slice market.long/price_panel to [start - warmup_calendar_days, None]."""
    lo = pd.Timestamp(start) - pd.Timedelta(days=warmup_calendar_days)
    long_ = market.long[market.long.index.get_level_values(0) >= lo]
    price = market.price_panel[market.price_panel.index >= lo]
    market.long = long_
    market.price_panel = price
    return market


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.data.margin import load_margin_extras_from_store
    from src.ml import build_feature_matrix, load_artifact, score_artifact
    from src.paper.shadow import resolve_shadow_universe

    START, END = "2026-01-01", "2026-08-28"
    cfg = load_config()

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    extras = meta.get("metadata", {}).get("extra_features") or []
    print("artifact:", meta_path.name, "| extras:", extras, flush=True)

    symbols = resolve_shadow_universe(cfg, "hs300_500")

    base = _market_data(cfg, seed=1)
    market360 = _slice_market_direct(base, START, 360)
    market540 = _slice_market_direct(_market_data(cfg, seed=1), START, 540)
    print(f"slice360: {market360.long.index.get_level_values(0).min()} .. "
          f"{market360.long.index.get_level_values(0).max()} | {len(market360.long.columns)} cols", flush=True)
    print(f"slice540: {market540.long.index.get_level_values(0).min()} .. "
          f"{market540.long.index.get_level_values(0).max()} | {len(market540.long.columns)} cols", flush=True)

    f360 = build_feature_matrix(market360.long, meta["feature_formulas"], n_jobs=12)
    f540 = build_feature_matrix(market540.long, meta["feature_formulas"], n_jobs=12)
    margin = load_margin_extras_from_store(cfg)
    ext = pd.concat([margin[k].rename(k) for k in extras], axis=1)
    f360 = f360.join(ext, how="left")
    f540 = f540.join(ext, how="left")
    print("features built", flush=True)

    booster = load_artifact(meta_path.with_suffix(".txt"))
    s360 = score_artifact(booster, f360)
    s540 = score_artifact(booster, f540)
    print("scored", flush=True)

    rows = []
    for d in pd.date_range(START, END, freq="B"):
        if d not in s360.index.get_level_values(0):
            continue
        if d not in s540.index.get_level_values(0):
            continue
        a = s360.xs(d, level=0).dropna()
        b = s540.xs(d, level=0).dropna()
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue
        ar = a.loc[common].rank()
        br = b.loc[common].rank()
        rows.append((d, len(a), len(b), len(common), float(np.corrcoef(ar, br)[0, 1])))

    out = pd.DataFrame(rows, columns=["date", "n_360", "n_540", "n_common", "rank_corr"])
    print("\nmonthly rank correlation (360d vs 540d warmup):")
    out["ym"] = out["date"].dt.to_period("M")
    print(out.groupby("ym")[["n_360", "n_540", "rank_corr"]].agg(
        n_360=("n_360", "first"), n_540=("n_540", "first"), rank_corr=("rank_corr", "mean")
    ).to_string())
    print("\nfirst 12 rows:")
    print(out.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
