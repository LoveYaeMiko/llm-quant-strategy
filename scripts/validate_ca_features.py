"""Validate corporate-action event features: rank-IC vs 10d forward return on
the walk-forward test window (2022-2025) + Spearman corr with the deployed ML
score (marginal value proxy)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config
    from src.data.corp_actions import event_features
    from src.ml import load_artifact, score_artifact

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    close = market.price_panel
    feats = event_features(cfg, close)
    print("feature frames:", {k: v.shape for k, v in feats.items()}, flush=True)

    fwd10 = close.pct_change(10, fill_method=None).shift(-10)
    mask = (close.index >= pd.Timestamp("2022-01-01")) & (close.index <= pd.Timestamp("2025-12-31"))

    print("\nrank-IC vs fwd10 return (test window 2022-2025):")
    for k, f in feats.items():
        fw = f[mask]
        r = fwd10[mask]
        ics = []
        for d in fw.index.unique():
            x = fw.loc[d]
            y = r.loc[d]
            both = x.notna() & y.notna()
            if both.sum() < 30:
                continue
            ics.append(np.corrcoef(x[both].rank(), y[both].rank())[0, 1])
        ics = np.array(ics)
        if len(ics):
            print(f"  {k:14s}: mean_IC={ics.mean():+.4f}  ICIR={ics.mean() / ics.std() if ics.std() > 0 else float('nan'):+.2f}  n={len(ics)}")
        else:
            print(f"  {k:14s}: no data")

    # marginal value vs the deployed model
    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet")
    booster = load_artifact(meta_path.with_suffix(".txt"))
    scores = score_artifact(booster, frame).unstack()

    print("\nSpearman corr with deployed ML score (test window):")
    for k, f in feats.items():
        fw = f[mask].stack(dropna=False)
        sw = scores.reindex(index=mask.index, columns=f.columns).stack(dropna=False)
        both = fw.notna() & sw.notna()
        if both.sum() < 100:
            print(f"  {k:14s}: insufficient overlap")
            continue
        print(f"  {k:14s}: rho={np.corrcoef(fw[both].rank(), sw[both].rank())[0, 1]:+.3f}  (n={int(both.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
