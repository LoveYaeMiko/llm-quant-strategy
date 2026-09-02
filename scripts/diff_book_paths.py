"""Diff tune-style (full-panel) vs shadow-style (warmup-sliced) ML books for B track.

Answers: does the 360d warmup slice change the artifact's scores/books for the
2026 shadow window (long-lookback truncation)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def main() -> int:
    from src.cli import _build_market_for_paper, _market_data
    from src.config import load_config
    from src.data.margin import load_margin_extras_from_store
    from src.ml import build_feature_matrix, load_artifact, score_artifact
    from src.paper.shadow import resolve_shadow_universe

    START, END = "2026-01-01", "2026-08-28"
    cfg = load_config()

    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    extras = meta.get("metadata", {}).get("extra_features") or []
    print("artifact:", meta_path.name, "| features:", len(meta["features"]), "| extras:", extras)

    wins = []
    for f in meta["feature_formulas"]:
        for tok in f.replace("(", " ").replace(")", " ").replace(",", " ").split():
            if tok.isdigit():
                wins.append(int(tok))
    print("max window literal:", max(wins) if wins else None)

    symbols = resolve_shadow_universe(cfg, "hs300_500")
    full = _market_data(cfg, seed=1)
    sliced = _build_market_for_paper(cfg, symbols, START, None, seed=1)
    print(f"full panel: {full.long.index.min()} .. {full.long.index.max()}, {len(full.long.columns)} cols")
    print(f"sliced:     {sliced.long.index.min()} .. {sliced.long.index.max()}, {len(sliced.long.columns)} cols")

    frame_full = build_feature_matrix(full.long, meta["feature_formulas"], n_jobs=4)
    frame_sliced = build_feature_matrix(sliced.long, meta["feature_formulas"], n_jobs=4)

    margin = load_margin_extras_from_store(cfg)
    ext = pd.concat([margin[k].rename(k) for k in extras], axis=1)
    frame_full = frame_full.join(ext, how="left")
    frame_sliced = frame_sliced.join(ext, how="left")

    booster = load_artifact(meta_path.with_suffix(".txt"))
    s_full = score_artifact(booster, frame_full)
    s_sliced = score_artifact(booster, frame_sliced)
    print("scored both paths", flush=True)

    rows = []
    for d in pd.date_range(START, END, freq="B"):
        if d not in s_full.index.get_level_values(0):
            continue
        if d not in s_sliced.index.get_level_values(0):
            continue
        a = s_full.xs(d, level=0).dropna()
        b = s_sliced.xs(d, level=0).dropna()
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue
        ar = a.loc[common].rank()
        br = b.loc[common].rank()
        rho = float(np.corrcoef(ar, br)[0, 1])
        rows.append((d, len(a), len(b), len(common), rho))

    out = pd.DataFrame(rows, columns=["date", "n_full", "n_sliced", "n_common", "rank_corr"])
    print("\nmonthly rank correlation (full-panel vs sliced features):")
    out["ym"] = out["date"].dt.to_period("M")
    print(out.groupby("ym")[["n_full", "n_sliced", "rank_corr"]].agg(
        n_full=("n_full", "first"), n_sliced=("n_sliced", "first"), rank_corr=("rank_corr", "mean")
    ).to_string())
    print("\nfirst 8 rows:")
    print(out.head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
