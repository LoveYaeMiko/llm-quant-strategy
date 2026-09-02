"""GPU deep-model training run — baseline + full zoo (+ margin extras).

Runs IN PARALLEL with the LightGBM track (VRAM + RAM used together). Same
pre-registered protocol: purged CV + embargo, walk-forward windows.

Usage:  python scripts/ml_train_torch.py [--with-margin]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ml.torch_model import train_walk_forward  # noqa: E402
from scripts.ml_common import load_margin_extras, load_zoo_formulas  # noqa: E402
from scripts.scan_factor_zoo import BASELINE, _WINDOWS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-margin", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(2, os.cpu_count() - 2))
    args = ap.parse_args()

    cfg = load_config()
    t0 = time.time()
    market = _market_data(cfg, seed=1)
    print(f"market loaded [{time.time()-t0:.0f}s]", flush=True)

    features = list(dict.fromkeys(BASELINE + load_zoo_formulas()))
    print(f"features: {len(features)} formulas, jobs={args.jobs}", flush=True)

    extras = None
    if args.with_margin:
        extras = load_margin_extras(cfg)

    result = train_walk_forward(
        market,
        features,
        horizon=10,
        train_window=_WINDOWS["train"],
        val_window=_WINDOWS["val"],
        test_window=_WINDOWS["test"],
        n_folds=5,
        embargo_frac=0.01,
        cost_bps=5.0,
        out_dir=str(ROOT / "outputs" / "models"),
        extra_features=extras,
        n_jobs=args.jobs,
    )
    print("RESULT", json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
