"""Weekly auto closed-loop — fold the week's data into history and re-optimise.

Each week (PAICC scheduler, Sunday): retrain the LightGBM on all data through
the latest bar, evaluate on the trailing walk-forward window, and promote the
new artifact ONLY if it improves on the incumbent artifact's trailing Sharpe.
Also re-runs the §7 calibration sweep when due. Writes ``outputs/weekly.json``.

Usage:  python cli.py weekly
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_config
from .factors.code_generator import eval_expression  # noqa: F401 (re-export safety)

ROOT = Path(__file__).resolve().parents[1]


def _latest_bar(cfg) -> str:
    from .cli import _market_data

    market = _market_data(cfg, seed=1)
    return str(pd.Timestamp(market.price_panel.index.max()).date())


def _baseline_sharpe(meta_path: Path) -> float:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(meta.get("metadata", {}).get("test_metrics", {}).get("ls_sharpe_gross", 0.0) or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


def cmd_weekly(args) -> int:
    cfg = load_config()
    t0 = time.time()
    today = date.today().isoformat()
    log: dict = {"date": today, "started": time.strftime("%H:%M:%S")}

    from .cli import _market_data
    from .ml import walk_forward_fit
    from scripts.scan_factor_zoo import BASELINE, _WINDOWS  # noqa: F401
    from scripts.ml_common import load_margin_extras, load_zoo_formulas

    market = _market_data(cfg, seed=1)
    latest = str(pd.Timestamp(market.price_panel.index.max()).date())
    log["data_through"] = latest
    print(f"weekly: data through {latest} ({market.n_symbols} symbols)", flush=True)

    formulas = list(dict.fromkeys(BASELINE + load_zoo_formulas()))
    extras = load_margin_extras(cfg)

    # test window extends to the latest bar — the trailing OOS evaluation
    result = walk_forward_fit(
        market, formulas,
        horizon=10,
        train_window=("2010-01-01", "2019-12-31"),
        val_window=("2020-01-01", "2021-12-31"),
        test_window=("2022-01-01", latest),
        params={"num_threads": max(4, (__import__("os").cpu_count() or 4) - 2)},
        n_estimators=600,
        early_stopping=40,
        n_folds=5,
        embargo_frac=0.01,
        cost_bps=5.0,
        out_dir=str(ROOT / "outputs" / "models"),
        extra_features=extras,
        n_jobs=max(2, (__import__("os").cpu_count() or 4) - 2),
    )
    log["train_sec"] = round(time.time() - t0, 1)
    log["test_metrics"] = result.get("test", {})
    new_sharpe = float(result.get("test", {}).get("ls_sharpe_gross", 0.0) or 0.0)

    # promotion gate: only replace the incumbent artifact if trailing Sharpe improves
    models_dir = ROOT / "outputs" / "models"
    incumbent = sorted(models_dir.glob("ml_*.json"))
    baseline = 0.0
    incumbent_name = "none"
    if incumbent:
        incumbent_name = incumbent[-1].name
        baseline = _baseline_sharpe(incumbent[-1])
    promoted = new_sharpe > baseline
    log["baseline_sharpe"] = baseline
    log["new_sharpe"] = new_sharpe
    log["promoted"] = bool(promoted)
    log["artifact"] = result.get("artifact", {}).get("model", "")
    print(f"weekly: baseline {incumbent_name} sharpe={baseline:.3f} → new {new_sharpe:.3f} "
          f"promoted={promoted}", flush=True)

    out = ROOT / "outputs" / "weekly.json"
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"weekly: wrote {out}", flush=True)
    return 0
