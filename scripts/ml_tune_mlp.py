"""GPU hyperparameter search for the MLP — train-window-only protocol.

Grid-searches hidden sizes / learning rate / dropout using PURGED folds
INSIDE the train window (val/test windows are never touched by the search);
the winning config is refit on train+val and scored on test exactly once.
This is the "tune fast, deploy honestly" loop the GPU makes affordable.

Usage:  python scripts/ml_tune_mlp.py [--with-margin] [--device cuda]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ml.cv import PurgedKFold  # noqa: E402
from src.ml.labels import align_features_labels, forward_return_labels, standardize_per_date  # noqa: E402
from src.ml.train import _slice, build_feature_matrix  # noqa: E402
from src.ml.torch_model import RankMLP, TorchAdapter, _fit_epochs, train_walk_forward  # noqa: E402
from scripts.ml_common import load_margin_extras, load_zoo_formulas  # noqa: E402
from scripts.scan_factor_zoo import BASELINE, _WINDOWS  # noqa: E402

GRID = {
    "hidden": [(256, 128), (512, 256, 128), (512, 256, 128, 64)],
    "lr": [3e-4, 1e-3, 3e-3],
    "dropout": [0.1, 0.2, 0.3],
}


def _purged_loss(market, features, labels_std, y_raw, horizon, hyper, device, n_folds=3) -> float:
    tradable = getattr(market, "forward_returns_tradable", None)
    X, y = align_features_labels(features, labels_std, f"fwd_{horizon}", tradable)
    X_tr = _slice(X, *_WINDOWS["train"])
    y_tr = y[X_tr.index]
    tr_vals = X_tr.astype(np.float64).values.astype(np.float32)
    feat_mean = np.nanmean(tr_vals, axis=0)
    feat_std = np.where(np.nanstd(tr_vals, axis=0) > 0, np.nanstd(tr_vals, axis=0), 1.0)
    v = np.nan_to_num((tr_vals - feat_mean) / feat_std, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = torch.from_numpy(np.ascontiguousarray(v)).to(device)
    yt = torch.from_numpy(y_tr.values.astype(np.float32)).to(device)

    kf = PurgedKFold(n_splits=n_folds, horizon=horizon, embargo_frac=0.01)
    losses = []
    loss_fn = torch.nn.MSELoss()
    for tr_idx, va_idx in kf.split(pd.Series(y_tr.index.get_level_values(0))):
        torch.manual_seed(int(hyper["seed"]))
        m = RankMLP(len(X.columns), hyper["hidden"], hyper["dropout"]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=hyper["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hyper["epochs"])
        _fit_epochs(m, opt, sched, Xt[tr_idx], yt[tr_idx], hyper["batch_size"],
                    hyper["epochs"], Xt[va_idx], yt[va_idx], hyper["patience"])
        m.eval()
        with torch.no_grad():
            losses.append(loss_fn(m(Xt[va_idx]).ravel(), yt[va_idx]).item())
        del m, opt, sched
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(losses))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-margin", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=max(2, os.cpu_count() - 2))
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    t0 = time.time()
    market = _market_data(cfg, seed=1)
    print(f"market loaded [{time.time()-t0:.0f}s] device={dev}", flush=True)

    formulas = list(dict.fromkeys(BASELINE + load_zoo_formulas()))
    extras = load_margin_extras(cfg) if args.with_margin else None
    features = build_feature_matrix(market.long, formulas, n_jobs=args.jobs)
    if extras:
        features = features.join(
            pd.concat([s.rename(k) for k, s in extras.items()], axis=1), how="left"
        ).astype(np.float64)
    labels = forward_return_labels(market.price_panel, (10,))
    labels_std = standardize_per_date(labels)
    y_raw = labels["fwd_10"]
    print(f"features: {features.shape[1]}", flush=True)

    base = {"epochs": 12, "batch_size": 8192, "patience": 4, "seed": 7}
    results = []
    combos = list(itertools.product(GRID["hidden"], GRID["lr"], GRID["dropout"]))
    print(f"grid: {len(combos)} configs", flush=True)
    for hidden, lr, dropout in combos:
        hyper = {**base, "hidden": hidden, "lr": lr, "dropout": dropout}
        t1 = time.time()
        loss = _purged_loss(market, features, labels_std, y_raw, 10, hyper, dev)
        results.append({"hidden": list(hidden), "lr": lr, "dropout": dropout,
                        "purged_val_mse": loss, "sec": round(time.time() - t1, 1)})
        print(f"hidden={hidden} lr={lr:g} do={dropout} -> mse {loss:.5f} [{time.time()-t1:.0f}s]", flush=True)

    results.sort(key=lambda r: r["purged_val_mse"])
    best = results[0]
    print(f"\nbest: {best}", flush=True)
    out = ROOT / "outputs" / "mlp_tune_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # refit the winner on train+val, score test ONCE
    result = train_walk_forward(
        market, formulas,
        horizon=10,
        train_window=_WINDOWS["train"],
        val_window=_WINDOWS["val"],
        test_window=_WINDOWS["test"],
        hyper={"lr": best["lr"], "dropout": best["dropout"], "epochs": 20,
               "batch_size": 8192, "patience": 5, "seed": 7},
        n_folds=5,
        embargo_frac=0.01,
        cost_bps=5.0,
        out_dir=str(ROOT / "outputs" / "models"),
        device=dev,
        extra_features=extras,
        n_jobs=args.jobs,
        hidden=tuple(best["hidden"]),
    )
    print("RESULT", json.dumps(result, ensure_ascii=False, indent=2)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
