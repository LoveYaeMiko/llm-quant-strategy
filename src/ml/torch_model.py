"""GPU deep-model track — MLP rank predictor trained on the RTX 4060.

Reuses the SAME walk-forward plumbing as the LightGBM track
(:mod:`src.ml.train`'s labels/CV/feature builder): per-date z-scored
``h``-day forward returns, purged-fold early stopping inside the train
window, val scored once, train+val refit, test scored exactly once. Only the
estimator differs:

* **Train on GPU** (float32, AdamW + cosine schedule, batchnorm + dropout) —
  VRAM-bound on the RTX 4060, RAM stays free for the concurrent LightGBM run;
* **Deploy deterministically** — the artifact is a frozen state_dict + a
  feature/normalisation spec; :class:`TorchAdapter` scores on CPU in eval
  mode with fixed seed and ``no_grad``, so the online layer gets the same
  reproducibility guarantees as the LightGBM booster.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .cv import PurgedKFold
from .labels import align_features_labels, forward_return_labels, standardize_per_date
from .train import build_feature_matrix, _slice, _window_metrics  # noqa: F401 (re-export)

_DEFAULT_HIDDEN = (512, 256, 128)
_DEFAULT_HYPER = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "batch_size": 8192,
    "epochs": 20,
    "patience": 4,
    "seed": 7,
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RankMLP(nn.Module):
    """Cross-sectional rank predictor: batchnorm → ReLU → dropout stack."""

    def __init__(self, n_features: int, hidden: Sequence[int] = _DEFAULT_HIDDEN,
                 dropout: float = 0.2) -> None:
        super().__init__()
        dims = [n_features, *list(hidden)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TorchArtifact:
    state_dict: dict[str, torch.Tensor]
    features: list[str]
    feature_formulas: list[str]
    hidden: tuple[int, ...]
    dropout: float
    feat_mean: list[float]
    feat_std: list[float]
    horizon: int
    fit_window: tuple[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ml_mlp_torch",
            "features": self.features,
            "feature_formulas": self.feature_formulas,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "horizon": self.horizon,
            "fit_window": list(self.fit_window),
            "metadata": self.metadata,
        }

    def save(self, out_dir: Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        pt = out_dir / f"mlp_{stamp}.pt"
        meta = out_dir / f"mlp_{stamp}.json"
        torch.save(
            {
                "state_dict": {k: v.cpu().clone() for k, v in self.state_dict.items()},
                "feat_mean": self.feat_mean,
                "feat_std": self.feat_std,
            },
            pt,
        )
        meta.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"model": str(pt), "meta": str(meta)}


class TorchAdapter:
    """Deterministic CPU scorer — the deployable face of the trained model."""

    def __init__(self, model: nn.Module, feat_mean: np.ndarray, feat_std: np.ndarray) -> None:
        self.model = model
        self.model.eval()
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = np.where(feat_std > 0, feat_std, 1.0).astype(np.float32)
        torch.manual_seed(0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        vals = (X.values.astype(np.float32) - self.feat_mean) / self.feat_std
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            t = torch.from_numpy(vals)
            out = self.model(t)
        return out.numpy().ravel()


def load_torch_artifact(model_path: str | Path, meta_path: str | Path) -> TorchAdapter:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    model = RankMLP(len(meta["features"]), tuple(meta["hidden"]), meta["dropout"])
    model.load_state_dict(blob["state_dict"])
    return TorchAdapter(model, np.asarray(blob["feat_mean"]), np.asarray(blob["feat_std"]))


def _fit_epochs(
    model: nn.Module, opt: torch.optim.Optimizer, sched: Any,
    X: torch.Tensor, y: torch.Tensor, batch_size: int, epochs: int,
    val_X: torch.Tensor | None, val_y: torch.Tensor | None, patience: int,
) -> int:
    """Train up to ``epochs``; early-stop on validation MSE. Returns best epoch."""
    best_loss, best_epoch, bad = float("inf"), 0, 0
    loss_fn = nn.MSELoss()
    n = len(X)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[idx]).ravel(), y[idx])
            loss.backward()
            opt.step()
        sched.step()
        if val_X is not None and val_y is not None:
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(val_X).ravel(), val_y).item()
            if vloss < best_loss - 1e-6:
                best_loss, best_epoch, bad = vloss, ep, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            best_epoch = ep
    return best_epoch


def train_walk_forward(
    market,
    formulas: Sequence[str],
    *,
    horizon: int = 10,
    train_window: tuple[str, str] = ("2010-01-01", "2019-12-31"),
    val_window: tuple[str, str] = ("2020-01-01", "2021-12-31"),
    test_window: tuple[str, str] = ("2022-01-01", "2025-12-31"),
    hyper: Optional[dict[str, Any]] = None,
    n_folds: int = 5,
    embargo_frac: float = 0.01,
    cost_bps: float = 5.0,
    out_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    extra_features: Optional[dict[str, pd.Series]] = None,
    n_jobs: Optional[int] = None,
    hidden: Sequence[int] = _DEFAULT_HIDDEN,
) -> dict[str, Any]:
    h = dict(_DEFAULT_HYPER)
    if hyper:
        h.update(hyper)
    hidden = tuple(int(x) for x in hidden)
    dev = device or _device()
    torch.manual_seed(int(h["seed"]))
    np.random.seed(int(h["seed"]))
    print(f"[torch] device={dev} hidden={hidden} epochs={h['epochs']}", flush=True)

    features = build_feature_matrix(market.long, formulas, n_jobs=n_jobs)
    if extra_features:
        extras = pd.concat(
            [s.rename(k) for k, s in extra_features.items()], axis=1
        )
        features = features.join(extras, how="left")
    # plain float64 — joined extras may carry pd.NA (NAType)
    features = features.astype(np.float64)
    features = features.replace([np.inf, -np.inf], np.nan)
    labels = forward_return_labels(market.price_panel, (horizon,))
    y_raw = labels[f"fwd_{horizon}"].rename("fwd_raw")
    labels_std = standardize_per_date(labels)
    tradable = getattr(market, "forward_returns_tradable", None)
    X, y = align_features_labels(features, labels_std, f"fwd_{horizon}", tradable)
    y_raw = y_raw.reindex(X.index)

    X_tr = _slice(X, *train_window)
    y_tr = y[X_tr.index]
    if len(X_tr) < 200:
        raise ValueError("training window too small for a stable fit")

    # per-feature standardization from TRAIN ONLY (no val/test leakage)
    tr_vals = X_tr.values.astype(np.float32)
    feat_mean = np.nanmean(tr_vals, axis=0)
    feat_std = np.nanstd(tr_vals, axis=0)
    feat_std = np.where(feat_std > 0, feat_std, 1.0)

    def _tensor(df: pd.DataFrame) -> torch.Tensor:
        v = (df.values.astype(np.float32) - feat_mean) / feat_std
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(v)).to(dev)

    Xtr_t, ytr_t = _tensor(X_tr), torch.from_numpy(y_tr.values.astype(np.float32)).to(dev)

    # purged folds inside train → best epoch via validation loss
    kf = PurgedKFold(n_splits=n_folds, horizon=horizon, embargo_frac=embargo_frac)
    best_epochs: list[int] = []
    for tr_idx, va_idx in kf.split(pd.Series(y_tr.index.get_level_values(0))):
        m = RankMLP(len(X.columns), hidden, h["dropout"]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=h["epochs"])
        be = _fit_epochs(m, opt, sched, Xtr_t[tr_idx], ytr_t[tr_idx], h["batch_size"],
                         h["epochs"], Xtr_t[va_idx], ytr_t[va_idx], h["patience"])
        best_epochs.append(be)
        del m, opt, sched
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    best_epoch = int(np.mean(best_epochs))
    print(f"[torch] purged best epoch: {best_epoch} (folds {best_epochs})", flush=True)

    # stage 1: train on train-window, score val
    m1 = RankMLP(len(X.columns), hidden, h["dropout"]).to(dev)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=best_epoch)
    _fit_epochs(m1, opt1, sched1, Xtr_t, ytr_t, h["batch_size"], best_epoch, None, None, h["patience"])
    X_va = _slice(X, *val_window)
    adapter1 = TorchAdapter(m1.cpu(), feat_mean, feat_std)
    val_metrics = _window_metrics(adapter1, X_va, y_raw[X_va.index], horizon, cost_bps) if len(X_va) else {}

    # stage 2: refit on train+val for the artifact, score test
    X_tv = _slice(X, train_window[0], val_window[1])
    y_tv = y[X_tv.index]
    Xtv_t, ytv_t = _tensor(X_tv), torch.from_numpy(y_tv.values.astype(np.float32)).to(dev)
    final = RankMLP(len(X.columns), hidden, h["dropout"]).to(dev)
    optf = torch.optim.AdamW(final.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    schedf = torch.optim.lr_scheduler.CosineAnnealingLR(optf, T_max=best_epoch)
    _fit_epochs(final, optf, schedf, Xtv_t, ytv_t, h["batch_size"], best_epoch, None, None, h["patience"])
    X_te = _slice(X, *test_window)
    adapter_f = TorchAdapter(final.cpu(), feat_mean, feat_std)
    test_metrics = _window_metrics(adapter_f, X_te, y_raw[X_te.index], horizon, cost_bps) if len(X_te) else {}

    artifact = TorchArtifact(
        state_dict={k: v.cpu().detach().clone() for k, v in final.state_dict().items()},
        features=list(X.columns),
        feature_formulas=list(formulas),
        hidden=tuple(hidden),
        dropout=h["dropout"],
        feat_mean=feat_mean.tolist(),
        feat_std=feat_std.tolist(),
        horizon=horizon,
        fit_window=(train_window[0], val_window[1]),
        metadata={
            "train_window": list(train_window), "val_window": list(val_window),
            "test_window": list(test_window),
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "best_epoch": int(best_epoch), "device": str(dev),
            "extra_features": list(extra_features.keys()) if extra_features else [],
        },
    )
    paths: dict[str, str] = {}
    if out_dir is not None:
        paths = artifact.save(Path(out_dir))

    return {
        "model": "mlp_torch",
        "horizon": horizon,
        "n_features": int(len(X.columns)),
        "best_epoch": int(best_epoch),
        "val": val_metrics,
        "test": test_metrics,
        "artifact": paths,
    }


__all__ = [
    "RankMLP", "TorchAdapter", "TorchArtifact",
    "load_torch_artifact", "train_walk_forward",
]
