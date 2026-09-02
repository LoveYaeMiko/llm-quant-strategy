"""XGBoost GPU track — the GBDT twin of the LightGBM track, trained on CUDA.

The pip LightGBM wheel is CPU-only, so the GPU GBDT role goes to XGBoost
(``tree_method="hist", device="cuda"`` — verified 90% GPU utilisation on the
RTX 4060). Same walk-forward protocol as :mod:`src.ml.train`: purged-fold
early stopping inside the train window, val scored once, train+val refit,
test scored exactly once.

Deployment stays deterministic: the artifact is a saved booster + metadata;
:class:`XGBAdapter` scores on CPU with a single thread (fixed model → fixed
predictions), no GPU required at inference.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .cv import PurgedKFold
from .labels import align_features_labels, forward_return_labels, standardize_per_date
from .train import _slice, _window_metrics, build_feature_matrix

try:
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    xgb = None
    _HAS_XGB = False

_DEFAULT_PARAMS: dict[str, Any] = {
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 4,
    "random_state": 7,
}


@dataclass
class XGBArtifact:
    booster: Any
    features: list[str]
    feature_formulas: list[str]
    horizon: int
    fit_window: tuple[str, str]
    params: dict[str, Any]
    best_iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ml_xgb",
            "features": self.features,
            "feature_formulas": self.feature_formulas,
            "horizon": self.horizon,
            "fit_window": list(self.fit_window),
            "params": {k: v for k, v in self.params.items() if k != "random_state"},
            "best_iteration": self.best_iteration,
            "metadata": self.metadata,
        }

    def save(self, out_dir: Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        model = out_dir / f"xgb_{stamp}.json"
        meta = out_dir / f"xgb_{stamp}.meta.json"
        self.booster.save_model(str(model))
        meta.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"model": str(model), "meta": str(meta)}


def _dmatrix(frame: pd.DataFrame, label: Optional[pd.Series] = None) -> "xgb.DMatrix":
    """DMatrix from a feature frame — inf/huge values clipped to float32-safe.

    Some zoo formulas overflow (``Power(x, 240)``, near-zero divisions) — beyond
    XGBoost's float32 range the C++ layer rejects the data entirely. Clip to
    ±1e30 (far above any legitimate feature magnitude) and cast float32.
    """
    vals = frame.replace([np.inf, -np.inf], np.nan).values
    vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    vals = np.clip(vals.astype(np.float64), -1.0e30, 1.0e30).astype(np.float32)
    lab = label.values.astype(np.float32) if label is not None else None
    return xgb.DMatrix(vals, label=lab, feature_names=list(frame.columns), nthread=1)


class XGBAdapter:
    """Deterministic CPU scorer — the deployable face of the trained booster.

    ``ntree_limit`` pins the number of trees used (0 = all) — the final
    booster is trained without early stopping, so its ``best_iteration``
    attribute does not exist; the CV-selected round count is baked into the
    model by construction.
    """

    def __init__(self, booster: Any, ntree_limit: int = 0) -> None:
        self.booster = booster
        self.ntree_limit = int(ntree_limit)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dmat = _dmatrix(X)
        if self.ntree_limit and self.ntree_limit > 0:
            return self.booster.predict(dmat, iteration_range=(0, self.ntree_limit)).ravel()
        return self.booster.predict(dmat).ravel()


def load_xgb_artifact(model_path: str | Path) -> "xgb.Booster":
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed")
    booster = xgb.Booster()
    booster.load_model(str(Path(model_path)))
    return booster


def _purged_best_iterations(
    X: pd.DataFrame, y: pd.Series, params: dict[str, Any], horizon: int,
    n_folds: int, embargo_frac: float, n_estimators: int, early_stopping: int,
) -> int:
    kf = PurgedKFold(n_splits=n_folds, horizon=horizon, embargo_frac=embargo_frac)
    bests: list[int] = []
    for tr, va in kf.split(pd.Series(y.index.get_level_values(0))):
        tr_X, tr_y = X.iloc[tr], y.iloc[tr]
        va_X, va_y = X.iloc[va], y.iloc[va]
        dtr = _dmatrix(tr_X, label=tr_y)
        dva = _dmatrix(va_X, label=va_y)
        m = xgb.train(
            params, dtr, num_boost_round=n_estimators,
            evals=[(dva, "val")], early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        bests.append(int(m.best_iteration or n_estimators))
    return int(np.mean(bests))


def walk_forward_fit_xgb(
    market,
    formulas: Sequence[str],
    *,
    horizon: int = 10,
    train_window: tuple[str, str] = ("2010-01-01", "2019-12-31"),
    val_window: tuple[str, str] = ("2020-01-01", "2021-12-31"),
    test_window: tuple[str, str] = ("2022-01-01", "2025-12-31"),
    params: Optional[dict[str, Any]] = None,
    n_estimators: int = 300,
    early_stopping: int = 30,
    n_folds: int = 5,
    embargo_frac: float = 0.01,
    cost_bps: float = 5.0,
    out_dir: Optional[str | Path] = None,
    extra_features: Optional[dict[str, pd.Series]] = None,
    n_jobs: Optional[int] = None,
    device: str = "cuda",
) -> dict[str, Any]:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed")
    p = dict(_DEFAULT_PARAMS)
    p["device"] = device
    if params:
        p.update(params)
    print(f"[xgb] device={p['device']} max_depth={p['max_depth']}", flush=True)

    features = build_feature_matrix(market.long, formulas, n_jobs=n_jobs)
    if extra_features:
        extras = pd.concat([s.rename(k) for k, s in extra_features.items()], axis=1)
        features = features.join(extras, how="left")
    features = features.astype(np.float64)
    # XGBoost rejects inf ("Input data contains inf...") while LightGBM accepts
    # it — division near zero in some zoo formulas overflows. NaN is XGBoost's
    # native missing value.
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

    best_iter = _purged_best_iterations(
        X_tr, y_tr, p, horizon, n_folds, embargo_frac, n_estimators, early_stopping
    )
    print(f"[xgb] purged best iteration: {best_iter}", flush=True)
    params_fit = {k: v for k, v in p.items() if k != "random_state"}

    dtr = _dmatrix(X_tr, label=y_tr)
    model = xgb.train(params_fit, dtr, num_boost_round=int(best_iter), verbose_eval=False)
    adapter = XGBAdapter(model)
    X_va = _slice(X, *val_window)
    val_metrics = _window_metrics(adapter, X_va, y_raw[X_va.index], horizon, cost_bps) if len(X_va) else {}

    X_tv = _slice(X, train_window[0], val_window[1])
    y_tv = y[X_tv.index]
    dtv = _dmatrix(X_tv, label=y_tv)
    final = xgb.train(params_fit, dtv, num_boost_round=int(best_iter), verbose_eval=False)
    adapter_f = XGBAdapter(final)
    X_te = _slice(X, *test_window)
    test_metrics = _window_metrics(adapter_f, X_te, y_raw[X_te.index], horizon, cost_bps) if len(X_te) else {}

    artifact = XGBArtifact(
        booster=final,
        features=list(X.columns),
        feature_formulas=list(formulas),
        horizon=horizon,
        fit_window=(train_window[0], val_window[1]),
        params=params_fit,
        best_iteration=int(best_iter),
        metadata={
            "train_window": list(train_window),
            "val_window": list(val_window),
            "test_window": list(test_window),
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "extra_features": list(extra_features.keys()) if extra_features else [],
        },
    )
    paths: dict[str, str] = {}
    if out_dir is not None:
        paths = artifact.save(Path(out_dir))

    return {
        "model": "xgb",
        "horizon": horizon,
        "n_features": int(len(X.columns)),
        "best_iteration": int(best_iter),
        "val": val_metrics,
        "test": test_metrics,
        "artifact": paths,
    }


__all__ = [
    "XGBArtifact", "XGBAdapter", "load_xgb_artifact", "walk_forward_fit_xgb",
]
