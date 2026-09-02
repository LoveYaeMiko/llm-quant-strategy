"""ML track — walk-forward LightGBM trainer + deterministic frozen artifact.

The first ML model for FQA: a cross-sectional rank predictor trained on the
factor zoo (formulas evaluated by the closed operator library) with per-date
z-scored ``h``-day forward-return labels.

Discipline (the whole point of the track):

* **Walk-forward** — the model is fitted on ``train``, early-stopped with purged
  folds *inside* ``train`` only, scored on ``val``, refitted on ``train+val``,
  and reported on ``test`` exactly once. Nothing beyond the fit window is ever
  seen during fitting.
* **Purged K-fold + embargo** — label overlap across fold boundaries is removed
  (:mod:`src.ml.cv`), so the early-stopping signal is honest.
* **Frozen artifact** — the online layer never touches the training pipeline:
  the booster is exported as a text model + a metadata JSON, and
  :func:`score_artifact` evaluates it with a single-threaded booster for
  bit-reproducible predictions. No torch, no LLM, no dynamic code.
* **Cost-aware reporting** — the long-short tail spread on ``test`` is reported
  net of a per-side cost charge so "Sharpe" here means what the ledger means.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ..backtest.metrics import daily_ic, tail_long_short_returns
from .cv import PurgedKFold
from .labels import align_features_labels, forward_return_labels, standardize_per_date

try:
    import lightgbm as lgb  # type: ignore

    _HAS_LGB = True
except ImportError:  # pragma: no cover — offline tests skip the trainer
    lgb = None
    _HAS_LGB = False


_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_lambda": 1.0,
    "num_threads": 1,          # determinism: single-threaded fit + predict
    "verbose": -1,
    "seed": 7,
}


_EVAL_CTX = None  # per-worker FactorContext, set by _init_eval_worker


def _init_eval_worker(panel: pd.DataFrame) -> None:
    """Process-pool initializer — one FactorContext per worker (Windows spawn)."""
    global _EVAL_CTX
    from ..factors.code_generator import FactorContext

    _EVAL_CTX = FactorContext(panel)


def _eval_formula(item) -> tuple[str, pd.Series]:
    """Evaluate one formula in the worker's context (module-level: picklable)."""
    global _EVAL_CTX
    from ..factors.code_generator import eval_expression

    idx, f = item
    name = f"f{idx:03d}_{_sanitize(f)}"
    s = eval_expression(f, _EVAL_CTX)
    if not hasattr(s, "groupby"):
        raise ValueError(f"formula produced a scalar, not a signal: {f!r}")
    return name, s.astype(float).rename(name)


def build_feature_matrix(
    panel: pd.DataFrame,
    formulas: Sequence[str],
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate ``formulas`` on the (date, symbol) ``panel`` → feature frame.

    ``panel`` is the long market panel (``market.long``); each formula goes
    through the closed operator library (:mod:`src.factors.code_generator`), so
    every feature is PIT-clean by construction and whitelist-only.

    ``n_jobs`` parallelises across formulas with a process pool (each worker
    holds its own FactorContext over the shared panel) — the 300+ formula zoo
    build drops from ~25 min to ~2-4 min on this machine.
    """
    from ..factors.code_generator import FactorContext, eval_expression

    names = [f"f{i:03d}_{_sanitize(f)}" for i, f in enumerate(formulas)]
    if n_jobs and n_jobs > 1 and len(formulas) > 4:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_eval_worker, initargs=(panel,)) as pool:
            pairs = list(pool.map(_eval_formula, list(enumerate(formulas))))
        cols = dict(pairs)
    else:
        fctx = FactorContext(panel)
        cols = {}
        for i, f in enumerate(formulas):
            s = eval_expression(f, fctx)
            if not hasattr(s, "groupby"):
                raise ValueError(f"formula produced a scalar, not a signal: {f!r}")
            # rename explicitly — eval_expression may carry a formula/field name on
            # the series, which LightGBM would reject as duplicate column names
            cols[names[i]] = s.astype(float).rename(names[i])
    out = pd.concat(cols.values(), axis=1)
    out.index.names = ["date", "symbol"]
    return out


def _sanitize(formula: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in formula)
    return keep[:48].strip("_").lower()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index.get_level_values(0) >= pd.Timestamp(start)) & (
        df.index.get_level_values(0) <= pd.Timestamp(end)
    )
    return df[mask]


def _window_metrics(
    model: Any,
    X: pd.DataFrame,
    y_raw: pd.Series,
    horizon: int,
    cost_bps: float,
) -> dict[str, float]:
    """Rank-IC + cost-aware long-short tail metrics for one window.

    ``y_raw`` is the *raw* h-day forward return (not the z-scored training
    target), so the tail spread is a real return. The overlapping daily spread
    is de-overlapped by subsampling every ``horizon`` days (each kept entry is
    one non-overlapping h-day return of a book rebalanced every ``horizon``
    days); annualisation then uses ``252 / horizon`` periods per year.
    """
    pred = pd.Series(model.predict(X), index=X.index)
    df = pd.DataFrame({"sig": pred, "fwd": y_raw}).dropna()
    ic = daily_ic(df["sig"], df["fwd"], method="spearman").dropna()
    ls = tail_long_short_returns(df["sig"], df["fwd"])
    sub = ls.iloc[::horizon] if len(ls) else ls
    ppy = 252.0 / horizon
    ann = float(sub.mean() * ppy) if len(sub) else 0.0
    sharpe = float(sub.mean() / sub.std() * np.sqrt(ppy)) if len(sub) > 2 and sub.std() > 0 else 0.0
    maxdd = float(((1.0 + sub).cumprod().div((1.0 + sub).cumprod().cummax()) - 1.0).min()) if len(sub) else 0.0
    # cost: one round trip (two sides) per rebalance, i.e. per horizon days
    cost_ann = float(cost_bps / 10_000.0 * 2.0 * ppy)
    return {
        "rank_ic": float(ic.mean()),
        "icir": float(ic.mean() / ic.std() * np.sqrt(252.0)) if len(ic) > 2 and ic.std() > 0 else 0.0,
        "n_days": int(len(ic)),
        "ls_ann_gross": ann,
        "ls_sharpe_gross": sharpe,
        "ls_maxdd_gross": maxdd,
        "ls_ann_net": ann - cost_ann,
    }


def _purged_best_iterations(
    X: pd.DataFrame, y: pd.Series, params: dict[str, Any], horizon: int,
    n_folds: int, embargo_frac: float, n_estimators: int, early_stopping: int,
) -> int:
    """Mean best-iteration over purged folds inside the training window."""
    kf = PurgedKFold(n_splits=n_folds, horizon=horizon, embargo_frac=embargo_frac)
    bests: list[int] = []
    for tr, va in kf.split(pd.Series(y.index.get_level_values(0))):
        tr_X, tr_y = X.iloc[tr], y.iloc[tr]
        va_X, va_y = X.iloc[va], y.iloc[va]
        m = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
        m.fit(
            tr_X, tr_y,
            eval_X=va_X, eval_y=va_y,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        bests.append(int(m.best_iteration_ or n_estimators))
    return int(np.mean(bests))


@dataclass
class MLArtifact:
    """A frozen, deployable model — text booster + feature spec + fit window."""

    model_text: str
    features: list[str]
    feature_formulas: list[str]
    horizon: int
    fit_window: tuple[str, str]
    params: dict[str, Any]
    best_iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ml_lgbm",
            "features": self.features,
            # formulas rebuild the feature matrix at scoring time — column
            # names alone are not evaluable expressions
            "feature_formulas": self.feature_formulas,
            "horizon": self.horizon,
            "fit_window": list(self.fit_window),
            "params": {k: v for k, v in self.params.items() if k != "seed"},
            "best_iteration": self.best_iteration,
            "metadata": self.metadata,
        }

    def save(self, out_dir: Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        txt = out_dir / f"ml_{stamp}.txt"
        meta = out_dir / f"ml_{stamp}.json"
        # newline="\n" is load-bearing: LightGBM's C model parser ABORTS on CRLF
        # files (Windows write_text defaults to os.linesep) — see tests/test_ml.py.
        txt.write_text(self.model_text, encoding="utf-8", newline="\n")
        meta.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"model": str(txt), "meta": str(meta)}


def load_artifact(model_path: str | Path) -> "lgb.Booster":
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not installed")
    # Parse from the string (newlines normalised) instead of the C file reader:
    # the C parser aborts the process on CRLF model files.
    text = Path(model_path).read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    booster = lgb.Booster(model_str=text)
    booster.num_threads = 1
    return booster


def score_artifact(booster: "lgb.Booster", features: pd.DataFrame) -> pd.Series:
    """Deterministic single-threaded score of a frozen booster on a feature frame."""
    return pd.Series(booster.predict(features.values), index=features.index)


def walk_forward_fit(
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
) -> dict[str, Any]:
    """Full walk-forward fit + report. Returns metrics and the artifact paths."""
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not installed")
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    features = build_feature_matrix(market.long, formulas, n_jobs=n_jobs)
    if extra_features:
        extras = pd.concat(
            [s.rename(k) for k, s in extra_features.items()], axis=1
        )
        features = features.join(extras, how="left")
    # plain float64 — joined extras may carry pd.NA (NAType), which breaks
    # numpy astype and LightGBM alike
    features = features.astype(np.float64)
    features = features.replace([np.inf, -np.inf], np.nan)
    labels = forward_return_labels(market.price_panel, (horizon,))
    y_raw = labels[f"fwd_{horizon}"].rename("fwd_raw")
    labels_std = standardize_per_date(labels)
    tradable = getattr(market, "forward_returns_tradable", None)
    X, y = align_features_labels(features, labels_std, f"fwd_{horizon}", tradable)
    # raw label aligned to X (for cost-aware tail metrics — the model never sees it)
    y_raw = y_raw.reindex(X.index)

    X_tr = _slice(X, *train_window)
    y_tr = y[X_tr.index]
    if len(X_tr) < 200:
        raise ValueError("training window too small for a stable fit")

    best_iter = _purged_best_iterations(
        X_tr, y_tr, p, horizon, n_folds, embargo_frac, n_estimators, early_stopping
    )
    params_fit = dict(p)
    params_fit.pop("seed", None)

    model = lgb.LGBMRegressor(n_estimators=int(best_iter), **p)
    model.fit(X_tr, y_tr)

    X_va = _slice(X, *val_window)
    val_metrics = _window_metrics(model, X_va, y_raw[X_va.index], horizon, cost_bps) if len(X_va) else {}

    # refit on train+val for the final artifact
    X_tv = _slice(X, train_window[0], val_window[1])
    y_tv = y[X_tv.index]
    final = lgb.LGBMRegressor(n_estimators=int(best_iter), **p)
    final.fit(X_tv, y_tv)

    X_te = _slice(X, *test_window)
    test_metrics = _window_metrics(final, X_te, y_raw[X_te.index], horizon, cost_bps) if len(X_te) else {}

    artifact = MLArtifact(
        model_text=final.booster_.model_to_string(),
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
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "extra_features": list(extra_features.keys()) if extra_features else [],
        },
    )
    paths: dict[str, str] = {}
    if out_dir is not None:
        paths = artifact.save(Path(out_dir))

    return {
        "horizon": horizon,
        "n_features": int(len(X.columns)),
        "best_iteration": int(best_iter),
        "train": {"n": int(len(X_tr))},
        "val": val_metrics,
        "test": test_metrics,
        "artifact": paths,
        "feature_formulas": list(formulas),
    }


__all__ = [
    "MLArtifact",
    "build_feature_matrix",
    "load_artifact",
    "score_artifact",
    "walk_forward_fit",
]
