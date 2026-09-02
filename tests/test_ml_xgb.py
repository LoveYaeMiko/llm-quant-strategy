"""Tests for the XGBoost GPU track (src/ml/xgb_model.py) — offline, CPU device."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from src.ml.xgb_model import (  # noqa: E402
    XGBAdapter,
    load_xgb_artifact,
    walk_forward_fit_xgb,
)
from src.data.synthetic import make_synthetic_market  # noqa: E402

FORMULAS = [
    "TS_Return(Close, 5)",
    "TS_Std(Close, 20)",
    "TS_Mean(Volume, 20)",
    "Neg(Rank(TS_Std(Close, 60)))",
]


def test_walk_forward_fit_xgb_end_to_end(tmp_path):
    mkt = make_synthetic_market(symbols=40, days=504, seed=5, start="2019-01-01")
    result = walk_forward_fit_xgb(
        mkt,
        FORMULAS,
        horizon=10,
        train_window=("2019-01-01", "2019-09-30"),
        val_window=("2019-10-01", "2019-12-31"),
        test_window=("2020-01-01", "2020-04-30"),
        n_estimators=60,
        early_stopping=10,
        n_folds=3,
        cost_bps=5.0,
        out_dir=tmp_path,
        device="cpu",
    )
    assert result["n_features"] == len(FORMULAS)
    assert np.isfinite(result["test"]["rank_ic"])
    model_path = Path(result["artifact"]["model"])
    meta_path = Path(result["artifact"]["meta"])
    assert model_path.is_file() and meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "ml_xgb"
    assert meta["feature_formulas"] == FORMULAS

    # artifact roundtrip: deterministic scoring
    booster = load_xgb_artifact(model_path)
    adapter = XGBAdapter(booster)
    from src.ml.train import build_feature_matrix

    feats = build_feature_matrix(mkt.long, FORMULAS)
    p1 = adapter.predict(feats)
    p2 = adapter.predict(feats)
    assert np.allclose(p1, p2, atol=0.0)
    assert np.isfinite(p1).all()


def test_walk_forward_fit_xgb_parallel_features(tmp_path):
    """The multiprocessing feature path must produce identical columns."""
    mkt = make_synthetic_market(symbols=40, days=504, seed=5, start="2019-01-01")
    from src.ml.train import build_feature_matrix

    formulas = FORMULAS + ["TS_Return(Close, 10)", "TS_Max(High, 20)"]  # > 4 to hit the pool path
    seq = build_feature_matrix(mkt.long, formulas, n_jobs=None)
    par = build_feature_matrix(mkt.long, formulas, n_jobs=2)
    assert list(seq.columns) == list(par.columns)
    assert np.allclose(seq.values, par.values, equal_nan=True)


def test_dmatrix_clips_inf_and_huge_values():
    from src.ml.xgb_model import _dmatrix

    X = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf, 1e308],
            "b": [np.nan, 2.0, 3.0, 4.0],
        }
    )
    dmat = _dmatrix(X)
    assert dmat.num_row() == 4 and dmat.num_col() == 2  # no exception = clipped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
