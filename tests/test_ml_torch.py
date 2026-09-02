"""Tests for the GPU deep-model track (src/ml/torch_model.py) — offline.

The tests run on whatever device is available (CPU fallback keeps CI green);
the production training script targets the RTX 4060 explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.ml.torch_model import (  # noqa: E402
    RankMLP,
    TorchAdapter,
    load_torch_artifact,
    train_walk_forward,
)
from src.data.synthetic import make_synthetic_market  # noqa: E402

FORMULAS = [
    "TS_Return(Close, 5)",
    "TS_Std(Close, 20)",
    "TS_Mean(Volume, 20)",
    "Neg(Rank(TS_Std(Close, 60)))",
]


def test_rank_mlp_forward_shape():
    m = RankMLP(n_features=8)
    out = m(torch.zeros(16, 8))
    assert out.shape == (16, 1)


def test_adapter_predict_is_deterministic_and_scales():
    m = RankMLP(n_features=2)
    adapter = TorchAdapter(m, np.array([0.0, 100.0], dtype=np.float32), np.array([1.0, 10.0], dtype=np.float32))
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [100.0, 110.0, 120.0]})
    p1 = adapter.predict(X)
    p2 = adapter.predict(X)
    assert np.allclose(p1, p2, atol=0.0)
    assert p1.shape == (3,)


def test_train_walk_forward_end_to_end(tmp_path):
    mkt = make_synthetic_market(symbols=40, days=504, seed=5, start="2019-01-01")
    result = train_walk_forward(
        mkt,
        FORMULAS,
        horizon=10,
        train_window=("2019-01-01", "2019-09-30"),
        val_window=("2019-10-01", "2019-12-31"),
        test_window=("2020-01-01", "2020-04-30"),
        hyper={"epochs": 3, "batch_size": 256, "patience": 2},
        n_folds=3,
        cost_bps=5.0,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result["n_features"] == len(FORMULAS)
    assert np.isfinite(result["test"]["rank_ic"])
    model_path = Path(result["artifact"]["model"])
    meta_path = Path(result["artifact"]["meta"])
    assert model_path.is_file() and meta_path.is_file()

    # artifact roundtrip: deterministic scoring, feature spec intact
    adapter = load_torch_artifact(model_path, meta_path)
    from src.ml.train import build_feature_matrix

    feats = build_feature_matrix(mkt.long, FORMULAS)
    p1 = adapter.predict(feats)
    p2 = adapter.predict(feats)
    assert np.allclose(p1, p2, atol=0.0)
    assert np.isfinite(p1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
