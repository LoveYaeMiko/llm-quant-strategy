"""Tests for the ML book bridge (src/paper/ml_book.py) — offline, synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.ml import walk_forward_fit
from src.paper.ml_book import MLBookPortfolio, _resolve_artifact

FORMULAS = [
    "TS_Return(Close, 5)",
    "TS_Std(Close, 20)",
    "TS_Mean(Volume, 20)",
    "Neg(Rank(TS_Std(Close, 60)))",
]


def _train_artifact(tmp_path) -> tuple[str, str]:
    mkt = make_synthetic_market(symbols=24, days=504, seed=5, start="2019-01-01")
    result = walk_forward_fit(
        mkt, FORMULAS, horizon=10,
        train_window=("2019-01-01", "2019-09-30"),
        val_window=("2019-10-01", "2019-12-31"),
        test_window=("2020-01-01", "2020-04-30"),
        n_estimators=30, early_stopping=10, n_folds=3,
        out_dir=tmp_path,
    )
    return result["artifact"]["model"], result["artifact"]["meta"]


def test_ml_book_portfolio_builds_books(tmp_path):
    # 120 names: the 5% per-name cap does not bind at production scale
    # (per-name weight 1/(2*0.1*N) ≈ 4.2% < 5%)
    mkt = make_synthetic_market(symbols=120, days=504, seed=5, start="2019-01-01")
    _train_artifact(tmp_path)
    # point the module's artifact dir at the tmp dir
    import src.paper.ml_book as mb

    mb._ARTIFACT_DIR = Path(tmp_path)
    book = MLBookPortfolio(
        mkt, ["lgbm"], long_pct=0.10, short_pct=0.10, ensemble=False, cfg=None
    )
    d = pd.Timestamp("2020-01-02")
    w = book.compute_weights(list(mkt.price_panel.columns), d)
    assert isinstance(w, dict)
    longs = sum(1 for v in w.values() if v > 0)
    shorts = sum(1 for v in w.values() if v < 0)
    assert longs >= 1  # top decile long book
    assert shorts >= 1  # bottom decile short book
    # gross-normalised to ~1 (the 5% per-name cap may trim after a regime
    # short-scale — same construction as the validated showdown books)
    gross = sum(abs(v) for v in w.values())
    assert 0.9 <= gross <= 1.0 + 1e-6


def test_resolve_artifact_newest_of_kind(tmp_path):
    import src.paper.ml_book as mb

    mb._ARTIFACT_DIR = Path(tmp_path)
    _train_artifact(tmp_path)
    meta_path, model_path = _resolve_artifact("lgbm", "")
    assert meta_path.is_file() and model_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "ml_lgbm"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
