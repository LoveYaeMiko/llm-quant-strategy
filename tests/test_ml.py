"""Tests for the ML track (src/ml) — fully offline, synthetic data.

Covered:
* forward-return labels (PIT alignment, horizon correctness);
* per-date cross-sectional standardization;
* purged K-fold: no train label window overlaps any test label window;
* embargo widens the purge margin;
* end-to-end walk-forward fit on synthetic data + deterministic frozen artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml import (  # noqa: E402
    PurgedKFold,
    align_features_labels,
    forward_return_labels,
    score_artifact,
    standardize_per_date,
    walk_forward_fit,
)
from src.data.synthetic import make_synthetic_market  # noqa: E402

lgb = pytest.importorskip("lightgbm")


def _panel() -> pd.DataFrame:
    mkt = make_synthetic_market(symbols=12, days=260, seed=3)
    return mkt


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def test_forward_return_labels_horizon_and_pit():
    mkt = _panel()
    closes = mkt.price_panel
    labels = forward_return_labels(closes, horizons=(1, 5))
    assert labels.index.names == ["date", "symbol"]
    # h=1 equals the synthetic forward_returns panel (same (date, symbol) order)
    fwd1 = labels["fwd_1"].reindex(mkt.forward_returns.index)
    assert np.allclose(fwd1, mkt.forward_returns, equal_nan=True)
    # last h rows of every symbol are NaN
    last_date = closes.index.max()
    assert labels.loc[last_date, "fwd_5"].isna().all()
    # h=5: label at date d is close[d+5]/close[d]-1 for a known symbol
    sym = closes.columns[0]
    d = closes.index[10]
    expect = closes[sym].iloc[15] / closes[sym].iloc[10] - 1.0
    got = labels.loc[(d, sym), "fwd_5"]
    assert abs(got - expect) < 1e-12


def test_standardize_per_date():
    mkt = _panel()
    labels = standardize_per_date(forward_return_labels(mkt.price_panel, (1,)))
    s = labels["fwd_1"].dropna()
    means = s.groupby(level=0).mean()
    stds = s.groupby(level=0).std()
    assert np.allclose(means, 0.0, atol=1e-10)
    assert np.allclose(stds, 1.0, atol=1e-10)


def test_align_features_labels_tradable_mask():
    mkt = _panel()
    labels = forward_return_labels(mkt.price_panel, (1,))
    feats = labels[["fwd_1"]].rename(columns={"fwd_1": "f"})
    tradable = pd.Series(1.0, index=labels.index)
    tradable.iloc[::7] = 0.0  # blank every 7th row
    X, y = align_features_labels(feats, labels, "fwd_1", tradable)
    assert set(X.index).issubset(tradable[tradable == 1.0].index)
    assert len(X) < len(labels)


def test_align_keeps_rows_with_nan_features():
    """One all-NaN feature column must not erase the whole dataset (LightGBM
    handles NaN natively) — only missing LABELS are dropped."""
    mkt = _panel()
    labels = forward_return_labels(mkt.price_panel, (1,))
    feats = pd.DataFrame({"good": labels["fwd_1"], "bad": np.nan}, index=labels.index)
    X, y = align_features_labels(feats, labels, "fwd_1")
    assert len(X) == labels["fwd_1"].notna().sum()
    assert X["bad"].isna().all()
    assert y.notna().all()


# --------------------------------------------------------------------------- #
# purged k-fold
# --------------------------------------------------------------------------- #


def _sample_dates(n_dates: int, n_symbols: int = 3) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_symbols)]])
    return pd.Series(idx.get_level_values(0).values, index=range(len(idx)))


def test_purged_kfold_no_label_overlap():
    horizon = 10
    dates = _sample_dates(120)
    kf = PurgedKFold(n_splits=5, horizon=horizon, embargo_frac=0.0)
    uniq, pos = np.unique(dates.values, return_inverse=True)
    for tr, te in kf.split(dates):
        test_pos = set(pos[te])
        # every train label window [p, p+horizon] must be disjoint from
        # every test label window [q, q+horizon]
        for p in pos[tr]:
            for q in test_pos:
                assert (p + horizon < q) or (p > q + horizon), (
                    f"overlap: train label at {p} vs test label at {q}"
                )
        # test rows form contiguous date blocks
        assert test_pos == set(range(min(test_pos), max(test_pos) + 1))


def test_purged_kfold_embargo_widens_margin():
    horizon = 10
    dates = _sample_dates(200)
    kf0 = PurgedKFold(n_splits=4, horizon=horizon, embargo_frac=0.0)
    kf1 = PurgedKFold(n_splits=4, horizon=horizon, embargo_frac=0.05)
    for (tr0, _), (tr1, _) in zip(kf0.split(dates), kf1.split(dates)):
        assert len(tr1) <= len(tr0)  # embargo removes strictly more train rows


def test_purged_kfold_rejects_too_few_dates():
    with pytest.raises(ValueError):
        list(PurgedKFold(n_splits=5, horizon=10).split(_sample_dates(4)))


# --------------------------------------------------------------------------- #
# end-to-end walk-forward fit + artifact determinism
# --------------------------------------------------------------------------- #


FORMULAS = [
    "TS_Return(Close, 5)",
    "TS_Std(Close, 20)",
    "TS_Mean(Volume, 20)",
    "Neg(Rank(TS_Std(Close, 60)))",
]


def test_walk_forward_fit_end_to_end(tmp_path):
    mkt = make_synthetic_market(symbols=40, days=504, seed=5, start="2019-01-01")
    # synthetic window split: ~60/20/20
    result = walk_forward_fit(
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
    )
    assert result["n_features"] == len(FORMULAS)
    assert result["best_iteration"] >= 1
    assert result["test"]["n_days"] > 0
    assert np.isfinite(result["test"]["rank_ic"])
    assert np.isfinite(result["test"]["ls_ann_net"])

    # artifact files exist and roundtrip deterministically
    model_path = Path(result["artifact"]["model"])
    meta_path = Path(result["artifact"]["meta"])
    assert model_path.is_file() and meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "ml_lgbm"
    assert len(meta["features"]) == len(FORMULAS)
    assert meta["fit_window"] == ["2019-01-01", "2019-12-31"]

    from src.ml.train import build_feature_matrix, load_artifact

    booster = load_artifact(model_path)
    feats = build_feature_matrix(mkt.long, FORMULAS)
    s1 = score_artifact(booster, feats)
    s2 = score_artifact(booster, feats)
    assert np.allclose(s1, s2, atol=0.0)
    assert s1.index.equals(feats.index)
    assert np.isfinite(s1).all()


def test_walk_forward_fit_with_nan_extra_features(tmp_path):
    """Extra features with NaN (margin factors are sparse) must not break
    numpy/LightGBM — the NAType regression guard."""
    mkt = make_synthetic_market(symbols=40, days=504, seed=5, start="2019-01-01")
    extra = pd.Series(
        np.random.default_rng(3).normal(size=len(mkt.long)),
        index=mkt.long.index,
    )
    extra.iloc[::3] = np.nan  # sparse like real margin crowding factors
    result = walk_forward_fit(
        mkt,
        FORMULAS,
        horizon=10,
        train_window=("2019-01-01", "2019-09-30"),
        val_window=("2019-10-01", "2019-12-31"),
        test_window=("2020-01-01", "2020-04-30"),
        n_estimators=30,
        early_stopping=10,
        n_folds=3,
        cost_bps=5.0,
        out_dir=tmp_path,
        extra_features={"x_margin_fin_growth": extra},
    )
    assert result["n_features"] == len(FORMULAS) + 1
    assert np.isfinite(result["test"]["rank_ic"])
