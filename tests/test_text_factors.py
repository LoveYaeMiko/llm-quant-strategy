"""PHASE9_BERT_BLUEPRINT — text dispersion/novelty factor tests.

Covers the deterministic vector math, the PIT window slice, the NaN guards
(min-articles / min-history), and the cross-sectional percentile rank panel —
all against a small synthetic embedding store in a tmp dir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.text_factors import TextFactorCalculator, run_text_gate


def _write_cache(tmp_path, rows: list[dict]) -> str:
    """Write a small embedding store and return the cache filename."""
    df = pd.DataFrame(rows)
    df["title_embedding"] = [np.asarray(v, dtype=np.float64) for v in df["title_embedding"]]
    cache = "tiny.parquet"
    df.to_parquet(tmp_path / cache, index=False)
    return cache


def _calc(tmp_path, rows, **kw) -> TextFactorCalculator:
    cache = _write_cache(tmp_path, rows)
    return TextFactorCalculator(data_dir=str(tmp_path), cache_file=cache, **kw)


# ---------------------------------------------------------------------------
# pairwise cosine distance math
# ---------------------------------------------------------------------------


def test_mean_pairwise_distance_identical_vectors():
    # all same direction -> pairwise cosine distance 0
    emb = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert TextFactorCalculator._mean_pairwise_distance(emb) == pytest.approx(0.0, abs=1e-9)


def test_mean_pairwise_distance_orthogonal():
    # three axes: pairwise distance between orthogonal unit vectors = 1
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert TextFactorCalculator._mean_pairwise_distance(emb) == pytest.approx(1.0, abs=1e-9)


def test_mean_pairwise_distance_single_row_is_nan():
    emb = np.array([[1.0, 0.0]])
    assert np.isnan(TextFactorCalculator._mean_pairwise_distance(emb))


# ---------------------------------------------------------------------------
# dispersion
# ---------------------------------------------------------------------------


def test_dispersion_nan_below_min_articles(tmp_path):
    # only 2 rows in the window, min_dispersion_articles=3 -> NaN
    calc = _calc(
        tmp_path,
        [
            {"symbol": "A", "date": "2022-01-10", "title": "t1", "title_embedding": [1.0, 0.0]},
            {"symbol": "A", "date": "2022-01-11", "title": "t2", "title_embedding": [0.0, 1.0]},
        ],
        min_dispersion_articles=3,
    )
    assert np.isnan(calc.dispersion("A", "2022-01-12", window=20))


def test_dispersion_window_slices_by_date(tmp_path):
    # rows: one on 2022-01-10, one on 2022-03-20 (outside a 20-day window)
    rows = [
        {"symbol": "A", "date": "2022-01-10", "title": "t1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-20", "title": "t2", "title_embedding": [0.0, 1.0]},
        {"symbol": "A", "date": "2022-03-20", "title": "t3", "title_embedding": [0.0, 1.0]},
    ]
    calc = _calc(tmp_path, rows, min_dispersion_articles=2)
    # as_of 2022-01-25, window 20 -> only the two January rows count -> orthogonal -> 1.0
    assert calc.dispersion("A", "2022-01-25", window=20) == pytest.approx(1.0, abs=1e-9)
    # as_of 2022-04-01, window 20 -> only the March row -> NaN (< 2 rows)
    assert np.isnan(calc.dispersion("A", "2022-04-01", window=20))


def test_dispersion_unknown_symbol_is_nan(tmp_path):
    calc = _calc(tmp_path, [
        {"symbol": "A", "date": "2022-01-10", "title": "t1", "title_embedding": [1.0, 0.0]},
    ], min_dispersion_articles=1)
    assert np.isnan(calc.dispersion("ZZZ", "2022-01-15", window=20))


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------


def test_novelty_latest_vs_history_centroid(tmp_path):
    # history: two rows on the +x axis; latest on +y -> distance 1.0
    rows = [
        {"symbol": "A", "date": "2022-01-01", "title": "h1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-05", "title": "h2", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-10", "title": "new", "title_embedding": [0.0, 1.0]},
    ]
    calc = _calc(tmp_path, rows, min_novelty_history=2)
    assert calc.novelty("A", "2022-01-12", window=180) == pytest.approx(1.0, abs=1e-9)


def test_novelty_latest_similar_to_history(tmp_path):
    # latest agrees with history centroid -> distance ~0
    rows = [
        {"symbol": "A", "date": "2022-01-01", "title": "h1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-05", "title": "h2", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-10", "title": "new", "title_embedding": [1.0, 0.0]},
    ]
    calc = _calc(tmp_path, rows, min_novelty_history=2)
    assert calc.novelty("A", "2022-01-12", window=180) == pytest.approx(0.0, abs=1e-6)


def test_novelty_nan_without_latest_or_history(tmp_path):
    # no report on/before as_of
    calc = _calc(tmp_path, [
        {"symbol": "A", "date": "2022-02-01", "title": "t", "title_embedding": [1.0, 0.0]},
    ], min_novelty_history=1)
    assert np.isnan(calc.novelty("A", "2022-01-01", window=180))  # latest is future -> NaN
    # history too thin
    calc2 = _calc(tmp_path, [
        {"symbol": "A", "date": "2022-01-01", "title": "h", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-05", "title": "new", "title_embedding": [0.0, 1.0]},
    ], min_novelty_history=3)
    assert np.isnan(calc2.novelty("A", "2022-01-06", window=180))


# ---------------------------------------------------------------------------
# cross-sectional snapshot + panel
# ---------------------------------------------------------------------------


def test_factor_snapshot_cross_section(tmp_path):
    # A: no signal (needs 2 rows), B: two identical + latest same -> dispersion ~0
    rows = [
        {"symbol": "A", "date": "2022-01-01", "title": "a1", "title_embedding": [1.0, 0.0]},
        {"symbol": "B", "date": "2022-01-01", "title": "b1", "title_embedding": [1.0, 0.0]},
        {"symbol": "B", "date": "2022-01-05", "title": "b2", "title_embedding": [1.0, 0.0]},
    ]
    calc = _calc(tmp_path, rows, min_dispersion_articles=2)
    snap = calc.factor_snapshot(["A", "B"], "2022-01-10", "dispersion", window=20)
    assert np.isnan(snap["A"])
    assert snap["B"] == pytest.approx(0.0, abs=1e-9)


def test_score_panel_percentile_rank(tmp_path):
    # two symbols, one date: B has higher dispersion -> rank 1.0
    rows = [
        # A: two orthogonal reports -> dispersion 1.0
        {"symbol": "A", "date": "2022-01-01", "title": "a1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A", "date": "2022-01-02", "title": "a2", "title_embedding": [0.0, 1.0]},
        # B: two identical reports -> dispersion 0.0
        {"symbol": "B", "date": "2022-01-01", "title": "b1", "title_embedding": [1.0, 0.0]},
        {"symbol": "B", "date": "2022-01-02", "title": "b2", "title_embedding": [1.0, 0.0]},
    ]
    calc = _calc(tmp_path, rows, min_dispersion_articles=2)
    panel = calc.score_panel(["2022-01-10"], ["A", "B"], "dispersion", window=20)
    assert len(panel) == 2
    by_sym = {s: float(panel.xs(s, level="symbol").iloc[0]) for s in ("A", "B")}
    assert by_sym["A"] == pytest.approx(1.0)
    assert by_sym["B"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# run_text_gate against a tiny market
# ---------------------------------------------------------------------------


def test_run_text_gate_returns_verdict(tmp_path):
    from src.data.synthetic import make_synthetic_market

    # make_synthetic_market produces symbols A01, A02, ... A40
    _calc(tmp_path, [
        {"symbol": "A01", "date": "2022-01-01", "title": "a1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A01", "date": "2022-01-02", "title": "a2", "title_embedding": [0.0, 1.0]},
        {"symbol": "A02", "date": "2022-01-01", "title": "b1", "title_embedding": [1.0, 0.0]},
        {"symbol": "A02", "date": "2022-01-02", "title": "b2", "title_embedding": [1.0, 0.0]},
    ], min_dispersion_articles=2)
    market = make_synthetic_market(seed=1)
    res = run_text_gate(market, ["A01", "A02"], "dispersion", window=20,
                        cache_file="tiny.parquet", data_dir=str(tmp_path))
    assert set(res) >= {"kind", "window_days", "metrics", "portfolio", "gate", "n_signal_cells"}
    assert res["kind"] == "dispersion"
    assert res["window_days"] == 20
    assert res["gate"]["ic_threshold"] == 0.015
    assert isinstance(res["gate"]["passed"], bool)
