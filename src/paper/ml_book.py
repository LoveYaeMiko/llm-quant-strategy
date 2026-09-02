"""ML model book for the shadow loop — artifacts → daily decile long-short books.

The bridge between the trained artifacts (LightGBM / XGBoost / MLP) and the
paper-trading loop: score the universe with the chosen artifact(s), optionally
rank-ensemble several, then build the same market-neutral book construction the
showdown validated (top/bottom decile + regime-adaptive short leg, NO
post-hoc neutralization — the models internalised those exposures at train
time).

PIT discipline: artifact features are causal operator evaluations; the trend
gate compounds only realised returns (``_market_trend``); books are keyed by
date so the runner's daily clock never sees a future weight.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..ml import build_feature_matrix, load_artifact, score_artifact
from ..ml.ensemble import rank_ensemble
from ..ml.torch_model import load_torch_artifact
from ..portfolio.alpha_core import _market_trend, long_book_weights
from ..data.margin import load_margin_extras_from_store

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache" / "ml_book_features"
# Tail builds keep this many bars of history so rolling windows (≤252 + margin)
# on the appended bars match a full rebuild.
_TAIL_CONTEXT_BARS = 320


def _feature_frame(market, meta: dict, cfg, n_jobs: Optional[int]) -> pd.DataFrame:
    """Artifact feature frame, cached on disk and tail-appended on daily runs.

    The frame covers the market slice from its first bar to the latest bar.
    A re-run appends only the new tail bars (rolling windows get
    ``_TAIL_CONTEXT_BARS`` of history), so the deployed shadow loop rebuilds the
    full matrix once per (artifact, universe) and afterwards pays ~minutes.
    Cache identity = formulas + extras + symbols + slice start date.
    """
    extras = meta.get("metadata", {}).get("extra_features") or []
    symbols = list(market.price_panel.columns)
    first_date = market.long.index.get_level_values(0).min()
    blob = json.dumps([meta["feature_formulas"], extras, sorted(symbols), str(first_date)], sort_keys=True)
    key = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:24]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{key}.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        c_dates = cached.index.get_level_values(0).unique().sort_values()
        all_dates = market.long.index.get_level_values(0).unique().sort_values()
        new_dates = all_dates[all_dates > c_dates.max()]
        if len(new_dates) == 0:
            return cached
        pos = int(all_dates.searchsorted(c_dates.max()))
        context_start = all_dates[max(0, pos - _TAIL_CONTEXT_BARS)]
        tail_panel = market.long[market.long.index.get_level_values(0) >= context_start]
        tail = build_feature_matrix(tail_panel, meta["feature_formulas"], n_jobs=n_jobs)
        tail = tail[tail.index.get_level_values(0) > c_dates.max()]
        if extras:
            margin = load_margin_extras_from_store(cfg)
            assert set(extras) <= set(margin), f"missing extras {set(extras) - set(margin)}"
            tail = tail.join(pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left")
        frame = pd.concat([cached, tail]).sort_index()
    else:
        frame = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=n_jobs)
        if extras:
            margin = load_margin_extras_from_store(cfg)
            assert set(extras) <= set(margin), f"missing extras {set(extras) - set(margin)}"
            frame = frame.join(pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left")

    frame.to_parquet(cache_path)
    return frame


class MLBookPortfolio:
    """Daily decile books from one or more ML artifacts (+ optional ensemble)."""

    def __init__(
        self,
        market,
        artifacts: list[str],
        *,
        long_pct: float = 0.10,
        short_pct: float = 0.10,
        max_position_pct: float = 0.05,
        trend_days: int = 60,
        trend_gate: float = 0.03,
        short_scale: float = 0.5,
        ensemble: bool = False,
        cfg=None,
        n_jobs: Optional[int] = None,
    ) -> None:
        scores: list[pd.Series] = []
        feature_cache: dict[tuple, pd.DataFrame] = {}
        for spec in artifacts:
            kind, stem = (spec.split(":", 1) + [""])[:2]
            meta_path, model_path = _resolve_artifact(kind, stem)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            key = (
                tuple(meta["feature_formulas"]),
                tuple(meta.get("metadata", {}).get("extra_features") or []),
            )
            if key not in feature_cache:
                frame = _feature_frame(market, meta, cfg, n_jobs=n_jobs)
                feature_cache[key] = frame
            frame = feature_cache[key]
            assert list(frame.columns) == meta["features"], "artifact columns out of sync"
            kind_name = meta.get("kind", "ml_lgbm")
            if kind_name == "ml_mlp_torch":
                adapter = load_torch_artifact(model_path, meta_path)
                s = pd.Series(adapter.predict(frame), index=frame.index)
            elif kind_name == "ml_xgb":
                from ..ml.xgb_model import XGBAdapter, load_xgb_artifact

                adapter = XGBAdapter(load_xgb_artifact(model_path))
                s = pd.Series(adapter.predict(frame), index=frame.index)
            else:
                booster = load_artifact(model_path)
                s = score_artifact(booster, frame)
            scores.append(s.rename(meta_path.name))
        if ensemble and len(scores) > 1:
            scores = [rank_ensemble(scores).rename("RANK_ENSEMBLE")]
        if not scores:
            raise ValueError("no artifacts scored")
        composite = rank_ensemble(scores) if len(scores) > 1 else scores[0]

        close_wide = market.price_panel
        trend = _market_trend(close_wide, trend_days)
        self._books: dict[pd.Timestamp, dict[str, float]] = {}
        for d, day in composite.dropna().groupby(level=0):
            ss = 1.0
            if pd.Timestamp(d) in trend.index:
                t = trend[pd.Timestamp(d)]
                if np.isfinite(t) and t > trend_gate:
                    ss = short_scale
            self._books[pd.Timestamp(d)] = long_book_weights(
                day, long_pct=long_pct, short_pct=short_pct,
                max_position_pct=max_position_pct, short_scale=ss,
            )

    def compute_weights(self, symbols, date) -> dict[str, float]:
        w = self._books.get(pd.Timestamp(date), {})
        if symbols is None:
            return dict(w)
        keep = set(symbols)
        return {s: v for s, v in w.items() if s in keep}


def _resolve_artifact(kind: str, stem: str) -> tuple[Path, Path]:
    """Resolve an artifact spec ``kind[:stem]`` to (meta, model) paths.

    * ``stem`` given → exact match (any kind whose meta file name contains it);
    * empty stem → the NEWEST artifact of ``kind`` (lgbm / torch / xgb).
    """
    if stem:
        metas = sorted(_ARTIFACT_DIR.glob(f"*{stem}*.json"))
        if not metas:
            raise SystemExit(f"artifact spec {kind!r}:{stem!r} not found under {_ARTIFACT_DIR}")
        meta_path = metas[-1]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        k = meta.get("kind", "ml_lgbm")
        model_path = _model_for_kind(meta_path, k)
        return meta_path, model_path

    patterns = {
        "lgbm": "ml_*.json",
        "torch": "mlp_*.json",
        "xgb": "xgb_*.meta.json",
    }
    pattern = patterns.get(kind)
    if pattern is None:
        raise SystemExit(f"unknown artifact kind {kind!r} (lgbm/torch/xgb)")
    metas = sorted(_ARTIFACT_DIR.glob(pattern))
    if not metas:
        raise SystemExit(f"no artifacts of kind {kind!r} under {_ARTIFACT_DIR}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta_path, _model_for_kind(meta_path, meta.get("kind", "ml_lgbm"))


def _model_for_kind(meta_path: Path, kind: str) -> Path:
    if kind == "ml_mlp_torch":
        return meta_path.with_suffix(".pt")
    if kind == "ml_xgb":
        return meta_path.with_name(meta_path.name.replace(".meta.json", ".json"))
    return meta_path.with_suffix(".txt")


__all__ = ["MLBookPortfolio"]
