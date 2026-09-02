"""Model ensemble — per-date cross-sectional rank averaging.

Single models disagree on the mid-book; averaging their per-date cross-
sectional ranks typically raises IC and the tradable tail spread (the same
reason the MLP beat the GBDT — different inductive biases). The ensemble
scores feed the SAME book construction as any single model, so the showdown
stays a clean comparison.

Deterministic: rank-average of fixed artifact scorers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def cross_sectional_rank(scores: pd.Series) -> pd.Series:
    """Per-date percentile rank of a (date, symbol) score series."""
    return scores.groupby(level=0).rank(pct=True)


def rank_ensemble(scores: Iterable[pd.Series]) -> pd.Series:
    """Average the per-date cross-sectional ranks of several score series.

    All series must share the (date, symbol) index; NaN scores are dropped
    per model before ranking (a model with no opinion on a day/symbol is
    ignored by the others).
    """
    ranked = []
    for s in scores:
        clean = s.dropna()
        if not len(clean):
            continue
        ranked.append(cross_sectional_rank(clean))
    if not ranked:
        raise ValueError("no non-empty score series to ensemble")
    out = pd.concat(ranked, axis=1)
    return out.mean(axis=1).rename("ensemble")


def ensemble_scores(named_scores: Sequence[tuple[str, pd.Series]]) -> tuple[pd.Series, list[str]]:
    """``(ensemble_series, used_model_names)`` — the dropped models are reported."""
    used = [n for n, s in named_scores if len(s.dropna())]
    return rank_ensemble([s for n, s in named_scores if n in used]), used


__all__ = ["cross_sectional_rank", "rank_ensemble", "ensemble_scores"]
