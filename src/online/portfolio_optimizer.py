"""Portfolio optimizer — PCA neutralization + Correlation-Break Diversification.

Review.md (paper §4.4 "Correlation-Break Diversification") warns that correlated
signals collapse into a single concentrated bet, so the online portfolio layer:

1. **PCA neutralization** — per date, residualise each symbol's signal against
   the top-``n_components`` principal components (numpy SVD, no heavy deps).
   This removes the market/regime factor that most signals share, leaving the
   genuinely idiosyncratic part of the score.
2. **Correlation-Break Diversification** — after weights are formed, the
   correlation matrix is inspected; any cluster whose members are too highly
   correlated gets its combined weight re-binned down to a per-cluster cap, so
   no single idea family dominates the book.

Everything here is deterministic and vectorised over numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PCA neutralization (per-date SVD residualization)
# ---------------------------------------------------------------------------


def neutralize_pca(
    signals: pd.Series,
    n_components: int = 5,
    min_symbols: int = 8,
) -> pd.Series:
    """Residualise a signal panel against its own top principal components.

    ``signals`` is indexed by ``(date, symbol)``. For each date with at least
    ``min_symbols`` rows, the cross-section is demeaned, SVD-decomposed, and the
    projection onto the first ``n_components`` eigenvectors is subtracted.
    Dates that are too small pass through demeaned-only.
    """
    out = signals.copy()
    for date, grp in signals.groupby(level=0):
        vals = grp.to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue  # fully NaN date — leave as NaN
        mean = float(np.nanmean(vals))
        centered = vals - mean
        valid = ~np.isnan(vals)
        if int(valid.sum()) < min_symbols or n_components < 1:
            out.loc[grp.index] = centered
            continue
        filled = np.where(np.isnan(vals), 0.0, centered)
        # SVD of the cross-section: u s vh. The leading right singular vector
        # captures the shared "market" factor; residualise it away.
        try:
            _, s, vh = np.linalg.svd(filled, full_matrices=False)
            k = int(min(n_components, len(s), len(vh)))
            if k < 1:
                out.loc[grp.index] = centered
                continue
            # project each row onto the top-k PCs and subtract
            proj = (filled @ vh[:k].T) @ vh[:k]
            resid = centered - proj
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate matrix
            out.loc[grp.index] = centered
            continue
        resid = np.where(np.isnan(vals), np.nan, resid)
        out.loc[grp.index] = resid
    return out


# ---------------------------------------------------------------------------
# Correlation-Break Diversification
# ---------------------------------------------------------------------------


def _corr_clusters(
    weights: pd.Series,
    returns: pd.DataFrame,
    corr_threshold: float = 0.7,
) -> dict[str, list[str]]:
    """Greedy single-link clusters of symbols whose correlation exceeds the bar."""
    symbols = [s for s in weights.index if s in returns.columns]
    if len(symbols) < 2:
        return {}
    corr = returns[symbols].corr().fillna(0.0).to_numpy()
    n = len(symbols)
    clusters: dict[str, list[str]] = {}
    assigned: set[int] = set()
    for i in range(n):
        if i in assigned:
            continue
        members = [symbols[i]]
        assigned.add(i)
        for j in range(i + 1, n):
            if j not in assigned and corr[i, j] >= corr_threshold:
                members.append(symbols[j])
                assigned.add(j)
        if len(members) > 1:
            clusters[f"c{len(clusters)}"] = members
    return clusters


def break_correlations(
    weights: pd.Series,
    returns: pd.DataFrame,
    *,
    corr_threshold: float = 0.7,
    cluster_cap: float = 0.15,
) -> pd.Series:
    """Cap the combined weight of any highly-correlated cluster.

    Cluster members keep their relative proportions but are scaled so their
    total weight does not exceed ``cluster_cap``. Freed weight is redistributed
    proportionally to the symbols that were not clipped.
    """
    w = weights.fillna(0.0).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()  # normalise to 1 before clipping
    clusters = _corr_clusters(w, returns, corr_threshold)
    clipped_total = 0.0
    for members in clusters.values():
        total = float(w.loc[members].sum())
        if total > cluster_cap and total > 0:
            scale = cluster_cap / total
            w.loc[members] = w.loc[members] * scale
            clipped_total += total - cluster_cap
    if clipped_total > 0:
        # give freed weight back to symbols that were not clipped
        clipped = {s for members in clusters.values() for s in members}
        free = [s for s in w.index if s not in clipped]
        free_sum = float(w.loc[free].sum()) if free else 0.0
        if free and free_sum > 0:
            w.loc[free] = w.loc[free] + w.loc[free] / free_sum * clipped_total
        elif free:
            for s in free:
                w.loc[s] = clipped_total / len(free)
    return w


@dataclass
class OptimizationResult:
    weights: pd.Series
    neutralized_scores: pd.Series
    clusters: dict[str, list[str]]


class PortfolioOptimizer:
    """Turn a raw score panel into disciplined, diversified target weights.

    Pipeline per rebalance date: cross-sectional rank -> neutralise by PCA ->
    long/short selection -> correlation-break diversification -> position caps.
    """

    def __init__(
        self,
        *,
        long_pct: float = 0.10,
        short_pct: float = 0.10,
        n_components: int = 5,
        corr_threshold: float = 0.7,
        cluster_cap: float = 0.15,
        max_position_pct: float = 0.05,
        returns_panel: Optional[pd.DataFrame] = None,
    ) -> None:
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.n_components = n_components
        self.corr_threshold = corr_threshold
        self.cluster_cap = cluster_cap
        self.max_position_pct = max_position_pct
        self.returns_panel = returns_panel  # (symbol x date) or (date, symbol)

    def optimize(self, scores: pd.Series) -> OptimizationResult:
        """One full optimisation pass over the whole panel."""
        neutral = neutralize_pca(scores, n_components=self.n_components)
        # cross-sectional percentile rank of the neutral scores
        rank = neutral.groupby(level=0).rank(pct=True)
        dates = sorted(neutral.index.get_level_values(0).unique())
        symbols = sorted(neutral.index.get_level_values(1).unique())

        all_weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        clusters: dict[str, list[str]] = {}
        for date in dates:
            row = rank.loc[date].dropna()
            if len(row) < 4:
                continue
            n_long = max(1, int(np.ceil(self.long_pct * len(row))))
            n_short = max(1, int(np.ceil(self.short_pct * len(row))))
            long = row.nlargest(n_long).index
            short = row.nsmallest(n_short).index
            w = pd.Series(0.0, index=row.index)
            w[long] = 1.0 / max(n_long, 1)
            w[short] = -1.0 / max(n_short, 1)
            # correlation-break diversification (requires a returns panel)
            ret_df = self.returns_panel
            if ret_df is not None:
                ret = _as_cross(ret_df)
                w = break_correlations(
                    w, ret, corr_threshold=self.corr_threshold, cluster_cap=self.cluster_cap
                )
                clusters.update(_corr_clusters(w, ret, self.corr_threshold))
            # gross leverage & position cap
            gross = float(w.abs().sum())
            if gross > 0:
                w = w / gross * (self.long_pct + self.short_pct) if gross > 1 else w
                over = w.abs() > self.max_position_pct
                if over.any():
                    scale = self.max_position_pct / w[over].abs()
                    w[over] = w[over] * scale
            all_weights.loc[date, w.index] = w

        return OptimizationResult(
            weights=all_weights,
            neutralized_scores=neutral,
            clusters=clusters,
        )


def _as_cross(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalise a returns panel to ``symbol x date`` (rows are symbols)."""
    if isinstance(panel.index, pd.MultiIndex):
        return panel.unstack(level=1)
    return panel


__all__ = [
    "neutralize_pca",
    "break_correlations",
    "PortfolioOptimizer",
    "OptimizationResult",
]
