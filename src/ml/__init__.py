"""ML track — offline LightGBM training pipeline for the factor zoo.

Modules:
* :mod:`src.ml.labels` — multi-horizon forward-return labels + per-date z-score;
* :mod:`src.ml.cv` — purged K-fold with embargo (label-overlap safe);
* :mod:`src.ml.train` — walk-forward trainer + frozen deterministic artifact.
"""

from .cv import PurgedKFold
from .labels import (
    align_features_labels,
    forward_return_labels,
    standardize_per_date,
)
from .train import (
    MLArtifact,
    build_feature_matrix,
    load_artifact,
    score_artifact,
    walk_forward_fit,
)

__all__ = [
    "PurgedKFold",
    "align_features_labels",
    "forward_return_labels",
    "standardize_per_date",
    "MLArtifact",
    "build_feature_matrix",
    "load_artifact",
    "score_artifact",
    "walk_forward_fit",
]
