"""ML track — label construction (multi-horizon forward returns).

Labels are built from the PIT-clean close panel: the label at date ``d`` for
horizon ``h`` is ``close[d+h] / close[d] - 1``. Features only ever use data at or
before ``d`` (guaranteed upstream by the factor context), and the label is the
return the backtest engine actually harvests (its ``forward_returns`` is exactly
the ``h=1`` case). The optional ``tradable`` mask blanks labels whose entry/exit
window crosses a price-limit-locked bar — the LIMIT_DOWN blueprint rule that
portfolio Sharpe must not be computed on returns that cannot be transacted.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def forward_return_labels(
    close_wide: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """(date, symbol) frame of forward returns, one column per horizon.

    ``close_wide`` is the date × symbol close panel. The last ``h`` rows of
    every symbol are NaN (the return window runs past the panel end).
    """
    cols = {}
    for h in horizons:
        fwd = close_wide.shift(-int(h)) / close_wide - 1.0
        # dropna=False keeps the tail rows (NaN labels) so the index is a full
        # (date, symbol) grid — downstream alignment decides what to drop.
        cols[f"fwd_{h}"] = fwd.stack(dropna=False).rename(f"fwd_{h}")
    out = pd.concat(cols.values(), axis=1)
    out.index.names = ["date", "symbol"]
    return out


def standardize_per_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date — neutralises the market level.

    The market-neutral book is built from cross-sectionally ranked scores, so
    training on per-date z-scored labels matches the deployment semantics.
    """
    out = frame.copy()
    for col in out.columns:
        s = out[col].astype(float)
        mean = s.groupby(level=0).transform("mean")
        std = s.groupby(level=0).transform("std").replace(0.0, np.nan)
        out[col] = (s - mean) / (std.fillna(1.0) + 1e-12)
    return out


def align_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon_col: str,
    tradable: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join features and one label column; blank untradeable rows.

    ``tradable`` (optional) is a (date, symbol) 0/1 series from
    :func:`src.backtest.limit_locked.tradeable_forward_returns`; labels where it
    is 0 are dropped so the model never trains on unharvestable returns.
    """
    lab = labels[[horizon_col]].rename(columns={horizon_col: "label"})
    joined = features.join(lab, how="inner")
    # LightGBM handles feature NaN natively — only rows whose LABEL is missing
    # (or untradeable) are dropped. Dropping on all-feature NaN would let one
    # all-NaN feature column erase the whole dataset.
    joined = joined[joined["label"].notna()]
    if tradable is not None:
        joined = joined.join(tradable.rename("tradable"), how="left")
        joined = joined[joined["tradable"] != 0]
        joined = joined.drop(columns=["tradable"])
    return joined.drop(columns=["label"]), joined["label"]


__all__ = [
    "forward_return_labels",
    "standardize_per_date",
    "align_features_labels",
]
