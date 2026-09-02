"""ML track — purged cross-validation with embargo (AFML §7.4).

Daily cross-sectional samples whose labels are ``h``-day forward returns
overlap heavily: a sample taken 3 days before a test sample shares 17 of its
20 label days. A naive split leaks test-label information into training and
inflates validation IC. :class:`PurgedKFold` splits by date blocks and purges
every training sample whose label window intersects any test label window,
plus an embargo margin (a fraction of the total span) on each side.

Positions work in *trading-day* units on the sorted unique dates, so ``horizon``
is exact regardless of calendar gaps.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd


class PurgedKFold:
    """Date-block K-fold with label-window purging + embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 10,
        embargo_frac: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.0 <= embargo_frac <= 0.1:
            raise ValueError("embargo_frac must be in [0, 0.1]")
        self.n_splits = int(n_splits)
        self.horizon = int(horizon)
        self.embargo_frac = float(embargo_frac)

    def split(
        self,
        dates: pd.Series,
        X: Optional[pd.DataFrame] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` row-position arrays.

        ``dates`` is a Series aligned to the sample rows (a label-start date per
        sample). Folds are contiguous blocks of the sorted unique dates; a train
        row survives only if its label window ``[p, p+h]`` is disjoint from
        ``[a-h-e, b+h+e]`` — the test label span extended by the embargo ``e``.
        """
        dates = pd.Series(dates).reset_index(drop=True)
        uniq, positions = np.unique(dates.values, return_inverse=True)
        n = len(uniq)
        if n < self.n_splits * 2:
            raise ValueError(
                f"not enough unique dates ({n}) for {self.n_splits} folds"
            )
        bounds = [0] + [int(np.round(i * n / self.n_splits)) for i in range(1, self.n_splits + 1)]
        embargo = int(np.round(n * self.embargo_frac))
        for k in range(self.n_splits):
            a, b = bounds[k], bounds[k + 1] - 1  # test block [a, b] in date positions
            test_mask = (positions >= a) & (positions <= b)
            # no overlap iff p < a - h - e  or  p > b + h + e
            keep_before = positions < a - self.horizon - embargo
            keep_after = positions > b + self.horizon + embargo
            train_mask = keep_before | keep_after
            yield (
                np.flatnonzero(train_mask),
                np.flatnonzero(test_mask),
            )

    def purge_margins(self, n_dates: int) -> tuple[int, int]:
        """The positional margin ``(left, right)`` purged around every test block."""
        embargo = int(np.round(n_dates * self.embargo_frac))
        return self.horizon + embargo, self.horizon + embargo


__all__ = ["PurgedKFold"]
