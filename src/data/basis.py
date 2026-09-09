"""Price-basis contract for the PIT read layer (defect C2).

**The finding.** The point-in-time price payload is *not* on one price basis.
For ``record_type='price'`` the stored payload is::

    {"open": .., "high": .., "low": ..,
     "close": <ADJUSTED>, "raw_close": <RAW>,
     "adjust_factor": <float>, "volume": .., "amount": .., "name": ..}

ADR-0002 chose this deliberately: ``close`` is backward-adjusted
(``close = raw_close × adjust_factor``, the factor anchored at the newest bar)
so ``TS_Return`` has no ex-dividend gaps, while ``raw_close`` + ``adjust_factor``
preserve the unadjusted tape for consistency checks. What ADR-0002 did *not*
state — and what the storage layer never enforced — is that ``open``/``high``
``/low`` are written **raw**. ``src/cli.py::_market_from_records`` builds
``market.long`` and ``market.price_panel`` straight from those columns, so every
consumer inherits a panel whose ``close`` column is on a different basis than
its ``open``/``high``/``low`` columns. A true range, an intraday gap or a
high/low breakout computed off that panel is arithmetically wrong on every
corporate-action bar (a 10:1 split makes ``high/close ≈ 10``).

**The contract.** This module makes the basis *explicit, measurable and
assertable* instead of implicit:

* :func:`infer_basis` derives, **empirically from the data**, which basis each
  price column actually carries — never from the assumption above. A column is
  classified by comparing it against its two possible twins (``raw_close`` for
  the raw basis, ``raw_close × adjust_factor`` for the adjusted basis) on the
  rows where ``|adjust_factor − 1| > 0.02``, i.e. the rows where the two bases
  are measurably different.
* :func:`basis_report` returns a JSON-serializable snapshot of that inference
  plus the measured evidence (row/symbol counts, factor range, largest relative
  mismatch, mixed columns). The read layer attaches it as ``market.basis``, so
  every artifact can record which basis its numbers are on.
* :func:`to_adjusted` / :func:`to_raw` are the **sanctioned conversion
  functions**: a consumer that needs a single basis converts explicitly, with a
  per-symbol forward-filled ``adjust_factor``, instead of silently mixing.
* :func:`assert_single_basis` turns the contract into a hard failure.

**The decision (why the panel is not re-based).** We deliberately do **not**
silently re-base ``market.long`` / ``market.price_panel`` to a single basis.
Re-basing would rewrite the input of every historical backtest, every stored
evidence artifact (``docs/D_TRACK_EVIDENCE.md``, the shadow ledger, the ML
artifacts' training frames) and every already-published metric, so a defect fix
would masquerade as a strategy change and no old number could be reproduced
again. Instead:

1. the panel keeps its current numbers — byte-identical, no result moves;
2. the basis of every column is now explicit (``market.basis``) and assertable
   (:func:`assert_single_basis`) inside any artifact;
3. new consumers that genuinely need one basis call :func:`to_adjusted` /
   :func:`to_raw`, which are auditable and testable, and the one existing
   exposed consumer (``src/paper/pullback_book.py``, which already scales
   ``high``/``low`` by the per-bar factor) is provably equivalent to
   ``to_adjusted`` for the columns it uses.

Only numpy/pandas are required, so this module is safe to import from the
storage, read and strategy layers alike.

.. note::
   :func:`infer_basis` / :func:`basis_report` memoize the per-symbol factor panel
   on the identity of the frame handed in (a report asks for it once per column).
   Treat the input frame as read-only while a report is being built: mutating the
   *same object* between calls can return a stale panel. Callers that copy first
   (the read layer, and every conversion here) are unaffected.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

#: Sentinel distinguishing "factor panel not supplied" from "there is none".
_FACTOR_UNSET: object = object()

BASIS_ADJUSTED = "adjusted"
BASIS_RAW = "raw"
BASIS_UNKNOWN = "unknown"

#: The price columns governed by the basis contract.
PRICE_COLUMNS = ("open", "high", "low", "close")

#: |adjust_factor - 1| must exceed this for a row to carry basis evidence.
#: 2% is comfortably above float/rounding noise (payloads are rounded to ~1e-6)
#: and below the smallest real A-share ex-event (a 1-for-1 bonus is already
#: 50%), so the evidence rows are exactly the corporate-action bars.
ACTION_THRESHOLD = 0.02

#: Absolute relative tolerance below which a column counts as "exactly" on a
#: basis. A stored payload is float64 with ~1e-15 relative rounding and a price
#: column derived as ``raw × (1 + w)`` inherits another ~1e-16 per operation, so
#: 1e-9 is six orders of magnitude above the noise and seven below the smallest
#: real ex-event (a 1% factor move already gives a 1e-2 relative gap).
MATCH_TOL = 1e-9

#: A basis also wins when its median relative error is at most this fraction of
#: the alternative's. Together with :data:`MATCH_TOL` this keeps inference robust
#: to a payload that is only *approximately* on a basis. Measured on a 6-symbol /
#: 22,928-bar slice of the production store (21,901 action rows, factors
#: 0.041..1.0): the raw twin's median relative error is 9.3e-3 (``open``), 1.0e-2
#: (``high``), 9.6e-3 (``low``) — that residual is the real intraday move from the
#: previous close, not noise — while the adjusted twin is 3.1e-1, 3.2e-1 and
#: 2.9e-1 respectively, i.e. ~30× worse. ``close`` is the mirror image: 2.4e-1
#: against the raw twin and 1.3e-6 against the adjusted one. A column genuinely on
#: neither basis sees comparable errors on both twins and stays ``"unknown"`` —
#: the rule never guesses.
_DECISIVE_RATIO = 0.25

_FACTOR = "adjust_factor"
_RAW_CLOSE = "raw_close"
_CLOSE = "close"
_SYMBOL = "symbol"
_DATE_KEYS = ("valid_from", "date")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _num(records: pd.DataFrame, column: str) -> Optional[pd.Series]:
    """Numeric view of ``column`` (None when the column is absent)."""
    if column not in records.columns:
        return None
    return pd.to_numeric(records[column], errors="coerce")


def _has_basis_evidence(records: pd.DataFrame) -> bool:
    """True when both basis twins are present (raw_close and close)."""
    return _num(records, _RAW_CLOSE) is not None and _num(records, _CLOSE) is not None


def _raw_factor_series(records: pd.DataFrame) -> Optional[pd.Series]:
    """The stored ``adjust_factor`` as float, NaN preserved (None if absent)."""
    return _num(records, _FACTOR)


def _factor_series(records: pd.DataFrame) -> pd.Series:
    """The stored ``adjust_factor`` as a float Series (1.0 where absent/NaN)."""
    f = _raw_factor_series(records)
    if f is None:
        return pd.Series(1.0, index=records.index)
    return f.fillna(1.0).astype(float)


def _effective_factor(
    records: pd.DataFrame, wide: Optional[pd.DataFrame] = _FACTOR_UNSET
) -> pd.Series:
    """Per-row factor actually in force: per-symbol forward-filled, NaN → 1.0.

    The stored ``adjust_factor`` can be missing on a bar (the payload may omit it
    or carry null); the factor in force there is the symbol's last known one, the
    same rule :func:`to_adjusted` uses. Inference therefore compares columns
    against the *effective* factor so a missing cell cannot masquerade as "no
    evidence".

    ``wide`` lets a caller reuse an already-built panel (a report needs it once
    per column); passing None means "no stored factors at all".
    """
    stored = _factor_series(records)
    if wide is _FACTOR_UNSET:
        if _FACTOR not in getattr(records, "columns", []):
            return stored
        try:
            wide = _factor_frame(records)
        except (TypeError, ValueError, KeyError):  # noqa: BLE001 — degrade to stored
            return stored
    if wide is None or len(wide) == 0:
        return stored
    return _factor_at(records, wide)


def _evidence_mask(records: pd.DataFrame, threshold: float = ACTION_THRESHOLD) -> pd.Series:
    """Rows where the two bases measurably differ: ``|adjust_factor - 1| > threshold``."""
    return (_effective_factor(records) - 1.0).abs() > float(threshold)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """``num / den`` with zeros/NaN in the denominator turned into NaN."""
    den = den.where(den.abs() > 0)
    return num / den


def _twin_pair(
    records: pd.DataFrame, wide: Optional[pd.DataFrame] = _FACTOR_UNSET
) -> Optional[tuple[pd.Series, pd.Series]]:
    """``(raw_twin, adjusted_twin)`` for the tape, or None when undecidable.

    With ``raw_close`` stored, the twins are exact: ``raw_close`` and
    ``raw_close × factor``. Without it the only available anchor is the stored
    ``close``, which is *assumed* adjusted (the ADR-0002 contract) so the twins
    become ``close / factor`` and ``close``. That fallback cannot verify the
    assumption — :func:`infer_basis` will simply find ``close`` "adjusted" by
    construction — but it keeps the conversions working on panels that carry only
    ``close`` + ``adjust_factor``.
    """
    raw_close = _num(records, _RAW_CLOSE)
    close = _num(records, _CLOSE)
    if close is None:
        return None
    factor = _effective_factor(records, wide)
    safe = factor.where(factor.abs() > 0, 1.0)
    if raw_close is not None:
        return raw_close, raw_close * factor
    return close / safe, close


def _col_errors(
    records: pd.DataFrame, column: str, mask: pd.Series, wide: Optional[pd.DataFrame] = _FACTOR_UNSET
) -> tuple[float, float]:
    """Median ``|col/twin - 1|`` against the RAW twin and against the ADJUSTED twin.

    The twins are *constructed from the tape*, never read from the column under
    test::

        raw twin      = raw_close          (or close / factor when absent)
        adjusted twin = raw_close × factor (or close when raw_close is absent)

    Comparing ``close`` against the stored ``close`` would be circular (it would
    always match itself) and would hide a payload whose stored ``close`` was
    written raw. Only the rows selected by ``mask`` (the action rows) are used:
    on a factor-1.0 bar both twins coincide, so including those rows would dilute
    the median with uninformative zeros and hide the real evidence. Returns
    ``(raw_err, adj_err)``; ``inf`` for a basis whose twin is unavailable or has
    no usable evidence rows.
    """
    col = _num(records, column)
    pair = _twin_pair(records, wide)
    if col is None or pair is None:
        return float("inf"), float("inf")
    raw_twin, adj_twin = pair
    sub = mask.fillna(False) & col.notna() & raw_twin.notna() & adj_twin.notna()
    if not bool(sub.any()):
        return float("inf"), float("inf")
    c = col[sub]
    raw_err = float(np.nanmedian(np.abs(_safe_ratio(c, raw_twin[sub]) - 1.0).to_numpy()))
    adj_err = float(np.nanmedian(np.abs(_safe_ratio(c, adj_twin[sub]) - 1.0).to_numpy()))
    return raw_err, adj_err


def _expected_twin(
    records: pd.DataFrame, column: str, basis: str, wide: Optional[pd.DataFrame] = _FACTOR_UNSET
) -> Optional[pd.Series]:
    """The value ``column`` would have on ``basis``: ``raw_close × factor`` or ``raw_close``.

    ``close`` is its own adjusted twin; a raw column's twin is ``raw_close``.
    """
    raw_close = _num(records, _RAW_CLOSE)
    if raw_close is None:
        return None
    if basis == BASIS_ADJUSTED:
        return raw_close * _effective_factor(records, wide)
    return raw_close


def _mismatch(
    records: pd.DataFrame, column: str, basis: str, wide: Optional[pd.DataFrame] = _FACTOR_UNSET
) -> float:
    """Largest relative gap between ``column`` and its expected twin (0.0 if unknown).

    Uses the *effective* factor (per-symbol forward-filled), so a bar whose stored
    ``adjust_factor`` is missing is compared against the factor actually in force.
    """
    col = _num(records, column)
    twin = _expected_twin(records, column, basis, wide)
    if col is None or twin is None:
        return 0.0
    ratio = _safe_ratio(col, twin)
    rel = (ratio - 1.0).abs()
    rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
    if rel.empty:
        return 0.0
    value = float(rel.max())
    return value if np.isfinite(value) else 0.0


def _json_number(value: object) -> Optional[float]:
    """Coerce a numpy/pandas scalar to a plain JSON-safe float (None for NaN/inf)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _factor_column(records: pd.DataFrame, symbol_col: str, date_col: Optional[str]) -> pd.DataFrame:
    """Long frame ``(symbol, date, factor)`` for a per-symbol forward-fill.

    ``factor`` is the *backward*-adjustment factor in force at that bar, with
    missing values preserved as NaN so they can be forward-filled per symbol.
    When the records carry no date column the fill degenerates to one factor per
    symbol.
    """
    factor = _raw_factor_series(records)
    if factor is None:
        factor = pd.Series(1.0, index=records.index)
    out = pd.DataFrame({symbol_col: records[symbol_col].astype(str).to_numpy(), "factor": factor.to_numpy()})
    if date_col is not None:
        out["date"] = pd.to_datetime(records[date_col]).to_numpy()
    return out


def _factor_frame(records: pd.DataFrame) -> Optional[pd.DataFrame]:
    """``date × symbol`` factor panel, forward-filled per symbol, NaN → 1.0.

    This is the same rule as ``PullbackPortfolio._adjust_factor_frame``: the
    factor is anchored at the newest bar, so a missing factor (a bar before the
    first recorded action, or a payload that omits the key) legitimately means
    1.0. Bars with no stored factor inherit their symbol's last known factor
    *before* the pivot, so a missing cell can never clobber the forward-fill.
    """
    if _SYMBOL not in records.columns:
        return None
    date_col = next((c for c in _DATE_KEYS if c in records.columns), None)
    frame = _factor_column(records, _SYMBOL, date_col)
    if date_col is not None:
        frame = frame.sort_values([_SYMBOL, "date"], kind="stable")
    frame = frame.assign(factor=frame.groupby(_SYMBOL, sort=False)["factor"].ffill()).dropna(
        subset=["factor"]
    )
    if frame.empty:
        return None
    if date_col is None:
        wide = frame.groupby(_SYMBOL, sort=False)["factor"].last().to_frame().T
        return wide
    wide = frame.pivot_table(index="date", columns=_SYMBOL, values="factor", aggfunc="last")
    return wide.sort_index().ffill().fillna(1.0)


def _factor_at(records: pd.DataFrame, wide: Optional[pd.DataFrame]) -> pd.Series:
    """Per-row factor in force: stored value, else the per-symbol filled panel.

    A row with its own ``adjust_factor`` uses it verbatim (the panel is a
    fallback, not an override). A row whose factor is missing/NaN inherits the
    symbol's last known factor from ``wide`` (which is already forward-filled by
    :func:`_factor_frame`); a row outside the panel falls back to 1.0.
    """
    stored = _factor_series(records)
    raw = _raw_factor_series(records)
    missing = raw.isna() if raw is not None else pd.Series(True, index=records.index)
    if wide is None or len(wide) == 0 or _SYMBOL not in records.columns or not bool(missing.any()):
        return stored
    date_col = next((c for c in _DATE_KEYS if c in records.columns), None)
    if date_col is None:
        return stored
    wide_dates = pd.DatetimeIndex(wide.index)
    dates = pd.to_datetime(records[date_col])
    symbols = records[_SYMBOL].astype(str)
    col_of = {c: i for i, c in enumerate(wide.columns)}
    cols = symbols.map(col_of).to_numpy()
    pos = wide_dates.searchsorted(pd.DatetimeIndex(dates))
    values = wide.to_numpy()
    filled = np.full(len(records), np.nan, dtype=float)
    ok = pd.notna(cols) & (pos < len(wide_dates))
    if ok.any():
        filled[np.flatnonzero(ok)] = values[pos[ok].astype(int), cols[ok].astype(int)]
    filled_series = pd.Series(filled, index=records.index)
    return stored.where(~missing, filled_series.fillna(stored)).astype(float)


def _rebase(records: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a copy of ``records`` with every price column on ``target``.

    The stored bases are *inferred* first, so a column already on ``target`` is
    copied through untouched (the conversion is idempotent and never
    double-adjusts a column that is already on the wanted basis). A column whose
    basis cannot be inferred (``"unknown"``) is converted anyway, using the
    conservative default — assume raw when converting to adjusted and assume
    adjusted when converting to raw — so the returned frame is always on
    ``target`` as claimed and the two functions round-trip. Unknown columns are
    exactly the ones with no ``adjust_factor``/``raw_close`` evidence, where the
    factor is 1.0 anyway, so this default is a no-op on real payloads.
    """
    if target not in (BASIS_ADJUSTED, BASIS_RAW):
        raise ValueError(f"target basis must be {BASIS_ADJUSTED!r} or {BASIS_RAW!r}, got {target!r}")
    out = records.copy()
    inferred = infer_basis(out)
    factors = _factor_frame(out)
    factor = _factor_at(out, factors)
    # factor > 0 by construction; guard a degenerate zero payload value.
    safe = factor.where(factor.abs() > 0, 1.0)
    raw_close = _num(out, _RAW_CLOSE)
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            continue
        values = _num(out, col)
        if values is None:
            continue
        current = inferred.get(col, BASIS_UNKNOWN)
        if current == target:
            continue  # already on the wanted basis
        if target == BASIS_ADJUSTED:
            out[col] = (values * factor).to_numpy()
        elif col == _CLOSE and raw_close is not None:
            out[col] = raw_close.to_numpy()  # the stored raw tape is exact
        else:
            out[col] = (values / safe).to_numpy()
    out["_basis"] = target
    out.attrs["basis"] = target
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def infer_basis(records: pd.DataFrame) -> dict[str, str]:
    """Which basis each price column in ``records`` is actually on.

    **Inference rule (empirical, not assumed).** Only the rows with
    ``|adjust_factor − 1| > 0.02`` can distinguish the two bases — everywhere
    else ``raw_close == close`` and the question is unanswerable. On those
    *action rows* each price column is compared with its two candidate twins::

        raw twin      = raw_close
        adjusted twin = close  (== raw_close × adjust_factor)

    and the basis whose median ``|column / twin − 1|`` is smaller wins, provided
    that error is at most :data:`MATCH_TOL` (1e-9) or at most
    :data:`_DECISIVE_RATIO` (25%) of the alternative's. Ties, an absent twin
    (``raw_close`` missing), or a column with no action rows return
    ``"unknown"`` — this function never guesses.

    Returns a mapping over :data:`PRICE_COLUMNS` present in ``records``.
    """
    out: dict[str, str] = {}
    if records is None or len(records) == 0:
        return {col: BASIS_UNKNOWN for col in PRICE_COLUMNS}
    mask = _evidence_mask(records)
    for col in PRICE_COLUMNS:
        if col not in records.columns:
            continue
        raw_err, adj_err = _col_errors(records, col, mask)
        if not np.isfinite(raw_err) and not np.isfinite(adj_err):
            out[col] = BASIS_UNKNOWN
        elif raw_err <= MATCH_TOL and adj_err <= MATCH_TOL:
            # Both twins match: factor is 1 on every evidence row (no real action).
            out[col] = BASIS_UNKNOWN
        elif raw_err <= MATCH_TOL or raw_err <= _DECISIVE_RATIO * adj_err:
            out[col] = BASIS_RAW
        elif adj_err <= MATCH_TOL or adj_err <= _DECISIVE_RATIO * raw_err:
            out[col] = BASIS_ADJUSTED
        else:
            out[col] = BASIS_UNKNOWN
    return out


def basis_report(records: pd.DataFrame, threshold: float = ACTION_THRESHOLD) -> dict:
    """JSON-serializable evidence snapshot for the basis contract.

    Keys:

    ``columns``
        ``{price column: basis}`` — :func:`infer_basis` output.
    ``n_rows`` / ``n_symbols``
        Size of the frame handed in.
    ``n_action_rows`` / ``n_action_symbols``
        Rows and distinct symbols with ``|adjust_factor − 1| > threshold`` — the
        only rows that can carry basis evidence.
    ``factor_min`` / ``factor_max``
        Observed range of ``adjust_factor`` (None when the column is absent).
    ``median_rel_error``
        ``{column: {raw: .., adjusted: ..}}`` — the measured median relative
        error of each column against each twin on the action rows. This is the
        evidence behind ``columns``: the winning twin's error is the small one.
    ``max_rel_mismatch``
        Largest relative gap between a column and its expected twin
        (``close`` vs ``raw_close × factor``, ``raw`` columns vs ``raw_close``).
        For a raw ``high`` this is the day's real intraday move from the previous
        close (~1.7e-1 worst case on the production store), not a defect: it says
        the column is on the raw *tape*, not identical to ``raw_close``.
    ``consistent``
        True when at least one column was classified and every classified column
        sits on the same basis.
    ``mixed_columns``
        The classified columns whose basis differs from the dominant one.
    ``dominant_basis`` / ``threshold`` / ``basis_column``
        Echoed inputs and the majority verdict, so a stored artifact is
        self-describing.
    """
    columns = infer_basis(records) if records is not None and len(records) else {
        col: BASIS_UNKNOWN for col in PRICE_COLUMNS
    }
    n_rows = int(len(records)) if records is not None and hasattr(records, "__len__") else 0
    if records is not None and _SYMBOL in getattr(records, "columns", []):
        n_symbols = int(records[_SYMBOL].astype(str).nunique())
    else:
        n_symbols = 0
    if n_rows:
        mask = _evidence_mask(records, threshold)
        n_action_rows = int(mask.sum())
        if n_action_rows and _SYMBOL in records.columns:
            n_action_symbols = int(records.loc[mask, _SYMBOL].astype(str).nunique())
        else:
            n_action_symbols = 0
        factor = _num(records, _FACTOR)
        factor_min = _json_number(factor.min()) if factor is not None else None
        factor_max = _json_number(factor.max()) if factor is not None else None
        mismatches = {
            col: _mismatch(records, col, basis)
            for col, basis in columns.items()
            if basis != BASIS_UNKNOWN
        }
        max_rel_mismatch = _json_number(max(mismatches.values())) if mismatches else None
        median_rel_error: dict[str, dict[str, Optional[float]]] = {}
        for col in columns:
            raw_err, adj_err = _col_errors(records, col, mask)
            median_rel_error[col] = {
                BASIS_RAW: _json_number(raw_err),
                BASIS_ADJUSTED: _json_number(adj_err),
            }
    else:
        n_action_rows = n_action_symbols = 0
        factor_min = factor_max = max_rel_mismatch = None
        median_rel_error = {}
    classified = {c: b for c, b in columns.items() if b != BASIS_UNKNOWN}
    if classified:
        counts: dict[str, int] = {}
        for b in classified.values():
            counts[b] = counts.get(b, 0) + 1
        dominant = max(sorted(counts), key=lambda b: counts[b])
        mixed = sorted(c for c, b in classified.items() if b != dominant)
        consistent = not mixed
    else:
        dominant = BASIS_UNKNOWN
        mixed = []
        consistent = False
    return {
        "columns": {c: columns[c] for c in PRICE_COLUMNS if c in columns},
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "n_action_rows": n_action_rows,
        "n_action_symbols": n_action_symbols,
        "factor_min": factor_min,
        "factor_max": factor_max,
        "median_rel_error": median_rel_error,
        "max_rel_mismatch": max_rel_mismatch,
        "consistent": bool(consistent),
        "mixed_columns": mixed,
        "dominant_basis": dominant,
        "threshold": float(threshold),
        "basis_column": "_basis",
    }


def to_adjusted(records: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``records`` with every price column on the ADJUSTED basis.

    Each column's stored basis is inferred first: a raw ``open``/``high``/``low``
    is multiplied by the bar's per-symbol forward-filled ``adjust_factor`` (1.0
    where the factor is absent — the newest-bar anchor and factor-less panels),
    while ``close`` (already adjusted) is copied through untouched. A column whose
    basis cannot be inferred is converted on the conservative default (assume
    raw), so the result really is on one basis. The result carries
    ``_basis == "adjusted"`` as a column and in ``.attrs["basis"]``. The input is
    never mutated.
    """
    return _rebase(records, BASIS_ADJUSTED)


def to_raw(records: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``records`` with every price column on the RAW basis.

    ``close`` becomes ``raw_close`` when present (otherwise ``close / factor``);
    ``open``/``high``/``low`` are divided by the per-symbol forward-filled
    ``adjust_factor`` unless they are already on the raw basis. The result carries
    ``_basis == "raw"`` as a column and in ``.attrs["basis"]``. The input is never
    mutated.
    """
    return _rebase(records, BASIS_RAW)


def assert_single_basis(
    records: pd.DataFrame,
    target: str,
    columns: Optional[Iterable[str]] = None,
) -> None:
    """Raise ``ValueError`` unless every requested price column is on ``target``.

    ``target`` is ``"adjusted"`` or ``"raw"``. ``columns`` defaults to every
    :data:`PRICE_COLUMNS` present in the frame. A column whose basis cannot be
    inferred (``"unknown"`` — no corporate-action rows, or no ``raw_close`` to
    compare against) fails too: "unverified" is not "verified". The message names
    the offending columns, their inferred bases and the measured evidence.
    """
    if target not in (BASIS_ADJUSTED, BASIS_RAW):
        raise ValueError(f"target basis must be {BASIS_ADJUSTED!r} or {BASIS_RAW!r}, got {target!r}")
    want = list(columns) if columns is not None else [c for c in PRICE_COLUMNS]
    inferred = infer_basis(records)
    report = basis_report(records)
    offenders = {c: inferred.get(c, BASIS_UNKNOWN) for c in want if inferred.get(c, BASIS_UNKNOWN) != target}
    if offenders:
        detail = ", ".join(f"{c}={b}" for c, b in sorted(offenders.items()))
        raise ValueError(
            f"price-basis contract violated: expected every column on {target!r} but got {detail} "
            f"(columns={dict(sorted(inferred.items()))}, n_rows={report['n_rows']}, "
            f"n_action_rows={report['n_action_rows']}, "
            f"factor_min={report['factor_min']}, factor_max={report['factor_max']}, "
            f"max_rel_mismatch={report['max_rel_mismatch']}). "
            f"Convert explicitly with src.data.basis.to_adjusted / to_raw."
        )


def basis_of(records_or_report: object, column: str) -> str:
    """Basis of one column from a records frame *or* an already-built report."""
    if isinstance(records_or_report, Mapping):
        columns = records_or_report.get("columns") or {}
        return str(columns.get(column, BASIS_UNKNOWN))
    return infer_basis(records_or_report).get(column, BASIS_UNKNOWN)


__all__ = [
    "BASIS_ADJUSTED",
    "BASIS_RAW",
    "BASIS_UNKNOWN",
    "PRICE_COLUMNS",
    "ACTION_THRESHOLD",
    "MATCH_TOL",
    "infer_basis",
    "basis_report",
    "basis_of",
    "to_adjusted",
    "to_raw",
    "assert_single_basis",
]
