"""Offline tests for the price-basis contract (defect C2, ``src/data/basis.py``).

The PIT payload mixes bases: ``close`` is backward-adjusted while
``open``/``high``/``low`` are raw (ADR-0002). These tests pin the contract that
makes that mix explicit, measurable and assertable — inference from the data,
JSON-serializable evidence, and sanctioned conversions — without touching any
stored number.

Every fixture is hand-built (no database, no network).

Two fixtures:

``_records``
    the real defect — ``open``/``high``/``low`` raw, ``close = raw_close × f``;
``_all_raw_records``
    a panel where every column is on the raw basis (``close == raw_close``)
    while a real 0.5 action keeps the two candidate twins distinguishable.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.basis import (
    BASIS_ADJUSTED,
    BASIS_RAW,
    BASIS_UNKNOWN,
    PRICE_COLUMNS,
    assert_single_basis,
    basis_report,
    infer_basis,
    to_adjusted,
    to_raw,
)

DATES = pd.bdate_range("2026-01-05", periods=4)
RAW = {
    "A": np.array([10.0, 10.5, 11.0, 11.5]),
    "B": np.array([5.0, 5.2, 5.4, 5.6]),
}


def _records(
    factor_b: float = 0.5,
    intraday_range: float = 0.0,
    missing_factor: bool = False,
) -> pd.DataFrame:
    """Two symbols, 4 bars; ``A`` factor 1.0, ``B`` factor ``factor_b``.

    Layout = the real defect: ``open``/``high``/``low`` are RAW, while
    ``close = raw_close × factor`` is ADJUSTED.
    """
    rows = []
    for sym, raw in RAW.items():
        factor = 1.0 if sym == "A" else factor_b
        for i, (d, rc) in enumerate(zip(DATES, raw)):
            row = {
                "symbol": sym,
                "valid_from": d,
                "open": rc,
                "high": rc * (1.0 + intraday_range),
                "low": rc * (1.0 - intraday_range),
                "raw_close": rc,
                "close": rc * factor,
                "adjust_factor": factor,
                "volume": 1_000_000.0,
            }
            if missing_factor and sym == "B" and i == 1:
                row["adjust_factor"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _all_raw_records(missing_factor: bool = False) -> pd.DataFrame:
    """Every price column on the raw basis, with a real 0.5 action on ``B``.

    ``close`` is the raw close itself and the intraday columns sit ±2% around it,
    so the raw columns are *on* the raw tape without all being numerically
    identical to ``raw_close`` — exactly the shape of the production store. The
    action keeps the two twins distinguishable, so inference has evidence.
    """
    rows = []
    for sym, raw in RAW.items():
        factor = 1.0 if sym == "A" else 0.5
        for i, (d, rc) in enumerate(zip(DATES, raw)):
            row = {
                "symbol": sym,
                "valid_from": d,
                "open": rc * 1.01,
                "high": rc * 1.02,
                "low": rc * 0.98,
                "raw_close": rc,
                "close": rc,
                "adjust_factor": factor,
                "volume": 1_000_000.0,
            }
            if missing_factor and sym == "B" and i == 1:
                row["adjust_factor"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _all_adjusted_records() -> pd.DataFrame:
    """Every price column on the adjusted basis (a correctly converted panel)."""
    rec = _records()
    for col in ("open", "high", "low"):
        rec[col] = rec[col] * rec["adjust_factor"]
    return rec


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def test_infer_basis_only_close_is_adjusted():
    """The canonical C2 shape: raw OHLC-minus-close, adjusted close."""
    inferred = infer_basis(_records())
    assert inferred["open"] == BASIS_RAW
    assert inferred["high"] == BASIS_RAW
    assert inferred["low"] == BASIS_RAW
    assert inferred["close"] == BASIS_ADJUSTED
    assert set(inferred) == set(PRICE_COLUMNS)


def test_infer_basis_tolerates_an_approximate_raw_column():
    """A raw column that is not exactly ``raw_close`` still loses decisively."""
    inferred = infer_basis(_records(intraday_range=0.02))
    assert inferred["high"] == BASIS_RAW
    assert inferred["low"] == BASIS_RAW


def test_infer_basis_all_raw_frame():
    """``close == raw_close`` while a 0.5 action exists ⇒ every column raw."""
    inferred = infer_basis(_all_raw_records())
    assert set(inferred.values()) == {BASIS_RAW}


def test_infer_basis_all_adjusted_frame():
    """Every column pre-scaled by the factor ⇒ every column is adjusted."""
    inferred = infer_basis(_all_adjusted_records())
    assert set(inferred.values()) == {BASIS_ADJUSTED}


def test_infer_basis_no_action_rows_is_unknown():
    """No corporate action ⇒ the two bases coincide ⇒ no evidence, no guess."""
    rec = _all_raw_records()
    rec["adjust_factor"] = 1.0
    assert set(infer_basis(rec).values()) == {BASIS_UNKNOWN}


def test_infer_basis_without_raw_close_falls_back_to_close_over_factor():
    """Without ``raw_close`` the stored ``close`` is the only anchor available.

    The fallback assumes the ADR-0002 contract (``close`` is adjusted) and then
    measures the other columns against ``close / factor`` — so ``close`` itself
    reads back as adjusted by construction and the raw OHLC still resolves as raw.
    """
    rec = _records().drop(columns=["raw_close"])
    inferred = infer_basis(rec)
    assert inferred["open"] == BASIS_RAW
    assert inferred["close"] == BASIS_ADJUSTED


def test_infer_basis_on_neither_basis_is_unknown():
    """A column that matches neither twin is reported unknown, not forced."""
    rec = _records()
    rec["high"] = rec["high"] * 3.0
    assert infer_basis(rec)["high"] == BASIS_UNKNOWN


def test_infer_basis_empty_frame():
    assert set(infer_basis(pd.DataFrame()).values()) == {BASIS_UNKNOWN}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_basis_report_shape_and_evidence():
    rep = basis_report(_records())
    assert rep["columns"] == {
        "open": BASIS_RAW,
        "high": BASIS_RAW,
        "low": BASIS_RAW,
        "close": BASIS_ADJUSTED,
    }
    assert rep["n_rows"] == 8
    assert rep["n_symbols"] == 2
    assert rep["n_action_rows"] == 4          # symbol B's four bars
    assert rep["n_action_symbols"] == 1
    assert rep["factor_min"] == pytest.approx(0.5)
    assert rep["factor_max"] == pytest.approx(1.0)
    assert rep["max_rel_mismatch"] == pytest.approx(0.0, abs=1e-12)
    assert rep["consistent"] is False         # close differs from OHLC
    assert rep["mixed_columns"] == ["close"]
    assert rep["dominant_basis"] == BASIS_RAW
    # The measured evidence behind the verdict (twins constructed from the tape).
    assert rep["median_rel_error"]["close"][BASIS_RAW] == pytest.approx(0.5)
    assert rep["median_rel_error"]["close"][BASIS_ADJUSTED] == pytest.approx(0.0)
    assert rep["median_rel_error"]["open"][BASIS_RAW] == pytest.approx(0.0)
    assert rep["median_rel_error"]["open"][BASIS_ADJUSTED] == pytest.approx(1.0)


def test_basis_report_counts_action_rows_from_the_effective_factor():
    """A missing stored factor is forward-filled before counting evidence."""
    rep = basis_report(_records(missing_factor=True))
    assert rep["n_action_rows"] == 4          # row 1 still inherits 0.5
    assert rep["factor_min"] == pytest.approx(0.5)


def test_basis_report_is_json_serializable():
    rep = basis_report(_records())
    text = json.dumps(rep)                    # must not raise
    assert json.loads(text) == rep
    # No numpy scalars survive: every numeric leaf is a builtin.
    for key in ("n_rows", "n_symbols", "n_action_rows", "n_action_symbols"):
        assert type(rep[key]) is int
    for key in ("factor_min", "factor_max", "max_rel_mismatch", "threshold"):
        assert rep[key] is None or type(rep[key]) is float
    assert all(type(v) is str for v in rep["columns"].values())


def test_basis_report_consistent_when_single_basis():
    rep = basis_report(_all_raw_records())
    assert rep["consistent"] is True
    assert rep["mixed_columns"] == []
    assert rep["dominant_basis"] == BASIS_RAW
    # open/high/low are the raw tape, not raw_close itself: the 2% gap is the
    # intraday move, which is exactly why the twin is not an equality.
    assert rep["max_rel_mismatch"] == pytest.approx(0.02)
    assert rep["median_rel_error"]["close"][BASIS_RAW] == pytest.approx(0.0)


def test_basis_report_empty_frame_is_json_serializable():
    rep = basis_report(pd.DataFrame())
    assert rep["n_rows"] == 0
    assert rep["consistent"] is False
    json.dumps(rep)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def test_to_adjusted_puts_every_column_on_the_adjusted_basis():
    rec = _records(intraday_range=0.02)
    adj = to_adjusted(rec)
    factor = rec["adjust_factor"]
    for col in ("open", "high", "low"):
        np.testing.assert_allclose(adj[col].to_numpy(), (rec[col] * factor).to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(adj["close"].to_numpy(), rec["close"].to_numpy())
    assert set(adj["_basis"]) == {BASIS_ADJUSTED}
    assert adj.attrs["basis"] == BASIS_ADJUSTED
    assert infer_basis(adj) == {c: BASIS_ADJUSTED for c in PRICE_COLUMNS}


def test_to_raw_puts_every_column_on_the_raw_basis():
    rec = _records()
    raw = to_raw(rec)
    for col in ("open", "high", "low"):
        np.testing.assert_allclose(raw[col].to_numpy(), rec[col].to_numpy(), rtol=0, atol=1e-12)
    # close was adjusted ⇒ it becomes the stored raw tape.
    np.testing.assert_allclose(raw["close"].to_numpy(), rec["raw_close"].to_numpy())
    assert set(raw["_basis"]) == {BASIS_RAW}
    assert raw.attrs["basis"] == BASIS_RAW
    assert infer_basis(raw) == {c: BASIS_RAW for c in PRICE_COLUMNS}


def test_round_trip_adjusted_raw():
    rec = _records(intraday_range=0.02)
    adj = to_adjusted(rec)
    back = to_raw(adj)
    # open/high/low round-trip exactly; close round-trips to the raw tape.
    for col in ("open", "high", "low"):
        np.testing.assert_allclose(back[col].to_numpy(), rec[col].to_numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(back["close"].to_numpy(), rec["raw_close"].to_numpy(), rtol=1e-12)
    again = to_adjusted(back)
    for col in PRICE_COLUMNS:
        np.testing.assert_allclose(again[col].to_numpy(), adj[col].to_numpy(), rtol=1e-12, atol=1e-12)


def test_round_trip_from_raw_basis():
    """An all-raw frame survives adjusted → raw on every column."""
    rec = _all_raw_records()
    adj = to_adjusted(rec)
    assert set(infer_basis(adj).values()) == {BASIS_ADJUSTED}
    back = to_raw(adj)
    for col in ("open", "high", "low"):
        np.testing.assert_allclose(back[col].to_numpy(), rec[col].to_numpy(), rtol=1e-12, atol=1e-12)
    # close round-trips to the raw tape (identical to the input's raw close).
    np.testing.assert_allclose(back["close"].to_numpy(), rec["raw_close"].to_numpy(), rtol=1e-12)


def test_conversion_does_not_mutate_input():
    rec = _records()
    before = rec.copy(deep=True)
    to_adjusted(rec)
    to_raw(rec)
    pd.testing.assert_frame_equal(rec, before)
    assert "_basis" not in rec.columns


def test_to_adjusted_forward_fills_missing_factor():
    """A bar with a NaN factor inherits the symbol's last known factor."""
    rec = _records(missing_factor=True)
    adj = to_adjusted(rec)
    b = adj[adj["symbol"] == "B"].reset_index(drop=True)
    # Row 1 has no stored factor; it must be filled from row 0 (0.5), not 1.0.
    np.testing.assert_allclose(b.loc[1, "open"], 5.2 * 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.loc[0, "low"], 5.0 * 0.5, rtol=0, atol=1e-12)
    # A is untouched (factor 1.0 throughout).
    a = adj[adj["symbol"] == "A"].reset_index(drop=True)
    np.testing.assert_allclose(a["open"].to_numpy(), RAW["A"], rtol=0, atol=1e-12)


def test_to_raw_without_raw_close_divides_by_factor():
    rec = _records().drop(columns=["raw_close"])
    raw = to_raw(rec)
    b = raw[raw["symbol"] == "B"].reset_index(drop=True)
    np.testing.assert_allclose(b["close"].to_numpy(), [5.0, 5.2, 5.4, 5.6], rtol=1e-12)
    np.testing.assert_allclose(b["open"].to_numpy(), [5.0, 5.2, 5.4, 5.6], rtol=1e-12)


def test_to_adjusted_is_idempotent():
    rec = _records(intraday_range=0.02)
    once = to_adjusted(rec)
    twice = to_adjusted(once)
    for col in PRICE_COLUMNS:
        np.testing.assert_allclose(twice[col].to_numpy(), once[col].to_numpy(), rtol=0, atol=1e-12)


def test_conversion_rejects_unknown_target():
    with pytest.raises(ValueError, match="target basis must be"):
        assert_single_basis(_records(), "split-adjusted")


# ---------------------------------------------------------------------------
# assertion
# ---------------------------------------------------------------------------


def test_assert_single_basis_raises_on_mixed_panel():
    with pytest.raises(ValueError) as exc:
        assert_single_basis(_records(), BASIS_ADJUSTED)
    msg = str(exc.value)
    assert "price-basis contract violated" in msg
    assert "open=raw" in msg and "high=raw" in msg
    assert "n_action_rows=4" in msg

def test_assert_single_basis_passes_after_conversion():
    assert_single_basis(to_adjusted(_records()), BASIS_ADJUSTED)
    assert_single_basis(to_raw(_records()), BASIS_RAW)


def test_assert_single_basis_rejects_unknown_columns():
    """Unverifiable is not verified: no evidence ⇒ the assertion fails."""
    rec = _all_raw_records()
    rec["adjust_factor"] = 1.0
    with pytest.raises(ValueError) as exc:
        assert_single_basis(rec, BASIS_ADJUSTED)
    assert "close=unknown" in str(exc.value)


def test_assert_single_basis_restricts_to_requested_columns():
    # close is adjusted, OHLC is raw → asking only about close must pass.
    assert_single_basis(_records(), BASIS_ADJUSTED, columns=["close"])
    with pytest.raises(ValueError):
        assert_single_basis(_records(), BASIS_ADJUSTED, columns=["high"])


def test_assert_single_basis_does_not_raise_on_empty_request():
    assert_single_basis(_records(), BASIS_ADJUSTED, columns=[])
