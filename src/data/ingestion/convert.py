"""Convert vendor frames → PIT record frames (record_type + closed interval).

Every record emitted by the ingestion pipeline carries a ``record_type`` so the
shared ``pit_records`` table can hold bars, universe snapshots, fundamentals and
news without schema drift, and so the CLI market builder can filter to price
records alone.
"""

from __future__ import annotations

import pandas as pd

from ..point_in_time_loader import SYMBOL, VALID_FROM, VALID_TO

PRICE = "price"
UNIVERSE = "universe"
FUNDAMENTAL = "fundamental"
TEXT = "text"

_BAR_FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def _stamp_column(df: pd.DataFrame) -> str:
    if VALID_FROM in df.columns:
        return VALID_FROM
    if "date" in df.columns:
        return "date"
    raise ValueError(f"frame needs a {VALID_FROM!r} or 'date' column")


def price_records(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Long OHLCV frame → PIT price records (closed interval, ADR-0001).

    Expects ``symbol`` + ``date``/``valid_from`` + OHLCV (``raw_close`` and
    ``adjust_factor`` optional). Sets ``valid_to = valid_from + freq`` so a query
    at day T returns exactly the bars born on T.
    """
    out = df.copy()
    out = out.rename(columns={_stamp_column(out): VALID_FROM})
    out[VALID_FROM] = pd.to_datetime(out[VALID_FROM])
    out[VALID_TO] = out[VALID_FROM] + pd.to_timedelta(freq)
    out["record_type"] = PRICE
    cols = [SYMBOL, VALID_FROM, VALID_TO, "record_type"]
    for col in _BAR_FIELDS + ["raw_close", "adjust_factor", "name"]:
        if col in out.columns:
            cols.append(col)
    return out[cols]


def universe_records(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Universe snapshot rows (Baostock ``query_all_stock``) → PIT universe records.

    Expects ``symbol`` + ``date``/``valid_from`` plus any of
    ``name``/``ipo_date``/``out_date``/``status``.

    NOTE: deliberately CLOSED-interval (``valid_to = valid_from + 1D``), diverging
    from ADR-0001's "open-ended for universe" rule. A closed interval makes
    ``universe_as_of(T)`` mean "exactly the members listed in the snapshot at T",
    which is what B4 survivorship counting relies on — an open-ended record would
    return the union of every snapshot and the delisted count would always be 0.
    """

    out = df.copy()
    out = out.rename(columns={_stamp_column(out): VALID_FROM})
    out[VALID_FROM] = pd.to_datetime(out[VALID_FROM])
    out[VALID_TO] = out[VALID_FROM] + pd.to_timedelta(freq)
    out["record_type"] = UNIVERSE
    cols = [SYMBOL, VALID_FROM, VALID_TO, "record_type"]
    for col in ["name", "ipo_date", "out_date", "status"]:
        if col in out.columns:
            cols.append(col)
    return out[cols]


def fundamental_records(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Fundamental facts keyed ``(symbol, valid_from)``; extra columns pass through."""
    out = df.copy()
    out = out.rename(columns={_stamp_column(out): VALID_FROM})
    out[VALID_FROM] = pd.to_datetime(out[VALID_FROM])
    out[VALID_TO] = out[VALID_FROM] + pd.to_timedelta(freq)
    out["record_type"] = FUNDAMENTAL
    return out


def text_records(df: pd.DataFrame, kind: str = "news", freq: str = "1D") -> pd.DataFrame:
    """News / announcements / reports → PIT text records with a ``kind`` feature."""
    out = df.copy()
    out = out.rename(columns={_stamp_column(out): VALID_FROM})
    out[VALID_FROM] = pd.to_datetime(out[VALID_FROM])
    out[VALID_TO] = out[VALID_FROM] + pd.to_timedelta(freq)
    out["record_type"] = TEXT
    out["kind"] = kind
    return out
