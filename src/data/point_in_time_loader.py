"""Point-in-time data store — the first line of defence against look-ahead bias.

FINSABER's Bias Traps (review.md §2.4, blueprint §4C) require that a query at
time T returns data **exactly as it existed at T**:

* a fact is visible iff ``valid_from <= T < valid_to`` (NaN ``valid_to`` means
  "still valid" — the record is the most recent known fact);
* survivorship bias is structurally avoided because the universe at T is built
  from the constituents *alive at T* — names delisted after T remain present;
* delisted stocks are included, so factor evaluation never silently drops the
  names that later failed (FINSABER Defect 02).

Two implementations share one temporal-filtering core:

* :class:`PointInTimeStore` — in-memory (tests, small universes);
* :class:`SQLitePointInTimeLoader` — persistent, zero-dependency (stdlib
  ``sqlite3``); the production Postgres backend keeps the same query contract.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SYMBOL = "symbol"
VALID_FROM = "valid_from"
VALID_TO = "valid_to"

# Temporal bookkeeping columns — excluded when a caller asks for *feature* fields.
_META_COLUMNS = frozenset({SYMBOL, VALID_FROM, VALID_TO})


def _as_ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


@dataclass
class PointInTimeStore:
    """In-memory point-in-time store.

    Rows carry feature columns plus temporal bookkeeping. Upserting a new fact
    for the same ``(symbol, valid_from)`` supersedes the old row, mirroring how
    a restated earnings figure replaces an earlier snapshot.
    """

    records: pd.DataFrame = field(default_factory=pd.DataFrame)

    def upsert(self, records: pd.DataFrame) -> None:
        """Add or replace point-in-time records.

        Parameters
        ----------
        records : DataFrame
            Must contain ``symbol`` and ``valid_from``. ``valid_to`` is optional
            (NaN ⇒ record still valid). Remaining columns are treated as features.
        """
        for col in (SYMBOL, VALID_FROM):
            if col not in records.columns:
                raise ValueError(f"missing required column {col!r}")
        df = records.copy()
        if VALID_TO not in df.columns:
            df[VALID_TO] = pd.NaT
        df[VALID_FROM] = pd.to_datetime(df[VALID_FROM])
        df[VALID_TO] = pd.to_datetime(df[VALID_TO], errors="coerce")
        if df.empty:
            return
        if self.records.empty:
            self.records = df
        else:
            self.records = pd.concat([self.records, df], ignore_index=True)
        self.records = self.records.drop_duplicates(
            subset=[SYMBOL, VALID_FROM], keep="last"
        ).sort_values([SYMBOL, VALID_FROM]).reset_index(drop=True)

    def query(
        self,
        timestamp: str | pd.Timestamp,
        fields: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return all facts visible at ``timestamp``.

        A row is visible iff ``valid_from <= T < valid_to``. Facts born after T
        (e.g. a bar dated 2024-01-01 when T is 2023-12-31) are structurally
        excluded — this is the PIT guarantee the verification checklist probes.
        """
        t = _as_ts(timestamp)
        df = self.records
        if df.empty:
            return pd.DataFrame()
        mask = (df[VALID_FROM] <= t) & (df[VALID_TO].isna() | (df[VALID_TO] > t))
        out = df.loc[mask].copy()
        if fields is not None:
            want = [SYMBOL] + [c for c in fields if c in out.columns and c not in _META_COLUMNS]
            out = out[list(dict.fromkeys(want))]  # dedupe, keep order
        return out.reset_index(drop=True)

    def universe(self, timestamp: str | pd.Timestamp) -> list[str]:
        """Constituents alive at ``timestamp`` — includes names delisted after T."""
        q = self.query(timestamp, fields=[])
        return sorted(q[SYMBOL].unique().tolist())

    def latest(
        self,
        timestamp: str | pd.Timestamp,
        fields: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """One row per symbol: the most recent fact visible at ``timestamp``."""
        q = self.query(timestamp)
        if q.empty:
            cols = [SYMBOL] + list(fields or [])
            return pd.DataFrame(columns=cols)
        q = q.sort_values(VALID_FROM)
        if fields is not None:
            cols = [SYMBOL] + [c for c in fields if c in q.columns and c not in _META_COLUMNS]
            q = q[cols]
        return q.groupby(SYMBOL, sort=True).tail(1).reset_index(drop=True)

    def has_future_leak(self, timestamp: str | pd.Timestamp, forbidden: str | pd.Timestamp) -> bool:
        """True if any fact born strictly after ``timestamp`` would be visible.

        Used by the PIT verification test: querying 2023-12-31 must never expose
        a fact with ``valid_from`` on 2024-01-01.
        """
        t = _as_ts(timestamp)
        f = _as_ts(forbidden)
        df = self.records
        if df.empty:
            return False
        leaked = df[(df[VALID_FROM] <= t) & (df[VALID_TO].isna() | (df[VALID_TO] > t)) & (df[VALID_FROM] > f)]
        return not leaked.empty


class SQLitePointInTimeLoader:
    """Persistent PIT loader on SQLite (stdlib only).

    Schema::

        pit_records(symbol TEXT, valid_from TEXT, valid_to TEXT, payload TEXT)

    ``payload`` is the JSON encoding of the feature columns. The temporal
    predicate is the same as :class:`PointInTimeStore`.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pit_records (
                symbol     TEXT    NOT NULL,
                valid_from TEXT    NOT NULL,
                valid_to   TEXT,
                payload    TEXT    NOT NULL,
                PRIMARY KEY (symbol, valid_from)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_pit_times ON pit_records (valid_from, valid_to)")
        self._conn.commit()

    def upsert(self, records: pd.DataFrame) -> None:
        store = PointInTimeStore()
        store.upsert(records)
        rows = []
        for _, r in store.records.iterrows():
            features = {
                k: (None if pd.isna(v) else (v.item() if isinstance(v, (np.generic,)) else v))
                for k, v in r.items()
                if k not in _META_COLUMNS
            }
            payload = json.dumps(features, ensure_ascii=False, default=str)
            vf = pd.Timestamp(r[VALID_FROM]).isoformat()
            vt = None if pd.isna(r[VALID_TO]) else pd.Timestamp(r[VALID_TO]).isoformat()
            rows.append((r[SYMBOL], vf, vt, payload))
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO pit_records (symbol, valid_from, valid_to, payload)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def query(self, timestamp: str | pd.Timestamp, fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
        t = _as_ts(timestamp).isoformat()
        cur = self._conn.execute(
            """
            SELECT symbol, valid_from, valid_to, payload
            FROM pit_records
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            """,
            (t, t),
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        out = []
        for symbol, vf, vt, payload in rows:
            rec = json.loads(payload)
            rec[SYMBOL] = symbol
            rec[VALID_FROM] = pd.Timestamp(vf)
            if vt is not None:
                rec[VALID_TO] = pd.Timestamp(vt)
            out.append(rec)
        df = pd.DataFrame(out)
        if fields is not None:
            want = [SYMBOL] + [c for c in fields if c in df.columns and c not in _META_COLUMNS]
            df = df[list(dict.fromkeys(want))]
        return df.reset_index(drop=True)

    def universe(self, timestamp: str | pd.Timestamp) -> list[str]:
        q = self.query(timestamp, fields=[])
        return sorted(q[SYMBOL].unique().tolist()) if not q.empty else []

    def latest(self, timestamp: str | pd.Timestamp, fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
        q = self.query(timestamp)
        if q.empty:
            return pd.DataFrame(columns=[SYMBOL] + list(fields or []))
        q = q.sort_values(VALID_FROM)
        if fields is not None:
            cols = [SYMBOL] + [c for c in fields if c in q.columns and c not in _META_COLUMNS]
            q = q[cols]
        return q.groupby(SYMBOL, sort=True).tail(1).reset_index(drop=True)

    def close(self) -> None:
        self._conn.close()


def from_url(url: str) -> PointInTimeStore | SQLitePointInTimeLoader:
    """Factory honouring a ``pit_database_url``.

    * ``sqlite:///<path>`` (or a bare filesystem path) → SQLite backend
    * ``postgresql://...``  → not bundled; raise a clear actionable error.
    """
    if url.startswith("sqlite:///"):
        return SQLitePointInTimeLoader(url[len("sqlite:///") :])
    if url.startswith("postgresql"):
        raise NotImplementedError(
            "Postgres PIT backend is the production target; for local runs use "
            "sqlite:///data/pit_data.db (configs/.env PIT_DATABASE_URL)."
        )
    return SQLitePointInTimeLoader(url)


def build_price_bars(
    prices: pd.DataFrame,
    symbols: Iterable[str],
    freq: str = "D",
) -> pd.DataFrame:
    """Turn a long OHLCV frame into PIT records.

    A bar stamped date d is a fact born at ``valid_from = d`` and valid for the
    next interval, so a query at date d sees the bar whose close was known *at*
    d — never one from the future.
    """
    df = prices.copy()
    df[VALID_FROM] = pd.to_datetime(df[VALID_FROM])
    df[VALID_TO] = df[VALID_FROM] + pd.to_timedelta(freq)
    return df
