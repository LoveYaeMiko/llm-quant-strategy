"""Point-in-time data store — the first line of defence against look-ahead bias.

FINSABER's Bias Traps (review.md §2.4, blueprint §4C) require that a query at
time T returns data **exactly as it existed at T**:

* a fact is visible iff ``valid_from <= T < valid_to`` (NaN ``valid_to`` means
  "still valid" — the record is the most recent known fact);
* survivorship bias is structurally avoided because the universe at T is built
  from the constituents *alive at T* — names delisted after T remain present;
* delisted stocks are included, so factor evaluation never silently drops the
  names that later failed (FINSABER Defect 02).

Three implementations share one temporal-filtering core:

* :class:`PointInTimeStore` — in-memory (tests, small universes);
* :class:`SQLitePointInTimeLoader` — persistent, zero-dependency (stdlib
  ``sqlite3``);
* :class:`PostgresPointInTimeLoader` (src/data/postgres_loader.py) — the
  production JSONB backend; same query contract.

Every record carries a ``record_type`` feature discriminator — ``price``,
``universe``, ``fundamental`` or ``text`` — so one table holds bars, universe
snapshots and fundamentals without schema drift.
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

# A probe far after every ingested record: a query here is equivalent to "the
# whole known history", because every closed-interval record has expired.
_FULL_HISTORY_PROBE = "2099-01-01"


def _as_ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _payload_for(row: pd.Series) -> tuple[str, str, Optional[str], str, str]:
    """Serialize one store row into ``(symbol, valid_from, valid_to, record_type, payload_json)``.

    Shared by the SQLite and Postgres backends so both write byte-identical
    payloads. NumPy scalars are unwrapped; NaN becomes JSON ``null``.
    """
    features = {
        k: (None if pd.isna(v) else (v.item() if isinstance(v, (np.generic,)) else v))
        for k, v in row.items()
        if k not in _META_COLUMNS
    }
    payload = json.dumps(features, ensure_ascii=False, default=str)
    vf = _as_ts(row[VALID_FROM]).isoformat()
    vt = None if pd.isna(row[VALID_TO]) else _as_ts(row[VALID_TO]).isoformat()
    rt = str(row.get("record_type", "") or "")
    return (row[SYMBOL], vf, vt, rt, payload)


def _rows_to_frame(rows: Iterable[dict], fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Hydrate backend rows (dicts with symbol/valid_from/valid_to/payload keys).

    Always materializes ``valid_to`` (NaT for open-ended rows) and coerce-casts
    timestamps, so in-memory and DB backends return identical query shapes.

    Built column-wise in a single pass (no per-row dict list) so a full-history
    ``snapshot`` can stream the ~12M-row price panel without the ~18 GiB spike
    that previously OOM'd the daily shadow run. A key first seen on a later row
    back-fills ``None`` for the earlier rows, preserving the union-of-keys
    semantics of the old ``pd.DataFrame(list_of_dicts)`` construction.
    """
    columns: dict[str, list] = {}
    n = 0
    for row in rows:
        rec = json.loads(row["payload"])
        rec[SYMBOL] = row["symbol"]
        rec[VALID_FROM] = _as_ts(row["valid_from"])
        rec[VALID_TO] = pd.NaT if row["valid_to"] is None else _as_ts(row["valid_to"])
        for key in rec:
            if key not in columns:
                columns[key] = [None] * n
        for key in columns:
            columns[key].append(rec.get(key))
        n += 1
    if not columns:
        return pd.DataFrame()
    df = pd.DataFrame(columns)
    if fields is not None:
        want = [SYMBOL] + [c for c in fields if c in df.columns and c not in _META_COLUMNS]
        df = df[list(dict.fromkeys(want))]
    return df.reset_index(drop=True)


def _filter_record_type(df: pd.DataFrame, record_type: Optional[str]) -> pd.DataFrame:
    """Keep only rows of ``record_type`` (None ⇒ keep all)."""
    if record_type is None:
        return df
    if df.empty or "record_type" not in df.columns:
        return df.iloc[0:0]
    return df[df["record_type"] == record_type]


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
        # ``record_type`` is part of the uniqueness key: a price bar and a
        # universe snapshot for the same (symbol, valid_from) are DIFFERENT
        # facts and must coexist. Frames without the discriminator (hand-built
        # test fixtures) normalize to "" so the key stays stable across batches.
        if "record_type" not in df.columns:
            df["record_type"] = ""
        if VALID_TO not in df.columns:
            df[VALID_TO] = pd.NaT
        df[VALID_FROM] = pd.to_datetime(df[VALID_FROM])
        df[VALID_TO] = pd.to_datetime(df[VALID_TO], errors="coerce")
        if df.empty:
            return
        if self.records.empty:
            self.records = df
        else:
            if "record_type" not in self.records.columns:
                self.records["record_type"] = ""
            self.records = pd.concat([self.records, df], ignore_index=True)
        self.records = self.records.drop_duplicates(
            subset=[SYMBOL, VALID_FROM, "record_type"], keep="last"
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

    def universe_as_of(
        self, timestamp: str | pd.Timestamp, record_type: Optional[str] = None
    ) -> list[str]:
        """Universe as of ``timestamp``, optionally restricted to one record type.

        With ``record_type="universe"`` this reads the Baostock yearly snapshots;
        with ``None`` it falls back to whatever facts (e.g. price bars) are alive.
        """
        q = self.query(timestamp, fields=["record_type"] if record_type is not None else [])
        q = _filter_record_type(q, record_type)
        return sorted(q[SYMBOL].unique().tolist())

    def min_date(self, record_type: Optional[str] = None) -> pd.Timestamp:
        """Earliest ``valid_from`` among matching records (NaT when empty)."""
        if self.records.empty:
            return pd.NaT
        df = _filter_record_type(self.records, record_type)
        return pd.NaT if df.empty else df[VALID_FROM].min()

    def max_date(self, record_type: Optional[str] = None) -> pd.Timestamp:
        """Latest ``valid_from`` among matching records (NaT when empty)."""
        if self.records.empty:
            return pd.NaT
        df = _filter_record_type(self.records, record_type)
        return pd.NaT if df.empty else df[VALID_FROM].max()

    def max_valid_from(self, record_type: str = "price") -> pd.Timestamp:
        """Latest ingested bar date — the coverage end used by B5 freshness."""
        return self.max_date(record_type)

    def delisted_symbols(
        self, as_of: str | pd.Timestamp, record_type: Optional[str] = None
    ) -> list[str]:
        """Symbols alive at ``as_of`` that are absent from the newest snapshot.

        Feeds the B4 survivorship check: names that were investable at ``as_of``
        but have since left the market (delisted or suspended) — the exact set
        a naive "current constituents" backtest silently drops.
        """
        past = set(self.universe_as_of(as_of, record_type))
        latest = self.max_date(record_type)
        if pd.isna(latest):
            return []
        present = set(self.universe_as_of(latest, record_type))
        return sorted(past - present)

    def distinct_dates(self, record_type: Optional[str] = None) -> list[pd.Timestamp]:
        """Sorted distinct ``valid_from`` dates (coverage probe for B5 / ingest summary)."""
        df = _filter_record_type(self.records, record_type)
        if df.empty:
            return []
        return sorted(pd.DatetimeIndex(df[VALID_FROM].dropna().unique()).tolist())

    def snapshot(self, record_type: Optional[str] = None) -> pd.DataFrame:
        """All records of ``record_type`` regardless of validity (market builder)."""
        return _filter_record_type(self.records, record_type).reset_index(drop=True)

    def symbols(self, record_type: Optional[str] = None) -> list[str]:
        """Distinct symbols that ever had a record of ``record_type``."""
        df = _filter_record_type(self.records, record_type)
        return sorted(df[SYMBOL].unique().tolist())

    def history(self, symbol: str, record_type: Optional[str] = None) -> pd.DataFrame:
        """A symbol's full record history, oldest first (ignores validity windows)."""
        df = self.records
        if df.empty:
            return pd.DataFrame()
        df = df[df[SYMBOL] == symbol]
        df = _filter_record_type(df, record_type)
        return df.sort_values(VALID_FROM).reset_index(drop=True)

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

    ``payload`` is the JSON encoding of the feature columns (including
    ``record_type``). The temporal predicate is the same as
    :class:`PointInTimeStore`.
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
                symbol      TEXT    NOT NULL,
                valid_from  TEXT    NOT NULL,
                valid_to    TEXT,
                record_type TEXT    NOT NULL DEFAULT '',
                payload     TEXT    NOT NULL,
                PRIMARY KEY (symbol, valid_from, record_type)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_pit_times ON pit_records (valid_from, valid_to)")
        self._conn.commit()

    def _record_type_expr(self) -> str:
        return "json_extract(payload, '$.record_type')"

    def upsert(self, records: pd.DataFrame) -> None:
        store = PointInTimeStore()
        store.upsert(records)
        rows = [_payload_for(r) for _, r in store.records.iterrows()]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO pit_records (symbol, valid_from, valid_to, record_type, payload)
            VALUES (?, ?, ?, ?, ?)
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
        return _rows_to_frame(
            [
                {"symbol": s, "valid_from": vf, "valid_to": vt, "payload": p}
                for s, vf, vt, p in cur.fetchall()
            ],
            fields=fields,
        )

    def universe(self, timestamp: str | pd.Timestamp) -> list[str]:
        q = self.query(timestamp, fields=[])
        return sorted(q[SYMBOL].unique().tolist()) if not q.empty else []

    def universe_as_of(
        self, timestamp: str | pd.Timestamp, record_type: Optional[str] = None
    ) -> list[str]:
        q = self.query(timestamp, fields=["record_type"] if record_type is not None else [])
        q = _filter_record_type(q, record_type)
        return sorted(q[SYMBOL].unique().tolist()) if not q.empty else []

    def min_date(self, record_type: Optional[str] = None) -> pd.Timestamp:
        return self._agg_valid_from("MIN", record_type)

    def max_date(self, record_type: Optional[str] = None) -> pd.Timestamp:
        return self._agg_valid_from("MAX", record_type)

    def _agg_valid_from(self, agg: str, record_type: Optional[str]) -> pd.Timestamp:
        sql = f"SELECT {agg}(valid_from) FROM pit_records"
        params: tuple = ()
        if record_type is not None:
            sql += f" WHERE {self._record_type_expr()} = ?"
            params = (record_type,)
        row = self._conn.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return pd.NaT
        return _as_ts(row[0])

    def max_valid_from(self, record_type: str = "price") -> pd.Timestamp:
        return self.max_date(record_type)

    def delisted_symbols(
        self, as_of: str | pd.Timestamp, record_type: Optional[str] = None
    ) -> list[str]:
        past = set(self.universe_as_of(as_of, record_type))
        latest = self.max_date(record_type)
        if pd.isna(latest):
            return []
        present = set(self.universe_as_of(latest, record_type))
        return sorted(past - present)

    def distinct_dates(self, record_type: Optional[str] = None) -> list[pd.Timestamp]:
        sql = "SELECT DISTINCT valid_from FROM pit_records"
        params: tuple = ()
        if record_type is not None:
            sql += f" WHERE {self._record_type_expr()} = ?"
            params = (record_type,)
        rows = self._conn.execute(sql + " ORDER BY valid_from", params).fetchall()
        return [_as_ts(r[0]) for r in rows]

    def snapshot(self, record_type: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT symbol, valid_from, valid_to, payload FROM pit_records"
        params: tuple = ()
        if record_type is not None:
            sql += f" WHERE {self._record_type_expr()} = ?"
            params = (record_type,)
        cur = self._conn.execute(sql + " ORDER BY valid_from", params)
        rows = [
            {"symbol": s, "valid_from": vf, "valid_to": vt, "payload": p}
            for s, vf, vt, p in cur.fetchall()
        ]
        return _rows_to_frame(rows)

    def symbols(self, record_type: Optional[str] = None) -> list[str]:
        sql = "SELECT DISTINCT symbol FROM pit_records"
        params: tuple = ()
        if record_type is not None:
            sql += f" WHERE {self._record_type_expr()} = ?"
            params = (record_type,)
        rows = self._conn.execute(sql + " ORDER BY symbol", params).fetchall()
        return [r[0] for r in rows]

    def history(self, symbol: str, record_type: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT symbol, valid_from, valid_to, payload FROM pit_records WHERE symbol = ?"
        params: list = [symbol]
        if record_type is not None:
            sql += f" AND {self._record_type_expr()} = ?"
            params.append(record_type)
        cur = self._conn.execute(sql + " ORDER BY valid_from", params)
        rows = [
            {"symbol": s, "valid_from": vf, "valid_to": vt, "payload": p}
            for s, vf, vt, p in cur.fetchall()
        ]
        return _rows_to_frame(rows)

    def latest(self, timestamp: str | pd.Timestamp, fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
        q = self.query(timestamp)
        if q.empty:
            return pd.DataFrame(columns=[SYMBOL] + list(fields or []))
        q = q.sort_values(VALID_FROM)
        if fields is not None:
            cols = [SYMBOL] + [c for c in fields if c in q.columns and c not in _META_COLUMNS]
            q = q[cols]
        return q.groupby(SYMBOL, sort=True).tail(1).reset_index(drop=True)

    def has_future_leak(self, timestamp: str | pd.Timestamp, forbidden: str | pd.Timestamp) -> bool:
        t = _as_ts(timestamp).isoformat()
        f = _as_ts(forbidden).isoformat()
        cur = self._conn.execute(
            """
            SELECT 1 FROM pit_records
            WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
              AND valid_from > ?
            LIMIT 1
            """,
            (t, t, f),
        )
        return cur.fetchone() is not None

    def close(self) -> None:
        self._conn.close()


def from_url(url: str) -> PointInTimeStore | SQLitePointInTimeLoader:
    """Factory honouring a ``pit_database_url``.

    * ``sqlite:///<path>`` (or a bare filesystem path) → SQLite backend
    * ``postgresql://...`` → the Postgres JSONB backend (lazy import so
      ``psycopg2`` is only required when Postgres is actually used).
    """
    if not url:
        raise ValueError("empty pit_database_url — set PIT_DATABASE_URL (see .env)")
    if url.startswith("sqlite:///"):
        return SQLitePointInTimeLoader(url[len("sqlite:///") :])
    if url.startswith("postgresql"):
        from .postgres_loader import PostgresPointInTimeLoader

        return PostgresPointInTimeLoader(url)
    return SQLitePointInTimeLoader(url)


def build_price_bars(
    prices: pd.DataFrame,
    symbols: Iterable[str],
    freq: str = "1D",
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
