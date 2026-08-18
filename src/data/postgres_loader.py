"""Postgres JSONB PIT backend (ADR-0003) — the production store.

Raw ``psycopg2`` (no ORM). One shared table for every record kind::

    pit_records(
        symbol      TEXT       NOT NULL,
        valid_from  TIMESTAMP  NOT NULL,
        valid_to    TIMESTAMP,          -- NULL ⇒ still valid
        record_type TEXT       NOT NULL DEFAULT '',  -- price / universe / fundamental / text
        payload     JSONB      NOT NULL, -- feature columns incl. record_type
        updated_at  TIMESTAMP  NOT NULL DEFAULT now(),
        PRIMARY KEY (symbol, valid_from, record_type)
    )

Bulk writes go through ``execute_values`` (page_size 1000) with
``ON CONFLICT (symbol, valid_from) DO UPDATE`` so a restated fact replaces the
earlier snapshot in place. The temporal predicate and the query shape are
identical to :class:`~src.data.point_in_time_loader.SQLitePointInTimeLoader`.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

import pandas as pd
import psycopg2
import psycopg2.extras

from .point_in_time_loader import (
    SYMBOL,
    VALID_FROM,
    VALID_TO,
    _META_COLUMNS,
    _as_ts,
    _filter_record_type,
    _payload_for,
    _rows_to_frame,
    PointInTimeStore,
)

_PAGE_SIZE = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pit_records (
    symbol      TEXT       NOT NULL,
    valid_from  TIMESTAMP  NOT NULL,
    valid_to    TIMESTAMP,
    record_type TEXT       NOT NULL DEFAULT '',
    payload     JSONB      NOT NULL,
    updated_at  TIMESTAMP  NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, valid_from, record_type)
);
CREATE INDEX IF NOT EXISTS ix_pit_times ON pit_records (valid_from, valid_to);
"""


class PostgresPointInTimeLoader:
    """PIT loader on Postgres JSONB — same query contract as the SQLite loader."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._conn = psycopg2.connect(url)
        self._init_schema()

    # -- write --------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)
        self._conn.commit()

    def upsert(self, records: pd.DataFrame) -> None:
        """Bulk upsert, superseding prior facts for the same ``(symbol, valid_from, record_type)``.

        ``record_type`` is part of the key so a price bar and a universe snapshot
        on the same (symbol, valid_from) coexist instead of clobbering each other
        (the ADR-0003 collision that destroyed the 2015 universe cohort).
        """
        store = PointInTimeStore()
        store.upsert(records)
        rows = [_payload_for(r) for _, r in store.records.iterrows()]
        if not rows:
            return
        tuples = [(s, vf, vt, rt, p) for s, vf, vt, rt, p in rows]
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO pit_records (symbol, valid_from, valid_to, record_type, payload)
                VALUES %s
                ON CONFLICT (symbol, valid_from, record_type) DO UPDATE SET
                    valid_to   = EXCLUDED.valid_to,
                    payload    = EXCLUDED.payload,
                    updated_at = now()
                """,
                tuples,
                template="(%s, %s, %s, %s, %s::jsonb)",
                page_size=_PAGE_SIZE,
            )
        self._conn.commit()

    # -- read ---------------------------------------------------------------

    def query(self, timestamp: str | pd.Timestamp, fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
        t = _as_ts(timestamp).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, valid_from, valid_to, payload::text
                FROM pit_records
                WHERE valid_from <= %s AND (valid_to IS NULL OR valid_to > %s)
                """,
                (t, t),
            )
            rows = [
                {"symbol": s, "valid_from": vf, "valid_to": vt, "payload": p}
                for s, vf, vt, p in cur.fetchall()
            ]
        return _rows_to_frame(rows, fields=fields)

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
        params: list = []
        if record_type is not None:
            sql += " WHERE payload->>'record_type' = %s"
            params.append(record_type)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
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
        params: list = []
        if record_type is not None:
            sql += " WHERE payload->>'record_type' = %s"
            params.append(record_type)
        with self._conn.cursor() as cur:
            cur.execute(sql + " ORDER BY valid_from", params)
            rows = cur.fetchall()
        return [_as_ts(r[0]) for r in rows]

    def snapshot(self, record_type: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT symbol, valid_from, valid_to, payload::text FROM pit_records"
        params: list = []
        if record_type is not None:
            sql += " WHERE payload->>'record_type' = %s"
            params.append(record_type)
        sql += " ORDER BY valid_from"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return _rows_to_frame(
                {"symbol": s, "valid_from": vf, "valid_to": vt, "payload": p}
                for s, vf, vt, p in cur
            )

    def symbols(self, record_type: Optional[str] = None) -> list[str]:
        sql = "SELECT DISTINCT symbol FROM pit_records"
        params: list = []
        if record_type is not None:
            sql += " WHERE payload->>'record_type' = %s"
            params.append(record_type)
        with self._conn.cursor() as cur:
            cur.execute(sql + " ORDER BY symbol", params)
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def history(self, symbol: str, record_type: Optional[str] = None) -> pd.DataFrame:
        sql = (
            "SELECT symbol, valid_from, valid_to, payload::text FROM pit_records "
            "WHERE symbol = %s"
        )
        params: list = [symbol]
        if record_type is not None:
            sql += " AND payload->>'record_type' = %s"
            params.append(record_type)
        sql += " ORDER BY valid_from"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
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
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM pit_records
                WHERE valid_from <= %s AND (valid_to IS NULL OR valid_to > %s)
                  AND valid_from > %s
                LIMIT 1
                """,
                (t, t, f),
            )
            return cur.fetchone() is not None

    def count(self, record_type: Optional[str] = None) -> int:
        """Total records, optionally of one kind (cheap coverage probe for B5)."""
        sql = "SELECT count(*) FROM pit_records"
        params: list = []
        if record_type is not None:
            sql += " WHERE payload->>'record_type' = %s"
            params.append(record_type)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


__all__ = ["PostgresPointInTimeLoader"]
