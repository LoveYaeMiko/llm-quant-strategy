"""Postgres JSONB loader tests — psycopg2 fully mocked (no live DB needed)."""

from __future__ import annotations

import pandas as pd

import psycopg2  # noqa: F401  (installed via the data extra)

from src.data.postgres_loader import PostgresPointInTimeLoader


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.conn.last_sql = (sql, params)

    def fetchall(self):
        return self.conn.fetch_result

    def fetchone(self):
        return self.conn.fetch_result[0] if self.conn.fetch_result else None


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result or []
        self.last_sql = None
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def close(self):
        self.closed = True


def _loader(conn, monkeypatch):
    monkeypatch.setattr("src.data.postgres_loader.psycopg2.connect", lambda url: conn)
    return PostgresPointInTimeLoader("postgresql://pit:pit@localhost:5432/pit_data")


def test_upsert_builds_on_conflict_sql(monkeypatch):
    calls = []

    def fake_execute_values(cur, sql, tuples, template=None, page_size=1000):
        calls.append((sql, list(tuples), template, page_size))

    monkeypatch.setattr("psycopg2.extras.execute_values", fake_execute_values)
    loader = _loader(FakeConn(), monkeypatch)
    loader.upsert(
        pd.DataFrame(
            [{"symbol": "A", "valid_from": "2024-01-01", "close": 1.0, "record_type": "price"}]
        )
    )
    assert len(calls) == 1
    sql, tuples, template, page_size = calls[0]
    assert "ON CONFLICT" in sql
    assert "record_type" in sql
    assert template == "(%s, %s, %s, %s, %s::jsonb)"
    assert page_size == 1000
    sym, vf, vt, rt, payload = tuples[0]
    assert sym == "A"
    assert vf == "2024-01-01T00:00:00"
    assert vt is None
    assert rt == "price"
    assert '"close": 1.0' in payload
    assert "ON CONFLICT (symbol, valid_from, record_type)" in sql


def test_query_hydrates_jsonb_rows(monkeypatch):
    conn = FakeConn(
        [("A", "2024-01-01T00:00:00", None, '{"close": 1.0, "record_type": "price"}')]
    )
    loader = _loader(conn, monkeypatch)
    q = loader.query("2024-01-02")
    assert q["symbol"].iloc[0] == "A"
    assert q["valid_from"].iloc[0] == pd.Timestamp("2024-01-01")
    assert pd.isna(q["valid_to"].iloc[0])
    assert q["close"].iloc[0] == 1.0
    assert "price" in q["record_type"].values


def test_aggregations_and_has_future_leak(monkeypatch):
    conn = FakeConn()
    loader = _loader(conn, monkeypatch)
    # min_date / max_date read agg rows
    conn.fetch_result = [("2024-01-01T00:00:00",)]
    assert loader.min_date("price") == pd.Timestamp("2024-01-01")
    conn.fetch_result = [("2024-02-01T00:00:00",)]
    assert loader.max_valid_from("price") == pd.Timestamp("2024-02-01")
    # has_future_leak sees a leaked row
    conn.fetch_result = [(1,)]
    assert loader.has_future_leak("2024-01-01", "2024-01-01") is True
    conn.fetch_result = []
    assert loader.has_future_leak("2024-01-01", "2024-01-01") is False
    # distinct_dates
    conn.fetch_result = [("2024-01-01T00:00:00",), ("2024-01-02T00:00:00",)]
    assert len(loader.distinct_dates("price")) == 2


def test_close_closes_connection(monkeypatch):
    conn = FakeConn()
    loader = _loader(conn, monkeypatch)
    loader.close()
    assert conn.closed
