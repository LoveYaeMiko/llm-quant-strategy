"""Point-in-time store tests — the anti-look-ahead guarantee."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.point_in_time_loader import (
    PointInTimeStore,
    SQLitePointInTimeLoader,
    build_price_bars,
)


@pytest.fixture
def store():
    s = PointInTimeStore()
    s.upsert(
        pd.DataFrame(
            [
                {"symbol": "AAA", "valid_from": "2024-01-01", "valid_to": "2024-03-01", "close": 10.0},
                {"symbol": "AAA", "valid_from": "2024-03-01", "valid_to": pd.NaT, "close": 11.0},
                {"symbol": "BBB", "valid_from": "2024-02-01", "valid_to": pd.NaT, "close": 5.0},
                # a restated fact supersedes the earlier snapshot (same key)
                {"symbol": "AAA", "valid_from": "2024-01-01", "valid_to": "2024-03-01", "close": 10.5},
            ]
        )
    )
    return s


def test_query_visibility_window(store):
    # 2024-01-15: AAA's first record visible; BBB not yet born
    q = store.query("2024-01-15")
    assert set(q["symbol"]) == {"AAA"}
    assert (q["close"] == 10.5).all()  # restated value wins


def test_query_after_valid_to(store):
    q = store.query("2024-06-01")  # AAA now on the 03-01 record
    row = q[q["symbol"] == "AAA"].iloc[0]
    assert row["close"] == 11.0
    assert set(q["symbol"]) == {"AAA", "BBB"}


def test_future_fact_invisible(store):
    store.upsert(
        pd.DataFrame(
            [{"symbol": "CCC", "valid_from": "2024-12-01", "valid_to": pd.NaT, "close": 99.0}]
        )
    )
    assert store.query("2024-06-01").empty is False
    assert "CCC" not in store.universe("2024-06-01")
    assert store.has_future_leak("2024-06-01", "2024-12-01") is False


def test_universe_includes_delisted(store):
    # a name delisted after T remains in the universe at T
    assert "AAA" in store.universe("2024-06-01")
    assert "BBB" in store.universe("2024-06-01")


def test_latest_one_row_per_symbol(store):
    latest = store.latest("2024-06-01")
    assert set(latest["symbol"]) == {"AAA", "BBB"}
    assert len(latest) == 2


def test_has_future_leak_positive():
    s = PointInTimeStore()
    s.upsert(
        pd.DataFrame(
            [
                {"symbol": "A", "valid_from": "2024-01-01", "valid_to": "2024-01-05", "close": 1.0},
                {"symbol": "A", "valid_from": "2024-01-06", "valid_to": pd.NaT, "close": 2.0},
            ]
        )
    )
    # at 2024-01-08 the record born 01-06 is visible and born after 01-05 -> leak
    assert s.has_future_leak("2024-01-08", "2024-01-05") is True
    # at 2024-01-03 the only visible fact was born 01-01, before 01-02 -> no leak
    assert s.has_future_leak("2024-01-03", "2024-01-02") is False


def test_upsert_requires_columns():
    s = PointInTimeStore()
    with pytest.raises(ValueError):
        s.upsert(pd.DataFrame([{"close": 1.0}]))


def test_price_and_universe_records_coexist_same_date():
    """ADR-0003 collision: a price bar and a universe snapshot on the same
    (symbol, valid_from) are DIFFERENT facts — upserting one must not clobber
    the other. (Regression for the full-ingest bug that destroyed the 2015
    universe cohort when the price phase re-upserted matching bars.)"""
    s = PointInTimeStore()
    s.upsert(
        pd.DataFrame(
            [{"symbol": "600519.SH", "valid_from": "2015-01-05", "record_type": "universe", "name": "贵州茅台"}]
        )
    )
    s.upsert(
        pd.DataFrame(
            [{"symbol": "600519.SH", "valid_from": "2015-01-05", "record_type": "price", "close": 10.0}]
        )
    )
    q = s.query("2015-01-05")
    assert len(q) == 2, "price + universe must coexist on the same date"
    assert set(q["record_type"]) == {"price", "universe"}
    # same record_type on the same key still supersedes (restatement semantics)
    s.upsert(
        pd.DataFrame(
            [{"symbol": "600519.SH", "valid_from": "2015-01-05", "record_type": "price", "close": 11.0}]
        )
    )
    q = s.query("2015-01-05")
    assert len(q) == 2
    assert q.loc[q["record_type"] == "price", "close"].iloc[0] == 11.0


def test_sqlite_price_and_universe_coexist_same_date(tmp_path):
    db = tmp_path / "pit.db"
    loader = SQLitePointInTimeLoader(db)
    loader.upsert(
        pd.DataFrame(
            [{"symbol": "600519.SH", "valid_from": "2015-01-05", "record_type": "universe", "name": "贵州茅台"}]
        )
    )
    loader.upsert(
        pd.DataFrame(
            [{"symbol": "600519.SH", "valid_from": "2015-01-05", "record_type": "price", "close": 10.0}]
        )
    )
    q = loader.query("2015-01-05")
    assert len(q) == 2
    assert set(q["record_type"]) == {"price", "universe"}
    loader.close()


def test_sqlite_loader_roundtrip(tmp_path):
    db = tmp_path / "pit.db"
    loader = SQLitePointInTimeLoader(db)
    loader.upsert(
        pd.DataFrame(
            [
                {"symbol": "AAA", "valid_from": "2024-01-01", "valid_to": "2024-03-01", "close": 10.0},
                {"symbol": "BBB", "valid_from": "2024-02-01", "valid_to": pd.NaT, "close": 5.0},
            ]
        )
    )
    q = loader.query("2024-01-15")
    assert set(q["symbol"]) == {"AAA"}
    # at 2024-06-01, AAA's first record has expired (valid_to 03-01) and BBB
    # is still valid -> universe is only BBB
    assert loader.universe("2024-06-01") == ["BBB"]
    assert set(loader.universe("2024-02-15")) == {"AAA", "BBB"}
    loader.close()


def test_build_price_bars():
    bars = pd.DataFrame(
        [{"symbol": "A", "valid_from": "2024-01-01", "close": 10.0}]
    )
    out = build_price_bars(bars, symbols=["A"], freq="1D")
    assert out["valid_to"].iloc[0] == pd.Timestamp("2024-01-02")
