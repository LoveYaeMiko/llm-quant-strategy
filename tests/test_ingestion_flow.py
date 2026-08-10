"""End-to-end ingestion flow — vendor frames → PIT records → checks."""

from __future__ import annotations

import pandas as pd

from src.data.ingestion.alphafeed_adapter import to_price_records
from src.data.ingestion.convert import price_records, universe_records
from src.data.point_in_time_loader import PointInTimeStore, build_price_bars


def _klines():
    return {
        "600519.SH": pd.DataFrame(
            {
                "trade_date": ["2024-01-02", "2024-01-03"],
                "name": ["贵州茅台", "贵州茅台"],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.5, 10.5],
                "close": [10.0, 11.0],
                "volume": [100, 120],
                "amount": [1000, 1300],
            }
        )
    }


def test_full_price_flow_closed_interval_and_pit_query():
    frame = to_price_records(_klines(), None)
    recs = price_records(frame)
    assert (recs["record_type"] == "price").all()
    # closed interval: bar born 01-02 is visible at 01-02, gone at 01-03
    store = PointInTimeStore()
    store.upsert(recs)
    assert set(store.query("2024-01-02")["symbol"]) == {"600519.SH"}
    assert store.query("2024-01-02").iloc[0]["close"] == 10.0
    # no future leak across the boundary
    assert store.has_future_leak("2024-01-02", "2024-01-02") is False


def test_universe_records_support_survivorship():
    store = PointInTimeStore()
    snap = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2015-01-05", "name": "退市股"},
            {"symbol": "BBB", "date": "2015-01-05", "name": "存续股"},
        ]
    )
    store.upsert(universe_records(snap))
    later = pd.DataFrame([{"symbol": "BBB", "date": "2024-06-28", "name": "存续股"}])
    store.upsert(universe_records(later))
    delisted = store.delisted_symbols("2015-01-05", "universe")
    assert delisted == ["AAA"]


def test_build_price_bars_roundtrip():
    recs = price_records(to_price_records(_klines(), None))
    assert (recs["valid_to"] == recs["valid_from"] + pd.Timedelta("1D")).all()
    # build_price_bars stays the low-level helper for manual frames
    bars = build_price_bars(
        pd.DataFrame([{"symbol": "A", "valid_from": "2024-01-01", "close": 1.0}]),
        symbols=["A"],
    )
    assert bars["valid_to"].iloc[0] == pd.Timestamp("2024-01-02")
