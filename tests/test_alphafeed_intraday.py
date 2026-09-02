"""Tests for the AlphaFeed intraday-family adapter methods (mock client)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter


class _FakeKlines:
    def batch(self, symbols, period=None, count=None, to_dataframe=False, **kw):
        return {s: pd.DataFrame({"close": [1.0, 2.0]}) for s in symbols}

    def intraday_batch(self, symbols, period=None, count=None, to_dataframe=False, **kw):
        return {s: pd.DataFrame({"price": [10.0]}) for s in symbols}


class _FakeDepth:
    def batch(self, symbols, **kw):
        return {s: {"bids": [], "asks": []} for s in symbols}


class _FakeClient:
    def __init__(self):
        self.klines = _FakeKlines()
        self.depth = _FakeDepth()
        self.quotes = None


@pytest.fixture
def adapter():
    a = AlphaFeedAdapter(api_key="test-key")
    a._client = _FakeClient()
    return a


def test_fetch_minute_klines_batches_symbols(adapter):
    out = adapter.fetch_minute_klines(["600519.SH", "000001.SZ"], period="5m", count=100)
    assert set(out) == {"600519.SH", "000001.SZ"}
    assert list(out["600519.SH"].columns) == ["close"]


def test_fetch_intraday_batches_symbols(adapter):
    out = adapter.fetch_intraday(["600519.SH"], period="1m", count=240)
    assert set(out) == {"600519.SH"}
    assert "price" in out["600519.SH"].columns


def test_fetch_depth_batches_symbols(adapter):
    out = adapter.fetch_depth(["600519.SH", "000001.SZ"])
    assert set(out) == {"600519.SH", "000001.SZ"}
    assert "bids" in out["600519.SH"]
