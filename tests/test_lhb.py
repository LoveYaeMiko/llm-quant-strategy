"""Tests for the dragon-tiger list PIT module (src/data/lhb.py) — offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.lhb import (
    NET_BUY,
    RECORD_TYPE,
    REASON,
    _normalize,
    to_pit_records,
    upsert_lhb,
)
from src.data.point_in_time_loader import PointInTimeStore


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "代码": ["600519", "600519", "000001"],
            "名称": ["贵州茅台", "贵州茅台", "平安银行"],
            "上榜日": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "收盘价": [10.0, 10.0, 20.0],
            "涨跌幅": [1.0, 1.0, -2.0],
            "龙虎榜净买额": [1000.0, 500.0, -300.0],
            "龙虎榜买入额": [2000.0, 800.0, 100.0],
            "龙虎榜卖出额": [1000.0, 300.0, 400.0],
            "龙虎榜成交额": [3000.0, 1100.0, 500.0],
            "市场总成交额": [1e9, 1e9, 2e9],
            "净买额占总成交额比": [0.05, 0.03, -0.01],
            "换手率": [1.0, 1.0, 2.0],
            "流通市值": [2e11, 2e11, 1e11],
            "上榜原因": ["机构买入", "日涨幅偏离7%", "日跌幅偏离7%"],
            "上榜后1日": [1.0, 1.0, 1.0],   # future data — must be stripped
            "上榜后5日": [2.0, 2.0, 2.0],
            "上榜后10日": [3.0, 3.0, 3.0],
        }
    )


def test_normalize_strips_lookahead_columns():
    out = _normalize(_raw_frame())
    for c in ("上榜后1日", "上榜后5日", "上榜后10日"):
        assert c not in out.columns


def test_normalize_aggregates_same_symbol_same_day():
    out = _normalize(_raw_frame())
    row = out[(out["symbol"] == "600519.SH")]
    assert len(row) == 1  # two listings of one day collapse to one record
    assert row.iloc[0][NET_BUY] == pytest.approx(1500.0)  # summed
    assert "机构买入" in row.iloc[0][REASON] and "日涨幅偏离7%" in row.iloc[0][REASON]


def test_to_pit_records_visible_next_day():
    recs = to_pit_records(_normalize(_raw_frame()))
    assert list(recs["record_type"].unique()) == [RECORD_TYPE]
    # listing dated 2024-01-02 visible from 2024-01-03
    first = recs[recs["symbol"] == "600519.SH"].iloc[0]
    assert first["valid_from"] == pd.Timestamp("2024-01-03")
    # one open tail per symbol, closed by the next listing
    assert recs["valid_to"].isna().sum() == 2


def test_upsert_lhb_no_look_ahead_and_closing():
    store = PointInTimeStore()
    upsert_lhb(store, _normalize(_raw_frame()))
    # the 01-03 listing (net_buy = -300) publishes after close — invisible then
    q = store.query(pd.Timestamp("2024-01-03 15:00"), fields=[NET_BUY])
    assert q.empty or (q[NET_BUY] > -300 + 1e-9).all()
    # next day it is visible, one live record per symbol
    q2 = store.query(pd.Timestamp("2024-01-04 15:00"), fields=[NET_BUY])
    assert q2.groupby("symbol").size().eq(1).all()
    assert (q2[NET_BUY] < -300 + 1e-9).any()


def test_empty_lhb():
    assert to_pit_records(pd.DataFrame()).empty
    assert upsert_lhb(PointInTimeStore(), pd.DataFrame()) == {"records": 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
