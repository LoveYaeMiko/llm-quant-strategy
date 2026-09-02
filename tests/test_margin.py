"""Tests for the margin-trading PIT module (src/data/margin.py) — offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.margin import (
    FIN_BALANCE,
    FIN_BUY,
    RECORD_TYPE,
    SL_BALANCE,
    SL_SELL_VOLUME,
    SL_VOLUME,
    _norm_sse,
    _norm_szse,
    load_margin_cache,
    margin_factors,
    to_pit_records,
    upsert_margin,
)
from src.data.point_in_time_loader import PointInTimeStore


def _margin_frame(days: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for sym in ("000001.SZ", "600519.SH"):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    FIN_BALANCE: 1_000_000.0 + i * 10_000,
                    FIN_BUY: 100_000.0 + i * 1_000,
                    SL_BALANCE: 50_000.0 + i * 500,
                    SL_VOLUME: 1_000.0 + i,
                    SL_SELL_VOLUME: 100.0 + i,
                }
            )
    return pd.DataFrame(rows)


def test_to_pit_records_visibility_is_next_day():
    recs = to_pit_records(_margin_frame())
    assert list(recs["record_type"].unique()) == [RECORD_TYPE]
    first = recs.iloc[0]
    assert first["valid_from"] == pd.Timestamp("2024-01-01") + pd.to_timedelta("1D")
    # within-batch closed intervals: valid_to = next snapshot's valid_from
    assert not recs["valid_to"].isna().all()
    assert recs["valid_to"].isna().sum() == 2  # one open tail per symbol
    # closed interval holds for every row
    closed = recs.dropna(subset=["valid_to"])
    assert (closed["valid_to"] > closed["valid_from"]).all()
    for col in (FIN_BALANCE, FIN_BUY, SL_BALANCE, SL_VOLUME, SL_SELL_VOLUME):
        assert col in recs.columns


def test_upsert_margin_no_look_ahead():
    store = PointInTimeStore()
    upsert_margin(store, _margin_frame())
    # balances rise +10k/day: balance of 01-03 = 1,020,000 (visible only 01-04)

    # at close of day 01-03 the latest visible balance is 01-02's = 1,010,000
    q_same_day = store.query(pd.Timestamp("2024-01-03 15:00"), fields=[FIN_BALANCE])
    assert not q_same_day.empty
    assert q_same_day[FIN_BALANCE].max() == pytest.approx(1_010_000.0)

    # next day the 01-03 balance is visible — exactly ONE live snapshot per symbol
    q_next = store.query(pd.Timestamp("2024-01-04 15:00"), fields=[FIN_BALANCE])
    assert q_next[FIN_BALANCE].max() == pytest.approx(1_020_000.0)
    assert q_next.groupby("symbol").size().eq(1).all()


def test_upsert_margin_incremental_batch_closes_old():
    """A second batch must close the first batch's open-ended tail."""
    store = PointInTimeStore()
    upsert_margin(store, _margin_frame(days=30, start="2024-01-01"))
    upsert_margin(store, _margin_frame(days=5, start="2024-02-12"))

    # after both batches, exactly one open-ended record per symbol remains
    snap = store.snapshot(RECORD_TYPE)
    open_ended = snap[snap["valid_to"].isna()]
    assert len(open_ended) == 2
    # a query in the gap between batches sees the first batch's tail
    q_gap = store.query(pd.Timestamp("2024-02-12 15:00"), fields=["valid_from"])
    assert not q_gap.empty
    # ...and after the second batch's first record, exactly one live fact per symbol
    q_after = store.query(pd.Timestamp("2024-02-14 15:00"), fields=["valid_from"])
    assert q_after.groupby("symbol").size().eq(1).all()


def test_margin_factors():
    fac = margin_factors(_margin_frame())
    assert set(fac.columns) == {"fin_growth", "fin_buy_growth", "sl_growth", "sl_mix"}
    # steady +10k/day financing balance: 20d growth ≈ 200k / 1.0M = 0.2
    d20 = pd.Timestamp("2024-01-29")
    row = fac.loc[(d20, "000001.SZ")]
    assert abs(row["fin_growth"] - 0.2) < 0.01
    # sl_mix at day 0 = 50k / 1M = 0.05
    assert abs(fac.iloc[0]["sl_mix"] - 0.05) < 1e-9


def test_margin_factors_requires_date_symbol():
    with pytest.raises(ValueError):
        margin_factors(pd.DataFrame(columns=[FIN_BALANCE]))


def test_load_margin_cache(tmp_path):
    # two checkpoint months on disk → concatenated, no network
    df = _margin_frame(days=5, start="2024-01-01")
    df2 = _margin_frame(days=5, start="2024-02-01")
    df.to_parquet(tmp_path / "margin_202401.parquet", index=False)
    df2.to_parquet(tmp_path / "margin_202402.parquet", index=False)
    loaded = load_margin_cache(tmp_path)
    assert len(loaded) == len(df) + len(df2)
    assert loaded["date"].min() == pd.Timestamp("2024-01-01")
    assert loaded["date"].max() == pd.Timestamp("2024-02-07")


def test_empty_margin_frame():
    recs = to_pit_records(pd.DataFrame())
    assert recs.empty
    assert upsert_margin(PointInTimeStore(), pd.DataFrame()) == {"records": 0}


def test_normalizers_column_mapping():
    sse = pd.DataFrame(
        {
            "信用交易日期": ["20260820"],
            "标的证券代码": ["600519"],
            "标的证券简称": ["贵州茅台"],
            "融资余额": ["1.2e9"],
            "融资买入额": ["1.0e8"],
            "融资偿还额": ["9.0e7"],
            "融券余量": ["12345"],
            "融券卖出量": ["678"],
            "融券偿还量": ["111"],
        }
    )
    out = _norm_sse(sse, "20260820")
    assert out.iloc[0]["symbol"] == "600519.SH"
    assert out.iloc[0][FIN_BALANCE] == pytest.approx(1.2e9)
    assert out.iloc[0][FIN_BUY] == pytest.approx(1.0e8)
    assert np.isnan(out.iloc[0][SL_BALANCE])  # SSE carries no short balance
    assert out.iloc[0][SL_VOLUME] == pytest.approx(12345)

    szse = pd.DataFrame(
        {
            "证券代码": ["000001"],
            "证券简称": ["平安银行"],
            "融资买入额": ["2.0e8"],
            "融资余额": ["3.0e9"],
            "融券卖出量": ["999"],
            "融券余额": ["1.5e7"],
            "融券余量": ["888"],
            "融资融券余额": ["3.015e9"],
        }
    )
    out2 = _norm_szse(szse, "20260820")
    assert out2.iloc[0]["symbol"] == "000001.SZ"
    assert out2.iloc[0][FIN_BALANCE] == pytest.approx(3.0e9)  # 融资余额, not 买入额
    assert out2.iloc[0][FIN_BUY] == pytest.approx(2.0e8)
    assert out2.iloc[0][SL_BALANCE] == pytest.approx(1.5e7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
