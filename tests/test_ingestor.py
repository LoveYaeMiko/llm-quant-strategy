"""Ingestor tests — pipeline orchestration with adapters mocked."""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data.ingestion.convert import UNIVERSE
from src.data.ingestion.ingestor import Ingestor, resolve_research_universe


def _cfg(db_url: str, universe_dir: str) -> Config:
    return Config(
        {
            "data": {
                "pit_database_url": db_url,
                "alphafeed": {"api_key": "test-key", "batch_size": 100},
                "baostock": {"enabled": "true"},
                "akshare": {"enabled": "true"},
                "fundamentals": {"pilot_size": 50},
                "checks": {"survivorship_date": "2015-01-01"},
                "universe_dir": universe_dir,
            },
            "project": {"start_date": "2010-01-01"},
            "research": {
                "train_start": "2010-01-01", "train_end": "2019-12-31",
                "val_start": "2020-01-01", "val_end": "2021-12-31",
                "test_start": "2022-01-01", "test_end": "2025-12-31",
                "universe": "hs300_500",
                "decay": {"window_days": 90, "icir_threshold": 0.30},
            },
        }
    )


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


def test_ingest_prices_writes_records_and_summary(tmp_path, monkeypatch):
    ing = Ingestor(_cfg(f"sqlite:///{tmp_path}/pit.db", str(tmp_path / "univ")))
    monkeypatch.setattr(
        ing.primary, "fetch_klines",
        lambda chunk, start, end, adjust="none": _klines(),
    )
    monkeypatch.setattr(
        ing.primary, "fetch_ex_factors",
        lambda chunk: {"600519.SH": pd.DataFrame({"trade_date": ["2024-01-02"], "ex_factor": [1.0]})},
    )
    stats = ing.ingest(symbols=["600519.SH"])
    assert stats.bars == 2
    assert stats.symbols_ok == 1
    assert stats.coverage_days == 2
    assert stats.max_gap_days == 0  # contiguous trading days → no gap
    assert stats.min_date == pd.Timestamp("2024-01-02")
    s = stats.summary()
    assert "价格K线记录" in s and "幸存者自查" in s
    # store round-trips the bars
    q = ing.store.query("2024-01-02")
    assert set(q["symbol"]) == {"600519.SH"}


def test_ingest_universe_writes_snapshots_and_caches(tmp_path, monkeypatch):
    ing = Ingestor(_cfg(f"sqlite:///{tmp_path}/pit.db", str(tmp_path / "univ")))
    monkeypatch.setattr(
        ing.baostock, "fetch_universe",
        lambda day: pd.DataFrame({"symbol": ["600519.SH"], "date": [pd.Timestamp(day)], "name": ["贵州茅台"]}),
    )
    monkeypatch.setattr(ing.baostock, "fetch_index_constituents", lambda name, date: ["600519.SH"])
    monkeypatch.setattr(ing.primary, "fetch_klines", lambda chunk, start, end, adjust="none": {})
    monkeypatch.setattr(ing.primary, "fetch_ex_factors", lambda chunk: {})
    stats = ing.ingest()
    assert stats.universe_records >= 1
    assert (tmp_path / "univ" / "hs300.json").is_file()
    assert (tmp_path / "univ" / "zz500.json").is_file()


def test_ingest_universe_backs_off_when_today_empty(tmp_path, monkeypatch):
    """baostock publishes no 'today' universe until the day's data is finalised —
    _ingest_universe must walk back to the most recent non-empty snapshot."""
    ing = Ingestor(_cfg(f"sqlite:///{tmp_path}/pit.db", str(tmp_path / "univ")))
    today = str(pd.Timestamp.today().normalize().date())
    prior = str((pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date())

    def fake_universe(day):
        if day == today:
            return pd.DataFrame()  # not finalised yet -> empty
        return pd.DataFrame({"symbol": ["600519.SH"], "date": [pd.Timestamp(day)], "name": ["贵州茅台"]})

    monkeypatch.setattr(ing.baostock, "fetch_universe", fake_universe)
    monkeypatch.setattr(ing.baostock, "fetch_index_constituents", lambda name, date: ["600519.SH"])
    stats = ing.ingest(universe_only=True)
    # survivorship anchor (2015-01-01) + walked-back prior day (both 1 symbol)
    assert stats.universe_records == 2
    assert len(ing.store.universe_as_of(prior, UNIVERSE)) == 1


def test_ingest_universe_only_skips_price_pass(tmp_path, monkeypatch):
    ing = Ingestor(_cfg(f"sqlite:///{tmp_path}/pit.db", str(tmp_path / "univ")))
    monkeypatch.setattr(
        ing.baostock, "fetch_universe",
        lambda day: pd.DataFrame({"symbol": ["600519.SH"], "date": [pd.Timestamp(day)], "name": ["贵州茅台"]}),
    )
    monkeypatch.setattr(ing.baostock, "fetch_index_constituents", lambda name, date: ["600519.SH"])

    def boom(*a, **k):  # the price pass must NOT run under universe_only
        raise AssertionError("price pass ran under --universe-only")

    monkeypatch.setattr(ing.primary, "fetch_klines", boom)
    monkeypatch.setattr(ing.primary, "fetch_ex_factors", boom)
    stats = ing.ingest(universe_only=True)
    assert stats.universe_records >= 1
    assert stats.bars == 0
    assert stats.symbols_ok == 0


def test_ingest_batch_failure_is_isolated(tmp_path, monkeypatch):
    ing = Ingestor(_cfg(f"sqlite:///{tmp_path}/pit.db", str(tmp_path / "univ")))
    monkeypatch.setattr(ing.primary, "fetch_klines", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ing.primary, "fetch_ex_factors", lambda chunk: {})
    stats = ing.ingest(symbols=["600519.SH"])
    assert stats.symbols_failed == 1
    assert stats.errors


def test_resolve_research_universe_literal():
    cfg = Config({"research": {"universe": "600519.SH, 000001.SZ"}, "data": {"universe_dir": "x"}})
    assert resolve_research_universe(cfg) == ["000001.SZ", "600519.SH"]


def test_resolve_research_universe_all_from_store(tmp_path):
    from src.data.point_in_time_loader import PointInTimeStore

    store = PointInTimeStore()
    store.upsert(
        pd.DataFrame(
            [
                {"symbol": "AAA", "valid_from": "2024-01-02", "valid_to": "2024-01-03", "close": 1.0, "record_type": "price"},
                {"symbol": "BBB", "valid_from": "2024-01-02", "valid_to": "2024-01-03", "close": 2.0, "record_type": "price"},
            ]
        )
    )
    cfg = Config({"research": {"universe": "all"}, "data": {"universe_dir": str(tmp_path)}})
    assert set(resolve_research_universe(cfg, store=store)) == {"AAA", "BBB"}
