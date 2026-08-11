"""Phase 9.1 — news/report ingestion tests (dedup, state persistence, PIT)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment.ingestion import NewsIngestor, ReportIngestor, _normalize_symbol


# ---------------------------------------------------------------------------
# symbol normalization
# ---------------------------------------------------------------------------


def test_normalize_symbol():
    assert _normalize_symbol("600519") == "600519.SH"
    assert _normalize_symbol("000001") == "000001.SZ"
    assert _normalize_symbol("300750") == "300750.SZ"
    assert _normalize_symbol("688981.SH") == "688981.SH"


# ---------------------------------------------------------------------------
# news ingestor — dedup + state persistence
# ---------------------------------------------------------------------------


def _fake_items():
    return [
        {"symbol": "600519.SH", "title": "茅台增长", "content": "业绩增长",
         "publish_time": "2026-08-11 09:00", "source": "em", "url": "http://a/1"},
        {"symbol": "000001.SZ", "title": "银行回调", "content": "股价下跌",
         "publish_time": "2026-08-11 10:00", "source": "em", "url": "http://a/2"},
    ]


def test_news_collect_dedup_and_state(tmp_path, monkeypatch):
    from src.sentiment import ingestion as ing

    calls = {"n": 0}

    def fake_fetch(symbol, pause=0.0):
        calls["n"] += 1
        items = []
        for it in _fake_items():
            if it["symbol"].startswith(symbol.split(".")[0]):
                items.append(ing.NewsItem(**it))
        return items

    monkeypatch.setattr(ing, "fetch_symbol_news", fake_fetch)
    ingestor = NewsIngestor(str(tmp_path))
    counts = ingestor.collect(["600519.SH", "000001.SZ"])
    assert calls["n"] == 2
    assert sum(counts.values()) == 2

    # state persisted
    state_file = tmp_path / "crawl_state.json"
    assert state_file.is_file()
    assert ingestor._state["last_run"]

    # second sweep — same URLs are already seen → 0 fresh
    counts2 = ingestor.collect(["600519.SH", "000001.SZ"])
    assert sum(counts2.values()) == 0
    assert calls["n"] == 4  # still fetched, but deduped on URL

    shard = tmp_path / "news_2026-08-11.parquet"
    assert shard.is_file()
    assert len(pd.read_parquet(shard)) == 2


def test_news_get_news_pit_filter(tmp_path, monkeypatch):
    from src.sentiment import ingestion as ing

    def fake_fetch(symbol, pause=0.0):
        return [ing.NewsItem(**it) for it in _fake_items()]

    monkeypatch.setattr(ing, "fetch_symbol_news", fake_fetch)
    ingestor = NewsIngestor(str(tmp_path))
    ingestor.collect(["600519.SH", "000001.SZ"])
    got = ingestor.get_news("600519.SH", "2026-08-11")
    assert len(got) == 1 and got[0].symbol == "600519.SH"
    assert ingestor.get_news("600519.SH", "2020-01-01") == []


# ---------------------------------------------------------------------------
# report ingestor — normalization + load
# ---------------------------------------------------------------------------


def test_report_normalize_and_load(tmp_path):
    df = pd.DataFrame(
        {
            "序号": [1, 2],
            "股票代码": ["600519", "600519"],
            "股票简称": ["贵州茅台", "贵州茅台"],
            "报告名称": ["业绩稳健增长，超预期", "风险提示"],
            "东财评级": ["买入", "持有"],
            "机构": ["中邮证券", "国泰君安"],
            "近一月个股研报数": [2, 2],
            "2026-盈利预测-收益": [67.0, 66.0],
            "2026-盈利预测-市盈率": [19.0, 18.0],
            "2027-盈利预测-收益": [69.0, 68.0],
            "2027-盈利预测-市盈率": [18.0, 17.0],
            "2028-盈利预测-收益": [73.0, 72.0],
            "2028-盈利预测-市盈率": [17.0, 16.0],
            "行业": ["白酒", "白酒"],
            "日期": ["2026-07-23", "2026-07-20"],
            "报告PDF链接": ["http://pdf/1", "http://pdf/2"],
        }
    )
    rows = ReportIngestor._normalize(df, "600519.SH")
    assert len(rows) == 2
    assert rows[0]["symbol"] == "600519.SH"
    assert rows[0]["title"] == "业绩稳健增长，超预期"
    assert rows[0]["rating"] == "买入"
    assert rows[0]["date"] == "2026-07-23"

    ingestor = ReportIngestor(str(tmp_path))
    ingestor._symbol_cache("600519.SH").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ingestor.load().columns).to_parquet(
        ingestor._symbol_cache("600519.SH"), index=False
    )
    all_df = ingestor.load()
    assert len(all_df) == 2
    sliced = ingestor.load(start="2026-07-22")
    assert len(sliced) == 1
