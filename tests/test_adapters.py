"""Adapter tests — pure logic with network/db mocked away."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter, to_price_records
from src.data.ingestion.baostock_adapter import BaostockAdapter
from src.data.ingestion.akshare_adapter import AkshareAdapter


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


def test_to_price_records_factor_one_is_identity():
    factors = {
        "600519.SH": pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03"], "ex_factor": [1.0, 1.0]})
    }
    out = to_price_records(_klines(), factors)
    assert list(out["close"]) == [10.0, 11.0]
    assert list(out["raw_close"]) == [10.0, 11.0]
    assert (out["adjust_factor"] == 1.0).all()


def test_to_price_records_applies_factor_backward():
    # 1:1 split ex-date 2024-01-02 (per-event factor 2.0): the pre-split 01-01
    # bar is halved (factor 0.5); the ex-date bar and later are already ex → 1.0.
    klines = {
        "600519.SH": pd.DataFrame(
            {"trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"], "close": [20.0, 10.0, 11.0]}
        )
    }
    factors = {"600519.SH": pd.DataFrame({"trade_date": ["2024-01-02"], "ex_factor": [2.0]})}
    out = to_price_records(klines, factors)
    assert list(out["adjust_factor"]) == [0.5, 1.0, 1.0]
    # adjusted series is continuous across the split: 10.0 → 10.0
    assert list(out["close"]) == [10.0, 10.0, 11.0]


def test_to_price_records_missing_factors_default_one():
    out = to_price_records(_klines(), None)
    assert (out["adjust_factor"] == 1.0).all()
    assert list(out["close"]) == [10.0, 11.0]


def test_to_price_records_multisymbol_factors_not_swapped():
    """Regression: each symbol must get its OWN ex_factor (a label/index
    misalignment once swapped factors between symbols in a >1-symbol batch)."""
    klines = {
        "600519.SH": pd.DataFrame(
            {"trade_date": ["2024-06-18", "2024-06-19"], "close": [200.0, 100.0]}
        ),
        "000858.SZ": pd.DataFrame(
            {"trade_date": ["2024-06-18", "2024-06-19"], "close": [120.0, 60.0]}
        ),
    }
    factors = {
        "600519.SH": pd.DataFrame({"trade_date": ["2024-06-19"], "ex_factor": [2.0]}),
        "000858.SZ": pd.DataFrame({"trade_date": ["2024-06-19"], "ex_factor": [3.0]}),
    }
    out = to_price_records(klines, factors).set_index(["symbol", "date"])
    # 600519: pre-split bar factor 1/2, ex-date bar 1.0
    assert out.loc["600519.SH", "adjust_factor"].tolist() == [0.5, 1.0]
    # 000858: pre-split bar factor 1/3, ex-date bar 1.0
    assert out.loc["000858.SZ", "adjust_factor"].tolist() == [1 / 3, 1.0]
    # close = raw * OWN symbol's factor (a swap would give 100.0/120.0 pre-bar)
    assert out.loc["600519.SH", "close"].iloc[0] == pytest.approx(200.0 * 0.5, rel=1e-4)
    assert out.loc["000858.SZ", "close"].iloc[0] == pytest.approx(120.0 * (1 / 3), rel=1e-4)


def _stub_bs(ad):
    """Provide a stand-in _bs so fetch_universe's fn lookup doesn't touch baostock."""
    import types

    ad._bs = types.SimpleNamespace(query_all_stock=None)


def test_baostock_fetch_universe_filters_index_and_bj(monkeypatch):
    ad = BaostockAdapter(enabled=True)
    _stub_bs(ad)
    df = pd.DataFrame(
        {
            "code": ["sh.600519", "sh.000001", "sz.000001", "bj.430047"],
            "code_name": ["贵州茅台", "上证指数", "平安银行", "北证股"],
            "type": ["1", "2", "1", "1"],
        }
    )
    monkeypatch.setattr(ad, "_query", lambda fn, **kw: df)
    out = ad.fetch_universe("2024-06-28")
    assert set(out["symbol"]) == {"600519.SH", "000001.SZ"}
    assert out["name"].tolist() == ["贵州茅台", "平安银行"]


def test_baostock_lazy_login_resolves_fn_after_login(monkeypatch):
    """Regression: _query must resolve the callable by NAME after login. Grabbing
    self._bs.<fn> at the call site crashes on a fresh adapter (lazy _bs is None)."""
    import types

    ad = BaostockAdapter(enabled=True)

    class FakeResult:
        error_code = "0"
        error_msg = ""
        fields = ["code", "code_name", "type"]

        def __init__(self, rows):
            self.rows = rows
            self.i = 0

        def next(self):
            if self.i < len(self.rows):
                return True
            return False

        def get_row_data(self):
            r = self.rows[self.i]
            self.i += 1
            return r

    fake = types.SimpleNamespace(
        fields=["code", "code_name", "type"],
        query_all_stock=lambda day: FakeResult(
            [["sh.600519", "贵州茅台", "1"], ["sh.000001", "上证指数", "2"]]
        ),
    )
    monkeypatch.setattr(ad, "_ensure_login", lambda: setattr(ad, "_bs", fake))
    out = ad.fetch_universe("2024-06-28")  # must NOT touch self._bs at call site
    assert set(out["symbol"]) == {"600519.SH"}


def test_baostock_fetch_universe_prefix_fallback(monkeypatch):
    # no `type` column -> index filter falls back to the EXCHANGE-AWARE prefix
    # heuristic: `sh.000001` (上证综指) is an index despite the 000 prefix (a
    # stock prefix on SZ); `sz.399001` (深证成指) has a non-stock prefix; B-shares
    # (900/200) are excluded; SZ 000xxx stocks survive.
    ad = BaostockAdapter(enabled=True)
    _stub_bs(ad)
    df = pd.DataFrame(
        {
            "code": ["sh.600519", "sh.000001", "sh.900901", "sz.000001", "sz.399001", "sz.200001"],
            "code_name": ["贵州茅台", "上证指数", "B股", "平安银行", "深证成指", "B股"],
        }
    )
    monkeypatch.setattr(ad, "_query", lambda fn, **kw: df)
    out = ad.fetch_universe("2024-06-28")
    assert set(out["symbol"]) == {"600519.SH", "000001.SZ"}


def test_akshare_fetch_news_maps_columns(monkeypatch):
    ad = AkshareAdapter(enabled=True)

    def fake_ak():
        return type("AK", (), {})()  # placeholder

    import types

    fake = types.SimpleNamespace(
        stock_news_em=lambda symbol: pd.DataFrame(
            {
                "发布时间": ["2024-01-05 09:30:00"],
                "新闻标题": ["标题"],
                "新闻内容": ["内容"],
                "文章来源": ["东方财富"],
            }
        )
    )
    monkeypatch.setattr(ad, "_ak", lambda: fake)
    out = ad.fetch_news("600519.SH", limit=10)
    assert out["symbol"].iloc[0] == "600519.SH"
    assert pd.Timestamp(out["date"].iloc[0]).date().isoformat() == "2024-01-05"


def test_akshare_fetch_news_disabled_returns_empty():
    ad = AkshareAdapter(enabled=False)
    assert ad.fetch_news("600519.SH").empty


def test_alphafeed_constructor_lazy_no_client():
    ad = AlphaFeedAdapter(api_key="test")
    assert ad._client is None


def test_fetch_ex_factors_normalizes_long_frame(monkeypatch):
    """Real ex_factors(to_dataframe=True) returns ONE long frame, not a dict."""
    import types

    ad = AlphaFeedAdapter(api_key="test")
    monkeypatch.setattr(
        ad, "_client",
        types.SimpleNamespace(
            klines=types.SimpleNamespace(
                ex_factors=lambda symbols, to_dataframe=False: pd.DataFrame(
                    {
                        "symbol": ["600519.SH", "000858.SZ", "600519.SH"],
                        "timestamp": [1, 2, 3],
                        "trade_date": ["2020-01-03", "2020-01-06", "2021-06-01"],
                        "ex_factor": [1.5, 2.0, 1.7],
                    }
                )
            )
        ),
    )
    out = ad.fetch_ex_factors(["600519.SH", "000858.SZ"])
    assert isinstance(out, dict)
    assert set(out) == {"600519.SH", "000858.SZ"}
    assert list(out["600519.SH"]["ex_factor"]) == [1.5, 1.7]
    assert list(out["000858.SZ"]["ex_factor"]) == [2.0]
