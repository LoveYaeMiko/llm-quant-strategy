"""No-retroactive-trading invariants (landing item 4.2).

The user's standing rule: **日内操作必须实时，不允许对过去已知时点买卖**. Two
mechanisms enforce it and are pinned here:

1. ``pb_live_intraday_from`` — on and after that date the minute-bar intraday
   sweep must NOT replay stops (the real-time trader owns them); a fresh ledger
   with no live fills therefore records ZERO intraday fills on live dates.
2. ``protect_after`` — the close run's idempotent delete must never drop fills
   the real-time trader appended while the run was in flight.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger
from src.paper.pullback_book import PullbackParams, PullbackPortfolio
from src.paper.runner import PaperRunner


def _market(symbols=("000001.SZ",), days: int = 80):
    dates = pd.bdate_range("2026-01-01", periods=days)
    closes = pd.DataFrame({s: 10.0 for s in symbols}, index=dates)
    rows = [
        {"date": d, "symbol": s, "open": 10.0, "high": 10.1, "low": 9.9,
         "close": 10.0, "volume": 1_000_000.0}
        for d in dates for s in symbols
    ]
    long = pd.DataFrame(rows).set_index(["date", "symbol"])
    return type("M", (), {"price_panel": closes, "long": long})()


def _book(tmp_path, live_from: str | None, minute_provider=None):
    """Seed one open lot in ``ledger.sqlite`` and return ``(book, ledger)``.

    The seed fill goes into the SAME ledger the runner uses (as in production:
    the real-time trader and the close run share one file). No daily_state row
    is written, so the runner still processes the date instead of resuming past
    it, and its prior-fill merge restores the position.
    """
    params = PullbackParams(k=1, rank_source="momentum", rank_min=0.0, vol_shrink=False,
                            stop_trigger="close", stop_open_minutes=0)
    market = _market()
    path = str(tmp_path / "ledger.sqlite")
    seed = PaperLedger(path)
    # a lot entered at 10.0 with a ~2-3% stop; the minute bar crashes to 9.0
    seed.append_fill(Fill(date="2026-03-02", symbol="000001.SZ", side="buy",
                          shares=100.0, price=10.0, commission=5.0, notional=1000.0,
                          source="close"))
    seed.close()
    ledger = PaperLedger(path)
    book = PullbackPortfolio(market, params, symbols=["000001.SZ"], ledger=ledger,
                             minute_provider=minute_provider)
    book.live_intraday_from = live_from
    return book, ledger


def _crashing_minutes(d, s):
    # -5% from the previous close: below the 2.5% stop but well inside the 10%
    # limit band, so the limit-down guard (D-7 replay path) does not veto it.
    return pd.DataFrame({
        "timestamp": pd.to_datetime([f"{pd.Timestamp(d).date()} 10:00:00"]),
        "open": [9.5], "high": [9.5], "low": [9.5], "close": [9.5],
    })


def _limit_down_minutes(d, s):
    return pd.DataFrame({
        "timestamp": pd.to_datetime([f"{pd.Timestamp(d).date()} 10:00:00"]),
        "open": [9.0], "high": [9.0], "low": [9.0], "close": [9.0],
    })


def test_live_date_is_never_replayed(tmp_path):
    book, ledger = _book(tmp_path, live_from="2026-03-02", minute_provider=_crashing_minutes)
    runner = PaperRunner(book, _market(), ledger, symbols=["000001.SZ"], cash=1000.0,
                         rebalance_days=1, pit_strict=False, seed=1)
    # no preclose provider → the historical close path applies, but the intraday
    # sweep must be skipped for the live date
    runner.preclose_provider = None
    runner.run(start="2026-03-02", end="2026-03-02")
    fills = ledger.fills()
    intraday = fills[fills["time"].fillna("").astype(str) != ""] if len(fills) else fills
    assert len(intraday) == 0
    ledger.close()


def test_pre_live_date_is_replayed(tmp_path):
    book, ledger = _book(tmp_path, live_from="2026-03-10", minute_provider=_crashing_minutes)
    runner = PaperRunner(book, _market(), ledger, symbols=["000001.SZ"], cash=1000.0,
                         rebalance_days=1, pit_strict=False, seed=1)
    runner.preclose_provider = None
    runner.run(start="2026-03-02", end="2026-03-02")
    fills = ledger.fills()
    intraday = fills[fills["time"].fillna("").astype(str) != ""] if len(fills) else fills
    assert len(intraday) == 1
    assert intraday.iloc[0]["source"] == "replay"
    ledger.close()


def test_limit_down_bar_is_not_replayed_as_a_fill(tmp_path):
    """A locked limit-down print has no counterparty — the stop stays pending."""
    book, ledger = _book(tmp_path, live_from="2026-03-10",
                         minute_provider=_limit_down_minutes)
    runner = PaperRunner(book, _market(), ledger, symbols=["000001.SZ"], cash=1000.0,
                         rebalance_days=1, pit_strict=False, seed=1)
    runner.preclose_provider = None
    runner.run(start="2026-03-02", end="2026-03-02")
    fills = ledger.fills()
    intraday = fills[fills["time"].fillna("").astype(str) != ""] if len(fills) else fills
    assert len(intraday) == 0
    ledger.close()


def test_protect_after_never_drops_concurrent_live_fills(tmp_path):
    path = tmp_path / "ledger.sqlite"
    led = PaperLedger(str(path))
    # 1. the live trader appended a fill during the session
    led.append_fill(Fill(date="2026-03-02", symbol="000001.SZ", side="sell", shares=-100.0,
                         price=9.5, commission=5.0, notional=950.0, time="10:00:00",
                         source="live"))
    # 2. the close run snapshots the watermark and merges what it can see
    watermark = led.max_fill_seq("2026-03-02")
    prior = led.fills_for_date("2026-03-02")
    # 3. another live fill lands WHILE the close run is processing (higher seq)
    led.append_fill(Fill(date="2026-03-02", symbol="000003.SZ", side="sell", shares=-100.0,
                         price=7.5, commission=5.0, notional=750.0, time="14:55:00",
                         source="live"))
    # 4. the close run records the day: its own fill + the merged prior fills
    led.record_day(
        "2026-03-02", 950.0, 950.0, {}, [
            *prior,
            Fill(date="2026-03-02", symbol="000002.SZ", side="buy", shares=100.0,
                 price=5.0, commission=5.0, notional=500.0, source="close"),
        ], 500.0, protect_after=watermark,
    )
    fills = led.fills()
    led.close()
    # both the merged prior fill AND the concurrent one survive
    assert set(fills["symbol"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert fills[fills["symbol"] == "000001.SZ"].iloc[0]["source"] == "live"
    assert fills[fills["symbol"] == "000003.SZ"].iloc[0]["source"] == "live"
