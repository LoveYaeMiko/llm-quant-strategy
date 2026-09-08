"""Regression tests for defect D-7 (2026-09-08 audit): live-trader guards.

The real-time trader must never book a fill it could not have got in reality:

* decisions stop at 15:00 (the closing auction belongs to the preclose layer,
  and polling to 15:10 could double-trade the same name);
* a print at the board-aware limit-down price has no bid behind it → no fill;
* a stale print (suspension / data lag) never decides.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.live.trader import LiveTrader, _in_trading_hours, _limit_pct


@pytest.mark.parametrize(
    "hhmm,expected",
    [
        ((9, 29), False),
        ((9, 30), True),
        ((11, 29), True),
        ((11, 30), False),
        ((12, 0), False),
        ((13, 0), True),
        ((14, 59), True),
        ((15, 0), False),
        ((15, 9), False),
    ],
)
def test_decision_window_ends_at_1500(hhmm, expected):
    assert _in_trading_hours(datetime(2026, 9, 8, hhmm[0], hhmm[1])) is expected


def test_limit_pct_is_board_aware():
    assert _limit_pct("300001.SZ") == 0.20
    assert _limit_pct("301001.SZ") == 0.20
    assert _limit_pct("688001.SH") == 0.20
    assert _limit_pct("600000.SH") == 0.10
    assert _limit_pct("000001.SZ") == 0.10


def _trader_stub(max_age_minutes: float = 5.0, limit_down: dict[str, float] | None = None) -> LiveTrader:
    """A LiveTrader with only the attributes the quote filter touches."""
    obj = LiveTrader.__new__(LiveTrader)
    obj.max_quote_age_minutes = max_age_minutes
    obj._quote_blocks = {}
    obj._quote_ts = {}
    obj._logged_blocks = set()
    obj._limit_down_price = lambda sym: (limit_down or {}).get(sym)  # type: ignore[method-assign]
    return obj


def test_stale_quote_does_not_decide():
    t = _trader_stub(max_age_minutes=5.0)
    now = datetime(2026, 9, 8, 10, 30, 0)
    raw = {
        "600000.SH": (10.0, pd.Timestamp("2026-09-08 10:29:00")),  # fresh
        "600001.SH": (10.0, pd.Timestamp("2026-09-08 09:40:00")),  # 50 min stale
    }
    live = t._filter_quotes(raw, now)
    assert set(live) == {"600000.SH"}
    assert "stale" in t._quote_blocks["600001.SH"]


def test_limit_down_print_is_not_sold():
    t = _trader_stub(limit_down={"600000.SH": 9.0})
    now = datetime(2026, 9, 8, 10, 30, 0)
    fresh = pd.Timestamp("2026-09-08 10:29:00")
    live = t._filter_quotes({"600000.SH": (9.0, fresh)}, now)
    assert live == {}                                  # stop stays pending
    assert "limit-down" in t._quote_blocks["600000.SH"]
    # One tick above the limit is tradeable again.
    live2 = t._filter_quotes({"600000.SH": (9.01, fresh)}, now)
    assert live2 == {"600000.SH": 9.01}


def test_missing_timestamp_is_allowed_but_invalid_print_is_not():
    t = _trader_stub()
    now = datetime(2026, 9, 8, 10, 30, 0)
    live = t._filter_quotes(
        {"600000.SH": (10.0, None), "600001.SH": (float("nan"), None)}, now
    )
    assert live == {"600000.SH": 10.0}
    assert t._quote_blocks["600001.SH"] == "invalid print"


def test_prev_close_and_limit_down_use_panel_last_row():
    t = LiveTrader.__new__(LiveTrader)
    t.portfolio = type("P", (), {})()
    t.portfolio._close = pd.DataFrame(
        [{"600000.SH": 10.0}, {"600000.SH": 12.0}],
        index=pd.to_datetime(["2026-09-04", "2026-09-07"]),
    )
    assert t._prev_close("600000.SH") == pytest.approx(12.0)
    assert t._limit_down_price("600000.SH") == pytest.approx(10.8)
    assert t._prev_close("999999.SZ") is None
    assert t._limit_down_price("999999.SZ") is None
