"""``refresh_intraday_daily`` window resolution (code == docstring).

The intraday rollup must never include a PARTIAL current day: a half-day
``tail_vol`` would be read as a final one by the D-track entry gate. The day is
complete from the 15:00 close on, which is what the docstring promises and what
``_resolve_end_date`` now enforces.
"""

from __future__ import annotations

import pandas as pd

import src.data.intraday as idr


def test_before_close_excludes_today():
    """14:59 — the day is still trading, so the fetch must stop at yesterday."""
    now = pd.Timestamp("2026-09-08 14:59:00")
    assert idr._resolve_end_date(now, now) == pd.Timestamp("2026-09-07")


def test_at_close_includes_today():
    """15:00 — the close bar is in, the day is complete."""
    now = pd.Timestamp("2026-09-08 15:00:00")
    assert idr._resolve_end_date(now, now) == pd.Timestamp("2026-09-08")


def test_after_close_includes_today():
    """15:02 — the PAICC scheduler job (post-close) covers the current day."""
    now = pd.Timestamp("2026-09-08 15:02:00")
    assert idr._resolve_end_date(now, now) == pd.Timestamp("2026-09-08")


def test_explicit_past_date_is_always_included():
    """Self-heal for an earlier ``date``: that day is closed regardless of clock."""
    now = pd.Timestamp("2026-09-08 09:05:00")
    assert idr._resolve_end_date(now, "2026-09-07") == pd.Timestamp("2026-09-07")


def test_resolver_returns_normalized_timestamp():
    end = idr._resolve_end_date(pd.Timestamp("2026-09-08 15:30:12"), "2026-09-08")
    assert isinstance(end, pd.Timestamp)
    assert end == end.normalize()


def test_refresh_daily_forwards_resolved_end(monkeypatch):
    """``refresh_intraday_daily`` must fetch exactly through the resolved end."""
    seen: dict = {}

    def fake_refresh(cfg, symbols, start, end, batch: int = 25):
        seen.update(start=pd.Timestamp(start), end=pd.Timestamp(end), symbols=list(symbols))
        return {"calls": 0, "symbols_updated": 0}

    monkeypatch.setattr(idr, "refresh_intraday", fake_refresh)
    summary = idr.refresh_intraday_daily(object(), ["000001.SZ"], date="2026-09-08")

    assert summary["calls"] == 0
    assert seen["symbols"] == ["000001.SZ"]
    # 2026-09-08 is a Tuesday → the 5-day lookback starts on Thursday 09-03.
    assert seen["start"] == pd.Timestamp("2026-09-03")
    expected_end = idr._resolve_end_date(pd.Timestamp.now(), pd.Timestamp("2026-09-08"))
    assert seen["end"] == expected_end
    assert seen["end"] in (pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-07"))
