"""Real-time intraday trader ↔ close-run merge semantics.

The D-track live trader (``src/live/trader.py``) appends minute-precision SELL
fills to the ledger while the market is open; the 17:30 close run must (1) merge
those fills instead of replaying the day (the intraday sweep is disabled for
dates >= ``live_intraday_from``) and (2) never drop a live fill appended WHILE
the close run itself is processing (the ``protect_after`` seq watermark).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger
from src.paper.runner import PaperRunner


def _market(n_symbols=8, days=40, seed=7):
    return make_synthetic_market(symbols=n_symbols, days=days, seed=seed)


class _StaticPortfolio:
    def __init__(self, symbols):
        self.weights = {symbols[0]: 0.6, symbols[1]: 0.4}

    def compute_weights(self, symbols, date):
        return {s: w for s, w in self.weights.items() if s in symbols}


def _runner(portfolio, market, db, **kw):
    kw.setdefault("symbols", sorted(market.price_panel.columns))
    kw.setdefault("cash", 100_000.0)
    kw.setdefault("max_position_pct", 1.0)
    kw.setdefault("rebalance_days", 1)
    kw.setdefault("pit_strict", True)
    return PaperRunner(portfolio, market, PaperLedger(db), **kw)


# --------------------------------------------------------------------------- #
# ledger — live fill append + time precision
# --------------------------------------------------------------------------- #


def test_live_fill_append_roundtrip(tmp_path):
    led = PaperLedger(tmp_path / "l.sqlite")
    fill = Fill("2026-09-04", "A01", "sell", -100.0, 12.34, 1.23, 1234.0,
                time="14:03:25")
    led.append_fill(fill)
    got = led.fills_for_date("2026-09-04")
    assert len(got) == 1
    assert got[0].symbol == "A01"
    assert got[0].side == "sell"
    assert got[0].shares == pytest.approx(-100.0)
    assert got[0].time == "14:03:25"  # minute precision survives the round trip
    assert led.max_fill_seq("2026-09-04") >= 1
    assert led.max_fill_seq("1999-01-01") == 0
    led.close()


def test_record_day_protects_mid_run_live_fill(tmp_path):
    """A live fill appended after the close run's snapshot must survive."""
    led = PaperLedger(tmp_path / "l.sqlite")
    d = "2026-09-04"
    close_fills = [Fill(d, "A01", "buy", 100.0, 10.0, 1.0, 1000.0)]
    led.record_day(d, cash=90_000.0, equity=100_000.0, positions={"A01": 100.0},
                   fills=close_fills, gross_exposure=1000.0)
    snapshot = led.max_fill_seq(d)  # the close run snapshots this watermark

    # the live trader exits A01 while the close run is still processing
    led.append_fill(Fill(d, "A01", "sell", -100.0, 10.50, 1.05, 1050.0,
                         time="14:59:07"))

    # the close run re-records the day (e.g. a manual mid-day re-run): its own
    # fill rows re-insert, but the live fill above the watermark is preserved
    led.record_day(d, cash=91_000.0, equity=101_000.0, positions={"A01": 100.0},
                   fills=close_fills, gross_exposure=1000.0, protect_after=snapshot)

    got = led.fills_for_date(d)
    times = sorted((f.side, f.time) for f in got)
    assert ("buy", "") in times          # re-inserted close fill
    assert ("sell", "14:59:07") in times  # live fill survived the delete
    assert len(got) == 2                 # no duplicates, nothing dropped
    led.close()


def test_record_day_legacy_behaviour_still_idempotent(tmp_path):
    """protect_after=None (direct API callers) deletes all rows for the date."""
    led = PaperLedger(tmp_path / "l.sqlite")
    d = "2026-09-04"
    led.record_day(d, 90_000.0, 100_000.0, {"A01": 100.0},
                   [Fill(d, "A01", "buy", 100.0, 10.0, 1.0, 1000.0)], 1000.0)
    led.record_day(d, 80_000.0, 99_000.0, {"A01": 50.0},
                   [Fill(d, "A01", "buy", 50.0, 10.0, 0.5, 500.0)], 500.0)
    assert len(led.fills_for_date(d)) == 1
    assert led.fills_for_date(d)[0].shares == pytest.approx(50.0)
    led.close()


# --------------------------------------------------------------------------- #
# runner — live-date gate + prior-fill merge
# --------------------------------------------------------------------------- #


class _SpyPortfolio(_StaticPortfolio):
    """Static book that counts intraday_sweep invocations."""

    def __init__(self, symbols):
        super().__init__(symbols)
        self.sweep_calls = 0
        self.live_intraday_from = None

    def intraday_exits(self, date):
        self.sweep_calls += 1
        return []


def test_runner_skips_intraday_sweep_on_live_dates(tmp_path):
    market = _market()
    syms = sorted(market.price_panel.columns)
    portfolio = _SpyPortfolio(syms)
    portfolio.live_intraday_from = str(market.price_panel.index[0].date())
    r = _runner(portfolio, market, tmp_path / "l.sqlite")
    r.run()
    # every processed date is >= live_intraday_from → the replay sweep must
    # never run (those exits were executed in real time by the live trader)
    assert portfolio.sweep_calls == 0
    r.ledger.close()


def test_runner_merges_prior_live_fill_on_resume(tmp_path):
    market = _market()
    syms = sorted(market.price_panel.columns)
    dates = sorted(market.price_panel.index)
    mid = dates[len(dates) // 2]
    db = tmp_path / "l.sqlite"

    r1 = _runner(_StaticPortfolio(syms), market, db)
    r1.run(end=mid)
    r1.ledger.close()

    # the live trader exited syms[0] on the morning of mid+1 (the first day the
    # resumed run processes): full liquidation, minute-precision timestamp
    next_day = dates[dates.index(mid) + 1]
    d = str(next_day.date())
    led = PaperLedger(db)
    held = led.latest_state()[2]
    assert syms[0] in held
    led.append_fill(Fill(d, syms[0], "sell", -held[syms[0]], 10.0, 1.0,
                         abs(held[syms[0]]) * 10.0, time="09:41:12"))
    led.close()

    r2 = _runner(_StaticPortfolio(syms), market, db)
    assert r2.run()["resumed"] is True
    rows = r2.ledger.fills_for_date(d)
    live = [f for f in rows if f.time == "09:41:12"]
    assert len(live) == 1
    assert live[0].symbol == syms[0]
    assert live[0].side == "sell"
    # the merged fill was applied to the executor state, so the day's final
    # book reflects the exit (the static close rebalance re-buys is fine —
    # the fill itself must simply be carried, never dropped or duplicated)
    assert sum(1 for f in rows if f.time == "09:41:12") == 1
    r2.ledger.close()


def test_runner_live_gate_uses_getattr_default(tmp_path):
    """Portfolios without the live attr (A/B/C tracks) keep replaying sweeps."""
    market = _market()
    syms = sorted(market.price_panel.columns)
    portfolio = _SpyPortfolio(syms)  # live_intraday_from = None explicitly set
    r = _runner(portfolio, market, tmp_path / "l.sqlite")
    r.run()
    assert portfolio.sweep_calls > 0
    r.ledger.close()
