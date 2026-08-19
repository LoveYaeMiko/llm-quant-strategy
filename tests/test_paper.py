"""Paper-trading layer — ledger persistence + the resumable daily runner.

Covers the three paper-trading gaps: (2) cross-day persistence (ledger round-
trip / resume), (1) the daily walk-forward loop (execution-aware PnL with
slippage/commission), and (3) the point-in-time discipline (no future-dated or
beyond-slippage fills; a rebalance sells dropped names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger
from src.paper.runner import PaperRunner


def _market(n_symbols=10, days=60, seed=0):
    return make_synthetic_market(symbols=n_symbols, days=days, seed=seed)


def _symbols(market):
    return sorted(market.price_panel.columns)


class _FakePortfolio:
    """Static book: long two names, short one, gross 1."""

    def __init__(self, symbols):
        self.weights = {symbols[0]: 0.5, symbols[1]: 0.3, symbols[2]: -0.2}

    def compute_weights(self, symbols, date):
        return {s: w for s, w in self.weights.items() if s in symbols}


class _SwitchingPortfolio:
    """Holds {A,B} then switches to {C,D} at ``switch`` (drop-sell test)."""

    def __init__(self, symbols, switch):
        self.symbols = symbols
        self.switch = pd.Timestamp(switch)

    def compute_weights(self, symbols, date):
        s = self.symbols
        if pd.Timestamp(date) < self.switch:
            return {s[0]: 0.5, s[1]: 0.5}
        return {s[2]: 0.5, s[3]: 0.5}


def _runner(portfolio, market, db, **kw):
    kw.setdefault("symbols", _symbols(market))
    kw.setdefault("cash", 100_000.0)
    kw.setdefault("max_position_pct", 1.0)  # isolate loop from cap logic
    kw.setdefault("rebalance_days", 1)
    kw.setdefault("pit_strict", True)
    return PaperRunner(portfolio, market, PaperLedger(db), **kw)


# ---------------------------------------------------------------------------
# ledger — persistence + resume
# ---------------------------------------------------------------------------


def test_ledger_roundtrip(tmp_path):
    led = PaperLedger(tmp_path / "l.sqlite")
    fills = [Fill("2024-01-02", "A01", "buy", 10.0, 50.0, 2.5, 500.0)]
    led.record_day("2024-01-02", cash=90_000.0, equity=100_500.0,
                   positions={"A01": 10.0}, fills=fills, gross_exposure=500.0)
    d, cash, pos = led.latest_state()
    assert d == "2024-01-02"
    assert cash == pytest.approx(90_000.0)
    assert pos == {"A01": 10.0}
    assert led.equity_curve().iloc[0] == pytest.approx(100_500.0)
    assert led.n_fills() == 1
    assert led.total_commission() == pytest.approx(2.5)
    led.close()


def test_ledger_resume_returns_latest_day(tmp_path):
    led = PaperLedger(tmp_path / "l.sqlite")
    led.record_day("2024-01-02", 90_000.0, 100_500.0, {"A01": 10.0}, [], 500.0)
    led.record_day("2024-01-03", 88_000.0, 99_000.0, {"A01": 5.0, "B02": 3.0}, [], 450.0)
    led.close()
    # reopen — latest state must be the second day, not a fresh empty book
    led2 = PaperLedger(tmp_path / "l.sqlite")
    d, cash, pos = led2.latest_state()
    assert d == "2024-01-03"
    assert cash == pytest.approx(88_000.0)
    assert pos == {"A01": 5.0, "B02": 3.0}
    assert len(led2.equity_curve()) == 2
    led2.close()


def test_ledger_record_day_is_idempotent(tmp_path):
    led = PaperLedger(tmp_path / "l.sqlite")
    led.record_day("2024-01-02", 90_000.0, 100_000.0, {"A01": 10.0}, [], 500.0)
    led.record_day("2024-01-02", 80_000.0, 99_000.0, {"A01": 1.0}, [], 100.0)
    assert led.equity_curve().iloc[0] == pytest.approx(99_000.0)  # overwritten
    assert len(led.equity_curve()) == 1
    led.close()


# ---------------------------------------------------------------------------
# runner — daily loop, PnL, determinism
# ---------------------------------------------------------------------------


def test_runner_end_to_end(tmp_path):
    market = _market()
    syms = _symbols(market)
    r = _runner(_FakePortfolio(syms), market, tmp_path / "l.sqlite")
    res = r.run()
    m = res["metrics"]
    # one return per day after the first; at least one fill (initial rebalance)
    assert m["n_days"] == market.n_days - 1
    assert m["n_fills"] >= 3  # 3 names bought/sold on the first rebalance
    assert len(res["equity"]) == market.n_days
    # final equity is cash + mark-to-market and is finite
    assert np.isfinite(m["final_equity"])
    r.ledger.close()


def test_runner_is_deterministic(tmp_path):
    market = _market()
    syms = _symbols(market)
    r1 = _runner(_FakePortfolio(syms), market, tmp_path / "a.sqlite")
    r2 = _runner(_FakePortfolio(syms), market, tmp_path / "b.sqlite")
    e1 = r1.run()["equity"]
    e2 = r2.run()["equity"]
    assert e1 == e2
    r1.ledger.close()
    r2.ledger.close()


def test_runner_resume_mid_way(tmp_path):
    market = _market()
    syms = _symbols(market)
    db = tmp_path / "l.sqlite"
    dates = sorted(market.price_panel.index)
    mid = dates[len(dates) // 2]

    r1 = _runner(_FakePortfolio(syms), market, db)
    r1.run(end=mid)
    n_first = len(r1.ledger.equity_curve())
    r1.ledger.close()

    # resume from a fresh runner/ledger against the same db — must pick up at
    # the day after ``mid``, not restart from zero.
    r2 = _runner(_FakePortfolio(syms), market, db)
    res2 = r2.run()
    assert res2["resumed"] is True
    n_total = len(r2.ledger.equity_curve())
    r2.ledger.close()
    assert n_total == market.n_days
    assert n_first < n_total


def test_runner_resume_metrics_use_full_curve(tmp_path):
    # regression: on a resumed run the metrics were computed from the *incremental*
    # returns dict (only the days after the resume point), so n_days collapsed to
    # ~1 and total_return to ~0 while equity had actually moved. ``ret`` must be
    # derived from the full equity curve so a resumed run reports the same metrics
    # as a single continuous run.
    market = _market()
    syms = _symbols(market)
    dates = sorted(market.price_panel.index)
    mid = dates[len(dates) // 2]

    m_full = _runner(_FakePortfolio(syms), market, tmp_path / "full.sqlite").run()["metrics"]

    db = tmp_path / "l.sqlite"
    r1 = _runner(_FakePortfolio(syms), market, db)
    r1.run(end=mid)
    r1.ledger.close()
    m_resumed = _runner(_FakePortfolio(syms), market, db).run()["metrics"]

    assert m_resumed["n_days"] == m_full["n_days"] == market.n_days - 1
    assert m_resumed["total_return"] == pytest.approx(m_full["total_return"])
    assert m_resumed["final_equity"] == pytest.approx(m_full["final_equity"])


def test_runner_sells_dropped_names(tmp_path):
    market = _market(days=40, seed=1)
    syms = _symbols(market)
    dates = sorted(market.price_panel.index)
    switch = dates[len(dates) // 2]
    r = _runner(_SwitchingPortfolio(syms, switch), market, tmp_path / "l.sqlite")
    r.run()
    _, cash, positions = r.ledger.latest_state()
    # after the switch the book holds {C, D}, and {A, B} were sold to zero
    assert syms[0] not in positions and syms[1] not in positions
    assert positions.get(syms[2], 0.0) != 0 and positions.get(syms[3], 0.0) != 0
    r.ledger.close()


def test_runner_rebalance_frequency(tmp_path):
    market = _market(days=30, seed=2)
    syms = _symbols(market)
    r = _runner(_FakePortfolio(syms), market, tmp_path / "l.sqlite", rebalance_days=5)
    r.run()
    states = r.ledger.daily_states()
    # fills only on days 0, 5, 10, ... (the 5-day rebalance cadence)
    fill_days = states[states["n_fills"] > 0]["date"].tolist()
    assert len(fill_days) < market.n_days
    # every fill day is 5 trading days apart
    idx = states[states["n_fills"] > 0].index.tolist()
    gaps = np.diff(idx)
    assert (gaps == 5).all()
    r.ledger.close()


# ---------------------------------------------------------------------------
# point-in-time guard
# ---------------------------------------------------------------------------


def test_pit_guard_rejects_future_fill(tmp_path):
    market = _market()
    syms = _symbols(market)
    r = _runner(_FakePortfolio(syms), market, tmp_path / "l.sqlite")
    close = market.price_panel.iloc[0]
    # a fill dated after the loop day must raise under pit_strict
    bad = [Fill("2099-01-01", syms[0], "buy", 10.0, float(close[syms[0]]), 1.0, 500.0)]
    with pytest.raises(RuntimeError, match="PIT violation"):
        r._check_fills(pd.Timestamp(close.name), close, bad)
    r.ledger.close()


def test_pit_guard_rejects_out_of_band_price(tmp_path):
    market = _market()
    syms = _symbols(market)
    r = _runner(_FakePortfolio(syms), market, tmp_path / "l.sqlite")
    close = market.price_panel.iloc[0]
    d = str(pd.Timestamp(close.name).date())
    # price far above close + slippage must raise
    px = float(close[syms[0]])
    bad = [Fill(d, syms[0], "buy", 10.0, px * 10.0, 1.0, 500.0)]
    with pytest.raises(RuntimeError, match="PIT violation"):
        r._check_fills(pd.Timestamp(close.name), close, bad)
    r.ledger.close()


# ---------------------------------------------------------------------------
# integration — real AlphaCore book through the runner
# ---------------------------------------------------------------------------


def test_runner_drives_alpha_core(tmp_path):
    from src.factors.code_generator import FactorContext
    from src.portfolio.alpha_core import AlphaCore
    from src.portfolio.layer_integration import ThreeLayerPortfolio

    market = _market(n_symbols=20, days=90, seed=3)
    fctx = FactorContext(market.long)
    alpha = AlphaCore(fctx, ["Rank(Close)", "Neg(TS_Return(Close, 10))"],
                      long_pct=0.20, short_pct=0.20, max_position_pct=0.20,
                      neutralize=False)
    portfolio = ThreeLayerPortfolio(alpha)
    r = _runner(portfolio, market, tmp_path / "l.sqlite", max_position_pct=0.20)
    res = r.run()
    assert res["metrics"]["n_fills"] > 0
    assert res["metrics"]["n_days"] > 0
    assert np.isfinite(res["metrics"]["final_equity"])
    r.ledger.close()
