"""Call-site regression locks (audit A-3).

The fixes for D-1 (preclose could not sell exited names) and D-2 (kill-switch
never reached the pullback book) were only covered at the helper level: deleting
``merge_targets()`` from ``build_preclose_orders`` or the ``scale_getter`` wiring
from ``_build_account_portfolio`` left the whole suite green. These tests execute
the real call sites with the heavy I/O mocked.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import make_synthetic_market
from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger


def _tiny_market(symbols=("600000.SH", "000001.SZ"), days: int = 80):
    dates = pd.bdate_range("2026-01-01", periods=days)
    closes = pd.DataFrame({s: 10.0 + np.arange(days) * 0.01 for s in symbols}, index=dates)
    rows = [
        {"date": d, "symbol": s, "open": 10.0, "high": 10.2, "low": 9.8,
         "close": float(closes.loc[d, s]), "volume": 1_000_000.0}
        for d in dates for s in symbols
    ]
    long = pd.DataFrame(rows).set_index(["date", "symbol"])
    return type("M", (), {"price_panel": closes, "long": long})()


def test_build_account_portfolio_wires_the_kill_switch(monkeypatch):
    """D-2 call site: cli._build_account_portfolio must pass scale_getter."""
    from src.cli import _build_account_portfolio

    market = make_synthetic_market(symbols=12, days=120, seed=5)
    symbols = list(market.price_panel.columns)
    account = {
        "name": "D_5W", "alpha_source": "pullback", "cash": 50_000,
        "pb_rank_source": "momentum", "pb_k": 2, "pb_stop_lo": 0.035, "pb_stop_hi": 0.035,
        "pb_use_intraday": False, "pb_intraday_stops": False, "pb_full_invest": True,
        "pb_rank_min": 0.0, "pb_vol_shrink": False, "pb_pullback_min": 0.0,
        "pb_zone_band": 1.0, "pb_entry_gate": -1.0, "pb_exit_gate": -1.0,
    }
    book, _ = _build_account_portfolio(None, market, symbols, account, 1.0)
    assert book._scale_getter is not None, "kill-switch is not wired into the book"
    from src.cli import _book_fingerprint

    assert _book_fingerprint(book)["gross_scale_wired"] is True

    # halt (scale 0) must flatten: every symbol the executor sees maps to 0.0
    book_halt, _ = _build_account_portfolio(None, market, symbols, account, 0.0)
    d = book_halt._dates[-1]
    weights = book_halt.compute_weights(symbols, d)
    assert weights, "halt must return explicit zero weights, not an empty book"
    assert set(weights.values()) == {0.0}


def test_preclose_order_list_contains_exits_for_unwanted_holdings(monkeypatch, tmp_path):
    """D-1 call site: an exited holding must appear as a SELL in the 14:50 list."""
    from src import preclose as pc
    from src.paper.pullback_book import PullbackPortfolio

    # isolated ROOT with a seeded ledger holding 600000.SH
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pc, "ROOT", tmp_path)
    ledger = PaperLedger(str(tmp_path / "outputs" / "shadow_ledger_D_5W.sqlite"))
    # daily_state first (record_day clears that date's fills), then the fill
    ledger.record_day("2026-09-08", 0.0, 10_000.0, {"600000.SH": 1000.0}, [], 10_000.0)
    ledger.append_fill(Fill(date="2026-09-08", symbol="600000.SH", side="buy",
                            shares=1000.0, price=10.0, commission=5.0, notional=10_000.0,
                            source="close"))
    ledger.close()

    market = _tiny_market()
    today = pd.Timestamp.today().normalize()
    # today's provisional bar for the held name (and a candidate)
    prov = {
        "600000.SH": {"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e6},
        "000001.SZ": {"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e6},
    }
    # the deployed D config uses the ML scanner, so the scores series must be a
    # (date, symbol) MultiIndex (the book unstacks it)
    score_index = pd.MultiIndex.from_product(
        [market.price_panel.index, list(market.price_panel.columns)],
        names=["date", "symbol"],
    )
    monkeypatch.setattr(pc, "_today_provisional", lambda adapter, symbols, batch=25: (prov, {}))
    monkeypatch.setattr(
        pc, "_scores_as_of_yesterday", lambda *a, **k: pd.Series(0.5, index=score_index)
    )
    monkeypatch.setattr(pc, "_intraday_today", lambda *a, **k: {})
    monkeypatch.setattr("src.data.intraday.load_intraday_frames", lambda cfg, symbols: {})
    monkeypatch.setattr("src.cli._build_market_for_paper", lambda *a, **k: market)
    monkeypatch.setattr(PullbackPortfolio, "compute_weights", lambda self, symbols, date: {})

    class _Cfg:
        def section(self, name):
            return {"start_date": "2026-01-01"} if name == "shadow" else {}

        def get(self, key, default=None):
            return default

    account = {
        "name": "D_5W", "alpha_source": "pullback", "universe": "hs300",
        "pb_stop_lo": 0.035, "pb_stop_hi": 0.035, "pb_rank_source": "momentum",
        "max_position_pct": 0.4, "notional_floor": 2000.0, "band_frac": 0.0,
        "pb_use_intraday": False, "pb_intraday_stops": False,
    }
    out = pc.build_preclose_orders(_Cfg(), account, ["600000.SH", "000001.SZ"])
    payload = json.loads((tmp_path / "outputs" / "preclose_orders_D_5W.json").read_text(encoding="utf-8"))
    sells = [o for o in payload["orders"] if o["side"] == "sell"]
    assert out["ok"] is True
    assert any(o["symbol"] == "600000.SH" for o in sells), (
        f"exited holding missing from the order list: {payload['orders']}"
    )
