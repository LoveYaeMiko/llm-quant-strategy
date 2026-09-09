"""Synthetic correlated market — offline demo / test data (Phase 1 fixture).

Generating a PIT panel + forward-return panel that look like a real market is
what lets the whole pipeline run without any vendor data:

* prices follow ``ret_{i,t} = beta_i * mkt_t + idio_{i,t}`` (common factor +
  idiosyncratic noise), so cross-sectional factors are *measurable*;
* the PIT store is built so a query at date ``d`` only ever sees bars born at
  or before ``d`` — the anti-look-ahead guarantee the checks probe;
* ``forward_returns[(d, s)]`` is the return realised over ``[d, d+1)``, which is
  exactly the horizon the backtest engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .point_in_time_loader import PointInTimeStore


@dataclass
class SyntheticMarket:
    records: pd.DataFrame          # PIT records (symbol, valid_from, valid_to, OHLCV)
    long: pd.DataFrame             # MultiIndex (date, symbol) panel with OHLCV
    price_panel: pd.DataFrame      # date x symbol closes
    forward_returns: pd.Series     # MultiIndex (date, symbol): ret over [d, d+1)
    pit_store: PointInTimeStore    # populated PIT store (price records only)
    n_symbols: int
    n_days: int
    # Price + universe records, for the B1-B5 checklist (B4 needs the
    # survivorship snapshots that the price-only pit_store lacks).
    audit_store: Optional[PointInTimeStore] = None
    # Forward returns with price-limit-locked bars masked to NaN (LIMIT_DOWN
    # blueprint 方案 B): IC keeps the raw ``forward_returns``, portfolio
    # Sharpe/max-drawdown use this. None on the synthetic market (no limits).
    forward_returns_tradable: Optional[pd.Series] = None
    # Basis contract (defect C2, docs/BASIS_CONTRACT.md): the report from
    # ``src.data.basis.basis_report`` describing which price basis each column of
    # ``long``/``price_panel`` is on. None for markets built without the read
    # layer (e.g. this synthetic fixture, whose OHLC is all on one basis).
    # Appended with a default so positional construction stays compatible.
    basis: Optional[dict] = None


def make_synthetic_market(
    symbols: int = 40,
    days: int = 504,
    seed: int = 1,
    start: str = "2019-01-01",
    idio_vol: float = 0.02,
    market_drift: float = 0.0004,
    market_vol: float = 0.010,
) -> SyntheticMarket:
    """Build a synthetic market with a persistent shared factor."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    n = len(dates)

    # common market factor + per-symbol beta / idio
    mkt = rng.normal(market_drift, market_vol, n)
    beta = rng.uniform(0.5, 1.5, symbols)
    idio = rng.normal(0.0, idio_vol, (symbols, n))

    rets = beta[:, None] * mkt[None, :] + idio
    names = [f"{chr(65 + i % 26)}{i + 1:02d}" for i in range(symbols)]
    closes = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=1).T,  # (days, symbols)
        index=dates,
        columns=names,
    )

    # derive OHLCV (intraday bars loosely consistent with close)
    ohlc = {}
    for col in names:
        c = closes[col]
        prev = c.shift(1).fillna(c.iloc[0])
        o = prev * (1 + rng.normal(0, 0.001))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
        v = rng.integers(1_000, 5_000_000, n).astype(float)
        ohlc[col] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    panel = pd.concat(
        {s: pd.DataFrame(ohlc[s], index=dates) for s in names}, axis=0
    )
    panel.index.names = ["symbol", "date"]  # level 0 = symbol, level 1 = date
    panel["amount"] = panel["close"] * panel["volume"]  # synthetic turnover
    panel = panel[["open", "high", "low", "close", "volume", "amount"]]

    # PIT records: bar at date d is a fact valid over [d, d+1 day)
    rec = panel.reset_index().copy()
    rec["valid_from"] = pd.to_datetime(rec["date"])
    rec["valid_to"] = rec["valid_from"] + pd.to_timedelta("1D")
    rec["symbol"] = rec["symbol"]
    rec = rec[["symbol", "valid_from", "valid_to", "open", "high", "low", "close", "volume"]]

    store = PointInTimeStore()
    store.upsert(rec)

    # forward returns: (d, s) -> return over [d, next trading day)
    fwd_wide = closes.pct_change().shift(-1)
    fwd = fwd_wide.stack().rename("fwd")
    fwd.index.names = ["date", "symbol"]

    # FactorContext expects a (date, symbol) panel
    long = panel.swaplevel().sort_index().rename_axis(["date", "symbol"])
    return SyntheticMarket(
        records=rec,
        long=long,
        price_panel=closes,
        forward_returns=fwd,
        pit_store=store,
        n_symbols=symbols,
        n_days=days,
    )
