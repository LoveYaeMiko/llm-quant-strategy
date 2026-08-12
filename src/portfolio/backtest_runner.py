"""Full-sample three-layer backtest runner (PHASE10 §3.5 / §四).

Two concerns live here:

* **lean market loader** — builds the (date, symbol) forward-return / OHLCV
  panels for a symbol list directly from the PIT Postgres store via a windowed
  SQL query, *not* ``store.snapshot("price")`` (which materialises all
  12.48M bars and blew past 32 GB RAM). 300 HS300 names × 2010-2025 is
  ~1.2M bars — a few hundred MB.
* **weight-driven backtester** — unlike the score-rank
  :class:`~src.backtest.engine.PointInTimeBacktest`, the three-layer portfolio's
  weights are produced by the layer stack, so the runner consumes explicit
  per-date weights and prices them against the forward-return panel:

      port_ret[d] = Σ_i  weights[d, i] · fwd[d, i]

  PIT holds by construction: a day's book only uses data visible at ``d``
  (alpha composite, PIT SUE, PIT sentiment), and ``fwd[d, i]`` is the return
  realised over ``[d, d+1)``.

The four blueprint scenarios (baseline / +risk / +tilt / three-layer) share one
``run_layered_backtest``; each scenario gets a *fresh* risk overlay so the
5-day freeze state never leaks across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.metrics import annualized_return, max_drawdown, sharpe_ratio, t_statistic, turnover
from .layer_integration import ThreeLayerPortfolio

# ---------------------------------------------------------------------------
# lean market loader
# ---------------------------------------------------------------------------


def load_hs300_market(
    config,
    symbols: list[str],
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> SimpleNamespace:
    """Price/forward panels for ``symbols`` directly from the PIT Postgres store.

    Returns a ``SimpleNamespace`` exposing ``long`` (date × symbol OHLCV),
    ``price_panel`` (date × symbol closes), ``forward_returns`` and
    ``forward_returns_tradable`` (limit-locked bars masked, LIMIT_DOWN 方案 B).
    """
    import psycopg2

    from ..backtest.limit_locked import tradeable_forward_returns
    from ..data.point_in_time_loader import PointInTimeStore

    url = config.get("data.pit_database_url")
    if not url:
        raise SystemExit("PIT_DATABASE_URL not set — cannot load the HS300 market")
    conn = psycopg2.connect(url)
    sql = """
        SELECT symbol, valid_from,
               payload->>'open', payload->>'high', payload->>'low',
               payload->>'close', payload->>'volume'
        FROM pit_records
        WHERE payload->>'record_type' = 'price'
          AND valid_from >= %s AND valid_from <= %s
          AND symbol = ANY(%s)
        """
    cur = conn.cursor()
    cur.execute(sql, (start, end, symbols))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        raise SystemExit(f"no price rows for {len(symbols)} symbols in [{start} .. {end}]")

    df = pd.DataFrame(
        rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    long = df.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]].sort_index()
    close_wide = long["close"].unstack()
    # fill_method=None: a forward return only exists between two *consecutive*
    # bars of the same name (no pad-fabricated suspension gaps).
    fwd = close_wide.pct_change(fill_method=None).shift(-1).stack().rename("fwd")

    market = SimpleNamespace(
        long=long,
        price_panel=close_wide,
        forward_returns=fwd,
        n_symbols=len(close_wide.columns),
        n_days=len(close_wide),
        limit_threshold=0.095,
        limit_dynamic=True,
    )
    market.forward_returns_tradable = tradeable_forward_returns(
        fwd, long, base_threshold=0.095, dynamic_threshold=True
    )
    return market


def _trading_dates(market, start=None, end=None) -> list[pd.Timestamp]:
    dates = sorted(market.forward_returns.index.get_level_values(0).unique())
    lo = pd.Timestamp(start) if start else None
    hi = pd.Timestamp(end) if end else None
    return [d for d in dates if (lo is None or d >= lo) and (hi is None or d <= hi)]


# ---------------------------------------------------------------------------
# weight-driven backtest
# ---------------------------------------------------------------------------


def run_layered_backtest(
    market,
    alpha,
    *,
    tilt=None,
    risk=None,
    symbols: Optional[list[str]] = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Price one layer composition over ``[start, end]`` and return metrics.

    A fresh ``ThreeLayerPortfolio`` is built here so a caller passing a stateful
    risk overlay gets an isolated run (freeze state does not leak).
    """
    dates = _trading_dates(market, start, end)
    if not dates:
        return {"error": "no trading dates in window", "metrics": {}, "weights": {}}
    pf = ThreeLayerPortfolio(alpha, tilt=tilt, risk=risk)
    if symbols is None:
        symbols = sorted(alpha.composite.index.get_level_values(1).unique().tolist())
    weights = pf.weights_frame(symbols, dates)

    forward = getattr(market, "forward_returns_tradable", None)
    if forward is None:
        forward = market.forward_returns
    fwd_wide = forward.unstack(fill_value=0.0)
    aligned = fwd_wide.reindex(index=weights.index, columns=weights.columns, fill_value=0.0)
    port_ret = (weights * aligned).sum(axis=1).sort_index()  # chronological

    ret = port_ret.dropna()
    metrics: dict = {}
    if ret.empty:
        metrics = {
            "sharpe": 0.0, "max_drawdown": 0.0, "annualized_return": 0.0,
            "t_stat": 0.0, "turnover": 0.0, "n_days": 0,
        }
    else:
        metrics = {
            "total_return": float((1.0 + ret).prod() - 1.0),
            "annualized_return": float(annualized_return(ret)),
            "sharpe": float(sharpe_ratio(ret)),
            "max_drawdown": float(max_drawdown(ret)),
            "t_stat": float(t_statistic(ret)),
            "turnover": float(turnover(weights)),
            "n_days": int(len(ret)),
            "effective_start": str(ret.index[0].date()),
            "effective_end": str(ret.index[-1].date()),
            "n_positions_avg": float(weights.gt(0).sum(axis=1).mean()),
        }
    return {"metrics": metrics, "weights": weights}


def check_gate(
    results: dict,
    sharpe_required: float = 1.6,
    max_drawdown_limit: float = 0.10,
) -> tuple[bool, list[str]]:
    """Phase 10 gate: Sharpe > 1.6 AND max drawdown < 10% (hard requirements)."""
    m = results.get("metrics", {})
    reasons: list[str] = []
    if m.get("sharpe", 0.0) < sharpe_required:
        reasons.append(f"Sharpe {m.get('sharpe', 0):.2f} < {sharpe_required}")
    if m.get("max_drawdown", 1.0) > max_drawdown_limit:
        reasons.append(f"maxDD {m.get('max_drawdown', 1.0):.1%} > {max_drawdown_limit:.0%}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# scenario sweep
# ---------------------------------------------------------------------------


def run_phase10_scenarios(
    market,
    alpha,
    pead=None,
    sentiment_panel: pd.Series | None = None,
    *,
    symbols: Optional[list[str]] = None,
    start: str | None = None,
    end: str | None = None,
    risk_kwargs: Optional[dict] = None,
) -> dict:
    """Run the four blueprint scenarios and return a comparison table + gate."""
    from .risk_overlay import SentimentRiskOverlay
    from .seasonal_tilt import PEADSeasonalTilt

    # the SUE percentile is measured over the whole universe (全市场分位)
    if symbols is None:
        symbols = sorted(alpha.composite.index.get_level_values(1).unique().tolist())
    tilt = PEADSeasonalTilt(pead, universe=symbols) if pead is not None else None
    rk = dict(risk_kwargs or {})

    def _fresh_risk():
        return SentimentRiskOverlay(sentiment_panel, **rk) if sentiment_panel is not None else None

    scenarios = {
        "baseline": {},
        "alpha_risk": {"risk": _fresh_risk},
        "alpha_tilt": {"tilt": tilt},
        "three_layer": {"tilt": tilt, "risk": _fresh_risk},
    }

    out: dict[str, dict] = {}
    table: list[dict] = []
    for name, spec in scenarios.items():
        tilt_arg = spec.get("tilt")
        risk_arg = spec["risk"]() if "risk" in spec else None
        res = run_layered_backtest(
            market, alpha, tilt=tilt_arg, risk=risk_arg,
            symbols=symbols, start=start, end=end,
        )
        m = res.get("metrics", {})
        passed, reasons = check_gate({"metrics": m})
        out[name] = {
            "metrics": m,
            "gate": {"passed": passed, "reasons": reasons},
            "risk_triggers": len(risk_arg.trigger_log()) if risk_arg is not None else 0,
        }
        table.append({
            "scenario": name,
            "sharpe": round(m.get("sharpe", 0.0), 3),
            "annualized_return": round(m.get("annualized_return", 0.0), 4),
            "max_drawdown": round(m.get("max_drawdown", 0.0), 4),
            "turnover": round(m.get("turnover", 0.0), 3),
            "n_days": m.get("n_days", 0),
            "effective_start": m.get("effective_start", ""),
            "risk_triggers": out[name]["risk_triggers"],
            "gate_passed": out[name]["gate"]["passed"],
        })
    return {"scenarios": out, "table": table, "weight_scheme": "equal_zscore",
            "long_pct": alpha.long_pct, "max_position_pct": alpha.max_position_pct}


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
