"""Paper-trading runner — the daily walk-forward loop (gaps 1 & 3).

The Phase 10 three-layer portfolio (:class:`~src.portfolio.layer_integration.
ThreeLayerPortfolio`) already prices a gate-passing backtest, but it does so
*weight-driven*: ``port_ret[d] = Σ w[d,i] · fwd[d,i]``, with no order, no
slippage, no commission, no cash account. :class:`PaperRunner` closes that gap by
advancing the clock one trading day at a time and, on each rebalance day,

    1. ``portfolio.compute_weights(symbols, d)``   — the PIT-clean book at ``d``;
    2. mark yesterday's book at ``close[d]``       — the realised return over
       ``[d-1, d)`` (no future bar is ever read);
    3. ``OrderExecutor.execute`` a single-day rebalance at ``close[d]``;
    4. ``ledger.record_day``                       — persist cash/equity/positions/fills.

This is gap (1) — the missing orchestrator that ties signal → optimise →
execute → PnL into a resumable daily loop — and gap (3) — the execution-side
point-in-time discipline: a fill happens at ``close[d]`` only, and a position is
marked at ``close[d]`` only, so nothing beyond the current day is ever accessed.
The signal/weight side is already PIT-clean by construction (AlphaCore lookbacks
are trailing; the sentiment overlay filters ``series.index <= t``; the PEAD tilt
reads ``pubDate <= as_of``); ``pit_strict`` turns that claim into a hard check on
every fill.

The runner is strategy-agnostic: it takes any object exposing
``compute_weights(symbols, date) -> {symbol: weight}``, so it drives the
three-layer portfolio in production and a fake alpha in tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.metrics import annualized_return, max_drawdown, sharpe_ratio, t_statistic
from ..online.order_executor import Fill, OrderExecutor
from .ledger import PaperLedger


class PaperRunner:
    """Resumable daily paper-trading loop over a weight source."""

    def __init__(
        self,
        portfolio,
        market,
        ledger: PaperLedger,
        *,
        symbols: Optional[list[str]] = None,
        cash: float = 100_000.0,
        slippage_bps: float = 2.0,
        commission_bps: float = 5.0,
        min_commission: float = 1.0,
        stamp_tax_sell_bps: float = 0.0,
        transfer_fee_bps: float = 0.0,
        max_position_pct: float = 0.05,
        blacklist: Optional[set[str]] = None,
        rebalance_days: int = 1,
        pit_strict: bool = True,
        seed: int = 0,
        notional_floor: float = 0.0,
        band_frac: float = 0.0,
    ) -> None:
        self.portfolio = portfolio
        self.market = market
        self.ledger = ledger
        self.symbols = list(symbols) if symbols else sorted(market.price_panel.columns)
        self.cash = float(cash)
        self.slippage_bps = float(slippage_bps)
        self.commission_bps = float(commission_bps)
        self.min_commission = float(min_commission)
        self.stamp_tax_sell_bps = float(stamp_tax_sell_bps)
        self.transfer_fee_bps = float(transfer_fee_bps)
        self.max_position_pct = float(max_position_pct)
        self.blacklist = blacklist or set()
        self.rebalance_days = int(max(1, rebalance_days))
        self.pit_strict = bool(pit_strict)
        self.seed = int(seed)
        self.notional_floor = float(notional_floor)
        self.band_frac = float(band_frac)

    # ------------------------------------------------------------------ helpers
    def _executor_kwargs(self) -> dict:
        return dict(
            cash=self.cash,
            slippage_bps=self.slippage_bps,
            commission_bps=self.commission_bps,
            min_commission=self.min_commission,
            stamp_tax_sell_bps=self.stamp_tax_sell_bps,
            transfer_fee_bps=self.transfer_fee_bps,
            max_position_pct=self.max_position_pct,
            blacklist=self.blacklist,
            seed=self.seed,
            notional_floor=self.notional_floor,
            band_frac=self.band_frac,
        )

    @staticmethod
    def _trading_dates(prices: pd.DataFrame, start, end) -> list[pd.Timestamp]:
        dates = sorted(pd.DatetimeIndex(prices.index.unique()).tolist())
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        return [d for d in dates if (lo is None or d >= lo) and (hi is None or d <= hi)]

    def _check_fills(self, date: pd.Timestamp, close: pd.Series, fills: list[Fill]) -> None:
        """PIT guard: no fill may be dated beyond the loop day or priced beyond
        ``close[d]`` ± slippage (a buy fills at ask, a sell at bid)."""
        d = str(pd.Timestamp(date).date())
        slip = self.slippage_bps / 10_000.0
        for f in fills:
            f_d = str(pd.Timestamp(f.date).date())
            if f_d != d:
                raise RuntimeError(
                    f"PIT violation: fill dated {f_d!r} != loop date {d!r}"
                )
            px = float(close.get(f.symbol, np.nan))
            if not np.isfinite(px):
                raise RuntimeError(f"PIT violation: no close for {f.symbol} on {d}")
            if not (px * (1.0 - slip) - 1e-9 <= f.price <= px * (1.0 + slip) + 1e-9):
                raise RuntimeError(
                    f"PIT violation: fill price {f.price:.4f} for {f.symbol} outside "
                    f"[{px * (1 - slip):.4f}, {px * (1 + slip):.4f}] on {d}"
                )

    # -------------------------------------------------------------------- run
    def run(self, start: str | None = None, end: str | None = None) -> dict:
        prices = self.market.price_panel
        dates = self._trading_dates(prices, start, end)
        if not dates:
            return {"error": "no trading dates in window", "metrics": {}, "equity": {}, "resumed": False}

        ex = OrderExecutor(**self._executor_kwargs())
        last_date, saved_cash, saved_positions = self.ledger.latest_state()
        resumed = last_date is not None
        if resumed:
            ex.restore(saved_cash or self.cash, saved_positions)

        start_idx = 0
        if resumed:
            last_ts = pd.Timestamp(last_date)
            start_idx = next((i for i, d in enumerate(dates) if d > last_ts), len(dates))
        for i in range(start_idx, len(dates)):
            d = dates[i]
            close = prices.loc[d]
            # realised return over [d-1, d): yesterday's book marked at today's
            # close — before any rebalance, so no future price is touched.
            mtm = ex.settle(close)

            fills: list[Fill] = []
            if (i - start_idx) % self.rebalance_days == 0:
                weights = self.portfolio.compute_weights(self.symbols, d)
                if weights:
                    # Explicit 0.0 for any name not in the book — both symbols
                    # dropped by the optimizer and positions still held from a
                    # shrunken universe (e.g. a halt target that only covers the
                    # current universe) — so the executor *sells* them rather
                    # than leaving them held forever.
                    held = set(ex.positions)
                    universe = list(
                        dict.fromkeys([*self.symbols, *sorted(held - set(self.symbols))])
                    )
                    targets = pd.DataFrame(
                        [{s: weights.get(s, 0.0) for s in universe}], index=[d]
                    )
                    res = ex.execute(targets, prices.loc[[d]], equity=mtm)
                    fills = res.fills
                    if self.pit_strict:
                        self._check_fills(d, close, fills)

            end_equity = ex.settle(close)
            gross = 0.0
            for s, sh in ex.positions.items():
                px = close.get(s, np.nan)
                if np.isfinite(px):
                    gross += abs(sh) * float(px)
            self.ledger.record_day(d, ex.cash, end_equity, dict(ex.positions), fills, gross)

        eq = self.ledger.equity_curve()
        ret = eq.pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)
        return {
            "metrics": self._metrics(eq, ret),
            "equity": {str(pd.Timestamp(k).date()): float(v) for k, v in eq.items()},
            "resumed": resumed,
        }

    def _metrics(self, eq: pd.Series, ret: pd.Series) -> dict:
        base: dict = {
            "n_days": int(len(ret)),
            "n_fills": self.ledger.n_fills(),
            "total_commission": round(self.ledger.total_commission(), 2),
        }
        if ret.empty or len(ret) < 2:
            return {**base, "total_return": 0.0, "annualized_return": 0.0,
                    "sharpe": 0.0, "max_drawdown": 0.0, "t_stat": 0.0}
        _, cash, _ = self.ledger.latest_state()
        return {
            **base,
            "total_return": float((1.0 + ret).prod() - 1.0),
            "annualized_return": float(annualized_return(ret)),
            "sharpe": float(sharpe_ratio(ret)),
            "max_drawdown": float(max_drawdown(ret)),
            "t_stat": float(t_statistic(ret)),
            "final_equity": float(eq.iloc[-1]),
            "final_cash": float(cash or 0.0),
        }
