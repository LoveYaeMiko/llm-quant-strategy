"""Order executor — deterministic fills, slippage, commissions, caps (Phase 5).

The online execution layer must be *boring*: no ML, no LLM, no surprises. Given
a set of target positions and a price series, it:

* converts target weight deltas into share orders,
* applies a deterministic slippage model (bps * participation),
* deducts commission (bps, with a per-order minimum),
* enforces the ``max_position_pct`` gross cap and a symbol blacklist,
* returns a ledger of fills and the realised cash / position state.

All math is numpy/pandas; the result is reproducible bit-for-bit on identical
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Fill:
    date: str
    symbol: str
    side: str  # "buy" | "sell"
    shares: float
    price: float
    commission: float
    notional: float


@dataclass
class OrderResult:
    fills: list[Fill] = field(default_factory=list)
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_fills": len(self.fills),
            "cash": round(self.cash, 4),
            "gross_exposure": round(sum(abs(v) for v in self.positions.values()), 4),
            "positions": {k: round(v, 6) for k, v in self.positions.items()},
        }


class OrderExecutor:
    """Deterministic execution against a ``(date, symbol)`` price panel."""

    def __init__(
        self,
        *,
        slippage_bps: float = 2.0,
        commission_bps: float = 5.0,
        min_commission: float = 1.0,
        stamp_tax_sell_bps: float = 0.0,
        transfer_fee_bps: float = 0.0,
        max_position_pct: float = 0.05,
        blacklist: Optional[set[str]] = None,
        cash: float = 100_000.0,
        seed: int = 0,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        # Real A-share cost structure (§7): stamp tax 0.05% sell-only, transfer
        # fee 0.001% both sides. Default 0.0 keeps the legacy flat-commission
        # behaviour unchanged; the shadow/calibrate path injects real values.
        self.stamp_tax_sell_bps = stamp_tax_sell_bps
        self.transfer_fee_bps = transfer_fee_bps
        self.max_position_pct = max_position_pct
        self.blacklist = blacklist or set()
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.rng = np.random.default_rng(seed)  # deterministic per seed

    def restore(self, cash: float, positions: dict[str, float]) -> None:
        """Resume from a persisted account state (paper-trading ledger)."""
        self.cash = float(cash)
        self.positions = {k: float(v) for k, v in positions.items()}

    # -- order book ----------------------------------------------------------

    def _quote(self, price: float, side: str) -> float:
        """Buy at ask (up), sell at bid (down). Slippage is bps of price."""
        slip = price * self.slippage_bps / 10_000.0
        return price + slip if side == "buy" else price - slip

    def _fee(self, notional: float, side: str) -> float:
        """Total transaction cost on a fill: commission (min-capped) + transfer
        fee (both sides) + stamp tax (sell only). ``notional`` is unsigned. """
        commission = max(self.min_commission, notional * self.commission_bps / 10_000.0)
        transfer = notional * self.transfer_fee_bps / 10_000.0
        stamp = (notional * self.stamp_tax_sell_bps / 10_000.0) if side == "sell" else 0.0
        return commission + transfer + stamp

    def execute(
        self,
        targets: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        equity: Optional[float] = None,
    ) -> OrderResult:
        """Execute weight targets against a price panel.

        ``targets`` is ``date x symbol`` target weights (from the portfolio
        optimizer); ``prices`` is ``date x symbol`` close prices. Each row is a
        rebalance: deltas are turned into fills at the row's price, positions are
        capped, and cash is updated.
        """
        result = OrderResult(positions=self.positions)
        equity = equity if equity is not None else self.cash
        if not np.isfinite(equity) or equity <= 0:
            equity = self.cash

        for date in targets.index:
            row = targets.loc[date].dropna()
            prices_row = prices.loc[date] if date in prices.index else prices.iloc[0]
            px = prices_row.reindex(row.index).fillna(0.0)
            for symbol in row.index:
                if symbol in self.blacklist:
                    continue
                price = float(px.get(symbol, np.nan))
                if not np.isfinite(price) or price <= 0:
                    continue
                target_w = float(row[symbol])
                # position cap in weight terms vs current equity
                cap_shares = self.max_position_pct * equity / price
                current = self.positions.get(symbol, 0.0)
                target_shares = np.clip(target_w * equity / price, -cap_shares, cap_shares)
                delta = target_shares - current
                if abs(delta) < 1e-9:
                    continue
                side = "buy" if delta > 0 else "sell"
                fill_price = self._quote(price, side)
                fee = self._fee(abs(delta) * fill_price, side)
                self.cash -= delta * fill_price + fee
                new_shares = current + delta
                if abs(new_shares) < 1e-9:
                    self.positions.pop(symbol, None)  # fully closed — drop the name
                else:
                    self.positions[symbol] = new_shares
                result.fills.append(
                    Fill(
                        date=str(date),
                        symbol=symbol,
                        side=side,
                        shares=float(delta),
                        price=float(fill_price),
                        commission=float(fee),
                        notional=float(abs(delta) * fill_price),
                    )
                )
        result.cash = self.cash
        result.positions = dict(self.positions)
        return result

    def settle(self, prices: pd.Series) -> float:
        """Mark the book to market at ``prices`` and return total equity.

        A suspended name has no close on ``prices`` (the PIT panel keeps
        ``fill_method=None`` gaps) — skip it rather than letting a NaN propagate
        into cash via ``self.cash + NaN`` (SQLite then stores NaN as NULL and
        trips the ``daily_state.cash NOT NULL`` guard on ``record_day``).
        """
        mv = 0.0
        for s, shares in self.positions.items():
            px = prices.get(s, np.nan)
            if np.isfinite(px):
                mv += shares * float(px)
        return self.cash + mv
