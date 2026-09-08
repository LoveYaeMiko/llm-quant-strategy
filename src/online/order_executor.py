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
    time: str = ""  # intraday fill timestamp (HH:MM), "" for close fills
    #: provenance (2026-09-08 audit, defect D-4): "live" = executed by the
    #: real-time trader at the actual print, "replay" = minute-bar intraday
    #: sweep, "close" = close rebalance, "auction" = pre-submitted 15:00
    #: closing-auction order list. Reporting must never mix live with replay.
    source: str = ""


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


def _board_lot(symbol: str) -> int:
    """Minimum board lot: STAR Market (688/689) trades 200-share min in 1-share
    increments above it; everywhere else it is 100-share multiples."""
    return 200 if symbol[:3] in ("688", "689") else 100


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
        notional_floor: float = 0.0,
        band_frac: float = 0.0,
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
        # cost governance: skip fills whose notional is below the floor (a
        # 5-yuan min commission on a dust fill is structurally ruinous at small
        # capital), and skip ADJUSTMENTS whose weight drift stays inside the
        # band (band rebalancing) — exits always execute.
        self.notional_floor = float(notional_floor)
        self.band_frac = float(band_frac)

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
        limit_locked: Optional[pd.Series] = None,
        source: str = "close",
    ) -> OrderResult:
        """Execute weight targets against a price panel.

        ``targets`` is ``date x symbol`` target weights (from the portfolio
        optimizer); ``prices`` is ``date x symbol`` close prices. Each row is a
        rebalance: deltas are turned into fills at the row's price, positions are
        capped, and cash is updated.

        A-share legality: buys must be whole 100-share board lots (odd lots are
        only allowed when closing a position entirely), fills land on the 0.01
        tick, and no fill may execute against a limit-locked bar — buying into a
        limit-up close / selling into a limit-down close is queue-uncertain and
        is deferred to the next rebalance (``limit_locked`` = the day's
        close-to-close returns per symbol).
        """
        result = OrderResult(positions=self.positions)
        equity = equity if equity is not None else self.cash
        if not np.isfinite(equity) or equity <= 0:
            equity = self.cash

        for date in targets.index:
            row = targets.loc[date].dropna()
            prices_row = prices.loc[date] if date in prices.index else prices.iloc[0]
            px = prices_row.reindex(row.index).fillna(0.0)
            locked = limit_locked
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
                # limit-lock legality — defer the trade to the next rebalance
                if locked is not None:
                    lv = locked.get(symbol, np.nan)
                    if np.isfinite(lv):
                        # board- AND date-aware band (主板 10%, 创业板 10% until
                        # 2020-08-24 then 20%, 科创板 20%, 北交所 30%) — one
                        # source of truth with the backtest mask.
                        from ..backtest.limit_locked import board_limit

                        lim = board_limit(symbol, pd.Timestamp(date), True)
                        if delta > 0 and lv >= lim - 0.005:
                            continue  # buying into a limit-up close
                        if delta < 0 and lv <= -(lim - 0.005):
                            continue  # selling into a limit-down close
                # board lot — STAR Market (688) is 200-share min in 1-share
                # increments; elsewhere whole 100-share lots, odd lots only when
                # closing out
                lot = _board_lot(symbol)
                if delta > 0:
                    if current < 0:
                        # covering a short: the cover itself may close entirely
                        # (odd lot OK), but any overshoot into a new long must be
                        # a whole lot
                        new_shares = current + delta
                        if new_shares >= 0:
                            if lot == 200:
                                long_delta = np.floor(new_shares) if new_shares >= lot else 0.0
                            else:
                                long_delta = np.floor(new_shares / lot) * lot
                            delta = -current + long_delta
                            if delta <= 0:
                                continue
                        else:
                            rounded_new = -np.ceil(abs(new_shares) / lot) * lot
                            if rounded_new <= current:
                                continue  # rounding would re-deepen the short
                            delta = rounded_new - current
                    else:
                        if lot == 200:
                            delta = np.floor(delta)
                            if delta < lot:
                                continue
                        else:
                            delta = np.floor(delta / lot) * lot
                            if delta <= 0:
                                continue
                elif delta < 0:
                    if current > 0:
                        new_shares = current + delta
                        if new_shares > 0:
                            if lot == 200:
                                if new_shares < lot:
                                    delta = -current  # can't keep <200 — close out
                                else:
                                    delta = np.floor(new_shares) - current
                                    if delta >= 0:
                                        continue
                            else:
                                rounded_new = np.ceil(new_shares / lot) * lot
                                if rounded_new >= current:
                                    continue  # nothing sellable after lot rounding
                                delta = rounded_new - current
                        else:
                            # flips through zero: close the long entirely (odd
                            # lot OK), the overshoot (a new short) must be whole
                            short_lots = np.floor(-new_shares / lot) * lot
                            delta = -current - short_lots
                            if delta >= 0:
                                continue
                    else:
                        # adding to a short — whole lots
                        delta = -np.floor(abs(delta) / lot) * lot
                        if delta >= 0:
                            continue
                # cost governance — exits always execute (closing is one fill);
                # entries/adjustments skip when too small to be worth the fees
                is_exit = abs(target_w) < 1e-12
                if not is_exit:
                    trade_notional = abs(delta) * price
                    if trade_notional < self.notional_floor:
                        continue
                    if self.band_frac > 0:
                        current_w = abs(current) * price / equity
                        if abs(target_w - current_w) < self.band_frac:
                            continue
                side = "buy" if delta > 0 else "sell"
                # fill on the 0.01 tick nearest the quoted price (tick
                # quantization; the PIT band check tolerates one tick)
                fill_price = float(round(self._quote(price, side), 2))
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
                        source=source,
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

    def execute_orders(
        self,
        orders: list[dict],
        prices: pd.Series,
        date,
        *,
        limit_locked: Optional[pd.Series] = None,
        fill_time: str = "15:00",
        source: str = "auction",
    ) -> OrderResult:
        """Execute a PRE-SUBMITTED closing-auction order list.

        ``orders`` = ``[{symbol, shares}]`` with SIGNED shares, decided at
        14:55 from 14:55-known data; fills happen at the 15:00 closing-auction
        (closing) prices with the same slippage/tick/fee model as the intraday
        executor. Limit-locked or suspended names do NOT fill (the order simply
        lapses — as in reality). Fill ``time`` records the auction timestamp.
        """
        result = OrderResult()
        for o in orders:
            symbol = str(o["symbol"])
            shares = float(o["shares"])
            if abs(shares) < 1e-9:
                continue
            price = float(prices.get(symbol, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue  # suspended — no auction print, order lapses
            side = "buy" if shares > 0 else "sell"
            if limit_locked is not None:
                lv = limit_locked.get(symbol, np.nan)
                if np.isfinite(lv):
                    # board- AND date-aware band — same source of truth as the
                    # backtest mask and the close executor above.
                    from ..backtest.limit_locked import board_limit

                    lim = board_limit(symbol, pd.Timestamp(date), True)
                    if shares > 0 and lv >= lim - 0.005:
                        continue  # buying into a limit-up close
                    if shares < 0 and lv <= -(lim - 0.005):
                        continue  # selling into a limit-down close
            fill_price = float(round(self._quote(price, side), 2))
            if fill_price <= 0:
                continue
            fee = self._fee(abs(shares) * fill_price, side)
            self.cash -= shares * fill_price + fee
            new_shares = self.positions.get(symbol, 0.0) + shares
            if abs(new_shares) < 1e-9:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = new_shares
            result.fills.append(
                Fill(
                    date=str(date),
                    symbol=symbol,
                    side=side,
                    shares=shares,
                    price=float(fill_price),
                    commission=float(fee),
                    notional=float(abs(shares) * fill_price),
                    time=str(fill_time),
                    source=source,
                )
            )
        result.cash = self.cash
        result.positions = dict(self.positions)
        return result
