"""Pullback trading book — a Martin Luk / Qullamaggie-style daily-bar system.

Translation of the strategy discussed (低胜率 + 极高盈亏比 + 严格止损 + 趋势跟踪)
onto the A-share daily-bar shadow loop, with the verified improvements baked in:

* market-environment filter — new entries only when the equal-weight 60d market
  trend is above ``entry_gate``; a full-book exit when it drops below
  ``exit_gate`` (the "stop trading" skill, made systematic);
* pullback entry — strong-momentum names (top cross-sectional rank of the 63d
  return) in an uptrend (close > rising EMA50) that pull back into the EMA
  zone below their recent high, with shrinking volume (seller exhaustion);
* ATR-adaptive stop — stop distance clipped to [2.5%, 4%] around
  ``atr_mult × ATR20``, so volatile names get room and calm names tight stops;
* asymmetric exits — breakeven stop after +1R, EMA(9) trailing exit after
  +1.5R, trend exit below EMA50, and a max-hold recycle — profits run, losses
  are cut fast;
* risk-scaled sizing — a fixed number of equal slots (``K``) so each position
  risks roughly ``1/K × stop`` ≈ 0.3-0.5% of equity (the 0.5%-risk discipline).

PIT discipline: every decision at date ``d`` uses data ≤ ``d``; the state
(entries/stops/peaks) is fully reconstructable from the ledger fills, so the
resumable daily shadow loop keeps working across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio.alpha_core import _market_trend


@dataclass
class _OpenLot:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    stop: float
    stop_dist: float
    peak: float
    trail_active: bool
    qty: float = 0.0
    cost: float = 0.0


@dataclass
class PullbackParams:
    """Tunable knobs of the pullback system (grid-tested for the D track)."""

    k: int = 8                    # equal slots (position = 1/k of equity each)
    rank_source: str = "momentum"  # "momentum" | "ml" (the scanner feeding the book)
    rank_min: float = 0.8         # top cross-sectional rank threshold (either source)
    mom_window: int = 63          # momentum window when rank_source = "momentum"
    mom_long_rank_min: float = 0.0  # optional 126d HQM-style quality filter (0 = off)
    bounce_confirm: bool = False  # require close > prev close on the entry day
    ema_fast: int = 9             # trailing-exit EMA
    ema_zone: int = 21            # pullback zone EMA
    zone_band: float = 0.02       # |close/EMA_zone - 1| <= zone_band
    pullback_min: float = 0.03    # at least 3% off the 10d high (a real pullback)
    vol_shrink: bool = True       # 5d volume < 20d volume (seller exhaustion)
    atr_mult: float = 1.5         # stop = clip(atr_mult*ATR20, lo, hi)
    stop_lo: float = 0.025
    stop_hi: float = 0.04
    breakeven_r: float = 1.0      # raise stop to entry after +R × stop_dist
    trail_r: float = 1.5          # activate EMA trailing after +R × stop_dist
    exit_into_strength_r: float = 0.0  # parabolic scale-out after +R (0 = off)
    max_hold: int = 40            # recycle positions after N trading days
    entry_gate: float = 0.0       # market 60d trend must exceed this to enter
    exit_gate: float = -0.03      # below this → liquidate the whole book
    trend_days: int = 60
    # intraday (AlphaFeed minute klines) enhancements — 0/False = off
    vwap_filter: float = 0.0      # entry requires |close/daily_VWAP - 1| <= this
    stop_rv: bool = False         # stop width = max(ATR20, RV20) — realized vol
    tail_vol_max: float = 0.0     # entry requires last-30min volume share <= this
    open30_max: float = 0.0       # entry requires open-30min return <= this (0=off)
    range_max: float = 0.0        # entry requires intraday range <= this (0=off)
    full_invest: bool = False     # size each open name 1/n — always fully invested


class PullbackPortfolio:
    """Stateful daily pullback book — ``compute_weights(symbols, date)``."""

    def __init__(
        self,
        market,
        params: PullbackParams | None = None,
        *,
        symbols: Optional[list[str]] = None,
        ledger=None,
        scores: Optional[pd.Series] = None,
        intraday: Optional[dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.p = params or PullbackParams()
        close_wide = market.price_panel
        long_ = market.long
        syms = list(symbols) if symbols else list(close_wide.columns)
        close_wide = close_wide.reindex(columns=syms)
        high_wide = long_["high"].unstack().reindex(columns=syms)
        low_wide = long_["low"].unstack().reindex(columns=syms)
        vol_wide = long_["volume"].unstack().reindex(columns=syms)

        self._close = close_wide
        self._prev_close = close_wide.shift(1)
        self._ema_fast = close_wide.ewm(span=self.p.ema_fast, adjust=False).mean()
        self._ema_zone = close_wide.ewm(span=self.p.ema_zone, adjust=False).mean()
        self._ema50 = close_wide.ewm(span=50, adjust=False).mean()
        self._ema50_rise = self._ema50 > self._ema50.shift(5)
        self._high10 = close_wide.rolling(10, min_periods=10).max()
        self._ret_mom = close_wide.pct_change(self.p.mom_window, fill_method=None)
        self._mom_rank = self._ret_mom.rank(axis=1, pct=True)
        self._ret_long = close_wide.pct_change(126, fill_method=None)
        self._mom_long_rank = self._ret_long.rank(axis=1, pct=True)

        # the "scanner": either raw momentum or an external ML artifact's
        # cross-sectional rank (the validated A-share strong-stock scanner)
        self._scores: pd.Series | None = None
        if self.p.rank_source == "ml":
            if scores is None:
                raise ValueError("rank_source='ml' requires a scores series (artifact predictions)")
            if isinstance(scores.index, pd.MultiIndex):
                wide = scores.unstack()
            else:
                wide = scores.reindex(close_wide.index)
            wide = wide.reindex(index=close_wide.index, columns=syms)
            self._scores = wide
            self._rank = wide.rank(axis=1, pct=True)
        else:
            self._rank = self._mom_rank

        # intraday enhancements (true VWAP / realized vol / tail volume)
        self._vwap_gap = None
        self._rv20 = None
        self._tail_vol = None
        self._open30 = None
        self._range = None
        if intraday:
            for key, frame in intraday.items():
                f = frame.reindex(index=close_wide.index, columns=syms)
                if key == "vwap_gap":
                    self._vwap_gap = f
                elif key == "rv":
                    self._rv20 = f.rolling(20, min_periods=10).mean()
                elif key == "tail_vol":
                    self._tail_vol = f
                elif key == "open30":
                    self._open30 = f
                elif key == "range":
                    self._range = f
        self._vol5 = vol_wide.rolling(5, min_periods=5).mean()
        self._vol20 = vol_wide.rolling(20, min_periods=20).mean()

        prev_close = close_wide.shift(1)
        tr = pd.concat(
            [
                high_wide - low_wide,
                (high_wide - prev_close).abs(),
                (low_wide - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self._atr20 = tr.rolling(20, min_periods=20).mean()
        self._atr_pct = self._atr20 / close_wide

        self._trend = _market_trend(close_wide, self.p.trend_days)

        self._dates = close_wide.index.sort_values()
        self._date_pos = {d: i for i, d in enumerate(self._dates)}
        self._open: dict[str, _OpenLot] = {}
        if ledger is not None:
            self._seed_from_ledger(ledger)

    # ------------------------------------------------------------------ seed
    def _seed_from_ledger(self, ledger) -> None:
        """Rebuild open lots from the ledger fills (entry vwap + stop state)."""
        fills = ledger.fills()
        if fills is None or len(fills) == 0:
            return
        cols = ["date", "symbol", "side", "shares", "price"]
        if "seq" in fills.columns:
            cols = ["date", "seq", "symbol", "side", "shares", "price"]
        fills = fills[cols].sort_values(cols[:2])
        lots: dict[str, _OpenLot] = {}
        for _, f in fills.iterrows():
            sym = str(f["symbol"])
            side = str(f["side"]).lower()
            d = pd.Timestamp(f["date"])
            shares = float(f["shares"])
            price = float(f["price"])
            if side == "buy":
                lot = lots.get(sym)
                if lot is None:
                    stop_dist = self._stop_dist(sym, d)
                    lots[sym] = _OpenLot(
                        symbol=sym, entry_date=d, entry_price=price,
                        stop=price * (1.0 - stop_dist), stop_dist=stop_dist,
                        peak=price, trail_active=False,
                        qty=shares, cost=shares * price,
                    )
                else:
                    lot.qty += shares
                    lot.cost += shares * price
                    lot.entry_price = lot.cost / lot.qty
                    lot.stop = lot.entry_price * (1.0 - lot.stop_dist)
            else:  # sell reduces / closes the lot
                lot = lots.get(sym)
                if lot is not None:
                    lot.qty -= shares
                    if lot.qty <= 0:
                        lots.pop(sym, None)
        for sym, lot in lots.items():
            hist = self._close.loc[lot.entry_date:, sym].dropna()
            if len(hist):
                lot.peak = float(hist.max())
                if lot.peak >= lot.entry_price * (1.0 + self.p.breakeven_r * lot.stop_dist):
                    lot.stop = max(lot.stop, lot.entry_price)
                if lot.peak >= lot.entry_price * (1.0 + self.p.trail_r * lot.stop_dist):
                    lot.trail_active = True
        self._open = lots

    def _stop_dist(self, symbol: str, d: pd.Timestamp) -> float:
        try:
            atr_pct = float(self._atr_pct.loc[d, symbol])
        except Exception:  # noqa: BLE001
            atr_pct = np.nan
        vol_pct = atr_pct
        if self.p.stop_rv and self._rv20 is not None:
            try:
                rv_pct = float(self._rv20.loc[d, symbol])
            except Exception:  # noqa: BLE001
                rv_pct = np.nan
            if np.isfinite(rv_pct) and rv_pct > 0:
                vol_pct = max(vol_pct, rv_pct) if np.isfinite(vol_pct) else rv_pct
        if not np.isfinite(vol_pct) or vol_pct <= 0:
            return self.p.stop_lo
        return float(np.clip(self.p.atr_mult * vol_pct, self.p.stop_lo, self.p.stop_hi))

    # ---------------------------------------------------------------- signals
    def _entry_candidates(self, d: pd.Timestamp) -> pd.DataFrame:
        """Names that pass trend/pullback/volume filters + the scanner rank at date d."""
        px = self._close.loc[d]
        ema_zone = self._ema_zone.loc[d]
        ema50 = self._ema50.loc[d]
        ema50_rise = self._ema50_rise.loc[d]
        high10 = self._high10.loc[d]
        rank = self._rank.loc[d]
        ret_mom = self._ret_mom.loc[d]
        vol5 = self._vol5.loc[d]
        vol20 = self._vol20.loc[d]

        zone = (px / ema_zone - 1.0).abs() <= self.p.zone_band
        uptrend = (px > ema50) & ema50_rise.fillna(False)
        pulled_back = px <= high10 * (1.0 - self.p.pullback_min)
        rank_ok = rank >= self.p.rank_min
        ret_ok = ret_mom > 0 if self.p.rank_source == "momentum" else pd.Series(True, index=px.index)
        mask = (zone & uptrend & pulled_back & rank_ok & ret_ok & px.notna()).fillna(False)
        if self.p.mom_long_rank_min > 0:
            mask &= (self._mom_long_rank.loc[d] >= self.p.mom_long_rank_min).fillna(False)
        if self.p.bounce_confirm:
            mask &= (px > self._prev_close.loc[d]).fillna(False)
        if self.p.vol_shrink:
            mask &= (vol5 < vol20).fillna(False)
        if self.p.vwap_filter > 0 and self._vwap_gap is not None:
            mask &= (self._vwap_gap.loc[d].abs() <= self.p.vwap_filter).fillna(False)
        if self.p.tail_vol_max > 0 and self._tail_vol is not None:
            mask &= (self._tail_vol.loc[d] <= self.p.tail_vol_max).fillna(False)
        if self.p.open30_max > 0 and self._open30 is not None:
            mask &= (self._open30.loc[d] <= self.p.open30_max).fillna(False)
        if self.p.range_max > 0 and self._range is not None:
            mask &= (self._range.loc[d] <= self.p.range_max).fillna(False)
        cand = pd.DataFrame(
            {
                "symbol": px.index[mask],
                "px": px[mask],
                "rank": rank[mask],
            }
        )
        return cand.sort_values("rank", ascending=False)

    # --------------------------------------------------------------- weights
    def compute_weights(self, symbols, date) -> dict[str, float]:
        d = pd.Timestamp(date)
        px_row = self._close.loc[d]

        if d in self._trend.index:
            regime = float(self._trend.loc[d])
        else:
            regime = np.nan

        # 1. process exits (stops / trails / trend / max-hold) at close[d]
        for sym in list(self._open):
            lot = self._open[sym]
            px = px_row.get(sym, np.nan)
            if not np.isfinite(px):
                continue  # suspended — keep the lot, no mark today
            lot.peak = max(lot.peak, float(px))
            held_bars = self._date_pos[d] - self._date_pos[lot.entry_date]
            reason = None
            if px <= lot.stop:
                reason = "stop"
            elif (
                self.p.exit_into_strength_r > 0
                and px >= lot.entry_price * (1.0 + self.p.exit_into_strength_r * lot.stop_dist)
                and px > float(self._ema_fast.loc[d, sym]) * 1.08
            ):
                reason = "strength"  # parabolic scale-out — sell into the extension
            elif lot.trail_active and px < float(self._ema_fast.loc[d, sym]):
                reason = "trail"
            elif px < float(self._ema50.loc[d, sym]):
                reason = "trend"
            elif held_bars >= self.p.max_hold:
                reason = "max_hold"
            if reason:
                self._open.pop(sym)
                continue
            if not lot.trail_active and px >= lot.entry_price * (1.0 + self.p.trail_r * lot.stop_dist):
                lot.trail_active = True
                lot.stop = max(lot.stop, lot.entry_price)
            elif px >= lot.entry_price * (1.0 + self.p.breakeven_r * lot.stop_dist):
                lot.stop = max(lot.stop, lot.entry_price)

        # 2. regime: hard exit below the gate
        if np.isfinite(regime) and regime < self.p.exit_gate:
            self._open.clear()
            return {}

        # 3. new entries (regime-gated, fill free slots)
        if np.isfinite(regime) and regime > self.p.entry_gate:
            free = self.p.k - len(self._open)
            if free > 0:
                cand = self._entry_candidates(d)
                held = set(self._open)
                for _, row in cand.iterrows():
                    if free <= 0:
                        break
                    sym = str(row["symbol"])
                    if sym in held:
                        continue
                    px = float(row["px"])
                    stop_dist = self._stop_dist(sym, d)
                    self._open[sym] = _OpenLot(
                        symbol=sym, entry_date=d, entry_price=px,
                        stop=px * (1.0 - stop_dist), stop_dist=stop_dist,
                        peak=px, trail_active=False,
                        qty=0.0, cost=0.0,
                    )
                    held.add(sym)
                    free -= 1

        if not self._open:
            return {}
        if self.p.full_invest:
            w = 1.0 / len(self._open)
        else:
            w = 1.0 / self.p.k
        out = {sym: w for sym in self._open}
        if symbols is not None:
            keep = set(symbols)
            out = {s: v for s, v in out.items() if s in keep}
        return out


__all__ = ["PullbackParams", "PullbackPortfolio"]
