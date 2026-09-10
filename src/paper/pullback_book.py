"""Pullback trading book — a Martin Luk / Qullamaggie-style daily-bar system.

Translation of the strategy discussed (低胜率 + 极高盈亏比 + 严格止损 + 趋势跟踪)
onto the A-share daily-bar shadow loop, with the verified improvements baked in:

* market-environment filter — new entries only when the equal-weight 60d market
  trend is above ``entry_gate``; a full-book exit when it drops below
  ``exit_gate`` (the "stop trading" skill, made systematic);
* pullback entry — strong-momentum names (top cross-sectional rank of the 63d
  return) in an uptrend (close > rising EMA50) that pull back into the EMA
  zone below their recent high, with shrinking volume (seller exhaustion);
* stop distance — ``clip(atr_mult × ATR20, stop_lo, stop_hi)``. The DEPLOYED
  configuration is a FLAT 3.5% (``stop_lo == stop_hi``, see
  :data:`FLAT_STOP_DEFAULT`); the ATR-adaptive band remains available for
  research but is no longer the default (it was selected under the collapsed
  true-range defect and lost the clean-window comparison);
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


#: The DEPLOYED D-track stop width: a FLAT 3.5%. With ``stop_lo == stop_hi`` the
#: ATR channel is inert (``clip(mult*ATR, lo, hi) == lo`` for every finite ATR and
#: for NaN), so this single number is the whole stop rule.
#:
#: Selection trail (2026-09-10, on the CORRECTED 800-name pool —
#: ``outputs/d_stop_grid_is_2026_800.json`` / ``d_stop_grid_oos_2025h2_800.json``):
#: flat_3p5 is the only variant with a positive Sharpe in BOTH windows and wins the
#: project's cross-window max-min rule (worst-window Sharpe 0.40 vs
#: atr_1p0_25_35 0.29, atr_1p0_25_40 −0.12, atr_1p5_25_40 −0.13). The numbers this
#: comment used to cite (+16.36% / Sharpe 1.66 / flat_2p5 −7.21%) came from a
#: 301-name cross-section and are VOID — see docs/D_TRACK_EVIDENCE.md §三/§六.
#:
#: This is the CODE default so that deleting ``pb_stop_lo``/``pb_stop_hi`` from the
#: YAML cannot silently revert the book to the unvalidated ATR-adaptive band.
FLAT_STOP_DEFAULT = 0.035


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
    atr_mult: float = 1.5         # stop = clip(atr_mult*ATR20, lo, hi); inert when lo == hi
    stop_lo: float = FLAT_STOP_DEFAULT   # deployed flat 3.5% (see FLAT_STOP_DEFAULT)
    stop_hi: float = FLAT_STOP_DEFAULT   # == stop_lo ⇒ ATR channel disabled by design
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
    # intraday stop execution (minute bars < 15:00)
    stop_trigger: str = "low"     # "low" = any wick breach | "close" = confirmed
    stop_buffer: float = 0.0      # trigger threshold = stop × (1 - buffer)
    stop_open_minutes: int = 0    # ignore the first N minutes (open-auction noise)


class PullbackPortfolio:
    """Stateful daily pullback book — ``compute_weights(symbols, date)``."""

    #: The daily close rebalance must run even when the book returns NO weights:
    #: an empty book (regime flatten / all lots exited) means "sell what is
    #: held", not "skip the rebalance". The runner pins held names at 0.0.
    always_rebalance = True

    def __init__(
        self,
        market,
        params: PullbackParams | None = None,
        *,
        symbols: Optional[list[str]] = None,
        ledger=None,
        scores: Optional[pd.Series] = None,
        intraday: Optional[dict[str, pd.DataFrame]] = None,
        minute_provider=None,
        scale_getter=None,
    ) -> None:
        self.p = params or PullbackParams()
        # Kill-switch integration (defect D-6, 2026-09-08 audit): the autopilot's
        # gross multiplier was never applied to this book — a `de_risk`/`halt`
        # decision had no effect on the D track. ``scale_getter`` returns the live
        # multiplier (1.0 normal, 0.5 de-risk, 0.0 halt); it may only SHRINK.
        self._scale_getter = scale_getter
        close_wide = market.price_panel
        long_ = market.long
        syms = list(symbols) if symbols else list(close_wide.columns)
        close_wide = close_wide.reindex(columns=syms)
        high_wide = long_["high"].unstack().reindex(columns=syms)
        low_wide = long_["low"].unstack().reindex(columns=syms)
        vol_wide = long_["volume"].unstack().reindex(columns=syms)

        # Basis consistency (defect D-8, 2026-09-08 audit): the PIT panel stores
        # RAW open/high/low but an ADJUSTMENT-SCALED close (ADR-0002:
        # close = raw_close × adjust_factor, factor anchored at the newest bar).
        # Mixing raw high/low with the adjusted close in the true range inflates
        # the ATR on every corporate-action bar (a 10:1 split makes high/close
        # ≈ 10), which then blows up the ATR-based stop distance. Scale high/low
        # onto the close's basis with the per-bar factor so TR is self-consistent.
        self._factor = self._adjust_factor_frame(market, syms, close_wide.index)
        high_wide = high_wide * self._factor
        low_wide = low_wide * self._factor

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
        # True range, ELEMENT-WISE over the three candidates. The previous
        # ``pd.concat([...], axis=1).max(axis=1)`` collapsed the (date × symbol)
        # frame to a per-DATE Series, so ``_atr20 / close_wide`` aligned a
        # date-indexed Series against symbol columns and every
        # ``_atr_pct.loc[d, symbol]`` lookup returned NaN — the ATR-adaptive
        # stop silently degraded to its 2.5% floor for the whole track
        # (defect D-8b, 2026-09-08 audit). np.maximum.reduce keeps the frame.
        tr = pd.DataFrame(
            np.maximum.reduce(
                [
                    (high_wide - low_wide).to_numpy(),
                    (high_wide - prev_close).abs().to_numpy(),
                    (low_wide - prev_close).abs().to_numpy(),
                ]
            ),
            index=close_wide.index,
            columns=close_wide.columns,
        )
        self._atr20 = tr.rolling(20, min_periods=20).mean()
        self._atr_pct = self._atr20 / close_wide

        self._trend = _market_trend(close_wide, self.p.trend_days)

        self._dates = close_wide.index.sort_values()
        self._date_pos = {d: i for i, d in enumerate(self._dates)}
        self._open: dict[str, _OpenLot] = {}
        self._minute_provider = minute_provider
        self._reentry_block: dict[str, pd.Timestamp] = {}
        self.live_intraday_from: str | None = None  # set by the shadow wiring
        if ledger is not None:
            self._seed_from_ledger(ledger)

    # ------------------------------------------------------------------ seed
    @staticmethod
    def _adjust_factor_frame(market, syms: list[str], index) -> pd.DataFrame:
        """``date × symbol`` backward-adjustment factor (1.0 when unavailable).

        Reads ``market.records`` (the PIT price records keep ``adjust_factor``;
        the flattened ``long`` panel drops it). Missing factors default to 1.0,
        which is exact for the newest bars (the factor is anchored at the newest
        bar) and for the synthetic offline market.
        """
        ones = pd.DataFrame(1.0, index=index, columns=syms)
        rec = getattr(market, "records", None)
        if rec is None or not hasattr(rec, "columns") or "adjust_factor" not in rec.columns:
            return ones
        if "date" in rec.columns:
            dates = pd.to_datetime(rec["date"])
        elif "valid_from" in rec.columns:
            dates = pd.to_datetime(rec["valid_from"])
        else:
            return ones
        frame = pd.DataFrame(
            {"date": dates.to_numpy(), "symbol": rec["symbol"].to_numpy(),
             "factor": pd.to_numeric(rec["adjust_factor"], errors="coerce").to_numpy()}
        ).dropna(subset=["factor"])
        if frame.empty:
            return ones
        wide = frame.pivot_table(index="date", columns="symbol", values="factor", aggfunc="last")
        return wide.reindex(index=index, columns=syms).ffill().fillna(1.0)

    def _basis_factor(self, d: pd.Timestamp) -> pd.Series:
        """Per-symbol backward-adjustment factor at date ``d`` (1.0 fallback).

        Used to build the ATR's consistent basis and by diagnostics/tests; the
        minute-bar stop path must NOT use it (those prints are already adjusted).
        """
        if self._factor is None or len(self._factor) == 0:
            return pd.Series(1.0, index=self._close.columns)
        if d in self._factor.index:
            return self._factor.loc[d].fillna(1.0)
        # suspended / non-trading date: use the last known factor
        prior = self._factor.loc[self._factor.index <= d]
        if len(prior) == 0:
            return pd.Series(1.0, index=self._close.columns)
        return prior.iloc[-1].fillna(1.0)

    def _seed_from_ledger(self, ledger) -> None:
        """Rebuild open lots from the ledger fills (entry vwap + stop state).

        Fills carry SIGNED shares (buys positive, sells negative) — the same
        moving-average-cost accumulation as ``enrich_positions``: a fill that
        reduces the position removes closed shares at the running avg cost, and
        a fully closed position leaves NO lot behind (a previous sign bug made
        sells ADD shares and resurrected closed positions as "ghost" lots that
        the close rebalance then bought back — fixed 2026-09-04).
        """
        fills = ledger.fills()
        if fills is None or len(fills) == 0:
            return
        cols = ["date", "symbol", "side", "shares", "price"]
        if "seq" in fills.columns:
            cols = ["date", "seq", "symbol", "side", "shares", "price"]
        fills = fills[cols].sort_values(cols[:2])

        signed: dict[str, float] = {}
        basis: dict[str, float] = {}
        entry_date: dict[str, pd.Timestamp] = {}
        for _, f in fills.iterrows():
            sym = str(f["symbol"])
            qty = float(f["shares"])
            px = float(f["price"])
            d = pd.Timestamp(f["date"])
            s = signed.get(sym, 0.0)
            b = basis.get(sym, 0.0)
            if s == 0.0:
                # open a fresh position (either direction) at the trade price
                signed[sym] = qty
                basis[sym] = qty * px
                entry_date[sym] = d
                continue
            avg = b / s
            if (qty > 0) == (s > 0):
                # same direction: add at the trade price
                signed[sym] = s + qty
                basis[sym] = b + qty * px
            else:
                # reducing/reversing: remove closed shares at current avg cost
                closing = min(abs(qty), abs(s))
                sign = 1.0 if s > 0 else -1.0
                new_b = b - sign * closing * avg
                new_s = s + qty
                remaining = abs(qty) - closing
                if remaining > 0:
                    # reversed through flat into the opposite side at trade price
                    open_sign = 1.0 if qty > 0 else -1.0
                    new_s = open_sign * remaining
                    new_b = open_sign * remaining * px
                    entry_date[sym] = d
                signed[sym] = new_s
                basis[sym] = new_b
                if abs(new_s) < 1e-9:
                    entry_date.pop(sym, None)

        lots: dict[str, _OpenLot] = {}
        for sym, qty in signed.items():
            if abs(qty) < 1e-9:
                continue
            avg = basis[sym] / qty
            d0 = entry_date.get(sym) or pd.Timestamp(fills.iloc[0]["date"])
            stop_dist = self._stop_dist(sym, d0)
            lots[sym] = _OpenLot(
                symbol=sym, entry_date=d0, entry_price=avg,
                stop=avg * (1.0 - stop_dist), stop_dist=stop_dist,
                peak=avg, trail_active=False,
                qty=qty, cost=basis[sym],
            )
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

    # ------------------------------------------------------- intraday exits
    def live_check(self, prices: dict[str, float], now: pd.Timestamp) -> list[dict]:
        """REAL-TIME stop check against current traded prices.

        A quote is a traded print — a print at/below the stop is a *confirmed*
        breach (no wick ambiguity), so the trigger semantics reduce to
        ``price <= stop × (1 - buffer)``, with the open-auction exemption and
        the same-day re-entry block applied. Executed by the live trader, never
        by the close replay.
        """
        if not self._open:
            return []
        t = now.time()
        if self.p.stop_open_minutes > 0:
            cut = (pd.Timestamp("09:30") + pd.Timedelta(minutes=self.p.stop_open_minutes)).time()
            if t < cut:
                return []
        exits = []
        for sym in list(self._open):
            lot = self._open[sym]
            px = prices.get(sym)
            if px is None or not np.isfinite(px):
                continue
            thr = lot.stop * (1.0 - self.p.stop_buffer)
            if float(px) <= thr:
                exits.append({"symbol": sym, "time": now.strftime("%H:%M:%S"), "price": float(px)})
                self._open.pop(sym)
                self._reentry_block[sym] = now.normalize()
        return exits

    def intraday_exits(self, date) -> list[dict]:
        """Stop breaches DURING the trading day, from minute bars (times < 15:00).

        Returns ``[{symbol, time, price}]`` — ``price`` is the stop level capped
        by the breaching bar's low. The lots are removed from the book and
        blocked from same-day re-entry (the close book then never re-buys them).
        """
        if self._minute_provider is None or not self._open:
            return []
        d = pd.Timestamp(date)
        # The minute cache is ALREADY adjustment-scaled (verified 2026-09-09:
        # AlphaFeed's minute endpoint returns 前复权 bars — for 000001.SZ on
        # 2026-01-05 the cache's last print is 11.1336, exactly the PIT adjusted
        # close, while the raw close was 11.50). So the prints are directly
        # comparable with the book's adjusted stop level — converting them again
        # would double-adjust (~3% too low) and fire stops on noise.
        exits = []
        for sym in list(self._open):
            lot = self._open[sym]
            bars = self._minute_provider(pd.Timestamp(date), sym)
            if bars is None or len(bars) == 0 or "low" not in bars.columns:
                continue
            if self.p.stop_open_minutes > 0:
                cut = pd.Timestamp("09:30") + pd.Timedelta(minutes=self.p.stop_open_minutes)
                bars = bars[bars["timestamp"].dt.time >= cut.time()]
            trigger_col = "close" if self.p.stop_trigger == "close" else "low"
            thr = lot.stop * (1.0 - self.p.stop_buffer)
            hit = bars[bars[trigger_col] <= thr]
            if hit.empty:
                continue
            # Limit-down legality (defect D-7 for the REPLAY path): a print at or
            # below the board's limit-down price has no bid behind it — the stop
            # stays pending (as the real-time trader already does). The check is
            # lookahead-free: it only compares the breaching bar with the
            # PREVIOUS close (both on the panel's adjusted basis).
            hit = self._drop_limit_down(hit, sym, d, trigger_col)
            if hit.empty:
                continue
            bar = hit.iloc[0]
            # Real-time semantics: the trader polls the LATEST print, so a
            # confirmed (close) trigger fills at the breaching minute's close
            # print — never at the bar low (that would be a stop-order fill at
            # a price the poller could not have traded). Matches live_check.
            price = min(float(lot.stop), float(bar[trigger_col]))
            ts = bar.get("timestamp")
            t_str = str(ts.time() if hasattr(ts, "time") else ts)
            exits.append({"symbol": sym, "time": t_str, "price": float(price)})
            self._open.pop(sym)
            self._reentry_block[sym] = pd.Timestamp(date)
        return exits

    # --------------------------------------------------------------- weights
    def _drop_limit_down(self, hit: pd.DataFrame, sym: str, d: pd.Timestamp, trigger_col: str) -> pd.DataFrame:
        """Drop breaching bars that sit at/below the limit-down price.

        Selling into a locked limit-down has no counterparty, so the stop stays
        pending. Uses only the bar and the PREVIOUS close (no lookahead).
        """
        from ..backtest.limit_locked import board_limit

        try:
            prev = float(self._prev_close.loc[d, sym])
        except (KeyError, TypeError):
            return hit
        if not np.isfinite(prev) or prev <= 0:
            return hit
        band = board_limit(sym, d, True) - 0.005
        ret = hit[trigger_col].astype(float) / prev - 1.0
        return hit[ret > -band]

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
                    blocked = self._reentry_block.get(sym)
                    if blocked is not None and blocked >= d:
                        continue  # intraday-exited today — no same-day re-entry
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
            out = {}
        elif self.p.full_invest:
            out = {sym: 1.0 / len(self._open) for sym in self._open}
        else:
            w = 1.0 / self.p.k
            out = {sym: w for sym in self._open}
        if symbols is not None:
            keep = set(symbols)
            out = {s: v for s, v in out.items() if s in keep}

        # 4. kill-switch gross multiplier — SHRINK ONLY (never lever up)
        scale = 1.0
        if self._scale_getter is not None:
            try:
                scale = float(self._scale_getter() or 0.0)
            except (TypeError, ValueError):
                scale = 0.0  # fail closed: an unreadable state must not trade
        if scale >= 1.0:
            return out
        if scale <= 0.0:
            # halt: flatten. Return 0.0 for every symbol the executor looks at
            # (the universe plus anything held) so positions are SOLD rather
            # than frozen at their last weights.
            base = list(symbols) if symbols is not None else list(self._open)
            return {s: 0.0 for s in dict.fromkeys([*base, *self._open])}
        return {s: v * scale for s, v in out.items()}


__all__ = ["PullbackParams", "PullbackPortfolio"]
