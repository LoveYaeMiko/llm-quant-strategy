"""Real-time intraday trader for the pullback (D) track.

Runs 09:30-15:00 on trading days, polling the current minute price of every
held symbol and executing stop breaches AT THE MOMENT they are traded — never
against past timestamps:

* prices come from the AlphaFeed minute endpoint's LATEST bar (a traded print);
* a print at/below the stop is a confirmed breach → sell immediately, record
  the fill with the actual minute timestamp (HH:MM:SS) into the ledger;
* per-position P&L is marked at the latest print every poll and written to
  ``outputs/live_<account>.json`` for the PAICC panel;
* the 15:10 close run then MERGES today's live fills (it never re-trades a
  past timestamp — the replay sweep is disabled for live dates via
  ``pb_live_intraday_from``).

Execution realism guards (defect D-7, 2026-09-08 audit):

* **decision window ends at 15:00** — the last intraday decision is taken on a
  print before 15:00; the 15:00 closing auction belongs to the preclose order
  layer, and polling past 15:00 could double-trade the same name;
* **limit-down prints do not fill** — a print at the (board-aware) limit-down
  price has no bid behind it, so the stop stays pending and is retried on the
  next poll instead of booking an impossible fill;
* **stale quotes do not decide** — a print older than ``max_quote_age_minutes``
  (suspension, data lag) is ignored for decisions and reported in the status
  file.

Compliance: only sells intraday (entries stay at the close), so T+1 is
satisfied by construction; each exit is a full liquidation (odd lots legal);
fees use the real cost model.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.ingestion.alphafeed_adapter import normalize_bar_timestamps
from ..online.order_executor import Fill


def _is_trading_day(d: pd.Timestamp) -> bool:
    return d.weekday() < 5


def _in_trading_hours(now: datetime) -> bool:
    """Decision window: 09:30-11:30 and 13:00-15:00 (exclusive of 15:00)."""
    t = (now.hour, now.minute)
    return (9, 30) <= t < (11, 30) or (13, 0) <= t < (15, 0)


def _limit_pct(symbol: str, date: pd.Timestamp | None = None) -> float:
    """Daily price limit via the shared, board- AND date-aware rule.

    Single source of truth (``limit_locked.board_limit``): 主板 10%, 创业板 10%
    until 2020-08-24 then 20%, 科创板 20%, 北交所 30%. The earlier local copy was
    date-blind and treated 北交所 as 10%, which would have mis-flagged a 10–30%
    drop on a BSE name as "limit-down, cannot sell" and skipped a real stop.
    """
    from ..backtest.limit_locked import board_limit

    return board_limit(str(symbol), pd.Timestamp(date or pd.Timestamp.today()), True)



def _is_live_process(pid: int) -> bool:
    """True when ``pid`` is an EXISTING live-trader process (not a recycled pid).

    A hard crash leaves a stale pid file; if the OS recycled the pid to an
    unrelated process, a bare ``os.kill(pid, 0)`` would wrongly treat the
    trader as running and skip the whole session. The command-line check is the
    primary gate; the os.kill probe only degrades gracefully when psutil is
    missing.
    """
    try:
        import psutil
    except ImportError:
        try:
            os.kill(pid, 0)  # noqa: S101 — existence probe
            return True
        except OSError:
            return False
    try:
        proc = psutil.Process(pid)
        cmd = " ".join(proc.cmdline()).lower()
        return "cli.py" in cmd and "live" in cmd.split()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


class LiveTrader:
    """Polls live prices and executes D-track stop breaches in real time."""

    def __init__(self, cfg, portfolio, ledger, account: dict, adapter) -> None:
        self.cfg = cfg
        # Deployment channel: today the trader only ever writes PAPER fills. A
        # future broker adapter must pass the same gate (src/deploy.py), so the
        # channel is resolved here and reported in the status file.
        from ..deploy import deployment_status

        self.deployment = deployment_status(cfg)
        self.portfolio = portfolio
        self.ledger = ledger
        self.account = account
        self.adapter = adapter
        self.account_name = str(account.get("name", "D_5W"))
        self.poll_seconds = int((cfg.section("live") or {}).get("poll_seconds", 60) or 60)
        self.slippage_bps = float(cfg.section("paper").get("slippage_bps", 2.0))
        self.min_commission = float(cfg.section("paper").get("min_commission", 5.0))
        self.commission_bps = float(cfg.section("paper").get("commission_bps", 2.5))
        self.stamp_bps = float(cfg.section("s7_calibration").get("cost_model", {}).get("stamp_tax_sell_bps", 5.0))
        self.transfer_bps = float(cfg.section("s7_calibration").get("cost_model", {}).get("transfer_fee_bps", 0.1))
        self.status_path = Path("outputs") / f"live_{self.account_name}.json"
        self.max_quote_age_minutes = float(
            (cfg.section("live") or {}).get("max_quote_age_minutes", 5) or 5
        )
        self._quote_blocks: dict[str, str] = {}
        self._logged_blocks: set[tuple[str, str]] = set()
        self._quote_ts: dict[str, pd.Timestamp] = {}
        # reconstruct the CURRENT state: last daily_state + today's live fills
        _, cash, positions = self.ledger.latest_state()
        self.cash = float(cash if cash is not None else float(account.get("cash", 50_000)))
        self.positions: dict[str, float] = {k: float(v) for k, v in positions.items()}
        today = pd.Timestamp.today().date()
        for f in self.ledger.fills_for_date(today):
            self.cash += f.notional - f.commission
            new_sh = self.positions.get(f.symbol, 0.0) + f.shares
            if abs(new_sh) < 1e-9:
                self.positions.pop(f.symbol, None)
            else:
                self.positions[f.symbol] = new_sh

    def _fee(self, notional: float) -> float:
        commission = max(self.min_commission, notional * self.commission_bps / 10_000.0)
        transfer = notional * self.transfer_bps / 10_000.0
        stamp = notional * self.stamp_bps / 10_000.0
        return commission + transfer + stamp

    def _current_prices(self) -> dict[str, tuple[float, pd.Timestamp | None]]:
        """Latest traded print + its minute timestamp per held symbol.

        The minute endpoint returns epoch milliseconds (UTC); the timestamp is
        normalised to naive Asia/Shanghai before being compared with the local
        clock — otherwise every print looks 8 hours stale and the freshness
        guard would block the entire session (found by scripts/live_smoke.py on
        the 2026-09-09 pre-open check).
        """
        if not self.positions:
            return {}
        batch = self.adapter.fetch_minute_klines(list(self.positions), period="1m", count=1)
        out: dict[str, tuple[float, pd.Timestamp | None]] = {}
        for sym, df in (batch or {}).items():
            if df is None or df.empty:
                continue
            df = normalize_bar_timestamps(df)
            px = float(df["close"].iloc[-1])
            ts: pd.Timestamp | None = None
            if "timestamp" in df.columns:
                try:
                    ts = pd.Timestamp(df["timestamp"].iloc[-1])
                except (ValueError, TypeError, OverflowError):
                    ts = None
            out[sym] = (px, ts)
        return out

    def _prev_close(self, symbol: str) -> float | None:
        """Last close in the book's daily panel (adjusted basis)."""
        try:
            series = self.portfolio._close[symbol].dropna()
        except (KeyError, AttributeError):
            return None
        if series is None or len(series) == 0:
            return None
        value = float(series.iloc[-1])
        return value if np.isfinite(value) and value > 0 else None

    def _limit_down_price(self, symbol: str, date: pd.Timestamp | None = None) -> float | None:
        """Board-aware limit-down price from the previous close.

        The panel's close is adjustment-scaled while prints are raw, so this is
        only used as a *tight* blocker: a print must sit at/below the computed
        limit to be refused, which a basis mismatch alone cannot produce in the
        direction that matters (px >> limit).
        """
        prev = self._prev_close(symbol)
        if prev is None:
            return None
        return round(prev * (1.0 - _limit_pct(symbol, date)), 2)

    def _filter_quotes(
        self, raw: dict[str, tuple[float, pd.Timestamp | None]], now: datetime
    ) -> dict[str, float]:
        """Drop stale quotes and limit-down prints from this decision cycle.

        FAIL-CLOSED: a print without a usable timestamp cannot be proven fresh,
        so it never decides. (The pre-fix code let it through — an API change or
        a malformed frame would then have traded on an unknown-age price.)
        """
        now_ts = pd.Timestamp(now)
        self._quote_blocks = {}
        self._quote_ts = {}
        live: dict[str, float] = {}
        max_age = self.max_quote_age_minutes * 60.0
        for sym, (px, ts) in raw.items():
            if not np.isfinite(px) or px <= 0:
                self._quote_blocks[sym] = "invalid print"
                continue
            if ts is None:
                self._quote_blocks[sym] = "no quote timestamp (no decision)"
                continue
            self._quote_ts[sym] = ts
            age = (now_ts - ts).total_seconds()
            if age > max_age:
                self._quote_blocks[sym] = f"quote stale {int(age // 60)}m (no decision)"
                continue
            lim = self._limit_down_price(sym, now_ts)
            if lim is not None and px <= lim + 1e-9:
                self._quote_blocks[sym] = f"limit-down {lim:.2f} — cannot sell"
                continue
            live[sym] = px
        for sym, why in self._quote_blocks.items():
            key = (sym, why.split(" ")[0])
            if key not in self._logged_blocks:
                self._logged_blocks.add(key)
                print(f"{now:%H:%M:%S} {sym} skipped: {why}", flush=True)
        return live


    def run(self) -> int:
        pid_file = Path("outputs") / f"live_{self.account_name}.pid"
        if pid_file.is_file():
            try:
                old = int(pid_file.read_text().strip())
                if _is_live_process(old):
                    print(f"live trader already running (pid {old}) — exiting")
                    return 0
            except (OSError, ValueError):
                pass
        pid_file.write_text(str(os.getpid()))
        # Clear yesterday's status file: the panel must never present the
        # previous session's P&L as today's live data (the trader writes the
        # first fresh snapshot on its first poll of the day).
        self.status_path.unlink(missing_ok=True)

        try:
            while True:
                now = datetime.now()
                d = pd.Timestamp(now.date())
                if not _is_trading_day(d):
                    # weekends/holidays (weekday check only): exit at once instead
                    # of sleeping forever — manual runs must terminate cleanly.
                    print(f"{now:%H:%M:%S} not a trading day — live trader exiting")
                    break
                if not _in_trading_hours(now):
                    if now.hour >= 15:
                        print(f"{now:%H:%M:%S} 15:00 — intraday decisions closed; "
                              "the closing auction is handled by the preclose layer")
                        break
                    time.sleep(max(20, self.poll_seconds))
                    continue

                # Network resilience: a failed price poll (outage, API throttle)
                # must NOT kill the session — keep polling and retry next cycle.
                try:
                    raw = self._current_prices()
                except Exception as exc:  # noqa: BLE001
                    print(f"{now:%H:%M:%S} price poll failed "
                          f"({type(exc).__name__}: {exc}) — retrying next cycle", flush=True)
                    time.sleep(self.poll_seconds)
                    continue

                prices = self._filter_quotes(raw, now)

                try:
                    exits = self.portfolio.live_check(prices, pd.Timestamp(now))
                    for sig in exits:
                        px = float(sig["price"])
                        fill_px = float(
                            int(px * (1.0 - self.slippage_bps / 10_000.0) * 100) / 100.0
                        )
                        if fill_px <= 0:
                            continue
                        shares = self.positions.get(sig["symbol"], 0.0)
                        if abs(shares) < 1e-9:
                            continue
                        notional = abs(shares) * fill_px
                        fee = self._fee(notional)
                        fill = Fill(
                            date=str(now.date()), symbol=sig["symbol"], side="sell",
                            shares=-abs(shares), price=fill_px, commission=float(fee),
                            notional=float(notional), time=sig["time"],
                            source="live",
                        )
                        # persist FIRST: if the ledger write fails the in-memory
                        # book is left untouched, so no phantom/lost fills ever
                        # reach the close-run merge.
                        self.ledger.append_fill(fill)
                        self.cash += notional - fee
                        self.positions.pop(sig["symbol"], None)
                        print(f"LIVE EXIT {sig['time']} {sig['symbol']} {fill_px:.2f} "
                              f"({abs(shares):.0f} shares, fee {fee:.2f})", flush=True)

                    self._write_status(prices, now)
                except Exception as exc:  # noqa: BLE001 — transient cycle error, keep polling
                    print(f"{now:%H:%M:%S} cycle error "
                          f"({type(exc).__name__}: {exc}) — continuing", flush=True)
                time.sleep(self.poll_seconds)
        finally:
            pid_file.unlink(missing_ok=True)
        return 0

    def _write_status(self, prices: dict[str, float], now: datetime) -> None:
        lots = {sym: lot for sym, lot in self.portfolio._open.items()}
        positions_out = []
        equity = self.cash
        for sym, sh in self.positions.items():
            px = prices.get(sym)
            lot = lots.get(sym)
            entry = float(lot.entry_price) if lot else None
            if px is None:
                # no fresh decision print (suspended / stale / limit-down): keep
                # the position but mark it so the panel never shows a fake P&L.
                positions_out.append({
                    "symbol": sym,
                    "shares": round(sh, 0),
                    "last": None,
                    "entry": round(entry, 2) if entry is not None else None,
                    "stop": round(float(lot.stop), 2) if lot else None,
                    "pnl": None,
                    "pnl_pct": None,
                    "blocked": self._quote_blocks.get(sym, "no print this cycle"),
                })
                continue
            pnl = (px - entry) * sh if entry is not None and sh >= 0 else None
            equity += sh * px
            positions_out.append({
                "symbol": sym,
                "shares": round(sh, 0),
                "last": round(float(px), 2),
                "entry": round(entry, 2) if entry is not None else None,
                "stop": round(float(lot.stop), 2) if lot else None,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "pnl_pct": round((px / entry - 1.0) * 100, 2) if entry else None,
                "quote_ts": self._quote_ts[sym].strftime("%H:%M:%S") if sym in self._quote_ts else None,
            })
        payload = {
            "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
            "equity_live": round(equity, 2),
            "cash": round(self.cash, 2),
            "invested_pct": round((equity - self.cash) / equity * 100, 1) if equity > 0 else 0.0,
            "positions": positions_out,
            "blocked": dict(self._quote_blocks),
            "decision_window": "09:30-11:30 / 13:00-15:00 (auction handled by preclose)",
            "deployment": self.deployment,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["LiveTrader"]
