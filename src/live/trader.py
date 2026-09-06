"""Real-time intraday trader for the pullback (D) track.

Runs 09:30-15:10 on trading days, polling the current minute price of every
held symbol and executing stop breaches AT THE MOMENT they are traded — never
against past timestamps:

* prices come from the AlphaFeed minute endpoint's LATEST bar (a traded print);
* a print at/below the stop is a confirmed breach → sell immediately, record
  the fill with the actual minute timestamp (HH:MM:SS) into the ledger;
* per-position P&L is marked at the latest print every poll and written to
  ``outputs/live_<account>.json`` for the PAICC panel;
* the 17:30 close run then MERGES today's live fills (it never re-trades a
  past timestamp — the replay sweep is disabled for live dates via
  ``pb_live_intraday_from``).

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

import pandas as pd

from ..online.order_executor import Fill


def _is_trading_day(d: pd.Timestamp) -> bool:
    return d.weekday() < 5


def _in_trading_hours(now: datetime) -> bool:
    t = (now.hour, now.minute)
    return (9, 30) <= t < (11, 30) or (13, 0) <= t < (15, 10)


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

    def _current_prices(self) -> dict[str, float]:
        if not self.positions:
            return {}
        batch = self.adapter.fetch_minute_klines(list(self.positions), period="1m", count=1)
        out: dict[str, float] = {}
        for sym, df in (batch or {}).items():
            if df is None or df.empty:
                continue
            out[sym] = float(df["close"].iloc[-1])
        return out

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
                    if now.hour >= 15 and now.minute >= 10:
                        print(f"{now:%H:%M:%S} market closed — live trader exiting")
                        break
                    time.sleep(max(20, self.poll_seconds))
                    continue

                # Network resilience: a failed price poll (outage, API throttle)
                # must NOT kill the session — keep polling and retry next cycle.
                try:
                    prices = self._current_prices()
                except Exception as exc:  # noqa: BLE001
                    print(f"{now:%H:%M:%S} price poll failed "
                          f"({type(exc).__name__}: {exc}) — retrying next cycle", flush=True)
                    time.sleep(self.poll_seconds)
                    continue

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
            if px is None:
                continue
            lot = lots.get(sym)
            entry = float(lot.entry_price) if lot else None
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
            })
        payload = {
            "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
            "equity_live": round(equity, 2),
            "cash": round(self.cash, 2),
            "invested_pct": round((equity - self.cash) / equity * 100, 1) if equity > 0 else 0.0,
            "positions": positions_out,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["LiveTrader"]
