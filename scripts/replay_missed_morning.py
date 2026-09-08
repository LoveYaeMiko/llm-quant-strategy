"""One-off outage re-simulation: replay the MISSED morning session (≤11:30).

The 09:25 live trader never started on 2026-09-08 (Docker auto-start failed),
so the morning's intraday stop monitoring is replayed NOW with the exact
live-trader semantics — point-in-time, never a future bar:

* trigger = first minute whose CLOSE prints at/below the stop (confirmed
  breach, no wick ambiguity), after the 30-minute open exemption;
* fill = the breaching print × (1 − slippage), tick-rounded, timestamped at
  the trigger minute — identical to `cli.py live`;
* bars AFTER 11:30 are never read (the afternoon belongs to the live trader).

Triggered exits are appended to the D ledger as live fills; the live trader
is then restarted so its book reflects the corrected state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.cli import ROOT, _build_market_for_paper, load_config
from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
from src.d_cycle import _pullback_params
from src.ml.train import load_artifact, score_artifact
from src.paper.ledger import PaperLedger
from src.paper.ml_book import _feature_frame, _resolve_artifact
from src.paper.pullback_book import PullbackPortfolio
from src.paper.shadow import resolve_shadow_universe

TODAY = pd.Timestamp.today().normalize()
MORNING_END = TODAY + pd.Timedelta(hours=11, minutes=30)


def main() -> int:
    cfg = load_config()
    lcfg = cfg.get("live") or {}
    name = str(lcfg.get("account", "D_5W"))
    account = next(a for a in cfg.get("shadow.accounts") if a.get("name") == name)
    symbols = resolve_shadow_universe(cfg, account.get("universe"))
    start = str(cfg.section("shadow").get("start_date", "2026-01-01"))

    market = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    meta_path, model_path = _resolve_artifact("lgbm", "")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = _feature_frame(market, meta, cfg, n_jobs=6)
    assert list(frame.columns) == meta["features"], "artifact columns out of sync"
    scores = score_artifact(load_artifact(model_path), frame)

    ledger = PaperLedger(str(ROOT / "outputs" / f"shadow_ledger_{name}.sqlite"))
    book = PullbackPortfolio(
        market, _pullback_params(account), symbols=symbols, scores=scores, ledger=ledger
    )
    if not book._open:
        print("no open lots — nothing to replay")
        ledger.close()
        return 0

    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    held = list(book._open)
    got = adapter.fetch_minute_klines(held, period="1m", count=240,
                                      start=TODAY, end=TODAY + pd.Timedelta(days=1))

    slippage = 2.0 / 10_000.0
    cash = float(ledger.latest_state()[1])
    existing = {(f.symbol, f.time) for f in ledger.fills_for_date(TODAY)}
    fills = []
    for sym, lot in list(book._open.items()):
        df = (got or {}).get(sym)
        if df is None or df.empty:
            continue
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        cut = pd.Timestamp("09:30").time()
        if book.p.stop_open_minutes > 0:
            cut = (pd.Timestamp("09:30") + pd.Timedelta(minutes=book.p.stop_open_minutes)).time()
        bars = df[(df["timestamp"].dt.normalize() == TODAY)
                  & (df["timestamp"].dt.time <= MORNING_END.time())
                  & (df["timestamp"].dt.time >= cut)].copy()
        if bars.empty:
            continue
        thr = lot.stop * (1.0 - book.p.stop_buffer)
        hit = bars[bars["close"] <= thr]
        if hit.empty:
            continue
        bar = hit.iloc[0]
        ts = bar["timestamp"]
        px = min(float(lot.stop), float(bar["close"]))  # the confirmed print
        fill_px = float(int(px * (1.0 - slippage) * 100) / 100.0)
        notional = abs(lot.qty) * fill_px
        fee = max(5.0, notional * 2.5 / 10_000.0) + notional * 5.0 / 10_000.0 + notional * 0.1 / 10_000.0
        fills.append({
            "symbol": sym, "time": ts.strftime("%H:%M:%S"), "print": px,
            "fill_px": fill_px, "shares": -abs(lot.qty), "fee": fee,
        })
        # idempotent: re-runs (e.g. after the trader's startup status-clear)
        # must not double-append the same morning exit
        if (sym, ts.strftime("%H:%M:%S")) not in existing:
            from src.online.order_executor import Fill

            ledger.append_fill(Fill(
                date=str(TODAY.date()), symbol=sym, side="sell",
                shares=-abs(lot.qty), price=fill_px, commission=float(fee),
                notional=float(notional), time=ts.strftime("%H:%M:%S"),
            ))
        book._open.pop(sym)
        cash += notional - fee

    # write the refreshed live status (panel card) — positions marked at the
    # latest morning prints, honest "re-simulated" note
    px_now = {}
    for sym, df in (got or {}).items():
        if df is not None and not df.empty and pd.api.types.is_numeric_dtype(df["timestamp"]):
            t = pd.to_datetime(df["timestamp"], unit="ms")
            m = df[(t.dt.normalize() == TODAY) & (t.dt.time <= MORNING_END.time())]
            if len(m):
                px_now[sym] = float(m["close"].iloc[-1])
    positions = []
    equity = cash
    for sym, lot in book._open.items():
        px = px_now.get(sym)
        if px is None:
            continue
        positions.append({
            "symbol": sym, "shares": round(lot.qty, 0), "last": round(px, 2),
            "entry": round(float(lot.entry_price), 2), "stop": round(float(lot.stop), 2),
            "pnl": round((px - lot.entry_price) * lot.qty, 2),
            "pnl_pct": round((px / lot.entry_price - 1.0) * 100, 2),
        })
        equity += lot.qty * px
    status = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "equity_live": round(equity, 2),
        "cash": round(cash, 2),
        "invested_pct": round((equity - cash) / equity * 100, 1) if equity > 0 else 0.0,
        "positions": positions,
        "note": "morning re-simulation (live trader missed 09:25 due to Docker outage)",
    }
    status_path = ROOT / "outputs" / f"live_{name}.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.close()

    print("morning replay fills:")
    for f in fills:
        print(f"  {f['time']} {f['symbol']} sell {f['shares']:.0f} @ {f['fill_px']:.2f} "
              f"(print {f['print']:.2f}, fee {f['fee']:.2f})")
    print(f"no fills" if not fills else f"{len(fills)} fills appended")
    print(f"remaining positions: {[(s, round(l.qty)) for s, l in book._open.items()]}")
    print(f"wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
