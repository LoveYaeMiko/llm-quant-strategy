"""14:55 closing-auction order layer (D track, the live account).

A real close-rebalance order must be SUBMITTED at 14:57-15:00 (closing call
auction) and fills at the 15:00 auction price. This module simulates that
exactly:

* at 14:55 the system decides the order list from 14:55-known data:
  - prices/ATR/EMA/pullback/volume conditions on the 14:55 print (provisional
    daily bar from minute klines);
  - the ML scanner uses the last fully-known cross-section (T-1 ranks — at
    14:55 today's final features do not exist yet, no lookahead);
  - intraday features (tail_vol / open30 / rv / range) from today's minute
    bars so far — exactly what an operator could compute at 14:55;
  - exits first (stop / trail / trend / strength at the 14:55 print), then
    entries into free slots; the same same-day re-entry block as the live
    trader (today's intraday stop-outs cannot be re-bought);
* the order list (board-lot-rounded shares) is persisted to
  ``outputs/preclose_orders_<account>.json``;
* the daily close run then EXECUTES that list at the actual 15:00 closing
  (auction) prices — see ``OrderExecutor.execute_orders`` and the runner's
  preclose provider. If this job never ran, the close run trades NOTHING that
  day (as in reality — unsubmitted orders do not fill).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd

from .cli import ROOT  # noqa: F401 — re-export for callers


def merge_targets(weights: dict[str, float], positions: dict[str, float]) -> dict[str, float]:
    """Build the 14:55 target-weight row from the book's desired weights.

    Every currently-held symbol is pinned at 0.0 FIRST, then the book's weights
    override. Rationale (defect D-5, 2026-09-08 audit): ``compute_weights`` only
    returns names the book still wants to hold, and ``OrderExecutor.execute``
    iterates over the target frame's columns — so an exited name absent from the
    dict was never sold, and the 14:55 order list could not liquidate anything.
    Pinning held names at 0.0 turns "the book no longer wants this name" into an
    explicit exit order (stop / trail / trend gate / max_hold / strength).
    """
    row: dict[str, float] = {str(s): 0.0 for s in positions}
    row.update({str(s): float(w) for s, w in weights.items()})
    return row


def _today_provisional(adapter, symbols, batch: int = 25):
    """Fetch today's minute bars and roll each symbol into one provisional bar.

    Returns ``(prov, bars)`` — ``prov`` = ``{symbol: {open, high, low, close,
    volume}}`` (only symbols with data today), ``bars`` = the raw per-symbol
    minute frames (reused for the 14:55 intraday features).
    """
    now = pd.Timestamp.now().normalize()
    start = now
    end = now + pd.Timedelta(days=1)
    prov: dict[str, dict[str, float]] = {}
    bars: dict[str, pd.DataFrame] = {}
    syms = list(symbols)
    for i in range(0, len(syms), batch):
        chunk = syms[i : i + batch]
        got = None
        for attempt in range(4):
            try:
                got = adapter.fetch_minute_klines(
                    chunk, period="1m", count=240, start=start, end=end
                )
                break
            except Exception:  # noqa: BLE001 — retry with backoff
                time.sleep(4 * (attempt + 1))
        if got is None:
            continue
        for sym in chunk:
            df = got.get(sym)
            if df is None or df.empty:
                continue
            df = df.copy()
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            if not len(df):
                continue
            bars[sym] = df
            prov[sym] = {
                "open": float(df["open"].iloc[0]),
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "close": float(df["close"].iloc[-1]),
                "volume": float(df["volume"].sum()),
            }
    return prov, bars


def _provisional_market(base_market, prov: dict[str, dict[str, float]], today: pd.Timestamp):
    """Base market (daily panel through yesterday) + today's provisional row."""
    panel = base_market.price_panel.copy()
    long = base_market.long.copy()
    if today not in panel.index:
        for sym in panel.columns:
            panel.loc[today, sym] = np.nan
    for sym, bar in prov.items():
        if sym in panel.columns:
            panel.loc[today, sym] = bar["close"]
    if prov:
        extra = pd.DataFrame(
            [
                {"symbol": sym, "date": today,
                 "open": b["open"], "high": b["high"], "low": b["low"],
                 "close": b["close"], "volume": b["volume"]}
                for sym, b in prov.items()
            ]
        ).set_index(["date", "symbol"])
        if extra.index.names != long.index.names:
            extra.index.names = long.index.names
        # concat, NOT .loc setitem: pandas raises KeyError on .loc assignment
        # with entirely new MultiIndex labels. Also DROP any existing today
        # rows first (post-ingest re-runs) or the concat duplicates them and
        # the book's unstack() fails.
        long = long[long.index.get_level_values(0) < today]
        long = pd.concat([long, extra]).sort_index()
    return SimpleNamespace(price_panel=panel, long=long)


def _scores_as_of_yesterday(cfg, base_market, symbols, start, yesterday) -> pd.Series:
    """ML scanner scores through YESTERDAY, today's row = yesterday's (ffill).

    At 14:50 today's final features do not exist; using the last fully-known
    cross-section is the only lookahead-free choice (a real 14:57 operator has
    exactly this information). The REAL base market is passed straight to the
    cached feature builder (cache hit — no slow fresh build, and no synthetic
    market objects), then today's row is ALWAYS overwritten with yesterday's
    (even if the cache already contains today's final features — post-ingest —
    using them at 14:50 would be lookahead).
    """
    import json as _json
    import os

    from .ml.train import load_artifact, score_artifact
    from .paper.ml_book import _feature_frame, _resolve_artifact

    meta_path, model_path = _resolve_artifact("lgbm", "")
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    frame = _feature_frame(base_market, meta, cfg, n_jobs=min(6, (os.cpu_count() or 4) - 2))
    assert list(frame.columns) == meta["features"], "artifact columns out of sync"
    scores = score_artifact(load_artifact(model_path), frame)
    scores = scores[~scores.index.duplicated(keep="last")]
    wide = scores.unstack()
    today = pd.Timestamp.today().normalize()
    # keep only rows <= yesterday (drop any post-ingest final-day features),
    # then append today = yesterday's cross-section (ffill).
    wide = wide[wide.index <= yesterday]
    if len(wide) == 0:
        raise RuntimeError("no scores through yesterday — feature cache empty?")
    wide = wide.reindex(wide.index.append(pd.Index([today]))).ffill()
    return wide.stack(dropna=False)


def _intraday_today(bars: dict[str, pd.DataFrame], prev_close: pd.Series, today: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Today's intraday features from minute bars so far (14:55-known)."""
    from .data.intraday import _daily_features

    rows: dict[str, dict[str, float]] = {}
    for sym, df in bars.items():
        d = _daily_features(df)
        if len(d) == 0:
            continue
        row = dict(d.iloc[-1])
        # gap uses the day's OPEN (not kept in the feature frame) vs prev close
        pc = float(prev_close.get(sym, np.nan))
        open_px = float(df["open"].iloc[0]) if len(df) else np.nan
        if np.isfinite(pc) and pc > 0 and np.isfinite(open_px):
            row["gap"] = open_px / pc - 1.0
        else:
            row["gap"] = np.nan
        rows[sym] = row

    frames: dict[str, pd.DataFrame] = {}
    for key in ("vwap_gap", "rv", "tail_vol", "gap", "open30", "range", "afternoon", "vwap"):
        col = {s: r.get(key, np.nan) for s, r in rows.items()}
        frames[key] = pd.DataFrame(col, index=[today])
    return frames


def build_preclose_orders(
    cfg,
    account: dict[str, Any],
    symbols: list[str],
    *,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Decide the 14:55 closing-auction order list for one account.

    ``out_dir`` (optional) redirects ONLY the order-list JSON — used by the
    dry-run harness so it can exercise the real ledger/market path without
    leaving a stale list for the 15:10 close run to execute.
    """
    from .cli import _build_market_for_paper
    from .data.ingestion.alphafeed_adapter import AlphaFeedAdapter
    from .data.intraday import load_intraday_frames
    from .online.order_executor import OrderExecutor
    from .paper.ledger import PaperLedger
    from .paper.pullback_book import PullbackPortfolio
    from .paper.shadow import resolve_shadow_universe

    name = str(account.get("name", "D_5W"))
    universe = resolve_shadow_universe(cfg, account.get("universe"))
    symbols = list(symbols) or universe
    start = str(cfg.section("shadow").get("start_date", "2026-01-01"))
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)

    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    base = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    prev_close = base.price_panel.iloc[-1]
    prov, bars = _today_provisional(adapter, symbols)
    print(f"preclose [{name}]: provisional bars for {len(prov)}/{len(symbols)} symbols", flush=True)
    if not prov:
        return {"ok": False, "error": "no minute data — market not open?"}

    market = _provisional_market(base, prov, today)
    scores = _scores_as_of_yesterday(cfg, base, symbols, start, yesterday)
    rollup = load_intraday_frames(cfg, symbols)
    today_feats = _intraday_today(bars, prev_close, today)
    intraday: dict[str, pd.DataFrame] = {}
    for key in ("vwap_gap", "rv", "tail_vol", "gap", "open30", "range", "afternoon", "vwap"):
        parts: list[pd.DataFrame] = []
        r = rollup.get(key)
        if r is not None and len(r):
            r = r[r.index < today]  # drop any today rows (post-refresh re-runs)
            if len(r):
                parts.append(r)
        t = today_feats.get(key)
        if t is not None and len(t):
            parts.append(t)
        if parts:
            intraday[key] = pd.concat(parts).sort_index()

    # pullback params from the deployed D config (same builder as the daily run)
    from .autopilot.state import ControlState
    from .d_cycle import _pullback_params  # reuse — same deployed parameters

    params = _pullback_params(account)
    # Kill-switch (defect D-6): the 14:55 order list must honour the autopilot
    # gross multiplier exactly like the daily run — a halt decided overnight has
    # to liquidate at the next closing auction, not only at the next close run.
    state_path = str(
        ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
    ).replace(".json", f"_{name}.json")
    control = ControlState.load(state_path)
    print(f"preclose [{name}]: control={control.mode} (gross x{control.gross_scale:g})", flush=True)

    ledger_path = str(ROOT / "outputs" / f"shadow_ledger_{name}.sqlite")
    ledger = PaperLedger(ledger_path)
    book = PullbackPortfolio(market, params, symbols=symbols, scores=scores,
                             intraday=intraday, ledger=ledger,
                             scale_getter=lambda: control.gross_scale)
    # same-day re-entry block: today's live stop-outs cannot be re-bought at close
    for f in ledger.fills_for_date(today):
        if f.side == "sell":
            book._reentry_block[f.symbol] = today
    ledger.close()

    weights = book.compute_weights(symbols, today)

    # current state: latest ledger day + today's live fills (BEFORE today's
    # close). Needed to pin exited names into the target frame — see below.
    ledger2 = PaperLedger(str(ROOT / "outputs" / f"shadow_ledger_{name}.sqlite"))
    _, cash, positions = ledger2.latest_state()
    for f in ledger2.fills_for_date(today):
        cash += f.notional - f.commission
        sh = positions.get(f.symbol, 0.0) + f.shares
        if abs(sh) < 1e-9:
            positions.pop(f.symbol, None)
        else:
            positions[f.symbol] = sh
    ledger2.close()

    px_row = pd.Series({s: prov[s]["close"] for s in prov})
    # A ledger with no recorded day yet (first run / fresh deployment) returns
    # cash=None; fall back to the configured starting cash instead of crashing
    # the 14:50 job (the runner uses the same fallback).
    if cash is None:
        cash = float(account.get("cash", 50_000))
    equity = float(cash)
    for s, sh in positions.items():
        px = px_row.get(s, np.nan)
        if np.isfinite(px):
            equity += sh * float(px)
    rets_today = px_row / prev_close.reindex(px_row.index) - 1.0

    # EXITS must be expressible as target weights. `compute_weights` only
    # returns the names the book still WANTS to hold: a name it exited (stop /
    # trail / trend gate / max_hold / strength) is simply ABSENT from the dict.
    # The executor iterates over the target frame's columns, so an absent name
    # was never sold — the 14:55 order list could not liquidate anything
    # (defect D-5, found by the 2026-09-08 independent audit). Pin every
    # currently-held symbol at 0.0 first, then let the book's weights override.
    target_row = merge_targets(weights, positions)

    ex = OrderExecutor(
        cash=float(cash),
        slippage_bps=2.0, commission_bps=2.5, min_commission=5.0,
        stamp_tax_sell_bps=5.0, transfer_fee_bps=0.1,
        max_position_pct=float(account.get("max_position_pct", 0.40)),
        notional_floor=float(account.get("notional_floor", 2000.0)),
        band_frac=float(account.get("band_frac", 0.0)),
        seed=1,
    )
    ex.restore(float(cash), dict(positions))
    targets = pd.DataFrame([target_row], index=[today])
    prices_df = pd.DataFrame([px_row], index=[today])
    res = ex.execute(targets, prices_df, equity=equity, limit_locked=rets_today)

    orders = [{"symbol": f.symbol, "side": f.side, "shares": f.shares} for f in res.fills]
    if orders:
        note = ("decided at 14:55 from 14:55-known data (T-1 ML ranks); "
                "fills at the 15:00 auction close")
    else:
        note = ("no orders (no open position to exit and no entry candidate "
                "passing the 14:55 filters)")
    out_path = (Path(out_dir) if out_dir is not None else ROOT / "outputs") / f"preclose_orders_{name}.json"
    payload = {
        "date": str(today.date()),
        "ts": time.strftime("%H:%M:%S"),
        "orders": orders,
        "note": note,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "orders": len(orders), "path": str(out_path)}


def cmd_preclose(args) -> int:
    from .config import load_config

    cfg = load_config()
    now = pd.Timestamp.now()
    hh, mm = now.hour, now.minute
    if now.weekday() >= 5:
        print("preclose: not a trading day — skip")
        return 0
    if not ((14, 45) <= (hh, mm) <= (15, 10)):
        print(f"preclose: warning — running outside the 14:45-15:10 window ({hh:02d}:{mm:02d})")
    lcfg = cfg.get("live") or {}
    account_name = str(lcfg.get("account", "D_5W"))
    account = next((a for a in cfg.get("shadow.accounts") if a.get("name") == account_name), None)
    if account is None:
        print(f"preclose: account {account_name!r} not found")
        return 1
    r = build_preclose_orders(cfg, account, list(getattr(args, "symbols", None) or []))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r.get("ok") else 1


__all__ = ["build_preclose_orders", "cmd_preclose"]
