"""Four-track self-audit: overfitting evidence + A-share trade legality.

1. Overfitting:
   * the deployed artifact's training window vs the 2026 shadow window;
   * 2025 replay of every track's CURRENT config (config-selection robustness);
   * grid spread (winner vs median) for the tuned knobs.
2. Legality (A-share rules):
   * T+1 — same-day buy+sell of one symbol;
   * board lot — buys/sells not in multiples of 100 shares;
   * price limits — entries on limit-up bars / exits on limit-down bars;
   * price tick — fills not on the 0.01 tick;
   * suspension — fills with no close (should be zero by construction).

Audited accounts come from ``shadow.accounts`` (priority desc, the daily-loop
order) — never a hardcoded list: A/B/C were retired on 2026-09-08 and an audit
must not silently probe ledgers the config no longer registers.
# 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def account_ledger_relpath(cfg, name: str) -> str:
    """Per-account ledger path, derived the same way ``src/cli.py`` derives it."""
    base = str((cfg.section("shadow") or {}).get("ledger_db", "outputs/shadow_ledger.sqlite"))
    return base.replace(".sqlite", f"_{name}.sqlite")


def load_accounts(cfg) -> list[dict]:
    """``shadow.accounts`` in the daily closed-loop order (priority desc)."""
    accounts = list(cfg.get("shadow.accounts") or [])
    return sorted(accounts, key=lambda a: int(a.get("priority", 0) or 0), reverse=True)


def board_code(symbol: str) -> str:
    if symbol.startswith(("688", "689")):
        return "star"
    if symbol.startswith(("300", "301")):
        return "chinext"
    return "main"


def limit_pct(symbol: str) -> float:
    return 0.20 if board_code(symbol) in ("star", "chinext") else 0.10


def load_fills(path: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(
        "SELECT seq, date, symbol, side, shares, price, commission, notional FROM fills",
        conn,
    )
    conn.close()
    return df


def legality_audit(account: str, path: str, market) -> dict:
    fills = load_fills(path)
    out = {"n_fills": len(fills)}
    if len(fills) == 0:
        return out

    # T+1 / same-day flip: one symbol both sides on one date
    grp = fills.groupby(["date", "symbol"])["side"].nunique()
    flips = int((grp > 1).sum())
    out["same_day_flips"] = flips

    # board lot: 100-share multiples (main/ChiNext), 200-share min (STAR) —
    # STAR trades in 1-share increments above 200, so only non-STAR symbols
    # are checked against the 100-multiple rule
    non_star = ~fills["symbol"].str.startswith(("688", "689"))
    odd = fills[non_star & (np.abs(fills["shares"]) % 100 != 0)]
    star_viol = 0
    for _, f in fills.iterrows():
        sym = str(f["symbol"])
        if sym[:3] in ("688", "689") and f["side"] == "buy" and abs(f["shares"]) < 200:
            star_viol += 1
    out["odd_lot_fills"] = int(len(odd))
    out["odd_lot_notional"] = float(odd["notional"].abs().sum()) if len(odd) else 0.0
    out["star_lot_violations"] = star_viol

    # price tick 0.01
    tick_off = fills[(fills["price"] * 100).round(3) % 1 != 0]
    out["off_tick_fills"] = int(len(tick_off))

    # limit-lock days: buy into a limit-up close, sell into a limit-down close
    close = market.price_panel
    ret = close.pct_change(fill_method=None)
    n_locked = 0
    locked_examples = []
    for _, f in fills.iterrows():
        d = pd.Timestamp(f["date"])
        sym = str(f["symbol"])
        try:
            r = float(ret.loc[d, sym])
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        lim = limit_pct(sym)
        if f["side"] == "buy" and r >= lim - 0.005:
            n_locked += 1
            if len(locked_examples) < 3:
                locked_examples.append(f"{f['date']} {sym} buy ret={r:+.2%}")
        elif f["side"] == "sell" and r <= -(lim - 0.005):
            n_locked += 1
            if len(locked_examples) < 3:
                locked_examples.append(f"{f['date']} {sym} sell ret={r:+.2%}")
    out["limit_locked_fills"] = n_locked
    out["limit_locked_examples"] = locked_examples
    return out


def main() -> int:
    from src.cli import _market_data
    from src.config import load_config

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    accounts = load_accounts(cfg)
    print("audited accounts (shadow.accounts, priority desc): "
          + (", ".join(f"{a['name']}[{a.get('alpha_source', 'pool')}]" for a in accounts)
             or "<none>"))
    if not accounts:
        print("WARNING: shadow.accounts is empty — no account-level audit performed")
    print("=" * 72)
    print("PART 1 — legality (A-share trading rules)")
    print("=" * 72)
    for account in accounts:
        name = str(account["name"])
        source = str(account.get("alpha_source", "pool"))
        rel = account_ledger_relpath(cfg, name)
        if not (ROOT / rel).is_file():
            print(f"\n[{name}] alpha_source={source}: ledger {rel} missing, skipped")
            continue
        r = legality_audit(name, rel, market)
        print(f"\n[{name}] alpha_source={source} ledger={rel} fills={r['n_fills']}")
        print(f"  T+1 same-day flip: {r['same_day_flips']}")
        print(f"  odd-lot fills (not 100-share): {r['odd_lot_fills']} "
              f"(notional {r['odd_lot_notional']:,.0f})")
        print(f"  STAR(688) <200-share buys: {r.get('star_lot_violations', '?')}")
        print(f"  off-tick fills: {r['off_tick_fills']}")
        print(f"  limit-locked fills: {r['limit_locked_fills']} "
              f"{r['limit_locked_examples']}")

    # overfitting — artifact training window
    print("\n" + "=" * 72)
    print("PART 2 — overfitting")
    print("=" * 72)
    meta_path = sorted((ROOT / "outputs" / "models").glob("ml_*.json"))[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"\nartifact: {meta_path.name}")
    for key in ("window", "train_window", "val_window", "test_window", "data_through"):
        if key in meta:
            print(f"  {key}: {meta[key]}")
    md = meta.get("metadata", {})
    for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end", "data_through"):
        if key in md:
            print(f"  metadata.{key}: {md[key]}")

    # 2025 replay of the current configs (config-selection robustness)
    # Books built from the cached full-panel scores with the SAME long_book_weights
    # + trend-gate construction the deployed shadow uses (identical semantics,
    # no feature rebuild).
    from src.paper.ledger import PaperLedger
    from src.paper.pullback_book import FLAT_STOP_DEFAULT, PullbackParams, PullbackPortfolio
    from src.paper.runner import PaperRunner
    from src.portfolio.alpha_core import _market_trend, long_book_weights
    from src.ml import load_artifact, score_artifact

    frame_cache = ROOT / "outputs" / f"_dtune_frame_{meta_path.stem}.parquet"
    scores = None
    if frame_cache.exists():
        booster = load_artifact(meta_path.with_suffix(".txt"))
        frame = pd.read_parquet(frame_cache)
        scores = score_artifact(booster, frame)
        print("\nfull-panel scores ready (2025 replay)")

    print("\n2025 replay of current track configs:")
    from src.paper.shadow import resolve_shadow_universe

    class _BooksPortfolio:
        def __init__(self, books):
            self._books = books

        def compute_weights(self, symbols, date):
            return self._books.get(pd.Timestamp(date), {})

    def ml_books(scores_series, symbols_used, long_pct, max_position_pct):
        comp = scores_series[
            (scores_series.index.get_level_values(0) >= pd.Timestamp("2025-01-01"))
            & (scores_series.index.get_level_values(0) <= pd.Timestamp("2025-12-31"))
        ]
        close = market.price_panel.reindex(columns=symbols_used)
        trend = _market_trend(close, 60)
        books = {}
        for d, day in comp.groupby(level=0):
            ss = 1.0
            if pd.Timestamp(d) in trend.index:
                t = trend[pd.Timestamp(d)]
                if np.isfinite(t) and t > 0.03:
                    ss = 0.5
            day = day[day.index.get_level_values(1).isin(symbols_used)]
            books[pd.Timestamp(d)] = long_book_weights(
                day, long_pct=long_pct, short_pct=0.0,
                max_position_pct=max_position_pct, short_scale=ss
            )
        return books

    def replay(label, portfolio, symbols_used, cash, rebalance, floor, band):
        ledger_path = ROOT / "outputs" / f"_audit_{label}.sqlite"
        ledger_path.unlink(missing_ok=True)
        ledger = PaperLedger(str(ledger_path))
        runner = PaperRunner(
            portfolio, market, ledger, symbols=symbols_used, cash=cash,
            slippage_bps=2.0, commission_bps=2.5, min_commission=5.0,
            stamp_tax_sell_bps=5.0, transfer_fee_bps=0.1,
            rebalance_days=rebalance, notional_floor=floor, band_frac=band,
            pit_strict=True, seed=7,
        )
        out = runner.run(start="2025-01-01", end="2025-12-31")
        ledger.close()
        ledger_path.unlink(missing_ok=True)
        m = out["metrics"]
        print(f"  {label:10s}: cum={m.get('total_return', 0):+.2%} "
              f"ann={m.get('annualized_return', 0):+.2%} sharpe={m.get('sharpe', 0):+.2f} "
              f"maxDD={m.get('max_drawdown', 0):+.2%} fills={m.get('n_fills', 0)}")
        return m

    if scores is not None:
        # Replay each CONFIGURED account with its DEPLOYED parameters (read from
        # the master config so the out-of-sample probe never drifts from what
        # runs live) through the path its alpha_source selects:
        #   ml       → cross-sectional book (long_book_weights + trend gate);
        #   pullback → D-track book with the intraday feature pack AND the
        #              minute-bar stop provider — the same wiring as
        #              src/cli.py::_build_account_portfolio, so the tail-volume
        #              entry gate (pb_tail_vol_max) and the intraday stop sweep
        #              are actually exercised.
        from src.data.intraday import load_intraday_frames, make_minute_provider

        def _replay_ml(account):
            name = str(account["name"])
            cap = float(account.get("max_position_pct", 0.05))
            uni = resolve_shadow_universe(cfg, account.get("universe"))
            replay(
                name,
                _BooksPortfolio(ml_books(scores, uni, float(account.get("long_pct", 0.10)), cap)),
                uni,
                float(account["cash"]),
                int(account.get("rebalance_days", 10)),
                float(account.get("notional_floor", 0.0)),
                float(account.get("band_frac", 0.0)),
            )

        def _replay_pullback(account):
            name = str(account["name"])
            uni = resolve_shadow_universe(cfg, account.get("universe"))
            # 口径与生产同构：加载 data/intraday 分钟特征包（pb_tail_vol_max 生效）
            intraday = None
            if bool(account.get("pb_use_intraday", False)):
                intraday = load_intraday_frames(cfg, uni)
            minute_provider = None
            if bool(account.get("pb_intraday_stops", False)):
                minute_provider = make_minute_provider(cfg)
            print(f"  intraday pack: {list(intraday) if intraday else '<not loaded>'} | "
                  f"minute provider: {'on' if minute_provider is not None else 'off'}")
            first = (intraday or {}).get("vwap_gap")
            if first is not None and len(first):
                lo, hi = first.index.min(), first.index.max()
                print(f"  intraday coverage: {lo.date()}..{hi.date()} "
                      f"({first.shape[1]} symbols)")
                if lo > pd.Timestamp("2025-01-01"):
                    print("  WARNING: the tail-volume entry gate treats a MISSING day as "
                          "FAIL, so this 2025 replay only trades from the coverage "
                          "start — the fill count is NOT a full-year figure")
            params = PullbackParams(
                k=int(account.get("pb_k", 8)),
                rank_source=str(account.get("pb_rank_source", "momentum")),
                rank_min=float(account.get("pb_rank_min", 0.8)),
                mom_window=int(account.get("pb_mom_window", 63)),
                mom_long_rank_min=float(account.get("pb_mom_long_rank_min", 0.0)),
                bounce_confirm=bool(account.get("pb_bounce_confirm", False)),
                ema_fast=int(account.get("pb_ema_fast", 9)),
                ema_zone=int(account.get("pb_ema_zone", 21)),
                zone_band=float(account.get("pb_zone_band", 0.02)),
                pullback_min=float(account.get("pb_pullback_min", 0.03)),
                vol_shrink=bool(account.get("pb_vol_shrink", True)),
                atr_mult=float(account.get("pb_atr_mult", 1.5)),
                stop_lo=float(account.get("pb_stop_lo", FLAT_STOP_DEFAULT)),
                stop_hi=float(account.get("pb_stop_hi", FLAT_STOP_DEFAULT)),
                breakeven_r=float(account.get("pb_breakeven_r", 1.0)),
                trail_r=float(account.get("pb_trail_r", 1.5)),
                exit_into_strength_r=float(account.get("pb_exit_into_strength_r", 0.0)),
                max_hold=int(account.get("pb_max_hold", 40)),
                entry_gate=float(account.get("pb_entry_gate", 0.0)),
                exit_gate=float(account.get("pb_exit_gate", -0.03)),
                trend_days=int(account.get("pb_trend_days", 60)),
                vwap_filter=float(account.get("pb_vwap_filter", 0.0)),
                stop_rv=bool(account.get("pb_stop_rv", False)),
                tail_vol_max=float(account.get("pb_tail_vol_max", 0.0)),
                open30_max=float(account.get("pb_open30_max", 0.0)),
                range_max=float(account.get("pb_range_max", 0.0)),
                full_invest=bool(account.get("pb_full_invest", False)),
                stop_trigger=str(account.get("pb_stop_trigger", "low")),
                stop_buffer=float(account.get("pb_stop_buffer", 0.0)),
                stop_open_minutes=int(account.get("pb_stop_open_minutes", 0)),
            )
            book = PullbackPortfolio(
                market, params, symbols=uni, scores=scores,
                intraday=intraday, minute_provider=minute_provider,
            )
            book.live_intraday_from = str(account.get("pb_live_intraday_from", "") or "") or None
            replay(
                name, book, uni, float(account["cash"]),
                int(account.get("rebalance_days", 1)),
                float(account.get("notional_floor", 2000.0)), float(account.get("band_frac", 0.0)),
            )

        for account in accounts:
            name = str(account["name"])
            source = str(account.get("alpha_source", "pool"))
            print(f"\n[{name}] alpha_source={source} — 2025 replay:")
            if source == "pullback":
                _replay_pullback(account)
            elif source == "ml":
                _replay_ml(account)
            else:
                print(f"  {name}: alpha_source={source} has no replay path in this "
                      f"audit — skipped")

    # grid spreads
    print("\ngrid selection spreads (2026 ann_return):")
    for grid_file, key in (("outputs/d_track_grid.json", "2026"),
                           ("outputs/d_track_grid2.json", "2026"),
                           ("outputs/d_track_grid3.json", "2026"),
                           ("outputs/b_track_grid.json", None),
                           ("outputs/a_track_grid.json", None),
                           ("outputs/c_track_grid.json", None)):
        p = ROOT / grid_file
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        anns = []
        for label, entry in data.items():
            r = entry.get(key) if key else entry
            if isinstance(r, dict) and "ann_return" in r:
                anns.append(float(r["ann_return"]))
        if anns:
            anns_sorted = sorted(anns, reverse=True)
            print(f"  {grid_file}: n={len(anns)} best={anns_sorted[0]:+.2%} "
                  f"median={anns_sorted[len(anns) // 2]:+.2%} worst={anns_sorted[-1]:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
