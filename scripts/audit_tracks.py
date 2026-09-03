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

ACCOUNTS = [
    ("A_200W", "outputs/shadow_ledger_A_200W.sqlite"),
    ("B_10W", "outputs/shadow_ledger_B_10W.sqlite"),
    ("C_5W", "outputs/shadow_ledger_C_5W.sqlite"),
    ("D_5W", "outputs/shadow_ledger_D_5W.sqlite"),
]


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
    print("=" * 72)
    print("PART 1 — legality (A-share trading rules)")
    print("=" * 72)
    for account, path in ACCOUNTS:
        if not (ROOT / path).is_file():
            print(f"{account}: ledger missing, skipped")
            continue
        r = legality_audit(account, path, market)
        print(f"\n[{account}] fills={r['n_fills']}")
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
    from src.paper.pullback_book import PullbackParams, PullbackPortfolio
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

    uni800 = resolve_shadow_universe(cfg, "hs300_500")
    uni300 = resolve_shadow_universe(cfg, "hs300")

    class _BooksPortfolio:
        def __init__(self, books):
            self._books = books

        def compute_weights(self, symbols, date):
            return self._books.get(pd.Timestamp(date), {})

    def ml_books(scores_series, symbols_used, long_pct):
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
                day, long_pct=long_pct, short_pct=0.0, max_position_pct=0.05, short_scale=ss
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
        replay("A_config", _BooksPortfolio(ml_books(scores, uni300, 0.10)),
               uni300, 2_000_000, 10, 0.0, 0.0)
        replay("B_config", _BooksPortfolio(ml_books(scores, uni800, 0.05)),
               uni800, 100_000, 10, 2000.0, 0.001)
        replay("C_config", _BooksPortfolio(ml_books(scores, uni800, 0.05)),
               uni800, 50_000, 10, 2000.0, 0.001)
        d_params = PullbackParams(k=6, rank_source="ml", rank_min=0.8, ema_fast=21,
                                  ema_zone=21, zone_band=0.02, pullback_min=0.03,
                                  vol_shrink=True, atr_mult=1.5, stop_lo=0.025, stop_hi=0.04,
                                  breakeven_r=1.0, trail_r=1.5, exit_into_strength_r=3.0,
                                  max_hold=40, entry_gate=0.0, exit_gate=-0.03, trend_days=60)
        d = PullbackPortfolio(market, d_params, symbols=uni800, scores=scores)
        replay("D_config", d, uni800, 50_000, 1, 2000.0, 0.0)

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
