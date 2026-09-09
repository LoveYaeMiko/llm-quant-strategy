"""Backfill the 2025-10-27 → 2025-12-12 minute-data hole (audit P-1).

The AlphaFeed minute caches carry only ~274/800 symbols per day inside that
window (every SH name missing), which is why the 2025 OOS artifact is marked
`data_coverage_invalid`. This script re-fetches the window, merges the bars into
the per-symbol caches and rebuilds `data/intraday/daily_features.parquet`.

Run it AFTER the close (15:10+): a bulk fetch competes with the 14:40 depth
snapshot and the 14:50 preclose fetch for the same AlphaFeed rate limiter, and
the preclose order list must be decided before the 15:00 auction.

Usage::

    python scripts/backfill_minute_gap.py --dry-run          # 1 symbol probe
    python scripts/backfill_minute_gap.py                    # full universe
    python scripts/backfill_minute_gap.py --batch 8 --limit 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

START = "2025-10-20"   # a few days before the hole
END = "2025-12-15"     # a few days after it
HOLE = ("2025-10-27", "2025-12-12")


def _coverage(cfg, symbols: list[str], frames: dict[str, pd.DataFrame] | None = None) -> dict:
    from src.data.intraday import load_intraday_frames

    frames = frames if frames is not None else load_intraday_frames(cfg, symbols)
    tail = frames.get("tail_vol")
    if tail is None or not len(tail):
        return {"days": 0, "min_symbols": 0, "mean_symbols": 0.0}
    win = tail.loc[START:END]
    counts = win.notna().sum(axis=1)
    return {
        "days": int(len(win)),
        "min_symbols": int(counts.min()) if len(counts) else 0,
        "mean_symbols": float(counts.mean()) if len(counts) else 0.0,
        "low_days": int((counts < 0.9 * win.shape[1]).sum()) if len(counts) else 0,
        "universe": int(win.shape[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill the 2025 minute-data hole")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N symbols (debug)")
    ap.add_argument("--dry-run", action="store_true", help="probe one symbol, write nothing")
    args = ap.parse_args()

    from src.config import load_config
    from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter, normalize_bar_timestamps
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    if args.limit:
        symbols = symbols[: args.limit]

    if args.dry_run:
        adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
        sym = symbols[0]
        got = adapter.fetch_minute_klines(
            [sym], period="1m", count=10_000,
            start=pd.Timestamp(args.start), end=pd.Timestamp(args.end),
        )
        df = normalize_bar_timestamps((got or {}).get(sym))
        if df is None or len(df) == 0:
            print(f"[dry-run] {sym}: NO BARS for {args.start}..{args.end} "
                  "— the API lower bound has moved past the hole")
            return 1
        days = df["timestamp"].dt.normalize().nunique()
        print(f"[dry-run] {sym}: {len(df)} bars across {days} days "
              f"({df['timestamp'].min()} .. {df['timestamp'].max()})")
        print("[dry-run] the API still serves the window — safe to run the full backfill")
        return 0

    before = _coverage(cfg, symbols)
    print(f"[backfill] window {args.start}..{args.end} (hole {HOLE[0]}..{HOLE[1]}) "
          f"symbols={len(symbols)} batch={args.batch}", flush=True)
    print(f"[backfill] coverage BEFORE: {before}", flush=True)

    from src.data.intraday import refresh_intraday

    summary = refresh_intraday(cfg, symbols, args.start, args.end, batch=args.batch)
    print(f"[backfill] refresh: {summary}", flush=True)

    after = _coverage(cfg, symbols)
    print(f"[backfill] coverage AFTER:  {after}", flush=True)
    ok = after["min_symbols"] >= 0.9 * max(1, after["universe"])
    print(f"[backfill] {'OK' if ok else 'STILL INCOMPLETE'} — "
          f"min symbols/day {after['min_symbols']}/{after['universe']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
