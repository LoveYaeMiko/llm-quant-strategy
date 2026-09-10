"""Backfill a per-symbol price gap in the PIT store (D-track universe).

Why (2026-09-10 pre-open finding): the daily incremental ingest only covered
~300 names through 2026-08, so the 2026 PIT price panel has **301 symbols/day**
while the D track declares 800 (``hs300_500``). The ingest widened to 800 on
2026-09-08, but the 499 newly-added names start fresh — they have an eight-month
hole and therefore no warm indicators (ATR20 needs ~20 bars, EMA50 ~60).

``Ingestor.ingest(resume=True)`` CANNOT fix this: with ``resume`` the
already-stored symbols are re-fetched from the GLOBAL newest bar forwards, so a
per-symbol hole in the middle of the history is never revisited. This script
therefore calls the ingestor with an explicit ``start``/``end`` and
``resume=False`` for exactly the names whose coverage inside the window is
incomplete.

Usage::

    python scripts/backfill_price_gap.py --dry-run            # report only (default)
    python scripts/backfill_price_gap.py --apply              # fetch + upsert
    python scripts/backfill_price_gap.py --apply --start 2026-01-01 --end 2026-09-07
    python scripts/backfill_price_gap.py --apply --symbols 600000.SH,000001.SZ

Run it AFTER the close (>= 15:10): the mid-session constraint is that no bulk
fetch may contend with the live trader / 14:40 depth / 14:50 preclose jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _panel_coverage(store, symbols: list[str], start: str, end: str) -> dict[str, int]:
    """``{symbol: n_bars}`` inside ``[start, end]`` for the requested universe."""
    recs = store.snapshot("price")
    if recs is None or recs.empty:
        return {}
    df = recs[["symbol", "valid_from"]].copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    win = df[(df["valid_from"] >= pd.Timestamp(start)) & (df["valid_from"] <= pd.Timestamp(end))]
    win = win[win["symbol"].isin(set(symbols))]
    return win.groupby("symbol").size().to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill a price gap in the PIT store")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--symbols", default="", help="explicit comma list (default: the D universe)")
    ap.add_argument("--min-bars", type=int, default=1,
                    help="a symbol with fewer bars than this inside the window is backfilled")
    ap.add_argument("--apply", action="store_true", help="actually fetch + upsert (default: dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="report only (the default)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    from src.data.point_in_time_loader import from_url
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    url = cfg.get("data.pit_database_url")
    if not url:
        print("ERROR: data.pit_database_url is not set", file=sys.stderr)
        return 2

    if args.symbols:
        universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        universe = list(resolve_shadow_universe(cfg, "hs300_500"))
    print(f"[gap] universe={len(universe)} window=[{args.start}, {args.end}]", flush=True)

    store = from_url(url)
    try:
        cov = _panel_coverage(store, universe, args.start, args.end)
        # a full window has ~one bar per trading day; anything below the median
        # (and below ``--min-bars``) is a hole
        counts = pd.Series(cov)
        median = float(counts.median()) if len(counts) else 0.0
        missing = sorted(s for s in universe if cov.get(s, 0) < max(args.min_bars, 1))
        partial = sorted(s for s in universe
                         if cov.get(s, 0) >= 1 and median and cov.get(s, 0) < median * 0.9)
    finally:
        if hasattr(store, "close"):
            store.close()

    report = {
        "universe": len(universe), "window": [args.start, args.end],
        "with_any_bar": len(cov), "median_bars": median,
        "n_zero_bar": len(missing), "n_partial": len(partial),
        "zero_bar_sample": missing[:20], "partial_sample": partial[:20],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[gap] with any bar: {len(cov)}/{len(universe)} · median bars {median:.0f} · "
          f"zero-bar {len(missing)} · partial {len(partial)}", flush=True)
    if missing:
        print(f"[gap] zero-bar sample: {missing[:10]}", flush=True)

    targets = sorted(set(missing) | set(partial))
    if not targets:
        print("[gap] nothing to backfill — coverage looks complete", flush=True)
        return 0
    if not args.apply:
        print(f"[gap] DRY RUN — would fetch {len(targets)} symbol(s) for "
              f"[{args.start}, {args.end}] (pass --apply to execute, after the close)",
              flush=True)
        return 0

    from src.data.ingestion.ingestor import Ingestor

    print(f"[gap] fetching {len(targets)} symbol(s) …", flush=True)
    ing = Ingestor(cfg)
    stats = ing.ingest(symbols=targets, start=args.start, end=args.end, resume=False)
    bars = getattr(stats, "price_records", None) or getattr(stats, "price_bars", None)
    errors = list(getattr(stats, "errors", []) or [])
    print(f"[gap] ingested bars={bars} errors={len(errors)}", flush=True)
    for err in errors[:5]:
        print(f"  [err] {err}")

    store = from_url(url)
    try:
        cov2 = _panel_coverage(store, universe, args.start, args.end)
    finally:
        if hasattr(store, "close"):
            store.close()
    still = sorted(s for s in targets if cov2.get(s, 0) < max(args.min_bars, 1))
    print(f"[gap] after: with any bar {len(cov2)}/{len(universe)} · still empty {len(still)}",
          flush=True)
    if still:
        print(f"[gap] still empty sample: {still[:10]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
