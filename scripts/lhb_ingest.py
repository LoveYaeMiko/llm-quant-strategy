"""Ingest dragon-tiger list history into the PIT store (record_type="lhb").

One range fetch (the endpoint serves the whole span), cached to
``data/lhb/lhb.parquet``, aggregated per (symbol, date), look-ahead columns
stripped, then upserted with next-day visibility. Re-runnable.

Usage:  python scripts/lhb_ingest.py [start] [end]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.lhb import fetch_lhb_akshare, to_pit_records  # noqa: E402
from src.data.point_in_time_loader import from_url  # noqa: E402


def main() -> int:
    cfg = load_config()
    url = cfg.get("data.pit_database_url")
    if not url:
        print("PIT_DATABASE_URL not set — aborting", file=sys.stderr)
        return 2

    start = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-28"

    cache_dir = ROOT / "data" / "lhb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "lhb.parquet"
    if cache_file.is_file():
        lhb = pd.read_parquet(cache_file)
        print(f"cache: {len(lhb):,} rows ({lhb.date.min().date()} .. {lhb.date.max().date()})")
    else:
        lhb = fetch_lhb_akshare(start, end)
        lhb.to_parquet(cache_file, index=False)
        print(f"fetched: {len(lhb):,} rows ({lhb.date.min().date()} .. {lhb.date.max().date()})")

    recs = to_pit_records(lhb)
    store = from_url(url)
    store.upsert(recs)
    print(f"upserted {len(recs):,} lhb records (record_type='lhb')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
