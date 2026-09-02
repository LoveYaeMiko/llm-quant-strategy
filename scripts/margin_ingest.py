"""Ingest cached margin parquets into the PIT store (record_type="margin").

Reads the monthly checkpoints under ``data/margin/`` (written by
:func:`src.data.margin.fetch_margin_akshare`), converts them to PIT records
(balance dated d visible from d+1; snapshots closed by the next one), and
upserts them into the configured PIT database. Re-runnable: re-ingesting the
same months is idempotent (same PK, same payload).

Usage:  python scripts/margin_ingest.py [--symbols-file path]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.margin import load_margin_cache, to_pit_records  # noqa: E402
from src.data.point_in_time_loader import from_url  # noqa: E402


def main() -> int:
    cfg = load_config()
    url = cfg.get("data.pit_database_url")
    if not url:
        print("PIT_DATABASE_URL not set — aborting", file=sys.stderr)
        return 2

    margin = load_margin_cache()  # cached months only — never triggers fetches
    if margin.empty:
        print("no cached margin data — run the backfill first", file=sys.stderr)
        return 1
    print(f"margin frame: {len(margin):,} rows, {margin.symbol.nunique()} symbols, "
          f"{margin.date.min().date()} .. {margin.date.max().date()}")

    # restrict to the research universe when its member list is cached
    uni_path = ROOT / "data" / "universe"
    for name in ("hs300_500.json", "hs300.json"):
        p = uni_path / name
        if p.is_file():
            import json

            members = set(json.loads(p.read_text(encoding="utf-8")))
            before = len(margin)
            margin = margin[margin["symbol"].isin(members)]
            print(f"universe {name}: {before:,} -> {len(margin):,} rows")
            break

    recs = to_pit_records(margin)
    store = from_url(url)
    store.upsert(recs)
    print(f"upserted {len(recs):,} margin records (record_type='margin')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
