"""Fetch AlphaFeed minute klines for the shadow universe and build the intraday
feature rollup (true VWAP / realized vol / tail volume / gap).

Symbols are fetched in BATCHES of ``--batch`` (25) — one API call covers the
whole batch per time window, so the full backfill is ~100-150 calls instead of
thousands. Covers the trailing ~1 year (4 × 10000-minute windows ≈ 165 trading
days ≈ the whole 2026 shadow period).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
from src.data.intraday import _intraday_dir, build_intraday_frames


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    elif "trade_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["trade_time"])
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--total-minutes", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    from src.config import load_config
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = list(args.symbols) if args.symbols else resolve_shadow_universe(cfg, "hs300_500")
    out_dir = _intraday_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in symbols if not (out_dir / f"{s.replace('.', '_')}.parquet").is_file()]
    print(f"fetching minute klines for {len(todo)} symbols (batch {args.batch})", flush=True)

    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    buckets: dict[str, list[pd.DataFrame]] = {s: [] for s in todo}
    pulled = 0
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    windows = 0
    import time as _time

    def _call(chunk, count, start, end):
        last = None
        for attempt in range(4):
            try:
                return adapter.fetch_minute_klines(chunk, period="1m", count=count, start=start, end=end)
            except Exception as exc:  # noqa: BLE001 — SSL EOF etc., retry with backoff
                last = exc
                _time.sleep(4 * (attempt + 1))
        raise last

    while pulled < args.total_minutes:
        count = min(10_000, args.total_minutes - pulled)
        start = end - pd.Timedelta(days=int(count / 240 * 1.4))
        got_any = False
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            batch = _call(chunk, count, start, end)
            for s in chunk:
                df = batch.get(s)
                if df is not None and not df.empty:
                    buckets[s].append(df)
                    got_any = True
        if not got_any:
            break
        pulled += count
        end = start
        windows += 1
        print(f"  window {windows} done (start {start.date()})", flush=True)

    saved = 0
    for s in todo:
        frames = buckets[s]
        if not frames:
            continue
        out = pd.concat(frames, ignore_index=True)
        out = _normalize(out)
        out = out.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        out.to_parquet(out_dir / f"{s.replace('.', '_')}.parquet")
        saved += 1
    print(f"saved {saved}/{len(todo)} symbol caches", flush=True)

    wide = build_intraday_frames(cfg, symbols)
    out_path = out_dir / "daily_features.parquet"
    rollup = pd.concat({k: v for k, v in wide.items()}, axis=1)
    rollup.columns.names = ["feature", "symbol"]
    rollup.to_parquet(out_path)
    print(f"wrote {out_path} ({len(rollup)} days × {rollup.shape[1]} series)", flush=True)
    print("features:", list(wide), "| coverage:", len(wide.get("vwap_gap", pd.DataFrame()).columns), "symbols", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
