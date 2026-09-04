"""Collect AlphaFeed market-depth snapshots for the shadow universe.

Depth is real-time only (no history), so it cannot backtest — its role is the
LIVE execution layer: a rolling depth dataset powers slippage estimation and
execution timing when the shadow goes live. This script snapshots the book once
per trading day (scheduler: weekdays 14:50, just before the close) and stores
``data/depth/YYYYMMDD.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> int:
    from src.config import load_config
    from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))

    rows = []
    for i in range(0, len(symbols), 200):
        chunk = symbols[i : i + 200]
        try:
            batch = adapter.fetch_depth(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"depth batch {i // 200} failed: {exc}", flush=True)
            continue
        for sym, snap in (batch or {}).items():
            if isinstance(snap, dict):
                row = {"symbol": sym, "ts": pd.Timestamp.now().isoformat()}
                for k, v in snap.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
                rows.append(row)
            else:
                rows.append({"symbol": sym, "ts": pd.Timestamp.now().isoformat(), "raw": str(snap)})

    out_dir = ROOT / "data" / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"wrote {out_path} ({len(rows)} symbols)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
